"""Z-Image Turbo keyframe, then MiniMax H3 image-to-video, via a local ComfyUI.

This matches the verified colab-zimage H3 run (fl2va, no turbo LoRA) and the
official Z-Image Turbo template. The worker never calls OpenRouter /videos.
"""
import io
import json
import math
import os
import random
import re
import threading
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

ZIMAGE_UNET_DEFAULT = 'z_image_turbo_bf16.safetensors'
ZIMAGE_CLIP_DEFAULT = 'qwen_3_4b.safetensors'
ZIMAGE_VAE_DEFAULT = 'ae.safetensors'
H3_UNET_DEFAULT = 'minimax_h3_fl2va_pruned_fp8_scaled.safetensors'
H3_CLIP_DEFAULT = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
H3_VIDEO_VAE_DEFAULT = 'minimax_h3_video_vae_fp16.safetensors'
H3_AUDIO_VAE_DEFAULT = 'minimax_h3_audio_vae_fp32.safetensors'
H3_SEED_DEFAULT = 2026101903
# Episode scenes are 8 seconds. 8*24=192, which is already on the 17k+5 grid.
# 5 seconds snaps to 124 (~5.17s). 15 seconds snaps to 362; that length stays unused.
DEFAULT_SCENE_SECONDS = 8.0
MAX_SCENE_SECONDS = 8.0
MIN_SCENE_SECONDS = 5.0
H3_FRAMES_8S = 192
H3_FRAMES_5S = 124
# Official Z-Image Turbo template latent. 768 and 768×1344 make EmptySD3 emit stacked panels.
ZIMAGE_SIDE = 1024
KEYFRAME_ATTEMPTS = int(os.environ.get('KEYFRAME_ATTEMPTS', '10') or 10)
TILE_CORR_MAX = 0.75
SINGLE_FRAME_RULE = (
    'Full-bleed single cinematic frame filling the entire image. '
    'No comic panels, no triptych, no stacked strips, no collage, '
    'no split screen, no storyboard grid, no repeated bands.'
)

def _env(name, default):
    value = os.environ.get(name, '').strip()
    return value or default

def zimage_unet():
    return _env('ZIMAGE_UNET', ZIMAGE_UNET_DEFAULT)

def zimage_clip():
    return _env('ZIMAGE_CLIP', ZIMAGE_CLIP_DEFAULT)

def zimage_vae():
    return _env('ZIMAGE_VAE', ZIMAGE_VAE_DEFAULT)

def h3_unet():
    return _env('H3_UNET', H3_UNET_DEFAULT)

def h3_clip():
    return _env('H3_CLIP', H3_CLIP_DEFAULT)

def h3_video_vae():
    return _env('H3_VIDEO_VAE', H3_VIDEO_VAE_DEFAULT)

def h3_audio_vae():
    return _env('H3_AUDIO_VAE', H3_AUDIO_VAE_DEFAULT)

def h3_length(seconds=DEFAULT_SCENE_SECONDS):
    """Snap a duration to H3's 17k+5 frame grid at 24 fps.

    Same rule as ComfyUI align_frame_count: start at ``round(seconds * 24)``
    and step up until ``n % 17 == 5``. ``h3_length(8)`` is 192. ``h3_length(5)``
    is 124 because 120 is not on the grid.
    """
    raw = max(5, int(round(float(seconds) * 24)))
    frame_count = raw
    while frame_count % 17 != 5:
        frame_count += 1
    return frame_count

def _bounded_seconds(seconds):
    seconds = float(seconds)
    if seconds > MAX_SCENE_SECONDS:
        seconds = MAX_SCENE_SECONDS
    if seconds < MIN_SCENE_SECONDS:
        seconds = MIN_SCENE_SECONDS
    return seconds

def scene_seconds(job=None):
    """Job duration wins. ``H3_SECONDS`` is the fallback when a job omits it.

    Anything above 8 seconds is capped at 8 (192 frames). A 15 second request
    does not become the native 362-frame length.
    """
    seconds = None
    if job is not None and str(job.get('duration') or '').strip():
        try:
            seconds = float(job['duration'])
        except (TypeError, ValueError):
            raise SystemExit(f"Invalid job duration: {job.get('duration')}")
    if seconds is None:
        raw = os.environ.get('H3_SECONDS', '').strip()
        if raw:
            try:
                seconds = float(raw)
            except ValueError:
                raise SystemExit(f'Invalid H3_SECONDS: {raw}')
        else:
            seconds = DEFAULT_SCENE_SECONDS
    return _bounded_seconds(seconds)

def scene_length(job=None):
    return h3_length(scene_seconds(job))

def frame_size(aspect_ratio):
    """Square, H3 16:9 landscape, or 9:16 portrait. Short edge stays 768.

    Portrait is 768×1344, the same pixel budget as 1344×768, both divisible by 32.
    """
    text = (aspect_ratio or '').strip().lower().replace(' ', '')
    if '9:16' in text or '9x16' in text:
        return 768, 1344
    if '16:9' in text or '16x9' in text:
        return 1344, 768
    return 768, 768

def zimage_latent_size():
    """Z-Image always samples the official 1024 square, never the H3 canvas."""
    return ZIMAGE_SIDE, ZIMAGE_SIDE

def _Image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Pillow is required to fit and check Z-Image keyframes') from exc
    return Image

def _resample():
    Image = _Image()
    return getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')

def fit_cover_image(image, width, height):
    """Scale so the image covers the H3 canvas, then center-crop. No letterbox."""
    image = image.convert('RGB')
    width, height = int(width), int(height)
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(width, int(round(image.width * scale))), max(height, int(round(image.height * scale)))),
        _resample(),
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))

def _corr_shift(gray, width, height, shift):
    if shift <= 0 or shift >= height:
        return 0.0
    rows = height - shift
    count = rows * width
    sum_a = sum_b = 0
    for y in range(rows):
        a0 = y * width
        b0 = (y + shift) * width
        for x in range(width):
            sum_a += gray[a0 + x]
            sum_b += gray[b0 + x]
    mean_a = sum_a / count
    mean_b = sum_b / count
    num = var_a = var_b = 0.0
    for y in range(rows):
        a0 = y * width
        b0 = (y + shift) * width
        for x in range(width):
            da = gray[a0 + x] - mean_a
            db = gray[b0 + x] - mean_b
            num += da * db
            var_a += da * da
            var_b += db * db
    denom = math.sqrt(var_a * var_b)
    if denom < 1e-6:
        return 0.0
    return num / denom

TILE_SHIFTS = (2, 3, 4, 5)

def tile_scores(image):
    """Normalized autocorrelation at H/n (rows) and W/n (columns) on a small luma probe.

    Vertical keys h2..h5 catch stacked strips and odd row grids; horizontal
    keys w2..w5 catch side-by-side comic columns (dog-cat 3x5 contact sheet).
    """
    probe = image.convert('L')
    probe.thumbnail((96, 96), _resample())
    scores = {}
    for axis, img in (('h', probe), ('w', probe.transpose(_Image().Transpose.TRANSPOSE)
                                      if hasattr(_Image(), 'Transpose') else probe.transpose(5))):
        width, height = img.size
        gray = list(img.getdata())
        for n in TILE_SHIFTS:
            key = '%s%d' % (axis, n)
            if height < 2 * n or width < 2:
                scores[key] = 0.0
            else:
                scores[key] = _corr_shift(gray, width, height, max(1, height // n))
    return scores

def vertical_tile_scores(image):
    """Back-compat alias: now returns vertical and horizontal scores."""
    return tile_scores(image)

def portrait_grid_hits(image):
    """Find grids of centered, compact subjects with distinct adjacent cells."""
    best = None
    flagged = False
    for n in (2, 3):
        portraits = 0
        thumbs = []
        for row in range(n):
            for col in range(n):
                cell = image.crop((col * image.width // n, row * image.height // n,
                                   (col + 1) * image.width // n, (row + 1) * image.height // n))
                cell = cell.convert('L').resize((48, 48), _resample())
                thumbs.append(list(cell.resize((12, 12), _resample()).getdata()))
                gray = list(cell.getdata())
                mean = sum(gray) / len(gray)
                energy = []
                for y in range(48):
                    for x in range(48):
                        dx = (gray[y * 48 + min(x + 1, 47)]
                              - gray[y * 48 + max(x - 1, 0)]) / 2
                        dy = (gray[min(y + 1, 47) * 48 + x]
                              - gray[max(y - 1, 0) * 48 + x]) / 2
                        energy.append(abs(gray[y * 48 + x] - mean) + math.hypot(dx, dy))
                top = sorted(range(len(energy)), key=energy.__getitem__, reverse=True)[:len(energy) // 5]
                cx = sum(i % 48 + 0.5 for i in top) / len(top)
                cy = sum(i // 48 + 0.5 for i in top) / len(top)
                distance = math.hypot((cx - 24) / 24, (cy - 24) / 24)
                contrast = (sum(energy[i] for i in top) / len(top)
                            / max(sum(energy) / len(energy), 1e-6))
                spread = math.sqrt(sum(((i % 48 + 0.5 - cx) / 24) ** 2
                                       + ((i // 48 + 0.5 - cy) / 24) ** 2
                                       for i in top) / len(top))
                portraits += distance < 0.45 and contrast > 1.55 and spread < 0.78
        adjacent = []
        for row in range(n):
            for col in range(n):
                index = row * n + col
                neighbors = []
                if col + 1 < n:
                    neighbors.append(index + 1)
                if row + 1 < n:
                    neighbors.append(index + n)
                for neighbor in neighbors:
                    pair = thumbs[index] + thumbs[neighbor]
                    adjacent.append(_corr_shift(pair, 144, 2, 1))
        adj_mean = sum(adjacent) / len(adjacent)
        hit = portraits >= math.ceil(0.55 * n * n) and adj_mean < 0.50
        if best is None or (hit, portraits / (n * n)) > (flagged, best['portraits'] / best['n'] ** 2):
            best = {'n': n, 'portraits': portraits, 'adj_mean': adj_mean}
        flagged = flagged or hit
    return flagged, best

def _band_crops(image, key):
    bands = int(key[1:])
    if key[0] == 'h':
        return [image.crop((0, i * image.height // bands, image.width, (i + 1) * image.height // bands))
                for i in range(bands)]
    return [image.crop((i * image.width // bands, 0, (i + 1) * image.width // bands, image.height))
            for i in range(bands)]

def salvage_keyframe(image, scores, width, height, threshold=None):
    """Crop a central portrait cell, salvage remaining repeated bands, then fit."""
    if threshold is None:
        threshold = tile_corr_max()
    if scores.get('portrait_hits', 0) > 0:
        n = int(scores['portrait_n'])
        center = n // 2
        image = image.crop((center * image.width // n, center * image.height // n,
                            (center + 1) * image.width // n, (center + 1) * image.height // n))
        scores = tile_scores(image)
    scores = {key: value for key, value in scores.items() if not key.startswith('portrait_')}
    for _ in range(3):
        if not scores or max(scores.values()) <= threshold:
            break
        key = max(scores, key=scores.get)
        crops = [c for c in _band_crops(image, key) if c.width >= 8 and c.height >= 8]
        if not crops:
            break
        image = min(crops, key=lambda crop: max(tile_scores(crop).values()))
        scores = tile_scores(image)
    return fit_cover_image(image, width, height)

def contact_sheet_image(width=96, height=160, cols=3, rows=5):
    """The dog-cat-5s failure: a cols x rows comic grid of distinct-ish panels."""
    Image = _Image()
    pw, ph = max(1, width // cols), max(1, height // rows)
    panel = single_scene_image(pw, ph)
    canvas = Image.new('RGB', (pw * cols, ph * rows))
    for r in range(rows):
        for c in range(cols):
            canvas.paste(panel, (c * pw, r * ph))
    return canvas

def tile_corr_max():
    raw = os.environ.get('KEYFRAME_TILE_CORR', '').strip()
    return float(raw) if raw else TILE_CORR_MAX

def keyframe_is_tiled(image, threshold=None):
    if threshold is None:
        threshold = tile_corr_max()
    scores = tile_scores(image)
    auto = max(scores.values()) > threshold
    port, info = portrait_grid_hits(image)
    scores.update(portrait_n=float(info['n']), portrait_hits=float(info['portraits']),
                  portrait_adj=float(info['adj_mean']))
    return auto or port, scores

def single_scene_image(width=96, height=96):
    """A non-repeating still, for tests and the Comfy HTTP mock."""
    Image = _Image()
    rng = random.Random(1)
    image = Image.frombytes(
        'RGB', (width, height),
        bytes(rng.randrange(256) for _ in range(width * height * 3)),
    )
    pixels = image.load()
    for y in range(height // 3, height // 3 + max(1, height // 5)):
        for x in range(width // 4, width // 4 + max(1, width // 3)):
            pixels[x, y] = (240, 180, 40)
    return image

def stacked_strip_image(width=96, height=96, bands=3):
    """The dog-cat failure: the same strip repeated down the frame."""
    Image = _Image()
    band_h = max(1, height // bands)
    band = single_scene_image(width, band_h)
    canvas = Image.new('RGB', (width, band_h * bands))
    for index in range(bands):
        canvas.paste(band, (0, index * band_h))
    return canvas

def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()

def job_identity(job):
    jid = job.get('job_id') or job.get('id')
    if not jid:
        raise ValueError('job has no id')
    return str(jid)

def safe_id(value):
    cleaned = re.sub(r'[^A-Za-z0-9_-]+', '_', str(value)).strip('_')
    return (cleaned or 'job')[:80]

def scene_prompt(job):
    text = str(job.get('prompt') or '').strip()
    return text or 'a dog and a cat talking'

def seed_for(job_id):
    base = int(_env('H3_SEED', str(H3_SEED_DEFAULT)))
    mixed = zlib.crc32(str(job_id).encode()) & 0x7FFFFFFF
    return (base ^ mixed) % (2**31 - 1) or base

def keyframe_prompt(prompt):
    return (
        'Photoreal cinematic still photo, continuous single shot, one camera angle, '
        'full-bleed edge-to-edge scene. No text, no captions. '
        + prompt
        + ' STRICT: one unbroken frame only. Forbidden: comic, manga, panel, '
        'triptych, collage, storyboard, grid, split-screen, stacked strips, '
        'repeated bands, borders between frames.'
    )

def motion_prompt(prompt, seconds=DEFAULT_SCENE_SECONDS):
    shown = int(round(float(seconds)))
    return (
        'The clip opens exactly on the keyframe and keeps the same characters, '
        'wardrobe, and setting. '
        f'This is one continuous shot of about {shown} seconds. '
        + SINGLE_FRAME_RULE
        + ' '
        + prompt.strip()
        + ' Gentle camera move that follows the action. No on-screen text, no '
        'subtitles, no scene cuts. Audio: natural sound and light ambience that '
        'matches the action, no voiceover.'
    )

def zimage_workflow(prompt, width, height, seed, prefix):
    """Official Z-Image Turbo graph: UNET + lumina2 CLIP + ae VAE, 8 steps."""
    return {
        '28': {'class_type': 'UNETLoader', 'inputs': {
            'unet_name': zimage_unet(), 'weight_dtype': 'default'}},
        '30': {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': zimage_clip(), 'type': 'lumina2', 'device': 'default'}},
        '29': {'class_type': 'VAELoader', 'inputs': {'vae_name': zimage_vae()}},
        '27': {'class_type': 'CLIPTextEncode', 'inputs': {
            'text': prompt, 'clip': ['30', 0]}},
        '33': {'class_type': 'ConditioningZeroOut', 'inputs': {
            'conditioning': ['27', 0]}},
        '13': {'class_type': 'EmptySD3LatentImage', 'inputs': {
            'width': int(width), 'height': int(height), 'batch_size': 1}},
        '11': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {
            'shift': 3, 'model': ['28', 0]}},
        '3': {'class_type': 'KSampler', 'inputs': {
            'seed': int(seed), 'steps': 8, 'cfg': 1,
            'sampler_name': 'res_multistep', 'scheduler': 'simple', 'denoise': 1,
            'model': ['11', 0], 'positive': ['27', 0], 'negative': ['33', 0],
            'latent_image': ['13', 0]}},
        '8': {'class_type': 'VAEDecode', 'inputs': {
            'samples': ['3', 0], 'vae': ['29', 0]}},
        '9': {'class_type': 'SaveImage', 'inputs': {
            'filename_prefix': prefix, 'images': ['8', 0]}},
    }

def h3_workflow(prompt, width, height, length, seed, image_name, prefix):
    """Flattened MiniMax H3 I2V graph from the verified colab-zimage run.

    Joint video/audio sample, no turbo LoRA. ``first_frame`` is required.
    """
    return {
        '6': {'class_type': 'UNETLoader', 'inputs': {
            'unet_name': h3_unet(), 'weight_dtype': 'default'}},
        '13': {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': h3_clip(), 'type': 'minimax', 'device': 'default'}},
        '11': {'class_type': 'VAELoader', 'inputs': {'vae_name': h3_video_vae()}},
        '24': {'class_type': 'VAELoader', 'inputs': {'vae_name': h3_audio_vae()}},
        '114': {'class_type': 'LoadImage', 'inputs': {'image': image_name}},
        '104': {'class_type': 'MiniMaxH3ImageToVideo', 'inputs': {
            'clip': ['13', 0],
            'vae': ['11', 0],
            'first_frame': ['114', 0],
            'last_frame': ['114', 0],
            'prompt': prompt,
            'width': int(width),
            'height': int(height),
            'length': int(length),
        }},
        '15': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': int(seed)}},
        '17': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'res_multistep'}},
        '16': {'class_type': 'BasicGuider', 'inputs': {
            'model': ['6', 0], 'conditioning': ['104', 0]}},
        '9': {'class_type': 'BasicScheduler', 'inputs': {
            'model': ['6', 0], 'scheduler': 'simple', 'steps': 20, 'denoise': 1}},
        '14': {'class_type': 'SamplerCustomAdvanced', 'inputs': {
            'noise': ['15', 0], 'guider': ['16', 0], 'sampler': ['17', 0],
            'sigmas': ['9', 0], 'latent_image': ['104', 1]}},
        '10': {'class_type': 'VAEDecode', 'inputs': {
            'samples': ['14', 0], 'vae': ['11', 0]}},
        '23': {'class_type': 'VAEDecodeAudio', 'inputs': {
            'samples': ['14', 0], 'vae': ['24', 0]}},
        '91': {'class_type': 'CreateVideo', 'inputs': {
            'images': ['10', 0], 'audio': ['23', 0], 'fps': 24}},
        '92': {'class_type': 'SaveVideo', 'inputs': {
            'video': ['91', 0], 'filename_prefix': prefix,
            'format': 'auto', 'codec': 'auto'}},
    }

def assert_i2v(workflow):
    types = [node.get('class_type') for node in workflow.values()]
    if 'MiniMaxH3ImageToVideo' not in types:
        raise RuntimeError('H3 workflow is missing MiniMaxH3ImageToVideo')
    if any(name and 'Wan' in name for name in types):
        raise RuntimeError('Wan Animate is not the Uncle text-prompt path')
    if 'LoraLoaderModelOnly' in types:
        raise RuntimeError('H3 turbo LoRA stays off for this path')
    node = next(n for n in workflow.values() if n['class_type'] == 'MiniMaxH3ImageToVideo')
    if not node['inputs'].get('first_frame') or not node['inputs'].get('last_frame'):
        raise RuntimeError('Refusing H3 text-to-video. A Z-Image keyframe is required.')
    if node['inputs']['first_frame'] != node['inputs']['last_frame']:
        raise RuntimeError('H3 first_frame and last_frame must be the same keyframe.')
    length = int(node['inputs']['length'])
    if length % 17 != 5 or length < 5 or length > H3_FRAMES_8S:
        raise RuntimeError(
            f'H3 length {length} is outside 5–8 seconds on the 17k+5 grid (max {H3_FRAMES_8S} frames)'
        )
    width = int(node['inputs']['width'])
    height = int(node['inputs']['height'])
    if (width, height) not in {(768, 768), (1344, 768), (768, 1344)}:
        raise RuntimeError('H3 frame size must be 768x768, 1344x768, or 768x1344')
    return node

def output_files(history_item):
    found = []
    for node_out in (history_item.get('outputs') or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in ('videos', 'gifs', 'images'):
            for entry in node_out.get(key) or []:
                if isinstance(entry, dict) and entry.get('filename'):
                    found.append(entry)
    return found

def pick_image(files):
    for entry in files:
        if str(entry['filename']).lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            return entry
    if files:
        return files[0]
    raise RuntimeError('ComfyUI finished without a keyframe image')

def pick_video(files):
    for entry in files:
        if str(entry['filename']).lower().endswith(('.mp4', '.webm', '.mkv', '.mov')):
            return entry
    raise RuntimeError('ComfyUI finished without an mp4')

def format_comfy_error(status):
    parts = []
    for msg in (status or {}).get('messages') or []:
        if isinstance(msg, list) and len(msg) > 1 and isinstance(msg[1], dict):
            parts.append(str(msg[1].get('exception_message') or msg[1]))
        else:
            parts.append(str(msg))
    return ('; '.join(parts) or 'ComfyUI error')[:500]


class ComfyClient:
    def __init__(self, base, timeout=None, poll=None, session=None):
        self.base = base.rstrip('/')
        self.timeout = float(timeout if timeout is not None else os.environ.get('COMFY_TIMEOUT', '1800'))
        self.poll = float(poll if poll is not None else os.environ.get('COMFY_POLL', '2'))
        self.session = session or requests.Session()
        self.client_id = uuid.uuid4().hex

    def upload_image(self, path):
        path = Path(path)
        with path.open('rb') as handle:
            response = self.session.post(
                f'{self.base}/upload/image',
                files={'image': (path.name, handle, 'image/png')},
                data={'overwrite': 'true', 'type': 'input'},
                timeout=120,
            )
        response.raise_for_status()
        data = response.json()
        name = data.get('name')
        if not name:
            raise RuntimeError('ComfyUI image upload returned no name')
        subfolder = data.get('subfolder') or ''
        return f'{subfolder}/{name}' if subfolder else name

    def submit(self, workflow):
        response = self.session.post(
            f'{self.base}/prompt',
            json={'prompt': workflow, 'client_id': self.client_id},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if data.get('node_errors'):
            raise RuntimeError(f'ComfyUI rejected the prompt: {str(data["node_errors"])[:500]}')
        if data.get('error'):
            raise RuntimeError(str(data['error'])[:500])
        prompt_id = data.get('prompt_id')
        if not prompt_id:
            raise RuntimeError('ComfyUI returned no prompt_id')
        return prompt_id

    def wait(self, prompt_id):
        deadline = time.monotonic() + self.timeout
        url = f'{self.base}/history/{prompt_id}'
        while True:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            payload = response.json()
            item = None
            if isinstance(payload, dict):
                item = payload.get(prompt_id)
                if item is None and payload.get('outputs'):
                    item = payload
            if item:
                status = item.get('status') or {}
                if status.get('status_str') == 'error':
                    raise RuntimeError(format_comfy_error(status))
                if status.get('completed') or status.get('status_str') == 'success':
                    return item
            if time.monotonic() >= deadline:
                raise TimeoutError(f'ComfyUI prompt {prompt_id} timed out')
            time.sleep(self.poll)

    def download(self, ref, dest):
        response = self.session.get(
            f'{self.base}/view',
            params={
                'filename': ref['filename'],
                'subfolder': ref.get('subfolder') or '',
                'type': ref.get('type') or 'output',
            },
            timeout=120,
        )
        response.raise_for_status()
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        if dest.stat().st_size < 32:
            raise RuntimeError(f'ComfyUI returned an empty file for {ref["filename"]}')
        return dest

    def free(self):
        try:
            self.session.post(
                f'{self.base}/free',
                json={'unload_models': True, 'free_memory': True},
                timeout=30,
            )
        except requests.RequestException as exc:
            print('comfy free skipped', exc)


def _fitted_keyframe(image, width, height):
    """Gate the SaveImage still, then scale/center-crop onto the H3 canvas.

    The check runs on the generated pixels. Scoring the upscaled canvas instead
    would flag smooth interpolation as a repeated band.
    """
    tiled, scores = keyframe_is_tiled(image)
    if tiled:
        return None, scores
    return fit_cover_image(image, width, height), scores

def render_comfy(job, out_path, client, workdir):
    """1024 Z-Image still, tile gate, then H3 I2V. KEYFRAME_IMAGE skips only the still call."""
    prompt = scene_prompt(job)
    width, height = frame_size(job.get('aspect_ratio'))
    seconds = scene_seconds(job)
    length = h3_length(seconds)
    seed = seed_for(job_identity(job))
    jid = safe_id(job_identity(job))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    external = os.environ.get('KEYFRAME_IMAGE', '').strip()
    if external:
        source = Path(external)
        if not source.is_file():
            raise FileNotFoundError(f'KEYFRAME_IMAGE not found: {external}')
        print('keyframe hook', source)
        fitted, scores = _fitted_keyframe(_Image().open(source), width, height)
        if fitted is None:
            raise RuntimeError(f'KEYFRAME_IMAGE looks tiled and will not be sent to H3: {scores}')
    else:
        zw, zh = zimage_latent_size()
        fitted = None
        scores = None
        best = None
        for attempt in range(KEYFRAME_ATTEMPTS):
            attempt_seed = (seed + attempt * 1_000_003) % (2**31 - 1) or seed
            still = zimage_workflow(
                keyframe_prompt(prompt), zw, zh, attempt_seed, f'uncle_zimage/{jid}_{attempt}',
            )
            prompt_id = client.submit(still)
            item = client.wait(prompt_id)
            raw_path = client.download(
                pick_image(output_files(item)), workdir / f'{jid}_zimage_{attempt}.png',
            )
            with _Image().open(raw_path) as raw:
                image = raw.copy()
            fitted, scores = _fitted_keyframe(image, width, height)
            max_score = max(value for key, value in scores.items() if not key.startswith('portrait_'))
            if best is None or max_score < best[0]:
                best = (max_score, image, attempt_seed, scores)
            if fitted is None:
                print('keyframe tiled', scores, f"portrait={scores['portrait_hits']}", 'seed', attempt_seed)
                continue
            seed = attempt_seed
            print('keyframe ok', scores, 'seed', seed)
            break
        client.free()
        if fitted is None:
            _, image, seed, scores = best
            fitted = salvage_keyframe(image, scores, width, height)
            print('keyframe salvage', scores, 'seed', seed, 'size', fitted.size)
    key_path = workdir / f'{jid}_key.png'
    fitted.save(key_path, format='PNG')
    if fitted.size != (width, height):
        raise RuntimeError(f'Keyframe fit produced {fitted.size}, expected {(width, height)}')
    image_name = client.upload_image(key_path)
    video = h3_workflow(
        motion_prompt(prompt, seconds), width, height, length, seed, image_name,
        f'video/uncle_h3_{jid}',
    )
    assert_i2v(video)
    print(f'render {width}x{height} frames={length} fps=24 seed={seed} zimage={ZIMAGE_SIDE}')
    prompt_id = client.submit(video)
    item = client.wait(prompt_id)
    return client.download(pick_video(output_files(item)), out_path)


def check_comfy(base):
    base = base.rstrip('/')
    response = requests.get(f'{base}/object_info/MiniMaxH3ImageToVideo', timeout=30)
    response.raise_for_status()
    data = response.json()
    if 'MiniMaxH3ImageToVideo' not in data:
        raise SystemExit(f'ComfyUI at {base} has no MiniMaxH3ImageToVideo node')
    stats = requests.get(f'{base}/system_stats', timeout=30)
    stats.raise_for_status()
    print('comfy ok', base)
    return data


class _MockState:
    def __init__(self, jobs=None):
        self.prompts = []
        self.uploads = []
        self.history = {}
        self.fail_next = False
        self.frees = 0
        self.views = []
        self.jobs = list(jobs or [])
        self.results = []
        self.claims = []
        self.key_pngs = []
        self._clean_png = None

    def clean_png(self):
        if self._clean_png is None:
            self._clean_png = png_bytes(single_scene_image(64, 64))
        return self._clean_png


def _json_response(handler, code, payload):
    body = json.dumps(payload).encode()
    handler.send_response(code)
    handler.send_header('Content-Type', 'application/json')
    handler.send_header('Content-Length', str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _raw_response(handler, code, body, content_type):
    handler.send_response(code)
    handler.send_header('Content-Type', content_type)
    handler.send_header('Content-Length', str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler):
    length = int(handler.headers.get('Content-Length') or 0)
    return handler.rfile.read(length) if length else b''


class _ComfyHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        state = self.server.state
        parsed = urlparse(self.path)
        if parsed.path.startswith('/history/'):
            prompt_id = parsed.path.rsplit('/', 1)[-1]
            item = state.history.get(prompt_id)
            _json_response(self, 200, {prompt_id: item} if item else {})
            return
        if parsed.path == '/view':
            name = (parse_qs(parsed.query).get('filename') or [''])[0]
            state.views.append(name)
            if name.lower().endswith('.mp4'):
                blob = b'FAKEMP4' + b'0' * 128
            elif state.key_pngs:
                blob = state.key_pngs.pop(0)
            else:
                blob = state.clean_png()
            _raw_response(self, 200, blob, 'application/octet-stream')
            return
        if parsed.path == '/object_info/MiniMaxH3ImageToVideo':
            _json_response(self, 200, {'MiniMaxH3ImageToVideo': {'input': {}}})
            return
        if parsed.path == '/system_stats':
            _json_response(self, 200, {'system': {'comfyui_version': 'mock'}})
            return
        _json_response(self, 404, {'error': 'not found'})

    def do_POST(self):
        state = self.server.state
        body = _read_body(self)
        parsed = urlparse(self.path)
        if parsed.path == '/prompt':
            workflow = json.loads(body.decode()).get('prompt') or {}
            prompt_id = f'p{len(state.prompts) + 1}'
            state.prompts.append(workflow)
            if state.fail_next:
                state.fail_next = False
                state.history[prompt_id] = {
                    'outputs': {},
                    'status': {
                        'status_str': 'error',
                        'completed': False,
                        'messages': [['execution_error', {'exception_message': 'boom'}]],
                    },
                }
            else:
                is_video = any(node.get('class_type') == 'SaveVideo' for node in workflow.values())
                filename = 'out.mp4' if is_video else 'key.png'
                key = 'videos' if is_video else 'images'
                state.history[prompt_id] = {
                    'outputs': {'out': {key: [{
                        'filename': filename, 'subfolder': '', 'type': 'output',
                    }]}},
                    'status': {'status_str': 'success', 'completed': True},
                }
            _json_response(self, 200, {'prompt_id': prompt_id})
            return
        if parsed.path == '/upload/image':
            state.uploads.append(body)
            _json_response(self, 200, {'name': 'key.png', 'subfolder': '', 'type': 'input'})
            return
        if parsed.path == '/free':
            state.frees += 1
            _json_response(self, 200, {'ok': True})
            return
        _json_response(self, 404, {'error': 'not found'})


class _RelayHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        state = self.server.state
        body = _read_body(self)
        parsed = urlparse(self.path)
        if parsed.path == '/api/colab/next':
            state.claims.append(self.headers.get('X-Worker-Token', ''))
            job = state.jobs.pop(0) if state.jobs else None
            _json_response(self, 200, {'job': job})
            return
        if parsed.path.startswith('/api/colab/result/'):
            job_id = parsed.path.rsplit('/', 1)[-1]
            state.results.append({
                'job_id': job_id,
                'body': body,
                'token': self.headers.get('X-Worker-Token', ''),
            })
            _json_response(self, 200, {'ok': True})
            return
        _json_response(self, 404, {'error': 'not found'})


class _RunningServer:
    def __init__(self, handler, state):
        self.state = state
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.httpd.state = state
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self):
        host, port = self.httpd.server_address
        return f'http://{host}:{port}'

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        return False


def serve_mock_comfy():
    """In-process ComfyUI HTTP mock for --dry-run and unit tests. No GPU."""
    return _RunningServer(_ComfyHandler, _MockState())


def serve_mock_relay(jobs):
    return _RunningServer(_RelayHandler, _MockState(jobs))


def dry_run(prompt='a dog and a cat talking on a porch'):
    """Render one synthetic job against the Comfy HTTP mock. Does not claim the relay."""
    job = {
        'job_id': 'colab-dryrun',
        'prompt': prompt,
        'aspect_ratio': '1:1',
        'duration': 8,
    }
    with serve_mock_comfy() as comfy:
        out = Path(os.environ.get('DRY_RUN_OUT', '')) if os.environ.get('DRY_RUN_OUT') else None
        if out is None:
            out = Path(os.getcwd()) / 'dry_run_out.mp4'
        work = out.parent / 'dry_run_work'
        render_comfy(job, out, ComfyClient(comfy.url), work)
        if len(comfy.state.prompts) != 2:
            raise RuntimeError('dry-run did not submit Z-Image and H3')
        print('dry-run prompts', len(comfy.state.prompts), 'uploads', len(comfy.state.uploads))
        return out
