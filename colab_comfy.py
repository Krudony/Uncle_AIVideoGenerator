"""Z-Image Turbo keyframe, then MiniMax H3 image-to-video, via a local ComfyUI.

This matches the verified colab-zimage H3 run (fl2va, no turbo LoRA) and the
official Z-Image Turbo template. The worker never calls OpenRouter /videos.
"""
import json
import os
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
        'Single cinematic still frame of one moment. Same characters and setting '
        'as the scene, no text, no captions, no collage, no split screen. '
        + prompt.strip()
    )

def motion_prompt(prompt, seconds=DEFAULT_SCENE_SECONDS):
    shown = int(round(float(seconds)))
    return (
        'The clip opens exactly on the keyframe and keeps the same characters, '
        'wardrobe, and setting. '
        f'This is one continuous shot of about {shown} seconds. '
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
    if not node['inputs'].get('first_frame'):
        raise RuntimeError('Refusing H3 text-to-video. A Z-Image keyframe is required.')
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


def render_comfy(job, out_path, client, workdir):
    """Keyframe (Z-Image) then H3 I2V. KEYFRAME_IMAGE skips only the still call."""
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
        key_path = Path(external)
        if not key_path.is_file():
            raise FileNotFoundError(f'KEYFRAME_IMAGE not found: {external}')
        print('keyframe hook', key_path)
    else:
        still = zimage_workflow(
            keyframe_prompt(prompt), width, height, seed, f'uncle_zimage/{jid}',
        )
        prompt_id = client.submit(still)
        item = client.wait(prompt_id)
        key_path = client.download(pick_image(output_files(item)), workdir / f'{jid}_key.png')
        client.free()
    image_name = client.upload_image(key_path)
    video = h3_workflow(
        motion_prompt(prompt, seconds), width, height, length, seed, image_name,
        f'video/uncle_h3_{jid}',
    )
    assert_i2v(video)
    print(f'render {width}x{height} frames={length} fps=24 seed={seed}')
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
            blob = b'FAKEMP4' + b'0' * 128 if name.lower().endswith('.mp4') else b'\x89PNG' + b'0' * 128
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
