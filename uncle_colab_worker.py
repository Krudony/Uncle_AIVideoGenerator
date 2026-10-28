#!/usr/bin/env python3
"""Poll Uncle Colab queue, make a short clip, upload. For Colab or local.

Env:
  UNCLE_URL              public base URL (cloudflared)
  COLAB_WORKER_TOKEN     must match server .env
  INSTALL_MODEL=0        default: ffmpeg blue smoke clip (2s)
  INSTALL_MODEL=1        try damo-vilab/text-to-video-ms-1.7b on GPU
"""
import os, time, tempfile, subprocess
from getpass import getpass

try:
    import requests
except ImportError:
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'requests'])
    import requests

UNCLE_URL = (os.environ.get('UNCLE_URL') or input('Uncle public URL: ')).rstrip('/')
TOKEN = os.environ.get('COLAB_WORKER_TOKEN') or getpass('COLAB_WORKER_TOKEN: ')
H = {'X-Worker-Token': TOKEN}
USE_MODEL = os.environ.get('INSTALL_MODEL', '0') == '1'

def claim():
    r = requests.post(f'{UNCLE_URL}/api/colab/next', headers=H, timeout=60)
    r.raise_for_status()
    return r.json().get('job')

def fail(job_id, err):
    requests.post(
        f'{UNCLE_URL}/api/colab/result/{job_id}',
        headers=H,
        data={'status': 'failed', 'error': str(err)[:500]},
        timeout=60,
    )

def upload(job_id, path):
    with open(path, 'rb') as f:
        r = requests.post(
            f'{UNCLE_URL}/api/colab/result/{job_id}',
            headers=H,
            files={'video': ('out.mp4', f, 'video/mp4')},
            data={'status': 'done'},
            timeout=300,
        )
    r.raise_for_status()
    return r.json()

def make_clip(prompt, out_path, seconds=8):
    if not USE_MODEL:
        subprocess.check_call([
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', f'color=c=blue:s=640x360:d={max(2, min(seconds, 8))}',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path,
        ])
        return
    import torch, numpy as np, imageio
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        'damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    clips = []
    n = max(2, min(3, (seconds + 3) // 4))
    for _ in range(n):
        frames = pipe(prompt, num_inference_steps=25, num_frames=16).frames[0]
        if np.stack([np.asarray(f) for f in frames]).mean() < 10:
            raise RuntimeError('clip looks all-black')
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        imageio.mimwrite(tmp, frames, fps=8)
        clips.append(tmp)
    lst = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
    for c in clips:
        lst.write(f"file '{c}'\n")
    lst.close()
    subprocess.check_call(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', lst.name, '-c', 'copy', out_path])

print('polling', UNCLE_URL, 'model' if USE_MODEL else 'ffmpeg-smoke')
while True:
    job = claim()
    if not job:
        time.sleep(5)
        continue
    jid = job['id']
    print('claimed', jid, (job.get('prompt') or '')[:80])
    out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    try:
        make_clip(job.get('prompt') or 'a dog and a cat talking', out, int(job.get('duration') or 8))
        print(upload(jid, out))
    except Exception as e:
        print('fail', e)
        fail(jid, e)
