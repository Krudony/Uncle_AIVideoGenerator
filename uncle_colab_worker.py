#!/usr/bin/env python3
"""Poll the Uncle Colab relay, render a clip, upload it.

Real clips (default, BACKEND=comfy) on a Colab Pro A100 that already has ComfyUI:

  1. Z-Image Turbo still from the scene prompt
  2. MiniMax H3 fl2va image-to-video at the same frame size as the still:
     768x768, 1344x768 for 16:9, or 768x1344 for 9:16.
     Length follows job duration, capped at 8 seconds (192 frames at 24 fps).

BACKEND=ffmpeg writes a blue smoke mp4 for bridge tests only.
INSTALL_MODEL / damo-vilab is not a generate path.

Env:
  COLAB_RELAY_URL       preferred public relay base (falls back to UNCLE_URL)
  COLAB_WORKER_TOKEN    must match the relay
  BACKEND               comfy (default) or ffmpeg
  COMFY_URL             default http://127.0.0.1:8188
  KEYFRAME_IMAGE        optional existing Z-Image still; H3 still runs as I2V
  H3_SECONDS            fallback seconds when the job has no duration (default 8, max 8)
  COLAB_WORKER_LOCK     flock path, default /tmp/uncle_colab_worker.lock
  MAX_EMPTY_POLLS       stop after this many empty claims (0 keeps polling)

Do not start a second Colab runtime while this process holds the lock.
"""
import fcntl
import os
import subprocess
import sys
import tempfile
import time
from getpass import getpass
from pathlib import Path

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'requests'])
    import requests

import colab_comfy

def guard_env():
    install = os.environ.get('INSTALL_MODEL', '').strip()
    if install == '1':
        raise SystemExit(
            'INSTALL_MODEL=1 is disabled. damo-vilab is not a generate path. '
            'Use BACKEND=comfy for a Z-Image Turbo keyframe and MiniMax H3 I2V.'
        )
    if install:
        print('INSTALL_MODEL is ignored. Real gen is BACKEND=comfy. Smoke is BACKEND=ffmpeg.')
    backend = os.environ.get('BACKEND', 'comfy').strip().lower() or 'comfy'
    if backend in {'damo', 'diffusers'}:
        raise SystemExit(f'BACKEND={backend} is disabled. Use BACKEND=comfy.')
    if backend not in {'comfy', 'ffmpeg'}:
        raise SystemExit('BACKEND must be comfy or ffmpeg.')
    return backend

def resolve_relay():
    base = (os.environ.get('COLAB_RELAY_URL') or os.environ.get('UNCLE_URL') or '').strip()
    if not base and sys.stdin.isatty():
        base = input('Uncle public URL: ').strip()
    if not base:
        raise SystemExit('Set COLAB_RELAY_URL (preferred) or UNCLE_URL.')
    return base.rstrip('/')

def resolve_token():
    token = (os.environ.get('COLAB_WORKER_TOKEN') or '').strip()
    if not token and sys.stdin.isatty():
        token = getpass('COLAB_WORKER_TOKEN: ').strip()
    if not token:
        raise SystemExit('Set COLAB_WORKER_TOKEN.')
    return token

def acquire_lock(path):
    handle = open(path, 'a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(
            f'Another Uncle Colab worker holds {path}. '
            'Do not start a second Colab session.'
        )
    handle.write(str(os.getpid()))
    handle.flush()
    return handle

def claim(relay_url, headers):
    response = requests.post(f'{relay_url}/api/colab/next', headers=headers, timeout=60)
    response.raise_for_status()
    return response.json().get('job')

def fail(relay_url, headers, job_id, err):
    response = requests.post(
        f'{relay_url}/api/colab/result/{job_id}',
        headers=headers,
        data={'status': 'failed', 'error': str(err)[:500]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()

def upload(relay_url, headers, job_id, path):
    with open(path, 'rb') as handle:
        response = requests.post(
            f'{relay_url}/api/colab/result/{job_id}',
            headers=headers,
            files={'video': ('out.mp4', handle, 'video/mp4')},
            data={'status': 'done'},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()

def render_ffmpeg(out_path):
    subprocess.check_call(
        [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', 'color=c=blue:s=640x360:d=2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def render_job(job, out_path, backend, comfy_url, workdir):
    if backend == 'ffmpeg':
        render_ffmpeg(out_path)
        return out_path
    client = colab_comfy.ComfyClient(comfy_url or os.environ.get('COMFY_URL', 'http://127.0.0.1:8188'))
    return colab_comfy.render_comfy(job, out_path, client, workdir)

def process_available_jobs(relay_url, token, backend='comfy', comfy_url=None, max_jobs=None, max_empty=0):
    headers = {'X-Worker-Token': token}
    done = failed = empty = seen = 0
    idle = float(os.environ.get('POLL_IDLE', '5'))
    while True:
        if max_jobs is not None and seen >= max_jobs:
            break
        job = claim(relay_url, headers)
        if not job:
            empty += 1
            print('empty', empty, '/', max_empty or 'inf', 'done', done, 'fail', failed)
            if max_empty and empty >= max_empty:
                break
            time.sleep(idle)
            continue
        empty = 0
        seen += 1
        jid = colab_comfy.job_identity(job)
        print('claimed', jid, (job.get('prompt') or '')[:80])
        try:
            with tempfile.TemporaryDirectory(prefix='uncle_clip_') as tmp:
                out = Path(tmp) / 'out.mp4'
                render_job(job, out, backend, comfy_url, tmp)
                print(upload(relay_url, headers, jid, out))
            done += 1
        except Exception as exc:
            print('fail', exc)
            failed += 1
            try:
                fail(relay_url, headers, jid, exc)
            except Exception as report_exc:
                print('fail report error', report_exc)
    return done, failed

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--dry-run' in argv:
        path = colab_comfy.dry_run()
        print('dry-run ok', path, 'bytes', Path(path).stat().st_size)
        return 0
    backend = guard_env()
    if '--check' in argv:
        colab_comfy.check_comfy(os.environ.get('COMFY_URL', 'http://127.0.0.1:8188'))
        return 0
    lock = acquire_lock(os.environ.get('COLAB_WORKER_LOCK', '/tmp/uncle_colab_worker.lock'))
    try:
        relay = resolve_relay()
        token = resolve_token()
        label = 'zimage+h3' if backend == 'comfy' else 'ffmpeg-smoke'
        print('polling', relay, 'backend=' + backend, label)
        done, failed = process_available_jobs(
            relay,
            token,
            backend=backend,
            comfy_url=os.environ.get('COMFY_URL', 'http://127.0.0.1:8188'),
            max_empty=int(os.environ.get('MAX_EMPTY_POLLS', '0') or '0'),
        )
        print('DONE_SUMMARY', done, failed)
        return 0 if failed == 0 else 1
    finally:
        lock.close()

if __name__ == '__main__':
    sys.exit(main())
