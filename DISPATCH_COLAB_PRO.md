# Uncle Colab worker — Z-Image Turbo then MiniMax H3

## Goal

Drain the Colab relay into watchable scene clips for Uncle Generate.

Each claimed job:

1. Builds a keyframe with Z-Image Turbo at **1024×1024**.
2. Rejects a stacked comic/storyboard still (vertical autocorrelation at H/2 or H/3 above 0.85) and rerolls the seed, up to 3 times.
3. Scales and center-crops the clean still onto the H3 canvas. Runs MiniMax H3 image-to-video (`fl2va`) with that image as both `first_frame` and `last_frame`.
4. Uploads the mp4 to the relay.

Per scene on an A100 the **video** canvas is:

- **9:16** (dog-cat and the studio default): **768×1344**.
- **16:9**: **1344×768**.
- **1:1** or anything else: **768×768**.

Do not point EmptySD3 at those sizes. 768 and 768×1344 are what produced the three stacked dog-cat bands. The fix is a clean 1024 keyframe, not cropping the finished mp4.

Length follows `job['duration']`. Episodes are 8 seconds, so the worker submits **192 frames** at 24 fps (`h3_length(8)`). 5 seconds is 124 frames (120 is not on the `17k+5` grid). Anything above 8 seconds, including 15, is capped at 192 frames. Do not render the native 15 second / 362-frame length. Concatenate the scene files afterward.

## Do not

- Do not set `INSTALL_MODEL`. `damo-vilab/text-to-video-ms-1.7b` is not a generate path.
- Do not use Wan Animate for these text-prompt scenes. Wan is for character replace.
- Do not call OpenRouter `/videos`.
- Do not start a second Colab runtime while a worker is polling. The worker takes an exclusive flock (`COLAB_WORKER_LOCK`, default `/tmp/uncle_colab_worker.lock`), the same idea as `run_h3.sh`.

## Relay

Prefer `COLAB_RELAY_URL`. `UNCLE_URL` is only a fallback.

- Relay: `https://uncle-colab-relay.facebook-chatbot.workers.dev`
- Header: `X-Worker-Token: $COLAB_WORKER_TOKEN` (from the Uncle `.env`, do not paste it into chat)
- Claim: `POST /api/colab/next`
- Upload: `POST /api/colab/result/{job_id}` multipart field `video` (`out.mp4`) and form `status=done`. The relay stores the bytes before it marks the job done.
- Failure: same URL, multipart fields `status=failed` and `error`. A failure is never recorded as done with an empty file.

## What the Colab runtime already needs

One A100 notebook with ComfyUI listening on `127.0.0.1:8188`, laid out like `colab-zimage`. The worker does not install ComfyUI and does not download weights.

H3 (verified run, Hugging Face `Comfy-Org/MiniMax-H3` revision `e5eb578a89295337b8ff433a035929ce0279e0b6`):

- `models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors`
- `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `models/vae/minimax_h3_video_vae_fp16.safetensors`
- `models/vae/minimax_h3_audio_vae_fp32.safetensors`

Z-Image Turbo (`Comfy-Org/z_image_turbo`):

- `models/diffusion_models/z_image_turbo_bf16.safetensors`
- `models/text_encoders/qwen_3_4b.safetensors`
- `models/vae/ae.safetensors`

No turbo LoRA. H3: 20 steps, `res_multistep`, scheduler `simple`, denoise 1, BasicGuider, joint video and audio into `CreateVideo` / `SaveVideo`. Z-Image: 8 steps, cfg 1, AuraFlow shift 3.

Drive folder `1qwFEx1OTbP3GyiqEvjaFp2Ym-3j4jjKY` was the intended model cache and was **empty** on the verified run. Do not block on it. If a later sync actually contains those filenames, mount Drive and add that directory as a ComfyUI extra model path. Otherwise use the copies already in the runtime's `models/` tree.

If this runtime's ComfyUI was installed from the current official template, the H3 diffusion and video VAE may be the int8 names (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, `minimax_h3_video_vae_int8_convrot.safetensors`). Those are the same Comfy-Org repo. Point `H3_UNET` and `H3_VIDEO_VAE` at the files that are actually on disk. Do not substitute Wan or damo.

## One cell

Runtime → A100. Start ComfyUI first (the existing colab-zimage startup). Then:

```python
!git clone -b cursor/h3-zimage-worker-fb29 --single-branch https://github.com/Krudony/Uncle_AIVideoGenerator.git /content/Uncle_AIVideoGenerator
!curl -fsS http://127.0.0.1:8188/system_stats
import os
os.environ['COLAB_RELAY_URL'] = 'https://uncle-colab-relay.facebook-chatbot.workers.dev'
os.environ['COLAB_WORKER_TOKEN'] = 'PASTE_FROM_ENV'
os.environ['BACKEND'] = 'comfy'
os.environ['COMFY_URL'] = 'http://127.0.0.1:8188'
!python /content/Uncle_AIVideoGenerator/uncle_colab_worker.py --check
!python /content/Uncle_AIVideoGenerator/uncle_colab_worker.py
```

Stop the Colab runtime when the queue is drained, including after failures.

Optional: `KEYFRAME_IMAGE` is a still you already made with Z-Image. The worker skips only the keyframe call and still runs H3 I2V. It will not do text-to-video.

`MAX_EMPTY_POLLS=6` stops after six empty claims. Leave it unset to keep polling.

## Bridge smoke only

`BACKEND=ffmpeg` uploads a 2 second blue mp4 and does not touch ComfyUI. Use it to test claim/upload. It is not a real clip.

## Check the next clips for stacked panels

After this branch is on the Colab runtime, generate one **1:1** scene and one **9:16** scene. Grab a frame around 2 seconds. It should be one full-bleed picture. It should not be three stacked copies of the same dog and cat. `ffprobe` should report 768×768 or 768×1344, about 8 seconds.

## Local dry run (no GPU)

```sh
python -m unittest tests.test_colab_worker -v
python uncle_colab_worker.py --dry-run
```
