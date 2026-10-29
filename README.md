# Uncle AI Video Generator

เว็บแอปสำหรับวางแผนและจัดการการผลิตวิดีโอเป็น Episode ตั้งแต่คิดไอเดีย สร้างและปรับ Storyboard สร้างคลิปและเสียง ไปจนถึงรวมไฟล์และตรวจดูต้นทุนการใช้งาน OpenRouter

## ภาพตัวอย่างหน้าจอ

### Studio — Scene และคลิปที่สร้างสำเร็จ

![หน้า Studio แสดง Scene และคลิปวิดีโอที่สร้างสำเร็จ](SCREENSHOT1.png)

### Dashboard ต้นทุนคลิป

![Dashboard สรุปต้นทุนคลิปและ OpenRouter usage](SCREENSHOT3.png)

## ความต้องการของระบบ

- Windows, macOS หรือ Linux
- Python 3.10 ขึ้นไป
- บัญชีและ API key ของ [OpenRouter](https://openrouter.ai/)
- FFmpeg และ FFprobe ที่เรียกใช้ได้จาก PATH สำหรับประกอบวิดีโอและปรับความยาวไฟล์เสียง

## ติดตั้งและเปิดใช้งาน

เปิด PowerShell หรือเทอร์มินัลในโฟลเดอร์โปรเจค แล้วติดตั้งแพ็กเกจ Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

หากใช้ macOS หรือ Linux ให้เปิด virtual environment ด้วยคำสั่งนี้แทน:

```sh
source .venv/bin/activate
```

สร้างไฟล์ `.env` ที่โฟลเดอร์หลักของโปรเจค แล้วใส่ API key:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-ใส่คีย์ของคุณ
```

จากนั้นเปิดแอป:

```sh
python app.py
```

เปิดเว็บที่ [http://127.0.0.1:5000](http://127.0.0.1:5000)

ตั้งค่า API key จากหน้า **API Keys** ในเว็บก็ได้ โดยเพิ่ม key ตั้งชื่อ และเลือก key ที่ต้องการใช้เป็น Active แอปเก็บ key ที่เพิ่มไว้ใน `instance/api_keys.json` และไม่ส่งค่า key กลับไปแสดงในหน้าเว็บ

## ติดตั้ง FFmpeg และ FFprobe

การติดตั้ง FFmpeg และ FFprobe เป็นทางเลือก หากไม่ต้องการใช้ฟีเจอร์ประกอบ/รวมวิดีโอหรือปรับความยาวไฟล์ Sound ก็ไม่จำเป็นต้องดาวน์โหลด และยังใช้ฟีเจอร์อื่นของแอปได้ตามปกติ หากต้องการใช้ฟีเจอร์เหล่านี้ FFprobe มาพร้อมชุดติดตั้ง FFmpeg ในวิธีด้านล่าง และทั้งสองโปรแกรมต้องเรียกใช้ได้จาก PATH

### Windows

แนะนำให้ใช้ WinGet ใน PowerShell (แพ็กเกจ `Gyan.FFmpeg` ใน [คลัง WinGet](https://github.com/microsoft/winget-pkgs/tree/master/manifests/g/Gyan/FFmpeg)):

```powershell
winget install --id Gyan.FFmpeg --exact
```

ถ้าเครื่องไม่มี WinGet ให้ดาวน์โหลด Windows build จากหน้า [FFmpeg สำหรับ Windows](https://www.gyan.dev/ffmpeg/builds/) ซึ่งลิงก์จากหน้า [ดาวน์โหลด FFmpeg อย่างเป็นทางการ](https://ffmpeg.org/download.html) จากนั้นแตกไฟล์ แล้วเพิ่มโฟลเดอร์ `bin` ที่มี `ffmpeg.exe` และ `ffprobe.exe` เข้า PATH ของ Windows

### macOS

ถ้าติดตั้ง Homebrew แล้ว ให้เปิด Terminal แล้วรัน:

```sh
brew install ffmpeg
```

รายละเอียดแพ็กเกจดูได้ที่ [สูตร FFmpeg ของ Homebrew](https://formulae.brew.sh/formula/ffmpeg)

### Ubuntu / Debian

เปิด Terminal แล้วติดตั้งแพ็กเกจจากคลังของระบบ:

```sh
sudo apt update
sudo apt install ffmpeg
```

ดูข้อมูลแพ็กเกจได้ที่ [Ubuntu Packages: ffmpeg](https://packages.ubuntu.com/ffmpeg)

### ตรวจสอบการติดตั้ง

ปิดแล้วเปิด Terminal หรือ PowerShell ใหม่ จากนั้นตรวจสอบว่าเรียกใช้ได้ทั้งคู่:

```sh
ffmpeg -version
ffprobe -version
```

ถ้าขึ้นข้อความว่าไม่พบคำสั่ง ให้ตรวจสอบ PATH แล้วเปิด Terminal หรือแอปใหม่อีกครั้ง เพื่อให้ Flask โหลดค่า PATH ล่าสุด

## วิธีใช้งาน

### Studio — สร้างและจัดการวิดีโอ

หน้าแรกใช้เลือก Episode และจัดการ Scene แต่ละรายการได้ แก้ไขหัวข้อ Prompt และบทพากย์ได้ โดยระบบบันทึกการแก้ไขให้อัตโนมัติ

กด **Load video models** เพื่อโหลดโมเดลสร้างวิดีโอที่ใช้ได้ แล้วตรวจสอบราคา ความยาว และความละเอียดที่รองรับก่อนสร้างคลิป สามารถสร้างทีละ Scene หรือเลือกหลาย Scene เพื่อสั่ง Generate แบบกลุ่มได้ ระบบตรวจสถานะงานและแสดงคลิปที่เสร็จพร้อมให้ดาวน์โหลดโดยอัตโนมัติ

ในหน้าเดียวกันยังมี AI Improve สำหรับปรับ Scene, เครื่องมือร่าง Description และแฮชแท็กสำหรับโพสต์, ส่งออก Storyboard JSON และ Assemble วิดีโอแบบไม่มีเสียง

### คิดไอเดีย EP

หน้า **คิดไอเดีย** เป็นแบบฟอร์มทีละขั้น เริ่มจากชื่อคลิป เลือกสไตล์ กลุ่มผู้ชม แพลตฟอร์ม ภาษา และรายละเอียดแนวทาง จากนั้นให้ AI สร้างโครง Episode และ Scene มาแก้ไขก่อนบันทึกเป็น Episode ใหม่ แบบฟอร์มฉบับร่างเก็บไว้ในเบราว์เซอร์

### จัดการ Episodes

หน้า **Episodes** ใช้สร้าง แก้ไข และลบ Episode รวมถึงเพิ่มหรือลบ Scene ได้ หาก Scene มีงานสร้างวิดีโอที่กำลังทำงานอยู่ ต้องรอให้งานนั้นจบก่อนจึงจะลบได้

### คลังวิดีโอ

หน้า **คลังวิดีโอ** แสดงคลิปของแต่ละ Episode และรองรับการเพิ่มไฟล์วิดีโอหลายไฟล์ในครั้งเดียว (MP4, MOV, MKV, WebM และ M4V) จัดลำดับคลิป แล้วประกอบคลิปที่เลือกเป็นไฟล์ MP4 ได้ การอัปโหลดหนึ่งครั้งรองรับไม่เกิน 100 ไฟล์ และขนาดคำขอรวมไม่เกิน 1 GB

### สร้าง Sound และใส่เสียงพื้นหลัง

หน้า **Sound** ใช้ค้นหาโมเดล OpenRouter ที่สร้างเสียงจาก Prompt ได้ เลือกระยะเวลาเป้าหมายได้ 1–120 วินาที แล้วฟังหรือดาวน์โหลดไฟล์ WAV ที่สร้างไว้ เสียงที่สร้างสำเร็จจะถูกบันทึกไว้ในประวัติ

ใน Studio ส่วน **ประกอบวิดีโอพร้อม Background Sound** ให้เลือก Sound จากประวัติและกด Assemble ระบบจะรวมคลิปทั้งหมดในคลังของ Episode ตามลำดับที่จัดไว้ วางเสียงที่เลือกเป็น Background Sound โดยเล่นครั้งเดียวไม่วนซ้ำ และ Fade out ในช่วงท้ายไม่เกิน 3 วินาที หากเสียงสั้นกว่าวิดีโอ ภาพจะเล่นต่อจนจบ

### ตั้งค่า API Keys และโมเดล

หน้า **API Keys** ใช้เพิ่มและสลับ API key พร้อมบอกว่า key ใดกำลัง Active นอกจากนี้ยังเลือกโมเดลแยกตามงานได้ ได้แก่:

- คิดไอเดียและสร้างแผน EP
- ปรับปรุง Scene และ Prompt
- AI Improve ทั้ง Episode
- คุยกับ AI ในหน้าวิดีโอ
- เขียน Description สำหรับโพสต์
- โมเดลเริ่มต้นสำหรับสร้างวิดีโอและ Sound

รายการโมเดลดึงจาก OpenRouter การเลือกโมเดลเริ่มต้นสำหรับวิดีโอหรือ Sound ยังเปลี่ยนเป็นโมเดลอื่นได้ในหน้าสร้างสื่อนั้น

### ต้นทุนคลิป

หน้า **ต้นทุนคลิป** สรุปคลิปตาม Episode และช่วงเวลา แสดงค่าใช้จ่ายจริงที่ OpenRouter รายงานเมื่อมีข้อมูล พร้อม usage วันนี้ สัปดาห์ เดือน และสะสมของ API key ที่กำลัง Active ยอด usage ของ key อาจรวมการใช้งานฟีเจอร์อื่นด้วย จึงแยกออกจากยอดต้นทุนวิดีโอรายคลิป

ใช้ปุ่ม Sync เพื่อขอข้อมูล usage ของงานเก่า ค่าใช้จ่ายบางรายการอาจไม่มีให้เรียกดูแล้ว หากงานหมดอายุหรือสร้างด้วย API key คนละตัว

## ต้นทุนและข้อควรทราบ

- การสร้างข้อความ เสียง และวิดีโอเรียกใช้ API ภายนอกและอาจมีค่าใช้จ่าย ตรวจสอบราคาและผู้ให้บริการใน OpenRouter ก่อนเริ่มงาน
- การสร้างวิดีโอเป็นงานแยกตาม Scene และอาจใช้เวลารอ ระบบไม่เริ่มสร้างคลิปที่ต้องเสียเงินเองโดยไม่มีคำสั่งจากผู้ใช้
- ราคาในหน้า Studio เป็นค่าประมาณ ส่วนหน้า **ต้นทุนคลิป** แสดงต้นทุนจริงเมื่อ API ส่งข้อมูลกลับมา
- FFmpeg และ FFprobe ต้องติดตั้งและอยู่ใน PATH เพื่อประกอบคลิปหรือปรับความยาว Sound ได้
- การสร้างวิดีโอแต่ละ Scene แยกจากกัน จึงอาจมีความต่อเนื่องของตัวละครหรือภาพไม่สมบูรณ์ ควรตรวจคลิปและ Storyboard ก่อนนำไปเผยแพร่
- ข้อความหรือข้อมูลอ้างอิงที่ป้อนให้ AI เป็นข้อมูลตั้งต้น ไม่ได้หมายความว่าระบบได้ตรวจสอบแหล่งข้อมูลเหล่านั้นแล้ว ควรตรวจสอบข้อเท็จจริงก่อนเผยแพร่

## ข้อมูลและไฟล์

- `projects/episodes.json` — ข้อมูล Episode และ Scene
- `projects/video_jobs.json` — ประวัติงานสร้างวิดีโอและต้นทุนที่ได้รับ
- `projects/sounds.json` — ประวัติเสียงที่สร้าง
- `outputs/` — คลิปวิดีโอและไฟล์เสียง
- `instance/api_keys.json` — API key ที่เพิ่มผ่านหน้าเว็บและค่าตั้งค่าโมเดล
- `.env` — API key ที่กำหนดผ่านตัวแปรแวดล้อม

สำรองโฟลเดอร์ `projects/`, `outputs/`, `instance/` และไฟล์ `.env` อย่างปลอดภัยหากต้องการย้ายเครื่อง ข้อมูล API key และไฟล์สื่ออาจมีความอ่อนไหว

## API ที่ใช้ในแอป

เว็บเซิร์ฟเวอร์ Flask มี API ภายในสำหรับจัดการ Episode, Scene, งานสร้างวิดีโอ, Sound, API key, การตั้งค่าโมเดล, คลังคลิป และต้นทุน ตัวอย่างเส้นทางที่หน้าเว็บใช้:

- `GET /api/episodes` — อ่านรายการ Episode
- `POST /api/episode-plan` — สร้างแผน Episode ด้วย AI
- `POST /api/episodes/create` — บันทึก Episode ใหม่
- `POST /api/generate` และ `POST /api/check` — ส่งและตรวจงานสร้างวิดีโอ
- `POST /api/sound/generate` และ `GET /api/sounds` — สร้างและอ่านประวัติ Sound
- `POST /api/assemble` และ `POST /api/assemble-with-sound` — รวมคลิปแบบไม่มีเสียงหรือพร้อมเสียงพื้นหลัง
- `GET /api/costs` และ `POST /api/costs/sync` — อ่านและซิงก์ข้อมูลต้นทุน

## ความปลอดภัย

แอปนี้ออกแบบมาสำหรับใช้งานบนเครื่องส่วนตัวและเปิดรับการเชื่อมต่อเฉพาะ `127.0.0.1` โดยไม่มีระบบล็อกอิน อย่าเปิดพอร์ตให้เข้าถึงจากอินเทอร์เน็ต และอย่าเผยแพร่ไฟล์ `.env`, `instance/api_keys.json` หรือ API key ใน Git

## เรียนต่อ: AI Video Production Pipeline 101

เรียนรู้การผลิตวิดีโอด้วย AI ตั้งแต่ค้นหาไอเดีย เขียนบท สร้างภาพ คลิป และเสียง ไปจนถึงตัดต่อและเผยแพร่บน YouTube โดยใช้ GPT-6 Astra เป็นผู้ช่วยวางแผนและลงมือทำงานตลอดกระบวนการ

ฝึกให้ AI ช่วยตัดต่อทั้งผ่าน **Computer Use เพื่อควบคุม CapCut** และผ่าน **Backend ด้วย Python + FFmpeg** สำหรับประกอบคลิป ใส่เสียงและซับไตเติล รวมถึงผลิตวิดีโอหลายเวอร์ชัน พร้อมเรียนรู้การเชื่อม YouTube API เพื่ออัปโหลด ตั้งเวลาเผยแพร่ และนำข้อมูลการรับชมมาพัฒนาคอนเทนต์ถัดไป

เรียนผ่านโปรเจกต์จริง ตั้งแต่การบรีฟและมอบหมายงานให้ AI ไปจนถึงตรวจคุณภาพและสั่งแก้ พร้อมเทมเพลตและ Workflow ที่นำกลับไปใช้ซ้ำได้ เหมาะสำหรับครีเอเตอร์ นักการตลาด เจ้าของธุรกิจ และผู้เริ่มต้น โดยไม่จำเป็นต้องมีพื้นฐานเขียนโค้ด

**เส้นทางการเรียนรู้**

ค้นหาไอเดีย → วางบรีฟและเขียนบท → ทำ Storyboard → สร้างภาพ คลิป และเสียง → ให้ AI ตัดต่อผ่าน CapCut หรือ Python + FFmpeg → ตรวจและปรับแก้ → อัปโหลดและตั้งเวลาเผยแพร่ผ่าน YouTube API → วิเคราะห์ผล → พัฒนาคลิปถัดไป

**รูปแบบหลักสูตร:** 10 EP · สัปดาห์ละ 2 ชั่วโมง · รวม 20 ชั่วโมง

[ดูรายละเอียดหลักสูตร AI Video Production Pipeline 101](https://uncle-engineer.com/course/ai-video-pipeline-101/)

## Colab — Z-Image Turbo, then MiniMax H3

Video Generate queues a Colab job. `POST /api/generate` does not call OpenRouter `/videos`. The box cannot run cloudflared (outbound port 7844 returns 530), so the public relay is `https://uncle-colab-relay.facebook-chatbot.workers.dev`. `COLAB_RELAY_URL` is preferred over `UNCLE_URL`. Source: `colab_relay/worker.js`. Runbook: [DISPATCH_COLAB_PRO.md](DISPATCH_COLAB_PRO.md).

Real clips use one path:

1. Z-Image Turbo still from the scene prompt, same frame size as the video.
2. MiniMax H3 `fl2va` image-to-video with that still as `first_frame`.

Each scene is **768×768**, **1344×768** when the job aspect is 16:9, or **768×1344** when it is 9:16. The Z-Image still uses that same size. Length follows the job duration. Episodes are 8 seconds, and `h3_length(8)` is **192 frames** (exactly 8.0s on the `17k+5` grid). A 5 second job is 124 frames, because 120 is not a legal H3 length. Durations above 8 seconds are capped at 192 frames, so a 15 second request does not become 362 frames.

`damo-vilab` is not a backend. `INSTALL_MODEL` does not select a model. Wan Animate is not this queue. Smoke mp4s for bridge tests only: `BACKEND=ffmpeg`.

ComfyUI must already be running on the Colab A100 at `http://127.0.0.1:8188` with the weights below. The worker does not download checkpoints and does not open a second Colab session. A flock on `COLAB_WORKER_LOCK` (default `/tmp/uncle_colab_worker.lock`) makes a second process on the same machine exit.

Verified H3 weights (`Comfy-Org/MiniMax-H3`, revision `e5eb578a89295337b8ff433a035929ce0279e0b6`):

- `models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors`
- `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `models/vae/minimax_h3_video_vae_fp16.safetensors`
- `models/vae/minimax_h3_audio_vae_fp32.safetensors`

Z-Image Turbo keyframes (`Comfy-Org/z_image_turbo`):

- `models/diffusion_models/z_image_turbo_bf16.safetensors`
- `models/text_encoders/qwen_3_4b.safetensors` (CLIP type `lumina2`)
- `models/vae/ae.safetensors`

H3 sampling matches the verified graph: 20 steps, `res_multistep`, `simple`, denoise 1, BasicGuider, no turbo LoRA, joint video and audio. Z-Image Turbo: 8 steps, cfg 1, AuraFlow shift 3, `res_multistep`, `simple`.

The Drive folder `1qwFEx1OTbP3GyiqEvjaFp2Ym-3j4jjKY` was the intended cache and was empty on that run. Mount it only when the files are actually there. The worker never downloads weights.

```python
import os
os.environ['COLAB_RELAY_URL'] = 'https://uncle-colab-relay.facebook-chatbot.workers.dev'
os.environ['COLAB_WORKER_TOKEN'] = 'PASTE_FROM_ENV'
os.environ['BACKEND'] = 'comfy'
os.environ['COMFY_URL'] = 'http://127.0.0.1:8188'
!python /content/Uncle_AIVideoGenerator/uncle_colab_worker.py
```

`KEYFRAME_IMAGE=/path/to/still.png` is an optional hook when the still was already rendered with Z-Image. H3 still runs as image-to-video and will not fall back to text-to-video.

Checkpoint names can be overridden with `ZIMAGE_UNET`, `ZIMAGE_CLIP`, `ZIMAGE_VAE`, `H3_UNET`, `H3_CLIP`, `H3_VIDEO_VAE`, and `H3_AUDIO_VAE` only when that runtime already has a different file from the same Comfy-Org repos (for example the newer int8 H3 names). Do not point them at Wan or damo.

Local check without a GPU:

```sh
python -m unittest tests.test_colab_worker -v
python uncle_colab_worker.py --dry-run
```

`--dry-run` talks to an in-process Comfy HTTP mock, writes `dry_run_out.mp4`, and does not claim the relay.

Worker API (header `X-Worker-Token`):

- `POST /api/colab/next`
- `POST /api/colab/result/<id>` multipart field `video`, form `status=done`, or `status=failed` with `error`
- `GET /api/colab/jobs`
- `POST /api/colab/requeue/<id>`
