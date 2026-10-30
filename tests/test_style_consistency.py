"""CPU regression checks for approved-keyframe style continuity."""
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import uncle_colab_worker as worker
from pathlib import Path
from PIL import Image, ImageEnhance
import colab_comfy

FIXTURES = Path(__file__).parent / 'fixtures'


def encode_clip(frames, path):
    size = frames[0].size
    subprocess.run([
        'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{size[0]}x{size[1]}', '-r', '24', '-i', 'pipe:0',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path),
    ], input=b''.join(frame.convert('RGB').tobytes() for frame in frames), check=True)


class StyleTests(unittest.TestCase):
    def setUp(self):
        self.corgi = Image.open(FIXTURES / 'style_corgi.png').convert('RGB')
        self.cats = Image.open(FIXTURES / 'style_cats.png').convert('RGB')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'clip.mp4'

    def check_clip(self, frames, keyframe):
        encode_clip(frames, self.path)
        return colab_comfy.assert_video_style(self.path, keyframe)

    def test_hard_cut_to_flat_cartoon_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'style switch'):
            self.check_clip([self.corgi] * 4 + [self.cats] * 4, self.corgi)

    def test_soft_switch_is_rejected(self):
        for length in (13, 24, 48):
            frames = [Image.blend(self.corgi, self.cats, n / (length - 1))
                      for n in range(length)]
            with self.subTest(length=length), self.assertRaisesRegex(RuntimeError, 'style switch'):
                self.check_clip(frames, self.corgi)

    def test_keyframe_mismatch_is_rejected_in_both_directions(self):
        for key, frame in ((self.corgi, self.cats), (self.cats, self.corgi)):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'style switch'):
                self.check_clip([frame] * 4, key)

    def test_consistent_medium_survives_motion_and_brightness(self):
        for image in (self.corgi, self.cats):
            moved = image.crop((8, 8, image.width - 8, image.height - 8)).resize(image.size)
            frames = [image, moved, ImageEnhance.Brightness(image).enhance(0.85), image]
            with self.subTest(image=image):
                self.check_clip(frames, image)

    def test_scene_change_with_same_medium_is_not_style_evidence(self):
        from PIL import ImageOps
        self.check_clip([self.corgi, ImageOps.mirror(self.corgi), self.corgi], self.corgi)
        self.check_clip([self.cats, ImageOps.mirror(self.cats), self.cats], self.cats)

    def test_dimmed_cartoon_switch_is_still_rejected(self):
        dimmed = ImageEnhance.Brightness(self.cats).enhance(0.5)
        with self.assertRaisesRegex(RuntimeError, 'style switch'):
            self.check_clip([self.corgi, dimmed], self.corgi)

    def test_overexposure_is_not_a_style_switch(self):
        bright = ImageEnhance.Brightness(self.corgi).enhance(2)
        self.check_clip([self.corgi, bright, self.corgi], self.corgi)

    def test_temporary_switch_returning_to_keyframe_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'style switch'):
            self.check_clip([self.corgi] * 6 + [self.cats] + [self.corgi] * 6, self.corgi)

    def test_worker_rejects_style_mismatch_before_upload(self):
        encode_clip([self.corgi] * 4, self.path)
        video_bytes = self.path.read_bytes()
        original = colab_comfy.ComfyClient.download

        def download(client, ref, dest):
            if ref['filename'].endswith('.mp4'):
                Path(dest).write_bytes(video_bytes)
                return Path(dest)
            return original(client, ref, dest)

        job = {'job_id': 'style-regression', 'prompt': 'dog and cat', 'duration': 5,
               'aspect_ratio': '9:16'}
        with colab_comfy.serve_mock_comfy() as comfy, colab_comfy.serve_mock_relay([job]) as relay:
            comfy.state.key_pngs = [colab_comfy.png_bytes(self.cats)]
            with patch.object(colab_comfy.ComfyClient, 'download', download):
                result = worker.process_available_jobs(
                    relay.url, 'test-token', comfy_url=comfy.url, max_jobs=1, max_empty=1)
        self.assertEqual(result, (0, 1))
        self.assertEqual(len(relay.state.results), 1)
        body = relay.state.results[0]['body']
        self.assertIn(b'failed', body)
        self.assertIn(b'style switch', body)
        self.assertNotIn(b'ftyp', body)

    def test_undecodable_video_is_rejected(self):
        self.path.write_bytes(b'not a video')
        with self.assertRaisesRegex(RuntimeError, 'decode'):
            colab_comfy.assert_video_style(self.path, self.corgi)


if __name__ == '__main__':
    unittest.main()
