"""Relay claim/upload glue with a mocked ComfyUI. No GPU."""
import os
import tempfile
import unittest
from pathlib import Path

import colab_comfy
import uncle_colab_worker as worker

ENV_KEYS = (
    'KEYFRAME_IMAGE', 'H3_SECONDS', 'INSTALL_MODEL', 'BACKEND',
    'COMFY_POLL', 'COMFY_TIMEOUT', 'H3_SEED', 'POLL_IDLE', 'DRY_RUN_OUT',
)


class EnvGuard(unittest.TestCase):
    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in ENV_KEYS}
        os.environ.pop('KEYFRAME_IMAGE', None)
        os.environ.pop('H3_SECONDS', None)
        os.environ.pop('INSTALL_MODEL', None)
        os.environ.pop('BACKEND', None)
        os.environ['COMFY_POLL'] = '0.01'
        os.environ['COMFY_TIMEOUT'] = '5'
        os.environ['POLL_IDLE'] = '0.01'

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class GridTests(EnvGuard):
    def test_five_seconds_snaps_to_verified_h3_length(self):
        self.assertEqual(colab_comfy.h3_length(5), 124)
        self.assertNotEqual(colab_comfy.h3_length(5), 120)
        self.assertEqual(colab_comfy.scene_length(), 124)

    def test_longer_than_five_seconds_is_rejected(self):
        os.environ['H3_SECONDS'] = '15'
        with self.assertRaises(SystemExit):
            colab_comfy.scene_length()

    def test_frame_size(self):
        self.assertEqual(colab_comfy.frame_size(''), (768, 768))
        self.assertEqual(colab_comfy.frame_size('1:1'), (768, 768))
        self.assertEqual(colab_comfy.frame_size('9:16'), (768, 768))
        self.assertEqual(colab_comfy.frame_size('16:9'), (1344, 768))
        self.assertEqual(colab_comfy.frame_size('16 : 9'), (1344, 768))

    def test_prompts_keep_the_scene_text(self):
        scene = 'Cute cartoon dog waving hello, talking to a cat'
        self.assertIn(scene, colab_comfy.keyframe_prompt(scene))
        self.assertIn(scene, colab_comfy.motion_prompt(scene))
        self.assertIn('five-second', colab_comfy.motion_prompt(scene))

    def test_job_identity_accepts_relay_and_local_shapes(self):
        self.assertEqual(colab_comfy.job_identity({'job_id': 'colab-a', 'id': 'other'}), 'colab-a')
        self.assertEqual(colab_comfy.job_identity({'id': 'colab-b'}), 'colab-b')

    def test_seed_is_stable_per_job(self):
        self.assertEqual(colab_comfy.seed_for('colab-abc'), colab_comfy.seed_for('colab-abc'))
        self.assertNotEqual(colab_comfy.seed_for('colab-abc'), colab_comfy.seed_for('colab-def'))

    def test_h3_graph_is_i2v_without_lora(self):
        graph = colab_comfy.h3_workflow('move', 768, 768, 124, 1, 'key.png', 'video/uncle')
        node = colab_comfy.assert_i2v(graph)
        self.assertEqual(node['inputs']['first_frame'], ['114', 0])
        self.assertEqual(node['inputs']['length'], 124)
        self.assertEqual(graph['6']['inputs']['unet_name'], colab_comfy.H3_UNET_DEFAULT)
        self.assertEqual(graph['9']['inputs']['steps'], 20)
        self.assertEqual(graph['9']['inputs']['scheduler'], 'simple')
        self.assertEqual(graph['17']['inputs']['sampler_name'], 'res_multistep')
        types = {n['class_type'] for n in graph.values()}
        self.assertIn('VAEDecodeAudio', types)
        self.assertNotIn('LoraLoaderModelOnly', types)

    def test_zimage_graph_uses_turbo_files(self):
        graph = colab_comfy.zimage_workflow('still', 768, 768, 3, 'uncle_zimage/job')
        self.assertEqual(graph['28']['inputs']['unet_name'], colab_comfy.ZIMAGE_UNET_DEFAULT)
        self.assertEqual(graph['30']['inputs']['type'], 'lumina2')
        self.assertEqual(graph['3']['inputs']['steps'], 8)
        self.assertEqual(graph['3']['inputs']['cfg'], 1)
        self.assertEqual(graph['11']['inputs']['shift'], 3)
        self.assertNotIn('MiniMaxH3ImageToVideo', {n['class_type'] for n in graph.values()})


class RelayGlueTests(EnvGuard):
    def _run(self, job, backend='comfy', fail_next=False, keyframe=None):
        if keyframe:
            os.environ['KEYFRAME_IMAGE'] = keyframe
        with colab_comfy.serve_mock_comfy() as comfy, colab_comfy.serve_mock_relay([job]) as relay:
            comfy.state.fail_next = fail_next
            done, failed = worker.process_available_jobs(
                relay.url, 'secret-token', backend=backend, comfy_url=comfy.url,
                max_jobs=1, max_empty=1,
            )
            return done, failed, comfy.state, relay.state

    def test_claim_renders_zimage_then_h3_and_uploads(self):
        job = {
            'job_id': 'colab-dog',
            'prompt': 'Cute cartoon dog waving hello, talking to a cat',
            'aspect_ratio': '9:16',
            'duration': 15,
        }
        done, failed, comfy, relay = self._run(job)
        self.assertEqual((done, failed), (1, 0))
        self.assertEqual(relay.claims, ['secret-token'])
        self.assertEqual(len(comfy.prompts), 2)
        still, video = comfy.prompts
        self.assertEqual(still['28']['inputs']['unet_name'], colab_comfy.ZIMAGE_UNET_DEFAULT)
        self.assertIn(job['prompt'], still['27']['inputs']['text'])
        node = colab_comfy.assert_i2v(video)
        self.assertEqual(node['inputs']['width'], 768)
        self.assertEqual(node['inputs']['height'], 768)
        self.assertEqual(node['inputs']['length'], 124)
        self.assertIn(job['prompt'], node['inputs']['prompt'])
        self.assertEqual(video['114']['inputs']['image'], 'key.png')
        self.assertEqual(len(comfy.uploads), 1)
        self.assertEqual(comfy.frees, 1)
        self.assertEqual(len(relay.results), 1)
        result = relay.results[0]
        self.assertEqual(result['job_id'], 'colab-dog')
        self.assertEqual(result['token'], 'secret-token')
        self.assertIn(b'name="status"', result['body'])
        self.assertIn(b'done', result['body'])
        self.assertIn(b'FAKEMP4', result['body'])
        self.assertIn(b'out.mp4', result['body'])

    def test_sixteen_by_nine_uses_native_h3_canvas(self):
        job = {'job_id': 'colab-wide', 'prompt': 'wide street', 'aspect_ratio': '16:9', 'duration': 8}
        _, _, comfy, _relay = self._run(job)
        video = comfy.prompts[1]
        self.assertEqual(video['104']['inputs']['width'], 1344)
        self.assertEqual(video['104']['inputs']['height'], 768)
        self.assertEqual(comfy.prompts[0]['13']['inputs']['width'], 1344)
        self.assertEqual(comfy.prompts[0]['13']['inputs']['height'], 768)

    def test_local_job_id_field_is_accepted(self):
        job = {'id': 'colab-local', 'prompt': 'cat answers', 'duration': 8}
        done, failed, _comfy, relay = self._run(job)
        self.assertEqual((done, failed), (1, 0))
        self.assertEqual(relay.results[0]['job_id'], 'colab-local')

    def test_comfy_error_fails_the_job(self):
        job = {'job_id': 'colab-bad', 'prompt': 'broken', 'duration': 8}
        done, failed, comfy, relay = self._run(job, fail_next=True)
        self.assertEqual((done, failed), (0, 1))
        self.assertEqual(len(comfy.prompts), 1)
        self.assertEqual(len(relay.results), 1)
        body = relay.results[0]['body']
        self.assertIn(b'failed', body)
        self.assertIn(b'boom', body)
        self.assertNotIn(b'FAKEMP4', body)

    def test_external_keyframe_skips_zimage_and_still_runs_i2v(self):
        fd, name = tempfile.mkstemp(suffix='.png')
        os.write(fd, b'\x89PNG' + b'0' * 64)
        os.close(fd)
        try:
            job = {'job_id': 'colab-hook', 'prompt': 'same dog and cat', 'duration': 8}
            done, failed, comfy, _relay = self._run(job, keyframe=name)
        finally:
            os.environ.pop('KEYFRAME_IMAGE', None)
            os.remove(name)
        self.assertEqual((done, failed), (1, 0))
        self.assertEqual(len(comfy.prompts), 1)
        self.assertEqual(comfy.prompts[0]['104']['class_type'], 'MiniMaxH3ImageToVideo')
        self.assertTrue(comfy.prompts[0]['104']['inputs']['first_frame'])
        self.assertEqual(comfy.frees, 0)
        self.assertEqual(len(comfy.uploads), 1)

    def test_ffmpeg_smoke_uploads_without_comfy(self):
        job = {'job_id': 'colab-smoke', 'prompt': 'ignored', 'duration': 8}
        done, failed, comfy, relay = self._run(job, backend='ffmpeg')
        self.assertEqual((done, failed), (1, 0))
        self.assertEqual(comfy.prompts, [])
        body = relay.results[0]['body']
        self.assertIn(b'ftyp', body)
        self.assertIn(b'done', body)

    def test_empty_queue_does_not_upload(self):
        with colab_comfy.serve_mock_relay([]) as relay:
            done, failed = worker.process_available_jobs(
                relay.url, 'secret-token', max_jobs=1, max_empty=1,
            )
        self.assertEqual((done, failed), (0, 0))
        self.assertEqual(relay.state.results, [])


class ProcessTests(EnvGuard):
    def test_damo_flag_is_rejected(self):
        os.environ['INSTALL_MODEL'] = '1'
        with self.assertRaises(SystemExit) as caught:
            worker.guard_env()
        self.assertIn('damo-vilab', str(caught.exception))

    def test_default_backend_is_comfy(self):
        self.assertEqual(worker.guard_env(), 'comfy')

    def test_second_worker_does_not_take_the_lock(self):
        path = Path(tempfile.mkdtemp()) / 'worker.lock'
        first = worker.acquire_lock(path)
        try:
            with self.assertRaises(SystemExit):
                worker.acquire_lock(path)
        finally:
            first.close()

    def test_dry_run_uses_mock_comfy_http(self):
        out = Path(tempfile.mkdtemp()) / 'dry.mp4'
        os.environ['DRY_RUN_OUT'] = str(out)
        path = colab_comfy.dry_run('a dog and a cat talking')
        self.assertTrue(Path(path).is_file())
        self.assertIn(b'FAKEMP4', Path(path).read_bytes())

    def test_check_talks_to_comfy_http(self):
        with colab_comfy.serve_mock_comfy() as comfy:
            colab_comfy.check_comfy(comfy.url)

    def test_worker_source_has_no_damo_pipeline(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('uncle_colab_worker.py', 'colab_comfy.py'):
            text = (root / name).read_text(encoding='utf-8')
            self.assertNotIn('text-to-video-ms', text)
            self.assertNotIn('DiffusionPipeline', text)
            self.assertNotIn('openrouter.ai', text.lower())
            self.assertNotIn('WanAnimate', text)
        worker_text = (root / 'uncle_colab_worker.py').read_text(encoding='utf-8')
        self.assertIn("os.environ.get('COLAB_RELAY_URL')", worker_text)
        self.assertIn('/api/colab/next', worker_text)
        self.assertIn('/api/colab/result/', worker_text)
        self.assertIn("data={'status': 'failed'", worker_text)
        self.assertIn("data={'status': 'done'}", worker_text)
