"""Relay claim/upload glue with a mocked ComfyUI. No GPU."""
import io
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


def guttered_contact_sheet(cols, rows):
    from PIL import Image, ImageEnhance
    panel_width, panel_height, gutter = 120, 144, 6
    image = Image.new('RGB', (cols * panel_width + (cols - 1) * gutter,
                              rows * panel_height + (rows - 1) * gutter), 'black')
    panel = colab_comfy.single_scene_image(panel_width, panel_height)
    for row in range(rows):
        for col in range(cols):
            cell = ImageEnhance.Brightness(panel).enhance(0.85 + (row + col) * 0.04)
            image.paste(cell, (col * (panel_width + gutter), row * (panel_height + gutter)))
    return image


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
        self.assertEqual(colab_comfy.scene_length({'duration': 5}), 124)

    def test_eight_seconds_is_192_frames(self):
        self.assertEqual(colab_comfy.h3_length(8), 192)
        self.assertEqual(192 / 24, 8)
        self.assertEqual(colab_comfy.scene_length(), 192)
        self.assertEqual(colab_comfy.scene_length({'duration': 8}), 192)
        os.environ['H3_SECONDS'] = '8'
        self.assertEqual(colab_comfy.scene_length(), 192)
        self.assertEqual(colab_comfy.scene_length({'duration': 8}), 192)

    def test_fifteen_seconds_caps_at_eight(self):
        self.assertEqual(colab_comfy.h3_length(15), 362)
        self.assertEqual(colab_comfy.scene_length({'duration': 15}), 192)
        os.environ['H3_SECONDS'] = '15'
        self.assertEqual(colab_comfy.scene_length(), 192)

    def test_frame_size(self):
        self.assertEqual(colab_comfy.frame_size(''), (768, 768))
        self.assertEqual(colab_comfy.frame_size('1:1'), (768, 768))
        self.assertEqual(colab_comfy.frame_size('9:16'), (768, 1344))
        self.assertEqual(colab_comfy.frame_size('9 : 16'), (768, 1344))
        self.assertEqual(colab_comfy.frame_size('16:9'), (1344, 768))
        self.assertEqual(colab_comfy.frame_size('16 : 9'), (1344, 768))

    def test_portrait_i2v_accepts_768_by_1344(self):
        graph = colab_comfy.h3_workflow('move', 768, 1344, 192, 1, 'key.png', 'video/uncle')
        node = colab_comfy.assert_i2v(graph)
        self.assertEqual(node['inputs']['width'], 768)
        self.assertEqual(node['inputs']['height'], 1344)
        self.assertEqual(node['inputs']['length'], 192)
        self.assertEqual(node['inputs']['last_frame'], node['inputs']['first_frame'])
        self.assertEqual(colab_comfy.zimage_latent_size(), (1024, 1024))
        still = colab_comfy.zimage_workflow('still', *colab_comfy.zimage_latent_size(), 1, 'uncle_zimage/job')
        self.assertEqual(still['13']['inputs']['width'], 1024)
        self.assertEqual(still['13']['inputs']['height'], 1024)
        self.assertNotEqual((still['13']['inputs']['width'], still['13']['inputs']['height']), (768, 1344))
        too_long = colab_comfy.h3_workflow('move', 768, 1344, 362, 1, 'key.png', 'video/uncle')
        with self.assertRaises(RuntimeError):
            colab_comfy.assert_i2v(too_long)

    def test_prompts_keep_the_scene_text(self):
        scene = 'Cute cartoon dog waving hello, talking to a cat'
        motion = colab_comfy.motion_prompt(scene, 8)
        self.assertIn(scene, colab_comfy.keyframe_prompt(scene))
        self.assertIn(scene, motion)
        self.assertIn('about 8 seconds', motion)
        self.assertNotIn('five-second', motion)
        self.assertNotIn('Ignore any 8-second', motion)
        for phrase in ('comic', 'triptych', 'stacked strips', 'storyboard', 'grid', 'repeated bands'):
            self.assertIn(phrase, colab_comfy.keyframe_prompt(scene))
            self.assertIn(phrase, motion)

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
        self.assertEqual(node['inputs']['last_frame'], ['114', 0])
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

    def test_fit_cover_maps_1024_onto_each_h3_canvas(self):
        source = colab_comfy.single_scene_image(1024, 1024)
        self.assertEqual(colab_comfy.fit_cover_image(source, 768, 768).size, (768, 768))
        self.assertEqual(colab_comfy.fit_cover_image(source, 1344, 768).size, (1344, 768))
        self.assertEqual(colab_comfy.fit_cover_image(source, 768, 1344).size, (768, 1344))

    def test_tile_gate_rejects_stacked_strips_and_keeps_a_single_scene(self):
        tiled, scores = colab_comfy.keyframe_is_tiled(colab_comfy.stacked_strip_image(96, 96))
        self.assertTrue(tiled)
        self.assertGreater(scores['h3'], 0.85)
        clean, clean_scores = colab_comfy.keyframe_is_tiled(colab_comfy.single_scene_image(96, 96))
        self.assertFalse(clean)
        self.assertLessEqual(max(clean_scores['%s%d' % (axis, n)]
                                 for axis in ('h', 'w') for n in colab_comfy.TILE_SHIFTS), 0.75)

    def test_tile_gate_rejects_mixed_portrait_grid(self):
        from PIL import Image
        fixture = Path(__file__).parent / 'fixtures' / 'dogcat_9657_early.png'
        with Image.open(fixture) as img:
            tiled, scores = colab_comfy.keyframe_is_tiled(img)
        self.assertTrue(tiled, scores)
        self.assertEqual(scores['portrait_n'], 3.0)
        self.assertGreaterEqual(scores['portrait_hits'], 5.0)
        self.assertLess(scores['portrait_adj'], 0.50)

    def test_tile_gate_rejects_f1b2_via_gutters(self):
        from PIL import Image
        fixture = Path(__file__).parent / 'fixtures' / 'dogcat_f1b2_2s.png'
        with Image.open(fixture) as image:
            gutter, gutter_scores = colab_comfy.comic_gutter_hits(image)
            tiled, scores = colab_comfy.keyframe_is_tiled(image)
            gutter_only, _ = colab_comfy.keyframe_is_tiled(image, threshold=1.0)
        self.assertTrue(gutter, gutter_scores)
        self.assertTrue(tiled, scores)
        self.assertTrue(gutter_only, scores)
        self.assertGreaterEqual(scores['gutter_h'], 2)
        self.assertGreaterEqual(scores['gutter_v'], 1)
        self.assertGreaterEqual(scores['gutter_d'], 30)

    def test_tile_gate_rejects_guttered_contact_sheets(self):
        for cols, rows in ((3, 4), (4, 3), (3, 3)):
            with self.subTest(cols=cols, rows=rows):
                image = guttered_contact_sheet(cols, rows)
                gutter, scores = colab_comfy.comic_gutter_hits(image)
                self.assertTrue(gutter, scores)
                self.assertEqual(scores['gutter_h'], rows - 1)
                self.assertEqual(scores['gutter_v'], cols - 1)
                self.assertGreaterEqual(scores['gutter_d'], 30)
                tiled, scores = colab_comfy.keyframe_is_tiled(image)
                self.assertTrue(tiled, scores)

    def test_tile_gate_keeps_single_scene_duo(self):
        from PIL import Image, ImageDraw
        image = Image.new('RGB', (384, 256), '#9ecfe0')
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 175, 383, 255), fill='#70a458')
        # Dog and cat share a ground plane, with no panels or repeated rows.
        draw.ellipse((40, 100, 160, 235), fill='#b67c48')
        draw.ellipse((30, 60, 165, 170), fill='#c99761')
        draw.ellipse((25, 70, 60, 155), fill='#805436')
        draw.ellipse((140, 70, 175, 155), fill='#805436')
        draw.ellipse((65, 98, 77, 111), fill='black')
        draw.ellipse((117, 98, 129, 111), fill='black')
        draw.ellipse((86, 123, 110, 140), fill='#322b27')
        draw.ellipse((230, 120, 335, 235), fill='#8c9299')
        draw.polygon(((225, 110), (230, 50), (265, 85), (300, 85),
                      (330, 50), (340, 110)), fill='#8c9299')
        draw.ellipse((225, 80, 340, 175), fill='#a4aab0')
        draw.ellipse((247, 115, 262, 128), fill='#475b23')
        draw.ellipse((301, 115, 316, 128), fill='#475b23')
        draw.polygon(((275, 140), (287, 140), (281, 148)), fill='#a6606b')
        tiled, scores = colab_comfy.keyframe_is_tiled(image)
        self.assertFalse(tiled, scores)

    def test_tile_gate_rejects_horizontal_comic_grid(self):
        tiled, scores = colab_comfy.keyframe_is_tiled(colab_comfy.contact_sheet_image(96, 160, 3, 5))
        self.assertTrue(tiled, scores)
        self.assertGreater(max(scores['w3'], scores['h5']), 0.85)
        regression = os.path.join('/workspace/uncle-tile-diag', 'frame_002.png')
        if os.path.exists(regression):
            from PIL import Image
            tiled, scores = colab_comfy.keyframe_is_tiled(Image.open(regression))
            self.assertTrue(tiled, scores)

    def test_salvage_of_comic_grid_yields_untiled_full_frame(self):
        grid = colab_comfy.contact_sheet_image(300, 500, 3, 5)
        _, scores = colab_comfy.keyframe_is_tiled(grid)
        out = colab_comfy.salvage_keyframe(grid, scores, 768, 1344)
        self.assertEqual(out.size, (768, 1344))
        tiled, after = colab_comfy.keyframe_is_tiled(out)
        self.assertFalse(tiled, after)

    def test_salvage_of_guttered_grid_yields_untiled_full_frame(self):
        grid = guttered_contact_sheet(4, 3)
        tiled, scores = colab_comfy.keyframe_is_tiled(grid)
        self.assertTrue(tiled, scores)
        out = colab_comfy.salvage_keyframe(grid, scores, 768, 1344)
        self.assertEqual(out.size, (768, 1344))
        tiled, after = colab_comfy.keyframe_is_tiled(out)
        self.assertFalse(tiled, after)


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
            'duration': 8,
        }
        done, failed, comfy, relay = self._run(job)
        self.assertEqual((done, failed), (1, 0))
        self.assertEqual(relay.claims, ['secret-token'])
        self.assertEqual(len(comfy.prompts), 2)
        still, video = comfy.prompts
        self.assertEqual(still['28']['inputs']['unet_name'], colab_comfy.ZIMAGE_UNET_DEFAULT)
        self.assertEqual(still['13']['inputs']['width'], 1024)
        self.assertEqual(still['13']['inputs']['height'], 1024)
        self.assertIn(job['prompt'], still['27']['inputs']['text'])
        node = colab_comfy.assert_i2v(video)
        self.assertEqual(node['inputs']['width'], 768)
        self.assertEqual(node['inputs']['height'], 1344)
        self.assertEqual(node['inputs']['length'], 192)
        self.assertIn('about 8 seconds', node['inputs']['prompt'])
        self.assertNotIn('five-second', node['inputs']['prompt'])
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

    def test_tiled_keyframes_reroll_then_salvage_before_h3(self):
        from PIL import Image
        job = {'job_id': 'colab-tile', 'prompt': 'dog and cat', 'aspect_ratio': '9:16', 'duration': 8}
        tiled = colab_comfy.png_bytes(colab_comfy.stacked_strip_image(90, 90))
        with colab_comfy.serve_mock_comfy() as comfy, colab_comfy.serve_mock_relay([job]) as relay:
            comfy.state.key_pngs = [tiled] * colab_comfy.KEYFRAME_ATTEMPTS
            done, failed = worker.process_available_jobs(
                relay.url, 'secret-token', comfy_url=comfy.url, max_jobs=1, max_empty=1,
            )
        self.assertEqual((done, failed), (1, 0))
        self.assertEqual(len(comfy.state.prompts), colab_comfy.KEYFRAME_ATTEMPTS + 1)
        self.assertTrue(all(
            node.get('class_type') != 'MiniMaxH3ImageToVideo'
            for prompt in comfy.state.prompts[:-1] for node in prompt.values()
        ))
        self.assertEqual(comfy.state.prompts[-1]['104']['class_type'], 'MiniMaxH3ImageToVideo')
        self.assertEqual(len(comfy.state.uploads), 1)
        upload = comfy.state.uploads[0]
        with Image.open(io.BytesIO(upload[upload.index(b'\x89PNG\r\n\x1a\n'):])) as image:
            self.assertEqual(image.size, (768, 1344))
            is_tiled, scores = colab_comfy.keyframe_is_tiled(image)
            self.assertFalse(is_tiled, scores)
        self.assertIn(b'done', relay.state.results[0]['body'])

    def test_tiled_keyframe_rerolls_then_accepts_a_clean_still(self):
        job = {'job_id': 'colab-reroll', 'prompt': 'dog and cat', 'aspect_ratio': '1:1', 'duration': 8}
        tiled = colab_comfy.png_bytes(colab_comfy.stacked_strip_image(90, 90))
        clean = colab_comfy.png_bytes(colab_comfy.single_scene_image(80, 80))
        with colab_comfy.serve_mock_comfy() as comfy, colab_comfy.serve_mock_relay([job]) as relay:
            comfy.state.key_pngs = [tiled, clean]
            done, failed = worker.process_available_jobs(
                relay.url, 'secret-token', comfy_url=comfy.url, max_jobs=1, max_empty=1,
            )
        self.assertEqual((done, failed), (1, 0))
        stills = [prompt for prompt in comfy.state.prompts if '13' in prompt and prompt['13']['class_type'] == 'EmptySD3LatentImage']
        self.assertEqual(len(stills), 2)
        self.assertEqual(stills[0]['13']['inputs']['width'], 1024)
        self.assertEqual(stills[1]['3']['inputs']['seed'],
                         (stills[0]['3']['inputs']['seed'] + 1_000_003) % (2**31 - 1))
        h3 = [prompt for prompt in comfy.state.prompts if any(node.get('class_type') == 'MiniMaxH3ImageToVideo' for node in prompt.values())]
        self.assertEqual(len(h3), 1)
        self.assertEqual(h3[0]['104']['inputs']['width'], 768)
        self.assertEqual(h3[0]['104']['inputs']['height'], 768)
        self.assertEqual(relay.state.results[0]['job_id'], 'colab-reroll')

    def test_duration_15_stays_at_192_frames(self):
        job = {
            'job_id': 'colab-long',
            'prompt': 'a long take',
            'aspect_ratio': '9:16',
            'duration': 15,
        }
        _, _, comfy, _relay = self._run(job)
        self.assertEqual(comfy.prompts[1]['104']['inputs']['length'], 192)
        self.assertNotEqual(comfy.prompts[1]['104']['inputs']['length'], 362)
        self.assertEqual(comfy.prompts[0]['13']['inputs']['width'], 1024)
        self.assertEqual(comfy.prompts[0]['13']['inputs']['height'], 1024)

    def test_sixteen_by_nine_uses_native_h3_canvas(self):
        job = {'job_id': 'colab-wide', 'prompt': 'wide street', 'aspect_ratio': '16:9', 'duration': 8}
        _, _, comfy, _relay = self._run(job)
        video = comfy.prompts[1]
        self.assertEqual(video['104']['inputs']['width'], 1344)
        self.assertEqual(video['104']['inputs']['height'], 768)
        self.assertEqual(comfy.prompts[0]['13']['inputs']['width'], 1024)
        self.assertEqual(comfy.prompts[0]['13']['inputs']['height'], 1024)
        self.assertEqual(video['104']['inputs']['last_frame'], video['104']['inputs']['first_frame'])

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
        os.close(fd)
        colab_comfy.single_scene_image(80, 80).save(name)
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
        self.assertIn("'status': (None, 'failed')", worker_text)
        self.assertIn("data={'status': 'done'}", worker_text)
