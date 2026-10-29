import assert from 'node:assert/strict';
import test from 'node:test';
import worker from '../colab_relay/worker.js';

const TOKEN = 'secret-token';
const JOB_ID = 'colab-af0f967d96ba4ac4b5288dc05e255835';

function memoryEnv(jobs = []) {
  const kv = new Map();
  const clips = new Map();
  if (jobs.length) kv.set('jobs', JSON.stringify(jobs));
  return {
    COLAB_WORKER_TOKEN: TOKEN,
    QUEUE: {
      async get(key, type) {
        const raw = kv.get(key);
        if (raw == null) return null;
        return type === 'json' ? JSON.parse(raw) : raw;
      },
      async put(key, value) {
        kv.set(key, value);
      },
    },
    CLIPS: {
      async put(key, buf) {
        clips.set(key, buf);
      },
      async get(key) {
        const buf = clips.get(key);
        if (!buf) return null;
        return { body: buf };
      },
    },
    clips,
  };
}

function runningJob() {
  return { job_id: JOB_ID, status: 'running', file: '', error: '', prompt: 'dog and cat' };
}

async function jobsOf(env) {
  return env.QUEUE.get('jobs', 'json');
}

function resultRequest(body, headers = {}) {
  return new Request(`https://relay.example/api/colab/result/${JOB_ID}`, {
    method: 'POST',
    headers: { 'X-Worker-Token': TOKEN, ...headers },
    body,
  });
}

test('urlencoded failure is failed with the error, not done without a file', async () => {
  const env = memoryEnv([runningJob()]);
  const response = await worker.fetch(resultRequest(
    new URLSearchParams({ status: 'failed', error: 'Keyframe still looks tiled after 3 seeds' }),
  ), env);
  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.status, 'failed');
  assert.equal(payload.file, '');
  const job = (await jobsOf(env))[0];
  assert.equal(job.status, 'failed');
  assert.match(job.error, /tiled after 3 seeds/);
  assert.equal(job.file, '');
  assert.equal(env.clips.size, 0);
});

test('json failure keeps the error and does not mark done', async () => {
  const env = memoryEnv([runningJob()]);
  const response = await worker.fetch(resultRequest(
    JSON.stringify({ status: 'failed', error: 'render exploded' }),
    { 'content-type': 'application/json' },
  ), env);
  const payload = await response.json();
  assert.equal(payload.status, 'failed');
  assert.equal(payload.file, '');
  const job = (await jobsOf(env))[0];
  assert.equal(job.status, 'failed');
  assert.equal(job.error, 'render exploded');
  assert.equal(env.clips.size, 0);
});

test('empty json is not a silent done', async () => {
  const env = memoryEnv([runningJob()]);
  const response = await worker.fetch(resultRequest('{}', { 'content-type': 'application/json' }), env);
  const payload = await response.json();
  assert.equal(payload.status, 'failed');
  assert.equal(payload.file, '');
  const job = (await jobsOf(env))[0];
  assert.equal(job.status, 'failed');
  assert.equal(job.error, 'missing video');
  assert.notEqual(job.status, 'done');
  assert.equal(env.clips.size, 0);
});

test('json done with a file key and no bytes is not done', async () => {
  const env = memoryEnv([runningJob()]);
  const response = await worker.fetch(resultRequest(
    JSON.stringify({ status: 'done', file: 'videos/not-uploaded.mp4' }),
    { 'content-type': 'application/json' },
  ), env);
  const payload = await response.json();
  assert.equal(payload.status, 'failed');
  assert.equal(payload.file, '');
  assert.equal(env.clips.size, 0);
  const download = await worker.fetch(new Request(`https://relay.example/api/colab/download/${JOB_ID}`, {
    headers: { 'X-Worker-Token': TOKEN },
  }), env);
  assert.equal(download.status, 404);
});

test('multipart video upload is done only after a non-empty object is stored', async () => {
  const env = memoryEnv([runningJob()]);
  const bytes = new Uint8Array([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70]);
  const form = new FormData();
  form.set('status', 'done');
  form.set('video', new Blob([bytes], { type: 'video/mp4' }), 'out.mp4');
  const response = await worker.fetch(resultRequest(form), env);
  const payload = await response.json();
  assert.equal(payload.status, 'done');
  assert.equal(payload.file, `videos/${JOB_ID}.mp4`);
  assert.equal(env.clips.size, 1);
  const stored = env.clips.get(payload.file);
  assert.equal(stored.byteLength, bytes.byteLength);
  const download = await worker.fetch(new Request(`https://relay.example/api/colab/download/${JOB_ID}`, {
    headers: { 'X-Worker-Token': TOKEN },
  }), env);
  assert.equal(download.status, 200);
  const body = new Uint8Array(await download.arrayBuffer());
  assert.deepEqual(body, bytes);
});

test('multipart without a video is failed, not done', async () => {
  const env = memoryEnv([runningJob()]);
  const form = new FormData();
  form.set('status', 'done');
  const response = await worker.fetch(resultRequest(form), env);
  const payload = await response.json();
  assert.equal(payload.status, 'failed');
  assert.equal(payload.file, '');
  assert.equal((await jobsOf(env))[0].error, 'missing video');
  assert.equal(env.clips.size, 0);
});

test('empty multipart video is not stored as done', async () => {
  const env = memoryEnv([runningJob()]);
  const form = new FormData();
  form.set('status', 'done');
  form.set('video', new Blob([], { type: 'video/mp4' }), 'out.mp4');
  const response = await worker.fetch(resultRequest(form), env);
  const payload = await response.json();
  assert.equal(payload.status, 'failed');
  assert.equal(payload.file, '');
  assert.equal(env.clips.size, 0);
});

test('multipart failure keeps the error and stores no clip', async () => {
  const env = memoryEnv([runningJob()]);
  const form = new FormData();
  form.set('status', 'failed');
  form.set('error', 'Keyframe still looks tiled after 3 seeds');
  const response = await worker.fetch(resultRequest(form), env);
  const payload = await response.json();
  assert.equal(payload.status, 'failed');
  assert.equal(payload.file, '');
  const job = (await jobsOf(env))[0];
  assert.match(job.error, /tiled after 3 seeds/);
  assert.equal(env.clips.size, 0);
});
