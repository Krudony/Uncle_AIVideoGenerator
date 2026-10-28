const enc = new TextEncoder();

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json', 'access-control-allow-origin': '*' },
  });
}

function auth(req, env) {
  const t = req.headers.get('X-Worker-Token') || '';
  return t && t === env.COLAB_WORKER_TOKEN;
}

async function listJobs(env) {
  const raw = await env.QUEUE.get('jobs', 'json');
  return Array.isArray(raw) ? raw : [];
}

async function saveJobs(env, jobs) {
  await env.QUEUE.put('jobs', JSON.stringify(jobs.slice(0, 500)));
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'access-control-allow-origin': '*',
          'access-control-allow-methods': 'GET,POST,OPTIONS',
          'access-control-allow-headers': 'content-type,x-worker-token',
        },
      });
    }

    if (url.pathname === '/health') return json({ ok: true, backend: 'colab-relay' });

    if (!auth(request, env) && url.pathname !== '/health') {
      return json({ error: 'unauthorized' }, 401);
    }

    // Enqueue from Uncle/Volt
    if (request.method === 'POST' && url.pathname === '/api/colab/enqueue') {
      const body = await request.json();
      const job_id = body.job_id || `colab-${crypto.randomUUID().replace(/-/g, '')}`;
      const job = {
        job_id,
        slug: body.slug || '',
        episode_title: body.episode_title || '',
        scene_id: body.scene_id,
        scene_title: body.scene_title || '',
        model: 'colab/free',
        duration: body.duration || 8,
        resolution: body.resolution || '720p',
        aspect_ratio: body.aspect_ratio || '9:16',
        status: 'queued',
        file: '',
        prompt: body.prompt || '',
        usage_cost: 0,
        created_at: new Date().toISOString(),
        api_key_id: 'colab-relay',
      };
      const jobs = await listJobs(env);
      jobs.unshift(job);
      await saveJobs(env, jobs);
      return json({ id: job_id, status: 'pending', usage: { cost: 0 } });
    }

    // Claim next
    if (request.method === 'POST' && url.pathname === '/api/colab/next') {
      const jobs = await listJobs(env);
      const job = jobs.find((j) => j.status === 'queued');
      if (!job) return json({ job: null });
      job.status = 'running';
      job.started_at = new Date().toISOString();
      await saveJobs(env, jobs);
      return json({ job });
    }

    // Result upload
    const m = url.pathname.match(/^\/api\/colab\/result\/([^/]+)$/);
    if (request.method === 'POST' && m) {
      const job_id = decodeURIComponent(m[1]);
      const jobs = await listJobs(env);
      const job = jobs.find((j) => j.job_id === job_id);
      if (!job) return json({ error: 'not found' }, 404);
      const ct = request.headers.get('content-type') || '';
      if (ct.includes('multipart/form-data')) {
        const form = await request.formData();
        const status = form.get('status') || 'done';
        const err = form.get('error') || '';
        const video = form.get('video');
        if (status === 'failed') {
          job.status = 'failed';
          job.error = String(err).slice(0, 500);
        } else if (video && typeof video.arrayBuffer === 'function') {
          const buf = await video.arrayBuffer();
          const key = `videos/${job_id}.mp4`;
          await env.CLIPS.put(key, buf, { httpMetadata: { contentType: 'video/mp4' } });
          job.status = 'done';
          job.file = key;
          job.error = '';
          job.completed_at = new Date().toISOString();
        } else {
          job.status = 'failed';
          job.error = 'missing video';
        }
      } else {
        const body = await request.json().catch(() => ({}));
        job.status = body.status === 'failed' ? 'failed' : 'done';
        job.error = body.error || '';
        if (body.file) job.file = body.file;
        job.completed_at = new Date().toISOString();
      }
      await saveJobs(env, jobs);
      return json({ ok: true, status: job.status, file: job.file || '' });
    }

    // List
    if (request.method === 'GET' && url.pathname === '/api/colab/jobs') {
      const jobs = await listJobs(env);
      return json({ data: jobs });
    }

    // Download clip
    const d = url.pathname.match(/^\/api\/colab\/download\/([^/]+)$/);
    if (request.method === 'GET' && d) {
      const job_id = decodeURIComponent(d[1]);
      const jobs = await listJobs(env);
      const job = jobs.find((j) => j.job_id === job_id);
      if (!job || !job.file) return json({ error: 'not found' }, 404);
      const obj = await env.CLIPS.get(job.file);
      if (!obj) return json({ error: 'missing object' }, 404);
      return new Response(obj.body, {
        headers: {
          'content-type': 'video/mp4',
          'content-disposition': `attachment; filename="${job_id}.mp4"`,
        },
      });
    }

    // Requeue
    const rq = url.pathname.match(/^\/api\/colab\/requeue\/([^/]+)$/);
    if (request.method === 'POST' && rq) {
      const job_id = decodeURIComponent(rq[1]);
      const jobs = await listJobs(env);
      const job = jobs.find((j) => j.job_id === job_id);
      if (!job) return json({ error: 'not found' }, 404);
      job.status = 'queued';
      job.error = '';
      job.file = '';
      await saveJobs(env, jobs);
      return json({ ok: true });
    }

    return json({ error: 'not found' }, 404);
  },
};
