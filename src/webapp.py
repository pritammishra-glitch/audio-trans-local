"""Upload an audio file, get the transcript back as a .txt download.

A small local web front end over run.sh. Everything stays on this machine -
the audio is never sent anywhere.

    ./serve.sh          then open http://127.0.0.1:5001
"""
import os, re, shutil, subprocess, threading, time, uuid, pathlib
from flask import (Flask, request, send_file, redirect, url_for,
                   render_template_string, abort)

import pipeline
import asr_backend

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pipeline.data_root()
UPLOADS = DATA / "audio"
OUT = DATA / "out"
ALLOWED = {".wav", ".mp3", ".m4a", ".aif", ".aiff", ".caf", ".flac", ".mp4"}
MAX_BYTES = 500 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

JOBS = {}          # id -> {name, state, log, error}
JOBS_LOCK = threading.Lock()


def ollama_up():
    """Is the annotation model reachable?

    Checked per page load rather than once at startup, so starting Ollama
    while the app is open is noticed without a restart. Fast mode needs
    nothing, so the app stays useful either way.
    """
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


def safe_stem(filename):
    stem = pathlib.Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return stem or "call"


def run_job(job_id, audio_path, name, language, second, fast):
    def set_state(**kw):
        with JOBS_LOCK:
            JOBS[job_id].update(kw)

    log = []

    def emit(line):
        line = str(line).rstrip()
        if not line or "NotOpenSSL" in line:
            return
        log.append(line)
        set_state(log=log[-40:])

    try:
        pipeline.run(audio_path, OUT / name, language=language,
                     second=second, fast=fast, log=emit)
        if not (OUT / name / "transcript.txt").exists():
            set_state(state="error", error="the pipeline produced no transcript")
            return
        set_state(state="done")
    except Exception as e:                       # noqa: BLE001
        set_state(state="error", error=str(e))


PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transcribe</title>
<style>
 :root{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e2dc;--acc:#1f5f4b}
 @media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#ececea;--mut:#9a9a94;--line:#2e2e34;--acc:#5fcfa8}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,-apple-system,system-ui,sans-serif}
 .wrap{max-width:720px;margin:0 auto;padding:48px 16px 80px}
 h1{font-size:1.5rem;margin:0 0 .25rem} p.sub{color:var(--mut);margin:0 0 2rem}
 .card{border:1px solid var(--line);border-radius:12px;padding:20px}
 label{display:block;font-weight:600;margin:0 0 .4rem}
 input[type=file],select{width:100%;padding:10px;border:1px solid var(--line);
   border-radius:8px;background:var(--bg);color:var(--fg)}
 .row{margin-bottom:1rem}
 button,a.btn{display:inline-block;background:var(--acc);color:#fff;border:0;
   border-radius:8px;padding:11px 18px;font-size:1rem;font-weight:600;
   cursor:pointer;text-decoration:none}
 a.btn.alt{background:transparent;color:var(--acc);border:1px solid var(--acc)}
 pre{background:rgba(127,127,127,.09);border-radius:8px;padding:14px;
   overflow:auto;font-size:13px;max-height:420px;white-space:pre-wrap;
   word-break:break-word}
 .err{color:#c0392b} .mut{color:var(--mut);font-size:.9rem}
 .note{border:1px solid var(--line);border-left:3px solid var(--acc);
   border-radius:8px;padding:12px 14px;margin:0 0 1.25rem;font-size:.93rem}
 code{background:rgba(127,127,127,.14);padding:1px 5px;border-radius:4px}
 .status{border:1px solid var(--line);border-radius:10px;padding:12px 14px;
   margin:0 0 1.25rem;font-size:.92rem;display:grid;gap:.45rem}
 .dot{display:inline-block;width:8px;height:8px;border-radius:50%;
   margin-right:8px;vertical-align:middle}
 .dot.ok{background:#2e9e6b} .dot.wait{background:#d79a21}
 .dot.off{background:#9a9a94}
 ul.files{list-style:none;padding:0;margin:0} ul.files li{margin:.3rem 0}
</style>
<div class="wrap">
<h1>Audio transcription &amp; annotation</h1>
<p class="sub">Runs entirely on this machine. Nothing is uploaded anywhere.</p>

{% if not job %}
<div class="status">
  <div><span class="dot ok"></span><strong>App</strong> &mdash; installed and running</div>
  <div>
    {% if model_ready %}
      <span class="dot ok"></span><strong>Speech model</strong> &mdash; downloaded, works offline
    {% else %}
      <span class="dot wait"></span><strong>Speech model</strong> &mdash; not downloaded yet.
      About 1.5 GB, fetched automatically the first time you transcribe
      something. Needs internet once.
    {% endif %}
  </div>
  <div>
    {% if ollama %}
      <span class="dot ok"></span><strong>Annotation model</strong> &mdash; Ollama running, full mode available
    {% else %}
      <span class="dot off"></span><strong>Annotation model</strong> &mdash; not installed, Fast mode only
    {% endif %}
  </div>
</div>
{% endif %}

{% if not job and not ollama %}
<div class="note">
  <strong>Fast mode is on.</strong> The full annotation pass needs Ollama,
  which is not running, so the Clean line, &lt;fs&gt;/&lt;sc&gt; and
  &lt;LOCATION&gt; are unavailable. Everything else works.
  <br><span class="mut">To enable it: install Ollama from ollama.com, run
  <code>ollama pull qwen2.5:7b</code>, and reload this page.</span>
</div>
{% endif %}

{% if not job %}
<form class="card" method="post" action="{{ url_for('upload') }}"
      enctype="multipart/form-data">
  <div class="row">
    <label for="f">Audio file</label>
    <input id="f" type="file" name="audio" required
           accept=".wav,.mp3,.m4a,.aif,.aiff,.caf,.flac,.mp4">
    <p class="mut">Stereo with one speaker per channel gives the best results.</p>
  </div>
  <div class="row">
    <label for="l">Main language</label>
    <select id="l" name="language">
      <option value="auto">Detect automatically</option>
      <option value="en" selected>English</option>
      <option value="hi">Hindi</option>
      <option value="kn">Kannada</option>
      <option value="ta">Tamil</option>
      <option value="te">Telugu</option>
      <option value="ml">Malayalam</option>
      <option value="mr">Marathi</option>
      <option value="bn">Bengali</option>
      <option value="gu">Gujarati</option>
      <option value="pa">Punjabi</option>
      <option value="ur">Urdu</option>
    </select>
  </div>
  <div class="row">
    <label for="s">Other language on the call</label>
    <select id="s" name="second">
      <option value="" selected>None / not sure</option>
      <option value="kn">Kannada</option>
      <option value="hi">Hindi</option>
      <option value="ta">Tamil</option>
      <option value="te">Telugu</option>
      <option value="ml">Malayalam</option>
      <option value="mr">Marathi</option>
      <option value="bn">Bengali</option>
      <option value="gu">Gujarati</option>
      <option value="pa">Punjabi</option>
      <option value="ur">Urdu</option>
      <option value="en">English</option>
    </select>
    <p class="mut">Turns not in the main language get wrapped, e.g.
       &lt;Kannada&gt;…&lt;/Kannada&gt;. Leave this as "not sure" and they are
       flagged for review instead of labelled - the detector can tell a turn
       is not English but cannot reliably say which language it is.</p>
  </div>
  <div class="row">
    <label><input type="checkbox" name="fast" value="1" style="width:auto"
      {% if not ollama %}checked disabled{% endif %}>
      Fast mode</label>
    {% if not ollama %}<input type="hidden" name="fast" value="1">{% endif %}
    <p class="mut">Skips the language-model pass, which is most of the
       runtime - minutes become seconds. You still get timestamps, speakers,
       pauses, silences and the PII, DATE and TIME tags. You lose the Clean
       line, &lt;fs&gt;/&lt;sc&gt; and &lt;LOCATION&gt;.</p>
  </div>
  <button type="submit">Transcribe</button>
</form>
{% else %}
<div class="card">
  {% if job.state == 'running' %}
    <p><strong>Working on {{ job.name }}…</strong> this takes roughly as long as
       the recording.</p>
    <pre>{{ "\n".join(job.log) or "starting…" }}</pre>
    <p class="mut">This page refreshes itself.</p>
  {% elif job.state == 'error' %}
    <p class="err"><strong>It failed.</strong> {{ job.error }}</p>
    <pre>{{ "\n".join(job.log) }}</pre>
    <p><a class="btn alt" href="{{ url_for('index') }}">Try another file</a></p>
  {% else %}
    <p><strong>Done.</strong> {{ job.name }}</p>
    <ul class="files">
      <li><a class="btn" href="{{ url_for('download', job_id=job.id, which='transcript') }}">
          Download transcript.txt</a></li>
      <li style="margin-top:.6rem"><a class="btn alt"
          href="{{ url_for('download', job_id=job.id, which='review') }}">
          Download review version</a></li>
    </ul>
    <p class="mut" style="margin-top:1rem">The review version adds the metadata
       block, the Clean lines and the &gt;&gt; REVIEW flags.</p>
    <pre>{{ preview }}</pre>
    <p><a class="btn alt" href="{{ url_for('index') }}">Transcribe another</a></p>
  {% endif %}
</div>
{% endif %}
</div>
{% if job and job.state == 'running' %}<meta http-equiv="refresh" content="2">{% endif %}
"""


@app.get("/")
def index():
    return render_template_string(PAGE, job=None, ollama=ollama_up(),
                                  model_ready=asr_backend.model_cached())


@app.post("/upload")
def upload():
    f = request.files.get("audio")
    if not f or not f.filename:
        abort(400, "no file")
    ext = pathlib.Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        abort(400, f"unsupported file type {ext}")

    name = safe_stem(f.filename)
    if (OUT / name).exists():
        name = f"{name}_{time.strftime('%H%M%S')}"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / f"{name}{ext}"
    f.save(dest)

    CODES = {"auto", "en", "hi", "kn", "ta", "te", "ml", "mr", "bn", "gu",
             "pa", "ur"}
    language = request.form.get("language", "en")
    if language not in CODES:
        language = "auto"
    second = request.form.get("second", "")
    if second not in CODES or second == "auto":
        second = ""

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "name": name, "state": "running",
                        "log": [], "error": None}
    # Belt and braces: without Ollama the LLM pass cannot run, whatever the
    # form said.
    fast = request.form.get("fast") == "1" or not ollama_up()
    threading.Thread(target=run_job,
                     args=(job_id, dest, name, language, second, fast),
                     daemon=True).start()
    return redirect(url_for("status", job_id=job_id))


@app.get("/job/<job_id>")
def status(job_id):
    job = JOBS.get(job_id) or abort(404)
    preview = ""
    if job["state"] == "done":
        preview = (OUT / job["name"] / "transcript.txt").read_text()[:4000]
    return render_template_string(PAGE, job=job, preview=preview,
                                  ollama=ollama_up(),
                                  model_ready=asr_backend.model_cached())


@app.get("/job/<job_id>/download/<which>")
def download(job_id, which):
    job = JOBS.get(job_id) or abort(404)
    fname = {"transcript": "transcript.txt",
             "review": "transcript_review.txt"}.get(which) or abort(404)
    path = OUT / job["name"] / fname
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True,
                     download_name=f"{job['name']}_{fname}",
                     mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
