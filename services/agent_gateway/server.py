"""OpenMontage Agent Gateway — public front door for the agent video service.

This is the HTTP entry point that lets a user drive OpenMontage's video service over a
domain (e.g. https://kekaye.ai). It is a **transport + session/job layer only** — it holds
NO creative or orchestration logic (that stays with the OpenMontage agent + skills, per
AGENT_GUIDE / PROJECT_CONTEXT "Python = tools + persistence"). The gateway:

  1. accepts a request (prompt + options),
  2. opens a job, runs the VALIDATED distributed-WAN backend as a subprocess
     (`services/distributed_wan/grpc_orchestrator.py`, Mac-MLX proposer + CUDA refiner),
  3. streams progress + serves the resulting .mp4.

Two job modes:
  - mode="video"  (default): direct text->video via the distributed-WAN cluster. Real, works
    today with the workers from `services/distributed_wan`.
  - mode="agent": full multi-stage OpenMontage pipeline. The gateway enqueues the brief for an
    agent runtime (LLM) worker; it does NOT fake creative decisions. If no agent runtime is
    attached (AGENT_RUNTIME_CMD unset) it returns a clear, honest "not attached" status.

Run:
    pip install -r services/agent_gateway/requirements.txt
    WAN_WORKERS="127.0.0.1:50051,127.0.0.1:55051" \
    AGENT_GATEWAY_API_KEY="<optional-secret>" \
    python services/agent_gateway/server.py --host 0.0.0.0 --port 8088
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# Orchestrator location is env-overridable so the gateway can live apart from the
# distributed_wan checkout (e.g. on a rented GPU box).
ORCH = Path(os.environ.get("ORCHESTRATOR_PATH",
                           str(REPO / "services" / "distributed_wan" / "grpc_orchestrator.py")))
JOBS_DIR = Path(os.environ.get("AGENT_GATEWAY_JOBS_DIR", str(REPO / "projects" / "_gateway_jobs")))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("AGENT_GATEWAY_API_KEY", "").strip()
WAN_WORKERS = os.environ.get("WAN_WORKERS", "").strip()
AGENT_RUNTIME_CMD = os.environ.get("AGENT_RUNTIME_CMD", "").strip()
MAX_LOG = 200

_PCT = re.compile(r"(\w[\w ()@,]*?):\s+(\d+)%")
_DONE = re.compile(r"^ORCH_DONE\s+(\{.*\})\s*$")


@dataclass
class Job:
    id: str
    prompt: str
    mode: str
    params: dict
    status: str = "queued"        # queued | running | done | error
    stage: str = "queued"
    pct: float = 0.0
    log: list = field(default_factory=list)
    out: Optional[str] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {"id": self.id, "status": self.status, "stage": self.stage,
                "pct": round(self.pct, 3), "prompt": self.prompt, "mode": self.mode,
                "error": self.error, "created": self.created, "updated": self.updated,
                "log": self.log[-25:], "result": f"/v1/jobs/{self.id}/video" if self.out else None}


JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
# GPU backend is serialized; one job at a time keeps the cluster honest.
_POOL = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="OpenMontage Agent Gateway", version="1.0.0")


def _auth(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
    return True


class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)
    mode: str = Field("video", pattern="^(video|agent)$")
    frames: int = 25
    fw_width: int = 480
    fw_height: int = 256
    fw_frames: int = 13
    proposer_steps: int = 6
    refine_steps: int = 16
    seed: int = 11


def _log(job: Job, line: str):
    job.log.append(line)
    if len(job.log) > MAX_LOG:
        job.log = job.log[-MAX_LOG:]
    job.updated = time.time()


def _run_video_job(job: Job):
    if not WAN_WORKERS:
        job.status, job.error = "error", "WAN_WORKERS not configured on the gateway host"
        return
    if not ORCH.exists():
        job.status, job.error = "error", f"orchestrator not found: {ORCH}"
        return
    job.status, job.stage = "running", "framework"
    out_path = JOBS_DIR / job.id / "video.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = job.params
    cmd = ["python3", str(ORCH), "--prompt", job.prompt, "--frames", str(p["frames"]),
           "--fw-width", str(p["fw_width"]), "--fw-height", str(p["fw_height"]),
           "--fw-frames", str(p["fw_frames"]), "--proposer-steps", str(p["proposer_steps"]),
           "--refine-steps", str(p["refine_steps"]), "--seed", str(p["seed"]),
           "--out", str(out_path)]
    env = {**os.environ, "WAN_WORKERS": WAN_WORKERS}
    _log(job, f"$ grpc_orchestrator --prompt {job.prompt!r} (workers={WAN_WORKERS})")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        _log(job, line)
        m = _DONE.match(line)
        if m:
            try:
                info = json.loads(m.group(1))
                job.out = info.get("out", str(out_path))
            except json.JSONDecodeError:
                job.out = str(out_path)
            continue
        mp = _PCT.search(line)
        if mp:
            label, pct = mp.group(1).strip(), int(mp.group(2))
            job.stage = "refine" if label.startswith("tile") else "framework"
            job.pct = pct / 100.0
    proc.wait()
    if proc.returncode != 0 or not (job.out and Path(job.out).exists()):
        job.status = "error"
        job.error = job.error or f"orchestrator exited rc={proc.returncode}; see log"
        return
    job.status, job.stage, job.pct = "done", "done", 1.0
    _log(job, f"done -> {job.out}")


def _run_agent_job(job: Job):
    # Full multi-stage OpenMontage pipeline needs an agent runtime (LLM) — not faked here.
    if not AGENT_RUNTIME_CMD:
        job.status = "error"
        job.error = ("agent mode requires an attached agent runtime (set AGENT_RUNTIME_CMD to a "
                     "command that drives the OpenMontage pipeline for a brief). The gateway does "
                     "not fabricate creative decisions. Use mode='video' for direct generation.")
        return
    job.status, job.stage = "running", "agent"
    out_path = JOBS_DIR / job.id / "video.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = AGENT_RUNTIME_CMD.split() + ["--prompt", job.prompt, "--out", str(out_path)]
    _log(job, f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            _log(job, line)
    proc.wait()
    if proc.returncode == 0 and out_path.exists():
        job.out, job.status, job.stage, job.pct = str(out_path), "done", "done", 1.0
    else:
        job.status, job.error = "error", f"agent runtime rc={proc.returncode}; see log"


def _dispatch(job: Job):
    try:
        (_run_agent_job if job.mode == "agent" else _run_video_job)(job)
    except Exception as exc:  # noqa: BLE001
        job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        job.updated = time.time()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "workers": WAN_WORKERS or None, "orchestrator": ORCH.exists(),
            "agent_runtime": bool(AGENT_RUNTIME_CMD), "jobs": len(JOBS)}


@app.get("/v1/capabilities")
def capabilities():
    # Best-effort registry summary; never let a heavy import break the endpoint.
    try:
        import sys
        sys.path.insert(0, str(REPO))
        from tools.tool_registry import registry  # type: ignore
        registry.discover()
        return registry.provider_menu_summary()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"available": False, "note": f"registry summary unavailable: {exc}",
                             "backend": "distributed_wan", "workers": WAN_WORKERS or None})


@app.post("/v1/videos")
def create_video(req: VideoRequest, _=Depends(_auth)):
    job = Job(id=uuid.uuid4().hex[:12], prompt=req.prompt.strip(), mode=req.mode,
              params=req.model_dump())
    with _LOCK:
        JOBS[job.id] = job
    _POOL.submit(_dispatch, job)
    return {"job_id": job.id, "status": job.status, "poll": f"/v1/jobs/{job.id}"}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job.public()


@app.get("/v1/jobs/{job_id}/video")
def get_video(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.out or not Path(job.out).exists():
        raise HTTPException(status_code=404, detail="video not ready")
    return FileResponse(job.out, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


_INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenMontage — Agent Video</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font:16px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  background:radial-gradient(1200px 600px at 70% -10%,#1b2942,#0b1120);color:#e7eefc;min-height:100vh}
.wrap{max-width:780px;margin:0 auto;padding:48px 20px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.02em}
.sub{color:#8fa6c8;margin:0 0 28px}
textarea{width:100%;min-height:96px;padding:14px;border-radius:12px;border:1px solid #2a3a59;
  background:#0e1830;color:#e7eefc;resize:vertical;font:inherit}
.row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
button{background:linear-gradient(180deg,#4f8cff,#2f6bff);color:#fff;border:0;padding:12px 22px;
  border-radius:10px;font-weight:600;cursor:pointer;font-size:15px}
button:disabled{opacity:.5;cursor:not-allowed}
.muted{color:#8fa6c8;font-size:13px}
.card{margin-top:24px;background:#0e1830aa;border:1px solid #22324f;border-radius:14px;padding:18px}
.bar{height:8px;border-radius:8px;background:#1a2742;overflow:hidden;margin:10px 0}
.bar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#4f8cff,#7ee0ff);transition:width .4s}
pre{white-space:pre-wrap;word-break:break-word;max-height:200px;overflow:auto;background:#0a1124;
  border:1px solid #1c2942;border-radius:10px;padding:12px;font-size:12.5px;color:#a9c2e6}
video{width:100%;border-radius:12px;margin-top:12px;background:#000}
input[type=password]{padding:10px;border-radius:8px;border:1px solid #2a3a59;background:#0e1830;color:#e7eefc}
</style></head><body><div class="wrap">
<h1>OpenMontage — Agent Video</h1>
<p class="sub">Describe a clip. The distributed cluster (MLX proposer + CUDA refiner) renders it.</p>
<textarea id="p" placeholder="a red fox walking through a snowy forest, cinematic, soft winter light"></textarea>
<div class="row">
  <button id="go">Generate</button>
  <input id="key" type="password" placeholder="API key (if required)" style="flex:1;min-width:160px">
</div>
<p class="muted">Direct text→video via <code>/v1/videos</code>. First render can take a few minutes.</p>
<div class="card" id="status" style="display:none">
  <div id="stage" class="muted">queued…</div>
  <div class="bar"><i id="fill"></i></div>
  <video id="vid" controls style="display:none"></video>
  <pre id="log"></pre>
</div>
<script>
const $=id=>document.getElementById(id);
let timer=null;
async function poll(jid){
  const r=await fetch('/v1/jobs/'+jid); const j=await r.json();
  $('stage').textContent=j.status+' · '+j.stage+' · '+Math.round((j.pct||0)*100)+'%';
  $('fill').style.width=Math.round((j.pct||0)*100)+'%';
  $('log').textContent=(j.log||[]).join('\\n');
  if(j.status==='done'&&j.result){clearInterval(timer);$('go').disabled=false;
    const v=$('vid');v.src=j.result;v.style.display='block';$('stage').textContent='done · 100%';}
  if(j.status==='error'){clearInterval(timer);$('go').disabled=false;
    $('stage').textContent='error: '+(j.error||'see log');}
}
$('go').onclick=async()=>{
  const prompt=$('p').value.trim(); if(prompt.length<3)return;
  $('go').disabled=true;$('status').style.display='block';$('vid').style.display='none';
  $('stage').textContent='submitting…';$('fill').style.width='0';$('log').textContent='';
  const h={'Content-Type':'application/json'}; const k=$('key').value.trim(); if(k)h['X-API-Key']=k;
  const r=await fetch('/v1/videos',{method:'POST',headers:h,body:JSON.stringify({prompt})});
  if(!r.ok){$('go').disabled=false;$('stage').textContent='error: '+r.status+' '+(await r.text());return;}
  const {job_id}=await r.json();
  timer=setInterval(()=>poll(job_id),2500);poll(job_id);
};
</script></div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8088)
    args = ap.parse_args()
    import uvicorn
    print(f"[agent_gateway] http://{args.host}:{args.port}  workers={WAN_WORKERS or '(unset)'}  "
          f"auth={'on' if API_KEY else 'off'}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
