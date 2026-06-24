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
import math
import os
import queue
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
WORKER_LIST = [a.strip() for a in WAN_WORKERS.split(",") if a.strip()]
# Dynamic membership (non-pool modes): if set, the worker list is re-read from this file on every
# job + healthz, so an EPHEMERAL refiner (e.g. a vast box that is released/recreated) can be added
# or removed by a supervisor WITHOUT restarting the gateway. Falls back to WAN_WORKERS when unset
# or unreadable. One line, comma- or newline-separated "host:port" entries.
WORKERS_FILE = os.environ.get("AGENT_GATEWAY_WORKERS_FILE", "").strip()
AGENT_RUNTIME_CMD = os.environ.get("AGENT_RUNTIME_CMD", "").strip()


def current_workers() -> list[str]:
    """Live worker list: the workers-file (if set+readable) else the static WAN_WORKERS env."""
    if WORKERS_FILE:
        try:
            txt = Path(WORKERS_FILE).read_text().strip()
            if txt:
                return [a.strip() for a in txt.replace("\n", ",").split(",") if a.strip()]
        except OSError:
            pass
    return WORKER_LIST
# Worker-pool mode: treat each WAN_WORKERS entry as an INDEPENDENT GPU (e.g. two Thunderbolt-
# bridged Mac minis). Each job runs on ONE worker (DIRECT no-refine), N jobs in parallel = N×
# throughput. Off => classic single-orchestrator call using ALL workers (proposer+CUDA-refiner).
def _resolve_mode() -> str:
    """Gateway scheduling mode (AGENT_GATEWAY_MODE), with back-compat for the old POOL flag.

      auto        (default): ONE job uses ALL workers; the orchestrator auto-picks single-pass
                  (MLX refiner) or tiled (CUDA refiner) refine. Covers 2-Mac (head+headless) and
                  3-node (head + headless + vast) without a flag.
      distributed: like auto but forces tiled refine and round-robin tile spread so EVERY refiner
                  (incl. a slow MLX Mac) gets work — head=proposer, headless+vast=refiners.
      pipeline:   force single-pass refine on ONE refiner (head=proposer, one refiner).
      pool:       each job runs DIRECT on ONE worker, N jobs in parallel (throughput; every
                  worker must be framework-capable).
    """
    m = os.environ.get("AGENT_GATEWAY_MODE", "").strip().lower()
    if not m:
        m = ("pool" if os.environ.get("AGENT_GATEWAY_WORKER_POOL", "0").lower()
             in ("1", "true", "yes") else "auto")
    if m not in ("auto", "distributed", "pipeline", "pool"):
        m = "auto"
    if m == "pool" and len(WORKER_LIST) <= 1:
        m = "auto"  # pool needs >1 worker to mean anything
    return m


GW_MODE = _resolve_mode()
POOL_MODE = (GW_MODE == "pool")
PIPELINE_MODE = (GW_MODE == "pipeline")
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
                "quality": self.params.get("quality"), "error": self.error,
                "created": self.created, "updated": self.updated,
                "log": self.log[-25:], "result": f"/v1/jobs/{self.id}/video" if self.out else None}


JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
# In pool mode, run one job per worker concurrently; otherwise serialize (the single cluster
# is one render at a time). A queue hands out free worker addresses to jobs.
_POOL = ThreadPoolExecutor(max_workers=(len(WORKER_LIST) if POOL_MODE else 1))
_WORKER_Q: queue.Queue = queue.Queue()
for _a in (WORKER_LIST if POOL_MODE else []):
    _WORKER_Q.put(_a)

app = FastAPI(title="OpenMontage Agent Gateway", version="1.0.0")


def _auth(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
    return True


# Quality presets (ADR 0015): knobs only — topology stays with AGENT_GATEWAY_MODE, except 'high'
# routes a seam-free full-frame generative refine to the best (CUDA) refiner via refine_mode=single.
QUALITY_PRESETS: dict[str, dict] = {
    "draft": {"fw_width": 384, "fw_height": 224, "fw_frames": 13, "proposer_steps": 5,
              "refine_steps": 12, "out_width": 640, "out_height": 384, "frames": 25,
              "fps": 12, "interpolate": 1, "interp_method": "linear", "refine_mode": "direct"},
    # standard = native T2V @ 832x480 DIRECT (same reliable path as 'high'; the tiled v2v refine
    # hangs in the worker thread, so all tiers use direct native generation, differing by res/frames).
    "standard": {"fw_width": 832, "fw_height": 480, "fw_frames": 25, "proposer_steps": 6,
                 "refine_steps": 16, "out_width": 832, "out_height": 480, "frames": 25,
                 "fps": 16, "interpolate": 1, "interp_method": "linear", "refine_mode": "direct"},
    # 'high' = native WAN T2V @ 1280x720 DIRECT (reliable). Hunyuan (13B) would be断崖式 better, but
    # its heavy VAE decode HANGS when run inside the worker's gRPC daemon thread (same stall as WAN
    # v2v; standalone Hunyuan/v2v both work — only heavy decode in the servicer thread hangs). Until
    # the worker runs the heavy op off the daemon thread (main thread / subprocess), high stays on the
    # reliable WAN native path. The --prefer-backend='hunyuan' routing is wired and ready (ADR 0017);
    # set prefer_backend below to flip to Hunyuan once the worker concurrency fix lands.
    "high": {"fw_width": 1280, "fw_height": 720, "fw_frames": 25, "proposer_steps": 8,
             "refine_steps": 24, "out_width": 1280, "out_height": 720, "frames": 25,
             "fps": 24, "interpolate": 2, "interp_method": "mci", "refine_mode": "direct"},
}


class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)
    mode: str = Field("video", pattern="^(video|agent)$")
    quality: str = Field("standard", pattern="^(draft|standard|high)$")
    # Duration / smoothness (ADR 0015). seconds>0 overrides frames (= round(seconds*fps)).
    seconds: Optional[float] = Field(None, ge=0, le=20)
    # Optional per-knob overrides; when None the quality preset value is used.
    frames: Optional[int] = None
    fps: Optional[int] = None
    interpolate: Optional[int] = Field(None, ge=1, le=4)
    interp_method: Optional[str] = Field(None, pattern="^(linear|mci)$")
    fw_width: Optional[int] = None
    fw_height: Optional[int] = None
    fw_frames: Optional[int] = None
    proposer_steps: Optional[int] = None
    refine_steps: Optional[int] = None
    out_width: Optional[int] = None
    out_height: Optional[int] = None
    refine_mode: Optional[str] = Field(None, pattern="^(auto|direct|single|tiled)$")
    prefer_backend: Optional[str] = Field(None, pattern="^[a-z0-9_-]{0,24}$")
    # Long-form (ADR 0015 Phase 2): chunks>1 (or longform=true + seconds) = autoregressive I2V gen.
    longform: bool = False
    chunks: Optional[int] = Field(None, ge=1, le=20)
    chunk_frames: Optional[int] = None
    chunk_overlap: Optional[int] = Field(None, ge=0, le=12)
    seed: int = 11

    def resolved(self) -> dict:
        """Merge preset + explicit overrides into a concrete param dict for the orchestrator."""
        p = dict(QUALITY_PRESETS.get(self.quality, QUALITY_PRESETS["standard"]))
        for k in ("frames", "fps", "interpolate", "interp_method", "fw_width", "fw_height",
                  "fw_frames", "proposer_steps", "refine_steps", "out_width", "out_height", "refine_mode"):
            v = getattr(self, k)
            if v is not None:
                p[k] = v
        p["seed"] = self.seed
        p["seconds"] = self.seconds or 0.0
        p["chunks"] = self.chunks or 1
        p["chunk_frames"] = self.chunk_frames or 25
        p["chunk_overlap"] = self.chunk_overlap if self.chunk_overlap is not None else 4
        # Long-form (opt-in): derive chunk count from target seconds when longform is requested.
        if self.longform and p["chunks"] == 1 and p["seconds"] > 0:
            net = max(1, p["chunk_frames"] - p["chunk_overlap"])
            want = round(p["seconds"] * p["fps"])
            if want > p["chunk_frames"]:
                p["chunks"] = max(1, math.ceil((want - p["chunk_overlap"]) / net))
        if p["chunks"] == 1 and p["seconds"] > 0:  # single-pass: seconds drives frame count
            p["frames"] = max(2, round(p["seconds"] * p["fps"]))
        if self.prefer_backend is not None:
            p["prefer_backend"] = self.prefer_backend
        p["quality"] = self.quality
        return p


def _log(job: Job, line: str):
    job.log.append(line)
    if len(job.log) > MAX_LOG:
        job.log = job.log[-MAX_LOG:]
    job.updated = time.time()


def _run_video_job(job: Job):
    live = current_workers()
    if not live:
        job.status, job.error = "error", "no workers configured on the gateway host"
        return
    if not ORCH.exists():
        job.status, job.error = "error", f"orchestrator not found: {ORCH}"
        return
    # In pool mode acquire ONE worker (blocks until a Mac is free); else use the LIVE worker list
    # (dynamic when AGENT_GATEWAY_WORKERS_FILE is set, so an ephemeral vast refiner can come/go).
    worker_addr = _WORKER_Q.get() if POOL_MODE else None
    job_workers = worker_addr if POOL_MODE else ",".join(live)
    job.status, job.stage = "running", "framework"
    out_path = JOBS_DIR / job.id / "video.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = job.params
    cmd = ["python3", str(ORCH), "--prompt", job.prompt, "--frames", str(p["frames"]),
           "--fw-width", str(p["fw_width"]), "--fw-height", str(p["fw_height"]),
           "--fw-frames", str(p["fw_frames"]), "--proposer-steps", str(p["proposer_steps"]),
           "--refine-steps", str(p["refine_steps"]), "--seed", str(p["seed"]),
           "--out-width", str(p.get("out_width", 832)), "--out-height", str(p.get("out_height", 480)),
           "--fps", str(p.get("fps", 12)), "--interpolate", str(p.get("interpolate", 1)),
           "--interp-method", str(p.get("interp_method", "linear")),
           "--out", str(out_path)]
    if p.get("seconds", 0):
        cmd += ["--seconds", str(p["seconds"])]
    if p.get("refine_mode"):  # quality preset / explicit override of refine topology
        cmd += ["--refine-mode", p["refine_mode"]]
    if p.get("prefer_backend"):  # pin a tier to a model worker (e.g. 'high' -> hunyuan)
        cmd += ["--prefer-backend", p["prefer_backend"]]
    if p.get("chunks", 1) > 1:  # long-form (autoregressive I2V continuity)
        cmd += ["--chunks", str(p["chunks"]), "--chunk-frames", str(p["chunk_frames"]),
                "--chunk-overlap", str(p["chunk_overlap"])]
    if POOL_MODE:
        cmd.append("--no-refine")  # each pooled worker is a single framework-only GPU (Mac MLX)
    elif GW_MODE == "pipeline":
        cmd.append("--single-refine")  # head proposes, one refiner does the whole clip (SR/V2V)
    elif GW_MODE == "distributed":
        # head proposes, ALL refiners share the tiled refine; round-robin so every refiner
        # (incl. the slow MLX Mac) gets at least its share of tiles.
        cmd += ["--refine-spread", "roundrobin"]
    # auto: pass neither -> orchestrator decides single-pass (MLX refiner) vs tiled (CUDA refiner).
    env = {**os.environ, "WAN_WORKERS": job_workers}
    _log(job, f"$ grpc_orchestrator --prompt {job.prompt!r} (worker={job_workers})")
    try:
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
                job.stage = "refine" if (label.startswith("tile") or label.startswith("refine")) else "framework"
                job.pct = pct / 100.0
        proc.wait()
    finally:
        if POOL_MODE and worker_addr is not None:
            _WORKER_Q.put(worker_addr)  # release the Mac for the next job
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
    live = current_workers()
    return {"status": "ok", "workers": live or None, "mode": GW_MODE,
            "pool_mode": POOL_MODE, "pipeline_mode": PIPELINE_MODE,
            "workers_file": WORKERS_FILE or None, "dynamic_workers": bool(WORKERS_FILE),
            "parallel": (len(WORKER_LIST) if POOL_MODE else 1), "orchestrator": ORCH.exists(),
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
              params=req.resolved())
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
select{padding:9px 10px;border-radius:8px;border:1px solid #2a3a59;background:#0e1830;color:#e7eefc;font:inherit}
label.fld{display:flex;flex-direction:column;gap:4px;font-size:12px;color:#8fa6c8}
label.fld select:disabled{opacity:.45}
</style></head><body><div class="wrap">
<h1>OpenMontage — Agent Video</h1>
<p class="sub">Describe a clip. The distributed cluster (MLX proposer + CUDA refiner) renders it.</p>
<textarea id="p" placeholder="a red fox walking through a snowy forest, cinematic, soft winter light"></textarea>
<div class="row">
  <label class="fld">Mode
    <select id="mode">
      <option value="video">video · direct (fast)</option>
      <option value="agent">agent · full pipeline</option>
    </select>
  </label>
  <label class="fld">Quality
    <select id="quality">
      <option value="draft">draft · 640×384</option>
      <option value="standard" selected>standard · 832×480</option>
      <option value="high">high · 1280×720</option>
    </select>
  </label>
  <label class="fld">Length
    <select id="len">
      <option value="">auto</option>
      <option value="3">3s</option>
      <option value="5">5s</option>
      <option value="8">8s</option>
      <option value="10">10s</option>
    </select>
  </label>
</div>
<div class="row">
  <button id="go">Generate</button>
  <input id="key" type="password" placeholder="API key (remembered in this browser)" style="flex:1;min-width:160px">
</div>
<p class="muted" id="hint">Direct text→video via <code>/v1/videos</code>. First render can take a few minutes.</p>
<div class="card" id="status" style="display:none">
  <div id="stage" class="muted">queued…</div>
  <div class="bar"><i id="fill"></i></div>
  <video id="vid" controls style="display:none"></video>
  <pre id="log"></pre>
</div>
<script>
const $=id=>document.getElementById(id);
let timer=null;
// Remember the API key in this browser so it only has to be entered once.
try{const sk=localStorage.getItem('om_api_key');if(sk)$('key').value=sk;}catch(e){}
function syncMode(){
  const agent=$('mode').value==='agent';
  // quality/length apply to direct video; agent mode plans its own scenes, pacing and length.
  $('quality').disabled=agent; $('len').disabled=agent;
  $('hint').innerHTML=agent
    ? 'Agent mode: the LLM plans script → scenes → directed prompts, the cluster renders each, then composes one video. Quality/Length below apply to <em>direct video</em> mode only. This takes several minutes.'
    : 'Direct text→video via <code>/v1/videos</code>. First render can take a few minutes.';
}
$('mode').onchange=syncMode; syncMode();
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
  const h={'Content-Type':'application/json'}; const k=$('key').value.trim();
  if(k){h['X-API-Key']=k; try{localStorage.setItem('om_api_key',k);}catch(e){}}
  const mode=$('mode').value; const body={prompt,mode};
  if(mode==='video'){ body.quality=$('quality').value; const s=$('len').value; if(s)body.seconds=parseFloat(s); }
  const r=await fetch('/v1/videos',{method:'POST',headers:h,body:JSON.stringify(body)});
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
    print(f"[agent_gateway] http://{args.host}:{args.port}  workers={WORKER_LIST or '(unset)'}  "
          f"mode={GW_MODE} parallel={len(WORKER_LIST) if POOL_MODE else 1}  "
          f"auth={'on' if API_KEY else 'off'}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
