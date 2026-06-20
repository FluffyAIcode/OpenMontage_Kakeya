"""Distributed WAN tile worker (CUDA) — one per vast GPU/box.

A FastAPI server that does the two WAN data-plane operations the distributed
orchestrator dispatches over HTTP (region-tolerant: only prompts + base64-mp4 cross
the wire, never per-step tensors):

  POST /v1/framework    {prompt, width, height, num_frames, steps, seed}
        -> distilled CausVid proposer -> low-res framework, returns {mp4_b64}
  POST /v1/refine_tile  {prompt, width, height, num_frames, steps, strength, seed, mp4_b64}
        -> full-WAN vid2vid refine of the given (framework-crop) clip, returns {mp4_b64}
  GET  /healthz

This is CUDA-only (WAN is a CUDA diffusers model). The Mac mini's MLX GPU CANNOT run
WAN — it serves the separate text plane via Kakeya (see orchestrator KAKEYA_ENDPOINT).
Run >=2 of these on co-located CUDA GPUs for true data-plane parallelism.

Start:
    pip install fastapi "uvicorn[standard]" diffusers transformers peft imageio imageio-ffmpeg
    python worker.py --port 9000
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
import time
from typing import Optional

import numpy as np
import torch

# One GPU pipeline per worker -> serialize generation within a worker. Real
# parallelism comes from running MULTIPLE workers (one per co-located CUDA GPU).
_gpu_lock = threading.Lock()
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LORA_REPO = "Kijai/WanVideo_comfy"
LORA_FILE = "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"

_state = {}


class FrameworkReq(BaseModel):
    prompt: str
    negative_prompt: str = "worst quality, low quality, blurry"
    width: int = 832
    height: int = 480
    num_frames: int = 25
    steps: int = 6
    seed: int = 42


class RefineReq(BaseModel):
    prompt: str
    negative_prompt: str = "worst quality, low quality, blurry"
    mp4_b64: str
    width: int = 832
    height: int = 480
    num_frames: int = 25
    steps: int = 16
    strength: float = 0.6
    seed: int = 7


def _frames_to_b64_mp4(frames: np.ndarray, fps: int = 12) -> str:
    import imageio.v2 as iio
    buf = io.BytesIO()
    iio.mimwrite(buf, [frames[i] for i in range(frames.shape[0])], format="mp4", fps=fps, codec="libx264")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_mp4_to_frames(b64: str) -> np.ndarray:
    import imageio.v2 as iio
    data = base64.b64decode(b64)
    r = iio.get_reader(io.BytesIO(data), format="mp4")
    return np.stack([f for f in r])


def _np(frames):
    if isinstance(frames, np.ndarray):
        arr = frames
    elif hasattr(frames[0], "convert"):
        return np.stack([np.asarray(f.convert("RGB"), np.uint8) for f in frames])
    else:
        arr = np.stack([np.asarray(f) for f in frames])
    if arr.dtype != np.uint8:
        arr = (arr.clip(0, 1) * 255).round().astype(np.uint8) if float(arr.max()) <= 1.0 + 1e-3 else arr.clip(0, 255).astype(np.uint8)
    return arr


def _pil(arr):
    from PIL import Image
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]


def _load():
    if _state:
        return
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanVideoToVideoPipeline
    torch.set_grad_enabled(False)
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, adapter_name="causvid")
    v2v = WanVideoToVideoPipeline(**{k: pipe.components[k] for k in
                                     ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")})
    _state.update(pipe=pipe, v2v=v2v)


app = FastAPI(title="Distributed WAN tile worker", version="1.0.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model": MODEL, "loaded": bool(_state)}


@app.post("/v1/framework")
def framework(req: FrameworkReq):
    _load(); pipe = _state["pipe"]
    t = time.time()
    with _gpu_lock:
        pipe.set_adapters("causvid")  # distilled proposer
        g = torch.Generator("cpu").manual_seed(req.seed)
        out = pipe(prompt=req.prompt, negative_prompt=req.negative_prompt, num_frames=req.num_frames,
                   width=req.width, height=req.height, num_inference_steps=req.steps,
                   guidance_scale=1.0, generator=g)
        frames = _np(out.frames[0])
    return JSONResponse({"mp4_b64": _frames_to_b64_mp4(frames), "gen_s": round(time.time() - t, 2)})


@app.post("/v1/refine_tile")
def refine_tile(req: RefineReq):
    _load(); pipe = _state["pipe"]; v2v = _state["v2v"]
    frames = _b64_mp4_to_frames(req.mp4_b64)
    t = time.time()
    with _gpu_lock:
        pipe.disable_lora()  # full WAN verifier for refine
        g = torch.Generator("cpu").manual_seed(req.seed)
        out = v2v(prompt=req.prompt, negative_prompt=req.negative_prompt, video=_pil(frames),
                  strength=req.strength, num_inference_steps=req.steps, guidance_scale=5.0, generator=g)
        refined = _np(out.frames[0])
    return JSONResponse({"mp4_b64": _frames_to_b64_mp4(refined), "gen_s": round(time.time() - t, 2)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--preload", action="store_true")
    args = ap.parse_args()
    if args.preload:
        _load()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
