"""Warm video-inference gateway — REAL diffusers serving (ADR 0002 §5).

A standalone FastAPI service that hosts the open-source video-diffusion models
(WAN 2.1 / Hunyuan / CogVideo / LTX) **warm** behind one HTTP API, so OpenMontage's
local video tools route here instead of cold-loading a pipeline per call.

This is NOT a mock and NOT Kakeya. It loads real model weights, runs a real VAE
decode, and returns a real, decodable .mp4. Kakeya is an LLM token engine and
cannot host video diffusion (ADR 0002 §2).

Run:
    pip install torch diffusers transformers accelerate fastapi "uvicorn[standard]" \
        imageio imageio-ffmpeg sentencepiece pillow safetensors
    python server.py --host 0.0.0.0 --port 8080

Contract (ADR 0002 §5):
    GET  /healthz                -> {"status":"ok","device":...,"loaded":[...],"known":[...]}
    POST /v1/video/generations   -> raw video/mp4 bytes
"""

from __future__ import annotations

import argparse
import base64
import io
import tempfile
import threading
import time
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# hf model id -> how to serve it. Mirrors tools/video/_shared.py variants so the
# gateway speaks the same model ids OpenMontage already sends.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {"pipeline": "WanPipeline", "fps": 16},
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": {"pipeline": "WanPipeline", "fps": 16},
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": {"pipeline": "WanImageToVideoPipeline", "fps": 16},
    "tencent/HunyuanVideo-1.5": {"pipeline": "HunyuanVideoPipeline", "fps": 24},
    "Lightricks/LTX-2": {"pipeline": "LTXPipeline", "fps": 24},
    "Lightricks/LTX-Video": {"pipeline": "LTXPipeline", "fps": 24},
    "THUDM/CogVideoX-5b": {"pipeline": "CogVideoXPipeline", "fps": 8},
    "THUDM/CogVideoX-2b": {"pipeline": "CogVideoXPipeline", "fps": 8},
}


class GenRequest(BaseModel):
    model: str
    prompt: str
    operation: str = "text_to_video"
    width: int = 768
    height: int = 512
    num_frames: int = 49
    num_inference_steps: int = 30
    fps: Optional[int] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    image_b64: Optional[str] = None
    image_url: Optional[str] = None


class WarmPool:
    """Lazy-load and keep diffusers pipelines warm, keyed by hf model id."""

    def __init__(self) -> None:
        self._pipes: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16
            if (self.device == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def loaded(self) -> list[str]:
        return sorted(self._pipes.keys())

    def get(self, model_id: str):
        if model_id not in MODEL_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"unknown model {model_id!r}; known: {sorted(MODEL_REGISTRY)}",
            )
        with self._lock:
            if model_id in self._pipes:
                return self._pipes[model_id]
            import diffusers

            spec = MODEL_REGISTRY[model_id]
            pipe_cls = getattr(diffusers, spec["pipeline"])
            t0 = time.time()
            pipe = pipe_cls.from_pretrained(model_id, torch_dtype=self.dtype)
            pipe = pipe.to(self.device)
            if getattr(pipe, "vae", None) is not None:
                if hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
                if hasattr(pipe.vae, "enable_slicing"):
                    pipe.vae.enable_slicing()
            print(f"[warm] loaded {model_id} in {time.time() - t0:.1f}s on {self.device}")
            self._pipes[model_id] = pipe
            return pipe


pool = WarmPool()
app = FastAPI(title="OpenMontage Video Inference Gateway", version="1.0.0")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": pool.device,
        "dtype": str(pool.dtype),
        "loaded": pool.loaded(),
        "known": sorted(MODEL_REGISTRY),
    }


def _decode_image(req: GenRequest):
    from PIL import Image

    if req.image_b64:
        raw = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    elif req.image_url:
        import requests

        resp = requests.get(req.image_url, timeout=60)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        raise HTTPException(status_code=400, detail="image_to_video requires image_b64 or image_url")
    return img.resize((req.width, req.height), Image.LANCZOS)


@app.post("/v1/video/generations")
def generate(req: GenRequest) -> Response:
    spec = MODEL_REGISTRY.get(req.model)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
    pipe = pool.get(req.model)

    kwargs: dict[str, Any] = {
        "prompt": req.prompt,
        "num_frames": req.num_frames,
        "width": req.width,
        "height": req.height,
        "num_inference_steps": req.num_inference_steps,
    }
    if req.negative_prompt:
        kwargs["negative_prompt"] = req.negative_prompt
    if req.seed is not None:
        kwargs["generator"] = torch.Generator(device="cpu").manual_seed(int(req.seed))
    if req.operation == "image_to_video":
        kwargs["image"] = _decode_image(req)

    t0 = time.time()
    try:
        output = pipe(**kwargs)
    except Exception as exc:  # noqa: BLE001 - surface the real engine error
        raise HTTPException(status_code=500, detail=f"generation failed: {type(exc).__name__}: {exc}")
    frames = output.frames[0] if hasattr(output, "frames") else output.images

    from diffusers.utils import export_to_video

    fps = req.fps or spec["fps"]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = tmp.name
    export_to_video(frames, out_path, fps=fps)
    with open(out_path, "rb") as fh:
        data = fh.read()
    print(
        f"[gen] {req.model} {req.width}x{req.height} f={req.num_frames} "
        f"steps={req.num_inference_steps} -> {len(data)} bytes in {time.time() - t0:.1f}s"
    )
    return Response(
        content=data,
        media_type="video/mp4",
        headers={"X-Gen-Seconds": f"{time.time() - t0:.1f}", "X-Model": req.model},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--preload",
        default="",
        help="comma-separated hf model ids to load warm at startup",
    )
    args = ap.parse_args()
    for mid in [m.strip() for m in args.preload.split(",") if m.strip()]:
        pool.get(mid)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
