"""Distributed WAN gRPC worker (ADR 0010) — pluggable backend.

Implements the `distwan.v1.VideoWorker` gRPC service with server-streaming progress
and capability negotiation. Two backends:

  --backend cuda : diffusers WanPipeline + CausVid LoRA (vast/H200). Full ops
                   (framework + refine). VALIDATED live.
  --backend mlx  : wraps `mlx-video` (Blaizzy/mlx-video) on a Mac mini. Ops are
                   advertised from --mlx-ops (default "framework"); owner-run.

The orchestrator routes by Health.ops + relative_speed (speed-weighted), so a slow
Mac and a fast vast box are scheduled appropriately — "Mac as another GPU".

Run (vast / CUDA):
    pip install grpcio diffusers transformers peft accelerate imageio imageio-ffmpeg
    python grpc_worker.py --backend cuda --port 50051 --preload

Run (Mac mini / MLX) — see services/distributed_wan/mac_setup.sh:
    python grpc_worker.py --backend mlx --port 50051 \
        --mlx-model-dir ~/wan21_mlx --mlx-ops framework
"""

from __future__ import annotations

import argparse
import io
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_worker_pb2 as pb
import video_worker_pb2_grpc as pb_grpc

NEG_DEFAULT = "worst quality, low quality, blurry, distorted"


# --------------------------------------------------------------------------- #
# mp4 <-> frames helpers
# --------------------------------------------------------------------------- #
def frames_to_mp4(frames, fps: int = 12) -> bytes:
    import imageio.v2 as iio
    buf = io.BytesIO()
    iio.mimwrite(buf, [frames[i] for i in range(frames.shape[0])], format="mp4", fps=fps, codec="libx264")
    return buf.getvalue()


def mp4_to_frames(data: bytes):
    import imageio.v2 as iio
    import numpy as np
    r = iio.get_reader(io.BytesIO(data), format="mp4")
    return np.stack([f for f in r])


def _to_uint8(frames):
    import numpy as np
    if isinstance(frames, np.ndarray):
        arr = frames
    elif hasattr(frames[0], "convert"):
        return np.stack([np.asarray(f.convert("RGB"), np.uint8) for f in frames])
    else:
        arr = np.stack([np.asarray(f) for f in frames])
    if arr.dtype != np.uint8:
        arr = (arr.clip(0, 1) * 255).round().astype(np.uint8) if float(arr.max()) <= 1.0 + 1e-3 else arr.clip(0, 255).astype(np.uint8)
    return arr


# --------------------------------------------------------------------------- #
# CUDA backend (diffusers WAN + CausVid) — VALIDATED
# --------------------------------------------------------------------------- #
class CudaBackend:
    MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    LORA_REPO = "Kijai/WanVideo_comfy"
    LORA_FILE = "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        import torch
        from diffusers import (AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline,
                               WanVideoToVideoPipeline)
        torch.set_grad_enabled(False)
        vae = AutoencoderKLWan.from_pretrained(self.MODEL, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanPipeline.from_pretrained(self.MODEL, vae=vae, torch_dtype=torch.bfloat16)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=3.0)
        self.pipe.to("cuda")
        self.pipe.load_lora_weights(self.LORA_REPO, weight_name=self.LORA_FILE, adapter_name="causvid")
        self.v2v = WanVideoToVideoPipeline(**{k: self.pipe.components[k] for k in
                                              ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")})
        self._loaded = True

    def health(self):
        import torch
        return pb.HealthReply(device="cuda" if torch.cuda.is_available() else "cpu",
                              backend="cuda-diffusers", model=self.MODEL,
                              ops=["framework", "refine", "t2v"], ready=self._loaded,
                              note=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                              relative_speed=1.0)

    def _cb(self, q, total):
        def cb(pipe, i, t, kw):
            try:
                q.put_nowait(("p", (i + 1) / max(total, 1)))
            except queue.Full:
                pass
            return kw
        return cb

    def framework(self, req, q):
        import torch
        self.load()
        with self._lock:
            self.pipe.set_adapters("causvid")
            g = torch.Generator("cpu").manual_seed(req.seed or 42)
            out = self.pipe(prompt=req.prompt, negative_prompt=req.negative_prompt or NEG_DEFAULT,
                            num_frames=req.num_frames or 25, width=req.width or 832, height=req.height or 480,
                            num_inference_steps=req.steps or 6, guidance_scale=1.0, generator=g,
                            callback_on_step_end=self._cb(q, req.steps or 6))
            return _to_uint8(out.frames[0])

    def refine(self, req, q):
        import torch
        self.load()
        frames = mp4_to_frames(req.mp4)
        from PIL import Image
        pil = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
        with self._lock:
            self.pipe.disable_lora()
            g = torch.Generator("cpu").manual_seed(req.seed or 7)
            out = self.v2v(prompt=req.prompt, negative_prompt=req.negative_prompt or NEG_DEFAULT, video=pil,
                           strength=req.strength or 0.6, num_inference_steps=req.steps or 16,
                           guidance_scale=5.0, generator=g, callback_on_step_end=self._cb(q, req.steps or 16))
            return _to_uint8(out.frames[0])


# --------------------------------------------------------------------------- #
# MLX backend (wraps mlx-video) — Mac mini; OWNER-RUN (untested in CI; no Mac here)
# --------------------------------------------------------------------------- #
class MlxBackend:
    """Wraps `mlx-video` via its CLI. Ops advertised from --mlx-ops because the
    exact T2V/I2V/vid2vid surface depends on the installed mlx-video version.
    Adjust the command template to match your mlx-video if flags differ."""

    def __init__(self, model_dir: str, ops: list[str]):
        self.model_dir = model_dir
        self.ops = ops or ["framework"]
        # Version-adaptable knobs (no code edits needed):
        self.module = os.environ.get("MLX_T2V_MODULE", "mlx_video.wan_2.generate")
        self.pass_dims = os.environ.get("MLX_PASS_DIMS", "0").lower() in ("1", "true", "yes")
        self.v2v_flag = os.environ.get("MLX_V2V_FLAG", "")  # e.g. "--video" if your build has vid2vid

    def health(self):
        return pb.HealthReply(device="mlx", backend="mlx-video", model=self.model_dir,
                              ops=self.ops, ready=True, note="Apple Silicon (MLX)",
                              relative_speed=float(os.environ.get("MLX_RELATIVE_SPEED", "0.12")))

    def _run(self, prompt, seed, num_frames, steps, q, extra=None):
        out = Path(tempfile.mkstemp(suffix=".mp4")[1])
        # Documented mlx-video flags (README): --model-dir/--prompt/--output-path/--seed.
        cmd = ["python", "-m", self.module, "--model-dir", self.model_dir,
               "--prompt", prompt, "--output-path", str(out), "--seed", str(seed)]
        # num-frames/steps may be config-driven in some mlx-video versions; opt-in.
        if self.pass_dims:
            if num_frames:
                cmd += ["--num-frames", str(num_frames)]
            if steps:
                cmd += ["--steps", str(steps)]
        if extra:
            cmd += extra
        q.put_nowait(("p", 0.05))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        import re
        tail = []
        for line in proc.stdout:  # best-effort progress from mlx-video stdout ("k/N")
            tail.append(line)
            m = re.search(r"(\d+)\s*/\s*(\d+)", line)
            if m:
                try:
                    q.put_nowait(("p", int(m.group(1)) / max(int(m.group(2)), 1)))
                except queue.Full:
                    pass
        proc.wait()
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"mlx-video failed (rc={proc.returncode}). Last output:\n"
                               + "".join(tail[-15:]) + "\nAdjust MLX_T2V_MODULE/MLX_PASS_DIMS to your mlx-video.")
        data = out.read_bytes()
        out.unlink(missing_ok=True)
        return _to_uint8(mp4_to_frames(data))

    def framework(self, req, q):
        if not ({"framework", "t2v"} & set(self.ops)):
            raise RuntimeError("this mlx worker is not configured for framework/t2v")
        return self._run(req.prompt, req.seed or 42, req.num_frames or 25, req.steps or 6, q)

    def refine(self, req, q):
        # mlx-video typically has NO vid2vid; the Mac usually serves framework/T2V only and
        # refines run on a CUDA worker. Enable refine ONLY if your mlx-video has vid2vid.
        if "refine" not in self.ops:
            raise RuntimeError("this mlx worker advertises no 'refine' op (mlx-video usually lacks "
                               "vid2vid). Route refines to a CUDA worker; use the Mac for framework/T2V.")
        if not self.v2v_flag:
            raise RuntimeError("set MLX_V2V_FLAG (e.g. --video) to your mlx-video's vid2vid input flag")
        inp = Path(tempfile.mkstemp(suffix=".mp4")[1]); inp.write_bytes(req.mp4)
        try:
            return self._run(req.prompt, req.seed or 7, req.num_frames or 25, req.steps or 16, q,
                             extra=[self.v2v_flag, str(inp), "--strength", str(req.strength or 0.6)])
        finally:
            inp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Test backend (NO GPU) — validates the gRPC TRANSPORT only (proto/streaming/
# routing/geometry), not the model. The model path is the cuda backend above.
# --------------------------------------------------------------------------- #
class TestBackend:
    def health(self):
        return pb.HealthReply(device=os.environ.get("TEST_DEVICE", "cpu"), backend="test-synthetic",
                              model="none", ops=["framework", "refine", "t2v"], ready=True,
                              note="transport smoke test (no model)",
                              relative_speed=float(os.environ.get("TEST_SPEED", "1.0")))

    def framework(self, req, q):
        import numpy as np
        for i in range(4):
            q.put_nowait(("p", (i + 1) / 4)); time.sleep(0.03)
        n, h, w = req.num_frames or 8, req.height or 480, req.width or 832
        arr = np.zeros((n, h, w, 3), np.uint8); arr[..., :] = (40, 80, 160)
        return arr

    def refine(self, req, q):
        import numpy as np
        for i in range(4):
            q.put_nowait(("p", (i + 1) / 4)); time.sleep(0.03)
        frames = mp4_to_frames(req.mp4).copy()
        frames[..., 0] = np.clip(frames[..., 0].astype(int) + 50, 0, 255)  # tint to show the refine ran
        return frames


# --------------------------------------------------------------------------- #
# gRPC servicer (streaming progress)
# --------------------------------------------------------------------------- #
class VideoWorkerServicer(pb_grpc.VideoWorkerServicer):
    def __init__(self, backend):
        self.backend = backend

    def Health(self, request, context):
        return self.backend.health()

    def _stream(self, op_fn, request, context):
        q: queue.Queue = queue.Queue(maxsize=256)
        result = {}

        def run():
            try:
                t = time.time()
                result["frames"] = op_fn(request, q)
                result["t"] = time.time() - t
            except Exception as exc:  # noqa: BLE001
                result["err"] = f"{type(exc).__name__}: {exc}"
            finally:
                q.put(("done", None))

        threading.Thread(target=run, daemon=True).start()
        yield pb.Progress(pct=0.0, stage="load")
        while True:
            kind, val = q.get()
            if kind == "done":
                break
            yield pb.Progress(pct=float(val), stage="denoise")
        if "err" in result:
            context.abort(grpc.StatusCode.INTERNAL, result["err"])
            return
        yield pb.Progress(pct=1.0, stage="done", done=True,
                          mp4=frames_to_mp4(result["frames"]), gen_seconds=float(result["t"]))

    def GenerateFramework(self, request, context):
        yield from self._stream(self.backend.framework, request, context)

    def RefineTile(self, request, context):
        yield from self._stream(self.backend.refine, request, context)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cuda", "mlx", "test"], default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--mlx-model-dir", default=os.environ.get("MLX_MODEL_DIR", ""))
    ap.add_argument("--mlx-ops", default=os.environ.get("MLX_OPS", "framework"))
    args = ap.parse_args()

    if args.backend == "cuda":
        backend = CudaBackend()
        if args.preload:
            backend.load()
    elif args.backend == "test":
        backend = TestBackend()
    else:
        backend = MlxBackend(args.mlx_model_dir, [o.strip() for o in args.mlx_ops.split(",") if o.strip()])

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8),
                         options=[("grpc.max_send_message_length", 256 * 1024 * 1024),
                                  ("grpc.max_receive_message_length", 256 * 1024 * 1024)])
    pb_grpc.add_VideoWorkerServicer_to_server(VideoWorkerServicer(backend), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    print(f"[grpc_worker] {args.backend} backend listening on {args.host}:{args.port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
