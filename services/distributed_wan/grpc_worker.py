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
        # Optional I2V model for long-form continuity (ADR 0015 Phase 2). T2V 1.3B has no I2V, so
        # I2V needs a separate checkpoint (e.g. "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"). Only when
        # CUDA_I2V_MODEL is set do we advertise "i2v" and honor FrameworkRequest.init_image.
        self.i2v_model = os.environ.get("CUDA_I2V_MODEL", "").strip()
        self._i2v = None

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
        # VAE tiling/slicing: REQUIRED for the 720p refine path. The Wan VAE encode+decode of a
        # full-frame 1280x720 clip in fp32 otherwise allocates a huge contiguous tensor and HANGS
        # (observed: refine denoise completes, then the worker is stuck in VAE decode at GPU 0%).
        # Tiling processes the frame in spatial tiles; slicing splits the temporal batch.
        for _vae in (vae,):
            try:
                _vae.enable_tiling(); _vae.enable_slicing()
            except Exception:  # noqa: BLE001
                pass
        self.pipe.load_lora_weights(self.LORA_REPO, weight_name=self.LORA_FILE, adapter_name="causvid")
        self.v2v = WanVideoToVideoPipeline(**{k: self.pipe.components[k] for k in
                                              ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")})
        self._loaded = True

    def _load_i2v(self):
        if self._i2v is not None or not self.i2v_model:
            return
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
        from transformers import CLIPVisionModel
        # diffusers Wan I2V recipe: CLIP image encoder + Wan VAE (fp32) + transformer (bf16);
        # flow_shift=5.0 is the 720P setting. CUDA_I2V_OFFLOAD=1 enables CPU offload if VRAM-tight.
        image_encoder = CLIPVisionModel.from_pretrained(self.i2v_model, subfolder="image_encoder",
                                                        torch_dtype=torch.float32)
        vae = AutoencoderKLWan.from_pretrained(self.i2v_model, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(self.i2v_model, vae=vae, image_encoder=image_encoder,
                                                       torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
        try:  # tiling/slicing keeps 720P I2V VAE encode/decode within memory (see load())
            vae.enable_tiling(); vae.enable_slicing()
        except Exception:  # noqa: BLE001
            pass
        if os.environ.get("CUDA_I2V_OFFLOAD", "0").lower() in ("1", "true", "yes"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self._i2v = pipe

    def health(self):
        import torch
        ops = ["framework", "refine", "t2v"] + (["i2v"] if self.i2v_model else [])
        return pb.HealthReply(device="cuda" if torch.cuda.is_available() else "cpu",
                              backend="cuda-diffusers", model=self.MODEL, ops=ops, ready=self._loaded,
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
        # I2V continuity chunk: generate conditioned on the seed image (long-form, ADR 0015 Phase 2).
        if getattr(req, "init_image", b"") and self.i2v_model:
            self._load_i2v()
            from PIL import Image
            w = max(16, ((req.width or 1280) // 16) * 16)      # Wan needs multiples of 16
            h = max(16, ((req.height or 720) // 16) * 16)
            nf = req.num_frames or 81
            if nf % 4 != 1:                                     # Wan needs num_frames == 4n+1
                nf = (nf // 4) * 4 + 1
            img = Image.open(io.BytesIO(req.init_image)).convert("RGB").resize((w, h))
            with self._lock:
                g = torch.Generator("cpu").manual_seed(req.seed or 42)
                out = self._i2v(image=img, prompt=req.prompt, negative_prompt=req.negative_prompt or NEG_DEFAULT,
                                num_frames=nf, width=w, height=h, num_inference_steps=req.steps or 30,
                                guidance_scale=5.0, generator=g,
                                callback_on_step_end=self._cb(q, req.steps or 30))
                return _to_uint8(out.frames[0])
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
# HunyuanVideo backend (diffusers) — a higher-quality open-source provider (ADR 0017).
# T2V via HunyuanVideoPipeline, I2V via HunyuanVideoImageToVideoPipeline. 13B; native 720p,
# ~5s clips. Heavy: CPU offload on by default (HUNYUAN_OFFLOAD=1). CUDA/vast only (no MLX).
# --------------------------------------------------------------------------- #
class HunyuanBackend:
    MODEL = os.environ.get("HUNYUAN_MODEL", "hunyuanvideo-community/HunyuanVideo")

    def __init__(self, ops):
        self.ops = ops or ["framework"]
        self._lock = threading.Lock()
        self.pipe = None
        self._i2v = None
        self.offload = os.environ.get("HUNYUAN_OFFLOAD", "1").lower() in ("1", "true", "yes")

    def _load_t2v(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        torch.set_grad_enabled(False)
        tr = HunyuanVideoTransformer3DModel.from_pretrained(self.MODEL, subfolder="transformer",
                                                            torch_dtype=torch.bfloat16)
        p = HunyuanVideoPipeline.from_pretrained(self.MODEL, transformer=tr, torch_dtype=torch.float16)
        p.vae.enable_tiling()
        p.enable_model_cpu_offload() if self.offload else p.to("cuda")
        self.pipe = p

    def _load_i2v(self):
        if self._i2v is not None:
            return
        import torch
        from diffusers import HunyuanVideoImageToVideoPipeline
        p = HunyuanVideoImageToVideoPipeline.from_pretrained(self.MODEL, torch_dtype=torch.float16)
        p.vae.enable_tiling()
        p.enable_model_cpu_offload() if self.offload else p.to("cuda")
        self._i2v = p

    def health(self):
        import torch
        return pb.HealthReply(device="cuda" if torch.cuda.is_available() else "cpu",
                              backend="hunyuan-diffusers", model=self.MODEL, ops=self.ops, ready=False,
                              note=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                              relative_speed=float(os.environ.get("HUNYUAN_RELATIVE_SPEED", "0.8")))

    def _cb(self, q, total):
        def cb(pipe, i, t, kw):
            try:
                q.put_nowait(("p", (i + 1) / max(total, 1)))
            except queue.Full:
                pass
            return kw
        return cb

    @staticmethod
    def _snap(req):
        w = max(16, (req.width or 1280) // 16 * 16)
        h = max(16, (req.height or 720) // 16 * 16)
        nf = req.num_frames or 61
        if nf % 4 != 1:                       # Hunyuan needs num_frames == 4k+1
            nf = (nf // 4) * 4 + 1
        return w, h, nf

    def framework(self, req, q):
        import torch
        if getattr(req, "init_image", b"") and "i2v" in self.ops:
            return self.i2v(req, q)
        self._load_t2v()
        w, h, nf = self._snap(req)
        with self._lock:
            g = torch.Generator("cpu").manual_seed(req.seed or 42)
            out = self.pipe(prompt=req.prompt, height=h, width=w, num_frames=nf,
                            num_inference_steps=req.steps or 30, generator=g,
                            callback_on_step_end=self._cb(q, req.steps or 30))
            return _to_uint8(out.frames[0])

    def i2v(self, req, q):
        import torch
        from PIL import Image
        self._load_i2v()
        w, h, nf = self._snap(req)
        img = Image.open(io.BytesIO(req.init_image)).convert("RGB").resize((w, h))
        with self._lock:
            g = torch.Generator("cpu").manual_seed(req.seed or 42)
            out = self._i2v(image=img, prompt=req.prompt, height=h, width=w, num_frames=nf,
                            num_inference_steps=req.steps or 30, generator=g,
                            callback_on_step_end=self._cb(q, req.steps or 30))
            return _to_uint8(out.frames[0])

    def refine(self, req, q):
        raise RuntimeError("hunyuan backend has no refine op; use framework/i2v")


# --------------------------------------------------------------------------- #
# MLX backend (wraps mlx-video) — Mac mini; OWNER-RUN (untested in CI; no Mac here)
# --------------------------------------------------------------------------- #
class MlxBackend:
    """Wraps `mlx-video` (Blaizzy/mlx-video) via its real CLI.

    Verified against the package source (mlx_video/models/wan_2/generate.py):
        python -m mlx_video.models.wan_2.generate \
            --model-dir DIR --prompt P --output-path OUT \
            --width W --height H --num-frames N --steps S --seed SEED [--lora PATH STR]
    where --num-frames must be 4n+1. mlx-video does T2V/I2V but has no vid2vid.

    refine op on MLX (for a two-Mac proposer->refiner pipeline, ADR 0014):
      - if MLX_V2V_FLAG is set (a build that DOES have vid2vid) -> generative V2V refine.
      - otherwise -> a SPATIAL super-resolution refine (Lanczos upscale + unsharp mask).
        This is NON-generative: it raises the proposer's resolution and crispens it, but
        does not synthesize new detail. It lets the headless Mac serve a real 'refine'
        role on stock mlx-video; for true generative detail, set MLX_V2V_FLAG or route
        refine to a CUDA worker. Knobs below are env-overridable for other versions."""

    def __init__(self, model_dir: str, ops: list[str]):
        self.model_dir = model_dir
        self.ops = ops or ["framework"]
        # Verified defaults for current mlx-video; env-overridable for other versions.
        self.module = os.environ.get("MLX_T2V_MODULE", "mlx_video.models.wan_2.generate")
        self.pass_dims = os.environ.get("MLX_PASS_DIMS", "1").lower() in ("1", "true", "yes")
        self.pass_wh = os.environ.get("MLX_PASS_WH", "1").lower() in ("1", "true", "yes")
        # Optional CausVid/Self-Forcing LoRA for a faithful few-step proposer: "path,strength"
        self.lora = os.environ.get("MLX_LORA", "").strip()
        self.v2v_flag = os.environ.get("MLX_V2V_FLAG", "")  # e.g. "--image" only if your build has vid2vid
        # VAE tiling cuts decode-time Metal memory on unified-memory Macs (OOM is usually the
        # VAE decode before writeout). "aggressive" is safest for small Mac minis.
        self.tiling = os.environ.get("MLX_TILING", "aggressive")
        # Spatial-SR refine knobs (used when refine runs without vid2vid).
        self.sr_sharpen = float(os.environ.get("MLX_SR_SHARPEN", "60"))  # unsharp percent; 0 disables

    def _refine_mode(self) -> str:
        return "v2v" if self.v2v_flag else "sr"

    def health(self):
        note = "Apple Silicon (MLX)"
        if "refine" in self.ops:
            note += f" · refine={self._refine_mode()}"
        return pb.HealthReply(device="mlx", backend="mlx-video", model=self.model_dir,
                              ops=self.ops, ready=True, note=note,
                              relative_speed=float(os.environ.get("MLX_RELATIVE_SPEED", "0.12")))

    def _run(self, prompt, seed, num_frames, steps, q, extra=None, width=0, height=0):
        out = Path(tempfile.mkstemp(suffix=".mp4")[1])
        # Verified mlx-video flags: --model-dir/--prompt/--output-path/--seed.
        cmd = ["python", "-m", self.module, "--model-dir", self.model_dir,
               "--prompt", prompt, "--output-path", str(out), "--seed", str(seed)]
        if self.pass_wh:
            if width:
                cmd += ["--width", str(width)]
            if height:
                cmd += ["--height", str(height)]
        if self.pass_dims:
            if num_frames:
                cmd += ["--num-frames", str(num_frames)]
            if steps:
                cmd += ["--steps", str(steps)]
        if self.lora:
            path, _, strength = self.lora.partition(",")
            cmd += ["--lora", path.strip(), (strength.strip() or "1.0")]
        if self.tiling and self.tiling != "auto":
            cmd += ["--tiling", self.tiling]
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
        # mlx-video requires num_frames == 4n+1; snap down if the caller sent something else.
        nf = req.num_frames or 25
        if nf % 4 != 1:
            nf = (nf // 4) * 4 + 1
        return self._run(req.prompt, req.seed or 42, nf, req.steps or 6, q,
                         width=req.width or 832, height=req.height or 480)

    def refine(self, req, q):
        if "refine" not in self.ops:
            raise RuntimeError("this mlx worker advertises no 'refine' op. Launch with "
                               "--mlx-ops refine (or framework,refine) to make this Mac a refiner.")
        # A) generative V2V — only if this mlx-video build has vid2vid (MLX_V2V_FLAG set).
        if self.v2v_flag:
            inp = Path(tempfile.mkstemp(suffix=".mp4")[1]); inp.write_bytes(req.mp4)
            try:
                return self._run(req.prompt, req.seed or 7, req.num_frames or 25, req.steps or 16, q,
                                 extra=[self.v2v_flag, str(inp), "--strength", str(req.strength or 0.6)],
                                 width=req.width or 832, height=req.height or 480)
            finally:
                inp.unlink(missing_ok=True)
        # B) stock mlx-video has no vid2vid -> spatial super-resolution refine (Lanczos + unsharp).
        #    Honest: this upscales/crispens the proposer; it does not synthesize new detail.
        return self._sr_refine(req, q)

    def _sr_refine(self, req, q):
        """Non-generative refine: Lanczos-upscale the proposer to (width,height) and crispen.

        Pure PIL/numpy (no MLX/Metal), so it is cheap, OOM-safe and CI-testable. Lets the
        headless Mac serve a real 'refine' role: head proposes low-res, this worker raises it
        to the target resolution. For generative detail set MLX_V2V_FLAG / use a CUDA refiner."""
        import numpy as np
        from PIL import Image, ImageFilter
        frames = mp4_to_frames(req.mp4)
        tw, th = int(req.width or 832), int(req.height or 480)
        n = int(frames.shape[0]) or 1
        sharpen = ImageFilter.UnsharpMask(radius=2, percent=int(self.sr_sharpen), threshold=2)
        out = []
        for i in range(frames.shape[0]):
            q.put_nowait(("p", (i + 1) / n))
            im = Image.fromarray(_to_uint8(frames[i:i + 1])[0]).convert("RGB").resize((tw, th), Image.LANCZOS)
            if self.sr_sharpen > 0:
                im = im.filter(sharpen)
            out.append(np.asarray(im, np.uint8))
        return np.stack(out)


# --------------------------------------------------------------------------- #
# Test backend (NO GPU) — validates the gRPC TRANSPORT only (proto/streaming/
# routing/geometry), not the model. The model path is the cuda backend above.
# --------------------------------------------------------------------------- #
class TestBackend:
    def health(self):
        return pb.HealthReply(device=os.environ.get("TEST_DEVICE", "cpu"), backend="test-synthetic",
                              model="none", ops=["framework", "refine", "t2v", "i2v"], ready=True,
                              note="transport smoke test (no model)",
                              relative_speed=float(os.environ.get("TEST_SPEED", "1.0")))

    def framework(self, req, q):
        import numpy as np
        for i in range(4):
            q.put_nowait(("p", (i + 1) / 4)); time.sleep(0.03)
        n, h, w = req.num_frames or 8, req.height or 480, req.width or 832
        arr = np.zeros((n, h, w, 3), np.uint8); arr[..., :] = (40, 80, 160)
        # I2V continuity: mark frame 0 from the seed image so a stitch test can verify continuity.
        if getattr(req, "init_image", b""):
            arr[0, ..., 1] = int(np.frombuffer(req.init_image[:256], np.uint8).mean()) % 256
        return arr

    def refine(self, req, q):
        import numpy as np
        from PIL import Image
        for i in range(4):
            q.put_nowait(("p", (i + 1) / 4)); time.sleep(0.03)
        frames = mp4_to_frames(req.mp4)
        tw, th = int(req.width or frames.shape[2]), int(req.height or frames.shape[1])
        if (frames.shape[2], frames.shape[1]) != (tw, th):  # mimic SR: resize to target dims
            frames = np.stack([np.asarray(Image.fromarray(frames[i]).resize((tw, th)), np.uint8)
                               for i in range(frames.shape[0])])
        frames = frames.copy()
        frames[..., 0] = np.clip(frames[..., 0].astype(int) + 50, 0, 255)  # tint to show the refine ran
        return frames


# --------------------------------------------------------------------------- #
# gRPC servicer (streaming progress)
# --------------------------------------------------------------------------- #
class VideoWorkerServicer(pb_grpc.VideoWorkerServicer):
    def __init__(self, backend, ops_override=None):
        self.backend = backend
        self.ops_override = ops_override  # restrict advertised ops (e.g. CUDA box = refine-only)

    def Health(self, request, context):
        h = self.backend.health()
        if self.ops_override:
            allowed = [o for o in h.ops if o in self.ops_override]
            h2 = pb.HealthReply(device=h.device, backend=h.backend, model=h.model,
                                ops=allowed or list(self.ops_override), ready=h.ready,
                                note=h.note, relative_speed=h.relative_speed)
            return h2
        return h

    def _stream(self, op_fn, request, context):
        q: queue.Queue = queue.Queue(maxsize=256)
        result = {}

        def run():
            try:
                t = time.time()
                result["frames"] = op_fn(request, q)
                result["t"] = time.time() - t
            except Exception as exc:  # noqa: BLE001
                import traceback as _tb
                result["err"] = f"{type(exc).__name__}: {exc}"
                print(f"[grpc_worker] op error: {result['err']}", flush=True)
                _tb.print_exc()
            finally:
                q.put(("done", None))

        threading.Thread(target=run, daemon=True).start()
        yield pb.Progress(pct=0.0, stage="load")
        # Heartbeat: MLX model load (umt5-xxl T5) is a long SILENT phase; without periodic
        # messages the idle HTTP/2 stream gets dropped over the SOCKS5/tailnet tunnel. Emit a
        # keepalive Progress every few seconds while the worker thread is still running.
        last = 0.0
        while True:
            try:
                kind, val = q.get(timeout=5)
            except queue.Empty:
                yield pb.Progress(pct=last, stage="load")  # keepalive (keeps bytes flowing)
                continue
            if kind == "done":
                break
            last = float(val)
            yield pb.Progress(pct=last, stage="denoise")
        if "err" in result:
            context.abort(grpc.StatusCode.INTERNAL, result["err"])
            return
        nfr = len(result["frames"]) if result.get("frames") is not None else 0
        print(f"[grpc_worker] denoise done ({nfr} frames); encoding mp4...", flush=True)
        _mp4 = frames_to_mp4(result["frames"])
        print(f"[grpc_worker] encoded mp4 {len(_mp4)} bytes; sending final message...", flush=True)
        yield pb.Progress(pct=1.0, stage="done", done=True, mp4=_mp4, gen_seconds=float(result["t"]))
        print("[grpc_worker] final message sent OK", flush=True)

    def GenerateFramework(self, request, context):
        yield from self._stream(self.backend.framework, request, context)

    def RefineTile(self, request, context):
        yield from self._stream(self.backend.refine, request, context)


# --------------------------------------------------------------------------- #
# Subprocess backend — runs the heavy CUDA generation in a CLEAN child process.
#
# Why: heavy video VAE encode/decode (Hunyuan, WAN v2v) HANGS when run inside the gRPC servicer's
# worker thread (denoise completes, then stuck at GPU 0%). The same op runs fine in a standalone
# process's main thread. So we run the real backend in a spawned child process whose MAIN thread does
# all CUDA work; the gRPC side only does IPC + a small file read (no CUDA on the gRPC threads).
# Progress is streamed back per step; the result frames are handed over via a temp .npy file.
# --------------------------------------------------------------------------- #
_REQ_TYPES = {"framework": pb.FrameworkRequest, "refine": pb.RefineRequest}


def _make_real_backend(kind, ops):
    if kind == "cuda":
        return CudaBackend()
    if kind == "hunyuan":
        return HunyuanBackend(ops)
    raise ValueError(f"subproc unsupported backend: {kind}")


def _subproc_main(cmd_q, res_q, kind, ops):
    """Child process (spawn): load the model and run ops in the MAIN thread. CUDA lives only here."""
    import os
    import numpy as np
    import tempfile
    import traceback
    try:
        backend = _make_real_backend(kind, ops)
        if kind == "cuda":
            backend.load()
        elif kind == "hunyuan":
            backend._load_t2v()
        res_q.put(("ready", None))
    except Exception as exc:  # noqa: BLE001
        res_q.put(("fatal", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        return

    class _PQ:  # forwards backend progress (q.put_nowait(("p", pct))) to the parent
        def put_nowait(self, item):
            try:
                res_q.put(("p", float(item[1])))
            except Exception:  # noqa: BLE001
                pass
        put = put_nowait

    tmpdir = os.environ.get("DISTWAN_TMP", "/workspace")
    while True:
        job = cmd_q.get()
        if job is None:
            break
        op_name, req_bytes, req_type = job
        try:
            req = _REQ_TYPES[req_type]()
            req.ParseFromString(req_bytes)
            frames = getattr(backend, op_name)(req, _PQ())
            fd, path = tempfile.mkstemp(suffix=".npy", dir=tmpdir)
            os.close(fd)
            np.save(path, frames)
            res_q.put(("done", path))
        except Exception as exc:  # noqa: BLE001
            res_q.put(("err", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


class SubprocBackend:
    """Proxy that runs `kind` in a persistent spawned child (CUDA off the gRPC threads)."""

    def __init__(self, kind, ops):
        import multiprocessing as mp
        self.kind = kind
        self.ops = ops or ["framework"]
        self._lock = threading.Lock()
        ctx = mp.get_context("spawn")
        self.cmd_q = ctx.Queue()
        self.res_q = ctx.Queue()
        self.proc = ctx.Process(target=_subproc_main, args=(self.cmd_q, self.res_q, kind, self.ops),
                                daemon=True)
        self.proc.start()
        print(f"[grpc_worker] subproc backend '{kind}' child pid={self.proc.pid} loading model...",
              flush=True)

    def health(self):
        spd = float(os.environ.get("SUBPROC_RELATIVE_SPEED", "0.8" if self.kind == "hunyuan" else "1.0"))
        return pb.HealthReply(device="cuda", backend=f"{self.kind}-subproc", model=self.kind,
                              ops=self.ops, ready=self.proc.is_alive(),
                              note=f"subprocess worker pid={self.proc.pid}", relative_speed=spd)

    def _run(self, op_name, req, req_type, q):
        import numpy as np
        with self._lock:
            self.cmd_q.put((op_name, req.SerializeToString(), req_type))
            while True:
                kind, val = self.res_q.get()
                if kind == "p":
                    try:
                        q.put_nowait(("p", val))
                    except Exception:  # noqa: BLE001
                        pass
                elif kind == "ready":
                    continue  # model finished loading; keep waiting for this job's result
                elif kind == "done":
                    frames = np.load(val)
                    try:
                        os.unlink(val)
                    except OSError:
                        pass
                    return frames
                elif kind in ("err", "fatal"):
                    raise RuntimeError(val)

    def framework(self, req, q):
        return self._run("framework", req, "framework", q)

    def refine(self, req, q):
        return self._run("refine", req, "refine", q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cuda", "mlx", "test", "hunyuan"], default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--mlx-model-dir", default=os.environ.get("MLX_MODEL_DIR", ""))
    ap.add_argument("--mlx-ops", default=os.environ.get("MLX_OPS", "framework"))
    ap.add_argument("--ops", default=os.environ.get("WORKER_OPS", ""),
                    help="restrict advertised ops, e.g. 'refine' to make a CUDA box refine-only")
    ap.add_argument("--subproc", action="store_true",
                    help="run the (cuda/hunyuan) backend in a spawned child process so heavy CUDA "
                         "decode runs off the gRPC threads (fixes the in-thread VAE-decode hang).")
    args = ap.parse_args()

    if args.subproc and args.backend in ("cuda", "hunyuan"):
        _default = "framework,refine,i2v" if args.backend == "cuda" else "framework,i2v"
        _ops = [o.strip() for o in (args.ops or _default).split(",") if o.strip()]
        backend = SubprocBackend(args.backend, _ops)
    elif args.backend == "cuda":
        backend = CudaBackend()
        if args.preload:
            backend.load()
    elif args.backend == "test":
        backend = TestBackend()
    elif args.backend == "hunyuan":
        backend = HunyuanBackend([o.strip() for o in (args.ops or "framework,i2v").split(",") if o.strip()])
    else:
        backend = MlxBackend(args.mlx_model_dir, [o.strip() for o in args.mlx_ops.split(",") if o.strip()])

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8),
                         options=[("grpc.max_send_message_length", 256 * 1024 * 1024),
                                  ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                                  ("grpc.keepalive_time_ms", 15000),
                                  ("grpc.keepalive_timeout_ms", 30000),
                                  ("grpc.keepalive_permit_without_calls", 1),
                                  ("grpc.http2.max_pings_without_data", 0),
                                  ("grpc.http2.min_ping_interval_without_data_ms", 5000)])
    ops_override = [o.strip() for o in args.ops.split(",") if o.strip()] or None
    pb_grpc.add_VideoWorkerServicer_to_server(VideoWorkerServicer(backend, ops_override), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    print(f"[grpc_worker] {args.backend} backend listening on {args.host}:{args.port}"
          + (f" (ops={ops_override})" if ops_override else ""), flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
