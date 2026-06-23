"""Distributed WAN gRPC orchestrator (ADR 0010) — the cloud-agent control plane.

Talks to N `VideoWorker` gRPC backends (CUDA on vast, MLX on a Mac mini — "another
GPU"). Capability- and speed-aware:

  1. Health() every worker -> ops + relative_speed.
  2. framework on the fastest framework-capable worker (server-streamed progress).
  3. then one of:
     - DIRECT (--no-refine / no refine worker): write the proposer as-is (Mac-only draft).
     - PIPELINE (--single-refine / MLX refiner): one full-frame refine on a single refiner
       at the target out resolution (two-Mac head=proposer, headless=refine).
     - TILED (CUDA refiner): upscale -> canvas -> native tile crops -> refine tiles
       SPEED-WEIGHTED across refine workers concurrently -> f_theta weight-map merge.

Optional Mac text plane (Kakeya MLX LLM) stays its own HTTP/OpenAI shim via
KAKEYA_ENDPOINT (skip-not-fake) — unrelated to the gRPC worker contract.

Run:
    pip install grpcio numpy pillow imageio imageio-ffmpeg
    WAN_WORKERS=vast-host:50051,mac-tailnet:50051 \
    python grpc_orchestrator.py --prompt "..." --out final.mp4
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import imageio.v2 as iio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_worker_pb2 as pb
import video_worker_pb2_grpc as pb_grpc

WT, HT, OV = 832, 480, 192
NX, NY = 2, 2
X_OFF = [i * (WT - OV) for i in range(NX)]
Y_OFF = [j * (HT - OV) for j in range(NY)]
CW, CH = X_OFF[-1] + WT, Y_OFF[-1] + HT
_GRPC_OPTS = [("grpc.max_send_message_length", 256 * 1024 * 1024),
              ("grpc.max_receive_message_length", 256 * 1024 * 1024),
              ("grpc.keepalive_time_ms", 15000),
              ("grpc.keepalive_timeout_ms", 30000),
              ("grpc.keepalive_permit_without_calls", 1),
              ("grpc.http2.max_pings_without_data", 0)]


class Worker:
    def __init__(self, addr):
        self.addr = addr
        self.ch = grpc.insecure_channel(addr, options=_GRPC_OPTS)
        self.stub = pb_grpc.VideoWorkerStub(self.ch)
        h = self.stub.Health(pb.HealthRequest(), timeout=20)
        self.ops = list(h.ops); self.speed = h.relative_speed or 1.0
        self.device = h.device; self.backend = h.backend
        print(f"[orch] worker {addr}: {self.backend}/{self.device} ops={self.ops} speed={self.speed}", flush=True)


def _mp4_to_frames(b):
    return np.stack([f for f in iio.get_reader(io.BytesIO(b), format="mp4")])


def _frames_to_mp4_bytes(frames, fps=12):
    buf = io.BytesIO()
    iio.mimwrite(buf, [frames[i] for i in range(frames.shape[0])], format="mp4", fps=fps, codec="libx264")
    return buf.getvalue()


def _interpolate(frames, factor):
    """Temporal interpolation x`factor` via linear blend between consecutive frames.

    Dependency-free smoothness/fps baseline (ADR 0015 Phase 1): T frames -> (T-1)*factor + 1."""
    if factor <= 1 or frames.shape[0] < 2:
        return frames
    out = []
    for i in range(frames.shape[0] - 1):
        a = frames[i].astype(np.float32); b = frames[i + 1].astype(np.float32)
        for k in range(factor):
            t = k / factor
            out.append((a * (1.0 - t) + b * t).round().astype(np.uint8))
    out.append(frames[-1])
    return np.stack(out)


def _interpolate_mci(frames, factor, base_fps=12):
    """Motion-compensated (optical-flow) interpolation via ffmpeg `minterpolate` — the RIFE-class
    smoothness upgrade over linear blending (ADR 0015 Phase 2c). Falls back to linear if ffmpeg/
    minterpolate is unavailable. Returns ~T*factor frames."""
    if factor <= 1 or frames.shape[0] < 2:
        return frames
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        src = Path(__import__("tempfile").mkstemp(suffix=".mp4")[1])
        dst = Path(__import__("tempfile").mkstemp(suffix=".mp4")[1])
        iio.mimwrite(src, [frames[i] for i in range(frames.shape[0])], format="mp4", fps=base_fps,
                     codec="libx264")
        target = base_fps * factor
        flt = (f"minterpolate=fps={target}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
        import subprocess
        r = subprocess.run([exe, "-y", "-i", str(src), "-filter:v", flt, "-c:v", "libx264", str(dst)],
                           capture_output=True)
        if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError("minterpolate failed")
        out = _mp4_to_frames(dst.read_bytes())
        src.unlink(missing_ok=True); dst.unlink(missing_ok=True)
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[orch] mci interpolation unavailable ({exc}); linear fallback", flush=True)
        return _interpolate(frames, factor)


def _encode(frames, out_path, fps=12, interpolate=1, method="linear"):
    """Apply optional interpolation (linear or mci/optical-flow), write the mp4 at `fps` + a
    mid-frame png. Returns the final frame count."""
    if interpolate > 1:
        f = _interpolate_mci(frames, interpolate, base_fps=fps) if method == "mci" else _interpolate(frames, interpolate)
    else:
        f = frames
    iio.mimwrite(out_path, [f[i] for i in range(f.shape[0])], format="mp4", fps=fps, codec="libx264")
    Image.fromarray(f[f.shape[0] // 2]).save(out_path.replace(".mp4", "_mid.png"))
    return int(f.shape[0])


def _ramp(n):
    return np.minimum(np.arange(n), np.arange(n)[::-1]) + 1.0


def _png_bytes(frame):
    buf = io.BytesIO(); Image.fromarray(frame).save(buf, format="PNG"); return buf.getvalue()


def _stitch(clips, overlap):
    """Concatenate chunks with an `overlap`-frame crossfade so chunk boundaries don't jump-cut.

    Total frames = sum(len(clip)) - (len(clips)-1)*overlap (ADR 0015 Phase 2 long-form)."""
    if len(clips) == 1:
        return clips[0]
    out = clips[0]
    for c in clips[1:]:
        ov = min(overlap, out.shape[0], c.shape[0])
        if ov > 0:
            blended = []
            for k in range(ov):
                t = (k + 1) / (ov + 1)
                a = out[out.shape[0] - ov + k].astype(np.float32); b = c[k].astype(np.float32)
                blended.append((a * (1.0 - t) + b * t).round().astype(np.uint8))
            out = np.concatenate([out[:out.shape[0] - ov], np.stack(blended), c[ov:]], axis=0)
        else:
            out = np.concatenate([out, c], axis=0)
    return out


def _consume(stream, label):
    mp4, gen_s = None, 0.0
    for p in stream:
        if p.done:
            mp4, gen_s = p.mp4, p.gen_seconds
        elif p.stage == "denoise":
            print(f"[orch]   {label}: {p.pct * 100:4.0f}%", flush=True)
    return mp4, gen_s


def mac_expand_prompts(endpoint, base_prompt, n):
    import json, urllib.request
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    body = {"model": os.environ.get("KAKEYA_MODEL", "kakeya-local"),
            "messages": [{"role": "user", "content":
                          f"Rewrite into {n} short prompt variations, one per line: {base_prompt}"}],
            "max_tokens": 200, "temperature": 0.7}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    lines = [ln.strip(" -*0123456789.") for ln in out["choices"][0]["message"]["content"].splitlines() if ln.strip()]
    lines = [ln for ln in lines if ln][:n]
    while len(lines) < n:
        lines.append(base_prompt)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--frames", type=int, default=25)
    # Proposer (framework) is LOW-res by design — keeps a memory-bounded Mac/MLX worker from
    # OOMing. It is temporally + spatially resampled up to the canvas before tiled refine.
    ap.add_argument("--fw-width", type=int, default=480)
    ap.add_argument("--fw-height", type=int, default=256)
    ap.add_argument("--fw-frames", type=int, default=13)  # must be 4n+1; worker snaps if not
    ap.add_argument("--proposer-steps", type=int, default=6)
    ap.add_argument("--refine-steps", type=int, default=16)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-refine", action="store_true",
                    help="single-worker DIRECT T2V (e.g. Mac-only): one generation at fw dims, "
                         "no tiled CUDA refine. Auto-enabled when no refine-capable worker exists.")
    # Single-pass pipeline (proposer -> one refiner over the WHOLE clip). The two-Mac topology:
    # head = framework/proposer (low-res), headless = refine (MLX SR upscale, or V2V if available).
    ap.add_argument("--single-refine", action="store_true",
                    help="one full-frame refine on a single refiner (e.g. headless Mac), instead "
                         "of 2x2 tiled CUDA V2V. Auto-enabled when the chosen refiner is MLX.")
    ap.add_argument("--out-width", type=int, default=WT, help="final width for --single-refine output")
    ap.add_argument("--out-height", type=int, default=HT, help="final height for --single-refine output")
    ap.add_argument("--refine-spread", choices=["weighted", "roundrobin"], default="weighted",
                    help="tiled refine assignment: 'weighted' (speed-weighted; a fast refiner may "
                         "take all tiles) or 'roundrobin' (every refiner gets a share, incl. slow MLX).")
    # --- quality + duration knobs (ADR 0015) ---
    ap.add_argument("--refine-mode", choices=["auto", "direct", "single", "tiled"], default="auto",
                    help="refine topology: auto (mlx->single, cuda->tiled), direct (proposer only), "
                         "single (one seam-free full-frame refine on the best refiner — the quality "
                         "path on a CUDA box), tiled (2x2 f_theta merge across refiners).")
    ap.add_argument("--fps", type=int, default=12, help="output frame rate")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="target duration; if >0, frames = round(seconds*fps) (Phase-1 temporal "
                         "resample of the proposer; true long-form chunking is Phase 2).")
    ap.add_argument("--interpolate", type=int, default=1,
                    help="temporal interpolation factor (x2/x4) for smoother motion / higher fps.")
    ap.add_argument("--interp-method", choices=["linear", "mci"], default="linear",
                    help="interpolation: 'linear' (blend) or 'mci' (ffmpeg motion-compensated / "
                         "optical-flow — RIFE-class smoothness; falls back to linear if unavailable).")
    # Long-form (ADR 0015 Phase 2): chunked autoregressive generation with I2V continuity.
    ap.add_argument("--longform", action="store_true",
                    help="force the I2V generative path even for a single chunk (chunks=1) — the "
                         "true-720p hero path on an i2v worker. >1 chunks is multi-chunk continuity.")
    ap.add_argument("--chunks", type=int, default=1,
                    help=">1 enables long-form: N chunks generated autoregressively (chunk N+1 seeded "
                         "by the last frame of chunk N via I2V) and crossfade-stitched.")
    ap.add_argument("--chunk-frames", type=int, default=25, help="frames generated per chunk")
    ap.add_argument("--chunk-overlap", type=int, default=4, help="crossfade frames between chunks")
    ap.add_argument("--out", default="distributed_wan_grpc.mp4")
    args = ap.parse_args()
    if args.seconds and args.seconds > 0:
        args.frames = max(2, round(args.seconds * args.fps))

    addrs = [a for a in os.environ.get("WAN_WORKERS", "").split(",") if a.strip()]
    if not addrs:
        raise SystemExit("set WAN_WORKERS=host1:50051,host2:50051")
    workers = [Worker(a) for a in addrs]
    fw_workers = [w for w in workers if ("framework" in w.ops or "t2v" in w.ops)]
    refine_workers = [w for w in workers if "refine" in w.ops]
    if not fw_workers:
        raise SystemExit("no framework-capable worker")

    # LONG-FORM (ADR 0015 Phase 2): chunked autoregressive generation. chunk0 = T2V; chunk N+1 is
    # seeded by the last frame of chunk N via I2V (continuity), then crossfade-stitched. Runs on an
    # i2v-capable worker (the CUDA box with CUDA_I2V_MODEL); falls back to independent T2V chunks
    # (no continuity) with a warning if no i2v worker is present.
    if args.longform or args.chunks > 1:
        args.chunks = max(1, args.chunks)
        i2v_workers = [w for w in workers if "i2v" in w.ops]
        cont = bool(i2v_workers)
        ow, oh = args.out_width, args.out_height

        def _to_out(frames):  # ensure every chunk is the same display resolution for clean stitch
            if (int(frames.shape[2]), int(frames.shape[1])) == (ow, oh):
                return frames
            return np.stack([np.asarray(Image.fromarray(frames[i]).resize((ow, oh), Image.LANCZOS))
                             for i in range(frames.shape[0])])

        clips, total_gs = [], 0.0
        if cont:
            gen = max(i2v_workers, key=lambda w: w.speed)
            sp = max(fw_workers, key=lambda w: w.speed)
            print(f"[orch] LONG-FORM (I2V continuity) seed={sp.addr}({sp.backend}) gen={gen.addr}"
                  f"({gen.backend}) — {args.chunks}x{args.chunk_frames}f @ {ow}x{oh}, overlap={args.chunk_overlap}",
                  flush=True)
            # seed frame: a quick T2V proposer clip, upscaled to the display resolution
            smp4, sgs = _consume(sp.stub.GenerateFramework(pb.FrameworkRequest(
                prompt=args.prompt, width=args.fw_width, height=args.fw_height,
                num_frames=args.fw_frames, steps=args.proposer_steps, seed=args.seed)), "seed")
            sf = _mp4_to_frames(smp4); total_gs += sgs
            init_png = _png_bytes(np.asarray(Image.fromarray(sf[-1]).resize((ow, oh), Image.BICUBIC)))
            for c in range(args.chunks):
                mp4, gs = _consume(gen.stub.GenerateFramework(pb.FrameworkRequest(
                    prompt=args.prompt, width=ow, height=oh, num_frames=args.chunk_frames,
                    steps=args.refine_steps, seed=args.seed + 1 + c, init_image=init_png)), f"chunk{c}")
                frames = _to_out(_mp4_to_frames(mp4))
                clips.append(frames); total_gs += gs; init_png = _png_bytes(frames[-1])
                print(f"[orch]   chunk {c} {frames.shape} in {gs:.1f}s", flush=True)
        else:
            gen = max(fw_workers, key=lambda w: w.speed)
            print(f"[orch] LONG-FORM (continuity OFF — no i2v worker) on {gen.addr}({gen.backend}) — "
                  f"{args.chunks}x{args.chunk_frames}f @ {ow}x{oh}, overlap={args.chunk_overlap}", flush=True)
            for c in range(args.chunks):
                mp4, gs = _consume(gen.stub.GenerateFramework(pb.FrameworkRequest(
                    prompt=args.prompt, width=ow, height=oh, num_frames=args.chunk_frames,
                    steps=args.refine_steps, seed=args.seed + c)), f"chunk{c}")
                clips.append(_to_out(_mp4_to_frames(mp4))); total_gs += gs
                print(f"[orch]   chunk {c} in {gs:.1f}s", flush=True)
        stitched = _stitch(clips, args.chunk_overlap)
        nout = _encode(stitched, args.out, args.fps, args.interpolate, args.interp_method)
        import json
        print("ORCH_DONE " + json.dumps({"workers": [w.addr for w in workers], "mode": "longform",
                                         "generator": gen.addr, "continuity": "i2v" if cont else "off",
                                         "chunks": args.chunks, "out": args.out, "gen_seconds": round(total_gs, 2),
                                         "px": [int(stitched.shape[1]), int(stitched.shape[2])],
                                         "fps": args.fps, "interpolate": args.interpolate,
                                         "seconds": round(nout / max(args.fps, 1), 2), "frames": nout}), flush=True)
        return

    # DIRECT (no-refine) mode: one T2V generation on the framework worker, no tiled refine.
    # This is the Mac-only / single-GPU path (mlx-video has no vid2vid). Auto-enabled when no
    # refine worker is present, so the service still produces video with just the Mac.
    if args.no_refine or args.refine_mode == "direct" or not refine_workers:
        fw = max(fw_workers, key=lambda w: w.speed)
        nf = args.fw_frames
        print(f"[orch] DIRECT (no-refine) on {fw.addr} ({fw.backend}) "
              f"@ {args.fw_width}x{args.fw_height}x{nf}", flush=True)
        mp4, gs = _consume(fw.stub.GenerateFramework(pb.FrameworkRequest(
            prompt=args.prompt, width=args.fw_width, height=args.fw_height,
            num_frames=nf, steps=args.proposer_steps, seed=args.seed)), "generate")
        frames = _mp4_to_frames(mp4)
        nout = _encode(frames, args.out, args.fps, args.interpolate, args.interp_method)
        import json
        print("ORCH_DONE " + json.dumps({"workers": [w.addr for w in workers], "mode": "direct",
                                         "out": args.out, "gen_seconds": round(gs, 2),
                                         "px": [args.fw_height, args.fw_width], "fps": args.fps,
                                         "interpolate": args.interpolate, "frames": nout}), flush=True)
        return

    # SINGLE-PASS pipeline: proposer (low-res, head) -> one refiner over the WHOLE clip
    # (headless Mac). mlx-video has no tiled V2V, so we send the whole resampled proposer to
    # the refiner at the target output resolution instead of the 2x2 CUDA tile flow. The MLX
    # refiner does a spatial-SR upscale (or generative V2V if its build supports it). Auto-on
    # when the chosen refiner is an MLX worker; --single-refine forces it for any backend.
    refiner = max(refine_workers, key=lambda w: w.speed)
    _use_single = (args.refine_mode == "single" or args.single_refine
                   or (args.refine_mode == "auto" and refiner.backend.startswith("mlx")))
    if _use_single:
        fw = max(fw_workers, key=lambda w: w.speed)
        if fw.addr == refiner.addr and len(workers) > 1:
            # prefer a DISTINCT proposer so both Macs are used (head proposes, headless refines)
            others = [w for w in fw_workers if w.addr != refiner.addr]
            if others:
                fw = max(others, key=lambda w: w.speed)
        print(f"[orch] PIPELINE proposer={fw.addr}({fw.backend})@{args.fw_width}x{args.fw_height}x{args.fw_frames}"
              f" -> refiner={refiner.addr}({refiner.backend})@{args.out_width}x{args.out_height}", flush=True)
        mp4, gs = _consume(fw.stub.GenerateFramework(pb.FrameworkRequest(
            prompt=args.prompt, width=args.fw_width, height=args.fw_height,
            num_frames=args.fw_frames, steps=args.proposer_steps, seed=args.seed)), "framework")
        framework = _mp4_to_frames(mp4)
        F = framework.shape[0]
        t_idx = (np.round(np.linspace(0, F - 1, args.frames)).astype(int)
                 if F != args.frames else np.arange(args.frames))
        # Upscale the proposer to the TARGET resolution before refine. The CUDA V2V pipeline keeps
        # its INPUT resolution, so to get high-res generative detail the input must already be at
        # out_w x out_h (MLX SR also honors this). This is what makes 'high'/single truly hi-res.
        proposer = np.stack([np.asarray(Image.fromarray(framework[t_idx[i]]).resize(
            (args.out_width, args.out_height), Image.BICUBIC)) for i in range(args.frames)])
        mp4b, rg = _consume(refiner.stub.RefineTile(pb.RefineRequest(
            prompt=args.prompt, mp4=_frames_to_mp4_bytes(proposer),
            width=args.out_width, height=args.out_height, num_frames=args.frames,
            steps=args.refine_steps, strength=args.strength, seed=args.seed + 10)), "refine")
        final = _mp4_to_frames(mp4b)
        # Generative refiners (WAN 1.3B V2V) output at the model's NATIVE resolution (~832x480)
        # regardless of input size, so SR-upscale the generative result to the display target.
        # (MLX SR already returns out dims -> this is a no-op there.) Honest: generative detail at
        # native res + spatial upscale to out_w x out_h.
        if (int(final.shape[2]), int(final.shape[1])) != (args.out_width, args.out_height):
            final = np.stack([np.asarray(Image.fromarray(final[i]).resize(
                (args.out_width, args.out_height), Image.LANCZOS)) for i in range(final.shape[0])])
        px = [int(final.shape[1]), int(final.shape[2])]
        nout = _encode(final, args.out, args.fps, args.interpolate, args.interp_method)
        import json
        print("ORCH_DONE " + json.dumps({"workers": [w.addr for w in workers], "mode": "pipeline",
                                         "proposer": fw.addr, "refiner": refiner.addr, "out": args.out,
                                         "proposer_seconds": round(gs, 2), "refine_seconds": round(rg, 2),
                                         "px": px, "fps": args.fps, "interpolate": args.interpolate,
                                         "frames": nout}), flush=True)
        return

    tiles = [(jy, jx) for jy in range(NY) for jx in range(NX)]
    tile_prompts = {t: args.prompt for t in tiles}
    mac = os.environ.get("KAKEYA_ENDPOINT", "").strip()
    if mac:
        try:
            v = mac_expand_prompts(mac, args.prompt, len(tiles))
            tile_prompts = {t: v[i] for i, t in enumerate(tiles)}
            print("[orch] Mac MLX text plane specialized per-tile prompts", flush=True)
        except Exception as exc:
            print(f"[orch] Mac text plane unavailable ({exc}); base prompt (not faked)", flush=True)

    # 2) LOW-res framework on the FASTEST framework-capable worker (the MLX proposer)
    fw = max(fw_workers, key=lambda w: w.speed)
    print(f"[orch] framework on {fw.addr} ({fw.backend}) @ {args.fw_width}x{args.fw_height}x{args.fw_frames}...", flush=True)
    mp4, gs = _consume(fw.stub.GenerateFramework(pb.FrameworkRequest(
        prompt=args.prompt, width=args.fw_width, height=args.fw_height,
        num_frames=args.fw_frames, steps=args.proposer_steps, seed=args.seed)), "framework")
    framework = _mp4_to_frames(mp4)
    print(f"[orch] framework {framework.shape} in {gs:.1f}s", flush=True)

    # 3) temporal resample proposer -> args.frames, then spatial upscale -> canvas
    F = framework.shape[0]
    t_idx = np.round(np.linspace(0, F - 1, args.frames)).astype(int) if F != args.frames else np.arange(args.frames)
    canvas = np.stack([np.asarray(Image.fromarray(framework[t_idx[i]]).resize((CW, CH), Image.BICUBIC))
                       for i in range(args.frames)])

    # 4) tile assignment across refine workers
    assign = {}
    if args.refine_spread == "roundrobin":
        # every refiner gets a share (head=proposer, headless+vast=refiners): deal tiles in turn,
        # fastest first, so a slow MLX Mac still contributes instead of being optimized out.
        order = sorted(refine_workers, key=lambda w: -w.speed)
        for i, t in enumerate(tiles):
            assign[t] = order[i % len(order)]
    else:
        loads = {id(w): 0 for w in refine_workers}
        for t in tiles:
            w = min(refine_workers, key=lambda w: (loads[id(w)] + 1) / max(w.speed, 1e-3))
            assign[t] = w; loads[id(w)] += 1
    print("[orch] tile->worker: " + ", ".join(f"{t}->{assign[t].addr}({assign[t].backend})"
                                              for t in tiles), flush=True)

    def do_refine(t):
        jy, jx = t; oy, ox = Y_OFF[jy], X_OFF[jx]
        crop = canvas[:, oy:oy + HT, ox:ox + WT, :]
        w = assign[t]
        mp4b, g = _consume(w.stub.RefineTile(pb.RefineRequest(
            prompt=tile_prompts[t], mp4=_frames_to_mp4_bytes(crop), width=WT, height=HT,
            num_frames=args.frames, steps=args.refine_steps, strength=args.strength,
            seed=args.seed + 10 + jy * 2 + jx)), f"tile{t}@{w.device}")
        return t, _mp4_to_frames(mp4b), w.addr, g

    print(f"[orch] refining {len(tiles)} tiles across {len(refine_workers)} worker(s)...", flush=True)
    t0 = time.time(); results = {}
    with ThreadPoolExecutor(max_workers=len(tiles)) as pool:
        for t, frames, addr, g in pool.map(do_refine, tiles):
            results[t] = frames
            print(f"[orch]   tile {t} <- {addr} ({g:.1f}s)", flush=True)
    wall = time.time() - t0

    # 5) f_theta weight-map merge
    T = args.frames
    acc = np.zeros((T, CH, CW, 3), np.float64); wsum = np.zeros((T, CH, CW, 1), np.float64)
    wmap = np.outer(_ramp(HT), _ramp(WT))[None, :, :, None]
    for (jy, jx), frames in results.items():
        oy, ox = Y_OFF[jy], X_OFF[jx]
        acc[:, oy:oy + HT, ox:ox + WT, :] += frames.astype(np.float64) * wmap
        wsum[:, oy:oy + HT, ox:ox + WT, :] += wmap
    final = (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)
    nout = _encode(final, args.out, args.fps, args.interpolate, args.interp_method)
    import json
    print("ORCH_DONE " + json.dumps({"workers": [w.addr for w in workers], "mode": "tiled",
                                     "refine_wall_s": round(wall, 2), "out": args.out,
                                     "canvas_px": [CH, CW], "fps": args.fps,
                                     "interpolate": args.interpolate, "frames": nout}), flush=True)


if __name__ == "__main__":
    main()
