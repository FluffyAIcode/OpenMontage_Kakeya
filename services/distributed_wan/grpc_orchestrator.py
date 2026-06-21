"""Distributed WAN gRPC orchestrator (ADR 0010) — the cloud-agent control plane.

Talks to N `VideoWorker` gRPC backends (CUDA on vast, MLX on a Mac mini — "another
GPU"). Capability- and speed-aware:

  1. Health() every worker -> ops + relative_speed.
  2. framework on the fastest framework-capable worker (server-streamed progress).
  3. upscale -> canvas -> native tile crops.
  4. refine tiles, assigned SPEED-WEIGHTED across refine-capable workers, dispatched
     concurrently (each a streamed RefineTile call).
  5. weight-map (f_theta) merge -> final video.

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
              ("grpc.max_receive_message_length", 256 * 1024 * 1024)]


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


def _ramp(n):
    return np.minimum(np.arange(n), np.arange(n)[::-1]) + 1.0


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
    ap.add_argument("--out", default="distributed_wan_grpc.mp4")
    args = ap.parse_args()

    addrs = [a for a in os.environ.get("WAN_WORKERS", "").split(",") if a.strip()]
    if not addrs:
        raise SystemExit("set WAN_WORKERS=host1:50051,host2:50051")
    workers = [Worker(a) for a in addrs]
    fw_workers = [w for w in workers if ("framework" in w.ops or "t2v" in w.ops)]
    refine_workers = [w for w in workers if "refine" in w.ops]
    if not fw_workers:
        raise SystemExit("no framework-capable worker")
    if not refine_workers:
        raise SystemExit("no refine-capable worker (e.g. a CUDA worker)")

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

    # 4) SPEED-WEIGHTED tile assignment across refine workers
    loads = {id(w): 0 for w in refine_workers}
    assign = {}
    for t in tiles:
        w = min(refine_workers, key=lambda w: (loads[id(w)] + 1) / max(w.speed, 1e-3))
        assign[t] = w; loads[id(w)] += 1
    print("[orch] tile->worker: " + ", ".join(f"{t}->{assign[t].device}" for t in tiles), flush=True)

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
    iio.mimwrite(args.out, [final[i] for i in range(T)], format="mp4", fps=12, codec="libx264")
    Image.fromarray(final[T // 2]).save(args.out.replace(".mp4", "_mid.png"))
    import json
    print("ORCH_DONE " + json.dumps({"workers": [w.addr for w in workers], "refine_wall_s": round(wall, 2),
                                     "out": args.out, "canvas_px": [CH, CW]}), flush=True)


if __name__ == "__main__":
    main()
