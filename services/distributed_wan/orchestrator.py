r"""Distributed WAN orchestrator — runs on the cloud agent, glues Mac + vast.

Region-aware, latency-tolerant heterogeneous pipeline (only text + base64-mp4 cross
the wire; NO per-step tensors — see ADR 0006):

  1. (Mac mini, MLX, optional)  expand the prompt into per-tile prompts via a Kakeya
     MLX text server at KAKEYA_ENDPOINT (the Mac's REAL capability). If unreachable,
     SKIP (logged) and use the base prompt — never faked as if the Mac ran.
  2. (vast CUDA worker[0])      distilled CausVid proposer -> low-res framework.
  3. (orchestrator)             upscale framework -> canvas, crop into native tiles.
  4. (vast CUDA workers, N)     full-WAN vid2vid refine of each tile crop, dispatched
     round-robin and CONCURRENTLY -> true data-plane parallelism across co-located
     CUDA GPUs. (The framework anchors overlaps, so independent tiles stay seamless —
     ADR 0004 capstone.)
  5. (orchestrator)             weight-map merge -> final video.

WAN runs ONLY on the CUDA workers; the Mac contributes text only (it cannot run WAN).

Run:
    pip install requests imageio imageio-ffmpeg numpy pillow
    WAN_WORKERS=http://w1:9000,http://w2:9000 \
    KAKEYA_ENDPOINT=http://mac-tailnet:8000 \   # optional (Mac MLX text)
    python orchestrator.py --prompt "..." --out final.mp4
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import imageio.v2 as iio
import numpy as np
from PIL import Image

# tiling (matches ADR 0004 capstone): 2x2 native 832x480, 192px overlap -> 1472x768
WT, HT, OV = 832, 480, 192
NX, NY = 2, 2
x_off = [i * (WT - OV) for i in range(NX)]      # [0, 640]
y_off = [j * (HT - OV) for j in range(NY)]      # [0, 288]
CW, CH = x_off[-1] + WT, y_off[-1] + HT          # 1472 x 768


def _post(url: str, payload: dict, timeout: int = 1200) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _frames_to_b64(frames: np.ndarray, fps: int = 12) -> str:
    buf = io.BytesIO()
    iio.mimwrite(buf, [frames[i] for i in range(frames.shape[0])], format="mp4", fps=fps, codec="libx264")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_frames(b64: str) -> np.ndarray:
    r = iio.get_reader(io.BytesIO(base64.b64decode(b64)), format="mp4")
    return np.stack([f for f in r])


def _ramp(n):
    return np.minimum(np.arange(n), np.arange(n)[::-1]) + 1.0


def mac_expand_prompts(endpoint: str, base_prompt: str, n: int) -> list[str]:
    """Use the Mac mini's Kakeya MLX text server to specialize per-tile prompts.
    Returns n prompts; on any failure raises (caller decides skip-not-fake)."""
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    msg = (f"Rewrite this video prompt into {n} short variations, one per line, each "
           f"emphasizing a different spatial region (top-left, top-right, bottom-left, "
           f"bottom-right). Prompt: {base_prompt}")
    body = {"model": os.environ.get("KAKEYA_MODEL", "kakeya-local"),
            "messages": [{"role": "user", "content": msg}], "max_tokens": 200, "temperature": 0.7}
    out = _post(url, body, timeout=120)
    text = out["choices"][0]["message"]["content"]
    lines = [ln.strip(" -*0123456789.") for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if ln][:n]
    while len(lines) < n:
        lines.append(base_prompt)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--proposer-steps", type=int, default=6)
    ap.add_argument("--refine-steps", type=int, default=16)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="distributed_wan_final.mp4")
    args = ap.parse_args()

    workers = [w for w in os.environ.get("WAN_WORKERS", "").split(",") if w.strip()]
    if not workers:
        raise SystemExit("set WAN_WORKERS=http://host:port[,http://host2:port2,...]")
    mac = os.environ.get("KAKEYA_ENDPOINT", "").strip()
    print(f"[orch] {len(workers)} WAN worker(s); mac_text={'yes' if mac else 'no'}", flush=True)
    for w in workers:
        h = _get(w.rstrip("/") + "/healthz")
        assert h.get("device") == "cuda", f"worker {w} not on CUDA: {h}"
        print(f"[orch] worker {w} OK device={h['device']}", flush=True)

    tiles = [(jy, jx) for jy in range(NY) for jx in range(NX)]

    # 1) Mac text plane (skip-not-fake)
    tile_prompts = {t: args.prompt for t in tiles}
    if mac:
        try:
            variants = mac_expand_prompts(mac, args.prompt, len(tiles))
            tile_prompts = {t: variants[i] for i, t in enumerate(tiles)}
            print(f"[orch] Mac MLX specialized {len(variants)} per-tile prompts", flush=True)
        except Exception as exc:
            print(f"[orch] Mac text plane UNAVAILABLE ({exc}); using base prompt (NOT faked)", flush=True)
    else:
        print("[orch] KAKEYA_ENDPOINT unset -> Mac text plane skipped (base prompt)", flush=True)

    # 2) framework on worker[0]
    print("[orch] framework (distilled proposer) on worker[0]...", flush=True)
    fw = _post(workers[0].rstrip("/") + "/v1/framework",
               {"prompt": args.prompt, "width": WT, "height": HT, "num_frames": args.frames,
                "steps": args.proposer_steps, "seed": args.seed})
    framework = _b64_to_frames(fw["mp4_b64"])
    print(f"[orch] framework {framework.shape} in {fw['gen_s']}s", flush=True)

    # 3) upscale -> canvas -> tile crops
    canvas = np.stack([np.asarray(Image.fromarray(framework[i]).resize((CW, CH), Image.BICUBIC))
                       for i in range(framework.shape[0])])

    # 4) dispatch tile refines CONCURRENTLY across workers (data-plane parallelism)
    def refine(idx, t):
        jy, jx = t; oy, ox = y_off[jy], x_off[jx]
        crop = canvas[:, oy:oy + HT, ox:ox + WT, :]
        w = workers[idx % len(workers)]
        r = _post(w.rstrip("/") + "/v1/refine_tile",
                  {"prompt": tile_prompts[t], "mp4_b64": _frames_to_b64(crop), "width": WT, "height": HT,
                   "num_frames": args.frames, "steps": args.refine_steps, "strength": args.strength,
                   "seed": args.seed + 10 + jy * 2 + jx})
        return t, _b64_to_frames(r["mp4_b64"]), w, r["gen_s"]

    print(f"[orch] refining {len(tiles)} tiles across {len(workers)} worker(s)...", flush=True)
    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=len(tiles)) as pool:
        for t, frames, w, gs in pool.map(lambda a: refine(*a), list(enumerate(tiles))):
            results[t] = frames
            print(f"[orch]   tile {t} <- {w} ({gs}s)", flush=True)
    wall = time.time() - t0

    # 5) weight-map merge
    T = args.frames
    acc = np.zeros((T, CH, CW, 3), np.float64); wsum = np.zeros((T, CH, CW, 1), np.float64)
    wmap = np.outer(_ramp(HT), _ramp(WT))[None, :, :, None]
    for (jy, jx), frames in results.items():
        oy, ox = y_off[jy], x_off[jx]
        acc[:, oy:oy + HT, ox:ox + WT, :] += frames.astype(np.float64) * wmap
        wsum[:, oy:oy + HT, ox:ox + WT, :] += wmap
    final = (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)
    iio.mimwrite(args.out, [final[i] for i in range(T)], format="mp4", fps=12, codec="libx264")
    Image.fromarray(final[T // 2]).save(args.out.replace(".mp4", "_mid.png"))

    meta = {"workers": len(workers), "mac_text": bool(mac), "canvas_px": [CH, CW],
            "tiles": f"{NX}x{NY}", "tile_refine_wall_s": round(wall, 2), "out": args.out}
    print("ORCH_DONE " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
