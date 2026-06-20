"""Single-GPU TIME-DIVISION validation of the full proposer/verifier/f_theta architecture.

Validates the ADR 0004 architecture on ONE GPU by time-sharing it across parts:

  1. PROPOSER (distilled CausVid WAN, 6-step)  -> low-res video framework (832x480)
  2. upscale framework -> wide canvas (1472x480), split into TWO overlapping parts
  3. VERIFIER (full WAN vid2vid) refines PART 1, THEN PART 2  -- sequential on one GPU
     (time-division: only one part's activations resident at a time -> BOUNDED memory)
  4. f_theta (boundary-consistency weight-map) INTEGRATES the two parts -> full video

Also runs a full-canvas single-pass refine as a memory REFERENCE, to show time-division
bounds peak GPU memory (more parts != more memory) -- the original "bounded memory" theme.

Real WAN 2.1 1.3B on a single CUDA GPU. No mock/fake/fallback/simplify.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LORA_REPO = "Kijai/WanVideo_comfy"
LORA_FILE = "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"
PROMPT = "a serene koi pond at golden hour, gentle ripples, lily pads, orange koi, cinematic, highly detailed"
NEG = "worst quality, low quality, blurry, distorted"
SEED = 21
import os
FRAMES = int(os.environ.get("TD_FRAMES", "25"))
NPARTS = int(os.environ.get("TD_PARTS", "2"))      # >=2 horizontal parts (time-division)
PROP_STEPS, VER_STEPS, STRENGTH = 6, 14, 0.55
PART_W, PART_H = 832, 480           # native part
OVX = 192                            # horizontal overlap (px)
X_OFF = [i * (PART_W - OVX) for i in range(NPARTS)]
CW = X_OFF[-1] + PART_W              # canvas width = 832 + (N-1)*640
CH = PART_H
# internal tile-boundary seam lines (px): start of each non-first part + end of first part
SEAM_X = sorted(set([X_OFF[i] for i in range(1, NPARTS)] + [PART_W]))
OUT = Path("/workspace/kakeya_int/timediv")
OUT.mkdir(parents=True, exist_ok=True)


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


def _pil(a):
    return [Image.fromarray(a[i]) for i in range(a.shape[0])]


def _ramp(n):
    return np.minimum(np.arange(n), np.arange(n)[::-1]) + 1.0


def _seam_excess(canvas, x_edges, win=48):
    g = canvas.astype(np.float64).mean(-1)
    dv = np.abs(g[:, :, 1:] - g[:, :, :-1]).mean(axis=(0, 1))
    def excess(e):
        lo, hi = max(1, e - win), min(len(dv), e + win)
        return float(dv[min(e, len(dv) - 1)] / (np.median(dv[lo:hi]) + 1e-6))
    return round(max(excess(e) for e in x_edges), 3)


@torch.no_grad()
def main():
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanVideoToVideoPipeline
    torch.set_grad_enabled(False)

    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, adapter_name="causvid")
    v2v = WanVideoToVideoPipeline(**{k: pipe.components[k] for k in
                                     ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")})

    def refine(video_arr, seed):
        g = torch.Generator("cpu").manual_seed(seed)
        out = v2v(prompt=PROMPT, negative_prompt=NEG, video=_pil(video_arr), strength=STRENGTH,
                  num_inference_steps=VER_STEPS, guidance_scale=5.0, generator=g)
        return _np(out.frames[0])

    def peak_gb():
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)

    # 1) PROPOSER -> framework
    print("[1] proposer (CausVid 6-step) -> framework...", flush=True)
    pipe.set_adapters("causvid")
    g = torch.Generator("cpu").manual_seed(SEED)
    t = time.time()
    framework = _np(pipe(prompt=PROMPT, negative_prompt=NEG, num_frames=FRAMES, width=PART_W,
                         height=PART_H, num_inference_steps=PROP_STEPS, guidance_scale=1.0,
                         generator=g).frames[0])
    t_prop = time.time() - t
    pipe.disable_lora()

    # 2) upscale -> wide canvas, split into N overlapping parts
    canvas = np.stack([np.asarray(Image.fromarray(framework[i]).resize((CW, CH), Image.BICUBIC))
                       for i in range(FRAMES)])

    # 3) VERIFIER — TIME-DIVISION: refine each part sequentially on the ONE GPU
    parts, part_times, part_peaks = [], [], []
    for pi, ox in enumerate(X_OFF):
        print(f"[3.{pi}] verifier refines PART {pi + 1}/{NPARTS} (x={ox})...", flush=True)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        t = time.time()
        out = refine(canvas[:, :, ox:ox + PART_W, :], SEED + 1 + pi)
        part_times.append(round(time.time() - t, 2)); part_peaks.append(peak_gb())
        parts.append(out)
        print(f"      part{pi + 1} {part_times[-1]}s peak={part_peaks[-1]}GB", flush=True)

    # 4) f_theta — integrate all parts (boundary-consistency weight-map merge)
    acc = np.zeros((FRAMES, CH, CW, 3), np.float64); wsum = np.zeros((FRAMES, CH, CW, 1), np.float64)
    wmap = np.outer(_ramp(PART_H), _ramp(PART_W))[None, :, :, None]
    for ox, out in zip(X_OFF, parts):
        acc[:, :, ox:ox + PART_W, :] += out.astype(np.float64) * wmap
        wsum[:, :, ox:ox + PART_W, :] += wmap
    full = (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)

    # reference: full-canvas SINGLE-pass refine (memory comparison; may OOM at scale)
    print("[ref] full-canvas single-pass refine (memory reference)...", flush=True)
    mem_full, t_full, full_oom = None, None, False
    try:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        t = time.time(); _ = refine(canvas, SEED + 99); t_full = round(time.time() - t, 2); mem_full = peak_gb()
        print(f"     full-canvas {t_full}s peak={mem_full}GB", flush=True)
    except torch.cuda.OutOfMemoryError:
        full_oom = True
        print("     full-canvas single pass OOM -> time-division SUCCEEDED where full pass cannot fit", flush=True)

    import imageio.v2 as iio
    iio.mimwrite(str(OUT / "framework.mp4"), [framework[i] for i in range(FRAMES)], fps=12, codec="libx264")
    iio.mimwrite(str(OUT / "timediv_integrated.mp4"), [full[i] for i in range(FRAMES)], fps=12, codec="libx264")
    Image.fromarray(full[FRAMES // 2]).save(OUT / "timediv_integrated_mid.png")

    metrics = {
        "model": MODEL, "canvas_px": [CH, CW], "parts": NPARTS, "overlap_px": OVX, "frames": FRAMES,
        "proposer_s": round(t_prop, 2),
        "verifier_part_times_s": part_times, "verifier_total_timedivision_s": round(sum(part_times), 2),
        "peak_gb_per_part": part_peaks, "peak_gb_timedivision": max(part_peaks),
        "peak_gb_full_canvas_single_pass": mem_full, "full_canvas_oom": full_oom,
        "memory_saved_vs_full_pass_gb": (None if mem_full is None else round(mem_full - max(part_peaks), 2)),
        "f_theta_seam_excess": _seam_excess(full, SEAM_X),
        "full_single_s": t_full,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
