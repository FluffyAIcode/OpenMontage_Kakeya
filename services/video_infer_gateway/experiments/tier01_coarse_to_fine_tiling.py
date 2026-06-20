"""Tier 0 + Tier 1 real-GPU experiments for ADR 0004.

Validates, on REAL model weights on a REAL GPU (no mock/fake/fallback/simplify):

  Tier 0 — coarse-to-fine: a cheap few-step PROPOSER pass produces a rough
           framework; a VERIFIER vid2vid pass refines it. Compared against a
           monolithic full-step baseline. Measures wall-clock + whether the
           refine preserves the proposer's layout (the "alignment" question).

  Tier 1 — distributed tiling + merge: the upscaled coarse framework is
           DECOMPOSED into native-resolution OVERLAPPING tiles; each tile is
           refined independently (the "parallel verifiers on different regions");
           tiles are MERGED two ways — (A) hard placement, (B) spatial-weight-map
           blending (the f_θ "decompose/merge consistency" role, heuristic form).
           Measures seam energy + temporal flicker for A vs B -> tells us whether
           a *learned* f_θ is even needed.

Model: CogVideoX-2b (a DiT video-diffusion model already resident on the box).
WAN 2.1 weights do not fit the 3.2 GB free disk alongside the running gateway, so
the MECHANISM is validated here; per ADR 0004 §1-3 the seam/coherence/speed
findings transfer to WAN (same DiT-video paradigm). This is a model substitution,
NOT a simplification of the experiment's rigor: real diffusion, real VAE decode,
real native-resolution tiles, real metrics.

Outputs real artifacts + a METRICS json block to stdout.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "THUDM/CogVideoX-2b"
W, H, FRAMES = 720, 480, 49          # CogVideoX-2b native grid
PROMPT = "a serene koi pond at golden hour, gentle ripples on the water, cinematic, highly detailed"
NEG = "worst quality, low quality, blurry, distorted, watermark"
SEED = 42
OUT = Path("/workspace/kakeya_int/tier01")
OUT.mkdir(parents=True, exist_ok=True)


def _np(frames):
    """list[PIL] -> uint8 array [T,H,W,3]."""
    return np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames])


def _pil(arr):
    """array [T,H,W,3] -> list[PIL]."""
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]


def _save_mp4(arr, path, fps=8):
    import imageio.v2 as imageio

    imageio.mimsave(str(path), [arr[i] for i in range(arr.shape[0])], fps=fps,
                    codec="libx264", quality=8)


def _psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse)


def _ncc(a, b):
    """Normalized cross-correlation on grayscale — structural alignment proxy."""
    a = a.astype(np.float64).mean(-1).ravel(); b = b.astype(np.float64).mean(-1).ravel()
    a -= a.mean(); b -= b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


def _seam_metric(canvas, x_edges, y_edges):
    """Mean abs luminance discontinuity ACROSS the TRUE tile-edge seam lines,
    normalized by the mean abs gradient in tile INTERIORS (value ~1.0 = seam no
    more visible than ordinary texture; >>1 = a visible stitch line).

    IMPORTANT: measure at the tile EDGES (where a hard merge actually discontinues),
    not the overlap centre — at the centre the later tile fully overwrites the
    earlier one, so a hard merge is coincidentally continuous there. (Fixed after
    the first run mislocated the seam; see loop log I10.)

    canvas: [T,H,W,3] uint8. x_edges/y_edges: lists of seam x/y positions.
    """
    g = canvas.astype(np.float64).mean(-1)  # [T,H,W]
    v = max(float(np.abs(g[:, :, x] - g[:, :, x - 1]).mean()) for x in x_edges)
    h = max(float(np.abs(g[:, y, :] - g[:, y - 1, :]).mean()) for y in y_edges)
    gx = np.abs(np.diff(g, axis=2))
    gy = np.abs(np.diff(g, axis=1))
    ref = 0.5 * (gx.mean() + gy.mean()) + 1e-6
    return {
        "seam_v_abs": round(float(v), 3),
        "seam_h_abs": round(float(h), 3),
        "interior_grad_abs": round(float(ref), 3),
        "seam_v_ratio": round(float(v / ref), 3),
        "seam_h_ratio": round(float(h / ref), 3),
    }


def _temporal_flicker(canvas, bx, by, band=16):
    """Mean abs frame-to-frame diff in a band around the seams (lower = steadier)."""
    g = canvas.astype(np.float64).mean(-1)
    T = g.shape[0]
    if T < 2:
        return 0.0
    xs = slice(max(0, bx - band), bx + band)
    ys = slice(max(0, by - band), by + band)
    region = np.concatenate([g[:, :, xs].reshape(T, -1), g[:, ys, :].reshape(T, -1)], axis=1)
    return round(float(np.abs(np.diff(region, axis=0)).mean()), 3)


def main():
    torch.manual_seed(SEED)
    metrics = {"model": MODEL_ID, "device": None, "tier0": {}, "tier1": {}}

    import diffusers

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    metrics["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    t2v = diffusers.CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to("cuda")
    t2v.vae.enable_tiling(); t2v.vae.enable_slicing()
    # vid2vid shares the SAME weights (no extra disk / negligible extra VRAM)
    v2v = diffusers.CogVideoXVideoToVideoPipeline(**t2v.components)

    def gen(steps, gen_seed):
        g = torch.Generator(device="cpu").manual_seed(gen_seed)
        out = t2v(prompt=PROMPT, negative_prompt=NEG, num_frames=FRAMES,
                  width=W, height=H, num_inference_steps=steps, guidance_scale=6.0,
                  generator=g)
        return _np(out.frames[0])

    def refine(video_arr, strength, steps, gen_seed):
        g = torch.Generator(device="cpu").manual_seed(gen_seed)
        out = v2v(prompt=PROMPT, negative_prompt=NEG, video=_pil(video_arr),
                  strength=strength, num_inference_steps=steps, guidance_scale=6.0,
                  generator=g)
        return _np(out.frames[0])

    # ---------------- Tier 0 ----------------
    print("[tier0] proposer (8 steps)...", flush=True)
    t = time.time(); coarse = gen(8, SEED); t_prop = time.time() - t

    print("[tier0] verifier refine (vid2vid, strength=0.6, 24 steps)...", flush=True)
    t = time.time(); refined = refine(coarse, 0.6, 24, SEED + 1); t_ver = time.time() - t

    print("[tier0] monolithic baseline (40 steps)...", flush=True)
    t = time.time(); full = gen(40, SEED); t_full = time.time() - t

    _save_mp4(coarse, OUT / "t0_coarse.mp4")
    _save_mp4(refined, OUT / "t0_refined.mp4")
    _save_mp4(full, OUT / "t0_full.mp4")

    metrics["tier0"] = {
        "t_proposer_s": round(t_prop, 2),
        "t_verifier_s": round(t_ver, 2),
        "t_coarse_to_fine_s": round(t_prop + t_ver, 2),
        "t_monolithic_s": round(t_full, 2),
        "wallclock_speedup_x": round(t_full / (t_prop + t_ver), 3),
        # alignment: does the refine keep the proposer's layout?
        "align_refined_vs_coarse_ncc": round(np.mean([_ncc(refined[i], coarse[i]) for i in range(0, FRAMES, 6)]), 4),
        "align_refined_vs_coarse_psnr": round(np.mean([_psnr(refined[i], coarse[i]) for i in range(0, FRAMES, 6)]), 2),
        # quality proximity to monolithic
        "refined_vs_full_psnr": round(np.mean([_psnr(refined[i], full[i]) for i in range(0, FRAMES, 6)]), 2),
    }
    print("[tier0] done:", json.dumps(metrics["tier0"]), flush=True)

    # ---------------- Tier 1 ----------------
    # Target canvas 1280x800 from 2x2 native 720x480 tiles overlapping by 160px.
    xs = [0, 560]   # tile x offsets (720 wide) -> canvas width 560+720=1280
    ys = [0, 320]   # tile y offsets (480 tall) -> canvas height 320+480=800
    CW, CH = 1280, 800
    # TRUE tile-edge seam lines (where a hard merge discontinues): the inner
    # edges of the second column/row tiles, and the inner edge of the first.
    x_edges = [560, 720]
    y_edges = [320, 480]
    seam_x, seam_y = 560, 320  # for the temporal-flicker band

    # upscaled coarse framework at target size (the low-res framework, enlarged)
    up = np.stack([np.asarray(Image.fromarray(coarse[i]).resize((CW, CH), Image.BICUBIC))
                   for i in range(FRAMES)])

    print("[tier1] refining 4 native-res tiles independently...", flush=True)
    tiles = {}
    t = time.time()
    for ti, oy in enumerate(ys):
        for tj, ox in enumerate(xs):
            crop = up[:, oy:oy + H, ox:ox + W, :]
            tiles[(ti, tj)] = refine(crop, 0.5, 20, SEED + 10 + ti * 2 + tj)
    t_tiles = time.time() - t

    def merge(blend: bool):
        acc = np.zeros((FRAMES, CH, CW, 3), np.float64)
        wsum = np.zeros((FRAMES, CH, CW, 1), np.float64)
        for (ti, tj), tile in tiles.items():
            oy, ox = ys[ti], xs[tj]
            if blend:
                # linear ramp weight map (MultiDiffusion/PatchVSR-style)
                wx = np.minimum(np.arange(W), np.arange(W)[::-1]) + 1.0
                wy = np.minimum(np.arange(H), np.arange(H)[::-1]) + 1.0
                wmap = np.outer(wy, wx)[None, :, :, None]
            else:
                wmap = np.ones((1, H, W, 1))
            acc[:, oy:oy + H, ox:ox + W, :] += tile.astype(np.float64) * wmap
            wsum[:, oy:oy + H, ox:ox + W, :] += wmap
        out = (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)
        return out

    hard = merge(blend=False)
    blended = merge(blend=True)
    _save_mp4(hard, OUT / "t1_merge_hard.mp4")
    _save_mp4(blended, OUT / "t1_merge_blended.mp4")
    # save a mid-frame png for visual evidence
    Image.fromarray(hard[FRAMES // 2]).save(OUT / "t1_hard_mid.png")
    Image.fromarray(blended[FRAMES // 2]).save(OUT / "t1_blended_mid.png")

    sm_hard = _seam_metric(hard, x_edges, y_edges)
    sm_blend = _seam_metric(blended, x_edges, y_edges)
    metrics["tier1"] = {
        "tiles": "2x2 native 720x480, 160px overlap, canvas 1280x800",
        "t_tiles_total_s": round(t_tiles, 2),
        "t_per_tile_s": round(t_tiles / 4, 2),
        "hard_merge": {**sm_hard, "temporal_flicker": _temporal_flicker(hard, seam_x, seam_y)},
        "blended_merge": {**sm_blend, "temporal_flicker": _temporal_flicker(blended, seam_x, seam_y)},
        "seam_v_reduction_pct": round(100 * (1 - sm_blend["seam_v_ratio"] / max(sm_hard["seam_v_ratio"], 1e-6)), 1),
        "seam_h_reduction_pct": round(100 * (1 - sm_blend["seam_h_ratio"] / max(sm_hard["seam_h_ratio"], 1e-6)), 1),
    }
    print("[tier1] done:", json.dumps(metrics["tier1"]), flush=True)

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
