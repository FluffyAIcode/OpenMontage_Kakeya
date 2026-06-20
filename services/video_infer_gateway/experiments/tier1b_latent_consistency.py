"""Tier 1b — consistency test for the decompose/merge (f_θ) role (ADR 0004 §6, I11).

Tier 1 showed a heuristic spatial-weight-map blend removes seams on LOW-frequency
content, but flagged (I11) that on content where independently-refined tiles
HALLUCINATE DIVERGENT structure, post-hoc pixel blending *ghosts* instead of fixing.

This experiment isolates the cause and tests a real consistency mechanism, on
HIGH-frequency content, on REAL GPU (no mock/fake/simplify):

  - DIFF  : refine each tile with a DIFFERENT seed (independent denoising).
  - SHARED: refine each tile with a SHARED seed + shared conditioning — the cheap,
            robust form of denoise-time consistency (the essence a denoise-time f_θ
            / latent MultiDiffusion would enforce: overlapping regions co-evolve to
            the SAME content, so the merge has nothing to ghost).

Metrics:
  - overlap_disagreement: mean abs diff between the two tiles' content IN their
    shared overlap region (high = divergence → ghosting potential).
  - seam ratio at true tile edges (as in Tier 1).

Model: resident CogVideoX-2b (disk-safe; WAN won't fit 3.2 GB free). Per ADR 0004
the mechanism transfers. NOTE: full per-step latent MultiDiffusion (fuse every
denoise step) is the production-grade version; shared-noise is the tractable lever
validated here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "THUDM/CogVideoX-2b"
W, H, FRAMES = 720, 480, 49
# high-frequency / structured prompt: divergent hallucination is likely per tile
PROMPT = ("a tall ornate wooden bookshelf packed with colorful books with sharp "
          "detailed spines and titles, a round wall clock, intricate fine textures, "
          "sharp focus, highly detailed")
NEG = "worst quality, low quality, blurry, smudged"
SEED = 7
OUT = Path("/workspace/kakeya_int/tier1b")
OUT.mkdir(parents=True, exist_ok=True)

xs, ys = [0, 560], [0, 320]            # 2x2 native tiles, 160px overlap
CW, CH = 1280, 800
X_EDGES, Y_EDGES = [560, 720], [320, 480]
OVX = (560, 720)   # canvas x-range of the vertical overlap band
OVY = (320, 480)   # canvas y-range of the horizontal overlap band


def _np(frames):
    return np.stack([np.asarray(f.convert("RGB"), np.uint8) for f in frames])


def _pil(arr):
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]


def _save_mid(arr, path):
    Image.fromarray(arr[arr.shape[0] // 2]).save(path)


def _seam(canvas):
    g = canvas.astype(np.float64).mean(-1)
    v = max(float(np.abs(g[:, :, x] - g[:, :, x - 1]).mean()) for x in X_EDGES)
    h = max(float(np.abs(g[:, y, :] - g[:, y - 1, :]).mean()) for y in Y_EDGES)
    ref = 0.5 * (np.abs(np.diff(g, axis=2)).mean() + np.abs(np.diff(g, axis=1)).mean()) + 1e-6
    return {"v_ratio": round(v / ref, 3), "h_ratio": round(h / ref, 3)}


def main():
    import diffusers

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    t2v = diffusers.CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to("cuda")
    t2v.vae.enable_tiling(); t2v.vae.enable_slicing()
    v2v = diffusers.CogVideoXVideoToVideoPipeline(**t2v.components)

    # coarse framework (rough) + upscale to canvas size
    g = torch.Generator(device="cpu").manual_seed(SEED)
    coarse = _np(t2v(prompt=PROMPT, negative_prompt=NEG, num_frames=FRAMES, width=W, height=H,
                     num_inference_steps=8, guidance_scale=6.0, generator=g).frames[0])
    up = np.stack([np.asarray(Image.fromarray(coarse[i]).resize((CW, CH), Image.BICUBIC))
                   for i in range(FRAMES)])

    def refine_tile(crop, seed):
        gg = torch.Generator(device="cpu").manual_seed(seed)
        return _np(v2v(prompt=PROMPT, negative_prompt=NEG, video=_pil(crop),
                       strength=0.7, num_inference_steps=20, guidance_scale=6.0,
                       generator=gg).frames[0])

    def run(mode):
        tiles = {}
        t = time.time()
        for ti, oy in enumerate(ys):
            for tj, ox in enumerate(xs):
                seed = SEED + 100 if mode == "shared" else SEED + 100 + ti * 2 + tj
                tiles[(ti, tj)] = refine_tile(up[:, oy:oy + H, ox:ox + W, :], seed)
        dt = time.time() - t

        # overlap disagreement (vertical band between left/right tiles, row 0)
        l, r = tiles[(0, 0)], tiles[(0, 1)]   # (0,0) covers x0-720; (0,1) covers x560-1280
        # shared canvas x in [560,720] -> local x in left[560:720], right[0:160]
        dis_v = float(np.abs(l[:, :, 560:720, :].astype(np.float64)
                             - r[:, :, 0:160, :].astype(np.float64)).mean())
        top, bot = tiles[(0, 0)], tiles[(1, 0)]  # canvas y in [320,480] -> top[320:480], bot[0:160]
        dis_h = float(np.abs(top[:, 320:480, :, :].astype(np.float64)
                             - bot[:, 0:160, :, :].astype(np.float64)).mean())

        # weight-map blended merge (same as Tier 1)
        acc = np.zeros((FRAMES, CH, CW, 3), np.float64)
        wsum = np.zeros((FRAMES, CH, CW, 1), np.float64)
        wx = (np.minimum(np.arange(W), np.arange(W)[::-1]) + 1.0)
        wy = (np.minimum(np.arange(H), np.arange(H)[::-1]) + 1.0)
        wmap = np.outer(wy, wx)[None, :, :, None]
        for (ti, tj), tile in tiles.items():
            oy, ox = ys[ti], xs[tj]
            acc[:, oy:oy + H, ox:ox + W, :] += tile.astype(np.float64) * wmap
            wsum[:, oy:oy + H, ox:ox + W, :] += wmap
        blended = (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)
        return {
            "t_tiles_s": round(dt, 2),
            "overlap_disagreement_v": round(dis_v, 3),
            "overlap_disagreement_h": round(dis_h, 3),
            "blended_seam": _seam(blended),
        }, blended

    print("[tier1b] DIFF-seed tiles...", flush=True)
    diff_m, diff_img = run("diff")
    _save_mid(diff_img, OUT / "diff_blended_mid.png")
    print("[tier1b] DIFF:", json.dumps(diff_m), flush=True)

    print("[tier1b] SHARED-seed tiles...", flush=True)
    shared_m, shared_img = run("shared")
    _save_mid(shared_img, OUT / "shared_blended_mid.png")
    print("[tier1b] SHARED:", json.dumps(shared_m), flush=True)

    ghost_reduction = round(100 * (1 - shared_m["overlap_disagreement_v"]
                                   / max(diff_m["overlap_disagreement_v"], 1e-6)), 1)
    metrics = {"model": MODEL_ID, "device": "cuda" if torch.cuda.is_available() else "cpu",
               "prompt_kind": "high_frequency_structured",
               "diff_seed": diff_m, "shared_seed": shared_m,
               "overlap_disagreement_reduction_pct_v": ghost_reduction}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
