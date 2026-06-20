"""Latent MultiDiffusion for WAN — the denoise-time merge-consistency fix (ADR 0004 I12).

Tier 1 (post-hoc pixel blend) and Tier 1b (shared noise) both FAILED to remove
cross-tile divergence, because independently-refined tiles diverge in their overlaps
(context/position-driven). The production fix: do the tiling INSIDE the denoiser —
one shared canvas latent, and at EVERY denoise step fuse the overlapping tiles'
predictions (weighted) before the scheduler step. Tiles then co-evolve and cannot
diverge → seamless larger-than-native generation.

This builds it for real WAN 2.1 1.3B (a bidirectional DiT) and compares, on the SAME
shared noise + prompt + tiling + scheduler, decoded once each:

  A) INDEPENDENT: each tile denoised in its own loop (Tier-1b mechanism) → merge latents.
  B) MULTIDIFFUSION: shared canvas latent, per-step prediction fusion.

Real GPU, real weights, no mock/fake/fallback/simplify. Metric: seam ratio at the true
tile edges on the decoded pixels (>1 = visible stitch; ~1 = invisible).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
PROMPT = "a serene koi pond at golden hour, gentle ripples on the water, lily pads, cinematic, highly detailed"
NEG = "worst quality, low quality, blurry, distorted"
SEED = 42
STEPS = 20
FRAMES = 25                       # 4*6+1  -> T_lat=7
HT, WT = 60, 104                  # native latent tile (480x832 / 8)
OV = 24                           # latent overlap
NX, NY = 2, 2                     # 2x2 tiles
OUT = Path("/workspace/kakeya_int/mdiff")
OUT.mkdir(parents=True, exist_ok=True)

# canvas latent size from 2x2 native tiles with overlap
x_off = [i * (WT - OV) for i in range(NX)]           # [0, 80]
y_off = [j * (HT - OV) for j in range(NY)]           # [0, 36]
CWL = x_off[-1] + WT                                  # 184
CHL = y_off[-1] + HT                                  # 96
# TRUE tile-boundary seam lines (latent): start of tile-1 and end of tile-0.
# (Earlier bug I17 used duplicate/over-lap-start columns; fixed.) NOTE: this seam
# metric is confounded by real horizontal/vertical SCENE edges that happen to fall
# near a boundary — the decoded-frame visual remains the primary arbiter.
X_EDGES = [x_off[1], WT]                               # latent [80, 104] -> px [640, 832]
Y_EDGES = [y_off[1], HT]                               # latent [36, 60]  -> px [288, 480]


def _ramp(n):
    return (np.minimum(np.arange(n), np.arange(n)[::-1]) + 1.0)


def _tile_weight(device, dtype):
    w = np.outer(_ramp(HT), _ramp(WT)).astype(np.float32)
    return torch.from_numpy(w).to(device, dtype)[None, None, None]  # [1,1,1,HT,WT]


def _seam(canvas_uint8):
    g = canvas_uint8.astype(np.float64).mean(-1)  # [T,H,W]
    xs = [e * 8 for e in X_EDGES]
    ys = [e * 8 for e in Y_EDGES]
    v = max(float(np.abs(g[:, :, x] - g[:, :, x - 1]).mean()) for x in xs if 0 < x < g.shape[2])
    h = max(float(np.abs(g[:, y, :] - g[:, y - 1, :]).mean()) for y in ys if 0 < y < g.shape[1])
    ref = 0.5 * (np.abs(np.diff(g, axis=2)).mean() + np.abs(np.diff(g, axis=1)).mean()) + 1e-6
    return {"seam_v_ratio": round(v / ref, 3), "seam_h_ratio": round(h / ref, 3)}


@torch.no_grad()
def main():
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

    torch.set_grad_enabled(False)  # custom loop: no autograd graph (avoids OOM)
    dtype = torch.bfloat16
    print("[load] WAN 2.1 1.3B ...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")
    dev = pipe._execution_device
    tdtype = pipe.transformer.dtype

    prompt_embeds, _ = pipe.encode_prompt(prompt=PROMPT, negative_prompt=NEG,
                                          do_classifier_free_guidance=False, device=dev)
    prompt_embeds = prompt_embeds.to(tdtype)

    # shared canvas noise
    g = torch.Generator(device="cpu").manual_seed(SEED)
    from diffusers.utils.torch_utils import randn_tensor
    canvas_noise = randn_tensor((1, 16, 7, CHL, CWL), generator=g, device=dev, dtype=torch.float32)

    wmap = _tile_weight(dev, torch.float32)
    tiles = [(jy, jx) for jy in range(NY) for jx in range(NX)]

    def denoise_tile(lat, sched):
        for t in sched.timesteps:
            mo = pipe.transformer(hidden_states=lat.to(tdtype), timestep=t.expand(1),
                                  encoder_hidden_states=prompt_embeds, return_dict=False)[0]
            lat = sched.step(mo, t, lat, return_dict=False)[0]
        return lat

    # ---- A) INDEPENDENT tiles (Tier-1b mechanism) ----
    print("[A] independent tile denoise...", flush=True)
    t0 = time.time()
    accA = torch.zeros((1, 16, 7, CHL, CWL), device=dev, dtype=torch.float32)
    wsumA = torch.zeros((1, 1, 1, CHL, CWL), device=dev, dtype=torch.float32)
    for (jy, jx) in tiles:
        oy, ox = y_off[jy], x_off[jx]
        sched = copy.deepcopy(pipe.scheduler); sched.set_timesteps(STEPS, device=dev)
        lat = canvas_noise[:, :, :, oy:oy + HT, ox:ox + WT].clone()
        lat = denoise_tile(lat, sched)
        accA[:, :, :, oy:oy + HT, ox:ox + WT] += lat * wmap
        wsumA[:, :, :, oy:oy + HT, ox:ox + WT] += wmap
    latA = accA / wsumA.clamp_min(1e-6)
    tA = time.time() - t0

    # ---- B) MULTIDIFFUSION (shared canvas, per-step fusion) ----
    print("[B] latent MultiDiffusion (per-step fusion)...", flush=True)
    t0 = time.time()
    schedB = copy.deepcopy(pipe.scheduler); schedB.set_timesteps(STEPS, device=dev)
    latB = canvas_noise.clone()
    for t in schedB.timesteps:
        acc = torch.zeros_like(latB); wsum = torch.zeros((1, 1, 1, CHL, CWL), device=dev, dtype=torch.float32)
        for (jy, jx) in tiles:
            oy, ox = y_off[jy], x_off[jx]
            crop = latB[:, :, :, oy:oy + HT, ox:ox + WT].to(tdtype)
            mo = pipe.transformer(hidden_states=crop, timestep=t.expand(1),
                                  encoder_hidden_states=prompt_embeds, return_dict=False)[0].float()
            acc[:, :, :, oy:oy + HT, ox:ox + WT] += mo * wmap
            wsum[:, :, :, oy:oy + HT, ox:ox + WT] += wmap
        fused = acc / wsum.clamp_min(1e-6)                 # fused canvas prediction
        latB = schedB.step(fused, t, latB, return_dict=False)[0]
    tB = time.time() - t0

    # ---- decode both (once each) ----
    def decode(lat):
        lat = lat.to(pipe.vae.dtype)
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 16, 1, 1, 1).to(lat.device, lat.dtype)
        std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, 16, 1, 1, 1).to(lat.device, lat.dtype)
        lat = lat / std + mean
        vid = pipe.vae.decode(lat, return_dict=False)[0]   # [1,3,T,H,W] in [-1,1]
        vid = ((vid.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]  # [3,T,H,W]
        return vid.permute(1, 2, 3, 0).cpu().numpy()       # [T,H,W,3]

    print("[decode] ...", flush=True)
    vidA = decode(latA); vidB = decode(latB)

    import imageio.v2 as iio
    iio.mimsave(str(OUT / "independent.mp4"), [vidA[i] for i in range(vidA.shape[0])], fps=12, codec="libx264", quality=8)
    iio.mimsave(str(OUT / "multidiffusion.mp4"), [vidB[i] for i in range(vidB.shape[0])], fps=12, codec="libx264", quality=8)
    Image.fromarray(vidA[vidA.shape[0] // 2]).save(OUT / "independent_mid.png")
    Image.fromarray(vidB[vidB.shape[0] // 2]).save(OUT / "multidiffusion_mid.png")

    metrics = {
        "model": MODEL, "device": str(dev), "canvas_latent": [CHL, CWL], "canvas_px": [CHL * 8, CWL * 8],
        "tiles": f"{NX}x{NY} native {WT*8}x{HT*8}, overlap {OV*8}px", "steps": STEPS, "frames": FRAMES,
        "independent_merge": {**_seam(vidA), "t_s": round(tA, 2)},
        "multidiffusion": {**_seam(vidB), "t_s": round(tB, 2)},
    }
    a, b = metrics["independent_merge"], metrics["multidiffusion"]
    metrics["seam_v_reduction_pct"] = round(100 * (1 - b["seam_v_ratio"] / max(a["seam_v_ratio"], 1e-6)), 1)
    metrics["seam_h_reduction_pct"] = round(100 * (1 - b["seam_h_ratio"] / max(a["seam_h_ratio"], 1e-6)), 1)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
