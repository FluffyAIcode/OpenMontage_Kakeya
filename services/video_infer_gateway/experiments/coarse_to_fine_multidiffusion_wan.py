"""Full ADR 0004 pipeline on HIGH-FREQUENCY content + boundary-isolating seam metric.

Wires latent MultiDiffusion into the coarse-to-fine vid2vid refine, driven by the
distilled proposer's framework, and stress-tests it where independent tiles diverge:

  1. distilled proposer (WAN + CausVid LoRA, 6-step)  -> low-res framework
  2. upscale framework -> high-res canvas (1472x768)
  3. VAE-encode + SDEdit noise (strength)             -> noised canvas latent
  4. tiled refine (full WAN, LoRA off), two ways:
       A) INDEPENDENT: each tile denoised on its own crop, merge latents
       B) MULTIDIFFUSION: shared canvas, per-step prediction fusion
  5. decode both; seam metric that ISOLATES tile boundaries from scene edges
     (peak discontinuity AT the boundary vs the median over a window around it).

Real WAN 2.1 1.3B on GPU. No mock/fake/fallback/simplify.
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
LORA_REPO = "Kijai/WanVideo_comfy"
LORA_FILE = "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"
# HIGH-FREQUENCY / structured content: independent tiles will hallucinate divergent detail
PROMPT = ("a dense bookshelf fully packed with colorful books with sharp detailed spines, "
          "ornate patterns, fine intricate texture, sharp focus, highly detailed")
NEG = "worst quality, low quality, blurry, smooth, out of focus"
SEED = 11
FRAMES = 25
STEPS = 16
STRENGTH = 0.6
WT, HT = 104, 60                  # native latent tile (832x480/8)
OV = 24
NX, NY = 2, 2
OUT = Path("/workspace/kakeya_int/c2fmd")
OUT.mkdir(parents=True, exist_ok=True)

x_off = [i * (WT - OV) for i in range(NX)]
y_off = [j * (HT - OV) for j in range(NY)]
CWL, CHL = x_off[-1] + WT, y_off[-1] + HT      # 184 x 96
CW, CH = CWL * 8, CHL * 8                       # 1472 x 768
X_EDGES_PX = [x_off[1] * 8, WT * 8]             # [640, 832]
Y_EDGES_PX = [y_off[1] * 8, HT * 8]             # [288, 480]


def _np(frames):
    if isinstance(frames, np.ndarray):
        arr = frames
    elif hasattr(frames[0], "convert"):
        return np.stack([np.asarray(f.convert("RGB"), np.uint8) for f in frames])
    else:
        arr = np.stack([np.asarray(f) for f in frames])
    if arr.dtype != np.uint8:
        arr = (arr.clip(0, 1) * 255).round().astype(np.uint8) if float(arr.max()) <= 1.0 + 1e-3 \
            else arr.clip(0, 255).astype(np.uint8)
    return arr


def _ramp(n):
    return np.minimum(np.arange(n), np.arange(n)[::-1]) + 1.0


def _seam_excess(canvas, x_edges, y_edges, win=48):
    """Seam ISOLATED from scene edges: discontinuity AT the boundary line / median
    discontinuity over a +/-win window around it. ~1 = looks like scene texture;
    >>1 = a tiling stitch spike at exactly the boundary."""
    g = canvas.astype(np.float64).mean(-1)  # [T,H,W]
    dv = np.abs(g[:, :, 1:] - g[:, :, :-1]).mean(axis=(0, 1))  # per-column disc [W-1]
    dh = np.abs(g[:, 1:, :] - g[:, :-1, :]).mean(axis=(0, 2))  # per-row disc [H-1]
    def excess(prof, e):
        lo, hi = max(1, e - win), min(len(prof), e + win)
        med = float(np.median(prof[lo:hi])) + 1e-6
        return float(prof[min(e, len(prof) - 1)] / med)
    return {
        "seam_v_excess": round(max(excess(dv, e) for e in x_edges), 3),
        "seam_h_excess": round(max(excess(dh, e) for e in y_edges), 3),
    }


def _overlap_disagree(tiles):
    """Mean abs latent diff between adjacent tiles in their shared overlap (ghosting)."""
    l, r = tiles[(0, 0)], tiles[(0, 1)]
    dv = float((l[:, :, :, :, WT - OV:WT] - r[:, :, :, :, :OV]).abs().mean())
    t, b = tiles[(0, 0)], tiles[(1, 0)]
    dh = float((t[:, :, :, HT - OV:HT, :] - b[:, :, :, :OV, :]).abs().mean())
    return round(0.5 * (dv + dh), 4)


@torch.no_grad()
def main():
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanVideoToVideoPipeline
    torch.set_grad_enabled(False)

    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")
    dev = pipe._execution_device
    tdtype = pipe.transformer.dtype
    v2v = WanVideoToVideoPipeline(**{k: pipe.components[k] for k in
                                     ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")})

    # 1) distilled proposer (CausVid) -> low-res framework
    print("[proposer] CausVid 6-step framework...", flush=True)
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, adapter_name="causvid")
    pipe.set_adapters("causvid")
    g = torch.Generator(device="cpu").manual_seed(SEED)
    t0 = time.time()
    coarse = _np(pipe(prompt=PROMPT, negative_prompt=NEG, num_frames=FRAMES, width=832, height=480,
                      num_inference_steps=6, guidance_scale=1.0, generator=g).frames[0])
    t_prop = time.time() - t0
    pipe.disable_lora()

    # 2) upscale framework -> canvas
    canvas = np.stack([np.asarray(Image.fromarray(coarse[i]).resize((CW, CH), Image.BICUBIC))
                       for i in range(FRAMES)])

    # 3) encode + SDEdit noise via v2v helpers
    prompt_embeds, _ = v2v.encode_prompt(prompt=PROMPT, negative_prompt=NEG,
                                         do_classifier_free_guidance=False, device=dev)
    prompt_embeds = prompt_embeds.to(tdtype)
    video_t = v2v.video_processor.preprocess_video(
        [Image.fromarray(canvas[i]) for i in range(FRAMES)], height=CH, width=CW).to(dev)
    v2v.scheduler.set_timesteps(STEPS, device=dev)
    timesteps, _ = v2v.get_timesteps(STEPS, v2v.scheduler.timesteps, STRENGTH, dev)
    init_latent = v2v.prepare_latents(video=video_t, batch_size=1, num_channels_latents=16,
                                      height=CH, width=CW, dtype=torch.float32, device=dev,
                                      generator=torch.Generator("cpu").manual_seed(SEED + 5),
                                      latents=None, timestep=timesteps[:1])
    tiles_ij = [(jy, jx) for jy in range(NY) for jx in range(NX)]
    wmap = torch.from_numpy(np.outer(_ramp(HT), _ramp(WT)).astype(np.float32)).to(dev)[None, None, None]

    def step_sched(sched, mo, t, lat):
        return sched.step(mo, t, lat, return_dict=False)[0]

    # 4A) INDEPENDENT tiles
    print("[A] independent tile vid2vid...", flush=True)
    t0 = time.time()
    accA = torch.zeros((1, 16, 7, CHL, CWL), device=dev); wsumA = torch.zeros((1, 1, 1, CHL, CWL), device=dev)
    final_tiles = {}
    for (jy, jx) in tiles_ij:
        oy, ox = y_off[jy], x_off[jx]
        sched = copy.deepcopy(v2v.scheduler)
        lat = init_latent[:, :, :, oy:oy + HT, ox:ox + WT].clone()
        for t in timesteps:
            mo = pipe.transformer(hidden_states=lat.to(tdtype), timestep=t.expand(1),
                                  encoder_hidden_states=prompt_embeds, return_dict=False)[0].float()
            lat = step_sched(sched, mo, t, lat)
        final_tiles[(jy, jx)] = lat
        accA[:, :, :, oy:oy + HT, ox:ox + WT] += lat * wmap
        wsumA[:, :, :, oy:oy + HT, ox:ox + WT] += wmap
    latA = accA / wsumA.clamp_min(1e-6)
    disagree = _overlap_disagree(final_tiles)
    tA = time.time() - t0

    # 4B) MULTIDIFFUSION
    print("[B] latent MultiDiffusion vid2vid...", flush=True)
    t0 = time.time()
    schedB = copy.deepcopy(v2v.scheduler)
    latB = init_latent.clone()
    for t in timesteps:
        acc = torch.zeros_like(latB); wsum = torch.zeros((1, 1, 1, CHL, CWL), device=dev)
        for (jy, jx) in tiles_ij:
            oy, ox = y_off[jy], x_off[jx]
            crop = latB[:, :, :, oy:oy + HT, ox:ox + WT].to(tdtype)
            mo = pipe.transformer(hidden_states=crop, timestep=t.expand(1),
                                  encoder_hidden_states=prompt_embeds, return_dict=False)[0].float()
            acc[:, :, :, oy:oy + HT, ox:ox + WT] += mo * wmap
            wsum[:, :, :, oy:oy + HT, ox:ox + WT] += wmap
        latB = step_sched(schedB, acc / wsum.clamp_min(1e-6), t, latB)
    tB = time.time() - t0

    # 5) decode + metrics
    def decode(lat):
        lat = lat.to(pipe.vae.dtype)
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 16, 1, 1, 1).to(lat.device, lat.dtype)
        std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, 16, 1, 1, 1).to(lat.device, lat.dtype)
        vid = pipe.vae.decode(lat / std + mean, return_dict=False)[0]
        return ((vid.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 3, 0).cpu().numpy()

    vidA, vidB = decode(latA), decode(latB)
    import imageio.v2 as iio
    for nm, v in (("independent", vidA), ("multidiffusion", vidB)):
        iio.mimsave(str(OUT / f"{nm}.mp4"), [v[i] for i in range(v.shape[0])], fps=12, codec="libx264", quality=8)
        Image.fromarray(v[v.shape[0] // 2]).save(OUT / f"{nm}_mid.png")

    a, b = _seam_excess(vidA, X_EDGES_PX, Y_EDGES_PX), _seam_excess(vidB, X_EDGES_PX, Y_EDGES_PX)
    metrics = {
        "model": MODEL, "content": "high_frequency_bookshelf", "canvas_px": [CH, CW],
        "strength": STRENGTH, "steps": STEPS, "frames": FRAMES,
        "t_proposer_s": round(t_prop, 2), "overlap_disagreement_independent": disagree,
        "independent": {**a, "t_s": round(tA, 2)},
        "multidiffusion": {**b, "t_s": round(tB, 2)},
        "seam_v_excess_reduction_pct": round(100 * (1 - b["seam_v_excess"] / max(a["seam_v_excess"], 1e-6)), 1),
        "seam_h_excess_reduction_pct": round(100 * (1 - b["seam_h_excess"] / max(a["seam_h_excess"], 1e-6)), 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
