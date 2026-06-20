"""Tier 0 with a FAITHFUL distilled WAN proposer (ADR 0004 §3.1 / §6).

The CogVideoX Tier-0 run used a NON-distilled 8-step proposer → only 1.35× single-GPU.
This run uses a GENUINE distilled few-step WAN proposer (CausVid LoRA on Wan2.1-T2V-1.3B)
to measure the real distilled-proposer speedup:

  - monolithic full WAN (30-step, CFG)              = quality reference (t_full)
  - distilled proposer: WAN + CausVid LoRA, 6-step  = cheap framework (t_prop)
  - verifier refine: FULL WAN vid2vid (LoRA off)    = high-detail completion (t_ver)

Real GPU, real weights, no mock/fake/fallback/simplify. If the LoRA fails to load we
report the failure (we do NOT silently fall back to a non-distilled proposer).
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
W, H, FRAMES = 832, 480, 49           # WAN 480p native; 49 = 4*12+1
PROMPT = "a serene koi pond at golden hour, gentle ripples on the water, cinematic, highly detailed"
NEG = "worst quality, low quality, blurry, distorted"
SEED = 42
OUT = Path("/workspace/kakeya_int/tier0wan")
OUT.mkdir(parents=True, exist_ok=True)


def _np(frames):
    return np.stack([np.asarray(f.convert("RGB"), np.uint8) for f in frames])


def _pil(a):
    return [Image.fromarray(a[i]) for i in range(a.shape[0])]


def _save(a, p, fps=16):
    import imageio.v2 as imageio
    imageio.mimsave(str(p), [a[i] for i in range(a.shape[0])], fps=fps, codec="libx264", quality=8)


def _psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    m = float(np.mean((a - b) ** 2))
    return 99.0 if m < 1e-9 else 10.0 * np.log10(255.0 ** 2 / m)


def _ncc(a, b):
    a = a.astype(np.float64).mean(-1).ravel(); b = b.astype(np.float64).mean(-1).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


def main():
    from diffusers import (
        AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanVideoToVideoPipeline,
    )

    dtype = torch.bfloat16
    print("[load] WAN 2.1 T2V 1.3B (downloads ~14GB on first run)...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")
    v2v = WanVideoToVideoPipeline(**pipe.components)  # shares weights with pipe

    def gen(steps, guidance, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return _np(pipe(prompt=PROMPT, negative_prompt=NEG, num_frames=FRAMES, width=W, height=H,
                        num_inference_steps=steps, guidance_scale=guidance, generator=g).frames[0])

    def refine(video, strength, steps, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return _np(v2v(prompt=PROMPT, negative_prompt=NEG, video=_pil(video), strength=strength,
                       num_inference_steps=steps, guidance_scale=5.0, generator=g).frames[0])

    metrics = {"model": MODEL, "lora": LORA_FILE, "dims": f"{W}x{H}x{FRAMES}"}

    # 1) monolithic full WAN (quality reference)
    print("[baseline] monolithic full WAN, 30-step...", flush=True)
    t = time.time(); full = gen(30, 5.0, SEED); t_full = time.time() - t
    _save(full, OUT / "full.mp4")
    print(f"[baseline] {t_full:.1f}s", flush=True)

    # 2) distilled proposer: load CausVid LoRA. NO silent fallback if this fails.
    print(f"[proposer] loading CausVid LoRA {LORA_FILE} ...", flush=True)
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE, adapter_name="causvid")
    pipe.set_adapters("causvid")
    print("[proposer] distilled WAN, 6-step, guidance=1.0 ...", flush=True)
    t = time.time(); coarse = gen(6, 1.0, SEED); t_prop = time.time() - t
    _save(coarse, OUT / "causvid_coarse.mp4")
    print(f"[proposer] {t_prop:.1f}s", flush=True)

    # 3) verifier refine with FULL WAN (LoRA disabled on the shared transformer)
    pipe.disable_lora()
    print("[verifier] full WAN vid2vid refine, strength=0.5, 15-step ...", flush=True)
    t = time.time(); refined = refine(coarse, 0.5, 15, SEED + 1); t_ver = time.time() - t
    _save(refined, OUT / "refined.mp4")
    print(f"[verifier] {t_ver:.1f}s", flush=True)

    idx = list(range(0, FRAMES, 6))
    metrics["tier0_distilled"] = {
        "t_monolithic_30step_s": round(t_full, 2),
        "t_distilled_proposer_6step_s": round(t_prop, 2),
        "t_verifier_refine_15step_s": round(t_ver, 2),
        "t_coarse_to_fine_s": round(t_prop + t_ver, 2),
        "speedup_distilled_proposer_alone_x": round(t_full / t_prop, 2),
        "speedup_coarse_to_fine_x": round(t_full / (t_prop + t_ver), 2),
        "align_refined_vs_coarse_ncc": round(float(np.mean([_ncc(refined[i], coarse[i]) for i in idx])), 4),
        "distilled_vs_monolithic_psnr": round(float(np.mean([_psnr(coarse[i], full[i]) for i in idx])), 2),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
