# ADR 0017 — HunyuanVideo as the default quality video provider (A/B vs WAN)

**Status:** Accepted (decision: adopt Hunyuan as the quality layer). Productization pending.
**Date:** 2026-06-23
**Related:** ADR 0015/0016 (quality/duration, agent runtime), the "four open-source models as
providers" design.

## 1. Question

Does replacing/augmenting WAN 2.1 with **HunyuanVideo** (13B) improve video quality enough to make it
the default quality layer? Decided by a **real A/B** on the same prompt, judged on actual frames.

## 2. Setup

- vast RTX PRO 6000 Blackwell (97 G). Same prompt: *"a border collie dog running through a snowy
  forest, cinematic, soft winter light."* Both 1280×720, ~2 s.
- **WAN** = current `quality=high` path (1.3B T2V seed → I2V-14B @720p), via the public gateway.
- **Hunyuan** = `hunyuanvideo-community/HunyuanVideo` (diffusers `HunyuanVideoPipeline`), 45 frames,
  30 steps, no offload. New `HunyuanBackend` added to `grpc_worker.py` (`--backend hunyuan`, ops
  framework/i2v).

## 3. Result (frame-level visual judgment)

- **WAN**: a chaotic, abstract blob field — **no recognizable dog or forest**, heavy SR artifacts.
  Effectively unusable. (Root cause: the 1.3B low-res T2V *seed* is garbage; I2V amplifies it.)
- **Hunyuan**: a **coherent, photorealistic** dog walking through a snowy forest, cinematic shallow
  depth of field, realistic snow + blurred trunks. **Clearly production-looking** for a single clip.

**Verdict: adopt HunyuanVideo as the default quality layer.** The gap is night-and-day.

## 4. Honest caveats

- **Speed/VRAM:** Hunyuan ~13–16 s/step × 30 + VAE decode ≈ **~8 min for a ~2 s 720p clip**, and it
  needs the GPU largely to itself (~49 GB active; the resident 85 GB WAN worker had to be evicted to
  avoid OOM). Slower + heavier than WAN.
- **Prompt adherence:** it rendered a Shiba/Akita-type dog, not specifically a *border collie* — scene
  quality is excellent but fine-grained subject control needs prompt work (Layer-3 skill).
- **Long-form:** multi-shot >5 s still needs `HunyuanVideo-I2V` chunking (drift risk remains).
- **License:** Tencent community license (region/MAU limits) — stricter than WAN's Apache-2.0.

## 5. Decision / next

1. **Hunyuan = default quality provider** (single-clip T2V hero path); WAN demoted to fast/draft +
   fallback. Both stay (the four-model provider design).
2. **Productize:** register `hunyuan_video` as a `video_generation` provider so the agent/
   `video_selector` routes to it for quality; resolve the VRAM tradeoff (can't keep WAN-14B (85 G) +
   Hunyuan resident — load the quality model on demand or dedicate the box to Hunyuan).
3. Wire HunyuanVideo-I2V for long-form continuity; improve prompting for subject fidelity.
