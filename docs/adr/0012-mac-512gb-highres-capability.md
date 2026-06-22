# ADR 0012 — What high-res can a ~512 GB Apple-Silicon Mac actually do?

**Status:** Accepted (analytical estimate; not yet benchmarked on the owner's box).
**Date:** 2026-06-22
**Related:** ADR 0008 (WAN-on-MLX feasibility), ADR 0013 (re-anchored architecture).
**Why:** owner feedback — re-evaluate whether the Mac can be a *high-res* path, or only low-res
drafts, given a ~512 G configuration (and only attach vast when it can't).

## 1. The earlier "low-res only" verdict was a small-Mac artifact

The OOM we hit (`832×480×25` → Metal "Insufficient Memory") happened on a **small** Mac
(16–64 GB unified). That is a *memory* limit, not a model limit. A ~512 GB Apple-Silicon machine
(M3 Ultra Mac Studio tops out at **512 GB** unified, ~819 GB/s; or two Thunderbolt-bridged Macs)
**removes that ceiling**:

- WAN 2.1 **14B** weights ≈ 28 GB (bf16); umt5-xxl T5 ≈ 11 GB; VAE small. A `1280×720×81` decode
  with VAE tiling is well within 512 GB — by a wide margin.
- So **memory is no longer the constraint for high-res** on this box. The earlier "drafts only"
  framing does **not** apply to 512 G.

## 2. The real constraint is GPU compute time (not resolution)

Apple's GPU is graphics-tuned and far slower than a datacenter NVIDIA part. Public benchmarks:

| Workload | M3 Ultra (Mac) | RTX 4090 | H200 (our vast) |
|---|---|---|---|
| WAN 2.2, 5 s, **480p** | ~11 min | ~2m40s | ~1–2 min |
| (14B @ 720p, full steps) | extrapolated ~20–40+ min | — | ~minutes |

So the 512 G Mac **can** render high-res (720p, even 14B) — it's a *capability* it has — but at
**minutes-to-tens-of-minutes per clip** at full step counts.

## 3. The lever that makes Mac high-res practical: few-step distillation

A CausVid / Self-Forcing LoRA cuts inference steps from ~30–50 to **4–8** (≈5–6× fewer), which we
already use for the proposer. Applied to a 512 G Mac running 14B @ 720p, that brings a hero clip
from "tens of minutes" down to **a few minutes** — acceptable for our **async** API (submit→poll).
`mlx-video`'s `generate` supports `--lora PATH STRENGTH`, so this is a config knob, not new code.

## 4. Honest capability tiers for the 512 G Mac

| Tier | Model / res / steps | Est. latency | Use |
|---|---|---|---|
| Draft | WAN 1.3B, 480p, few-step | ~1–2 min | previews, iteration |
| Standard | WAN 14B, 720p, few-step LoRA | ~3–8 min | default deliverable |
| Hero (Mac-only) | WAN 14B, 720p, full steps | ~20–40+ min | best quality, patient |
| Hero (accelerated) | Mac proposer + **on-demand vast** refine | ~1–3 min | when latency matters |

## 5. Recommendation (updates ADR 0008 / my earlier "Mac = drafts only")

- **The 512 G Mac IS a legitimate high-res generator** (720p / 14B). Reposition it from "low-res
  drafts only" to "high-res capable, latency-bound." Memory does not force low-res here.
- **Default:** Mac does Standard tier (14B @ 720p, few-step LoRA) — high-res, a few minutes, async.
- **Attach vast on-demand for SPEED, not for resolution:** when a request needs fast turnaround or a
  final refine polish, set `VAST_REFINE_WORKER` to a freshly-spun-up vast box; detach it otherwise.
  This matches the owner's "no co-located CUDA; temporary vast for refine."
- **Verify on the real box:** these are estimates. First action when the box is reachable — run
  `mlx_video.models.wan_2.generate` at 14B/720p (few-step) and record real latency to fill in §2/§4.

## 6. Caveats

- `mlx-video` must convert/run the 14B WAN (its config supports `wan21_t2v_14b`; conversion is
  heavier and quantization `--quantize --bits 4/8` can halve/quarter weight memory and speed decode).
- Thunderbolt-bridged-pair memory is **not** auto-pooled for one generation (ADR 0011 / Iteration 23
  — `mlx-video` has no sharding). A single high-res generation must fit on **one** Mac; with a
  512 G single machine that's fine, with two smaller bridged Macs it is not.
