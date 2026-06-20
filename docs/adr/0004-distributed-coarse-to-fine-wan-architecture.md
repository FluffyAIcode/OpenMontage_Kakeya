# ADR 0004 — Distributed coarse-to-fine WAN (proposer→parallel tiled verifiers→merge): objective evaluation

- **Status:** Evaluation (refines the Phase-2b-for-video direction; no code change yet)
- **Date:** 2026-06-20
- **Deciders:** OpenMontage maintainers
- **Supersedes framing of:** ADR 0003's "spec-decode-on-video" rejection — this is the
  *accurate* architecture the maintainer intended.
- **Related:** ADR 0002 (video gateway), ADR 0003 (why Kakeya's AR trio doesn't port)

---

## 1. Accurate restatement (the intended architecture)

> "proposer = a distilled small model of Wan2.1; Wan2.1 itself = verifier; use Kakeya's
> parallel inference so the proposer first generates a **low-resolution content
> framework**, then **simultaneously dispatches to multiple verifiers** that complete
> **different regions** to high resolution; then **merge** into the final video. f_θ
> performs the **mapping during decompose/merge** of the low-res framework so regions do
> not stitch wrong."

Named accurately, this is:

**Cascaded coarse-to-fine generation + distributed spatial/temporal *tiled* generative
super-resolution/refinement + a boundary-consistency (decompose↔merge) map.**

This is a real, established family — **not** the LLM spec-decode paradigm ADR 0003 ruled
out. It is a *correct* use of "parallel inference." This ADR evaluates it objectively.

## 2. Mapping to real, current systems

| Your component | Real technique it is | Reference |
|---|---|---|
| Proposer = distilled WAN → low-res framework | few-step distilled WAN (coarse base of a cascade) | CausVid (2412.07772), Self-Forcing (2506.08009) |
| Verifier = full WAN refines a region → high-res | generative (space-time) super-resolution / refinement conditioned on the low-res prior | **VEnhancer** (2407.07667) — ControlNet on a frozen video prior |
| Many verifiers, different regions, in parallel | patch/tile-parallel video diffusion | **PatchVSR** (2509.26025), DistriFusion, AsyncDiff, PipeFusion, **SuperGen** (2508.17756) |
| f_θ = decompose/merge mapping (anti-stitch) | boundary consistency: overlap averaging, **spatial weight maps**, **tile shifting** | **MultiDiffusion** (2302.08113), PatchVSR, SuperGen |

## 3. Objective findings

### 3.1 The proposer MUST be aligned (answers the running question)

For coarse-to-fine to work, the distilled proposer's low-res framework must be a
**faithful coarse sample of full-WAN's distribution**. If it is misaligned, the high-res
refiners "fight" the layout and hallucinate region-inconsistent detail. So **yes — an
aligned (distilled-from-WAN) proposer is genuinely required** in this architecture
(unlike the training-free feature-caching path in ADR 0003). Distilled Wan2.1-1.3B
checkpoints already exist (CausVid / Self-Forcing), so the alignment work is largely
upstream — OpenMontage serves the checkpoint.

### 3.2 The dominant 硬伤: full WAN is NOT a native tile-verifier

PatchVSR states it plainly: *"pre-trained video diffusion models are not native for
patch-level detail generation."* Making "full WAN refine a high-res region conditioned on
the low-res framework" requires **one of**:

- **A.** a **trained conditioning adapter** — VEnhancer-style ControlNet copying the WAN
  prior's encoder/middle block and training it to accept (low-res frames + noisy
  latents). **Extra training**, but principled and high quality; or
- **B.** **training-free tiled diffusion** (MultiDiffusion / PatchVSR) — no training, but
  with *documented* limits: naive overlap averaging yields **"black holes or seamlines,"**
  and fixed tiles **"fail to maintain temporal consistency in video."**

Either way, "WAN as verifier" is **not a drop-in** — it is a conditioning/adaptation
problem, not a config change.

### 3.3 f_θ here is a NEW artifact, not Kakeya's f_θ

Kakeya's f_θ restores a **causal-token KV cache** (ADR 0003). Your f_θ is a **spatial/
temporal decompose↔merge consistency map** — same name, different math and objective.
Today the *proven* mechanisms are heuristic:

- overlap regions + averaging (MultiDiffusion),
- **spatial weight maps** that down-weight auxiliary patches toward boundaries (PatchVSR),
- **deterministic tile shifting** across timesteps so seams at step *t* are corrected at
  *t+1* (SuperGen / SpotDiffusion).

Whether a **learned** f_θ beats these heuristics is an **open research question**, not a
solved component. Recommendation: treat heuristic blending as the baseline and only invest
in a learned f_θ if measured seams/temporal-flicker are unacceptable.

### 3.4 Not lossless (the "verifier" does not verify)

LLM spec-decode is distribution-preserving (rejection sampling → identical output).
Coarse-to-fine + tiling is a **quality/speed tradeoff**: the merged high-res result is
**not** guaranteed to equal monolithic full-res WAN. The "verifier" *refines*, it does not
*verify* in the lossless sense. Acceptable for a video tool, but it must be stated — there
is no correctness guarantee transferred from Kakeya.

### 3.5 Real wall-clock parallelism needs multiple GPUs

"Many verifiers in parallel" only reduces wall-clock if regions run on **separate GPUs**
(DistriFusion/AsyncDiff/PipeFusion partition patches across devices). On the current
single H200 (144 GB) it degenerates to **batched tiles on one device** — a throughput win,
not a latency win. Multi-GPU is where Kakeya's distributed transport *could* contribute.

### 3.6 What Kakeya actually contributes here (honest)

Not its verifier/drafter/f_θ **math** — those are AR-token constructs (ADR 0003). What is
reusable is Kakeya's **distributed execution fabric**: multi-tenant scheduling, worker
placement, and tensor transport (`distributed/{placement,exchange,tensor_codec}.py`) — as
the **scheduler that fans tile-refinement jobs to workers and gathers/merges results**.
Caveat: Kakeya's distributed plane is spec-decode-specific and partly **design-only** (its
ADR 0014), so realistically only the low-level transport primitives transfer; the
tile-scheduler itself is net-new.

## 4. Verdict

The architecture is **coherent and buildable**, and is the *right* mental model for
"parallel video inference" (far better than spec-decode-on-video). But it is **not a port
of Kakeya's trio** — it reuses the names with diffusion semantics, borrows at most
Kakeya's *distributed transport*, and has two genuine costs:

1. **Conditioning full WAN to refine tiles** (train a VEnhancer-style ControlNet, or accept
   training-free tiling's coherence limits), and
2. **The f_θ consistency map** (heuristic now; learned = research).

Plus: it is **not lossless**, and **needs multi-GPU** for true parallel speedup.

## 5. Staged validation plan (empirical, testable on the live H200)

Build bottom-up so each tier proves the next is worth it. No mocks; real models (ADR 0002 §0).

- **Tier 0 — coarse-to-fine, no tiling (cheapest).** distilled-WAN low-res → whole-frame
  generative refine (SDEdit/img2img, or VEnhancer). Measure quality + speed vs monolithic
  full-res WAN. Proves the proposer→verifier flow and the alignment requirement (§3.1).
- **Tier 1 — parallel tiling + seam handling.** Overlap tiles + spatial-weight-map blending
  (PatchVSR/MultiDiffusion-style) on one H200 (batched tiles). **Measure seams + temporal
  flicker** — this empirically decides whether a learned f_θ (§3.3) is even needed.
- **Tier 2 — learned consistency map (only if Tier-1 seams unacceptable).** Train the f_θ
  analog. Research effort; gate on Tier-1 evidence.
- **Tier 3 — multi-GPU distribution (only with >1 GPU).** Fan tiles across devices via
  Kakeya transport; measure wall-clock scaling.

Each tier emits real artifacts (mp4 + ffprobe + seam/flicker metrics) recorded in the loop
log, per the no-fake/no-simplify guideline.

## 6. Real-GPU results — Tier 0 + Tier 1 (measured, H200)

Run on the live H200 with **CogVideoX-2b** (a DiT video model; WAN weights did not fit
the 3.2 GB free disk alongside the running gateway — the *mechanism* is what is
validated, and per §1–3 it transfers to WAN). Script:
[`services/video_infer_gateway/experiments/tier01_coarse_to_fine_tiling.py`](../../services/video_infer_gateway/experiments/tier01_coarse_to_fine_tiling.py).
Raw metrics: [`tier01_evidence/metrics.json`](tier01_evidence/metrics.json).

### Tier 0 — coarse-to-fine (proposer → verifier-refine vs monolithic)

| Metric | Value | Reading |
|---|---|---|
| proposer (8-step T2V) | 10.6 s | the cheap "framework" pass |
| verifier (vid2vid, strength 0.6, 24-step) | 17.3 s | high-detail completion |
| coarse-to-fine total | 27.9 s | |
| monolithic baseline (40-step) | 37.8 s | |
| **wall-clock speedup** | **1.35×** | modest on ONE GPU with a NON-distilled proposer |
| align refined↔coarse (NCC) | **0.932** | refine **preserves** the proposer's layout → the alignment premise (§3.1) holds |
| refined↔monolithic (PSNR) | 13.2 dB | **not the same sample** → confirms §3.4: coarse-to-fine is *not lossless* |

**Conclusion:** the proposer→verifier flow works and is layout-faithful (high NCC), but
single-GPU speedup with a non-distilled proposer is only ~1.35×. The real speedup needs
(a) a **distilled few-step proposer** (much cheaper framework pass) and (b) **multi-GPU
parallel verifiers** — exactly §3.1 + §3.5.

### Tier 1 — decompose → independent tile refine → merge (the f_θ question)

2×2 native-720×480 tiles, 160 px overlap, 1280×800 canvas; each tile refined
independently (vid2vid, strength 0.5, 20-step); 4 tiles in **55.7 s** (13.9 s/tile,
serial on one GPU → parallelizable on multi-GPU).

Seam discontinuity at the **true tile edges**, normalized by interior texture
(>1 = visible stitch; ~1 = invisible):

| Merge | vertical seam | horizontal seam | verdict |
|---|---|---|---|
| **Hard** (naive overwrite) | **5.24×** | **5.12×** | glaring stitch lines — the 拼接错误 |
| **Blended** (spatial weight-map f_θ) | **0.85×** | **1.15×** | seam ≈ ordinary texture |
| reduction | **−84 %** | **−78 %** | |

Visual evidence (mid-frame): hard merge shows obvious rectangular seams; blended is
coherent.

![hard merge — visible seams](tier01_evidence/t1_hard_mid.png)
![weight-map blended — seams gone](tier01_evidence/t1_blended_mid.png)

**Conclusions (honest):**

1. **拼接错误 is real and severe** without a merge map: naive hard merge = **~5× seams**.
2. A **heuristic f_θ** (spatial-weight-map blending, MultiDiffusion/PatchVSR-style)
   removes ~**80 %** of the seam on this content — so a *learned* f_θ is **not required
   for low-frequency content**.
3. **Caveat (content-dependent):** this clip is low-frequency (water/light). Where
   independently-refined tiles **hallucinate divergent structure** (faces, text, hard
   edges crossing a boundary), post-hoc pixel averaging **ghosts** rather than fixes —
   which is why the robust solution is **denoise-time fusion** (MultiDiffusion in latent
   space, fusing every step) **or a learned f_θ**. The "refine fully, then merge" order
   the proposal implies is the *fragile* case; consistency should act **during** denoise.
4. **Per-tile cost is serial here** (one GPU); the parallel-verifier speedup needs
   multi-GPU (§3.5).

This empirically confirms ADR 0004's verdict: the architecture is sound; the f_θ role is
**necessary** (hard merge is unacceptable) and **partly solvable by a heuristic**, with a
learned/denoise-time version reserved for hard content.

## References

1. VEnhancer — https://arxiv.org/abs/2407.07667 ; https://github.com/Vchitect/VEnhancer
2. PatchVSR — https://arxiv.org/abs/2509.26025
3. MultiDiffusion — https://arxiv.org/abs/2302.08113
4. SuperGen (tiling + tile-shift, distributed) — https://arxiv.org/abs/2508.17756
5. CausVid — https://arxiv.org/abs/2412.07772 ; Self-Forcing — https://arxiv.org/abs/2506.08009
