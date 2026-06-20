# ADR 0003 — Can Kakeya's verifier / drafter / f_θ be re-pointed at WAN? (objective evaluation)

- **Status:** Evaluation (informs the Phase-2b-for-video plan; no code change)
- **Date:** 2026-06-20
- **Deciders:** OpenMontage maintainers
- **Related:** ADR 0001 (Kakeya text), ADR 0002 (video gateway + architecture diagram)
- **Question:** In the planned integration, can the three Kakeya components
  (Verifier, Drafter/Proposer, f_θ) be replaced by WAN-corresponding versions —
  i.e. *WAN as verifier + an aligned WAN drafter + a corresponding f_θ* — and does
  that require WAN to have an **aligned drafter**?

---

## 1. Why this is not a "component swap"

Kakeya's three components are not generic building blocks; they are bound to **one
paradigm**: **autoregressive (AR) token generation with a causal KV cache and
lossless speculative decoding (rejection sampling).**

- **Verifier** = the ground-truth AR model; in spec-decode it verifies K draft
  tokens in one parallel forward and accepts/rejects so the output distribution is
  *identical* to the verifier alone (rejection sampling).
- **Drafter/Proposer** = a cheap model proposing K next tokens; the win exists only
  because (a) verifying K tokens in parallel is cheaper than K sequential passes and
  (b) drafts are often accepted.
- **f_θ** = a trained projection that **restores evicted causal-KV entries** so a
  bounded sliding-window KV preserves recall.

All three presuppose: a **token-sequential** dependency, a **growing causal KV
cache**, and a **discrete accept/reject** rule. The question is whether WAN provides
any of these.

## 2. The decisive fact about WAN

WAN 2.1 is a **bidirectional Diffusion Transformer** (flow-matching; T5 cross-attention;
1.3B/14B) — confirmed from the model card. Generation is **iterative denoising of the
whole latent clip in parallel** over ~20–50 steps; each step is a full forward over all
spatio-temporal latent patches with **bidirectional** attention.

Consequences for the three components, on **vanilla (bidirectional) WAN**:

| Kakeya component | Object it needs | Present in vanilla WAN? | Verdict |
|---|---|---|---|
| Verifier (AR accept/reject) | token-sequential decode + discrete acceptance | **No** (parallel latent denoise; continuous latents) | no analog |
| Drafter (spec-decode) | a thing to "verify K of" | **No** | nothing to draft/accept |
| f_θ (KV restoration) | a **growing causal KV cache** to evict/restore | **No** (bidirectional, no causal KV across steps) | nothing to restore |

The WAN bottleneck is the **number of denoising steps**, not token-sequential latency.
So re-pointing Kakeya's trio at vanilla WAN is a **category mismatch**, not a swap.

> Note: WAN's *VAE* is "3D causal" (temporal causality in the autoencoder). That is
> unrelated to a DiT attention KV cache; it does not give f_θ an object to restore.

## 3. The only regime where a KV-cache/f_θ analog appears: autoregressive WAN

There **is** a way to give WAN a causal KV cache — but it is reached by **distillation**,
not by porting Kakeya:

- **CausVid** ([arXiv 2412.07772](https://arxiv.org/abs/2412.07772)) distills the
  bidirectional WAN teacher into an **autoregressive causal student** that, "similar to
  decoder-only LLMs, achieves efficient autoregressive inference through **key-value (KV)
  caching**," using **asymmetric distillation + DMD**, 4 denoising steps.
- **Self-Forcing** ([arXiv 2506.08009](https://arxiv.org/abs/2506.08009), "Self-Forcing
  Wan 2.1") does AR rollout during training, **KV caching** at inference, real-time on a
  single RTX 4090.
- **WorldScape** uses Self-Forcing-style distillation with a **monotonic / sliding-window
  KV cache** and reports ~**8.7×** from a 4-step AR student.

In this regime the analogs exist — "verifier" ≈ teacher/full pass, a cheap chunk
"proposer," and a **bounded monotonic KV** that is the legitimate video cousin of
Kakeya's bounded-KV + restoration. **But the speedup comes from the distillation itself,
not from an inference-time draft-verify loop**: the fast student *replaces* the teacher;
there is **no separate verifier doing rejection sampling and no f_θ at serving time.**

A *true* speculative draft-verify with alignment does exist — but only for diffusion
**language** models: **DiffuSpec** ([arXiv 2510.02358](https://arxiv.org/abs/2510.02358))
and **SimSD** use a dLLM drafter + AR verifier with causal-consistency path search and
rejection-sampling acceptance. The continuous-latent acceptance rule needed to do this
for a **video DiT** has **no established equivalent** — it would be new research.

## 4. Direct answers

**Q: Can we replace verifier/drafter/f_θ with WAN versions?**
Not as a substitution. On bidirectional WAN, the drafter (spec-decode) and f_θ
(KV-restoration) have **no object to act on**. Only by first converting WAN into an
AR causal student do KV/f_θ-like ideas become meaningful — and that conversion is a
heavy DMD distillation, which already delivers the speed **without** a runtime
verifier or f_θ.

**Q: Does WAN need an *aligned drafter*?** Depends on the acceleration chosen:

| Acceleration path | Aligned drafter? | Verifier at inference? | f_θ / KV-restore? | Maturity / cost |
|---|---|---|---|---|
| **Feature caching** (TeaCache, DeepCache — reuse DiT features across steps) | **No** | No | No | Training-free, ~1.5–2.3×; TeaCache already supports CogVideoX/LTX |
| **Step distillation** (CausVid / Self-Forcing / DMD / Turbo) | The **distilled student is the "aligned" artifact** — there is no separate drafter | No (student replaces teacher) | Only as the student's own monotonic KV | Distillation = heavy training, **already done** for Wan2.1-T2V-1.3B → just serve the checkpoint |
| **True spec draft-verify on video latents** (Kakeya-faithful) | **Yes** — *and* you must invent continuous-latent verification | Yes | Yes | Research-frontier; nonexistent for video DiTs |

So: **for the realistic paths you do NOT need a separate aligned drafter.** Either no
drafter at all (caching), or the drafter and model **collapse into one distilled
student** (distillation), whose "alignment" is the distillation pipeline itself —
already performed upstream for Wan2.1-1.3B.

> **Refinement:** the maintainer later specified a more precise architecture —
> distilled-WAN *proposer* → low-res framework → **parallel full-WAN refiners** on
> different regions → merge, with f_θ as a decompose/merge consistency map. That is a
> *coarse-to-fine + tiled super-resolution* design (not spec-decode), evaluated in
> **ADR 0004**. It still does not port Kakeya's AR trio, but it is the right mental
> model for "parallel video inference" and is the basis for Phase-2b-for-video.

## 5. Recommendation for Phase-2b-for-video

Do **not** re-create Kakeya's verifier/drafter/f_θ on WAN. Instead, in the gateway:

1. **Tier 1 (now, low risk):** integrate **feature caching** (TeaCache/DeepCache) — no
   drafter, no alignment, model-preserving, ~1.5–2.3×. Empirically testable today on the
   live H200 gateway (CogVideoX/LTX are TeaCache-supported).
2. **Tier 2 (medium, no training by us):** serve **distilled few-step WAN** (CausVid /
   Self-Forcing checkpoints for Wan2.1-T2V-1.3B) as additional `model` ids — the
   "alignment" is the upstream distillation; OpenMontage just hosts the checkpoint. Large
   speedup (real-time class).
3. **Tier 3 (not recommended as a deliverable):** speculative draft-verify on video
   latents — requires a purpose-built aligned drafter **and** new verification theory.

Keep **f_θ / bounded-KV restoration on the text path** (Kakeya gRPC, ADR 0001 Phase 2b),
where the AR token paradigm makes it correct. For video, the honest analog is the
**monotonic/sliding-window KV of an AR distilled video model** — reached by distillation,
not by porting Kakeya's trio.

## 6. Consequence for the architecture diagram

The `kakeya_llm -.-> gRPC (Drafter+Verifier+f_θ)` edge in ADR 0002 §3a is a **text-only**
construct. It must **not** be redrawn as feeding the video gateway: the video plane's
"Phase 2b" is *caching + distilled-student serving*, a separate mechanism. This ADR is the
reference for that distinction.

## References

1. Wan 2.1 — bidirectional DiT model card: https://github.com/Wan-Video/Wan2.1
2. CausVid — https://arxiv.org/abs/2412.07772
3. Self-Forcing (Wan 2.1) — https://arxiv.org/abs/2506.08009
4. TeaCache (CVPR 2025) — https://github.com/ali-vilab/TeaCache
5. DeepCache — https://horseee.github.io/Diffusion_DeepCache/
6. DiffuSpec (spec-decode for diffusion *language* models) — https://arxiv.org/abs/2510.02358
