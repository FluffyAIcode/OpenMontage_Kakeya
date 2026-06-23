# Kakeya Integration — Iterate → Test → Improve Loop Log

Companion to `docs/adr/0001-kakeya-llm-inference-integration.md`. This log records
the develop → test → collect-issues → analyze → improve cycles, as the task asks.

---

## Iteration 1 — Phase 1 scaffolding

**Built**

- `tools/text/kakeya_llm.py` — `text_generation` provider talking to a user-run
  Kakeya OpenAI-compatible HTTP server. Operations: `generate`, `batch`, `health`.
- `tools/text/llm_selector.py` — capability router (auto-discovers
  `text_generation` providers, mirrors `tts_selector`).
- `tests/tools/test_kakeya_llm.py` — 15 offline tests using a stdlib mock server.

**Tested**

- Registry discovery picks up both tools; both report `unavailable` with no
  `KAKEYA_ENDPOINT` (default users unaffected) and `available` once it is set.
- generate / batch / health round-trips, response parsing, input validation.

**Issues collected**

| ID | Issue | Severity |
|----|-------|----------|
| I1 | Selector returned a dead-end *"No available provider matched the request"* when `kakeya_llm` was registered but unavailable (endpoint unset). Unhelpful — didn't tell the user how to fix it. | UX |
| I2 | `get_status()` must not perform a network probe (called repeatedly during preflight). Confirmed env-only; reachability deferred to `execute()`. | design (resolved by design) |
| I3 | Client-side `batch` fan-out is sequential, so the CUDA batched-scheduler throughput win (W2) is only realized if the *server* absorbs concurrent in-flight requests. Sequential client calls leave that on the table. | perf (deferred) |

**Analyzed + improved**

- **I1 → fixed:** selector now returns an actionable message ("Set KAKEYA_ENDPOINT
  …", lists registered providers) whenever no provider is *available*, not just
  when none is *registered*. Re-tested: all 15 pass.
- **I2:** kept env-only status by design; documented the rationale in the tool.
- **I3:** deferred to Phase 2 (see below). Documented honestly in the tool's
  `supports` metadata (`high_throughput_batching_requires_gpu_server: True`) so the
  agent never over-promises a speedup the client path can't deliver alone.

**Result:** 15/15 tests pass. Default OpenMontage behavior unchanged (tools dormant
until configured).

---

## Iteration 2 — concurrent batch fan-out (resolves I3)

**Why:** The headline benefit of "distributed parallel inference" is *throughput via
concurrency*. A sequential client loop never puts more than one request in flight, so
Kakeya's batched scheduler (the thing that delivers the 8.45× number) would have
nothing to batch. Iteration 1's batch path left W2 entirely on the table.

**Built**

- `kakeya_llm` `batch` now fans prompts out through a bounded `ThreadPoolExecutor`
  (`concurrency`, default 8 to match Kakeya's 8-session sweet spot, hard cap 64).
- Order is preserved by index; per-item errors stay isolated; usage is aggregated.
- `_resolve_concurrency()` clamps to `[1, min(requested_cap, n_prompts)]`.

**Tested (+3 tests, 18 total)**

- Order preserved under concurrency=8 across 20 prompts.
- Concurrency clamping (over prompt count → n; <1 → 1; over cap → 64; garbage → default).
- **Actual parallelism:** a 0.1s-per-call stub × 8 prompts finishes < 0.5s
  (sequential would be ~0.8s), proving requests overlap.

**Issues collected**

| ID | Issue | Severity | Disposition |
|----|-------|----------|-------------|
| I4 | `concurrency=1` must remain available as an escape hatch for servers with `--capacity 1` (single-tenant CPU shim) to avoid 429s. | correctness | Handled: `concurrency=1` takes the sequential path; users set it for capacity-1 servers. |
| I5 | Client concurrency > server capacity will get 429s from Kakeya's admission control; we surface them per-item rather than crashing, but we can't auto-discover server capacity over the HTTP shim. | perf/UX | Documented; Phase 2b gRPC `GetSessionInfo` could expose capacity for auto-tuning. |

**Result:** 18/18 tests pass. W2 is now actually reachable on a GPU server; the tool's
`supports` metadata remains honest that the *win itself* still depends on a GPU server.

---

## Iteration 3 — course correction: unify the LOCAL VIDEO models (ADR 0002)

**Trigger:** maintainer clarified the real intent — not to replace an OpenMontage
module, but to run the four open-source video models (WAN 2.1, Hunyuan, CogVideo,
LTX) *on a unified inference engine* serving OpenMontage's orchestration + video
stream, ideally Kakeya.

**Investigated (code-level, not README-level)**

- Read Kakeya's `runtime.proto`, `backends/`, `bridge/`, and grepped for
  diffusion/video/vae/latent. **Finding (硬伤):** Kakeya is an LLM *token* engine —
  token-only gRPC contract, "diffusion" = text-diffusion LLM, features are AR-decode
  concepts. It has **no path** to host DiT video-diffusion models. The one
  "multimodal" note is image/audio *input* to an LLM, output still tokens.
- Read OpenMontage's `tools/video/_shared.py`: all four tools share
  `generate_local_video()` which does `Pipeline.from_pretrained()` **per call** —
  cold load every time, no warm reuse / VRAM pool / batching. Real inefficiency.
- Found the right precedent already in-repo: `generate_ltx_modal_video()` routes to a
  standalone HTTP inference server (`MODAL_LTX2_ENDPOINT_URL`).

**Decided (ADR 0002):** decouple the goal (unified warm backend) from the wrong
mechanism (Kakeya). Add an engine-agnostic **video-inference gateway** seam; the four
tools route to one warm server over HTTP (`VIDEO_INFER_ENDPOINT`), falling back to
in-process diffusers. Kakeya stays the **text** engine (ADR 0001). The honest
"unification" is a two-engine local inference plane (text→Kakeya, video→diffusion
gateway), both behind existing selectors.

**Built**

- `tools/video/video_infer_client.py` — gateway client + ADR 0002 §5 contract
  (`/healthz`, `POST /v1/video/generations`; accepts raw mp4 / video_url / video_b64).
- Routing seam in `_shared.generate_local_video()` (short-circuits before any torch
  import). `local_generation_status()` now reports AVAILABLE when a gateway is set —
  **without** a local torch/diffusers install.

**Tested (+11, suite now 363 passed)**

- Endpoint resolution, availability widening, health, all three response shapes,
  i2v image_b64 encoding, unknown-variant + unsupported-i2v errors, and an
  **end-to-end routing test that passes with torch NOT installed** (proves the seam
  short-circuits the in-process path).

**Issues collected**

| ID | Issue | Disposition |
|----|-------|-------------|
| I6 | We don't ship the gateway *server* (needs GPU; can't CI here). | Documented; client+contract+tests shipped now, reference server is a follow-up PR. |
| I7 | Client-side has no cross-model batching; the throughput win lives *inside* the gateway (warm pool + admission). | By design — gateway owns batching; OpenMontage stays a thin client. |

---

## Iteration 4 — REAL GPU run: de-mock everything (H200, 144 GB)

**Trigger:** GPU provisioned (vast.ai **H200 NVL, 144 GB VRAM**, CUDA 13.2). Directive:
replace every GPU-absent mock with real implementation + tests; merge a binding
"no fallback / mock / fake / simplify" guideline (ADR 0002 §0, ADR 0001 §0).

**Built / deployed (all real)**

- Installed real CUDA torch (`2.6.0+cu124`, `cuda.is_available()=True`) + diffusers.
- **Wrote the real video gateway server** (`services/video_infer_gateway/server.py`):
  FastAPI + diffusers warm pool implementing ADR 0002 §5. Deployed on the H200.
- Ran a **real Kakeya HTTP shim** (`scripts/serve.py`, Qwen3-0.6B) on the H200.
- SSH-tunnelled VM → GPU; pointed `VIDEO_INFER_ENDPOINT` / `KAKEYA_ENDPOINT` at them.
- Added `tests/integration/test_real_gpu.py` — the **binding correctness gate**;
  env-gated, **skips (never fakes)** when endpoints are absent.

**Real evidence (measured, not targets)**

| Engine | Real result |
|--------|-------------|
| Video gateway | `CogVideoX-2b`, 720×480, 49 frames, 20 steps → **1.36 MB mp4 in 21.4 s on H200**; ffprobe: **h264 720×480, 49 frames**. End-to-end through `CogVideoVideo.execute()` → gateway (`mode=remote_gateway`). |
| Video health | `{"status":"ok","device":"cuda",...}` advertising all 8 model ids. |
| Kakeya text | real completion (`'PING.'`), `usage.completion_tokens≥1`; sequential batch **4/4** real completions. |

**Issues collected (real testing)**

| ID | Issue | Severity | Disposition |
|----|-------|----------|-------------|
| I8 | **Kakeya HTTP shim is single-session**: concurrent requests (`concurrency>1`) return **HTTP 500** (verified: 1/4 ok, 3/4 → 500). Its batched/throughput path is gRPC+CUDA, not the shim. | correctness | **Fixed:** changed `kakeya_llm` default batch concurrency **8 → 1** (safe for the only stable transport). Raising it is documented as requiring a concurrency-capable backend. Added a real test that *documents* the 500 behaviour and asserts the tool isolates it (no crash/fake). |
| I9 | H200 box has only ~23 GB free disk (no large data volume) → can hold **one** video model's weights at a time (T5/UMT5 text encoders dominate). | env | Hardware limit, reported honestly. Gateway design already routes by model id; more disk → more warm models. Not a simplification. |

**Honest correction to Iteration 2:** the concurrent client fan-out (added as "the W2
win") is correct *client* behaviour and genuinely helps a concurrency-capable gRPC/CUDA
backend — but it is **counterproductive against the HTTP shim** (500s). Real testing
forced the default back to sequential. The throughput win remains **gRPC-path-gated**
(Phase 2b), now backed by evidence rather than assumption.

**Result:** 6/6 real integration tests pass on live GPU; 29 offline smoke tests pass;
mock tests explicitly demoted to "not the correctness gate" (ADR 0002 §0).

---

## Iteration 5 — Tier 0 / Tier 1 real-GPU experiments (ADR 0004 §6)

**Trigger:** "proceed tier 0 and tier 1 on the GPU real environment."

**Built / ran (real, H200, CogVideoX-2b — WAN weights didn't fit 3.2 GB free disk; the
mechanism transfers per ADR 0004 §1–3):**
`services/video_infer_gateway/experiments/tier01_coarse_to_fine_tiling.py`.

**Tier 0 (coarse-to-fine):** proposer 8-step (10.6 s) → vid2vid verifier 24-step (17.3 s)
= 27.9 s vs monolithic 40-step = 37.8 s → **1.35× wall-clock** (single GPU, non-distilled
proposer). Refine↔coarse NCC **0.932** (layout preserved → alignment holds); refined↔full
PSNR 13.2 (**not lossless**). Confirms: real speedup needs a distilled proposer + multi-GPU.

**Tier 1 (decompose → independent tile refine → merge):** 2×2 native tiles, 160 px overlap.
Corrected seam metric (measured at true tile edges): **hard merge = 5.2×/5.1× interior
texture (glaring seams)**; **weight-map blend = 0.85×/1.15× (≈80 % reduction, visually
gone)**. Visual evidence committed (`tier01_evidence/`).

**Issues collected**

| ID | Issue | Disposition |
|----|-------|-------------|
| I10 | First in-script seam metric sampled the overlap **centre** (where the hard merge is coincidentally continuous because the later tile overwrites it) → reported blending as *worse*, contradicting the obvious visual seams. | **Fixed:** metric now measures at **true tile edges**; recomputed on the saved real frames → hard 5.2×, blended 0.85× (matches the eye). |
| I11 | Heuristic post-hoc blending fixes low-frequency seams but **ghosts divergent hallucinated structure** → robust fix is **denoise-time fusion (latent MultiDiffusion) or a learned f_θ**; the "refine fully then merge" order is the fragile case. | Recorded in ADR 0004 §6; gates whether a learned f_θ is worth building. |

**Result:** the proposer→parallel-verifier→merge architecture is empirically sound on real
GPU; f_θ is **necessary** (hard merge unacceptable) and **partly solvable by a heuristic**;
single-GPU speedup is modest (distilled proposer + multi-GPU needed for the real win).

---

## Iteration 6 — Tier 1b consistency test (resolves I11; ADR 0004 §6)

**Trigger:** "proceed to a distilled-proposer run or a multi-GPU/latent-fusion test."
Multi-GPU = impossible (one H200). Faithful distilled-WAN = disk-blocked (3.2 GB free).
So ran the **latent/consistency test** (disk-safe, resident CogVideoX) to resolve I11.

**Experiment:** high-frequency prompt; independent-seed tiles vs **shared-noise** tiles;
metric = cross-tile **overlap disagreement**.

**Result (real, H200):** shared noise did **NOT** reduce overlap disagreement
(v 3.63→4.02, h 3.81→2.58; net −11 %). The divergence is **context/position-driven**
(different crop, different 3D-RoPE positions, different global attention context), not
noise-driven — PatchVSR's "DiTs not native for patch-level," confirmed.

| ID | Finding | Disposition |
|----|---------|-------------|
| I12 | Neither post-hoc pixel blend (I11) nor shared-noise fixes cross-tile divergence on non-trivial content. f_θ/merge-consistency **must** be **denoise-time latent fusion** (latent MultiDiffusion) or a **learned** consistency model. | Strengthens the case for f_θ as a necessary component; per-step latent MultiDiffusion on CogVideoX's 3D-RoPE DiT is a research-grade build (future). |
| I13 | Multi-GPU + faithful distilled-WAN not runnable on this box (1 GPU, 3.2 GB disk). | Flagged as next steps needing more GPUs / freeing the gateway model + ~13 GB. |

**Result:** I11 resolved — the merge-consistency must act *during* denoising or be learned;
cheap levers (post-hoc blend, shared noise) are insufficient on hard content. This is a
precise, evidence-backed refinement of the f_θ requirement.

---

## Iteration 7 — faithful distilled-WAN Tier-0: BLOCKED by disk (I14)

**Trigger:** "free CogVideoX/Gemma and run the real distilled-WAN Tier-0 speedup."

**Done:** freed everything (CogVideoX already gone with the dead gateway; Gemma-4 26B was
**never** resident — only Qwen3-0.6B ran; freed it + pip cache) → 18 GB free / 32 GB disk.
Confirmed the diffusers pieces exist: `WanVideoToVideoPipeline` (native vid2vid) + a 1.3B
**CausVid** distill LoRA (`Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors`).
Wrote the ready experiment `services/video_infer_gateway/experiments/tier0_distilled_wan.py`
(monolithic WAN vs CausVid-distilled proposer + full-WAN verifier refine).

**Blocker I14 (hard, environmental):** the WAN 2.1 1.3B **diffusers** repo is **~29 GB**
because its **UMT5-XXL text encoder is fp32 = 22.7 GB**. Measured across 6 re-uploads —
all carry the fp32 encoder. A bf16 text encoder exists only in **original-Wan `.pth`**
format (11.4 GB; key layout ≠ diffusers, needs remapping, and converting it needs ~22 GB
transient). No clean drop-in **bf16 diffusers** WAN repo exists; no transformers-format
bf16 UMT5 encoder found. So the fp32 repo **does not fit** the 32 GB disk (18 GB free),
and the bf16 assembly is fragile (in-memory key remap of the original `.pth`).

**Decision (no fake):** do NOT relabel a CogVideoX run as distilled-WAN. **Escalate the
disk blocker.** The script is committed and ready.

**Unblock (either):**
1. **Resize the vast.ai instance disk to ≥ 64 GB** (or relaunch with a bigger disk) →
   then `tier0_distilled_wan.py` runs the faithful CausVid-distilled-WAN Tier-0 in ~5 min.
2. Authorize a **best-effort bf16 assembly** (techfreakworm bf16 diffusers transformer +
   official VAE + in-memory remap of the original bf16 UMT5 `.pth`) — fits 18 GB but the
   text-encoder remap may fail; higher risk, may burn GPU time.

| ID | Item | Status |
|----|------|--------|
| I14 | WAN-diffusers = 29 GB (fp32 UMT5 22.7 GB) > 32 GB disk; bf16 only in original format. | **Blocked** — needs ≥64 GB disk (clean) or a fragile bf16 remap. Experiment committed & ready. |

---

## Iteration 8 — faithful distilled-WAN Tier-0 UNBLOCKED + run (resolves I14)

**Trigger:** new GPU box provided (`104.202.252.41`) — **H200, 144 GB VRAM, 605 GB disk**
(vs the 32 GB box). I14 unblocked.

**Done:** installed torch/diffusers/peft; downloaded WAN 2.1 1.3B diffusers (~29 GB, fits
easily) + the 1.3B CausVid LoRA; ran `tier0_distilled_wan.py`.

**Real result (H200):** monolithic 30-step = 30.0 s; **distilled proposer (CausVid 6-step)
= 5.26 s → 5.71×**; verifier refine 15-step = 10.0 s; coarse-to-fine total = 15.24 s →
**1.97×**; refined↔coarse **NCC 0.966** (alignment holds on real WAN); CausVid 6-step
frame is high-quality (evidence committed). Headline: a **genuine distilled proposer is
5.7× cheaper** than monolithic — vs 1.35× for the non-distilled CogVideoX proposer
(confirms ADR 0004 §3.1).

**Bugs found + fixed (real testing loop):**

| ID | Issue | Fix |
|----|-------|-----|
| I15 | `WanVideoToVideoPipeline(**pipe.components)` failed: WanPipeline carries a `transformer_2` slot the vid2vid pipeline rejects. | Pass only its 5 expected modules. |
| I16 | WAN pipelines return **np frames** (not PIL like CogVideoX) → `.convert()` crashed. | `_np()` now robust to PIL / np.ndarray / float[0,1]. |

**Result:** I14 resolved on a real ≥64 GB-disk H200; the faithful distilled-WAN Tier-0
number is **5.7× (proposer) / 1.97× (coarse-to-fine)**, with strong layout alignment.

---

## Iteration 9 — latent-MultiDiffusion build (the denoise-time f_θ fix)

**Trigger:** "start latent-MultiDiffusion build." Built the production-grade merge fix
flagged by I12: a single shared canvas latent at larger-than-native resolution, denoised
with **per-step prediction fusion** over overlapping native-size tiles (real WAN 2.1 1.3B,
reimplemented denoise loop). `latent_multidiffusion_wan.py`.

**Result (real H200):** the build works — produces a **coherent 1472×768 video** (from 2×2
native 832×480 tiles) that WAN can't do natively in one pass. **Visually more coherent**
than independent-then-merge (independent shows an upper-third tonal band; MultiDiffusion is
one continuous scene). Vertical seam 1.27×→**1.14×** (~10 %); horizontal ~2.6× in BOTH is a
**real shoreline edge** (metric confounded by scene content). On low-frequency content both
overlap-blended methods are close.

| ID | Finding/bug | Disposition |
|----|-------------|-------------|
| (oom) | custom denoise loop lacked `torch.no_grad()` → autograd graph OOM'd at 140 GB. | Fixed: `@torch.no_grad()` + expandable_segments. |
| I17 | seam metric measured overlap-start/duplicate columns, not true tile boundaries. | Fixed: edges at `[x_off[1], WT]` / `[y_off[1], HT]`. |
| I18 | seam metric confounded by real scene edges near boundaries; low-freq content blends easily. | A definitive quantitative MD win needs high-freq content + a boundary-isolating metric (next). |

**Verdict:** latent MultiDiffusion is **implementable and correct on real WAN** and is the
right home for the f_θ/merge-consistency role (co-evolution beats post-hoc blend, visually).
Definitive quantitative advantage pending high-frequency content + a cleaner metric.

---

## Iteration 10 — high-frequency stress + MultiDiffusion wired into coarse-to-fine (capstone)

**Trigger:** "run the high-frequency stress test and/or wire MultiDiffusion into the
coarse-to-fine vid2vid refine driven by the distilled proposer."

**Built:** `coarse_to_fine_multidiffusion_wan.py` — full ADR 0004 pipeline on high-frequency
content (dense ornate bookshelf): CausVid proposer (6-step) → upscale → SDEdit (strength
0.6) → tiled full-WAN refine, **independent-merge vs MultiDiffusion**, with a
boundary-isolating seam metric (peak ÷ window-median).

**Result (real H200):** both seamless — seam_v_excess 0.96 (both), seam_h_excess 1.05 vs
1.09; **latent overlap-disagreement only 0.073**; the two decoded frames are near-identical
and sharp (high-freq detail preserved). **MultiDiffusion gave no measurable benefit here.**

**Capstone finding (I19):** in the coarse-to-fine regime the **shared low-res framework
anchors the tile overlaps** → independent tiles barely diverge → **independent parallel
refinement is already seamless** at moderate strength; per-step latent fusion is
unnecessary. Opposite of Tier 1b (from-scratch tiling diverges). ⇒ the f_θ role is largely
provided by the framework conditioning; **independent tiles parallelize trivially across
GPUs** (no cross-tile sync) — clean for the distributed goal. MultiDiffusion is the
safety-net for from-scratch / high-strength refine. Bug fixed: `get_timesteps` needs the
scheduler's timesteps (not None).

**Synthesis of ADR 0004:** distilled proposer (5.7×) + framework-anchored **independent**
parallel tile refinement (seamless, trivially distributable) is the efficient design;
MultiDiffusion is the consistency safety-net for the unanchored/high-drift regime. Next: a
**strength sweep** to find the crossover where independent tiling breaks and MultiDiffusion
becomes necessary; and **multi-GPU** independent-tile parallelism (needs ≥2 GPUs).

---

## Iteration 11 — Kakeya "mac bridge" / Mac mini MLX evaluation (ADR 0005)

**Trigger:** "use the Kakeya mac bridge to connect to a local Mac mini and utilize its GPU."

**Read the bridge design** (`docs/design/mac-bridge-cloud-agent-access.md`, `kakeya_mac.py`).
**Finding:** the bridge is a **git-bus + GitHub Actions self-hosted-runner** dispatch for
**allowlisted MLX eval/bench presets** — not a GPU connection and not a serving channel.

**Objective verdict (ADR 0005):** this cloud agent **cannot** use it to "utilize the Mac
mini GPU" — D1 no inbound path to the Mac; D2 needs owner setup (register Mac as a runner +
workflow on a pushable repo); D3 the bridge lives in the Kakeya engine repo, not
OpenMontage_Kakeya; D4 it's CI/eval, not serving; D5 MLX serves LLM **text**, not WAN video;
D6 WAN latency kills the spec-decode data plane (control/tool plane over WAN, data plane on
LAN). Per the no-fake guideline, did not fabricate a connection to an unreachable Mac.

**Correct path (no OpenMontage code change):** the owner runs Kakeya's **MLX server**
(`serve.py --backend mlx`) on the Mac mini and points OpenMontage's `kakeya_llm`
(`KAKEYA_ENDPOINT`) at it over LAN/Tailscale — reusing the ADR 0001 seam. The Mac becomes a
local, private text backend; video stays on CUDA (ADR 0002/0004).

---

## Iteration 12 — distributed WAN across Mac mini + vast (different regions) — ADR 0006

**Trigger:** "complete script for the cloud agent to use Mac mini GPU + vast GPU for
distributed WAN inference."

**Objective blockers stated:** B1 — WAN is CUDA-only; the Mac's MLX **cannot run WAN**
(text only). B2 — cross-region RTT forbids tensor/pipeline parallelism (per-step tensors
can't cross the wire). ⇒ tensor-parallel WAN across Mac+vast is impossible; the feasible
design is a **heterogeneous, coarse-grained pipeline** (text on Mac, WAN on vast, only
prompts+mp4 cross the wire).

**Built (complete script set):** `services/distributed_wan/{worker.py,orchestrator.py,README.md}`.
Worker = CUDA WAN tile node (distilled proposer + vid2vid refine, GPU-locked). Orchestrator =
cloud-agent glue (Mac text via `KAKEYA_ENDPOINT` skip-not-fake + concurrent tile fan-out to
`WAN_WORKERS` + weight-map merge).

**Validated (real H200):** ran orchestrator **on the cloud VM** → vast worker **over a
tunnel** → framework 3.9s → 4 tile refines → **seamless 1472×768** koi-pond video
(`tier01_evidence/dwan_distributed_mid.png`). Mac plane skipped honestly (no reachable Mac).
N workers ⇒ real parallel refine (per-worker GPU lock); 1 worker ⇒ serialized (expected).

**Verdict:** "distributed WAN across Mac+vast" is delivered as the only feasible shape —
**heterogeneous (Mac text + vast WAN), coarse-grained, region-tolerant**; tensor-parallel WAN
across the two is not possible (B1/B2). Real multi-GPU WAN speedup = add co-located CUDA
workers.

---

## Iteration 13 — single-GPU time-division validation (ADR 0007)

**Trigger:** "single GPU: proposer framework → split in 2 → verifier does part 1 then part 2
(time-division) → f_θ integrates → validate the architecture."

**Built/ran** `time_division_2part_wan.py` (N-part, real WAN on one H200):
- **2-part** (1472×480×25): proposer 2.7s; verifier 4.4s+4.4s time-division; f_θ seam-excess
  **1.17 (seamless)**; peak 24.1GB.
- **4-part** (2752×480×49): proposer 5.6s; verifier 4×9.5s; f_θ seam-excess **1.24
  (seamless 2752×480 = 3.3× native width)**; peak 24.4GB.

**Findings:**
- **Feasibility VALIDATED** ✓ — proposer → time-division verifier → f_θ → seamless
  beyond-native-resolution video on ONE GPU; linear time in #parts; framework anchors parts
  → seamless.
- **Bounded memory NOT realized on the H200 (I20):** peak constant ~24GB regardless of
  parts/frames/canvas; full-canvas single pass didn't OOM even at 2752×480×49. WAN is already
  memory-bounded (WAN-VAE bounded-length design + SDPA attention + resident weights dominate).
  Time-division's memory benefit is **conditional** — only on a GPU too small to hold the
  constant footprint (16/24GB cards, WAN-14B). On the H200 the win is resolution scaling +
  seamless integration + (with more GPUs) parallelism, not memory.

**Verdict:** architecture feasible + validated; bounded-memory is a real but conditional
benefit (memory-constrained GPUs only).

---

## Iteration 14 — WAN-on-Apple-Silicon (MLX) feasibility (ADR 0008)

**Trigger:** "evaluate feasibility of porting WAN 2.1 to MLX."

**Finding:** **already done / highly feasible.** WAN 2.1/2.2 run on Apple Silicon via
maintained MLX ports (`Blaizzy/mlx-video` — Wan2.1 1.3B/14B + LoRA/4-step; `Wan2.2-mlx`;
`mlx-gen`), and via **PyTorch MPS + `mps-conv3d` + fp16** with no port. The historical
blocker (3D-conv VAE on Apple Silicon) is solved both ways.

**Honest correction:** ADR 0005 D5 / ADR 0006 B1 said "WAN can't run on the Mac/MLX" — that
was true only for *vanilla diffusers-on-MPS without patches* (bf16 + Conv3D-MPS fail), our
stack. Corrected in those ADRs + ADR 0008.

**Upshot:** the Mac can now be a (slow, RAM-bound, single-device) **WAN tile worker** in the
ADR 0006 task-parallel pipeline (same HTTP worker contract, MLX backend, speed-weighted tile
assignment) — not just the text plane. Unchanged: cross-region tensor-distribution still
impossible (latency, B2); MLX has no multi-device. Constraints are now **performance +
memory** (1.3B on ≥32GB, 14B needs ≥64GB+q8), not feasibility. Recommendation: don't port
from scratch — use `mlx-video` / MPS+`mps-conv3d`; keep heavy video on CUDA.

---

## Iteration 15 — worker transport: HTTP vs gRPC (ADR 0009)

**Trigger:** "why not gRPC for the worker contract?"

**Decision:** HTTP/JSON for the **worker/tool plane** because the workload is **coarse**
(1 call = a whole tile, seconds–minutes compute, few-MB payload), **cross-region + NAT**, and
**latency-tolerant**. gRPC's strengths (binary efficiency, HTTP/2 streaming/mux) are
negligible here (~33% base64 on 2MB vs minutes of diffusion; no high-frequency/per-step
traffic — that's the data plane B2 rules out cross-region), while its costs are real (grpcio
+ stubs on both ends → breaks the zero-dep stdlib orchestrator; HTTP/2 fussier through
relays). HTTP's wins (stdlib-only client, easy NAT traversal, backend-agnostic CUDA+MLX
workers, curl-debuggable) match the plane.

This **matches Kakeya's own split** (mac-bridge §4.2): control/tool plane = coarse +
latency-tolerant; **data plane = gRPC on a LAN** (which is what Kakeya uses `RuntimeService`
for, and which can't cross regions anyway). gRPC is reserved for a LAN fleet node / streaming
progress (SSE-over-HTTP covers progress without grpc deps).

---

## Iteration 16 — gRPC worker contract (product decision, ADR 0010)

**Trigger:** "usable product, not a toy → use gRPC; run mlx-video on the Mac as another GPU."

**Built:** `proto/video_worker.proto` (+ stubs), `grpc_worker.py` (backends: **cuda** diffusers
full ops, **mlx** wrapping mlx-video owner-run, **test** transport-only), `grpc_orchestrator.py`
(capability + **speed-weighted** routing, **server-streaming progress**, concurrent refine,
f_θ merge), `mac_setup.sh`. ADR 0010 supersedes ADR 0009's HTTP recommendation for the product.

**Validated (gRPC transport, local, no GPU):** two test workers (speed 3.0/1.0) → capability
negotiation; **exact 3:1 speed-weighted tile split** (cuda 3, mlx 1 of 4); per-tile streamed
progress interleaved across concurrent workers; f_θ merge → 1472×768 mp4. ✓

**Pending → done in iteration 17:** CUDA-over-gRPC real video (vast box was unreachable at
decision time; came back on a new port). MLX worker stays owner-run (no Mac here).

---

## Iteration 17 — CUDA-over-gRPC real video on the live H200 (completes ADR 0010)

**Trigger:** vast box returned (new port). Reinstalled the root Python env (torch/diffusers/
grpc); the **27 GB WAN cache on `/workspace` persisted** (no re-download). Bumped grpcio to
≥1.81.1 to match the stubs.

**Validated (real H200):** cloud-agent gRPC orchestrator → `--backend cuda` worker over a
tunnel → **server-streamed per-tile progress** → framework + 4 streamed `RefineTile` + f_θ
merge → **real h264 1472×768/25f seamless koi-pond video** (ffprobe-verified;
`tier01_evidence/grpc_cuda_real_mid.png`). The full gRPC product path works end-to-end with
the real WAN model. Single worker → tiles serialized (lock); N workers → parallel.

**Status:** gRPC worker contract (ADR 0010) is **built + validated** (transport locally,
CUDA real-video on GPU). MLX worker stays owner-run on the Mac (no Mac access).

---

## Iteration 18 — complete Mac mini MLX gRPC worker script

**Trigger:** "give the complete script to run on the Mac mini."

**Delivered:** `services/distributed_wan/mac_setup.sh` — turnkey: preflight (arm64/macOS≥14/
py≥3.11) → venv + mlx/mlx-video/grpc deps → clone repo + protoc stubs → convert
Wan2.1-T2V-1.3B→MLX → Tailscale hint → run `grpc_worker.py --backend mlx`. Made `MlxBackend`
**config-driven** (env: `MLX_T2V_MODULE`, `MLX_PASS_DIMS`, `MLX_OPS`, `MLX_V2V_FLAG`,
`MLX_RELATIVE_SPEED`) so it adapts to the installed mlx-video without code edits and **fails
loudly** (no silent garbage) if flags differ. bash -n + py_compile clean.

**Honest:** mlx-video = T2V/I2V (usually no vid2vid) → Mac advertises **framework/T2V** by
default; refines stay on CUDA unless the owner's mlx-video has vid2vid (`MLX_OPS+=refine`,
`MLX_V2V_FLAG`). Not runnable here (no Mac); owner runs it, then `WAN_WORKERS` includes the
Mac and the orchestrator speed-weights it.

---

## Iteration 19 — live cross-region Mac(MLX)+vast(CUDA) gRPC cluster wired; MLX module-path bug fixed

**Trigger:** owner reported the Mac mini MLX gRPC worker is up and reachable on the tailnet
(`TCP *:50051 LISTEN`, `nc -vz 100.78.64.43 50051 succeeded`). Goal: actually run the
distributed WAN job across both GPUs.

**What worked (verified, not claimed):**
- **Cross-region transport over a userspace-networking tailnet.** The vast H200 container has
  **no `/dev/net/tun`**, so tailscaled runs in userspace mode (SOCKS5 on `:1055`); a normal
  `connect()` to `100.x` does not route, and the box has neither `socat` nor `ncat`. Built
  `services/distributed_wan/socks5_forward.py` (stdlib only): `127.0.0.1:55051` → SOCKS5(1055)
  → `mac:50051`. gRPC then dials the local forward as plaintext h2c.
- **Mac Health over gRPC** through the forward: `backend=mlx-video device=mlx ops=['framework']
  speed=0.12` — a **real MLX worker** (not the test backend), ~214 ms tailnet RTT.
- **Two-GPU cluster staged:** vast CUDA restarted **refine-only** (`--ops refine`,
  `ops=['refine']`, warm) on `:50051`; Mac framework-only on `:55051`. Orchestrator routes
  framework→Mac, the 4 refine tiles→vast — genuinely both GPUs.

**Bug found + fixed (the real blocker):** a live `GenerateFramework` to the Mac streamed
progress 0%→5% then surfaced a clean gRPC `INTERNAL`: `No module named 'mlx_video.wan_2'`.
Checked the actual Blaizzy/mlx-video source: the entrypoints are
`mlx_video.models.wan_2.generate` / `.convert` (not `mlx_video.wan_2.*`), with verified flags
`--model-dir/--prompt/--output-path/--width/--height/--num-frames/--steps/--seed/--lora`
(num-frames must be 4n+1). Fixed `MlxBackend` defaults + `mac_setup.sh`
(download native `Wan-AI/Wan2.1-T2V-1.3B` checkpoint → `mlx_video.models.wan_2.convert
--checkpoint-dir/--output-dir`). Added `--ops`/`WORKER_OPS` so a CUDA box can be refine-only.

**Honest status:** the orchestrator → Mac → mlx-video path is wired and proven end-to-end at
the transport/streaming layer; the final pixels require the **Mac worker to restart on the
fixed code** (the running worker still has the old module path baked in) and a valid converted
MLX model dir — both Mac-side actions the cloud agent cannot perform remotely. On restart +
confirmation, the orchestrator produces the real Mac-proposer + vast-refiner video.

---

## Iteration 20 — REAL Mac(MLX)+vast(CUDA) distributed video produced ✅

**Result:** the full cross-region distributed WAN pipeline ran end-to-end and produced a real,
seamless **1472×768 × 25-frame h264** video (`docs/adr/tier01_evidence/dwan_mac_vast.mp4`,
mid-frame `dwan_mac_vast_mid.png`) — a red fox in a snowy forest, no visible tile seams.

**Measured (live, two regions):**
- **Proposer = Mac mini Apple-Silicon MLX** (`mlx-video`): low-res framework `480×256`, returned
  `(16, 256, 480, 3)` in **98.4 s** (includes per-call model load: umt5-xxl T5 + transformer + VAE).
- **Refine = vast H200 CUDA** (diffusers WAN vid2vid): 4 tiles, per-tile 6.7–23.1 s,
  **refine wall 24.5 s** (concurrent dispatch, serialized by the worker GPU lock).
- **Transport:** orchestrator on vast → `localhost:50051` (CUDA, refine-only) + `localhost:55051`
  → `socks5_forward.py` → SOCKS5(tailscaled) → Mac `:50051`, all gRPC server-streaming.

**Two bugs fixed to get here (both real, surfaced not masked):**
1. **MLX OOM** (Metal "Insufficient Memory") at full `832×480×25` VAE decode → made the proposer
   low-res by design (`--fw-width/--fw-height/--fw-frames`, temporal+spatial resample to canvas)
   + aggressive VAE tiling (`MLX_TILING`). Fox proposer at `480×256×13` fits comfortably.
2. **Idle stream drop** (`Stream removed (Socket closed)`): the MLX worker emits 5% then goes
   silent during the long T5 load; the idle HTTP/2 stream was cut over the SOCKS5/tailnet tunnel.
   Added a worker **heartbeat** (keepalive Progress every 5 s) + gRPC keepalive on both ends.

**What this proves:** WAN runs on Apple Silicon via MLX (ADR 0008), the gRPC worker contract
(ADR 0010) federates heterogeneous GPUs (MLX + CUDA) across regions, and the coarse-to-fine
proposer/refiner split (ADR 0004) yields a seamless beyond-native-resolution result from two
modest, geographically-separated machines. The Mac is genuinely "another GPU".

**Honest limits:** cross-region latency + per-call MLX model reload make the MLX proposer the
wall-clock bottleneck (~98 s vs ~25 s refine); this is a *capability/feasibility* win, not a
throughput win. For raw speed, co-locate CUDA workers (ADR 0006 §5).

---

## Iteration 21 — public agent video service via a domain (ADR 0011, kekaye.ai)

**Trigger:** "continue integrating OpenMontage so the agent video service is usable directly via
the kekaye.ai domain."

**Built:** `services/agent_gateway/` — a FastAPI **front door** (REST + minimal web UI) that turns
a request into a video by driving the validated distributed-WAN cluster, and the domain/TLS
deployment config (`deploy/Caddyfile`, `deploy/agent-gateway.service`, Tailscale-Funnel path).

**Architecture honored:** the gateway is a **transport + job layer only** — no creative/pipeline
logic in Python (Rule Zero). `mode="video"` shells out to `grpc_orchestrator.py` (capability call);
`mode="agent"` enqueues for an external agent runtime (`AGENT_RUNTIME_CMD`) and never fakes creative
decisions. Endpoints: `/`, `/healthz`, `/v1/capabilities`, `POST /v1/videos`, `GET /v1/jobs/{id}`,
`GET /v1/jobs/{id}/video`. Optional `X-API-Key` auth.

**Tested:** 6 offline tests (`tests/tools/test_agent_gateway.py`) using a fake orchestrator that
emits the real progress/`ORCH_DONE` protocol — full submit→run→poll→download lifecycle, auth gate,
honest agent-mode-without-runtime, 404s. All pass.

**Live-verified:** deployed on the vast H200, submitted `POST /v1/videos` (sea-turtle prompt) with
`X-API-Key`, the gateway drove the Mac+vast cluster, and `GET /v1/jobs/{id}/video` returned a real
h264 1472×768×25 mp4 (`docs/adr/tier01_evidence/gateway_demo.{mp4,_mid.png}`).

**Owner-dependent (cannot be done by the cloud agent):** pointing `kekaye.ai` DNS at a host, or
enabling Tailscale Funnel/HTTPS (admin) for the zero-DNS `*.ts.net` public URL. `tailscale serve`
hung waiting on tailnet HTTPS provisioning — that's an admin toggle, not a code issue.

---

## Iteration 22 — Mac-only pivot: DIRECT no-refine mode + all-on-Mac service (vast shut down)

**Trigger:** owner shut down vast and asked to prioritize the Mac mini GPU for connectivity +
OpenMontage integration testing.

**Reality check:** with vast down, the cloud-agent VM lost its only path to the Mac (it relayed
through vast's SOCKS5 forwarder; this VM has no TUN/tailnet of its own). So the durable answer is
**all-on-Mac**: the Mac is both the GPU and the control host.

**Built:**
- `grpc_orchestrator.py` **DIRECT (no-refine) mode** — auto-enabled when no refine-capable worker
  exists (Mac MLX is framework-only). One MLX T2V generation at fw dims, no tiled CUDA refine, so
  the gateway still produces video with just the Mac. Explicit `--no-refine` too.
- `deploy/mac_all_in_one.sh` + `deploy/mac-all-in-one.md` — turnkey: MLX worker + agent_gateway
  (:8088) + cloudflared → kakeya.ai, all on the Mac over localhost. mac_setup.sh now also installs
  fastapi/uvicorn.

**Honest:** the cloud agent cannot shell into the Mac, and vast (its relay) is down, so this
iteration is code + a runbook the owner runs on the Mac. Output in Mac-only mode is a low-res
T2V clip (memory-bounded); adding a CUDA refine worker re-enables the high-res refined pipeline
with no gateway change. Gateway tests (6) still pass; orchestrator compiles.

---

## Iteration 23 — two Mac minis over Thunderbolt: gateway worker-POOL (2× throughput)

**Trigger:** owner bridged two Mac minis over Thunderbolt (~536 G combined) and asked to
re-architect the video agent for it.

**Grounded finding (checked the package source):** `mlx-video` has **no** distributed/tensor-
parallel support (no `mlx.distributed`/`mlx.launch`/collectives) and **no vid2vid**. So:
- a single WAN generation can't be sharded across the two Macs — the combined memory is **not**
  one pool for one generation (each gen is bounded by one Mac);
- the high-res proposer→tiled-refine pipeline still needs a CUDA refiner.

**Built — the honest win is throughput:** gateway **POOL mode** (`AGENT_GATEWAY_WORKER_POOL=1`):
each `WAN_WORKERS` entry is an independent GPU; a `queue` hands each job one free Mac (DIRECT
no-refine), N jobs run in parallel = N× throughput, with backpressure when all Macs are busy.
- `server.py`: pool queue + per-worker dispatch + `--no-refine`; `/healthz` reports
  `pool_mode`/`parallel`.
- `mac_all_in_one.sh`: `PEERS="<macB-bridge-ip>:50051"` → head Mac runs gateway in pool mode.
- `two-mac-thunderbolt.md`: bridge IPs, per-Mac worker, head gateway + cloudflared, verify 2×.
- 7 gateway tests pass (added a two-Mac pool test).

**Honest:** still owner-run (no cloud-agent path to the Macs after vast shutdown). Output remains
Mac-grade DIRECT T2V; adding a CUDA refiner re-enables high-res refine with no code change.

---

## Iteration 24 — re-anchor to the agent: WAN cluster = registered provider tool (ADR 0012/0013)

**Trigger:** owner accepted the architecture review and directed: localize OpenMontage + expose via
Cloudflare; no co-located CUDA (on-demand vast for refine); re-evaluate the 512 G Mac's high-res
ceiling; harden before public; collapse the two video backends.

**Keystone shipped — re-anchor + collapse:** the distributed-WAN cluster is now a third transport
behind the existing `generate_local_video()` seam, so `wan_video` (already
`capability="video_generation"`, discoverable by `video_selector`, usable by every pipeline) routes
to the **local Mac cluster** — the agent uses it like any provider. One unified local-video
abstraction (distributed gRPC | warm HTTP gateway | in-process diffusers), not parallel stacks.
- `WAN_WORKERS` → Mac-only DIRECT no-refine T2V at requested W×H.
- `VAST_REFINE_WORKER` (optional) → appends an **on-demand** vast CUDA worker → proposer+refine.
- `local_generation_status()` AVAILABLE when `WAN_WORKERS` set. Tests:
  `tests/tools/test_local_wan_provider.py` (3) + gateway (7) = 10 pass.

**Mac high-res re-evaluation (ADR 0012):** the earlier "low-res only" was a *small-Mac* OOM
artifact. A ~512 G Apple-Silicon box (M3 Ultra 512 G / 819 GB/s) **removes the memory ceiling** —
WAN 14B @ 720p fits easily. The real limit is **GPU time** (e.g. WAN 480p 5 s ≈ 11 min on M3 Ultra
vs ~2–3 min on a 4090/H200). Few-step LoRA brings 14B/720p to a few minutes (async-friendly).
Verdict: **the 512 G Mac IS high-res capable** (latency-bound, not memory-bound); attach vast for
*speed*, not to *enable* resolution.

**Honest:** still owner-run on the Mac (no cloud-agent path to the Macs; vast off). Remaining
sequenced work in ADR 0013 §5: gateway `mode=agent` runtime, hardening (durable jobs/supervision/
rate-limit/per-key auth), parameterize the refine canvas, benchmark the real box.

---

## Iteration 25 — link check + relay moved to the Mac (subdomain on Cloudflare)

**Link test (`检测链接`):** `kakeya.ai` resolves to Cloudflare (`172.67.167.146`) and returns
HTTP 200, but it **serves an unrelated "AgentMate" site**, not our gateway (`/healthz` returns that
HTML, not JSON). No `agent/video/api/...` subdomain exists. **Verdict: the OpenMontage link is NOT
wired** — the apex is taken by another app.

**Decision (owner):** the **relay lives on the Mac mini**, not on a GPU. The Mac cluster is the
always-on entry (cloudflared) + proposer/refiner; vast is an on-demand refiner only. This removes
the prior "can't reach the cluster because vast is off" failure mode.

**Done:** updated all deploy docs to (a) run `cloudflared` (the relay) on the **Mac**, and (b)
expose the agent on a **subdomain `agent.kakeya.ai`** so it coexists with the existing apex site —
`cloudflare.md`, `mac-all-in-one.md`, `two-mac-thunderbolt.md`, `mac_all_in_one.sh`.

**To go live (owner, on the Mac):** run the gateway (`mac_all_in_one.sh`) + a Cloudflare Tunnel
with public hostname `agent.kakeya.ai → http://localhost:8088`. Then `https://agent.kakeya.ai/healthz`
returns JSON and I can verify end-to-end.

---

## Iteration 26 — render test PASSED on the Mac MLX GPU (cloud agent drove it via SSH-over-Cloudflare)

**Access:** owner exposed `ssh.kakeya.ai → Cloudflare Tunnel → Mac:22` and authorized the cloud
agent's key. The agent now SSHes in via `cloudflared access ssh` (config `Host mac`) and drives
everything itself — no more human copy-paste.

**Mac facts:** head Mac has a **display** (3440×1440), runs the MLX worker on `127.0.0.1:50051`
under `caffeinate -dimsu MLX_TILING=aggressive`; headless peer is `169.254.27.104:50051`.

**Two real clips rendered (agent-driven, verified):**
1. **Direct orchestrator** → display worker, DIRECT no-refine `480×256` → `(16,256,480)` in **104 s**,
   real h264 fox-in-snow (`tier01_evidence/mac_render_proof.{mp4,_mid.png}`). No watchdog timeout
   (display + tiling + caffeinate).
2. **Through the gateway** (`POST /v1/videos` → job `e7fe26437000` → `done`), downloaded via
   `/v1/jobs/{id}/video` → real h264 otter-in-kelp (`tier01_evidence/gateway_render_proof.{mp4,_mid.png}`).
   Proves gateway → orchestrator → MLX worker → served mp4.

**Open issues (owner-side, not integration):**
- **Public `agent.kakeya.ai` DNS/route flapping:** it resolved + returned `healthz 200` earlier
  this session, then stopped resolving entirely — the Cloudflare DNS record/tunnel public-hostname
  for the subdomain needs restoring. The gateway + tunnel connector are healthy locally.
- **Headless peer watchdog:** round-robin still sends ~half of jobs to `169.254.27.104`, which trips
  `kIOGPUCommandBufferCallbackErrorTimeout`. Drop it from `WAN_WORKERS` (route to the display Mac)
  or add an HDMI dummy plug to bring it back reliably.

---

## Iteration 27 — PUBLIC path proven end-to-end at agent.kakeya.ai ✅

**Root cause of the earlier outage (resolved):** the Mac's `cloudflared` was authenticated to a
different Cloudflare account/zone (`agentmate.build`), so `cloudflared tunnel route dns kakeya-gw …`
wrote junk records `agent.kakeya.ai.agentmate.build` against the wrong tunnel (`aeb49800…`).
`agent.kakeya.ai` / `ssh.kakeya.ai` never existed in the real `kakeya.ai` zone (NXDOMAIN). Fix:
re-auth to the `kakeya.ai` account and route by the GATEWAY tunnel **ID**:
`cloudflared tunnel route dns 99e33427-bf06-4c63-8678-8ee37bfc3921 agent.kakeya.ai` (+ `ssh`).

**Public proof (agent-driven):** from the cloud-agent VM, `https://agent.kakeya.ai/healthz`
returns the gateway JSON; `POST /v1/videos` (Cloudflare round-trip) → job `fe99ecd5d15d` → `done`
→ downloaded `/v1/jobs/{id}/video` = real h264 480×256×16 otter clip
(`tier01_evidence/public_render_proof.{mp4,_mid.png}`). Full chain verified:
**VM → Cloudflare → kakeya-gw tunnel → gateway → orchestrator → Mac MLX GPU → mp4 → back to VM.**

**Notes:** the cloud-agent VM's stub resolver (`10.0.0.2`) cached the old NXDOMAIN; worked around
by pinning the CF edge IP in `/etc/hosts` (VM-local, not repo). SSH-over-Cloudflare (`ssh.kakeya.ai`)
also restored. Headless peer (`169.254.27.104`) still trips the GPU watchdog on ~half of round-robin
jobs — drop it from `WAN_WORKERS` or add an HDMI dummy plug for reliable 2-worker serving.

---

## Iteration 28 — public demo opened (no key) + routed to the display Mac

**Trigger:** the web UI returned `401 missing or invalid X-API-Key` (the "API key" field was empty);
owner asked to open the demo.

**Done (agent-driven via SSH):** relaunched the Mac gateway with **no `AGENT_GATEWAY_API_KEY`**
(auth off) and **`WAN_WORKERS=127.0.0.1:50051`** (display Mac only, so the open demo never lands on
the headless peer's GPU watchdog). Verified end-to-end from the cloud VM with **no key**:
`POST /v1/videos {fox prompt}` → job `49d70fc2ce6e` → `done` → downloaded real h264 480×256×16 fox
clip (`tier01_evidence/public_open_demo.{mp4,_mid.png}`). The web UI now works with the key field
empty.

**Security note:** the public endpoint now accepts unauthenticated renders — fine for a watched
demo, but anyone can spend Mac GPU time. Recommend a Cloudflare WAF/rate-limit rule on
`POST /v1/videos` (owner dashboard) and re-enabling `AGENT_GATEWAY_API_KEY` after the demo. The
gateway runs via `nohup` (not launchd) — it won't auto-restart on reboot; durable supervision is
the ADR 0013 §5 hardening follow-up.

---

## Iteration 29 — BOTH Mac GPUs utilized in parallel over Thunderbolt ✅

**Headless Mac fixed:** the second Mac (`allen@Allens-Mac-mini`, `169.254.27.104`) now reports a
**display Online (2180×1200)** — a monitor/dummy plug is attached, which relaxes the macOS GPU
watchdog. A direct test render on its GPU completed cleanly (`ORCH_DONE`, 117 s, real h264 clip,
no `kIOGPUCommandBufferCallbackErrorTimeout`). So both Mac GPUs render.

**Topology clarified:** head Mac `fluffy314@fluffy314s-Mac-mini` (`169.254.187.239`, display, gateway
+ cloudflared); headless Mac `allen@Allens-Mac-mini` (`169.254.27.104`, display now attached). The
Thunderbolt bridge is a fast LAN link, not GPU pooling (mlx-video has no sharding) — "both GPUs" =
one MLX worker per Mac + the gateway distributing jobs.

**Both GPUs in parallel (verified):** the mac-bridge gateway variant uses
`AGENT_GATEWAY_WORKER_MODE` (`cluster`=distributed/serialized, `round_robin`=one job per worker,
N parallel). Relaunched with `WORKER_MODE=round_robin` + both workers → `max_video_jobs=2`. Two
public jobs submitted back-to-back (deer + whale) were **both `running` simultaneously** (one per
Mac) and **both finished** as real h264 480×256×16 clips
(`tier01_evidence/dualgpu_{deer,whale}.mp4`, `dualgpu_whale_mid.png`, `headless_gpu_proof.mp4`).

**State:** `agent.kakeya.ai` open demo now load-balances across **both Mac mini GPUs** in parallel
(2× throughput). Access to the headless Mac is via head Mac → `ssh allen@169.254.27.104` (key
trust established).

---

## Iteration 30 — launchd auto-start (cluster survives crashes/reboots)

**Done (agent-driven via SSH):** installed per-user **LaunchAgents** (not LaunchDaemons — they run
in the GUI/Aqua session so the GPU watchdog stays relaxed) with `RunAtLoad` + `KeepAlive`:
- Head Mac (`fluffy314`): `ai.kakeya.mlxworker` (`127.0.0.1:50051`, under `caffeinate`,
  `MLX_TILING=aggressive`) + `ai.kakeya.gateway` (`:8088`, `WORKER_MODE=round_robin`, both workers).
- Headless Mac (`allen`): `ai.kakeya.mlxworker` (`169.254.27.104:50051`).
All `state=running` via `launchctl print gui/501/…`; PATH is venv-first so the `mlx_video` subprocess
resolves. Verified end-to-end: a public job (`d588c5fe44e3`) ran on the launchd-managed cluster →
`done`.

**Artifacts:** `services/agent_gateway/deploy/launchd/` (README + templated
`ai.kakeya.{mlxworker,gateway}.plist`).

**Reboot caveat (documented):** LaunchAgents start at **user login**, so unattended reboot recovery
also needs **auto-login** enabled, and **FileVault** (if on) blocks unattended pre-boot unlock. Crash
recovery (`KeepAlive`) works regardless. `cloudflared` should also be made a service
(`sudo cloudflared service install <token>`) for full reboot durability.

---

## Iteration 31 — tunnel outage + recovery; rotation-resilient cloudflared LaunchAgent

**What broke (my mistakes, documented honestly):**
1. The first cloudflared LaunchAgent used `cloudflared tunnel run --token` and a hard-coded token;
   retiring the live `--url` connector left it unable to serve. The self-healing fallback relied on
   `setsid`, which **macOS lacks**, so it didn't fire → tunnel down, SSH lost.
2. While recovering, multiple connectors ran at once → `control stream … failure` churn.
3. The decisive cause: the owner had **rotated the tunnel token**, so every connector using the old
   (plist) token was rejected. Cloudflare's UI does not display the token after a rotate.

**Recovery (owner-run, since SSH was down):** `cloudflared tunnel token <UUID>` fetches the CURRENT
token from the CLI (no dashboard needed). Running a single `cloudflared tunnel --url
http://localhost:8088 run --token "$TOKEN"` registered 4/4 connections → tunnel `Healthy`, SSH +
`agent.kakeya.ai` back.

**Durable + rotation-resilient fix:** `deploy/launchd/run_cloudflared.sh` (wrapper that fetches the
token via `cloudflared tunnel token <UUID>` at startup) + `ai.kakeya.cloudflared.plist`
(RunAtLoad/KeepAlive). Installed live (connector pid healthy, 4/4 registered), foreground
connectors retired → exactly one durable connector. So auto-start survives reboots AND future token
rotations (as long as `cert.pem` stays valid).

**Lessons:** never hard-code a Cloudflare tunnel token in auto-start; never run >1 connector per
tunnel; macOS has no `setsid`; the tunnel carries the cloud agent's only SSH path, so changes to it
must be done with a self-healing/he-can-recover-it plan. Compute (workers/gateway/FileVault/
auto-login) was unaffected throughout.

---

## Iteration 32 — UI job failures: headless worker unreachable via link-local; fixed via LAN IP

**Symptom:** UI renders intermittently failed — `orchestrator exited rc=1`,
`No route to host ipv4:169.254.27.104:50051`. Round-robin sent ~half the jobs to the headless
worker, addressed by its **Thunderbolt link-local** IP.

**Cause:** the head Mac has `169.254.x` on multiple interfaces (`bridge0` + `en9`), so the route to
the peer's `169.254.27.104` was ambiguous → "No route to host". (The headless worker was up and
bound; the head just couldn't route to that link-local addr.)

**Fix:** rebind the headless worker to **`--host 0.0.0.0`** and point the gateway at the peer's
**stable LAN IP** `192.168.68.51:50051` (gRPC can't resolve `.local` mDNS). Verified: two public
jobs ran in parallel (head + headless) and both completed. `agent.kakeya.ai` healthy again.

**Durability note:** `192.168.68.51` is DHCP — set a **DHCP reservation** (router) or use the
peer's **Tailscale IP** so it can't change on a future lease/reboot. Docs updated
(`deploy/two-mac-thunderbolt.md`): never address a peer by `169.254.x`.

---

## Iteration 33 — two-Mac proposer→refiner PIPELINE (head=proposer, headless=refine) — ADR 0014

**Ask:** run the cluster as coarse-to-fine across the two Macs — head Mac = **proposer**
(framework/T2V), headless Mac = **refiner** — not as two independent pool jobs.

**Blocker (already documented):** stock `mlx-video` has **no vid2vid**, and the orchestrator's
refine path was a 2×2 **tiled CUDA V2V** flow that doesn't fit a single headless Mac. The MLX
worker's `refine` op hard-failed unless `MLX_V2V_FLAG` was set.

**What we built (honest, works on stock mlx-video):**
1. **MLX worker — functional `refine` without vid2vid** (`grpc_worker.py`). If `MLX_V2V_FLAG` is set
   → generative V2V (unchanged). Otherwise → **spatial super-resolution refine**: Lanczos upscale to
   `--out-width/--out-height` + unsharp mask (`MLX_SR_SHARPEN`, default 60). Pure PIL/numpy: OOM-safe,
   no Metal, CI-testable. `health().note` advertises `refine=sr` vs `refine=v2v` so callers aren't
   misled. **It raises resolution/crispness; it does NOT synthesize new detail** (that needs CUDA V2V).
2. **Orchestrator — single-pass PIPELINE** (`grpc_orchestrator.py`). New `--single-refine`
   (auto-on when the chosen refiner is MLX) + `--out-width/--out-height`. Proposer (low-res, head)
   → temporal resample → **one full-frame** `RefineTile` on the refiner (headless) at the target
   resolution → write. Picks a **distinct** proposer when possible so both Macs are used. Emits
   `ORCH_DONE {"mode":"pipeline","proposer":A,"refiner":B,...}`. CUDA refiners still get the tiled path.
3. **Gateway — PIPELINE mode** (`agent_gateway/server.py`). >1 worker + pool OFF ⇒ `PIPELINE_MODE`:
   one job spans both Macs, gateway passes `--single-refine`. `/healthz` now reports
   `{"mode":"pipeline|pool|single","pipeline_mode":bool}`. `refine:` progress parsed as the refine stage.
4. **Deploy** — worker `--mlx-ops` is now a role (`framework`=proposer / `refine`=refiner): launchd
   `mlxworker` plist `__MLX_OPS__` + `MLX_SR_SHARPEN`; gateway plist documents pipeline (default) vs
   pool; `mac_all_in_one.sh` gains `MODE=pipeline|pool` + `MLX_OPS`; `two-mac-thunderbolt.md` rewritten
   with the two-topology table and per-mode launch/verify.

**Tests (offline, no GPU):** `tests/tools/test_distributed_wan_pipeline.py` — (a) `MlxBackend` SR
refine upscales a tiny mp4 to the target res without vid2vid; (b) **end-to-end over real gRPC**: two
in-process `TestBackend` workers (framework + refine) + `grpc_orchestrator --single-refine` produce an
mp4 at the out-res with `mode=pipeline`, proposer=A, refiner=B. Plus a gateway pipeline-mode test
(asserts `--single-refine`, both workers, no `--no-refine`). **10/10 pass** with the existing gateway
suite.

**Honest verdict:** this is a *real* two-Mac coarse-to-fine path (head drafts low-res fast, headless
upscales/refines the whole clip), giving higher-res output than a single Mac's DIRECT draft — but the
MLX refine is **interpolative SR, not generative**. For generative detail, attach a CUDA refiner (the
orchestrator auto-promotes to the tiled V2V pipeline) or an mlx-video build with vid2vid (`MLX_V2V_FLAG`).

**New issue I33:** the previous one-line manual run (`WAN_WORKERS=127.0.0.1:50051`) only listed the
head, so the headless never got work — not a bug, a command-arg artifact. Pipeline mode makes the
two-role split the gateway default so the headless is always engaged on each job.

**Live verification (this iteration):** head Mac repo moved to this branch; Mac B's `grpc_worker.py`
updated (scp from head — Mac B's `git fetch` hangs on a GitHub HTTPS credential prompt) and its
launchd refiner restarted; head worker+gateway reloaded. Public job through `agent.kakeya.ai`:
`ORCH_DONE mode=pipeline proposer=127.0.0.1:50051 refiner=192.168.68.51:50051 px=[480,832] frames=25
proposer=105.7s refine=0.36s`; downloaded clip = h264 832×480 25f. The **headless Mac did the
refine** (refine≈0.36 s confirms the SR path). ✅

---

## Iteration 34 — 3-node: head=proposer, headless(MLX)+vast(CUDA)=refiners (gateway mode selector)

**Ask:** add a vast CUDA box so the cluster runs head Mac = proposer, **and BOTH** the headless Mac
(MLX SR refine) **and** vast (CUDA generative refine) as refiners.

**Code gap found:** the gateway *forced* `--single-refine` in the 2-Mac case, which picks only ONE
(fastest) refiner — with vast present that would idle the headless Mac. Fixed by replacing the
binary pool/pipeline switch with an explicit **`AGENT_GATEWAY_MODE`**:
- `auto` (default): pass ALL workers, no flag → orchestrator auto-picks single-pass (MLX refiner)
  vs **tiled** (CUDA refiner). 2-Mac → single on the Mac; 3-node → tiled across both refiners.
- `distributed`: tiled + **`--refine-spread roundrobin`** so EVERY refiner (incl. the slow MLX Mac)
  gets a share of tiles instead of being optimized out by speed-weighting.
- `pipeline`: force `--single-refine` (head + one refiner). `pool`: per-worker DIRECT (throughput).
- back-compat: `AGENT_GATEWAY_WORKER_POOL=1` ⇒ `pool`. `/healthz` reports the resolved `mode`.

**Orchestrator:** new `--refine-spread {weighted,roundrobin}`. In the tiled path, `roundrobin` deals
tiles fastest-first in turn so the MLX Mac contributes (2 tiles MLX-SR / 2 tiles CUDA-V2V on a
4-tile, 2-refiner job) rather than vast taking all 4 (an 8× speed gap zeroes the Mac under
`weighted`). Tile→worker log now prints `addr(backend)`.

**Vast refiner bringup:** `services/distributed_wan/vast_refiner_setup.sh` — installs gRPC+diffusers,
launches `grpc_worker --backend cuda --ops refine --preload` (refine-only so the head stays the sole
proposer), and documents reachability (head→vast SSH `-L 50052:localhost:50051` tunnel / Tailscale /
mapped port).

**Honesty:** with a fast CUDA refiner present, `weighted` would (correctly, for wall-time) route all
tiles to vast and idle the Mac; `distributed`/`roundrobin` is the explicit "use all three" mode at
some wall-time cost. MLX tiles are SR (interpolative); vast tiles are generative — blended by the
f_θ weight-map merge. **Tests:** gateway mode resolution (auto/pipeline/distributed/pool) +
orchestrator pipeline; 12/12 pass.

**Networking note:** the fresh vast box rejected both the cloud-agent and head keys at first
(`Permission denied (publickey)`). Resolved using the cloud-agent's own `~/.ssh/vast_key` (env
secret) to reach vast, then authorizing the head Mac's pubkey there. One gotcha: appending the head
key with `echo >>` onto a no-trailing-newline `authorized_keys` **merged** it into the existing
line (head key became a comment → still denied); fixed by rewriting `authorized_keys` with both keys
on separate lines.

**Live verification (3-node) ✅** — vast H200 (143 G): installed torch 2.12 + diffusers 0.38 +
**protobuf** + **ftfy** (WAN text-encoder needs ftfy; first refine RPC failed `NameError: ftfy`),
ran `grpc_worker --backend cuda --ops refine --preload` in tmux. Head→vast reachability = an SSH
local-forward `127.0.0.1:50052 -> vast:50051` from the head. Gateway set
`WAN_WORKERS=127.0.0.1:50051,192.168.68.51:50051,127.0.0.1:50052` + `AGENT_GATEWAY_MODE=distributed`.
Public job → `agent.kakeya.ai`:

```
[orch] tile (0,0) <- 127.0.0.1:50052 (7.0s)    vast   (CUDA generative V2V)
[orch] tile (0,1) <- 192.168.68.51:50051 (0.2s) Mac B  (MLX SR)
[orch] tile (1,0) <- 127.0.0.1:50052 (11.6s)   vast
[orch] tile (1,1) <- 192.168.68.51:50051 (0.2s) Mac B
ORCH_DONE refine_wall_s=14.25 canvas_px=[768,1472]
```

Downloaded clip = **h264 1472×768 25f** — head proposed, **both** Mac B and vast refined
(round-robin), higher-res than the 2-Mac pipeline's 832×480. ✅

**Durability TODO (this topology):** the vast worker runs in `tmux` (survives SSH disconnect, not a
vast reboot) and the head→vast tunnel is a `-fN` ssh (not auto-restarting). For a standing 3-node
service, wrap the vast worker (systemd/tmux-resurrect) and the tunnel (autossh / launchd on the head)
before relying on it unattended.

---

## Iteration 35 — ephemeral-vast durability: auto-recover refiner; 2-Mac fallback (no restart)

**Ask:** vast GPUs are released/recreated often — guarantee the cluster auto-recovers the vast
refiner after every recreate, and keeps producing video (head proposer + f_θ) while vast is gone.

**Built (3 parts, no gateway restarts):**
1. **Dynamic membership** — gateway `AGENT_GATEWAY_WORKERS_FILE`: the worker list is re-read from a
   file **per job + healthz**, so a refiner can be added/removed live. Falls back to `WAN_WORKERS`.
2. **Head supervisor** (`vast_refiner_supervisor.sh`, launchd KeepAlive): each cycle — if vast is
   reachable, scp worker files + run idempotent `vast_bootstrap.sh` (install deps incl. protobuf/ftfy,
   launch cuda refine-only in tmux), (re)open the `ssh -L 50052→vast:50051` tunnel, and write
   `BASE_WORKERS,127.0.0.1:50052` → **3-node**; if vast is unreachable or still loading, write
   `BASE_WORKERS` → **2-Mac fallback** (head proposer + headless SR via f_θ).
3. **Idempotent vast bootstrap** so a FRESH box self-converges to a running refiner.

**Live proof (full release→recreate cycle):**
```
22:51 vast refiner ONLINE -> 3-node (added 127.0.0.1:50052)     healthz workers=[head,MacB,vast]
22:52 vast 104.202.252.41:29999 unreachable -> 2-Mac fallback   healthz workers=[head,MacB]
22:53 [bootstrap] worker already listening; vast refiner ONLINE -> 3-node   healthz workers=[head,MacB,vast]
```
Membership flipped 3-node ↔ 2-Mac with **no gateway restart** (dynamic file). Deployed on the head as
`ai.kakeya.vastsupervisor` (launchd) + gateway switched to the dynamic file. Runbook:
`deploy/ephemeral-vast-refiner.md`. **After a recreate, only the SSH endpoint changes** — edit
`~/.kakeya/vast.env` (or set `VAST_RESOLVE_CMD` for zero-touch); the supervisor does the rest.
**Test:** dynamic workers-file membership (3-node↔2-Mac, no restart); 13/13 pass.

---

## Iteration 36 — quality & duration optimization Phase 1 (ADR 0015)

**Ask:** start optimizing two dimensions on the current architecture — **video quality** and **duration**.

**Built (Phase 1, offline-validated):**
- **Quality — seam-free full-frame refine.** Orchestrator `--refine-mode {auto,direct,single,tiled}`:
  `single` runs ONE generative V2V over the whole upscaled clip on the best (CUDA) refiner — no 2×2
  seams, no mixed MLX-SR tiles. `auto` keeps prior behavior.
- **Quality — presets/knobs.** Gateway `quality=draft|standard|high` (+ per-knob overrides): high =
  `refine_mode=single`, 1280×720, refine-steps 24, 24fps, interpolate ×2. Topology still from
  `AGENT_GATEWAY_MODE`; `high` forces the seam-free single path on the CUDA box.
- **Duration/smoothness.** `--fps`, `--seconds` (frames = round(seconds·fps)), `--interpolate {1,2,4}`
  (linear-blend temporal interpolation). All encode paths go through `_encode(fps, interpolate)`;
  `ORCH_DONE` reports fps/interpolate/frames.

**Honest scope:** `single`/`high` uses one refiner (vast) per job (seam-free, fully generative) — use
`distributed` to engage both refiners. Phase-1 `--seconds` resamples the proposer's motion (smoother,
not new content); true long-form = Phase-2 chunked I2V continuity. Linear interp is a baseline; RIFE/
FILM is Phase 2.

**Tests:** `_interpolate` frame-count; gateway quality presets + seconds/overrides; real-gRPC
`--refine-mode single` + fps/interpolate (ORCH_DONE fps=16, interpolate=2, frames=9). **20/20 pass.**
Roadmap + phases in ADR 0015. Live high-res (`high`→vast 1280×720 seam-free) benchmark next.

---

## Iteration 37 — quality+duration Phase 2: long-form (chunked I2V) + hi-res refine fix; live vast

**New vast** (RTX PRO 6000 Blackwell, 97G) brought up as refiner via the supervisor (fresh-cache fix
for a `Stale file handle` on the `/workspace` HF cache → `HF_HOME=/root/.hf_home`; re-authorized the
head key with the un-merge/dedup normalizer). Supervisor **auto-recovered the new endpoint** (tunnel +
3-node) after the vast.env edit — durability proven on a real recreate.

**Hi-res quality finding (two real-GPU runs):** the WAN **1.3B V2V model is locked to its native
~832×480** — feeding it 720p just gets downscaled, so generative output is 480p. The honest hi-res
pipeline is **generative refine at native res → spatial SR-upscale to the display target**. Fix:
`single` upscales the proposer before the refiner AND SR-upscales the generative result to
`out_w×out_h`. **Validated:** proposer(MLX)→generative V2V on vast(cuda, 9.1s)→SR→interpolate ×2 →
`ORCH_DONE px=[720,1280] frames=49`, ffprobe **h264 1280×720, 49f, 2.04s**. ✅ (True 720p needs a
720P/14B model — Phase 2b.)

**Phase 2 long-form** (ADR 0015 §3b): `--chunks` autoregressive generation with **I2V continuity**
(proto `init_image`; CUDA `WanImageToVideoPipeline` gated on `CUDA_I2V_MODEL`; advertises `i2v` only
then) + crossfade `_stitch`. Gateway `longform`/`chunks`. Tests: `_stitch` math, real-gRPC long-form
(chunks=3, continuity=i2v, frames=23), gateway long-form + presets/seconds. **23/23 pass.**

**Honest scope:** long-form continuity needs a Wan **I2V checkpoint** on the CUDA box
(`CUDA_I2V_MODEL`) — orchestration validated offline + the contract propagates to vast via the
supervisor; without it, chunks are independent T2V (continuity off). RIFE/FILM is Phase 2b.

---

## Iteration 38 — Phase 2b: I2V-14B-720P long-form on vast — TRUE 720p + continuity ✅

Deployed **Wan2.1-I2V-14B-720P** (84 G) on the new vast (RTX PRO 6000 Blackwell, 97 G). Worker does
the T2V seed (1.3B) + I2V chunks (14B) — launched with `--ops framework,refine,i2v` + `CUDA_I2V_MODEL`.
Long-form ran entirely on vast (`WAN_WORKERS=127.0.0.1:50052` via the head→vast tunnel).

**Live result:** `ORCH_DONE mode=longform continuity=i2v chunks=2 px=[720,1280] frames=46 seconds=2.88
gen=391s` → ffprobe **h264 1280×720, 46f, 2.875s**. **True native 720p generative** (the 14B I2V
generates at 1280×720 — no SR) with **I2V continuity** across the chunk boundary (25+25−4 crossfade).
~8.3 s/step at 720p.

**Issues hit + fixes (real-GPU):**
- `/workspace` HF cache `Stale file handle` → `HF_HOME=/root/.hf_home`.
- head-key merge on vast `authorized_keys` (no trailing newline) → un-merge/dedup normalizer.
- the 1.3B V2V caps at ~480p (Iter 37) → 14B I2V is the true-720p path.
- **vast image kills user processes on SSH logout** (no systemd/linger; killed tmux AND setsid) →
  ran the worker under a **persistent SSH** for the validation. Durable i2v under the supervisor on
  such images needs systemd/linger (Phase 2c). `vast_bootstrap.sh`/supervisor now env-gate i2v
  (`CUDA_I2V_MODEL`/`VAST_I2V_MODEL`) for images where tmux persists.
- orchestrator on the head used the head repo's OLD pb2 (no `init_image`) → scp'd the regenerated pb2.
- the MLX head seed hit the **Metal GPU watchdog** → moved the T2V seed onto vast's 1.3B (all-on-vast).

**Tests:** 20 offline (no GPU) still green. Code in PR #5 (ADR 0015 §3b).

---

## Iteration 39 — Phase 2c: optical-flow (mci) interpolation + durable held-worker (auto-recover) ✅

**Optical-flow interpolation:** `--interp-method {linear,mci}` (gateway `interp_method`; `high`→mci).
`_interpolate_mci` uses **ffmpeg `minterpolate`** (motion-compensated: `mi_mode=mci`, `mc_mode=aobmc`,
bidir, vsbmc) — RIFE-class smoothness over linear blend, linear fallback if unavailable. Confirmed
`minterpolate` is in the bundled ffmpeg (not falling back). Unit-tested (frame-count grows).

**Durable i2v on ephemeral vast** (`VAST_HOLD_WORKER=1`): the new Blackwell vast image **kills user
processes on SSH logout** (no systemd/linger — killed tmux AND setsid). Fix: the **launchd-durable
head supervisor holds the worker as a child of a persistent ssh** (`KAKEYA_HELD=<host> … exec python
grpc_worker --ops framework,refine,i2v`); `vast_bootstrap.sh` gains `BOOTSTRAP_NO_LAUNCH` (deps-only).
`vast_onstart.sh` authorizes the head key + persistent `HF_HOME` so a recreated box is auto-adopted.

**Validated live:** killed the manual persistent worker; the supervisor **spawned and now holds** the
i2v worker — 1 held ssh on the head, `ops=['framework','refine','i2v']` via the tunnel, 3-node — with
**no dependency on any interactive session**. On recreate: held ssh drops → 2-Mac fallback → new box
(key authorized, endpoint via vast.env/`VAST_RESOLVE_CMD`) → deps reinstall + held worker respawn →
auto-recovery. (Caveat: the ~84 G I2V model re-downloads unless `HF_HOME` is on a persistent volume.)

**Tests:** 21 offline green. ADR 0015 §3c. Phases 1/2/2b/2c all landed in PR #5.

---

## Iteration 41 — Gap B Phase 1: autonomous agent runtime (mode=agent) — ADR 0016

**Finding:** OpenMontage has **no Python pipeline runner** — the host LLM is normally the
orchestrator. So `mode=agent` integration = building a new **autonomous agent runtime** (a cloud-LLM
reasoner + tool-calling loop). In-repo LLM is only `kakeya_llm` (mechanical).

**Built:** `services/agent_runtime/` (`AGENT_RUNTIME_CMD`):
- `llm.py` — urllib-only cloud LLM client (`AGENT_LLM=anthropic|openai|kakeya|stub`, auto-detect).
- `run.py` — unattended orchestrator: LLM → brief→script→scene_plan (schema-validated + checkpointed
  via `lib/checkpoint`), assets via `video_selector` (auto-routes to best provider; premium if
  configured, else WAN draft) + `tts_selector` + `music_gen`, deterministic edit → `audio_mixer` +
  `video_compose` (ffmpeg) → final mp4. `AGENT_RUNTIME_FAKE_ASSETS=1` for CI.

**Topology:** agent runtime on the head Mac CPU (cloud LLM, doesn't steal video GPU); headless = video
worker; vast = heavy gen — the cost/stability recommendation.

**Validated offline (no keys/GPU):** stub LLM + ffmpeg fixtures → full plan→validate→checkpoint→
assets→mix→compose → valid h264 mp4 + checkpoints. 23 tests pass (runtime+gateway+pipeline).

**Honest scope:** MVP "auto" path (no research/proposal/approval/reviewer governance). **Needs a
cloud LLM key** to actually run; quality is provider-bounded (WAN draft unless a premium video
provider key is set — then `video_selector` auto-upgrades). Not yet "all skills". Live `mode=agent`
test on agent.kakeya.ai pending the key.

---

## Open follow-ups (next iterations)
- **Phase 2b — native gRPC transport.** Add an optional `kakeya` Python SDK transport
  for the bounded-memory long-context path (W3), behind the same tool, once the proto
  stabilizes and the dependency is opt-in (`pip install kakeya`).
- **Phase 2c — pipeline wiring.** Have `localization-dub` (subtitle translation) and
  `animated-explainer` (image-prompt expansion) call `llm_selector` when available,
  with graceful fallback to a cloud LLM or the host agent.
- **Benchmark harness.** Stand up a real GPU Kakeya server and replace ADR §2.3
  *target* numbers with measured throughput / latency / quality, so the perf claims
  are evidence-backed rather than aspirational.

## Standing honesty checks (do not regress)

1. Never route creative scripting through Kakeya (D6).
2. Never claim a speedup for CPU-only servers (D2).
3. Keep the tools `unavailable` and inert unless the user opts in (D5).
4. Keep OpenMontage free of torch/CUDA/gRPC hard deps in Phase 1 (D5/D7).
