# ADR 0001 — Integrating the Kakeya LLM Inference Engine into OpenMontage

- **Status:** Proposed (Phase 1 implemented behind availability gate)
- **Date:** 2026-06-20
- **Deciders:** OpenMontage maintainers
- **Upstream:** [FluffyAIcode/Kakeya-LLM-Inference-engine](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine) (v0.4 / v0.5-cuda, alpha)
- **Related:** `PROJECT_CONTEXT.md` (architecture), `AGENT_GUIDE.md` (tool contract), `docs/PROVIDERS.md`

---

## 0. Binding engineering guidelines (no fallback / mock / fake / simplify)

> Added 2026-06-20 once real GPU (H200, 144 GB) was provisioned.

The text integration is now proven against a **real, GPU-served Kakeya server**
(real model weights, real token generation). Mock servers remain only as fast unit
checks of pure helpers; they are **not** the correctness gate. A real integration
test must hit the **live Kakeya server** and fail loudly if unreachable — it must
never pass via a stub, a fake response, or a fallback path. When no
`KAKEYA_ENDPOINT` is configured the real test **skips**; it never fakes a pass. See
ADR 0002 §0 for the full statement (shared across the integration).

## 1. Context

The request: *"Integrate the Kakeya inference engine into OpenMontage, leveraging
Kakeya's distributed parallel inference and bounded-memory capabilities to improve
OpenMontage's performance."*

Before designing anything, we have to be precise about what each system actually is,
because the naive framing ("plug a faster inference engine in to make OpenMontage
faster") rests on an assumption that **does not hold**.

### 1.1 What Kakeya actually is

Kakeya is a **memory-bounded local LLM inference server**:

- Long-running process exposing a gRPC `RuntimeService`
  (`CreateSession` / `AppendTokens` / `Generate` (server-streaming) /
  `GetSessionInfo` / `CloseSession`).
- Holds **session state server-side**; its headline property is that the KV-cache
  footprint and per-turn latency stay **bounded over very long conversations**
  (evidence: +0.0093 s latency drift over a 4-hour / 480-turn Mac M4 run).
- Headline throughput claim — **8.45× served throughput at 8 sessions** — comes from
  a **batched scheduler that is CUDA-only**. On Apple-Silicon MLX the multi-tenant
  path is **serial-only** (upstream MLX `B>1, L=1` quantized-kernel bug, per their
  ADR 0014).
- The memory-bounded, recall-preserving engine uses **Gemma-4 26B-A4B** as the
  verifier plus a DFlash proposer — i.e. a large model that needs real VRAM.
- The Python/TS SDKs operate at the **token-id level** (`session.append([ids])`,
  `generate()` yields token ids). There is an OpenAI-compatible HTTP shim
  (`/v1/chat/completions`) but it is **explicitly deprecated** and, critically,
  **pure-autoregressive with no speculative decoding** — "roughly the same speed as
  `transformers`-vanilla AR generation." The perf-bearing path is gRPC + GPU.
- Packaging: **ships from source**, no PyPI/npm release yet; depends on
  `torch` + (MLX or CUDA). Tags `v0.4-mac` / `v0.4-cuda` / `v0.5-cuda`. **0 stars**,
  alpha maturity.

### 1.2 What OpenMontage actually is

OpenMontage is an **agent-orchestrated video production platform**. From
`PROJECT_CONTEXT.md`:

> The AI agent IS the intelligence. Python exists only for tools and persistence.

Concretely:

- **OpenMontage runs no LLM inference of its own.** All reasoning (script writing,
  scene planning, prompt authoring, review) is performed by the **host agent**
  (e.g. Claude/Cursor) at the top of the loop.
- The Python tools call **external generation APIs** — images (FLUX/Imagen/DALL·E),
  video (Veo/Kling/Seedance/HeyGen), TTS (ElevenLabs/Google/OpenAI/Piper/Doubao),
  music (Suno) — plus local FFmpeg/Remotion/HyperFrames composition.
- There is **no `text_generation` / LLM capability family** in the tool registry
  today. Grep confirms it: capabilities are `tts`, `image_generation`,
  `video_generation`, `music_generation`, `video_post`, `analysis`, `avatar`,
  `character_animation`, `enhancement`.

### 1.3 The consequence

**There is no existing LLM-inference workload inside OpenMontage for Kakeya to
accelerate.** "Make OpenMontage faster with distributed inference" is therefore a
*category error* as literally stated: you cannot speed up a token-generation step
that the platform never performs. The bottlenecks in an OpenMontage run are
external API latency (image/video/TTS), FFmpeg/Remotion render time, and agent
think-time — none of which Kakeya touches.

Integrating Kakeya is only meaningful if we **introduce a new, optional local-LLM
workload** that some pipelines can offload to it. That is the design below — framed
honestly, with the dealbreakers called out first.

---

## 2. Honest performance evaluation (do this before building)

The task explicitly asks: *evaluate the performance of the design, identify the real
performance-improvement points, and the technical dealbreakers (技术硬伤).* Here it is.

### 2.1 Technical dealbreakers (硬伤) — why the headline pitch does not deliver

| # | Dealbreaker | Why it matters for OpenMontage |
|---|-------------|-------------------------------|
| **D1** | **No inference workload exists to accelerate.** | OpenMontage doesn't generate tokens; the host agent does. Kakeya can only help a *net-new* offloaded text task we choose to add. This is added surface, not an optimization of a hot path. |
| **D2** | **The two perf features require a GPU the typical user doesn't have.** | Batched 8.45× throughput is **CUDA-only**; bounded-memory recall engine needs **Gemma-4 26B** (tens of GB VRAM). On a CPU box or the default cloud VM you get tiny Qwen3-0.6B, AR-only — *no* throughput win and *no* meaningful bounded-memory benefit. |
| **D3** | **Workload-shape mismatch.** | Kakeya optimizes **long-lived multi-turn sessions** (4 h / 480 turns, bounded KV drift). OpenMontage's plausible text needs are **short, stateless, batchy** (expand 30 image prompts, translate 40 subtitle lines, write 12 caption variants). The bounded-KV-over-long-conversation property is irrelevant to stateless batch jobs. |
| **D4** | **The text-friendly entrypoint is the deprecated, perf-stripped one.** | The OpenAI HTTP shim (text-in/text-out, trivial to call) is *deprecated* and *has no speculative decoding* ("≈ transformers-vanilla"). The perf-bearing gRPC path is **token-level**, forcing OpenMontage to own per-model tokenizers/chat-templates and a generated-protobuf dependency on a moving alpha API. |
| **D5** | **Heavy operational footprint vs. OpenMontage's current "agent + ffmpeg + API keys" baseline.** | A long-running gRPC server + multi-GB model download + `torch`/MLX/CUDA is a large new dependency for a video tool. Adds cold-start, lifecycle, and ops burden. |
| **D6** | **Quality regression risk if it ever touches creative reasoning.** | A local Qwen3-0.6B/Gemma is far weaker than the host agent. Routing *creative* decisions (script, hook, scene narrative) through it would **degrade** output. Kakeya must be confined to mechanical text transforms only. |
| **D7** | **Alpha stability / API churn.** | 0 stars, v0.4, no stable release, source-only install, gRPC proto still evolving. Pinning OpenMontage to it is a maintenance liability. |

**Verdict on the headline claim:** "distributed parallel inference + bounded memory →
faster OpenMontage" is **not real** for the default user. It is conditionally real
only for a narrow, self-hosted, GPU-equipped, batch-text use case.

### 2.2 The real (narrow but genuine) improvement points

Where a *user who already runs a GPU Kakeya server* gets actual value:

| # | Real win | Mechanism | Which Kakeya feature is actually used |
|---|----------|-----------|----------------------------------------|
| **W1** | **Zero marginal cost / private batch text** | Offload high-volume mechanical text (image-prompt expansion, subtitle translation, caption/hook variants, alt-text) to a local model instead of paying a cloud LLM API per token, and keep content on-device. | Local serving (the *server* itself), not the exotic features. |
| **W2** | **Throughput on embarrassingly-parallel batch jobs** | A `localization-dub` run translating hundreds of subtitle lines, or `clip-factory` writing dozens of caption variants, is exactly the multi-request fan-out the **CUDA batched scheduler** is built for (8.45× @ 8 concurrent). | Batched scheduler (**CUDA only** — honestly gated). |
| **W3** | **Long-context script analysis without OOM** | Feeding a very long source transcript (podcast-repurpose) into one prompt benefits from the **bounded sliding-window KV + restoration**: process long context without KV blow-up. | Bounded-memory KV / restoration (GPU restored-Gemma path). |

These are all **batch, mechanical, non-creative** text tasks. None replace the host
agent's reasoning. W2/W3 only materialize on GPU; W1 (cost/privacy) holds even on CPU
albeit at lower quality.

### 2.3 Performance targets & how we'd measure them

We will **not** claim numbers we cannot reproduce. The integration ships with a
benchmark harness (`tests/eval` style) and these *target* metrics, to be filled in
against a real server:

| Metric | Definition | Target (GPU/CUDA) | Reality (CPU/0.6B) |
|--------|------------|-------------------|--------------------|
| Batch prompt-expansion throughput | prompts/sec for N=32 image prompts | ≥ 4× vs serial single-request loop | ~1× (no batching) |
| Per-token cost | USD/1k output tokens vs cloud LLM | $0 marginal | $0 marginal |
| Subtitle translation latency | wall-clock for 200 lines | bounded by batch width | linear, slow |
| Long-context stability | KV bytes vs context length | bounded (flat) | n/a (tiny ctx) |
| Quality | human pass-rate on prompt usefulness | ≥ cloud-LLM baseline only on large models | **below baseline** on 0.6B |

The honest headline: **on the hardware most users have, the only guaranteed win is
cost/privacy, not speed.** Speed wins are GPU-gated and we label them as such in the
tool's `not_good_for` and `supports` fields so the agent never over-promises.

**Measured (Iteration 4, real H200 + real Qwen3-0.6B over the HTTP shim):** real
`generate` returns real completions with real token usage; sequential batch returns
4/4 real completions. **But the deprecated HTTP shim is single-session — concurrent
requests return HTTP 500** (verified 3/4 failures at concurrency=4). So the
client-side concurrent fan-out (W2) is *gated on Kakeya's gRPC multi-tenant/CUDA
path*, not the shim; the tool now defaults batch concurrency to 1 accordingly. See
`docs/adr/0001-kakeya-integration-loop-log.md` Iteration 4 + I8.

---

## 3. Decision

Adopt a **minimal, optional, strictly-gated** integration:

1. Add a new capability family **`text_generation`** to the tool registry.
2. Ship a provider tool **`kakeya_llm`** that talks to a **user-run** Kakeya server.
   - Default transport: the **OpenAI-compatible HTTP endpoint** (text-in/text-out,
     `requests` only — already a core dep). This is deprecated upstream but is the
     only stable text interface and needs no protobuf/tokenizer coupling. We document
     the tradeoff loudly.
   - The tool is **unavailable unless `KAKEYA_ENDPOINT` is set and reachable** — no
     silent install, no bundled server, no torch/CUDA added to OpenMontage's deps.
3. Ship a selector **`llm_selector`** (capability router) so future local/cloud LLM
   providers slot in via the same auto-discovery pattern as `tts_selector`.
4. Confine usage to **mechanical batch text** (prompt expansion, translation, caption
   variants, alt-text). The tool's contract explicitly marks creative scripting as
   `not_good_for`. **The host agent remains the creative intelligence (D6).**
5. Be honest in the tool metadata about GPU-gating of the speed features (D2) and
   alpha status (D7).

### What we explicitly do NOT do

- We do **not** vendor Kakeya, add `torch`/MLX/CUDA, or auto-start a server.
- We do **not** route the agent's script/scene-plan reasoning through Kakeya.
- We do **not** claim a speedup for the default (no-GPU) user.
- We do **not** take a hard dependency on Kakeya's gRPC proto in Phase 1.

---

## 4. Design

```
┌────────────────────────────────────────────────────────────┐
│ Host Agent (creative intelligence — unchanged)              │
└───────────────┬────────────────────────────────────────────┘
                │ calls a tool for MECHANICAL batch text only
                ▼
        ┌───────────────┐      auto-discovers capability="text_generation"
        │ llm_selector  │──────────────────────────────────────────────┐
        └──────┬────────┘                                               │
               │ routes to best available provider                     ▼
               ▼                                              (future: other
        ┌───────────────┐   HTTP /v1/chat/completions          local/cloud LLM
        │  kakeya_llm   │ ───────────────────────────────►      providers)
        └───────────────┘   (or batch fan-out)
               │
               ▼
   user-run Kakeya server  ($KAKEYA_ENDPOINT)  ── GPU? batched throughput
   (NOT started/owned by OpenMontage)          ── CPU? cost/privacy only
```

- `kakeya_llm.execute()` supports two operations:
  - `generate` — one prompt → one completion.
  - `batch` — list of prompts → list of completions (the throughput case W2).
- Availability = `KAKEYA_ENDPOINT` env set **and** a fast health probe succeeds.
- Cost is reported as `$0` (local), with a note that GPU electricity isn't modeled.

---

## 5. Consequences

**Positive**

- OpenMontage gains an optional, free, private text backend for batch chores.
- Net-new capability family with a clean selector seam for future LLM providers
  (local llama.cpp, Ollama, cloud) — the integration is *not* Kakeya-specific at the
  selector layer.
- No new heavy dependencies; default users are unaffected (tool simply reports
  `unavailable`).

**Negative / risks**

- Real speed benefit is GPU-gated and alpha-dependent (D2, D7).
- Adds a provider surface that must be maintained against a moving upstream.
- Temptation to over-use it for creative tasks must be resisted (mitigated by
  `not_good_for` metadata + this ADR).

**Follow-ups (the loop)**

- Phase 2: optional native gRPC transport (`kakeya` Python SDK) behind the same tool,
  for the bounded-memory long-context path (W3), once the proto stabilizes.
- Phase 2: wire `localization-dub` subtitle translation and `animated-explainer`
  prompt-expansion to call `llm_selector` when available, with cloud/agent fallback.
- Benchmark harness against a real GPU server to replace §2.3 targets with measured
  numbers.

See `docs/adr/0001-kakeya-integration-loop-log.md` for the iterate→test→improve log.
