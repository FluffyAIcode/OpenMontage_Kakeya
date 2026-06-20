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
