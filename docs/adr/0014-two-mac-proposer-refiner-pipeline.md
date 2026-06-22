# ADR 0014 — Two-Mac proposer→refiner pipeline (head proposes, headless refines)

**Status:** Accepted; implemented (offline-validated).
**Date:** 2026-06-22
**Related:** ADR 0004 (coarse-to-fine WAN), ADR 0006/0010 (distributed-WAN gRPC), ADR 0008 (MLX
feasibility — no vid2vid), ADR 0011 (agent gateway), ADR 0013 (re-anchor). Extends the two-Mac
Thunderbolt deployment from "pool throughput" to a "proposer→refiner pipeline" topology.

## 1. Problem

The two Mac minis ran in **pool** mode: each Mac served an independent DIRECT (no-refine) job, giving
2× throughput but no cross-Mac collaboration on a single clip. The ask: run the cluster
**coarse-to-fine** — **head Mac = proposer** (low-res framework/T2V), **headless Mac = refiner** —
so one clip benefits from both GPUs (a draft pass + a refine/upscale pass).

## 2. Constraint (the honest blocker)

Stock `mlx-video` (the WAN port) has **no vid2vid** and no tensor-parallel/distributed support
(ADR 0008). The orchestrator's refine path was a 2×2 **tiled CUDA V2V** flow (built for a diffusers
WAN `WanVideoToVideoPipeline` on a CUDA box), which does not fit a single headless Mac. The MLX
worker's `refine` op therefore hard-failed unless `MLX_V2V_FLAG` pointed at a vid2vid build.

So a *generative* refine on the headless Mac is **not possible with stock mlx-video**. What *is*
possible — and genuinely uses the proposer→refiner split — is a **spatial super-resolution refine**.

## 3. Decision

Add a single-pass pipeline with an MLX-native, non-generative refine, and keep the generative paths
(CUDA tiled, or MLX V2V if available) intact:

```
PIPELINE (two Macs, mlx-video):
  head (framework) ──low-res proposer──► temporal resample ──► headless (refine)
                                                                  │  SR: Lanczos↑ + unsharp
                                                                  ▼  (or generative V2V if MLX_V2V_FLAG)
                                                              final mp4 @ out-res
```

1. **MLX `refine` works without vid2vid** (`grpc_worker.py::MlxBackend`):
   - `MLX_V2V_FLAG` set → generative V2V (unchanged).
   - else → **`_sr_refine`**: Lanczos upscale to `(out_width,out_height)` + `UnsharpMask`
     (`MLX_SR_SHARPEN`, default 60; 0 disables). Pure PIL/numpy — OOM-safe, no Metal, CI-testable.
   - `health().note` advertises `refine=sr` vs `refine=v2v` (no silent misrepresentation).
2. **Orchestrator single-pass pipeline** (`grpc_orchestrator.py`): `--single-refine` (auto-on when
   the chosen refiner's backend is MLX) + `--out-width/--out-height`. Proposer on the fastest
   framework worker → one full-frame `RefineTile` on the refiner at the target resolution. Prefers a
   **distinct** proposer so both Macs are engaged. CUDA refiners still use the tiled generative path.
3. **Gateway pipeline mode** (`agent_gateway/server.py`): `>1 worker` + pool OFF ⇒ `PIPELINE_MODE`;
   one job spans both Macs and the orchestrator is called with `--single-refine`. `/healthz` reports
   `{"mode":"pipeline|pool|single","pipeline_mode":bool}`.
4. **Roles via `--mlx-ops`**: `framework` (proposer) / `refine` (refiner) / `framework,refine`
   (single-Mac coarse-to-fine). Deploy templates updated (launchd plists, `mac_all_in_one.sh`
   `MODE=pipeline|pool`, `two-mac-thunderbolt.md`).

## 4. What this buys (and what it does not)

- ✅ A real two-Mac coarse-to-fine path: head drafts low-res fast; headless upscales/refines the whole
  clip to a higher output resolution in one pass. Higher-res than a single Mac's DIRECT draft.
- ✅ Both GPUs engaged on one clip; no tile-seam machinery; OOM-safe on unified memory.
- ⚠️ **The MLX refine is interpolative SR, not generative** — it raises resolution and crispness but
  does not synthesize new detail. For generative detail: attach a CUDA refiner (orchestrator
  auto-promotes to the tiled V2V pipeline) or an mlx-video build with vid2vid (`MLX_V2V_FLAG`).
- Trade-off vs pool: pipeline = one clip at a time, both Macs busy (quality); pool = N clips in
  parallel (throughput). Selectable per deployment; no code change to switch.

## 5. Validation

`tests/tools/test_distributed_wan_pipeline.py` (offline, no GPU):
- `MlxBackend` SR refine upscales a tiny mp4 to the target resolution without vid2vid; `health` note
  reports `refine=sr`.
- **End-to-end over real gRPC**: two in-process `TestBackend` workers (framework + refine) +
  `grpc_orchestrator --single-refine` → mp4 at out-res, `ORCH_DONE mode=pipeline`, proposer=A,
  refiner=B.
- Gateway pipeline test: asserts `--single-refine`, all workers passed, no `--no-refine`, healthz
  `mode=pipeline`.

10/10 pass with the existing gateway suite.

## 6. Follow-ups

- Benchmark real SR-refine output quality/latency on the 512 G Macs vs DIRECT and vs CUDA V2V.
- If/when an mlx-video vid2vid build is available, wire `MLX_V2V_FLAG` and compare generative MLX
  refine to the SR fallback.
- Parameterize the proposer→output scale factor and optionally chain SR (e.g. 2× then 2×).
