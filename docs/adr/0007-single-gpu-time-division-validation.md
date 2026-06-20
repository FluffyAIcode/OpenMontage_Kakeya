# ADR 0007 — Single-GPU time-division validation of proposer / verifier / f_θ

- **Status:** Validated on real GPU (H200, single device)
- **Date:** 2026-06-20
- **Deciders:** OpenMontage maintainers
- **Question:** On ONE GPU, validate the full architecture by **time-division**: proposer
  builds the framework → split into parts → verifier completes part 1, **then** part 2
  (sequential, time-shared GPU) → **f_θ** integrates → full video. Is it feasible?
- **Related:** ADR 0004 (coarse-to-fine), ADR 0006 (distributed)
- **Script:** [`services/distributed_wan/time_division_2part_wan.py`](../../services/distributed_wan/time_division_2part_wan.py)

---

## 1. What was run (real WAN 2.1 1.3B, single H200)

```
PROPOSER (distilled CausVid, 6-step) -> low-res framework (832x480)
  -> upscale to a wide canvas, split into N overlapping native-832 parts
VERIFIER (full WAN vid2vid) refines part 1, THEN part 2, ... (TIME-DIVISION on one GPU)
f_θ (boundary-consistency weight-map) integrates the parts -> full wide video
```

Two configurations, plus a full-canvas single-pass **memory reference**:

| Run | Canvas | Proposer | Verifier (time-division) | f_θ seam-excess | Peak GPU |
|---|---|---|---|---|---|
| 2-part | 1472×480×25 | 2.7 s | 4.4 + 4.4 = 8.7 s | **1.17** (≈1 → seamless) | 24.1 GB |
| 4-part | 2752×480×49 | 5.6 s | 4 × 9.5 = 38.0 s | **1.24** (≈1 → seamless) | 24.4 GB |

Evidence: `tier01_evidence/timediv_2part_mid.png` (seamless 1472×480),
`timediv_4part_mid.png` (seamless **2752×480**, 3.3× native width).

## 2. Findings

### 2.1 Feasibility — VALIDATED ✓

The full proposer → time-division verifier → f_θ pipeline works end-to-end on **one GPU**
and produces **seamless beyond-native-resolution** video (832-wide native → 1472 and 2752
wide). f_θ seam-excess ≈ 1.0 at every part boundary = no visible stitch. Per-part verifier
time is constant (~9.5 s @ 49f); N parts cost **N × per-part** (linear time, as expected for
time-division on a single GPU). The framework anchors the parts so sequential independent
refinement integrates seamlessly (ADR 0004 capstone, confirmed in the time-division form).

### 2.2 Bounded memory — NOT realized on this GPU (honest)

Peak GPU memory is **constant ~24 GB regardless of #parts / frames / canvas width** —
identical for a single part **and** for the full-canvas single pass, which did **not OOM**
even at 2752×480×49. So **time-division did not reduce peak memory here.**

Why: WAN's memory is **already bounded** independent of time-division —
- the **WAN-VAE** is designed for bounded-memory, unlimited-length encode/decode (per its
  model card), so VAE memory doesn't blow up with canvas size;
- attention is **SDPA** (≈linear in tokens), not a materialized O(tokens²) matrix;
- the bulk of the 24 GB is **resident weights** (UMT5-XXL text encoder + transformer + fp32
  VAE), which are constant.

So on the 140 GB H200, the full canvas fits at a constant ~24 GB and time-division saves
nothing. Time-division's memory benefit is **conditional**: it only matters on a GPU **too
small to hold that constant footprint** (e.g., a 16/24 GB consumer card, or the WAN-14B
model), where a part fits but the full canvas does not — there, time-division is the enabler
(and the full-canvas pass would OOM, which it did not here).

### 2.3 What time-division actually buys on a large GPU

- **Resolution scaling beyond native** (832 → 2752 wide) on one GPU.
- **Seamless f_θ integration** of the parts.
- **Linear-time tiling** — N parts in N × per-part time; trivially convertible to
  **parallel** across GPUs (ADR 0006) when more devices exist.
- **NOT** a memory saving on a GPU that already holds the full pass.

## 3. Verdict

The architecture is **feasible and validated** on a single GPU via time-division, producing
seamless beyond-native-resolution video. The **bounded-memory** property is a real but
**conditional** benefit — it pays off only on memory-constrained GPUs (small cards / larger
models), not on the H200 where WAN's own design already bounds memory. For the H200, the win
is resolution scaling + seamless integration + (with more GPUs) parallel speedup, not memory.

## 4. Boundary (do not regress)

- Time-division gives **feasibility + resolution scaling**, and **memory headroom only on
  small GPUs**. Don't claim memory savings on a GPU that already fits the full pass.
- f_θ here = boundary-consistency weight-map merge; it stays seamless because the framework
  anchors the parts (moderate refine strength) — ADR 0004 capstone.
- WAN remains CUDA-only; this is all on one CUDA GPU (the Mac cannot run WAN — ADR 0006).
