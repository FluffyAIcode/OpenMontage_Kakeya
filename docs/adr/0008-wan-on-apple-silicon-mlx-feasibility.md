# ADR 0008 — Feasibility of running WAN 2.1 on Apple Silicon (MLX / MPS)

- **Status:** Evaluation (corrects ADR 0005 D5 / ADR 0006 B1)
- **Date:** 2026-06-21
- **Deciders:** OpenMontage maintainers
- **Question:** How feasible is it to **port WAN 2.1 to MLX** (Apple Silicon)?
- **Related:** ADR 0005 (mac bridge), ADR 0006 (distributed Mac+vast), ADR 0007 (time-division)

---

## 1. Headline: it's not just feasible — it already exists

A from-scratch MLX port of WAN 2.1 is **unnecessary**. WAN 2.1/2.2 already run on Apple
Silicon via maintained projects:

| Project | What it provides |
|---|---|
| [`Blaizzy/mlx-video`](https://github.com/Blaizzy/mlx-video) | MLX-native **Wan2.1 (1.3B/14B T2V)** + Wan2.2; flow-matching + CFG; LoRA incl. Wan2.2-Lightning 4-step; inference **and** finetuning. |
| [`antonpetrovmain/Wan2.2-mlx`](https://github.com/antonpetrovmain/Wan2.2-mlx) | **Pure MLX** port of Wan2.2 (all PyTorch removed); single-device, unified memory. |
| [`lpalbou/mlx-gen`](https://github.com/lpalbou/mlx-gen) (mflux fork) | Wan2.2 T2V/I2V incl. **TI2V-5B BF16/q8**, A14B; mixed quantization. |

And **without any port**, PyTorch's **MPS** backend runs diffusers `WanPipeline` once two
known issues are handled:
- **bf16 unsupported on MPS** → use **fp16** (or fp32).
- **Conv3D was CUDA-only on MPS** → [`mps-conv3d`](https://github.com/mpsops/mps-conv3d)
  (`pip install mps-conv3d; patch_conv3d()`) provides a native **Metal Conv3D** kernel,
  forward+backward, fp16/fp32, M1–M4. This was historically *the* blocker (the Wan-VAE's
  3D causal convs); it is now solved.

## 2. Correction to earlier ADRs (honest)

ADR 0005 (D5) and ADR 0006 (B1) stated *"WAN cannot run on the Mac / MLX cannot run WAN."*
That is **wrong as a general claim** and is corrected here. It was accurate only for the
**vanilla diffusers `WanPipeline` on `torch.device("mps")` without patches** — which is the
stack OpenMontage uses — where bf16 + Conv3D-on-MPS both fail. In reality, **WAN runs on
Apple Silicon today** via (a) dedicated MLX ports, or (b) PyTorch MPS + `mps-conv3d` + fp16.

What does **not** change from ADR 0006:
- **B2 (cross-region) still holds:** you cannot tensor/pipeline-distribute a single WAN
  forward across a Mac + a vast box over WAN latency. That conclusion is latency-driven and
  unaffected by the Mac now being able to run WAN locally.
- **MLX has no multi-device/distributed inference** ("Multi-GPU and distributed inference are
  not currently supported in MLX" — `Wan2.2-mlx`): a Mac MLX node is a single unified-memory
  serial generator.

## 3. Port-from-scratch effort (for completeness; NOT recommended)

If one ignored the existing ports, a clean-room MLX port breaks down as:

| Component | Effort | Risk |
|---|---|---|
| **Scheduler** (flow-matching / UniPC) | trivial | none — pure array math |
| **UMT5-XXL text encoder** | moderate | low — standard T5 encoder; re-impl in `mlx-nn` + weight convert |
| **WAN DiT transformer** | moderate–high | low–med — patch embed, **3D RoPE**, self+cross attention (`mlx.fast.scaled_dot_product_attention`), AdaLN modulation, FFN × 30/40 blocks; reference = diffusers `WanTransformer3DModel`; weight-name mapping |
| **Wan-VAE (3D causal conv + tiling)** | high | **was the blocker** — needs solid **conv3d**; MLX conv3d + the causal/tiling logic. Now de-risked (existing ports prove it; `mps-conv3d` proves the kernel) |
| **Weight conversion** | moderate | low — safetensors → MLX, name remap |
| **Numerical validation** | moderate | per-layer parity vs diffusers, then end-to-end |

Verdict: **a real but standard DiT-port project — and entirely redundant**, since
`mlx-video` already did it (including finetuning + LoRA).

## 4. Practical constraints (why "runs" ≠ "fast")

- **Speed:** Apple Silicon GPU ≪ H200. `mlx-gen` advertises "5-second M5 Max clips" — and the
  M-Max is far above a Mac **mini** (M4). Expect **minutes** per clip for 1.3B, much worse for
  14B. It runs locally/privately, slowly.
- **Memory (unified):** UMT5-XXL ≈ 11 GB fp16 + transformer (1.3B ≈ 2.6 GB / 14B ≈ 28 GB) +
  VAE must fit in RAM. **1.3B feasible on ≥ 32 GB Macs; 14B needs ≥ 64 GB and q8.** A 16 GB
  Mac mini is too small for the full 1.3B + UMT5-XXL comfortably.
- **Single device:** no MLX multi-GPU; one Mac = one serial worker.

## 5. Architectural upshot — the Mac can now be a (slow) WAN *tile* worker

This **upgrades ADR 0006**. Previously the Mac was "text plane only." Now, because WAN runs
on Apple Silicon, the Mac can also be a **coarse-grained WAN tile worker** in the
distributed task-parallel pipeline (ADR 0006 §2): it implements the **same worker HTTP
contract** (`/v1/framework`, `/v1/refine_tile`) but with an **MLX backend** (wrapping
`mlx-video`) instead of CUDA diffusers. (Why HTTP and not gRPC for this contract — given the
coarse, cross-region workload — is decided in **ADR 0009**.) The orchestrator then treats Mac + vast workers
uniformly, with **speed-weighted tile assignment** (give the slow Mac fewer/smaller tiles).

Still valid:
- **Coarse-grained only** (whole tiles/clips per worker; latency-tolerant; no per-step
  tensors over the wire) — ADR 0006 B2.
- The framework anchors tiles → independent per-worker refine stays seamless (ADR 0004
  capstone), so a heterogeneous Mac+vast tile split integrates cleanly via f_θ.

Not built here (needs a Mac to test; the owner's machine — ADR 0005): an MLX-backed worker is
a documented follow-up. The worker contract (ADR 0006 §5) is backend-agnostic, so it slots in.

## 6. Recommendation

1. **Do not port from scratch.** Use **`mlx-video`** (Wan2.1 1.3B/14B, MLX) on the Mac, or
   **diffusers + MPS + `mps-conv3d` + fp16** for a no-port path.
2. For OpenMontage, the highest-value Mac role is still the **text plane** (ADR 0005) given
   the Mac's slowness for video; but a **speed-weighted MLX WAN tile worker** (ADR 0006 §2,
   this ADR §5) is now a legitimate option to add the Mac as a (slow) video contributor.
3. Keep video's heavy lifting on CUDA (vast/H200); use the Mac for local/private/offline 1.3B
   generation or as a supplementary tile worker — never as a cross-region tensor peer (B2).

## 7. Net answer to "feasibility of porting WAN 2.1 to MLX"

**Highly feasible and already done.** The historical hard part (3D-conv VAE on Apple Silicon)
is solved both in MLX (existing ports) and in PyTorch MPS (`mps-conv3d`). The remaining
limits are **performance and memory** (slow, RAM-bound, single-device), not feasibility.
