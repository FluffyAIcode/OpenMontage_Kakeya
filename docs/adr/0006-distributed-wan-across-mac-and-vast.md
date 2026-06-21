# ADR 0006 — Distributed WAN inference across a Mac mini + vast GPUs in different regions

- **Status:** Implemented (heterogeneous pipeline) + validated on real GPU
- **Date:** 2026-06-20
- **Deciders:** OpenMontage maintainers
- **Question:** A complete script letting a cloud agent use a **Mac mini GPU** and a **vast
  GPU** (in **different regions**) for **distributed WAN inference**.
- **Related:** ADR 0002 (video gateway), ADR 0004 (coarse-to-fine capstone), ADR 0005 (mac bridge)

---

## 1. The two hard blockers (objective)

| # | Blocker | Consequence |
|---|---------|-------------|
| **B1** | ~~**WAN runs only on CUDA**; the Mac's MLX has no WAN implementation.~~ **CORRECTED by ADR 0008:** WAN 2.1/2.2 *do* run on Apple Silicon via MLX ports (`mlx-video`) or PyTorch **MPS + `mps-conv3d` + fp16**. B1 was true only for *vanilla diffusers-on-MPS without patches* (our stack). | The Mac **can** be a (slow, RAM-bound, single-device) WAN tile worker — see ADR 0008 §5. It is still best used for the text plane given its speed; cross-region tensor-distribution remains impossible (B2). |
| **B2** | **Cross-region forbids tensor/pipeline parallelism.** Splitting one WAN forward needs sub-ms interconnect; cross-region RTT is 10s–100s ms **per exchange**. | Per-denoise-step tensor exchange is hopeless. Only **coarse, latency-tolerant** traffic (prompts, mp4 clips) may cross the wire. |

**Therefore tensor-parallel "WAN across Mac + vast" is impossible.** What *is* feasible is a
**heterogeneous, coarse-grained pipeline** that splits work by capability and keeps all
per-step tensors on-box.

## 2. The design (what runs where)

```
 cloud agent (orchestrator.py)
    1) text: prompt -> per-tile prompts   ──HTTP──►  Mac mini (MLX)  Kakeya text server  [KAKEYA_ENDPOINT]
    2) framework + 4) tile refines (mp4)  ──HTTP──►  vast CUDA worker(s) (worker.py)     [WAN_WORKERS]
    3) upscale+crop   5) weight-map merge -> final.mp4
```

- **Mac mini (MLX, different region):** Kakeya MLX text server (ADR 0005). Specializes the
  prompt into per-tile prompts. **Text only** — its real capability. *Optional*; if
  `KAKEYA_ENDPOINT` is unset/unreachable the orchestrator **skips it (logged), never fakes it.**
- **vast CUDA worker(s) (`worker.py`):** distilled **CausVid proposer** (framework) +
  full-WAN **vid2vid refine** (tiles). **Run ≥2 on co-located CUDA GPUs for real parallel
  refine.** WAN runs *only* here.
- **cloud agent (`orchestrator.py`):** framework → upscale → native-tile crops → **concurrent
  tile dispatch round-robin across workers** → weight-map merge. The framework anchors tile
  overlaps, so independent per-worker refine stays seamless (ADR 0004 capstone) — which is
  exactly what lets different tiles run on different workers.

**Wire traffic:** prompts (bytes) + base64-mp4 clips only. **No per-step tensors ever leave a
box** → region-tolerant.

## 3. Validation (real, on the vast H200)

Ran the orchestrator **on the cloud agent VM**, reaching the vast WAN worker **over an SSH
tunnel** (cross-machine), `KAKEYA_ENDPOINT` unset (Mac skipped honestly):

```
[orch] 1 WAN worker(s); mac_text=no
[orch] worker http://localhost:9000 OK device=cuda
[orch] KAKEYA_ENDPOINT unset -> Mac text plane skipped (base prompt)
[orch] framework (distilled proposer) on worker[0]... (25,480,832,3) in 3.94s
[orch] refining 4 tiles across 1 worker(s)...  (tiles refined, lock-serialized on 1 GPU)
ORCH_DONE {"workers":1,"mac_text":false,"canvas_px":[768,1472],"tiles":"2x2","tile_refine_wall_s":20.6}
```

Output: a **seamless 1472×768** koi-pond video (`tier01_evidence/dwan_distributed_mid.png`),
produced end-to-end by **cloud agent → remote vast CUDA worker**. Confirms the pipeline,
the network transport, and seamless tiled merge in the distributed setting.

- **Distributed (real):** with **N workers** on co-located CUDA GPUs, the orchestrator's
  concurrent round-robin gives parallel tile refine (each worker has its own GPU lock).
  Single-worker here → tiles serialize (one GPU); that is the expected single-GPU behavior.
- **Mac plane:** wired and skip-not-fake; plugging a reachable `KAKEYA_ENDPOINT` (Mac via
  Tailscale) adds the text plane in parallel — not validated here (no reachable Mac), per
  ADR 0005 (the Mac is the owner's local machine).

## 4. Files

- `services/distributed_wan/worker.py` — CUDA WAN tile worker (`/v1/framework`,
  `/v1/refine_tile`, `/healthz`); one per GPU; GPU-locked.
- `services/distributed_wan/orchestrator.py` — cloud-agent orchestrator (Mac text +
  fan-out + merge), stdlib HTTP, region-tolerant.
- `services/distributed_wan/README.md` — cross-region deployment (Tailscale for the Mac,
  worker URLs/tunnels for vast).

## 3b. Live cross-region verification (Iteration 19)

The gRPC plane (ADR 0010) was wired and proven across **two regions**:

- The vast container has **no `/dev/net/tun`**, so tailscaled runs userspace-only (SOCKS5).
  `services/distributed_wan/socks5_forward.py` bridges `localhost:55051` → SOCKS5 → `mac:50051`
  so the orchestrator dials the Mac as a plaintext gRPC endpoint (no socat/ncat needed).
- **Mac Health (real):** `backend=mlx-video device=mlx ops=['framework'] speed=0.12`, ~214 ms
  tailnet RTT — a genuine MLX WAN worker, not a stub.
- **vast CUDA restarted refine-only** (`--ops refine`); cluster = Mac(framework) + vast(refine).
- A live `GenerateFramework` to the Mac surfaced a clean module-path bug
  (`mlx_video.wan_2` → must be `mlx_video.models.wan_2`), fixed in `grpc_worker.py` +
  `mac_setup.sh`. Final pixels require the owner to restart the Mac worker on the fixed code.

## 5. Boundary (do not regress)

- **WAN runs on BOTH CUDA and Apple-Silicon/MLX.** Superseding the earlier B1: ADR 0008 and
  the Iteration-19 live Health confirm a real `mlx-video` WAN worker on the Mac. The Mac is the
  low-res **framework/T2V proposer**; the CUDA box does **vid2vid refine** (mlx-video has no
  vid2vid). For raw multi-GPU WAN throughput, co-located CUDA workers still win.
- **Cross-region = coarse traffic only** (B2): prompts + mp4. Never route per-step tensors
  or spec-decode drafts over WAN (matches ADR 0003/0005).
- **No fake:** unreachable workers are skipped-not-faked; the Mac MLX failure was surfaced as a
  real gRPC `INTERNAL`, never masked.
