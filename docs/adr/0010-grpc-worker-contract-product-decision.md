# ADR 0010 — Worker contract is gRPC (product decision; supersedes ADR 0009's recommendation)

- **Status:** Decided + transport validated. Supersedes ADR 0009's HTTP recommendation.
- **Date:** 2026-06-21
- **Deciders:** OpenMontage maintainers (product directive: "usable product, not a toy → use gRPC")
- **Related:** ADR 0006 (distributed Mac+vast), ADR 0008 (MLX worker), ADR 0009 (HTTP-vs-gRPC analysis)
- **Implementation:** `services/distributed_wan/{proto/video_worker.proto, grpc_worker.py, grpc_orchestrator.py, mac_setup.sh}`

---

## 1. Decision

The distributed WAN worker contract is **gRPC** (`distwan.v1.VideoWorker`), not HTTP/JSON.

## 2. Why this overrides ADR 0009

ADR 0009 recommended HTTP, but explicitly scoped its key argument — *"zero-dep stdlib
client"* — to a **throwaway validation** context. For a **product**, the weights flip:

| Concern | Product value of gRPC |
|---|---|
| **Typed, versioned contract** | one `.proto` is the source of truth for **two heterogeneous backends** (CUDA diffusers + MLX/mlx-video); prevents drift. |
| **Server-streaming progress** | minutes-long generations need live progress; `RefineTile`/`GenerateFramework` stream `Progress{pct,stage}` then the final `mp4`. (HTTP needed SSE bolted on.) |
| **Deadlines / cancellation / backpressure** | first-class in gRPC — real for a product that must time out and cancel runs. |
| **Capability negotiation** | `Health.ops` + `relative_speed` lets the orchestrator route by what each backend supports and how fast it is (CUDA vs slow Mac). |
| **Binary payloads** | mp4 bytes sent raw (no base64 +33%); matters as tiles/resolution scale. |
| **Fleet-unification path** | aligns with Kakeya's own gRPC `RuntimeService` (ADR 0005 M3) if the worker ever becomes a fleet node. |

The "stdlib client" convenience is irrelevant to a product (it can depend on `grpcio`), and
the cross-region HTTP/2 worry from ADR 0009 is moot over **Tailscale** (flat network). So for
a product, gRPC is the right call. ADR 0009's analysis remains correct *for its scope*; this
ADR is the product decision.

## 3. Contract (`proto/video_worker.proto`)

```
service VideoWorker {
  rpc Health(HealthRequest) returns (HealthReply);                  // ops[] + relative_speed
  rpc GenerateFramework(FrameworkRequest) returns (stream Progress);// distilled proposer
  rpc RefineTile(RefineRequest) returns (stream Progress);          // full-model vid2vid refine
}
Progress { float pct; string stage; bool done; bytes mp4; float gen_seconds; }
```

Two backends implement it (capability-advertised):
- **`cuda`** (vast/H200): diffusers WanPipeline + CausVid LoRA; ops `framework, refine, t2v`;
  `relative_speed=1.0`. The model code is the **already-validated** ADR 0006 path.
- **`mlx`** (Mac mini): wraps `mlx-video` (ADR 0008); ops advertised via `--mlx-ops` (default
  `framework`; add `refine` only if your mlx-video has vid2vid); `relative_speed≈0.12`.
  **Owner-run** (`mac_setup.sh`) — the cloud agent can't reach a local Mac (ADR 0005).

The orchestrator (`grpc_orchestrator.py`): `Health()` all workers → runs the framework on the
fastest framework-capable worker → **speed-weighted** tile assignment across refine-capable
workers → concurrent streamed `RefineTile` → f_θ weight-map merge.

## 4. Validation

**gRPC transport — validated locally (no GPU), end-to-end:**
- Two `test`-backend workers (speeds 3.0 "cuda" and 1.0 "mlx").
- Capability negotiation OK; **speed-weighted routing gave the exact 3:1 split** (cuda 3 tiles,
  mlx 1 tile of 4); **server-streamed progress** (25→100% per tile, interleaved across
  concurrent workers); concurrent dispatch + f_θ merge → a 1472×768 mp4. ✓

**CUDA model backend:** unchanged diffusers code already validated on real GPU (ADR 0006 →
real h264 video). The transport swap (HTTP→gRPC) is what's new and is now proven.
**Pending:** a CUDA-over-gRPC real-video re-run on vast — the vast H200 was **unreachable**
(connection reset) at decision time; rerun `grpc_worker.py --backend cuda` there when it
returns. **MLX backend:** owner-run on the Mac; not testable here (no Mac).

## 5. Boundary (unchanged)

- gRPC here rides **Tailscale/tunnels**; still **coarse-grained** (whole tiles, latency-tolerant)
  — no per-step tensors over the wire (ADR 0006 B2). The streaming is *progress*, not tensors.
- WAN stays CUDA for heavy work; the Mac is a slow MLX worker / text plane (ADR 0008).
