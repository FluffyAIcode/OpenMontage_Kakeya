# ADR 0009 — Distributed WAN worker transport: HTTP vs gRPC

- **Status:** Decided (HTTP for the worker/tool plane; gRPC reserved for a LAN data plane)
- **Date:** 2026-06-21
- **Deciders:** OpenMontage maintainers
- **Question:** Why is the distributed WAN worker contract (ADR 0006/0008) **HTTP/JSON**
  and not **gRPC** (which Kakeya itself uses for `RuntimeService`)?
- **Related:** ADR 0006 (distributed Mac+vast), ADR 0008 (MLX worker), Kakeya
  `docs/design/mac-bridge-cloud-agent-access.md` §4.2

---

## 1. The deciding factor: workload shape

The worker calls (`/v1/framework`, `/v1/refine_tile`) are:

- **Coarse-grained** — one call = a whole tile generation: **seconds–minutes** of GPU
  compute, a **few-MB** mp4 payload, **once** per tile.
- **Cross-region + NAT** — the Mac is outbound-only; vast via tunnel/public IP.
- **Latency-tolerant, no per-step tensors** (ADR 0006 B2 forbids fine-grained exchange).

Transport efficiency is therefore **not** on the critical path: the dominant cost is the
diffusion compute, not serialization or framing.

## 2. HTTP vs gRPC for this shape

| Factor | HTTP/JSON (chosen) | gRPC |
|---|---|---|
| Cloud-agent client deps | **stdlib `urllib`, zero install** (matches Kakeya's stdlib-only bridge-client choice) | `grpcio` + generated stubs on **both** ends |
| NAT / tunnel / relay traversal | trivial (HTTP/1.1 over Tailscale / Caddy / Cloudflare / SSH tunnel) | HTTP/2 streaming fussier through relays/proxies |
| Backend-agnostic worker | CUDA-diffusers **and** MLX workers serve the *same* REST | both bound to a compiled proto |
| Debuggability | `curl` / health one-liners | grpcurl / stubs |
| Payload efficiency | base64-mp4 ~+33%, JSON parse | binary protobuf (no inflation) |
| Streaming / multiplexing | request/response (SSE if needed) | native bidi streaming + HTTP/2 mux |

**The two gRPC advantages don't apply here.** Binary efficiency: ~33 % on a 2 MB clip is
microseconds vs **minutes** of diffusion. Streaming/multiplexing: there is no
high-frequency, small-message, per-step traffic to multiplex — that is the *data plane*
case ADR 0006 B2 already rules out across regions. Meanwhile HTTP's wins (zero-dep stdlib
client, easy NAT traversal, backend-agnostic, debuggable) are exactly what a cross-region
heterogeneous plane needs.

## 3. This matches Kakeya's own split

Kakeya uses gRPC for `RuntimeService` because that is the **LLM token data plane**:
high-frequency `AppendTokens` / server-streaming `Generate`, low-latency, tensor/binary.
Kakeya's mac-bridge doc §4.2 concludes the general rule:

> **control/tool plane = coarse + latency-tolerant; data plane = gRPC on a LAN.**

Our WAN tile worker is a **tool/coarse** plane (whole tiles, cross-region) → HTTP. The gRPC
data plane is LAN-scoped and cannot stretch across regions anyway (B2). So HTTP here is
*consistent with* Kakeya's architecture, not contrary to it.

## 4. When gRPC would be the right choice
- A **LAN** setup with genuine **fine-grained** exchange (does not exist for diffusion — no
  token loop), **or**
- making the worker a true **Kakeya fleet node** sharing the `RuntimeService` contract
  (ADR 0005 M3 — LAN-scoped, mTLS), **or**
- **server-streaming progress** (denoise %/partial frames) to the orchestrator — the one real
  upside; but **SSE over HTTP** delivers it without adding grpc/proto deps.

## 5. Decision

- **Worker/tool plane (cross-region, coarse): HTTP/JSON.** Keep the zero-dep stdlib
  orchestrator + backend-agnostic REST worker (CUDA or MLX).
- **Data plane (LAN, fine-grained/streaming): gRPC** — Kakeya's `RuntimeService`, reserved
  for co-located nodes (not our cross-region case).
- If progress streaming is wanted on the worker plane, add **SSE** before reaching for gRPC.

A gRPC worker variant is straightforward to add later (define a `.proto`, dual-serve) **if**
a LAN fleet or fleet-node unification motivates it; it offers **no benefit** for the
current coarse, cross-region workload and costs the stdlib-only client.
