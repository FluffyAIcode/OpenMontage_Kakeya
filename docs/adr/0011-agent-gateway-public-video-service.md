# ADR 0011 — Public agent video service via a domain (kekaye.ai)

**Status:** Accepted (gateway implemented + tested; live exposure depends on owner DNS/Funnel).
**Date:** 2026-06-21
**Related:** ADR 0002 (video inference gateway), ADR 0006/0010 (distributed-WAN gRPC cluster),
skill `macmini2GPUcloud`.

## 1. Goal

Let users use **OpenMontage's agent video service directly over a domain** (e.g.
`https://kekaye.ai`) — submit a request in a browser/API and get a video back, powered by the
distributed-WAN cluster (Mac-mini MLX proposer + cloud CUDA refiner) we built and validated.

## 2. The architectural tension (and how we honor it)

OpenMontage is **agent-first**: "the AI agent IS the intelligence; Python is only tools +
persistence. No Python orchestrator, no Python reviewer" (`PROJECT_CONTEXT.md`,
`AGENT_GUIDE.md` Rule Zero). A naive "video API" that reimplements pipeline/creative decisions in
Python would violate this.

**Resolution:** the gateway is a **transport + session/job layer only**. It does NOT make
creative or orchestration decisions. It:
1. accepts a request and opens a job (persistence),
2. for `mode="video"`, shells out to the already-validated
   `services/distributed_wan/grpc_orchestrator.py` (a capability/tool call, not orchestration),
3. for `mode="agent"`, enqueues the brief for an **external agent runtime** (`AGENT_RUNTIME_CMD`)
   — the LLM agent + skills remain the intelligence; the gateway just manages the job and files.

This keeps Python in its allowed role (tools + persistence + transport) and leaves the creative
intelligence with the agent.

## 3. Design

```
  Browser / API  ──HTTPS──►  Caddy (auto-TLS, kekaye.ai)  ──►  agent_gateway (FastAPI :8088)
                                                                    │ subprocess
                                                                    ▼
                                              distributed_wan/grpc_orchestrator.py
                                                 │ gRPC                  │ gRPC (socks5_forward / tailnet)
                                                 ▼                       ▼
                                           CUDA worker (refine)     Mac mini MLX worker (framework)
```

- **Service:** `services/agent_gateway/server.py` (FastAPI). Endpoints: `/`, `/healthz`,
  `/v1/capabilities`, `POST /v1/videos`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/video`.
- **Jobs:** in-process store + on-disk mp4 under `projects/_gateway_jobs/`; a single-worker
  executor serializes GPU work (the cluster is one GPU per role).
- **Progress:** the gateway parses the orchestrator's stdout (`… N%`, `ORCH_DONE {json}`) into
  `stage`/`pct`/`log`, surfaced to a minimal web UI and the JSON API.
- **Auth:** optional `AGENT_GATEWAY_API_KEY` → `X-API-Key` required on `POST /v1/videos`.
- **TLS/domain:** Caddy (`deploy/Caddyfile`) gets a Let's Encrypt cert for `kekaye.ai`
  automatically; or Tailscale **Funnel** for instant public HTTPS on a NAT'd GPU box.

## 4. What is real vs. owner-dependent

- **Real, working today:** the gateway, the job lifecycle (tested offline, 6 passing tests with a
  fake orchestrator), and `mode="video"` driving the live Mac+vast cluster.
- **Owner-dependent:** pointing the `kekaye.ai` DNS A-record at a host (registrar/Cloudflare) and
  opening 80/443 — the cloud agent cannot control external DNS. Documented in `README.md`. A
  Tailscale-Funnel `*.ts.net` URL is the zero-DNS fallback for an immediate public demo.
- **Not faked:** `mode="agent"` (full multi-stage pipeline) returns an honest "agent runtime not
  attached" status unless `AGENT_RUNTIME_CMD` is set — the gateway never fabricates creative work.

## 4b. Live verification

Deployed on the vast H200 box against the live Mac+vast cluster and exercised through the HTTP API:

```
POST /v1/videos {"prompt":"a sea turtle gliding over a coral reef, sunbeams"}  (X-API-Key)
  -> job 8f4b540d…  -> framework on Mac MLX -> 4 tiles refined on vast CUDA
GET  /v1/jobs/8f4b540d…        -> {"status":"done","pct":1.0,...}
GET  /v1/jobs/8f4b540d…/video  -> HTTP 200, real h264 1472×768 × 25-frame mp4
```

![Agent gateway result — produced via POST /v1/videos](tier01_evidence/gateway_demo_mid.png)

The gateway → distributed-WAN → mp4 path is real end-to-end. Public exposure (`tailscale funnel`
or a `kekaye.ai` A-record + Caddy) is the only remaining owner step — Funnel/HTTPS must be enabled
by the tailnet admin; external DNS is out of the cloud agent's control.

## 5. Honest limits

- A capable autonomous `mode="agent"` needs a strong reasoning LLM in the loop; the local Kakeya
  text plane (small models) is not sufficient to drive full pipelines reliably. `mode="video"` is
  the production-ready path today.
- Single-job serialization caps throughput to the cluster's one-render-at-a-time reality; scaling
  means more co-located CUDA workers (ADR 0006 §5), not gateway changes.
- The gateway holds jobs in-process; a restart loses job *state* (not the mp4s on disk). A durable
  queue (Redis/DB) is a later upgrade if multi-replica serving is needed.

## 6. Boundary (do not regress)

- The gateway must stay **logic-free** about creative/pipeline decisions. Any "smart" routing
  beyond capability dispatch belongs in the agent + skills, not here.
- Never expose a public deployment without `AGENT_GATEWAY_API_KEY` (each call costs GPU time).
- TLS terminates at Caddy/Funnel; the app binds `127.0.0.1` only.
