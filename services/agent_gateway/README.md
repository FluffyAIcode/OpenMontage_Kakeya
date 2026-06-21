# OpenMontage Agent Gateway — public video service (kakeya.ai)

The public front door that lets anyone use OpenMontage's agent video service over a domain.
It is a **thin transport + job/session layer** — no creative or orchestration logic lives here
(that stays with the OpenMontage agent + skills, per `AGENT_GUIDE.md`). The gateway opens a job,
drives the **validated distributed-WAN cluster** (Mac-mini MLX proposer + cloud CUDA refiner,
see `services/distributed_wan/`), streams progress, and serves the resulting `.mp4`.

```
  Browser / API client
        │  HTTPS  https://kakeya.ai
        ▼
  Caddy (auto-TLS)  ──►  agent_gateway (FastAPI :8088)
                                │ subprocess
                                ▼
                 distributed_wan/grpc_orchestrator.py
                    │ gRPC                  │ gRPC (via socks5_forward over tailnet)
                    ▼                       ▼
              CUDA worker (refine)     Mac mini MLX worker (framework)
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/` | Minimal web UI (prompt → video) |
| `GET`  | `/healthz` | Liveness + config (`workers`, `orchestrator`, `agent_runtime`) |
| `GET`  | `/v1/capabilities` | Registry provider-menu summary (best-effort) |
| `POST` | `/v1/videos` | Submit `{prompt, mode?, frames?, fw_width?, …}` → `{job_id}` |
| `GET`  | `/v1/jobs/{id}` | Job status: `status`, `stage`, `pct`, `log`, `result` |
| `GET`  | `/v1/jobs/{id}/video` | Download the rendered mp4 |

`mode="video"` (default): direct text→video via the distributed-WAN cluster (works today).
`mode="agent"`: full multi-stage OpenMontage pipeline — requires an attached **agent runtime**
(`AGENT_RUNTIME_CMD`); the gateway never fabricates creative decisions.

### Example

```bash
curl -s -X POST https://kakeya.ai/v1/videos \
  -H 'Content-Type: application/json' -H 'X-API-Key: $KEY' \
  -d '{"prompt":"a red fox walking through a snowy forest, cinematic"}'
# {"job_id":"ab12…","status":"queued","poll":"/v1/jobs/ab12…"}
curl -s https://kakeya.ai/v1/jobs/ab12…            # poll until status=done
curl -s https://kakeya.ai/v1/jobs/ab12…/video -o out.mp4
```

## Run

```bash
pip install -r services/agent_gateway/requirements.txt
WAN_WORKERS="127.0.0.1:50051,127.0.0.1:55051" \
AGENT_GATEWAY_API_KEY="<optional-secret>" \
python services/agent_gateway/server.py --host 0.0.0.0 --port 8088
```

Env:

| Var | Meaning |
|---|---|
| `WAN_WORKERS` | gRPC worker addresses passed to the orchestrator (required for `mode=video`) |
| `AGENT_GATEWAY_API_KEY` | if set, `POST /v1/videos` requires header `X-API-Key` |
| `AGENT_GATEWAY_JOBS_DIR` | where job outputs are written (default `projects/_gateway_jobs`) |
| `ORCHESTRATOR_PATH` | path to `grpc_orchestrator.py` if the gateway lives apart from `distributed_wan` |
| `AGENT_RUNTIME_CMD` | command that drives the full OpenMontage pipeline for `mode=agent` |

## Exposing it at kakeya.ai

### Option A — your own host + Caddy (production, custom domain)

1. **DNS:** point `kakeya.ai` (A/AAAA) at the gateway host's public IP. (Cloudflare/registrar.)
2. Open ports **80** and **443** to the host.
3. Run the gateway on `127.0.0.1:8088` (systemd unit in `deploy/agent-gateway.service`).
4. Run Caddy with `deploy/Caddyfile` — it fetches a Let's Encrypt cert for `kakeya.ai` automatically.

```bash
sudo cp services/agent_gateway/deploy/agent-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now agent-gateway
sudo caddy run --config services/agent_gateway/deploy/Caddyfile     # SITE_HOST=kakeya.ai by default
```

> **On Cloudflare?** `kakeya.ai` is managed by Cloudflare and the GPU box is behind NAT — use a
> **Cloudflare Tunnel**. Full command set: [`deploy/cloudflare.md`](deploy/cloudflare.md).
>
> **Mac mini is the only GPU?** Run worker + gateway + `cloudflared` all on the Mac (self-contained,
> auto DIRECT no-refine mode): [`deploy/mac-all-in-one.md`](deploy/mac-all-in-one.md) +
> `deploy/mac_all_in_one.sh`.

### Option B — Tailscale Funnel (instant public HTTPS, no DNS/cert work)

Good for a rented GPU box behind NAT (no public IP). Gives a public `https://<host>.<tailnet>.ts.net`
URL with a valid cert. To later move to `kakeya.ai`, CNAME it to the funnel host or switch to Option A.

```bash
tailscale funnel --bg 8088        # exposes the local gateway publicly over HTTPS
tailscale funnel status           # prints the public https URL
```

> Funnel requires the tailnet admin to enable HTTPS certs + Funnel for the node (Tailscale
> admin console → DNS → enable MagicDNS/HTTPS; ACLs → allow `funnel`).

## Security

- Set `AGENT_GATEWAY_API_KEY` for any public deployment (each generation costs GPU time).
- Caddy terminates TLS; the gateway binds `127.0.0.1` only.
- The gateway runs untrusted prompt text only — it never `eval`s input and shells out with a
  fixed argv (no shell interpolation).
