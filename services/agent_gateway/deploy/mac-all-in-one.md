# All-in-one on the Mac mini — agent video service at kakeya.ai (no vast)

When the Mac mini is the **only GPU**, run the whole service on it. The Mac is both the GPU
worker and the control host; everything talks over `localhost`. No vast, no relay, no tailnet
hop — durable and owner-controlled.

```
User ─ https://agent.kakeya.ai ─► Cloudflare ──tunnel(outbound)──► Mac mini
                                                               ├─ cloudflared
                                                               ├─ agent_gateway (:8088)
                                                               │     │ subprocess → grpc_orchestrator (DIRECT, no-refine)
                                                               └─ MLX worker (:50051, framework/T2V)
```

Because the only worker is the Mac MLX (framework-only, no vid2vid), the orchestrator
**auto-selects DIRECT mode**: one MLX T2V generation, no tiled CUDA refine. Output is a
Mac-grade clip (low-res by memory necessity). Add a CUDA refine worker later (set
`WAN_WORKERS="127.0.0.1:50051,<cuda-host>:50051"`) to upgrade to the high-res refined pipeline
with zero gateway changes.

## 1. One-time setup (if not done)

```bash
# venv + mlx-video + WAN->MLX model at ~/wan21_mlx + gateway deps
bash services/distributed_wan/mac_setup.sh        # STEP=setup also fine
```

## 2. Start worker + gateway (one command)

```bash
API_KEY="<your-secret>" bash services/agent_gateway/deploy/mac_all_in_one.sh
# -> MLX worker on :50051, gateway on :8088; prints health + next steps. Logs in ~/.openmontage-logs
```

Verify locally on the Mac:

```bash
curl -s http://127.0.0.1:8088/healthz
curl -s -X POST http://127.0.0.1:8088/v1/videos -H 'Content-Type: application/json' \
     -H 'X-API-Key: <your-secret>' -d '{"prompt":"a red fox in a snowy forest, cinematic"}'
# poll /v1/jobs/<id> until done, then GET /v1/jobs/<id>/video
```

## 3. Expose at kakeya.ai (Cloudflare Tunnel — outbound, no open ports)

```bash
brew install cloudflared
```

**Option A — dashboard token (simplest):** Cloudflare dashboard → Zero Trust → Networks →
Tunnels → Create (Cloudflared) → copy token → on the Mac:

```bash
cloudflared service install <TOKEN>     # runs as a launchd service
```
Then in the tunnel's **Public Hostname**: `kakeya.ai` (HTTP) → `http://localhost:8088`. Cloudflare
auto-creates the proxied DNS record.

**Option B — CLI named tunnel:**
```bash
cloudflared tunnel login                                   # browser: pick the kakeya.ai zone
cloudflared tunnel create kakeya-gw
cloudflared tunnel route dns kakeya-gw agent.kakeya.ai   # subdomain; apex stays on the other site
cloudflared tunnel run --url http://localhost:8088 kakeya-gw
```

Expose on a **subdomain** (`agent.kakeya.ai`) — the `kakeya.ai` apex already serves another site.
Zone settings: SSL/TLS = **Full**, *Always Use HTTPS* = On, the `agent.kakeya.ai` record
**Proxied** (orange cloud). Full details in [`cloudflare.md`](cloudflare.md).

## 4. Verify publicly

```bash
curl -s https://agent.kakeya.ai/healthz
# open https://agent.kakeya.ai in a browser -> web UI -> type a prompt -> get a clip from the Mac GPU
```

## Notes

- Keep `API_KEY` set for any public deployment — each render costs Mac GPU time.
- Raise `--fw-width/--fw-height/--fw-frames` (orchestrator) as the Mac's unified memory allows; the
  gateway sends conservative defaults (480×256×13) that fit small Macs.
- The async API (submit → poll) keeps every request under Cloudflare's 100s proxy timeout even
  though a render takes a while.
- launchd/`pm2`/a `tmux` session keeps the worker + gateway alive across logout; `cloudflared
  service install` already runs as launchd.
