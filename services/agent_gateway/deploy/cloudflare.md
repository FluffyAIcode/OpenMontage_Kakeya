# Serve the Agent Gateway via Cloudflare (relay on the Mac mini)

**Relay lives on the Mac mini, not on a GPU box.** The Mac mini cluster is the always-on,
owner-controlled entry point: it runs `cloudflared` (the relay) + the agent gateway + the
proposer/refiner MLX worker. A vast GPU is attached **on-demand only** as an extra refiner — it is
never the relay (it's ephemeral). This is what makes the link durable regardless of vast.

> **`kakeya.ai` apex is already in use** (it currently serves another site, e.g. "AgentMate").
> Do **not** clobber it — expose the agent on a **subdomain**, e.g. **`agent.kakeya.ai`**
> (or `video.kakeya.ai`). All examples below use `agent.kakeya.ai`.

The Mac sits behind home NAT — no public inbound IP. So instead of an A-record + open 80/443, use a
**Cloudflare Tunnel (`cloudflared`)**: an *outbound* tunnel from the **Mac** to Cloudflare's edge.
Cloudflare serves `https://agent.kakeya.ai` with its own (Universal SSL) cert and forwards requests
down the tunnel to the gateway on `localhost:8088`. No open ports, no public IP.

```
User ─ https://kakeya.ai ─► Cloudflare edge (TLS)
                                   │  (Cloudflare Tunnel — outbound from the box)
                                   ▼
                         cloudflared ─► http://localhost:8088  (agent_gateway)
                                                                   │ subprocess → distributed-WAN
```

> Our API is **async** (`POST /v1/videos` returns a job id immediately; you poll), so no request
> ever hits Cloudflare's 100-second proxy timeout. The mp4 download is small. Cloudflare's orange-
> cloud proxy is fully compatible.

`cloudflared` is already installed on the box at `/usr/local/bin/cloudflared` (v2026.6.1).

---

## Option A — Dashboard token (simplest; best for a headless box)

No browser needed on the box. Create the tunnel in the Cloudflare dashboard, paste its token.

1. **Cloudflare dashboard → Zero Trust → Networks → Tunnels → Create a tunnel**
   - Connector: **Cloudflared**. Name: `kakeya-gw`. Save → it shows an install command with a
     long **token** (`eyJ...`). Copy the token.

2. **On the GPU box**, run the connector with that token (persist it in `tmux`/systemd):
   ```bash
   sudo cloudflared service install <PASTE_TOKEN>
   # or, non-root / ad-hoc:
   cloudflared tunnel run --token <PASTE_TOKEN>
   ```

3. **Back in the dashboard → the tunnel → Public Hostname → Add a public hostname:**
   - **Subdomain:** `agent`  **Domain:** `kakeya.ai`   (i.e. `agent.kakeya.ai` — keeps the apex site intact)
   - **Type:** `HTTP`  **URL:** `localhost:8088`
   - Save. Cloudflare **auto-creates the proxied DNS record** for `agent.kakeya.ai`.

4. **(Optional) also expose `www`** — add another public hostname `www.kakeya.ai → localhost:8088`,
   or add a redirect rule.

Done — `https://kakeya.ai` now reaches the gateway.

---

## Option B — CLI named tunnel (config-as-code)

Requires a one-time browser login to authorize the zone. If the box is headless, run
`cloudflared tunnel login` on your laptop and copy the resulting `~/.cloudflared/cert.pem` to the box.

```bash
# 1) authorize (browser): pick the kakeya.ai zone — writes ~/.cloudflared/cert.pem
cloudflared tunnel login

# 2) create a named tunnel — writes ~/.cloudflared/<UUID>.json credentials
cloudflared tunnel create kakeya-gw

# 3) create the proxied DNS record for the SUBDOMAIN automatically (apex stays untouched)
cloudflared tunnel route dns kakeya-gw agent.kakeya.ai

# 4) write the config (see deploy/cloudflared-config.yml), then run it
cloudflared tunnel --config services/agent_gateway/deploy/cloudflared-config.yml run kakeya-gw

# 5) install as a persistent service (optional)
sudo cloudflared --config services/agent_gateway/deploy/cloudflared-config.yml service install
sudo systemctl enable --now cloudflared
```

`deploy/cloudflared-config.yml` (fill in your UUID — `cloudflared tunnel list` prints it):

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: ~/.cloudflared/<TUNNEL_UUID>.json   # on the Mac (not root) the path is ~/.cloudflared/
ingress:
  - hostname: agent.kakeya.ai
    service: http://localhost:8088
  - service: http_status:404
```

---

## Cloudflare zone settings (both options)

In the `kakeya.ai` zone dashboard:

- **SSL/TLS → Overview:** mode **Full** (the tunnel is already encrypted end-to-end; Universal SSL
  serves the edge cert for `agent.kakeya.ai` automatically — no cert work on the Mac).
- **SSL/TLS → Edge Certificates:** *Always Use HTTPS* = **On**; *Automatic HTTPS Rewrites* = On.
- **DNS:** the tunnel's `agent.kakeya.ai` record must be **Proxied (orange cloud)** — Tunnels require
  it. Leave the existing apex `kakeya.ai` record alone (it serves the other site).
- **Security:** protect generation cost — set the gateway's `AGENT_GATEWAY_API_KEY` (callers send
  `X-API-Key`), and/or add a Cloudflare WAF rate-limit rule on `POST /v1/videos`.

## Run the gateway with the API key (on the box)

```bash
cd /workspace/om   # or your checkout
WAN_WORKERS="127.0.0.1:50051,127.0.0.1:55051" \
ORCHESTRATOR_PATH=/workspace/distwan_src/distributed_wan/grpc_orchestrator.py \
AGENT_GATEWAY_API_KEY="<your-secret>" AGENT_GATEWAY_JOBS_DIR=/workspace/gw_jobs \
python3 services/agent_gateway/server.py --host 127.0.0.1 --port 8088
```

## Verify

```bash
curl -s https://agent.kakeya.ai/healthz      # expect JSON {"status":"ok",...}, NOT the apex site's HTML
curl -s -X POST https://agent.kakeya.ai/v1/videos -H 'Content-Type: application/json' \
     -H 'X-API-Key: <your-secret>' -d '{"prompt":"a red fox in a snowy forest, cinematic"}'
# → {"job_id":"…"}; poll https://agent.kakeya.ai/v1/jobs/<id> until done; GET …/video
```

> If `/healthz` returns HTML (the apex "AgentMate" page) instead of JSON, the tunnel route isn't
> active for the subdomain yet — re-check the Public Hostname mapping and that `cloudflared` is
> running on the Mac.

## Troubleshooting `error 1033` / `502` on `agent.kakeya.ai`

Both mean Cloudflare's edge can't reach a healthy origin through the tunnel. Walk the chain on the
**Mac** (the relay), in order:

```bash
# 1) Is the gateway ORIGIN actually up on the Mac?  (502 == connector ok, origin down)
curl -s http://127.0.0.1:8088/healthz            # expect JSON {"status":"ok",...}
#    not up? start it:  API_KEY=<secret> bash services/agent_gateway/deploy/mac_all_in_one.sh
#    then watch:        tail -f ~/.openmontage-logs/gateway.log

# 2) Is cloudflared running AND connected?  (1033 == no live connector for the hostname)
cloudflared tunnel list                          # CONNECTIONS column must be non-empty
cloudflared tunnel info kakeya-gw                # shows active edge connections
ps aux | grep -v grep | grep cloudflared

# 3) Does the RUNNING tunnel map agent.kakeya.ai -> http://localhost:8088 ?
#    - CLI:        cloudflared tunnel --url http://localhost:8088 run kakeya-gw
#    - config.yml: ingress: [{hostname: agent.kakeya.ai, service: http://localhost:8088}, {service: http_status:404}]
#    - dashboard:  the tunnel's Public Hostname = agent.kakeya.ai (HTTP) -> localhost:8088
```

**Most common cause:** the DNS route (`cloudflared tunnel route dns kakeya-gw agent.kakeya.ai`)
points at tunnel `kakeya-gw`, but a *different* tunnel (or a dashboard token tunnel) is the one
actually running — so the edge has the route but no matching live connector → `1033`. Fix: run the
**same** tunnel that owns the route, pointed at the gateway:

```bash
cloudflared tunnel --url http://localhost:8088 run kakeya-gw
```

Once `curl http://127.0.0.1:8088/healthz` returns JSON on the Mac **and** that tunnel is running,
`https://agent.kakeya.ai/healthz` returns the same JSON.

## Alternative — real A-record (only if the host has a public IP + open 80/443)

Not the vast box's case, but for a normal VPS: point `kakeya.ai` A-record (DNS-only, grey cloud)
at the IP, open 80/443, and run `deploy/Caddyfile` (`SITE_HOST=kakeya.ai caddy run --config Caddyfile`)
for auto Let's Encrypt. Then in Cloudflare set SSL/TLS = **Full (strict)**.
