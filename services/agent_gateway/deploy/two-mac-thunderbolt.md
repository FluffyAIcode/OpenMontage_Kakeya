# Two Mac minis over Thunderbolt — agent video cluster at kakeya.ai

Two Thunderbolt-bridged Mac minis form a small MLX cluster. This runbook deploys the agent video
service across both, with one Mac as the head (gateway + `cloudflared`).

```
User ─ https://kakeya.ai ─► Cloudflare ──tunnel──► Mac A (HEAD)
                                                     ├─ cloudflared
                                                     ├─ agent_gateway (:8088, POOL mode)
                                                     └─ MLX worker A (:50051)
                                                            │  Thunderbolt bridge (fast LAN)
                                                            ▼
                                                   Mac B: MLX worker B (:50051)
```

## What the two-Mac bridge does — and does not — buy you (honest)

`mlx-video` (the WAN port) has **no distributed/tensor-parallel support** (no `mlx.distributed`,
no `mlx.launch`, no cross-device collectives) and **no vid2vid**. Consequences:

- ❌ A *single* WAN generation **cannot** be sharded across both Macs, so the combined ~536 G is
  **not** one memory pool for one generation. Each generation is bounded by ONE Mac's memory.
- ❌ The high-res proposer→tiled-refine pipeline still needs a CUDA refiner (MLX has no vid2vid).
- ✅ **Throughput: 2× — the real win.** Each Mac serves an independent job in parallel (gateway
  POOL mode). Two users / two prompts complete at once.
- ✅ **Per-Mac headroom:** if these are large-memory Macs, each can do a higher-res DIRECT T2V than
  a small Mac — raise `--fw-width/--fw-height/--fw-frames`.
- ✅ The fast Thunderbolt link makes the head→peer gRPC hop effectively LAN-local (low latency).

> If you later add a CUDA box (or `mlx-video` gains vid2vid/distributed), drop it into
> `WAN_WORKERS` and the orchestrator re-enables the high-res refined pipeline with no code change.

## 1. Find the Thunderbolt-bridge IPs

macOS auto-creates a `bridge0` interface for the Thunderbolt bridge. On each Mac:

```bash
ifconfig bridge0 | grep 'inet '      # e.g. inet 169.254.x.x  (or your static bridge IP)
# or System Settings → Network → Thunderbolt Bridge → Details → TCP/IP
```
Note **Mac B's** bridge IP (the head will dial it). A static bridge IP (e.g. `192.168.5.1` / `.2`)
is more stable than link-local `169.254.*`.

## 2. Mac B — run an MLX worker (bind to the bridge so Mac A can reach it)

```bash
# one-time: bash services/distributed_wan/mac_setup.sh   (venv + mlx-video + ~/wan21_mlx)
cd ~/openmontage-mac/services/distributed_wan
source ~/.venv-distwan/bin/activate
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE MLX_TILING=aggressive \
  python grpc_worker.py --backend mlx --host 0.0.0.0 --port 50051 \
      --mlx-model-dir ~/wan21_mlx --mlx-ops framework
# verify from Mac A:  nc -vz <macB-bridge-ip> 50051
```

## 3. Mac A (HEAD) — worker + gateway (POOL) + cloudflared

```bash
cd ~/openmontage-mac && git pull --ff-only origin AgentMemory/agent-gateway-kekaye-cc88
API_KEY="<your-secret>" PEERS="<macB-bridge-ip>:50051" \
  bash services/agent_gateway/deploy/mac_all_in_one.sh
# -> MLX worker A + gateway (pool, parallel=2). /healthz shows "pool_mode":true,"parallel":2
```

Then expose at `kakeya.ai` via Cloudflare Tunnel (see [`cloudflare.md`](cloudflare.md)):

```bash
brew install cloudflared
cloudflared service install <DASHBOARD_TOKEN>     # public hostname kakeya.ai -> http://localhost:8088
```

## 4. Verify the 2× parallelism

```bash
curl -s https://kakeya.ai/healthz        # {"pool_mode":true,"parallel":2,"workers":[A,B],...}
# fire two jobs; both go "running" immediately (one per Mac), not queued:
for p in "a red fox in snow" "a sea turtle over a reef"; do
  curl -s -X POST https://kakeya.ai/v1/videos -H 'Content-Type: application/json' \
       -H 'X-API-Key: <your-secret>' -d "{\"prompt\":\"$p\"}"; echo; done
```

## Notes

- Keep `API_KEY` set; each render costs Mac GPU time.
- The gateway POOL pool hands each job a free Mac and blocks a 3rd job until one frees — natural
  backpressure for a 2-GPU cluster.
- Logs: `~/.openmontage-logs/{worker,gateway}.log` on the head; worker log on Mac B's terminal.
