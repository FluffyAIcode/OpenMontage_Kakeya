# Two Mac minis over Thunderbolt — agent video cluster at kakeya.ai

Two Thunderbolt-bridged Mac minis form a small MLX cluster. This runbook deploys the agent video
service across both, with one Mac as the head (gateway + `cloudflared`).

```
User ─ https://agent.kakeya.ai ─► Cloudflare ──tunnel──► Mac A (HEAD)
                                                     ├─ cloudflared
                                                     ├─ agent_gateway (:8088)
                                                     └─ MLX worker A (:50051)
                                                            │  Thunderbolt bridge (fast LAN)
                                                            ▼
                                                   Mac B: MLX worker B (:50051)
```

## Two topologies — pick by goal

The cluster runs in one of two modes (set by the gateway + each worker's `--mlx-ops` role):

| Mode | Head Mac A | Headless Mac B | One job uses | Best for |
|---|---|---|---|---|
| **Pipeline** (quality) | `framework` = proposer | `refine` = refiner | BOTH Macs (A proposes low-res → B refines) | higher-res / refined single clip |
| **Pool** (throughput) | `framework` | `framework` | ONE Mac (the other takes the next job) | 2 prompts at once, 2× throughput |

## What the two-Mac bridge does — and does not — buy you (honest)

`mlx-video` (the WAN port) has **no distributed/tensor-parallel support** (no `mlx.distributed`,
no `mlx.launch`, no cross-device collectives) and **no native vid2vid**. Consequences:

- ❌ A *single* WAN generation **cannot** be sharded across both Macs, so the combined ~536 G is
  **not** one memory pool for one generation. Each generation is bounded by ONE Mac's memory.
- ⚠️ **Pipeline mode's refine is spatial super-resolution, not generative.** With stock mlx-video the
  headless refiner does a **Lanczos upscale + unsharp** of the proposer to the target resolution
  (`--out-width/--out-height`): it raises resolution and crispness but does **not** synthesize new
  detail. For *generative* refine you still need a CUDA refiner (diffusers WAN V2V) or an mlx-video
  build with vid2vid (then set `MLX_V2V_FLAG`).
- ✅ **Pipeline: a real two-Mac coarse-to-fine path** — head proposes a fast low-res draft, headless
  upscales/refines the whole clip in one pass. Output is higher-res than a single Mac's DIRECT draft.
- ✅ **Pool: throughput 2×** — each Mac serves an independent job in parallel.
- ✅ **Per-Mac headroom:** large-memory Macs can raise `--fw-width/--fw-height/--fw-frames` for a
  higher-res proposer (and `--out-width/--out-height` for the pipeline output).
- ✅ The fast Thunderbolt link makes the head→peer gRPC hop effectively LAN-local (low latency).

> Add a CUDA box (or an mlx-video with vid2vid) to `WAN_WORKERS` and the orchestrator promotes the
> refiner to the high-res **tiled generative** pipeline automatically — no code change.

## ⚠️ Do not address the peer by its `169.254.x` Thunderbolt link-local IP

Link-local `169.254.x` addresses are **not stable across reboots** and, when a Mac has `169.254.x`
on more than one interface (e.g. `bridge0` + `en9`), the head Mac can get **"No route to host"**
to the peer's bridge IP (ambiguous route) — this took the headless worker offline after a reboot
(ADR 0001 Iteration 32). Instead:

- Bind the peer's worker to **`--host 0.0.0.0`** (all interfaces), and
- Address it from the gateway by a **stable IP** — the peer's **LAN IP with a DHCP reservation**
  (e.g. `192.168.68.51`) or its **Tailscale IP**. (gRPC can't resolve `.local` mDNS names, so use
  an IP, not the `*.local` name, in `WAN_WORKERS`.)

## 1. Find the Thunderbolt-bridge IPs

macOS auto-creates a `bridge0` interface for the Thunderbolt bridge. On each Mac:

```bash
ifconfig bridge0 | grep 'inet '      # e.g. inet 169.254.x.x  (or your static bridge IP)
# or System Settings → Network → Thunderbolt Bridge → Details → TCP/IP
```
Note **Mac B's** bridge IP (the head will dial it). A static bridge IP (e.g. `192.168.5.1` / `.2`)
is more stable than link-local `169.254.*`.

## 2. Mac B (headless) — run the MLX worker in the role for your mode

```bash
# one-time: bash services/distributed_wan/mac_setup.sh   (venv + mlx-video + ~/wan21_mlx)
cd ~/openmontage-mac/services/distributed_wan
source ~/.venv-distwan/bin/activate
# PIPELINE: Mac B is the REFINER  -> --mlx-ops refine   (SR upscale; set MLX_V2V_FLAG for V2V)
# POOL:     Mac B is another GPU  -> --mlx-ops framework
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE MLX_TILING=aggressive MLX_SR_SHARPEN=60 \
  python grpc_worker.py --backend mlx --host 0.0.0.0 --port 50051 \
      --mlx-model-dir ~/wan21_mlx --mlx-ops refine
# verify from Mac A:  nc -vz <macB-stable-ip> 50051   (use the LAN/Tailscale IP, NOT 169.254.x)
```

## 3. Mac A (HEAD) — worker + gateway + cloudflared

```bash
cd ~/openmontage-mac && git pull --ff-only origin <this-branch>
# PIPELINE (default): head=proposer(framework), peer=refiner. ONE job spans both Macs.
API_KEY="<your-secret>" MODE=pipeline MLX_OPS=framework PEERS="<macB-stable-ip>:50051" \
  bash services/agent_gateway/deploy/mac_all_in_one.sh
# /healthz shows {"mode":"pipeline","pipeline_mode":true,"workers":[A,B]}

# POOL (throughput) instead: every worker=framework, each job DIRECT on one Mac, 2 in parallel.
# API_KEY=... MODE=pool MLX_OPS=framework PEERS="<macB-stable-ip>:50051" bash .../mac_all_in_one.sh
```

Then expose at `kakeya.ai` via Cloudflare Tunnel (see [`cloudflare.md`](cloudflare.md)):

```bash
brew install cloudflared
cloudflared service install <DASHBOARD_TOKEN>     # public hostname agent.kakeya.ai -> http://localhost:8088
```

## 4. Verify

```bash
# PIPELINE: one job, log shows  framework: …% (proposer on A)  then  refine: …% (refiner on B)
curl -s https://agent.kakeya.ai/healthz        # {"mode":"pipeline","pipeline_mode":true,...}
JID=$(curl -s -X POST https://agent.kakeya.ai/v1/videos -H 'Content-Type: application/json' \
      -H 'X-API-Key: <your-secret>' -d '{"prompt":"a red fox in snow, cinematic"}' \
      | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
curl -s https://agent.kakeya.ai/v1/jobs/$JID -H 'X-API-Key: <your-secret>'   # ORCH_DONE mode=pipeline

# POOL: fire two jobs; both go "running" immediately (one per Mac), not queued:
for p in "a red fox in snow" "a sea turtle over a reef"; do
  curl -s -X POST https://agent.kakeya.ai/v1/videos -H 'Content-Type: application/json' \
       -H 'X-API-Key: <your-secret>' -d "{\"prompt\":\"$p\"}"; echo; done
```

## Headless Mac GPU watchdog (`kIOGPUCommandBufferCallbackErrorTimeout`)

A Mac with **no display attached / driven over SSH only** runs macOS's GPU command-buffer watchdog
aggressively, so long Metal kernels (WAN's VAE decode or a denoise step) get aborted mid-flight —
`[METAL] Command buffer execution failed: Caused GPU Timeout Error`. This is **not** OOM and **not**
the network; it's the headless GPU watchdog. Fixes, in order of reliability:

1. **Attach a display (best, hardware):** plug a **headless HDMI/DisplayPort dummy plug** (~$10) into
   the headless Mac. macOS then thinks a display is present → relaxes the GPU watchdog **and** unlocks
   full GPU clocks. This is the standard fix for headless Apple-Silicon ML.
2. **Run the worker inside the logged-in GUI (Aqua) session**, not a bare SSH shell: enable auto-login
   and start the worker via a **LaunchAgent** (loaded in the user session) or a Screen-Sharing
   Terminal. GPU buffers from the console session aren't watchdog'd as hard as pure-SSH ones.
3. **Shorten each command buffer (software):** `MLX_TILING=aggressive` (splits the VAE decode into
   small buffers) + smaller resolution/frames/steps. Helps, but #1 is the durable fix.
4. **`caffeinate -dimsu`** so the Mac/GPU never sleeps or throttles mid-render.

**Immediate workaround:** route generation only to a **display-attached** Mac — start the gateway
with just that worker (omit the headless peer from `WAN_WORKERS`/`PEERS`) until the headless Mac has
a dummy plug + GUI-session worker.

## Notes

- Keep `API_KEY` set; each render costs Mac GPU time.
- **Pool** mode hands each job a free Mac and blocks a 3rd job until one frees — natural
  backpressure for a 2-GPU cluster. **Pipeline** mode runs one job at a time (both Macs busy on it).
- Switch modes by re-running `mac_all_in_one.sh` with `MODE=pipeline|pool` and restarting Mac B's
  worker with the matching `--mlx-ops` (`refine` for pipeline, `framework` for pool).
- Logs: `~/.openmontage-logs/{worker,gateway}.log` on the head; worker log on Mac B's terminal.
