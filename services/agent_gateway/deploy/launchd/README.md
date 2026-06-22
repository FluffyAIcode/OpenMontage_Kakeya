# launchd auto-start for the Mac mini cluster (survive crashes + reboots)

Runs the OpenMontage MLX worker (each Mac) and the agent gateway (head Mac) as **per-user
LaunchAgents**. LaunchAgents (not LaunchDaemons) run in the user's **GUI/Aqua session**, which is
what keeps the **GPU watchdog relaxed** for Metal/MLX work — a LaunchDaemon at boot has no display
session and would re-trigger `kIOGPUCommandBufferCallbackErrorTimeout`. `KeepAlive` restarts on
crash; `RunAtLoad` starts at login.

Two agents (Label → role):

| Label | Where | Role |
|---|---|---|
| `ai.kakeya.mlxworker` | every Mac | MLX worker (`grpc_worker.py --backend mlx`) under `caffeinate` |
| `ai.kakeya.gateway`   | head Mac only | agent gateway (`server.py :8088`) over all workers |
| `ai.kakeya.vastsupervisor` | head Mac only | keeps an EPHEMERAL vast CUDA refiner self-healing; flips the gateway's dynamic worker file 3-node↔2-Mac (see [`../ephemeral-vast-refiner.md`](../ephemeral-vast-refiner.md)) |

> For the ephemeral-vast supervisor, run the gateway with `AGENT_GATEWAY_WORKERS_FILE=~/.kakeya/wan_workers`
> (and `AGENT_GATEWAY_MODE=distributed`) so worker membership is dynamic — no gateway restart when
> vast comes/goes. Setup: [`../ephemeral-vast-refiner.md`](../ephemeral-vast-refiner.md).

**Worker ROLE is set per Mac via `--mlx-ops` (`__MLX_OPS__` in the plist):**

| Topology | Head Mac `--mlx-ops` | Headless Mac `--mlx-ops` | Gateway mode |
|---|---|---|---|
| **Pipeline** (head proposes, headless refines) | `framework` | `refine` | default (pool OFF) → `--single-refine` |
| **Pool** (throughput, each Mac a full job) | `framework` | `framework` | `AGENT_GATEWAY_WORKER_POOL=1` |

In **pipeline** mode the gateway runs ONE job across both Macs: the `framework` worker (head)
makes a low-res proposer, the `refine` worker (headless) upscales/refines the whole clip in one
pass (MLX SR by default — Lanczos+unsharp; generative V2V if `MLX_V2V_FLAG` is set). In **pool**
mode every worker advertises `framework` and each job runs DIRECT on one Mac, N in parallel.

The templates here use `__HOME__`, `__HOST__`, `__MLX_OPS__` etc. — substitute per Mac. Live values
used in this deployment (pipeline topology):

- **Head Mac** (`fluffy314`, has display) — PROPOSER: worker `--host 127.0.0.1 --mlx-ops framework`;
  gateway `WAN_WORKERS=127.0.0.1:50051,192.168.68.51:50051` (pool OFF → pipeline).
- **Headless Mac** (`allen`, dummy-plug attached) — REFINER: worker
  `--host 0.0.0.0 --mlx-ops refine` (reach it at its stable LAN IP `192.168.68.51`, NOT a
  `169.254.x` link-local — the head Mac has multiple such interfaces, see ADR 0001 Iteration 32).

## Install (per Mac)

```bash
mkdir -p ~/Library/LaunchAgents ~/.openmontage-logs
# copy the plist(s), substituting your paths, to ~/Library/LaunchAgents/
U=$(id -u)
# stop any manual instances first
pkill -f "caffeinate.*grpc_worker"; pkill -f grpc_worker.py; pkill -f "server.py.*8088"
# load into the GUI session (KeepAlive + RunAtLoad)
launchctl bootstrap "gui/$U" ~/Library/LaunchAgents/ai.kakeya.mlxworker.plist
launchctl enable    "gui/$U/ai.kakeya.mlxworker"
# head Mac only:
launchctl bootstrap "gui/$U" ~/Library/LaunchAgents/ai.kakeya.gateway.plist
launchctl enable    "gui/$U/ai.kakeya.gateway"
# verify
launchctl print "gui/$U/ai.kakeya.mlxworker" | grep -E "state =|pid ="
curl -s http://127.0.0.1:8088/healthz      # head Mac
```

Manage:
```bash
launchctl kickstart -k "gui/$U/ai.kakeya.gateway"   # restart
launchctl bootout "gui/$U/ai.kakeya.gateway"        # stop+unload
tail -f ~/.openmontage-logs/{worker,gateway}.log
```

## Critical for surviving REBOOTS (not just crashes)

LaunchAgents start at **user login**, so for an unattended reboot to bring the cluster back you
need **automatic login** for that user:

- System Settings → Users & Groups → **Automatically log in as** → select the user.
- **FileVault caveat:** if FileVault is ON, the disk needs the password at pre-boot, so auto-login
  can't proceed unattended after a cold boot. Either disable FileVault on these dedicated render
  Macs, or accept that a cold reboot needs one manual unlock. (After unlock/login, the agents start.)

Without auto-login, `KeepAlive` still restarts the services on crash and they start the moment
someone logs in — but a headless reboot won't auto-resume.

## cloudflared tunnel (head Mac) — `ai.kakeya.cloudflared` + `run_cloudflared.sh`

The third LaunchAgent keeps the Cloudflare tunnel (`agent.kakeya.ai` / `ssh.kakeya.ai`) up across
reboots. **Do not hard-code the token in the plist** — a dashboard *Refresh token* invalidates it
and silently breaks auto-start (this bit us, ADR 0001 Iteration 31). Instead the plist runs
`run_cloudflared.sh`, which calls `cloudflared tunnel token <UUID>` to fetch the **current** token
at startup (needs a valid `~/.cloudflared/cert.pem` for the zone) — rotation-resilient.

```bash
# install the wrapper (substitute __TUNNEL_ID__ and chmod +x) and the plist (substitute __HOME__)
cp run_cloudflared.sh ~/run_cloudflared.sh && chmod +x ~/run_cloudflared.sh
cp ai.kakeya.cloudflared.plist ~/Library/LaunchAgents/
U=$(id -u)
launchctl bootstrap "gui/$U" ~/Library/LaunchAgents/ai.kakeya.cloudflared.plist
launchctl enable "gui/$U/ai.kakeya.cloudflared"
grep "Registered tunnel connection" ~/.openmontage-logs/cloudflared.log    # expect connIndex 0..3
```

**CRITICAL — run exactly ONE connector for this tunnel.** Multiple connectors with the same token
(or a stale/rotated token) produce `control stream encountered a failure while serving` and the
tunnel goes `Down`. If you ever need to recover: `pkill -f "cloudflared tunnel"`, then start a
single connector with a CURRENT token — `cloudflared tunnel --url http://localhost:8088 run --token
$(cloudflared tunnel token <UUID>)` — and watch for `Registered tunnel connection`.

## Notes
- The worker plist sets `PATH` venv-first so the MLX subprocess (`python -m mlx_video…`) resolves
  to the venv that has `mlx-video`. `MLX_TILING=aggressive` keeps VAE-decode command buffers small.
- The gateway plist intentionally omits `AGENT_GATEWAY_API_KEY` (open demo). Add it back (an
  `EnvironmentVariables` key) to re-require `X-API-Key`.
