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
| `ai.kakeya.gateway`   | head Mac only | agent gateway (`server.py :8088`), `round_robin` over all workers |

The templates here use `__HOME__`, `__USER@HOST__` etc. — substitute per Mac. Live values used in
this deployment:

- **Head Mac** (`fluffy314`, has display): worker `--host 127.0.0.1`, gateway
  `WAN_WORKERS=127.0.0.1:50051,169.254.27.104:50051`, `AGENT_GATEWAY_WORKER_MODE=round_robin`.
- **Headless Mac** (`allen`, display/dummy-plug attached): worker `--host 169.254.27.104`.

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

## Notes
- The worker plist sets `PATH` venv-first so the MLX subprocess (`python -m mlx_video…`) resolves
  to the venv that has `mlx-video`. `MLX_TILING=aggressive` keeps VAE-decode command buffers small.
- The gateway plist intentionally omits `AGENT_GATEWAY_API_KEY` (open demo). Add it back (an
  `EnvironmentVariables` key) to re-require `X-API-Key`.
- `cloudflared` (the `kakeya-gw` tunnel serving `agent.kakeya.ai`/`ssh.kakeya.ai`) should also be a
  service for full reboot durability: `sudo cloudflared service install <tunnel-token>` (LaunchDaemon).
