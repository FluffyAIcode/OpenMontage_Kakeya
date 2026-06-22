# Ephemeral vast refiner — auto-recovery (head proposer + headless SR + vast V2V via f_θ)

vast.ai GPUs are frequently **released and recreated** (a recreate = a clean box: new SSH host/port,
no torch, no model cache). This runbook makes the 3-node cluster **self-healing**: after every vast
recreate it auto-converges back to a refiner, and while vast is gone the service keeps running on the
2-Mac pipeline (head Mac proposer + headless Mac SR refine, merged by f_θ).

## How it works

```
                    ~/.kakeya/wan_workers   (dynamic membership file)
                            ▲ writes
   head Mac:  vast_refiner_supervisor.sh ──ssh──► vast box (ephemeral)
      (launchd KeepAlive)   │  scp + vast_bootstrap.sh (install deps, launch worker)
      gateway reads ────────┘  ssh -L 50052->vast:50051 (gRPC tunnel)
      the file per job
```

- **Gateway** runs with `AGENT_GATEWAY_WORKERS_FILE=~/.kakeya/wan_workers` and re-reads it **per job**
  — so membership changes need **no gateway restart** (no dropped jobs).
- **Supervisor** (`vast_refiner_supervisor.sh`, launchd) every ~30 s:
  - vast reachable → scp worker files, run `vast_bootstrap.sh` (idempotent install + launch in tmux),
    (re)open the SSH gRPC tunnel, and write `BASE_WORKERS,127.0.0.1:50052` → **3-node**.
  - vast unreachable / still loading → write `BASE_WORKERS` → **2-Mac fallback** (head + headless).
- **f_θ merge**: with vast present the orchestrator runs the tiled refine (round-robin: vast = CUDA
  generative V2V tiles, headless = MLX SR tiles) and merges with the f_θ weight-map. With vast gone it
  auto-degrades to a single-pass SR refine on the headless Mac.

## One-time setup (head Mac)

1. Gateway reads the dynamic file — add to `ai.kakeya.gateway.plist` `EnvironmentVariables`:
   ```xml
   <key>AGENT_GATEWAY_WORKERS_FILE</key><string>__HOME__/.kakeya/wan_workers</string>
   <key>AGENT_GATEWAY_MODE</key><string>distributed</string>
   ```
   Keep `WAN_WORKERS` as the fallback (`127.0.0.1:50051,<headlessIP>:50051`). Reload the gateway.

2. Config `~/.kakeya/vast.env`:
   ```bash
   VAST_SSH_HOST=104.202.252.41
   VAST_SSH_PORT=20006
   VAST_SSH_USER=root
   VAST_SSH_KEY=/Users/fluffy314/.ssh/id_ed25519     # head key authorized on vast
   VAST_REMOTE_PORT=50051
   VAST_LOCAL_PORT=50052
   BASE_WORKERS=127.0.0.1:50051,192.168.68.51:50051   # head proposer + headless refiner
   WORKERS_FILE=/Users/fluffy314/.kakeya/wan_workers
   DISTWAN_LOCAL=/Users/fluffy314/openmontage-mac/services/distributed_wan
   INTERVAL=30
   ```

3. Install the supervisor LaunchAgent:
   ```bash
   sed "s#__HOME__#$HOME#g" services/agent_gateway/deploy/launchd/ai.kakeya.vastsupervisor.plist \
     > ~/Library/LaunchAgents/ai.kakeya.vastsupervisor.plist
   U=$(id -u)
   launchctl bootstrap "gui/$U" ~/Library/LaunchAgents/ai.kakeya.vastsupervisor.plist
   launchctl enable "gui/$U/ai.kakeya.vastsupervisor"
   tail -f ~/.openmontage-logs/vastsupervisor.log
   ```

## After each vast recreate

Only the SSH endpoint changes. Either:
- **Manual (one edit):** update `VAST_SSH_HOST`/`VAST_SSH_PORT` in `~/.kakeya/vast.env`. The
  supervisor picks it up next cycle: scp + bootstrap + tunnel + 3-node, all automatic.
- **Auto (optional):** set `VAST_RESOLVE_CMD` in `vast.env` to a command that prints `HOST PORT` for
  your instance (e.g. wrapping `vastai show instances`). Then recreate needs **zero** manual edits.

Make sure the recreated box authorizes the head Mac's pubkey (vast "On-start Script":
`echo '<head pubkey>' >> ~/.ssh/authorized_keys`), or bake it into your vast template.

## Verify

```bash
curl -s http://127.0.0.1:8088/healthz   # {"mode":"distributed","dynamic_workers":true,"workers":[...]}
cat ~/.kakeya/wan_workers                # 3 entries when vast up; 2 when vast gone
grep ONLINE ~/.openmontage-logs/vastsupervisor.log
```
