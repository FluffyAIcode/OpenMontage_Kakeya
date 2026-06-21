#!/usr/bin/env bash
# =============================================================================
# All-in-one OpenMontage agent video service ON THE MAC MINI.  RUN ON THE MAC.
# =============================================================================
# Self-contained: the Mac is the ONLY GPU and also the control host.
#   MLX worker (framework/T2V)  +  agent_gateway (:8088, no-refine DIRECT mode)
#   then expose via Cloudflare Tunnel -> https://kakeya.ai
#
# No vast, no relay: the gateway talks to its LOCAL MLX worker over localhost, and
# the orchestrator auto-uses DIRECT (no-refine) mode because the only worker is the
# Mac MLX (framework-only). Output is a Mac-grade low-res T2V clip — raise dims as
# your Mac's memory allows, or add a CUDA refine worker later to upgrade quality.
#
# Prereq: run mac_setup.sh once first (venv + mlx-video + WAN->MLX model at ~/wan21_mlx).
#
# Usage (from the repo's services/distributed_wan working dir is fine):
#   bash services/agent_gateway/deploy/mac_all_in_one.sh
#   API_KEY=mysecret PORT=8088 bash .../mac_all_in_one.sh
# =============================================================================
set -euo pipefail

VENV="${VENV:-$HOME/.venv-distwan}"
MODEL_DIR="${MODEL_DIR:-$HOME/wan21_mlx}"
WORKDIR="${WORKDIR:-$HOME/openmontage-mac}"
PORT="${PORT:-8088}"
WORKER_PORT="${WORKER_PORT:-50051}"
API_KEY="${API_KEY:-}"
MLX_TILING="${MLX_TILING:-aggressive}"
# Two-Mac Thunderbolt cluster: PEERS = comma-list of the OTHER Mac(s)' worker addresses on the
# Thunderbolt-bridge network, e.g. PEERS="192.168.5.2:50051". Each Mac runs its own MLX worker;
# the head Mac (this one) runs the gateway in worker-POOL mode -> one job per Mac, N× throughput.
PEERS="${PEERS:-}"

say() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
die() { printf "\n\033[1;31mERROR: %s\033[0m\n" "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "run this on the Mac mini (macOS)"
[ -d "$VENV" ] || die "venv $VENV missing — run mac_setup.sh first"
[ -d "$MODEL_DIR" ] || die "MLX model $MODEL_DIR missing — run mac_setup.sh (STEP=setup) first"
[ -d "$WORKDIR/services" ] || die "repo not at $WORKDIR — set WORKDIR or run mac_setup.sh first"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
say "ensure gateway deps (fastapi/uvicorn) in the venv"
python -m pip install -q "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6"

cd "$WORKDIR"
DWAN="services/distributed_wan"
ORCH="$WORKDIR/$DWAN/grpc_orchestrator.py"
LOG_DIR="${LOG_DIR:-$HOME/.openmontage-logs}"; mkdir -p "$LOG_DIR"

# 1) MLX worker (framework/T2V) on localhost
if ! nc -z 127.0.0.1 "$WORKER_PORT" 2>/dev/null; then
  say "starting MLX worker on 127.0.0.1:$WORKER_PORT (tiling=$MLX_TILING)"
  ( cd "$WORKDIR/$DWAN" && env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
      MLX_RELATIVE_SPEED=0.12 MLX_TILING="$MLX_TILING" \
      nohup python grpc_worker.py --backend mlx --host 127.0.0.1 --port "$WORKER_PORT" \
          --mlx-model-dir "$MODEL_DIR" --mlx-ops framework >"$LOG_DIR/worker.log" 2>&1 & )
  sleep 4
else
  say "MLX worker already listening on :$WORKER_PORT"
fi

# 2) agent gateway. WAN_WORKERS = this Mac's worker + any PEERS (other Macs over Thunderbolt).
#    With >1 worker, enable POOL mode: each job runs DIRECT (no-refine) on ONE Mac, N in parallel.
WORKERS="127.0.0.1:$WORKER_PORT"
POOL="0"
if [ -n "$PEERS" ]; then WORKERS="$WORKERS,$PEERS"; POOL="1"; fi
say "starting agent_gateway on 127.0.0.1:$PORT  (workers=$WORKERS, pool=$POOL)"
env WAN_WORKERS="$WORKERS" \
    AGENT_GATEWAY_WORKER_POOL="$POOL" \
    ORCHESTRATOR_PATH="$ORCH" \
    AGENT_GATEWAY_API_KEY="$API_KEY" \
    AGENT_GATEWAY_JOBS_DIR="$HOME/.openmontage-jobs" \
    nohup python services/agent_gateway/server.py --host 127.0.0.1 --port "$PORT" \
        >"$LOG_DIR/gateway.log" 2>&1 &
sleep 4
curl -s "http://127.0.0.1:$PORT/healthz" || die "gateway health check failed — see $LOG_DIR/gateway.log"

cat <<EOF

\033[1;32mLocal service is up:\033[0m  http://127.0.0.1:$PORT  (logs in $LOG_DIR)
  $([ -n "$API_KEY" ] && echo "auth: send header  X-API-Key: $API_KEY" || echo "auth: OFF (set API_KEY=... for a public deployment)")

Next: expose it at https://kakeya.ai with a Cloudflare Tunnel (outbound; no open ports):
  brew install cloudflared
  # Option A (dashboard token):  cloudflared service install <TOKEN>     # add public hostname kakeya.ai -> http://localhost:$PORT
  # Option B (CLI):              cloudflared tunnel login && cloudflared tunnel create kakeya-gw \\
  #                              && cloudflared tunnel route dns kakeya-gw kakeya.ai \\
  #                              && cloudflared tunnel run --url http://localhost:$PORT kakeya-gw
See services/agent_gateway/deploy/cloudflare.md for the full walk-through.
EOF
