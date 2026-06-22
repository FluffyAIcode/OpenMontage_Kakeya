#!/usr/bin/env bash
# =============================================================================
# Head-Mac supervisor for an EPHEMERAL vast CUDA refiner. RUN ON THE HEAD MAC
# (under launchd). Keeps the 3-node cluster self-healing as vast is released/recreated:
#
#   - vast reachable  -> scp worker files, run vast_bootstrap.sh (idempotent install+launch),
#                        (re)open the SSH tunnel  local:VAST_LOCAL_PORT -> vast:VAST_REMOTE_PORT,
#                        and ADD 127.0.0.1:VAST_LOCAL_PORT to the gateway's dynamic workers file.
#   - vast gone/loading -> REMOVE vast from the workers file -> gateway auto-falls back to the
#                        2-Mac pipeline (head proposer + Mac B refine via f_theta). No restart.
#
# The gateway must run with AGENT_GATEWAY_WORKERS_FILE=$WORKERS_FILE so membership is dynamic.
#
# Config: ~/.kakeya/vast.env (sourced). Example:
#   VAST_SSH_HOST=104.202.252.41
#   VAST_SSH_PORT=20006
#   VAST_SSH_USER=root
#   VAST_SSH_KEY=/Users/fluffy314/.ssh/id_ed25519
#   VAST_REMOTE_PORT=50051
#   VAST_LOCAL_PORT=50052
#   BASE_WORKERS=127.0.0.1:50051,192.168.68.51:50051   # head proposer + Mac B refiner (always-on)
#   WORKERS_FILE=/Users/fluffy314/.kakeya/wan_workers
#   DISTWAN_LOCAL=/Users/fluffy314/openmontage-mac/services/distributed_wan
#   INTERVAL=30
#   # optional auto-endpoint discovery (prints "HOST PORT"); overrides VAST_SSH_HOST/PORT:
#   # VAST_RESOLVE_CMD="vastai show instances --raw | python3 .../pick_endpoint.py kakeya-refiner"
# =============================================================================
set -uo pipefail

CONF="${VAST_ENV:-$HOME/.kakeya/vast.env}"
log() { printf '%s [vast-sup] %s\n' "$(date '+%H:%M:%S')" "$*"; }

write_workers() {  # atomic write of the gateway's dynamic worker list
  local content="$1" tmp
  tmp="$(mktemp)"; printf '%s\n' "$content" > "$tmp"; mv "$tmp" "$WORKERS_FILE"
}

reachable_local() {  # is the tunnel's local port accepting connections?
  nc -z 127.0.0.1 "$VAST_LOCAL_PORT" >/dev/null 2>&1
}

remote_listening() {  # is the worker listening on the vast box?
  ssh $SSH_OPTS -p "$VAST_SSH_PORT" "$VAST_SSH_USER@$VAST_SSH_HOST" \
    "bash -c 'exec 3<>/dev/tcp/127.0.0.1/$VAST_REMOTE_PORT' 2>/dev/null && echo up" 2>/dev/null | grep -q up
}

once() {
  [ -f "$CONF" ] || { log "no $CONF; nothing to supervise"; return; }
  # shellcheck disable=SC1090
  source "$CONF"
  : "${VAST_SSH_USER:=root}" "${VAST_REMOTE_PORT:=50051}" "${VAST_LOCAL_PORT:=50052}"
  : "${WORKERS_FILE:=$HOME/.kakeya/wan_workers}" "${INTERVAL:=30}"
  mkdir -p "$(dirname "$WORKERS_FILE")"
  SSH_OPTS="-i ${VAST_SSH_KEY:-$HOME/.ssh/id_ed25519} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"

  # optional auto-discovery of the current instance endpoint
  if [ -n "${VAST_RESOLVE_CMD:-}" ]; then
    read -r H P < <(eval "$VAST_RESOLVE_CMD" 2>/dev/null || true)
    [ -n "${H:-}" ] && VAST_SSH_HOST="$H"; [ -n "${P:-}" ] && VAST_SSH_PORT="$P"
  fi

  # vast not configured/known -> 2-Mac fallback
  if [ -z "${VAST_SSH_HOST:-}" ] || [ -z "${VAST_SSH_PORT:-}" ]; then
    write_workers "$BASE_WORKERS"; log "vast endpoint unknown -> 2-Mac fallback"; return
  fi

  # vast unreachable (released/destroyed) -> drop it, fall back
  if ! ssh $SSH_OPTS -p "$VAST_SSH_PORT" "$VAST_SSH_USER@$VAST_SSH_HOST" true 2>/dev/null; then
    pkill -f "L $VAST_LOCAL_PORT:localhost:$VAST_REMOTE_PORT" 2>/dev/null
    write_workers "$BASE_WORKERS"; log "vast $VAST_SSH_HOST:$VAST_SSH_PORT unreachable -> 2-Mac fallback"; return
  fi

  # reachable: push code + idempotent bootstrap (installs deps + launches worker on a fresh box)
  ssh $SSH_OPTS -p "$VAST_SSH_PORT" "$VAST_SSH_USER@$VAST_SSH_HOST" "mkdir -p /workspace/distwan" 2>/dev/null
  scp $SSH_OPTS -P "$VAST_SSH_PORT" \
      "$DISTWAN_LOCAL/grpc_worker.py" "$DISTWAN_LOCAL/video_worker_pb2.py" \
      "$DISTWAN_LOCAL/video_worker_pb2_grpc.py" "$DISTWAN_LOCAL/vast_bootstrap.sh" \
      "$VAST_SSH_USER@$VAST_SSH_HOST:/workspace/distwan/" 2>/dev/null
  ssh $SSH_OPTS -p "$VAST_SSH_PORT" "$VAST_SSH_USER@$VAST_SSH_HOST" \
      "bash /workspace/distwan/vast_bootstrap.sh" 2>/dev/null

  # worker still loading the model? keep vast OUT until it actually listens (no broken jobs)
  if ! remote_listening; then
    write_workers "$BASE_WORKERS"; log "vast reachable but worker not listening yet (loading) -> 2-Mac for now"; return
  fi

  # ensure the tunnel; re-open if down
  if ! reachable_local; then
    pkill -f "L $VAST_LOCAL_PORT:localhost:$VAST_REMOTE_PORT" 2>/dev/null; sleep 1
    ssh $SSH_OPTS -o ExitOnForwardFailure=yes -fN \
        -L "$VAST_LOCAL_PORT:localhost:$VAST_REMOTE_PORT" -p "$VAST_SSH_PORT" \
        "$VAST_SSH_USER@$VAST_SSH_HOST" 2>/dev/null
    sleep 2
  fi

  if reachable_local; then
    write_workers "$BASE_WORKERS,127.0.0.1:$VAST_LOCAL_PORT"; log "vast refiner ONLINE -> 3-node (added 127.0.0.1:$VAST_LOCAL_PORT)"
  else
    write_workers "$BASE_WORKERS"; log "tunnel failed -> 2-Mac fallback"
  fi
}

# Initialize fallback immediately so the gateway always has a valid list, then supervise forever.
if [ -f "$CONF" ]; then source "$CONF"; mkdir -p "$(dirname "${WORKERS_FILE:=$HOME/.kakeya/wan_workers}")"; [ -f "$WORKERS_FILE" ] || write_workers "${BASE_WORKERS:-127.0.0.1:50051}"; fi
while true; do once; sleep "${INTERVAL:-30}"; done
