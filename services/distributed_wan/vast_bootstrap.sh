#!/usr/bin/env bash
# =============================================================================
# Idempotent bootstrap for a (possibly FRESH) vast.ai box -> CUDA refiner worker.
# RUN ON THE VAST BOX. Safe to re-run; the head supervisor calls it every cycle.
# =============================================================================
# vast instances are ephemeral (released/recreated): each recreate is a clean box with no torch and
# no model cache. This script makes a fresh box re-converge to a running refiner:
#   1) install deps into the venv only if missing,
#   2) launch grpc_worker (cuda, refine-only, preloaded) in a tmux session if not already listening.
# The head Mac scps grpc_worker.py + video_worker_pb2*.py into $DISTWAN before calling this.
#
# Env (override as needed):
#   VENV_PY   python in the box's venv         (default /venv/main/bin/python)
#   DISTWAN   dir holding the worker files      (default /workspace/distwan)
#   PORT      gRPC listen port                  (default 50051)
#   HF_HOME   model cache (persist if possible) (default /workspace/.hf_home)
# =============================================================================
set -euo pipefail

VENV_PY="${VENV_PY:-/venv/main/bin/python}"
DISTWAN="${DISTWAN:-/workspace/distwan}"
PORT="${PORT:-50051}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
LOG="${LOG:-$DISTWAN/worker.log}"

[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"
PIP="$VENV_PY -m pip"
cd "$DISTWAN" 2>/dev/null || { echo "ERR: $DISTWAN not found (head should scp worker files here)"; exit 1; }
[ -f grpc_worker.py ] || { echo "ERR: grpc_worker.py missing in $DISTWAN"; exit 1; }

# 0) already healthy? (worker listening) -> no-op (idempotent fast path)
if "$VENV_PY" - "$PORT" <<'PY' 2>/dev/null
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(("127.0.0.1",int(sys.argv[1])))==0 else 1)
PY
then echo "[bootstrap] worker already listening on :$PORT"; exit 0; fi

# 1) deps: install only what's missing (torch is the big one; ftfy+protobuf are easy to forget)
need=$("$VENV_PY" - <<'PY'
mods = {"torch":"torch","diffusers":"diffusers","transformers":"transformers","peft":"peft",
        "accelerate":"accelerate","grpc":"grpcio","google.protobuf":"protobuf","ftfy":"ftfy",
        "imageio":"imageio","imageio_ffmpeg":"imageio-ffmpeg","safetensors":"safetensors"}
miss=[]
for imp,pkg in mods.items():
    try: __import__(imp)
    except Exception: miss.append(pkg)
print(" ".join(miss))
PY
)
if [ -n "${need// }" ]; then
  echo "[bootstrap] installing missing deps: $need"
  $PIP install --no-input -q $need
else
  echo "[bootstrap] deps already present"
fi

# 2) launch the worker. OPS/preload depend on whether an I2V-720P model is configured:
#   CUDA_I2V_MODEL set -> ops "framework,refine,i2v" for long-form (ADR 0015 Phase 2b), lazy i2v load.
#   else               -> ops "refine --preload" (the standard refiner).
# Durability note: vast images vary — some kill user processes on SSH logout (no systemd/linger),
# which also kills tmux. On those, run the worker under a persistent connection (or enable linger).
if [ -n "${CUDA_I2V_MODEL:-}" ]; then
  OPS="framework,refine,i2v"; PRE=""
else
  OPS="refine"; PRE="--preload"
fi
# BOOTSTRAP_NO_LAUNCH=1 -> install deps only; the caller (supervisor held-worker mode) launches the
# worker under a persistent connection (for vast images that kill processes on SSH logout).
if [ "${BOOTSTRAP_NO_LAUNCH:-0}" = "1" ]; then
  echo "[bootstrap] deps ready; NO_LAUNCH (caller holds the worker)"; exit 0
fi
command -v tmux >/dev/null || { apt-get update -y >/dev/null 2>&1 && apt-get install -y tmux >/dev/null 2>&1; }
tmux kill-session -t refiner 2>/dev/null || true
tmux new-session -d -s refiner \
  "cd '$DISTWAN' && HF_HOME='$HF_HOME' CUDA_I2V_MODEL='${CUDA_I2V_MODEL:-}' CUDA_I2V_OFFLOAD='${CUDA_I2V_OFFLOAD:-0}' '$VENV_PY' grpc_worker.py --backend cuda --host 0.0.0.0 --port '$PORT' --ops $OPS $PRE 2>&1 | tee '$LOG'"
echo "[bootstrap] launched worker (ops=$OPS) in tmux 'refiner'; log=$LOG"
