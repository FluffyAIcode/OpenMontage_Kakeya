#!/usr/bin/env bash
# =============================================================================
# Mac mini -> distributed-WAN MLX gRPC worker ("another GPU").  RUN ON THE MAC.
# =============================================================================
# Turns an Apple-Silicon Mac mini into a gRPC VideoWorker (ADR 0010) backed by
# mlx-video (ADR 0008), so the cloud-agent orchestrator can dispatch WAN work to
# it alongside a vast CUDA box.  The cloud agent CANNOT reach your local Mac, so
# YOU run this; expose the worker via Tailscale for off-LAN access.
#
# Honest scope: mlx-video does T2V (and I2V); it usually has NO vid2vid. So by
# default this worker advertises the FRAMEWORK/T2V op only, and high-res tile
# REFINES run on a CUDA worker. If your mlx-video build has vid2vid, set
# MLX_OPS="framework,refine" and MLX_V2V_FLAG to its video-input flag.
#
# Usage:
#   bash mac_setup.sh                 # full setup + run on :50051
#   PORT=50051 MLX_OPS=framework bash mac_setup.sh
#   STEP=run bash mac_setup.sh        # skip setup, just run (after first setup)
# =============================================================================
set -euo pipefail

# ---- config (override via env) ----
REPO_URL="${REPO_URL:-https://github.com/FluffyAIcode/OpenMontage_Kakeya}"
# Branch that contains the gRPC worker + proto + this script. Until PR #2 is merged to
# main, the gRPC files live ONLY on this branch — cloning main will NOT have them.
REPO_BRANCH="${REPO_BRANCH:-AgentMemory/wan-mlx-feasibility-cc88}"
WORKDIR="${WORKDIR:-$HOME/openmontage-mac}"
VENV="${VENV:-$HOME/.venv-distwan}"
HF_MODEL="${HF_MODEL:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
MODEL_DIR="${MODEL_DIR:-$HOME/wan21_mlx}"
PORT="${PORT:-50051}"
MLX_OPS="${MLX_OPS:-framework}"                 # add ',refine' only if your mlx-video has vid2vid
MLX_RELATIVE_SPEED="${MLX_RELATIVE_SPEED:-0.12}" # speed hint for the orchestrator's weighting
STEP="${STEP:-all}"                              # all | setup | run

say() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
die() { printf "\n\033[1;31mERROR: %s\033[0m\n" "$*" >&2; exit 1; }

# ---- 0. preflight ----
if [ "$STEP" != "run" ]; then
  say "0. preflight (Apple Silicon, macOS >= 14, Python >= 3.11)"
  [ "$(uname -s)" = "Darwin" ] || die "must run on macOS"
  [ "$(uname -m)" = "arm64" ] || die "must run on Apple Silicon (arm64); got $(uname -m)"
  osmajor="$(sw_vers -productVersion | cut -d. -f1)"; [ "$osmajor" -ge 14 ] || die "need macOS >= 14 (got $(sw_vers -productVersion))"
  command -v python3 >/dev/null || die "python3 not found (install Python 3.11+: brew install python@3.12)"
  pyok="$(python3 -c 'import sys; print(1 if sys.version_info[:2] >= (3,11) else 0)')"
  [ "$pyok" = "1" ] || die "need Python >= 3.11 (got $(python3 -V))"
  echo "OK: $(sw_vers -productName) $(sw_vers -productVersion) arm64, $(python3 -V)"
fi

# ---- 1. venv + deps ----
if [ "$STEP" = "all" ] || [ "$STEP" = "setup" ]; then
  say "1. virtualenv + dependencies"
  python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -U pip
  # mlx-video pulls mlx; the rest are the worker's needs. (imageio-ffmpeg bundles ffmpeg.)
  python -m pip install -U mlx mlx-video grpcio grpcio-tools \
      imageio imageio-ffmpeg numpy pillow huggingface_hub
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# ---- 2. worker code + gRPC stubs ----
if [ "$STEP" = "all" ] || [ "$STEP" = "setup" ]; then
  say "2. fetch worker code + generate gRPC stubs"
  if [ ! -d "$WORKDIR/.git" ]; then
    git clone -b "$REPO_BRANCH" --depth 1 "$REPO_URL" "$WORKDIR"
  else
    git -C "$WORKDIR" fetch origin "$REPO_BRANCH" && git -C "$WORKDIR" checkout "$REPO_BRANCH" && git -C "$WORKDIR" pull --ff-only || true
  fi
  [ -f "$WORKDIR/services/distributed_wan/grpc_worker.py" ] || die "grpc_worker.py not found after clone — wrong branch? ($REPO_BRANCH)"
  cd "$WORKDIR/services/distributed_wan"
  python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/video_worker.proto
  # generated grpc stub uses a flat import; ensure it resolves as a flat module
  sed -i '' 's/^from \. import video_worker_pb2 as/import video_worker_pb2 as/' video_worker_pb2_grpc.py 2>/dev/null || true
  echo "OK: stubs generated in $(pwd)"
fi
cd "$WORKDIR/services/distributed_wan"

# ---- 3. convert WAN 2.1 1.3B -> MLX (one-time) ----
if { [ "$STEP" = "all" ] || [ "$STEP" = "setup" ]; } && [ ! -d "$MODEL_DIR" ]; then
  say "3. convert $HF_MODEL -> MLX ($MODEL_DIR)"
  # Per Blaizzy/mlx-video. If your mlx-video's convert entrypoint/flags differ,
  # run its documented conversion once and point MODEL_DIR at the result.
  if python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('mlx_video.wan_2.convert') else 1)"; then
    python -m mlx_video.wan_2.convert --hf-model "$HF_MODEL" --out "$MODEL_DIR"
  else
    cat >&2 <<EOF
NOTE: couldn't find 'mlx_video.wan_2.convert'. Convert with your mlx-video's
documented command (see its README), e.g.:
    python -m mlx_video.convert --model $HF_MODEL --out $MODEL_DIR
then re-run:  MODEL_DIR=$MODEL_DIR STEP=run bash mac_setup.sh
EOF
    exit 1
  fi
fi

# ---- 4. (optional) Tailscale so the cloud orchestrator can reach this Mac ----
say "4. reachability"
cat <<EOF
For off-LAN access by the cloud-agent orchestrator, join a tailnet:
    brew install tailscale && sudo tailscale up      # note this node's MagicDNS name
Then the orchestrator host uses:  WAN_WORKERS="<vast-host>:50051,<this-mac>:${PORT}"
(On the same LAN you can skip Tailscale and use this Mac's LAN IP.)
EOF

# ---- 5. run the MLX gRPC worker ----
if [ "$STEP" = "all" ] || [ "$STEP" = "run" ]; then
  say "5. starting MLX gRPC worker on 0.0.0.0:${PORT}  (ops=${MLX_OPS})"
  [ -d "$MODEL_DIR" ] || die "MODEL_DIR '$MODEL_DIR' missing — run conversion first (STEP=setup)"
  exec env MLX_RELATIVE_SPEED="$MLX_RELATIVE_SPEED" \
       python grpc_worker.py --backend mlx --host 0.0.0.0 --port "$PORT" \
           --mlx-model-dir "$MODEL_DIR" --mlx-ops "$MLX_OPS"
fi
