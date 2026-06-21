#!/usr/bin/env bash
# Mac mini setup — run mlx-video behind the distributed-WAN gRPC worker (ADR 0010).
# RUN THIS ON THE MAC MINI (Apple Silicon, macOS 14+). The cloud agent cannot reach
# your local Mac; this turns it into a gRPC WAN worker ("another GPU") the orchestrator
# can dispatch to (over Tailscale for off-LAN access).
set -euo pipefail

PORT="${PORT:-50051}"
MODEL_DIR="${MODEL_DIR:-$HOME/wan21_mlx}"
# Ops your mlx-video build supports. "framework"/"t2v" are safe; add "refine" ONLY if
# your mlx-video exposes vid2vid (else the orchestrator routes refines to a CUDA worker).
MLX_OPS="${MLX_OPS:-framework}"

echo "== 1. deps =="
python3 -m pip install -U mlx mlx-video grpcio imageio imageio-ffmpeg numpy pillow

echo "== 2. convert WAN 2.1 weights to MLX (one-time) =="
# Per Blaizzy/mlx-video: convert HF Wan2.1-T2V-1.3B-Diffusers -> MLX dir.
if [ ! -d "$MODEL_DIR" ]; then
  python3 -m mlx_video.wan_2.convert \
      --hf-model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --out "$MODEL_DIR" \
    || echo "NOTE: adjust to your mlx-video's convert command if the flags differ."
fi

echo "== 3. (optional) Tailscale so the cloud orchestrator can reach this Mac =="
echo "   brew install tailscale && sudo tailscale up   # note this node's tailnet name"

echo "== 4. start the MLX gRPC worker =="
# grpc_worker.py + the generated stubs (video_worker_pb2*.py) must be present here.
MLX_RELATIVE_SPEED="${MLX_RELATIVE_SPEED:-0.12}" \
python3 grpc_worker.py --backend mlx --host 0.0.0.0 --port "$PORT" \
    --mlx-model-dir "$MODEL_DIR" --mlx-ops "$MLX_OPS"

# Then, on the cloud agent / orchestrator host:
#   WAN_WORKERS="<vast-host>:50051,<mac-tailnet-name>:50051" \
#   python grpc_orchestrator.py --prompt "..." --out final.mp4
# The orchestrator negotiates capabilities + speed-weights tile assignment, so the
# slower Mac gets fewer/smaller tiles and the CUDA box does the heavy lifting.
