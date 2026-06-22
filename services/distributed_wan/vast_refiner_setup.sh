#!/usr/bin/env bash
# =============================================================================
# Bring up a vast.ai (CUDA) box as a REFINER worker for the 3-node topology:
#   head Mac = proposer (framework)  +  headless Mac = refiner (MLX SR)  +  vast = refiner (CUDA V2V)
# RUN THIS ON THE VAST BOX.
# =============================================================================
# The CUDA backend is the *generative* refiner (diffusers WAN 2.1 + CausVid LoRA, real vid2vid),
# so on a 3-node job the orchestrator's tiled refine sends tiles here for true high-res detail
# while the slow MLX Mac picks up its share (round-robin). Advertised refine-only via --ops refine
# so the head Mac stays the sole proposer.
#
# Prereqs on vast: a CUDA torch (vast images ship one), the distributed_wan files in CWD
#   (git clone the repo, or have the head scp grpc_worker.py + video_worker_pb2*.py here).
#
# Usage (on vast):
#   cd <repo>/services/distributed_wan && bash vast_refiner_setup.sh
#   # then make it reachable from the head Mac (see REACHABILITY below)
# =============================================================================
set -euo pipefail

PORT="${PORT:-50051}"
say() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }

[ -f grpc_worker.py ] || { echo "run from services/distributed_wan (grpc_worker.py not found)"; exit 1; }

say "install gRPC + diffusers deps (torch assumed present on the vast image)"
python -m pip install -q grpcio "diffusers>=0.32" transformers peft accelerate \
    imageio imageio-ffmpeg safetensors

say "launch CUDA REFINER worker on 0.0.0.0:$PORT (refine-only, preloaded)"
# --ops refine  -> advertises refine only (head Mac remains the proposer)
# --preload     -> loads WAN 2.1 + CausVid LoRA now so the first tile isn't cold
nohup python grpc_worker.py --backend cuda --host 0.0.0.0 --port "$PORT" \
    --ops refine --preload >/tmp/vast_refiner.log 2>&1 &
sleep 3
echo "log: /tmp/vast_refiner.log   (model download/load can take a few minutes on first run)"

cat <<EOF

== REACHABILITY (head Mac must reach this worker) ==
Pick ONE:

  A) SSH forward tunnel FROM the head Mac (simplest; encrypted; no extra ports):
       # on the head Mac (needs its pubkey in this box's ~/.ssh/authorized_keys):
       ssh -p <vast-ssh-port> -N -f -L 50052:localhost:$PORT root@<vast-ip>
       # then add  127.0.0.1:50052  to the gateway's WAN_WORKERS as the vast refiner.

  B) Tailscale (durable across reboots):
       curl -fsSL https://tailscale.com/install.sh | sh && tailscale up
       tailscale ip -4        # head dials this 100.x.y.z:$PORT directly

  C) vast.ai mapped public port: map a host port -> container $PORT at instance creation,
       then the head dials <vast-ip>:<mapped-port>.

Verify from the head Mac:  nc -vz <addr> <port>
EOF
