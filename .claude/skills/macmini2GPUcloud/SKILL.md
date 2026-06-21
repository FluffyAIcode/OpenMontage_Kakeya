---
name: macmini2GPUcloud
description: SOP for federating a local Apple-Silicon Mac mini (MLX GPU) with a remote cloud CUDA GPU (e.g. vast.ai H200) into ONE distributed WAN video-inference cluster over a gRPC contract + Tailscale. Use when connecting a Mac mini to a cloud GPU across regions/NAT, bringing up the distributed_wan workers/orchestrator, or reproducing the Mac(MLX-proposer)+cloud(CUDA-refiner) pipeline in a new environment. Triggers: "connect Mac mini to cloud GPU", "distributed WAN", "MLX + CUDA cluster", "cross-region gRPC video", "tailnet GPU".
---

# macmini2GPUcloud — Mac mini (MLX) ⇄ Cloud GPU (CUDA) distributed inference SOP

A **migratable runbook** for turning a local Mac mini and a rented cloud GPU box (different
regions, both behind NAT) into a single heterogeneous WAN inference cluster:

- **Mac mini (Apple-Silicon / MLX)** = low-res **proposer** (WAN T2V framework via `mlx-video`).
- **Cloud GPU (NVIDIA / CUDA)** = high-res **refiner** (diffusers WAN vid2vid, tiled, parallel).
- **Transport** = a typed **gRPC** `VideoWorker` contract, server-streaming progress.
- **Connectivity** = **Tailscale** tailnet + (on locked-down cloud containers) a stdlib
  **SOCKS5→TCP forwarder**.

This was validated live: a Mac mini proposer + a vast H200 refiner produced a seamless
`1472×768 × 25-frame` clip across two regions. Code lives in `services/distributed_wan/`
(`grpc_worker.py`, `grpc_orchestrator.py`, `socks5_forward.py`, `mac_setup.sh`,
`proto/video_worker.proto`). Design rationale: ADR 0006/0008/0010.

> **Golden rule (why the split exists):** the Mac is memory-bounded. It must only ever generate
> a **low-res** proposer; the cloud GPU does the heavy high-res refine. Never ask the Mac for
> full-resolution generation — it OOMs on the Metal VAE decode.

---

## 0. Architecture at a glance

```
            ┌────────────────────────────┐         Tailscale tailnet          ┌──────────────────────┐
            │  Cloud GPU box (CUDA)      │  100.x  (WireGuard, NAT-traversal) │  Mac mini (MLX)      │
            │                            │◀──────────────────────────────────▶│                      │
 orchestrator (this box) ──localhost:50051──▶ CUDA worker (refine-only)        │  MLX worker          │
            │   │                        │                                     │  (framework/T2V)     │
            │   └─localhost:55051─▶ socks5_forward.py ─SOCKS5(:1055)─▶ mac:50051 (gRPC) ──────────────┘
            └────────────────────────────┘
```

- Orchestrator dials **two plaintext gRPC endpoints**: the local CUDA worker and a *local*
  forward port that tunnels to the Mac over the tailnet's SOCKS5 proxy.
- `framework` (proposer) → routed to the only framework-capable worker (Mac).
- `refine` tiles → routed to the CUDA worker (mlx-video has no vid2vid).

---

## 1. Prerequisites

| Where | Needs |
|---|---|
| Mac mini | Apple Silicon (arm64), macOS ≥ 14, Python ≥ 3.11, a Tailscale account/login, ~30 GB free disk |
| Cloud box | NVIDIA GPU (≥ 24 GB for WAN refine), Python 3.11+, `pip`, a shell (`tmux` strongly recommended), outbound internet |
| Tailnet | One Tailscale account; both nodes joined to the **same tailnet** |
| Repo | `services/distributed_wan/` checked out on **both** machines (same branch) |

---

## 2. Phase A — Mac mini MLX worker

Run **on the Mac**. The turnkey path is `services/distributed_wan/mac_setup.sh`; the manual
equivalent is below so you understand each step.

### A1. Environment + deps
```bash
python3 -m venv ~/.venv-distwan && source ~/.venv-distwan/bin/activate
python -m pip install -U mlx mlx-video grpcio grpcio-tools imageio imageio-ffmpeg numpy pillow huggingface_hub
```

### A2. Generate the gRPC stubs (from the repo's proto)
```bash
cd <repo>/services/distributed_wan
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/video_worker.proto
# macOS sed needs the '' arg; make the grpc stub use a FLAT import:
sed -i '' 's/^from \. import video_worker_pb2 as/import video_worker_pb2 as/' video_worker_pb2_grpc.py
```

### A3. Convert WAN → MLX (one-time). **Use the NATIVE Wan repo, not `-Diffusers`.**
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', local_dir='$HOME/wan21_ckpt')"
python -m mlx_video.models.wan_2.convert --checkpoint-dir ~/wan21_ckpt --output-dir ~/wan21_mlx --dtype bfloat16
```

### A4. Self-test the MLX path at PROPOSER size (confirms model + no OOM)
```bash
# Do NOT run in HF offline mode (it blocks the umt5-xxl T5 load). Keep frames = 4n+1.
python -m mlx_video.models.wan_2.generate --model-dir ~/wan21_mlx \
    --prompt "a red fox in snowy forest" --output-path /tmp/t.mp4 \
    --width 480 --height 256 --num-frames 13 --steps 6 --seed 11 --tiling aggressive
```
If `/tmp/t.mp4` appears, the worker will work. If it OOMs, shrink (`--width 384 --height 224
--num-frames 9`) and/or keep `--tiling aggressive`.

### A5. Start the MLX gRPC worker — **bind 0.0.0.0** (not 127.0.0.1) so the tailnet can reach it
```bash
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE MLX_RELATIVE_SPEED=0.12 MLX_TILING=aggressive \
  python grpc_worker.py --backend mlx --host 0.0.0.0 --port 50051 \
      --mlx-model-dir ~/wan21_mlx --mlx-ops framework
```
Confirm: `lsof -nP -iTCP:50051 -sTCP:LISTEN` shows `TCP *:50051 (LISTEN)`.

---

## 3. Phase B — Cloud GPU CUDA worker

Run **on the cloud box**. Persist it in `tmux` (cloud SSH sessions drop).

```bash
pip install -U "grpcio>=1.81.1" grpcio-tools diffusers transformers peft accelerate imageio imageio-ffmpeg
cd <repo>/services/distributed_wan
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/video_worker.proto
sed -i 's/^from \. import video_worker_pb2 as/import video_worker_pb2 as/' video_worker_pb2_grpc.py  # GNU sed (no '')

tmux new-session -d -s cuda "export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  python3 grpc_worker.py --backend cuda --host 127.0.0.1 --port 50051 --ops refine --preload 2>&1 | tee /workspace/cuda.log"
```
- `--ops refine` makes the CUDA box **refine-only** so the orchestrator is forced to use the Mac
  for the proposer (otherwise the faster CUDA box wins the framework and the Mac sits idle).
- `--preload` warms the model so the first refine is fast.

---

## 4. Phase C — Connectivity (Tailscale + SOCKS5 forwarder)

### C1. Join both nodes to the same tailnet
- **Mac:** `brew install tailscale && sudo tailscale up` (note its MagicDNS / `100.x` IP).
- **Cloud box, normal host:** `curl -fsSL https://tailscale.com/install.sh | sh && tailscale up`.
- **Cloud box, locked-down container (no `/dev/net/tun`)** — common on vast.ai:
  ```bash
  # userspace networking: no TUN device needed; exposes a SOCKS5 proxy + HTTP proxy
  mkdir -p /tmp/ts && curl -fsSL https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz | tar xz -C /tmp/ts --strip-components=1
  tmux new-session -d -s tsd "/tmp/ts/tailscaled --tun=userspace-networking \
      --socks5-server=localhost:1055 --outbound-http-proxy-listen=localhost:1056 --state=/workspace/ts.state"
  /tmp/ts/tailscale up   # prints an auth URL — the tailnet OWNER must approve the node
  ```
  Verify: `/tmp/ts/tailscale status` lists both nodes; `/tmp/ts/tailscale ping <mac-100.x>` → `pong`.

### C2. Bridge gRPC across userspace networking (the key trick)
With `--tun=userspace-networking`, the kernel **cannot** route `100.x`, so a normal `connect()`
(and thus gRPC) to the Mac fails. The box usually has **no `socat`/`ncat`** either. Use the
repo's stdlib bridge:
```bash
tmux new-session -d -s fwd "python3 socks5_forward.py \
    --listen 127.0.0.1:55051 --socks 127.0.0.1:1055 --target <mac-100.x>:50051"
```
Now `127.0.0.1:55051` on the cloud box tunnels to the Mac's gRPC worker. (A real host with a TUN
device skips this — just dial `<mac-100.x>:50051` directly.)

### C3. Confirm reachability end-to-end
```bash
python3 - <<'PY'
import grpc, video_worker_pb2 as pb, video_worker_pb2_grpc as g
for name,addr in [("cuda","127.0.0.1:50051"),("mac","127.0.0.1:55051")]:
    h=g.VideoWorkerStub(grpc.insecure_channel(addr)).Health(pb.HealthRequest(),timeout=25)
    print(name, h.backend, list(h.ops), "speed=", h.relative_speed)
PY
# Expect: cuda cuda-diffusers ['refine'] ... | mac mlx-video ['framework'] speed=0.12
```

---

## 5. Phase D — Run the distributed job

From the cloud box (the orchestrator):
```bash
WAN_WORKERS="127.0.0.1:50051,127.0.0.1:55051" \
python services/distributed_wan/grpc_orchestrator.py \
    --prompt "a red fox walking through a snowy forest, cinematic" \
    --frames 25 --fw-width 480 --fw-height 256 --fw-frames 13 \
    --proposer-steps 6 --refine-steps 16 --out dwan_mac_vast.mp4
```
- `--fw-*` set the **proposer** (Mac) resolution/frames — keep LOW. The orchestrator temporally
  + spatially resamples the proposer up to the high-res canvas before the CUDA refine.
- Tune Mac memory entirely from here (no Mac change): drop to `--fw-width 384 --fw-height 224
  --fw-frames 9` if the Mac OOMs.

Output: a seamless high-res MP4 + a `_mid.png` preview.

---

## 6. Troubleshooting (every issue hit in the real bring-up)

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'mlx_video.wan_2'` | wrong module path | use `mlx_video.models.wan_2.{generate,convert}` (note `.models.`) |
| Metal `Insufficient Memory` before writeout | full-res VAE decode on the Mac | low-res proposer (`--fw-*`) + `--tiling aggressive`; stop any 2nd model-holding process |
| `Stream removed (Socket closed)` mid-framework | idle HTTP/2 stream dropped during the **silent T5 load** | worker **heartbeat** (already in `grpc_worker.py`) + gRPC keepalive; ensure both ends run the latest code |
| `nc -vz <mac> 50051` → refused | Mac worker bound to `127.0.0.1` | start worker with `--host 0.0.0.0` |
| cloud→Mac connect fails though `tailscale ping` works | userspace networking can't route `100.x` for normal sockets | run `socks5_forward.py`; dial the local forward port |
| `curl -x socks5h://localhost:1055 ... → "Received HTTP/0.9"` | this is **success** — proxy reached the gRPC port; curl just can't parse HTTP/2 | ignore; proceed with gRPC |
| MLX generate fails loading T5 | `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` in env | unset them for the worker (`env -u ...`) |
| `--num-frames` rejected / odd output | WAN needs `4n+1` frames | use 9/13/17/21/25…; worker also snaps down |
| gRPC stub `ImportError` | generated relative import | `sed` it to a flat `import video_worker_pb2 as …` |
| cloud worker won't background | bare `&` killed by SSH session | use `tmux` |
| grpc version error from stubs | `grpcio` too old | `pip install -U "grpcio>=1.81.1"` |
| convert fails on `-Diffusers` repo | wrong checkpoint layout | convert from native `Wan-AI/Wan2.1-T2V-1.3B` |

---

## 7. Migration checklist (new environment)

1. **Pick the GPUs** → set `--mlx-ops framework` (Mac) and `--ops refine` (cloud).
2. **Replace addresses** → Mac `100.x` in `socks5_forward.py --target`; `WAN_WORKERS` on the orchestrator.
3. **TUN or not?** → if `/dev/net/tun` exists on the cloud box, run normal `tailscale up` and dial
   `<mac-100.x>:50051` directly (skip the forwarder). Else use userspace mode + forwarder.
4. **Model cache** → set `HF_HOME` on the cloud box; pre-convert `~/wan21_mlx` on the Mac.
5. **Persistence** → every long-lived process (`tailscaled`, `socks5_forward`, workers,
   orchestrator) in its own `tmux` session.
6. **Health-gate** → always run the Phase C3 health probe before Phase D.
7. **Honest expectation** → this is a *capability/feasibility* federation (heterogeneous MLX+CUDA,
   beyond-native resolution, cross-region), **not** a throughput win: the MLX proposer + per-call
   model reload + WAN latency dominate wall-clock. For raw speed, co-locate CUDA workers.

---

## 8. Security notes
- gRPC channels here are **plaintext (insecure)** — safe only because all traffic rides the
  encrypted WireGuard tailnet. Do **not** expose `:50051` on a public interface.
- The SOCKS5 proxy and forward ports bind to `localhost` only.
- Workers run untrusted model code; keep them on dedicated boxes.
