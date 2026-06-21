# Distributed WAN inference — Mac mini + vast CUDA, across regions

> **Transport: gRPC** (`proto/video_worker.proto`, `grpc_worker.py`, `grpc_orchestrator.py`)
> is the product contract — typed schema, **server-streaming progress**, deadlines,
> capability negotiation, speed-weighted routing (ADR 0010). The earlier HTTP
> `worker.py`/`orchestrator.py` remain as the simpler reference (ADR 0009). The Mac runs an
> **MLX** worker (`mac_setup.sh`, wrapping `mlx-video`, ADR 0008) as "another GPU"; vast runs
> the **CUDA** worker. Run `python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/video_worker.proto`
> to (re)generate stubs.



Lets a **cloud agent** drive a WAN video production that uses **both** a local **Mac mini**
GPU and one-or-more **vast** GPUs that sit in **different regions** — *correctly*, within the
real constraints (see `docs/adr/0006-distributed-wan-across-mac-and-vast.md`).

## The hard constraints (why it's heterogeneous, not tensor-parallel)

- **WAN runs ONLY on CUDA.** WAN 2.1 is a PyTorch/diffusers model. The Mac's Apple-Silicon
  **MLX** GPU cannot run WAN. So the Mac is **not** a WAN compute node.
- **Cross-region kills tensor/pipeline parallelism.** Per-step tensor exchange over
  tens–hundreds of ms RTT is hopeless. Only **coarse, latency-tolerant** traffic crosses the
  wire: prompts (bytes) and base64-mp4 clips. **No per-step tensors ever leave a box.**

## Mac mini — complete setup (one script)

Run on the Mac (Apple Silicon, macOS ≥ 14). It is **owner-run** — the cloud agent cannot
reach a local Mac (ADR 0005).

```bash
# full setup + run (clones repo, venv+deps, generates stubs, converts WAN->MLX, runs worker)
bash services/distributed_wan/mac_setup.sh
# later, just run:            STEP=run bash services/distributed_wan/mac_setup.sh
# enable refine if your mlx-video has vid2vid:
#   MLX_OPS="framework,refine" MLX_V2V_FLAG="--video" bash services/distributed_wan/mac_setup.sh
```

It performs: preflight (arm64 / macOS≥14 / Python≥3.11) → venv + `mlx mlx-video grpcio
grpcio-tools imageio ...` → clone repo + `protoc` the stubs → convert `Wan2.1-T2V-1.3B` to
MLX → (Tailscale hint) → start `grpc_worker.py --backend mlx`.

**Honest role:** mlx-video does **T2V/I2V**, usually **no vid2vid**, so the Mac advertises the
**framework/T2V** op by default and high-res **refines run on a CUDA worker**. Version-adaptable
env knobs (no code edits): `MLX_T2V_MODULE`, `MLX_PASS_DIMS`, `MLX_OPS`, `MLX_V2V_FLAG`,
`MODEL_DIR`, `PORT`, `MLX_RELATIVE_SPEED`. If a flag differs in your mlx-video build the worker
**fails loudly** with the last output (no silent garbage).

Verify, then point the orchestrator at it:
```bash
# on any host that can reach the Mac (LAN IP or Tailscale name):
python - <<'PY'
import grpc, video_worker_pb2 as pb, video_worker_pb2_grpc as g
h=g.VideoWorkerStub(grpc.insecure_channel("<mac>:50051")).Health(pb.HealthRequest(),timeout=15)
print(h.backend, h.device, list(h.ops), h.relative_speed)   # expect mlx-video mlx ['framework'...]
PY
WAN_WORKERS="<vast-host>:50051,<mac>:50051" python services/distributed_wan/grpc_orchestrator.py --prompt "..." --out final.mp4
```

## The design (what each node does)

```
 cloud agent (orchestrator)
        │  1) text: expand -> per-tile prompts        (HTTP, ms-tolerant)
        ├───────────────► Mac mini (MLX)  Kakeya text server  [KAKEYA_ENDPOINT]
        │  2) framework + 4) tile refines (base64-mp4) (HTTP, ms-tolerant)
        └───────────────► vast CUDA worker(s)  WAN proposer + vid2vid refine [WAN_WORKERS]
        3) upscale+crop  5) weight-map merge -> final.mp4   (on the orchestrator)
```

- **Mac mini (MLX)** — `worker`? No: the Mac runs **Kakeya's MLX *text* server**
  (`scripts/serve.py --backend mlx`, see ADR 0005). It only specializes prompts.
- **vast CUDA** — `worker.py`: distilled CausVid proposer (framework) + full-WAN vid2vid
  refine (tiles). **Run ≥2 workers on co-located CUDA GPUs for real parallel refine.**
- **cloud agent** — `orchestrator.py`: text → framework → tile crops → concurrent refine →
  merge. The framework anchors tile overlaps so independent refine stays seamless
  (ADR 0004 capstone), which is exactly why tiles can be refined on *different* workers.

## Run

**On each vast CUDA box (a worker):**
```bash
pip install fastapi "uvicorn[standard]" diffusers transformers peft accelerate \
    imageio imageio-ffmpeg sentencepiece safetensors ftfy
python services/distributed_wan/worker.py --port 9000 --preload
```

**On the Mac mini (optional text plane, run by the owner — see ADR 0005):**
```bash
PYTHONPATH=.:sdks/python python3 scripts/serve.py --backend mlx \
    --verifier-id mlx-community/Qwen3-1.7B-4bit --host 0.0.0.0 --port 8000
# expose via Tailscale so the cloud agent can reach it across regions
```

**On the cloud agent (orchestrator):**
```bash
pip install requests imageio imageio-ffmpeg numpy pillow
export WAN_WORKERS="http://<vast1-ip-or-tunnel>:9000,http://<vast2>:9000"  # 1+ workers
export KAKEYA_ENDPOINT="http://<mac-tailnet-name>:8000"                    # optional
python services/distributed_wan/orchestrator.py --prompt "a serene koi pond at golden hour" \
    --out final.mp4
```

Cross-region reachability: vast workers via public IP or an SSH tunnel; the Mac via
**Tailscale** (the Mac is NAT'd/outbound-only). Only prompts + mp4 clips cross the wire.

## What is and isn't "distributed"

- **Distributed (real):** the WAN **tile refines fan out concurrently to N CUDA workers** —
  add workers, get parallel speedup. The Mac adds the text plane in parallel.
- **Not possible:** splitting a single WAN forward (tensor/pipeline parallel) across Mac+vast
  or across regions. WAN on MLX doesn't exist; cross-region latency forbids it.
