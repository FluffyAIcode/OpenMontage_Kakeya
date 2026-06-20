# Video Inference Gateway (reference server)

A standalone, **real** diffusers-serving gateway that hosts OpenMontage's local
open-source video models (WAN 2.1 / Hunyuan / CogVideo / LTX) **warm** behind one
HTTP API. OpenMontage's `wan_video` / `hunyuan_video` / `cogvideo_video` /
`ltx_video_local` tools route here when `VIDEO_INFER_ENDPOINT` is set, instead of
cold-loading a diffusers pipeline on every call.

See `docs/adr/0002-unified-local-video-inference-backend.md` for the design and the
HTTP contract (§5). This server is **not Kakeya** — Kakeya is an LLM token engine
and cannot host video diffusion (ADR 0002 §2).

## Run (on a GPU box)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 8080 \
    --preload THUDM/CogVideoX-2b      # optional: warm a model at startup
```

## Point OpenMontage at it

```bash
# in OpenMontage .env
VIDEO_INFER_ENDPOINT=http://<gpu-host-or-tunnel>:8080
```

Then `wan_video` / `cogvideo_video` / … report `available` and generate via the
gateway — no local torch/diffusers needed in OpenMontage.

## Endpoints

- `GET /healthz` → `{"status":"ok","device":"cuda","loaded":[...],"known":[...]}`
- `POST /v1/video/generations` → raw `video/mp4` bytes (see ADR 0002 §5 for the body)

## Notes

- Models are lazy-loaded on first request and kept warm (keyed by HF model id).
- VRAM is the warm-pool budget; **disk** is the weights budget — each model family
  pulls multi-GB weights (text encoders dominate). Provision disk accordingly.
- Real throughput batching / VRAM-aware admission is a future enhancement; the
  warm-pool already removes the per-call cold-load cost.
