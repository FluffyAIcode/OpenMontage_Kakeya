"""Offline tests for the two-Mac proposer->refiner pipeline (ADR 0014).

No GPU / no mlx-video. Two things are covered without any model:

  1. MlxBackend._sr_refine: the stock-mlx-video (no vid2vid) refine path is a pure
     PIL/numpy Lanczos super-resolution upscale. We feed it a tiny mp4 and assert it
     returns frames at the requested target resolution.

  2. The orchestrator's --single-refine pipeline end-to-end over real gRPC, using two
     in-process TestBackend workers: one advertises 'framework' (proposer/head), the
     other 'refine' (refiner/headless). Proves capability routing + single-pass refine
     produce an mp4 at the target out resolution.

Run:
    pip install grpcio numpy pillow imageio imageio-ffmpeg pytest
    pytest tests/tools/test_distributed_wan_pipeline.py -q
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DWAN = REPO / "services" / "distributed_wan"

grpc = pytest.importorskip("grpc")
np = pytest.importorskip("numpy")
iio = pytest.importorskip("imageio.v2")
pytest.importorskip("PIL")

sys.path.insert(0, str(DWAN))
import video_worker_pb2 as pb  # noqa: E402
import video_worker_pb2_grpc as pb_grpc  # noqa: E402
import grpc_worker as gw  # noqa: E402


def _tiny_mp4(n=5, h=32, w=64) -> bytes:
    frames = (np.random.default_rng(0).integers(0, 255, (n, h, w, 3))).astype(np.uint8)
    buf = io.BytesIO()
    iio.mimwrite(buf, [frames[i] for i in range(n)], format="mp4", fps=12, codec="libx264")
    return buf.getvalue()


def test_mlx_sr_refine_upscales_without_vid2vid(monkeypatch):
    """No MLX_V2V_FLAG -> SR path: returns frames at the requested target resolution."""
    monkeypatch.delenv("MLX_V2V_FLAG", raising=False)
    backend = gw.MlxBackend(model_dir="/nonexistent", ops=["refine"])
    assert backend._refine_mode() == "sr"
    # health advertises the SR mode honestly
    assert "refine=sr" in backend.health().note

    req = types.SimpleNamespace(prompt="x", mp4=_tiny_mp4(n=5, h=32, w=64),
                                width=96, height=48, num_frames=5, steps=8, strength=0.6, seed=7)
    import queue
    out = backend.refine(req, queue.Queue(maxsize=64))
    assert out.shape == (5, 48, 96, 3)
    assert out.dtype == np.uint8


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _serve(ops):
    server = grpc.server(__import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(max_workers=4),
                         options=[("grpc.max_send_message_length", 64 * 1024 * 1024),
                                  ("grpc.max_receive_message_length", 64 * 1024 * 1024)])
    pb_grpc.add_VideoWorkerServicer_to_server(gw.VideoWorkerServicer(gw.TestBackend(), ops), server)
    port = _free_port()
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    return server, port


def test_orchestrator_single_refine_pipeline(tmp_path):
    """Two in-process workers (framework + refine) -> --single-refine produces an mp4 at out-res."""
    prop, p_prop = _serve(["framework"])   # head / proposer
    refn, p_refn = _serve(["refine"])      # headless / refiner
    import time as _t
    _t.sleep(0.5)  # let both gRPC servers be ready before the orchestrator subprocess dials
    try:
        out = tmp_path / "pipe.mp4"
        env = {**os.environ, "WAN_WORKERS": f"127.0.0.1:{p_prop},127.0.0.1:{p_refn}"}
        proc = subprocess.run(
            [sys.executable, str(DWAN / "grpc_orchestrator.py"), "--prompt", "a fox in snow",
             "--single-refine", "--frames", "5", "--fw-frames", "5", "--fw-width", "64",
             "--fw-height", "32", "--out-width", "96", "--out-height", "48", "--out", str(out)],
            env=env, capture_output=True, text=True, timeout=120)
        assert out.exists(), proc.stdout + proc.stderr
        done = [l for l in proc.stdout.splitlines() if l.startswith("ORCH_DONE")]
        assert done, proc.stdout + proc.stderr
        info = json.loads(done[-1].split(" ", 1)[1])
        assert info["mode"] == "pipeline"
        assert info["proposer"] == f"127.0.0.1:{p_prop}"
        assert info["refiner"] == f"127.0.0.1:{p_refn}"
        assert info["px"] == [48, 96]
        assert info["frames"] == 5
    finally:
        prop.stop(0); refn.stop(0)
