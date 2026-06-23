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


def test_interpolate_helper():
    """_interpolate turns T frames into (T-1)*factor + 1 (smoothness/fps lever)."""
    import grpc_orchestrator as orch
    frames = np.zeros((5, 8, 8, 3), np.uint8)
    frames[1] = 255
    assert orch._interpolate(frames, 1).shape[0] == 5
    assert orch._interpolate(frames, 2).shape[0] == 9    # (5-1)*2 + 1
    assert orch._interpolate(frames, 4).shape[0] == 17   # (5-1)*4 + 1
    blended = orch._interpolate(frames, 2)
    assert 0 < int(blended[1].max()) < 255  # midpoint between frame0(0) and frame1(255) is blended


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


def test_stitch_helper():
    """_stitch concatenates chunks with crossfade: total = sum(len) - (k-1)*overlap."""
    import grpc_orchestrator as orch
    a = np.zeros((10, 8, 8, 3), np.uint8); b = np.zeros((10, 8, 8, 3), np.uint8); cc = np.zeros((10, 8, 8, 3), np.uint8)
    assert orch._stitch([a], 4).shape[0] == 10
    assert orch._stitch([a, b], 4).shape[0] == 16           # 20 - 4
    assert orch._stitch([a, b, cc], 4).shape[0] == 22       # 30 - 2*4
    assert orch._stitch([a, b], 0).shape[0] == 20           # no overlap


def test_orchestrator_longform_chunks(tmp_path):
    """--chunks 3 on an i2v-capable worker -> autoregressive generation + crossfade stitch."""
    gen, p_gen = _serve(["framework", "t2v", "i2v"])  # TestBackend advertises i2v
    import time as _t
    _t.sleep(0.5)
    try:
        out = tmp_path / "long.mp4"
        env = {**os.environ, "WAN_WORKERS": f"127.0.0.1:{p_gen}"}
        proc = subprocess.run(
            [sys.executable, str(DWAN / "grpc_orchestrator.py"), "--prompt", "a flythrough",
             "--chunks", "3", "--chunk-frames", "9", "--chunk-overlap", "2",
             "--out-width", "64", "--out-height", "48", "--fps", "12", "--out", str(out)],
            env=env, capture_output=True, text=True, timeout=120)
        assert out.exists(), proc.stdout + proc.stderr
        info = json.loads([l for l in proc.stdout.splitlines() if l.startswith("ORCH_DONE")][-1].split(" ", 1)[1])
        assert info["mode"] == "longform"
        assert info["chunks"] == 3 and info["continuity"] == "i2v"
        assert info["frames"] == 9 * 3 - 2 * 2   # 23: sum - (k-1)*overlap
        assert info["px"] == [48, 64]
    finally:
        gen.stop(0)


def test_orchestrator_refine_mode_single_fps_interp(tmp_path):
    """--refine-mode single (quality path) + --fps/--interpolate: forces a seam-free full-frame
    refine on the best refiner and the output frame count reflects interpolation."""
    prop, p_prop = _serve(["framework"])
    refn, p_refn = _serve(["refine"])
    import time as _t
    _t.sleep(0.5)
    try:
        out = tmp_path / "q.mp4"
        env = {**os.environ, "WAN_WORKERS": f"127.0.0.1:{p_prop},127.0.0.1:{p_refn}"}
        proc = subprocess.run(
            [sys.executable, str(DWAN / "grpc_orchestrator.py"), "--prompt", "a fox in snow",
             "--refine-mode", "single", "--frames", "5", "--fw-frames", "5", "--fw-width", "64",
             "--fw-height", "32", "--out-width", "96", "--out-height", "48",
             "--fps", "16", "--interpolate", "2", "--out", str(out)],
            env=env, capture_output=True, text=True, timeout=120)
        assert out.exists(), proc.stdout + proc.stderr
        info = json.loads([l for l in proc.stdout.splitlines() if l.startswith("ORCH_DONE")][-1].split(" ", 1)[1])
        assert info["mode"] == "pipeline"
        assert info["fps"] == 16 and info["interpolate"] == 2
        assert info["frames"] == 9   # (5-1)*2 + 1 after interpolation
    finally:
        prop.stop(0); refn.stop(0)
