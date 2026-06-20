"""OFFLINE SMOKE tests for the warm video-inference gateway seam (ADR 0002).

NOT THE CORRECTNESS GATE. Per ADR 0002 §0 (no fallback/mock/fake/simplify), the
binding correctness gate is `tests/integration/test_real_gpu.py`, which drives a
REAL diffusers gateway on a GPU (CogVideoX-2b verified: h264 720x480, 49 frames).
The stdlib mock server below is fast offline coverage of routing/parsing/error
paths only — it is explicitly NOT evidence the integration generates real video.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tools.base_tool import ToolStatus
from tools.video import _shared
from tools.video.video_infer_client import (
    generate_remote_video,
    remote_health,
    video_infer_endpoint,
)

_FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42FAKEVIDEOBYTES"


class _GatewayHandler(BaseHTTPRequestHandler):
    mode = "bytes"  # "bytes" | "url" | "b64"
    base = ""

    def log_message(self, *args):
        pass

    def _json(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _video(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/healthz":
            self._json(200, {"status": "ok", "models": ["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"]})
        elif self.path == "/dl.mp4":
            self._video(_FAKE_MP4)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/video/generations":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        self._last_req = json.loads(self.rfile.read(length) or b"{}")
        type(self).last_payload = self._last_req
        mode = type(self).mode
        if mode == "bytes":
            self._video(_FAKE_MP4)
        elif mode == "b64":
            self._json(200, {"video_b64": base64.b64encode(_FAKE_MP4).decode()})
        elif mode == "url":
            self._json(200, {"video_url": f"{type(self).base}/dl.mp4"})


@pytest.fixture
def gateway():
    server = HTTPServer(("127.0.0.1", 0), _GatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    _GatewayHandler.base = base
    _GatewayHandler.mode = "bytes"
    try:
        yield base, _GatewayHandler
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# endpoint resolution + availability widening
# ---------------------------------------------------------------------------


def test_endpoint_resolution(monkeypatch):
    monkeypatch.delenv("VIDEO_INFER_ENDPOINT", raising=False)
    assert video_infer_endpoint() is None
    monkeypatch.setenv("VIDEO_INFER_ENDPOINT", "http://x:8000/")
    assert video_infer_endpoint() == "http://x:8000"  # trailing slash trimmed
    # inputs override env
    assert video_infer_endpoint({"video_infer_endpoint": "http://y:9/"}) == "http://y:9"


def test_status_available_with_gateway_without_torch(monkeypatch):
    # No VIDEO_GEN_LOCAL_ENABLED, torch not importable here — but a gateway makes
    # the local tools available anyway (the server owns the heavy deps).
    monkeypatch.delenv("VIDEO_GEN_LOCAL_ENABLED", raising=False)
    monkeypatch.setenv("VIDEO_INFER_ENDPOINT", "http://127.0.0.1:1")
    assert _shared.local_generation_status() == ToolStatus.AVAILABLE


def test_status_unavailable_without_anything(monkeypatch):
    monkeypatch.delenv("VIDEO_INFER_ENDPOINT", raising=False)
    monkeypatch.delenv("VIDEO_GEN_LOCAL_ENABLED", raising=False)
    assert _shared.local_generation_status() == ToolStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_remote_health(gateway):
    base, _ = gateway
    body = remote_health(base)
    assert body["status"] == "ok"
    assert "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" in body["models"]


# ---------------------------------------------------------------------------
# generation: all three response shapes
# ---------------------------------------------------------------------------


def _gen(base, tmp_path, **extra):
    inputs = {
        "prompt": "a calm ocean at dawn",
        "model_variant": "wan2.1-1.3b",
        "output_path": str(tmp_path / "out.mp4"),
        **extra,
    }
    return generate_remote_video(
        endpoint=base,
        tool_name="wan_video",
        variants=_shared.WAN_VARIANTS,
        default_variant="wan2.1-1.3b",
        inputs=inputs,
    )


def test_generate_raw_mp4_bytes(gateway, tmp_path):
    base, handler = gateway
    handler.mode = "bytes"
    result = _gen(base, tmp_path)
    assert result.success, result.error
    assert result.data["mode"] == "remote_gateway"
    assert result.data["model_id"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert (tmp_path / "out.mp4").read_bytes() == _FAKE_MP4
    # payload mapping: variant -> hf model id, defaults applied
    assert handler.last_payload["model"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert handler.last_payload["width"] == 832
    assert handler.last_payload["fps"] == 16


def test_generate_video_b64(gateway, tmp_path):
    base, handler = gateway
    handler.mode = "b64"
    result = _gen(base, tmp_path)
    assert result.success, result.error
    assert (tmp_path / "out.mp4").read_bytes() == _FAKE_MP4


def test_generate_video_url_download(gateway, tmp_path):
    base, handler = gateway
    handler.mode = "url"
    result = _gen(base, tmp_path)
    assert result.success, result.error
    assert (tmp_path / "out.mp4").read_bytes() == _FAKE_MP4


def test_generate_i2v_sends_image_b64(gateway, tmp_path):
    base, handler = gateway
    handler.mode = "bytes"
    img = tmp_path / "ref.png"
    img.write_bytes(b"PNGDATA")
    result = _gen(
        base, tmp_path, operation="image_to_video", reference_image_path=str(img)
    )
    assert result.success, result.error
    assert handler.last_payload["operation"] == "image_to_video"
    assert base64.b64decode(handler.last_payload["image_b64"]) == b"PNGDATA"


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_unknown_variant(gateway, tmp_path):
    base, _ = gateway
    result = generate_remote_video(
        endpoint=base,
        tool_name="wan_video",
        variants=_shared.WAN_VARIANTS,
        default_variant="wan2.1-1.3b",
        inputs={"prompt": "x", "model_variant": "nope"},
    )
    assert not result.success
    assert "Unknown model_variant" in result.error


def test_i2v_unsupported_variant(gateway, tmp_path):
    # cogvideo-2b has i2v=False
    base, _ = gateway
    result = generate_remote_video(
        endpoint=base,
        tool_name="cogvideo_video",
        variants=_shared.COGVIDEO_VARIANTS,
        default_variant="cogvideo-2b",
        inputs={"prompt": "x", "model_variant": "cogvideo-2b", "operation": "image_to_video"},
    )
    assert not result.success
    assert "does not support image_to_video" in result.error


# ---------------------------------------------------------------------------
# end-to-end routing through generate_local_video
# ---------------------------------------------------------------------------


def test_generate_local_video_routes_to_gateway(gateway, tmp_path, monkeypatch):
    base, handler = gateway
    handler.mode = "bytes"
    monkeypatch.setenv("VIDEO_INFER_ENDPOINT", base)
    # Note: torch/diffusers are NOT installed here. If routing failed and fell
    # through to the in-process path, this would ImportError. Success proves the
    # gateway seam short-circuits before any torch import.
    result = _shared.generate_local_video(
        tool_name="wan_video",
        variants=_shared.WAN_VARIANTS,
        default_variant="wan2.1-1.3b",
        inputs={"prompt": "routed", "output_path": str(tmp_path / "r.mp4")},
    )
    assert result.success, result.error
    assert result.data["mode"] == "remote_gateway"
    assert (tmp_path / "r.mp4").read_bytes() == _FAKE_MP4
