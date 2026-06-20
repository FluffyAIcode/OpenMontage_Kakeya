"""OFFLINE SMOKE tests for the Kakeya text-generation integration.

NOT THE CORRECTNESS GATE. Per ADR 0002 §0 (no fallback/mock/fake/simplify), the
binding correctness gate is `tests/integration/test_real_gpu.py`, which runs
against a REAL, GPU-served Kakeya server. The stdlib mock server below exists only
as fast offline coverage of pure-function logic and the HTTP round-trip shape; it
is explicitly NOT evidence that the integration works end-to-end.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tools.base_tool import ToolStatus
from tools.text.kakeya_llm import KakeyaLLM
from tools.text.llm_selector import LLMSelector


# ---------------------------------------------------------------------------
# Mock Kakeya OpenAI-compatible server
# ---------------------------------------------------------------------------


class _MockHandler(BaseHTTPRequestHandler):
    # Set by the fixture: maps prompt content -> behavior.
    fail_on: set = set()

    def log_message(self, *args):  # silence test output
        pass

    def _send_json(self, status: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok", "model": "mock-qwen"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        user_msg = ""
        for m in req.get("messages", []):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
        if user_msg in type(self).fail_on:
            self._send_json(500, {"error": {"message": "boom"}})
            return
        self._send_json(
            200,
            {
                "id": "chatcmpl-test",
                "model": req.get("model", "mock-qwen"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"echo:{user_msg}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_msg.split()),
                    "completion_tokens": 3,
                    "total_tokens": len(user_msg.split()) + 3,
                },
            },
        )


@pytest.fixture
def mock_server():
    _MockHandler.fail_on = set()
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", _MockHandler
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Availability gating
# ---------------------------------------------------------------------------


def test_unavailable_without_endpoint(monkeypatch):
    monkeypatch.delenv("KAKEYA_ENDPOINT", raising=False)
    tool = KakeyaLLM()
    assert tool.get_status() == ToolStatus.UNAVAILABLE
    result = tool.execute({"operation": "generate", "prompt": "hi"})
    assert not result.success
    assert "no endpoint" in result.error.lower()


def test_available_with_endpoint(monkeypatch):
    monkeypatch.setenv("KAKEYA_ENDPOINT", "http://127.0.0.1:1")
    assert KakeyaLLM().get_status() == ToolStatus.AVAILABLE


def test_get_status_does_not_probe_network(monkeypatch):
    # Pointing at a dead address must still report AVAILABLE instantly
    # (status is env-only; reachability is an execute()-time concern).
    monkeypatch.setenv("KAKEYA_ENDPOINT", "http://10.255.255.1:9")
    assert KakeyaLLM().get_status() == ToolStatus.AVAILABLE


# ---------------------------------------------------------------------------
# generate / health / batch against the mock server
# ---------------------------------------------------------------------------


def test_generate_roundtrip(mock_server, monkeypatch):
    endpoint, _ = mock_server
    monkeypatch.setenv("KAKEYA_ENDPOINT", endpoint)
    result = KakeyaLLM().execute({"operation": "generate", "prompt": "hello"})
    assert result.success, result.error
    assert result.data["completion"] == "echo:hello"
    assert result.data["usage"]["completion_tokens"] == 3
    assert result.cost_usd == 0.0


def test_health(mock_server, monkeypatch):
    endpoint, _ = mock_server
    monkeypatch.setenv("KAKEYA_ENDPOINT", endpoint)
    result = KakeyaLLM().execute({"operation": "health"})
    assert result.success
    assert result.data["healthy"] is True
    assert result.data["server"]["status"] == "ok"


def test_batch_all_succeed(mock_server, monkeypatch):
    endpoint, _ = mock_server
    monkeypatch.setenv("KAKEYA_ENDPOINT", endpoint)
    prompts = ["a", "b", "c"]
    result = KakeyaLLM().execute({"operation": "batch", "prompts": prompts})
    assert result.success
    assert result.data["completions"] == ["echo:a", "echo:b", "echo:c"]
    assert result.data["succeeded"] == 3
    assert result.data["failed"] == 0
    # usage is aggregated across the batch
    assert result.data["usage"]["completion_tokens"] == 9


def test_batch_partial_failure_is_isolated(mock_server, monkeypatch):
    endpoint, handler = mock_server
    handler.fail_on = {"b"}
    monkeypatch.setenv("KAKEYA_ENDPOINT", endpoint)
    result = KakeyaLLM().execute({"operation": "batch", "prompts": ["a", "b", "c"]})
    # success is True because at least one succeeded; failures are surfaced per-item
    assert result.success
    assert result.data["completions"][0] == "echo:a"
    assert result.data["completions"][1] is None
    assert result.data["completions"][2] == "echo:c"
    assert result.data["succeeded"] == 2
    assert result.data["failed"] == 1
    assert result.data["errors"][1] is not None
    assert "1/3 prompts failed" in result.error


def test_batch_preserves_order_under_concurrency(mock_server, monkeypatch):
    endpoint, _ = mock_server
    monkeypatch.setenv("KAKEYA_ENDPOINT", endpoint)
    prompts = [str(i) for i in range(20)]
    result = KakeyaLLM().execute(
        {"operation": "batch", "prompts": prompts, "concurrency": 8}
    )
    assert result.success
    assert result.data["completions"] == [f"echo:{i}" for i in range(20)]
    assert result.data["concurrency"] == 8


def test_batch_concurrency_clamped_to_prompt_count():
    # Requesting more workers than prompts is pointless; clamp to len(prompts).
    assert KakeyaLLM._resolve_concurrency({"concurrency": 50}, 3) == 3
    # Below 1 floors to 1.
    assert KakeyaLLM._resolve_concurrency({"concurrency": 0}, 5) == 1
    # Above the hard cap is capped.
    assert KakeyaLLM._resolve_concurrency({"concurrency": 9999}, 1000) == 64
    # Garbage falls back to the default (sequential — safe for the HTTP shim).
    assert KakeyaLLM._resolve_concurrency({"concurrency": "x"}, 100) == 1


def test_batch_actually_runs_concurrently(monkeypatch):
    # A slow per-request stub: if requests ran sequentially, 8 x 0.1s = 0.8s.
    # Concurrent fan-out should finish well under that.
    import tools.text.kakeya_llm as mod

    calls = []

    def slow_chat_once(self, endpoint, prompt, inputs):
        calls.append(prompt)
        time.sleep(0.1)
        return f"r:{prompt}", {}

    monkeypatch.setenv("KAKEYA_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setattr(mod.KakeyaLLM, "_chat_once", slow_chat_once)

    start = time.time()
    result = KakeyaLLM().execute(
        {"operation": "batch", "prompts": [str(i) for i in range(8)], "concurrency": 8}
    )
    elapsed = time.time() - start
    assert result.success
    assert result.data["succeeded"] == 8
    # Sequential would be ~0.8s; concurrent should be ~0.1-0.3s. Generous bound.
    assert elapsed < 0.5, f"batch did not run concurrently (took {elapsed:.2f}s)"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_generate_requires_prompt(mock_server, monkeypatch):
    endpoint, _ = mock_server
    monkeypatch.setenv("KAKEYA_ENDPOINT", endpoint)
    result = KakeyaLLM().execute({"operation": "generate", "prompt": "  "})
    assert not result.success
    assert "non-empty" in result.error


def test_batch_requires_list(mock_server, monkeypatch):
    endpoint, _ = mock_server
    monkeypatch.setenv("KAKEYA_ENDPOINT", endpoint)
    result = KakeyaLLM().execute({"operation": "batch", "prompts": []})
    assert not result.success
    assert "non-empty" in result.error


# ---------------------------------------------------------------------------
# Response parsing edge cases (no network)
# ---------------------------------------------------------------------------


def test_parse_completion_message_content():
    text, usage = KakeyaLLM._parse_completion(
        {"choices": [{"message": {"content": "hi"}}], "usage": {"x": 1}}
    )
    assert text == "hi"
    assert usage == {"x": 1}


def test_parse_completion_legacy_text_field():
    text, _ = KakeyaLLM._parse_completion({"choices": [{"text": "legacy"}]})
    assert text == "legacy"


def test_parse_completion_no_choices_raises():
    with pytest.raises(ValueError):
        KakeyaLLM._parse_completion({"choices": []})


# ---------------------------------------------------------------------------
# Selector routing
# ---------------------------------------------------------------------------


def test_selector_unavailable_without_provider(monkeypatch):
    monkeypatch.delenv("KAKEYA_ENDPOINT", raising=False)
    result = LLMSelector().execute({"operation": "generate", "prompt": "hi"})
    assert not result.success
    assert "no text_generation provider" in result.error.lower()


def test_selector_routes_to_kakeya(mock_server, monkeypatch):
    endpoint, _ = mock_server
    monkeypatch.setenv("KAKEYA_ENDPOINT", endpoint)
    result = LLMSelector().execute({"operation": "generate", "prompt": "routed"})
    assert result.success, result.error
    assert result.data["completion"] == "echo:routed"
    assert result.data["selected_provider"] == "kakeya"


def test_selector_status_follows_providers(monkeypatch):
    monkeypatch.delenv("KAKEYA_ENDPOINT", raising=False)
    assert LLMSelector().get_status() == ToolStatus.UNAVAILABLE
    monkeypatch.setenv("KAKEYA_ENDPOINT", "http://127.0.0.1:1")
    assert LLMSelector().get_status() == ToolStatus.AVAILABLE
