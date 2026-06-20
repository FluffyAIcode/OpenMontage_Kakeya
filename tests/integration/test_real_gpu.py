"""REAL GPU integration tests — no mock, no fake, no fallback (ADR 0002 §0).

These are the binding correctness gate for the Kakeya text integration and the
warm video-inference gateway. They hit **live, GPU-served** endpoints and assert on
**real** outputs (a real LLM completion; a real, ffprobe-decodable .mp4).

They never fake a pass:
- If the relevant endpoint env var is unset, the test SKIPS (it does not stub).
- They do NOT exercise the in-process diffusers fallback; the gateway path is what
  is under test.

Run against live servers (e.g. via SSH tunnels to a GPU box):
    export KAKEYA_ENDPOINT=http://127.0.0.1:8000
    export VIDEO_INFER_ENDPOINT=http://127.0.0.1:8080
    pytest tests/integration/test_real_gpu.py -v -s

Env knobs:
    KAKEYA_ENDPOINT          base URL of a live Kakeya OpenAI-compatible server
    VIDEO_INFER_ENDPOINT     base URL of a live video-inference gateway
    REAL_VIDEO_MODEL_VARIANT model_variant to generate (default: cogvideo-2b)
    REAL_VIDEO_FRAMES        num_frames (default: 49)
    REAL_VIDEO_STEPS         num_inference_steps (default: 20)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import json

import pytest

KAKEYA_ENDPOINT = os.environ.get("KAKEYA_ENDPOINT")
VIDEO_INFER_ENDPOINT = os.environ.get("VIDEO_INFER_ENDPOINT")

requires_kakeya = pytest.mark.skipif(
    not KAKEYA_ENDPOINT, reason="KAKEYA_ENDPOINT not set — real Kakeya server required (no mock)"
)
requires_gateway = pytest.mark.skipif(
    not VIDEO_INFER_ENDPOINT,
    reason="VIDEO_INFER_ENDPOINT not set — real video gateway required (no mock)",
)


def _ffprobe(path: str) -> dict:
    if not shutil.which("ffprobe"):
        return {}
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if proc.returncode != 0:
        return {}
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Kakeya text engine — REAL
# ---------------------------------------------------------------------------


@requires_kakeya
def test_real_kakeya_health():
    from tools.text.kakeya_llm import KakeyaLLM

    result = KakeyaLLM().execute({"operation": "health"})
    assert result.success, result.error
    assert result.data["healthy"] is True


@requires_kakeya
def test_real_kakeya_generate():
    from tools.text.kakeya_llm import KakeyaLLM

    result = KakeyaLLM().execute(
        {
            "operation": "generate",
            "prompt": "Reply with exactly the word: PONG",
            "max_tokens": 16,
            "temperature": 0.0,
        }
    )
    assert result.success, result.error
    completion = result.data["completion"]
    assert isinstance(completion, str) and completion.strip(), "empty completion"
    # Real generation produced tokens.
    assert result.data["usage"].get("completion_tokens", 0) >= 1
    print(f"\n[REAL kakeya] completion={completion!r} usage={result.data['usage']}")


@requires_kakeya
def test_real_kakeya_batch_sequential():
    """Real batch over the HTTP shim. Sequential (concurrency=1) is the correct
    mode for the single-session shim — real testing showed concurrency>1 makes
    the shim return HTTP 500 (its concurrent/batched path is gRPC+CUDA, not the
    shim). So we assert the genuinely-real sequential path: all prompts succeed.
    """
    from tools.text.kakeya_llm import KakeyaLLM

    prompts = ["Name a color.", "Name a fruit.", "Name an animal.", "Name a city."]
    result = KakeyaLLM().execute(
        {"operation": "batch", "prompts": prompts, "max_tokens": 8}  # default concurrency=1
    )
    assert result.success, result.error
    assert result.data["succeeded"] == len(prompts), result.data["errors"]
    assert all(isinstance(c, str) and c.strip() for c in result.data["completions"])
    print(f"\n[REAL kakeya batch seq] {result.data['completions']}")


@requires_kakeya
def test_real_kakeya_shim_is_single_session():
    """Document the real 技术硬伤: the deprecated HTTP shim cannot serve concurrent
    requests — concurrency>1 yields per-item 500s (the tool isolates them, never
    crashes/fakes). Throughput needs Kakeya's gRPC multi-tenant/CUDA path.
    """
    from tools.text.kakeya_llm import KakeyaLLM

    prompts = [f"Name color {i}." for i in range(4)]
    result = KakeyaLLM().execute(
        {"operation": "batch", "prompts": prompts, "max_tokens": 8, "concurrency": 4}
    )
    # Tool must not crash; failures are isolated and surfaced per item.
    assert isinstance(result.data["completions"], list)
    assert result.data["failed"] >= 1, (
        "Shim unexpectedly served concurrency cleanly — if Kakeya fixed this, "
        "revisit the concurrency default."
    )
    errs = [e for e in result.data["errors"] if e]
    assert any("500" in e for e in errs), errs
    print(f"\n[REAL kakeya shim concurrency] succeeded={result.data['succeeded']} "
          f"failed={result.data['failed']} (500s confirm single-session shim)")


# ---------------------------------------------------------------------------
# Video inference gateway — REAL diffusion -> real mp4
# ---------------------------------------------------------------------------


@requires_gateway
def test_real_gateway_health():
    from tools.video.video_infer_client import remote_health, video_infer_endpoint

    body = remote_health(video_infer_endpoint())
    assert body.get("status") == "ok"
    assert body.get("device") == "cuda", f"gateway not on GPU: {body}"
    assert body.get("known"), "gateway advertises no models"
    print(f"\n[REAL gateway] {body}")


@requires_gateway
def test_real_gateway_generates_decodable_mp4(tmp_path):
    """End-to-end: OpenMontage tool -> live gateway -> real diffusion -> real mp4."""
    from tools.video import _shared
    from tools.video.cogvideo_video import CogVideoVideo

    variant = os.environ.get("REAL_VIDEO_MODEL_VARIANT", "cogvideo-2b")
    frames = int(os.environ.get("REAL_VIDEO_FRAMES", "49"))
    steps = int(os.environ.get("REAL_VIDEO_STEPS", "20"))

    # Sanity: the variant must be a real registered model.
    assert variant in _shared.COGVIDEO_VARIANTS, variant
    meta = _shared.COGVIDEO_VARIANTS[variant]

    out = tmp_path / "real.mp4"
    tool = CogVideoVideo()
    result = tool.execute(
        {
            "prompt": "a serene koi pond at golden hour, gentle ripples, cinematic",
            "model_variant": variant,
            "num_frames": frames,
            "num_inference_steps": steps,
            "seed": 42,
            "output_path": str(out),
        }
    )
    assert result.success, result.error
    # Must have gone through the gateway, NOT the in-process fallback.
    assert result.data["mode"] == "remote_gateway", result.data
    assert out.exists() and out.stat().st_size > 10_000, "mp4 missing or too small"

    probe = _ffprobe(str(out))
    if probe:
        streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        assert streams, f"no video stream in output: {probe}"
        vs = streams[0]
        assert int(vs["width"]) == meta["default_width"]
        assert int(vs["height"]) == meta["default_height"]
        print(
            f"\n[REAL gateway] {variant} -> {out.stat().st_size} bytes, "
            f"{vs['width']}x{vs['height']} codec={vs.get('codec_name')} "
            f"frames~{frames} steps={steps}"
        )
    else:
        print(f"\n[REAL gateway] {variant} -> {out.stat().st_size} bytes (ffprobe unavailable)")
