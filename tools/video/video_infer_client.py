"""Client for an external warm video-inference gateway.

This is the seam that lets OpenMontage's local video tools (wan/hunyuan/
cogvideo/ltx) route generation to ONE standalone, warm inference server that
hosts the open-source models — instead of cold-loading a diffusers pipeline
in-process on every call. See docs/adr/0002-unified-local-video-inference-backend.md.

Engine-agnostic by design: the gateway can be a FastAPI+diffusers service, a
generalized Modal endpoint, a ComfyUI server, etc. OpenMontage depends only on
the HTTP contract documented in ADR 0002 §5, never on a specific server.

IMPORTANT: this gateway serves DiT *video diffusion* models. It is NOT Kakeya
(an LLM token engine, which cannot host video diffusion — ADR 0002 §2). Kakeya
serves the separate `text_generation` capability (ADR 0001).
"""

from __future__ import annotations

import base64
import os
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from tools.base_tool import ToolResult

VIDEO_INFER_ENDPOINT_ENV = "VIDEO_INFER_ENDPOINT"
VIDEO_INFER_API_KEY_ENV = "VIDEO_INFER_API_KEY"
_HEALTH_TIMEOUT_S = 3.0
_GEN_TIMEOUT_S = 900  # video diffusion is slow; generous ceiling


def video_infer_endpoint(inputs: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Resolve the gateway base URL from inputs or env. None if unconfigured."""
    raw = None
    if inputs:
        raw = inputs.get("video_infer_endpoint")
    raw = raw or os.environ.get(VIDEO_INFER_ENDPOINT_ENV)
    if not raw:
        return None
    return raw.rstrip("/")


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(VIDEO_INFER_API_KEY_ENV)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def remote_health(endpoint: str) -> dict[str, Any]:
    """Probe the gateway. Returns the parsed /healthz body."""
    import requests

    url = urllib.parse.urljoin(endpoint + "/", "healthz")
    resp = requests.get(url, headers=_headers(), timeout=_HEALTH_TIMEOUT_S)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text[:500]}


def _resolve_model_id(meta: dict[str, Any], operation: str) -> str:
    """Pick the HF model id to request, honoring i2v variants when present."""
    if operation == "image_to_video" and meta.get("hf_i2v_id"):
        return meta["hf_i2v_id"]
    return meta["hf_id"]


def _build_payload(
    meta: dict[str, Any], inputs: dict[str, Any],
) -> dict[str, Any]:
    operation = inputs.get("operation", "text_to_video")
    width = inputs.get("width", meta["default_width"])
    height = inputs.get("height", meta["default_height"])
    num_frames = inputs.get("num_frames", meta["default_num_frames"])
    payload: dict[str, Any] = {
        "model": _resolve_model_id(meta, operation),
        "prompt": inputs["prompt"],
        "operation": operation,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": inputs.get("num_inference_steps", 30),
        "fps": meta["fps"],
    }
    if inputs.get("seed") is not None:
        payload["seed"] = inputs["seed"]
    if meta["pipeline_class"] == "CogVideoXPipeline":
        payload["negative_prompt"] = (
            "worst quality, low quality, blurry, distorted, watermark"
        )
    if operation == "image_to_video":
        ref_path = inputs.get("reference_image_path")
        ref_url = inputs.get("reference_image_url")
        if ref_path:
            payload["image_b64"] = base64.b64encode(
                Path(ref_path).read_bytes()
            ).decode()
        elif ref_url:
            payload["image_url"] = ref_url
        # No image is a server-side error; let the gateway reject it so the
        # contract stays in one place. (The in-process path validates locally.)
    return payload


def _write_response_video(resp, output_path: Path) -> Optional[str]:
    """Persist the gateway response to output_path. Returns error string or None."""
    import requests

    content_type = resp.headers.get("content-type", "")
    if "video" in content_type or "octet-stream" in content_type:
        output_path.write_bytes(resp.content)
        return None

    try:
        body = resp.json()
    except ValueError:
        return f"unexpected non-JSON, non-video response (content-type={content_type!r})"

    if body.get("video_b64"):
        output_path.write_bytes(base64.b64decode(body["video_b64"]))
        return None

    video_url = body.get("video_url") or body.get("url")
    if video_url:
        dl = requests.get(video_url, timeout=300)
        dl.raise_for_status()
        output_path.write_bytes(dl.content)
        return None

    return f"no video bytes/url/b64 in gateway response: {body}"


def generate_remote_video(
    *,
    endpoint: str,
    tool_name: str,
    variants: dict[str, dict[str, Any]],
    default_variant: str,
    inputs: dict[str, Any],
) -> ToolResult:
    """Route a generation request to the warm video-inference gateway.

    Mirrors the success-shape of `_shared.generate_local_video` so callers and
    artifacts stay identical regardless of execution backend.
    """
    import requests

    from tools.video._shared import probe_output

    variant = inputs.get("model_variant", default_variant)
    if variant not in variants:
        return ToolResult(
            success=False,
            error=f"Unknown model_variant: {variant}. Available: {', '.join(sorted(variants))}",
        )
    meta = variants[variant]

    if inputs.get("operation") == "image_to_video" and not meta.get("i2v"):
        return ToolResult(
            success=False, error=f"{meta['name']} does not support image_to_video.",
        )
    if not inputs.get("prompt"):
        return ToolResult(success=False, error="prompt is required.")

    payload = _build_payload(meta, inputs)
    url = urllib.parse.urljoin(endpoint + "/", "v1/video/generations")
    resp = requests.post(
        url, headers=_headers(), json=payload, timeout=_GEN_TIMEOUT_S,
    )
    resp.raise_for_status()

    output_path = Path(inputs.get("output_path", f"{tool_name}_{variant}.mp4"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    err = _write_response_video(resp, output_path)
    if err:
        return ToolResult(success=False, error=f"video gateway: {err}")

    num_frames = payload["num_frames"]
    fps = payload["fps"]
    return ToolResult(
        success=True,
        data={
            "provider": tool_name,
            "model_variant": variant,
            "provider_name": meta["name"],
            "mode": "remote_gateway",
            "endpoint": endpoint,
            "prompt": inputs["prompt"],
            "model_id": payload["model"],
            "width": payload["width"],
            "height": payload["height"],
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": round(num_frames / fps, 2),
            "operation": payload["operation"],
            "output": str(output_path),
            "format": "mp4",
            "license": meta["license"],
            **probe_output(output_path),
        },
        artifacts=[str(output_path)],
        seed=inputs.get("seed"),
        model=payload["model"],
    )
