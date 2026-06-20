"""Kakeya LLM provider — text generation via a user-run Kakeya inference server.

Kakeya (https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine) is a
memory-bounded local LLM inference server. OpenMontage does NOT bundle, install,
or start it — this tool is a thin client that talks to a server the user runs
themselves and points us at via ``KAKEYA_ENDPOINT``.

Transport (Phase 1): the OpenAI-compatible HTTP shim ``/v1/chat/completions``.
This is the only stable *text-in/text-out* surface Kakeya exposes (the gRPC
``RuntimeService`` is token-id level). The shim is deprecated upstream and is
pure-autoregressive (no speculative decoding), so the speed-bearing features
(CUDA batched scheduler, bounded-memory restored engine) are only realized when
the *server* runs on GPU. On a CPU server you still get the cost/privacy benefit
of local inference, just no throughput win. See ADR 0001 §2 for the full,
honest performance evaluation.

Scope: mechanical, non-creative batch text only (prompt expansion, translation,
caption variants, alt-text). Creative scripting stays with the host agent.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

DEFAULT_ENDPOINT_ENV = "KAKEYA_ENDPOINT"
DEFAULT_MODEL_ENV = "KAKEYA_MODEL"
DEFAULT_API_KEY_ENV = "KAKEYA_API_KEY"
_DEFAULT_MODEL_LABEL = "kakeya-local"
_HEALTH_TIMEOUT_S = 2.0
_GEN_TIMEOUT_S = 300
# Default batch fan-out is SEQUENTIAL (1). Real testing against a live Kakeya
# server (ADR loop iter 4) showed the OpenAI-compatible HTTP shim — the only
# stable text transport today — is single-session and returns HTTP 500 on
# concurrent requests. Kakeya's batched-scheduler throughput (8.45x) lives on the
# gRPC multi-tenant/CUDA path, NOT the shim. So concurrency>1 must only be raised
# when pointed at a concurrency-capable backend; defaulting to 1 keeps the shim
# path correct out of the box.
_DEFAULT_BATCH_CONCURRENCY = 1
_MAX_BATCH_CONCURRENCY = 64


class KakeyaLLM(BaseTool):
    name = "kakeya_llm"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "text_generation"
    provider = "kakeya"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    # HYBRID: the model runs locally on the user's machine, but we reach it
    # over a (usually loopback) network socket. It can also point at a remote
    # self-hosted box.
    runtime = ToolRuntime.HYBRID

    # Availability is gated purely on the endpoint env var being set. We do
    # NOT add torch/grpc/CUDA to OpenMontage's dependency surface — the user
    # runs the server. Reachability is verified at execute() time, not during
    # registry discovery (which must stay fast and offline-safe).
    dependencies = [f"env:{DEFAULT_ENDPOINT_ENV}"]
    install_instructions = (
        "kakeya_llm needs a running Kakeya inference server you host yourself.\n"
        "  1. Clone + start a server (see the Kakeya repo's quickstart):\n"
        "       git clone https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine\n"
        "       # CPU (lightweight): start the OpenAI-compatible HTTP shim on a small model\n"
        "       # GPU (CUDA): start with the batched scheduler for real throughput\n"
        "  2. Point OpenMontage at it in .env:\n"
        "       KAKEYA_ENDPOINT=http://127.0.0.1:8000   # base URL of the HTTP shim\n"
        "       KAKEYA_MODEL=<model-id>                 # optional; defaults to the server's model\n"
        "       KAKEYA_API_KEY=<key>                    # optional; only if the server enforces auth\n"
        "NOTE: speed features (batched throughput, bounded-memory recall) require the\n"
        "server to run on GPU. On CPU you get cost/privacy benefits only. See\n"
        "docs/adr/0001-kakeya-llm-inference-integration.md."
    )
    agent_skills: list[str] = []

    capabilities = [
        "text_generation",
        "batch_text_generation",
        "offline_generation",
        "prompt_expansion",
        "translation",
    ]
    supports = {
        "offline": True,
        "batch": True,
        "streaming": False,  # Phase 1 collects full completions
        "local_cost_free": True,
        # Honest: these are GPU-server-gated, not guaranteed by this tool.
        "high_throughput_batching_requires_gpu_server": True,
        "bounded_memory_long_context_requires_gpu_server": True,
    }
    best_for = [
        "batch image-prompt expansion (scene_plan -> many generation prompts)",
        "subtitle/caption translation at volume (localization-dub)",
        "caption/hook/alt-text variants",
        "private, zero-marginal-cost mechanical text transforms",
    ]
    not_good_for = [
        "creative scripting or narrative decisions (keep with the host agent)",
        "high quality on a CPU-only server with a tiny model",
        "guaranteed speedups without a GPU-backed Kakeya server",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["generate", "batch", "health"],
                "default": "generate",
                "description": (
                    "'generate' = single prompt; 'batch' = list of prompts "
                    "(throughput path); 'health' = probe the server only."
                ),
            },
            "prompt": {
                "type": "string",
                "description": "User prompt for 'generate'.",
            },
            "prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of prompts for 'batch'.",
            },
            "system": {
                "type": "string",
                "description": "Optional system instruction applied to every prompt.",
            },
            "model": {
                "type": "string",
                "description": (
                    "Model id to request. Defaults to $KAKEYA_MODEL, then the "
                    "server's advertised model."
                ),
            },
            "max_tokens": {"type": "integer", "minimum": 1, "default": 512},
            "temperature": {
                "type": "number",
                "minimum": 0,
                "maximum": 2,
                "default": 0.7,
            },
            "endpoint": {
                "type": "string",
                "description": "Override base URL; defaults to $KAKEYA_ENDPOINT.",
            },
            "concurrency": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_BATCH_CONCURRENCY,
                "default": _DEFAULT_BATCH_CONCURRENCY,
                "description": (
                    "Max concurrent in-flight requests for 'batch'. Default 1 "
                    "(sequential) — the Kakeya HTTP shim is single-session and 500s "
                    "on concurrent requests. Raise ONLY when pointed at a "
                    "concurrency-capable backend (gRPC multi-tenant / CUDA), where "
                    "the batched scheduler turns concurrency into throughput."
                ),
            },
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "completion": {"type": "string"},
            "completions": {"type": "array", "items": {"type": "string"}},
            "model": {"type": "string"},
            "usage": {"type": "object"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=10, network_required=True,
    )
    retry_policy = RetryPolicy(
        max_retries=2, backoff_seconds=1.0, retryable_errors=["timeout", "connection"],
    )
    determinism = Determinism.STOCHASTIC
    idempotency_key_fields = ["operation", "prompt", "prompts", "model", "temperature"]
    side_effects = ["sends text to a user-run local inference server"]
    user_visible_verification = [
        "Inspect generated text for usefulness; tiny local models are weaker than the agent",
    ]

    # ---- configuration helpers ----

    def _endpoint(self, inputs: Optional[dict[str, Any]] = None) -> Optional[str]:
        """Resolve the server base URL from inputs or env. Returns None if unset."""
        raw = None
        if inputs:
            raw = inputs.get("endpoint")
        raw = raw or os.environ.get(DEFAULT_ENDPOINT_ENV)
        if not raw:
            return None
        return raw.rstrip("/")

    def _model(self, inputs: dict[str, Any]) -> str:
        return (
            inputs.get("model")
            or os.environ.get(DEFAULT_MODEL_ENV)
            or _DEFAULT_MODEL_LABEL
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(DEFAULT_API_KEY_ENV)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    # ---- status ----

    def get_status(self) -> ToolStatus:
        """Available iff an endpoint is configured.

        We deliberately do NOT perform a network probe here: get_status() is
        called repeatedly during registry discovery / preflight and must be
        fast and offline-safe. Reachability is checked at execute() time.
        """
        return (
            ToolStatus.AVAILABLE
            if self._endpoint()
            else ToolStatus.UNAVAILABLE
        )

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Local inference: no per-token API cost. GPU electricity is not modeled.
        return 0.0

    # ---- execution ----

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        endpoint = self._endpoint(inputs)
        if not endpoint:
            return ToolResult(
                success=False,
                error=(
                    "kakeya_llm unavailable: no endpoint. " + self.install_instructions
                ),
            )

        try:
            import requests  # noqa: F401
        except ImportError:
            return ToolResult(
                success=False,
                error="kakeya_llm requires the 'requests' package (pip install requests).",
            )

        operation = inputs.get("operation", "generate")
        start = time.time()
        try:
            if operation == "health":
                result = self._health(endpoint)
            elif operation == "batch":
                result = self._batch(endpoint, inputs)
            else:
                result = self._generate(endpoint, inputs)
        except Exception as exc:  # noqa: BLE001 - surface any client/server error
            return ToolResult(
                success=False,
                error=f"kakeya_llm {operation} failed: {type(exc).__name__}: {exc}",
                duration_seconds=round(time.time() - start, 3),
            )
        result.duration_seconds = round(time.time() - start, 3)
        return result

    # ---- operations ----

    def _health(self, endpoint: str) -> ToolResult:
        import requests

        url = urllib.parse.urljoin(endpoint + "/", "healthz")
        resp = requests.get(url, headers=self._headers(), timeout=_HEALTH_TIMEOUT_S)
        resp.raise_for_status()
        body: dict[str, Any] = {}
        try:
            body = resp.json()
        except (ValueError, json.JSONDecodeError):
            body = {"raw": resp.text[:500]}
        return ToolResult(
            success=True,
            data={"healthy": True, "endpoint": endpoint, "server": body},
        )

    def _build_messages(
        self, prompt: str, system: Optional[str],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _chat_once(
        self, endpoint: str, prompt: str, inputs: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """One /v1/chat/completions round-trip. Returns (text, usage)."""
        import requests

        url = urllib.parse.urljoin(endpoint + "/", "v1/chat/completions")
        payload = {
            "model": self._model(inputs),
            "messages": self._build_messages(prompt, inputs.get("system")),
            "max_tokens": int(inputs.get("max_tokens", 512)),
            "temperature": float(inputs.get("temperature", 0.7)),
            "stream": False,
        }
        resp = requests.post(
            url, headers=self._headers(), json=payload, timeout=_GEN_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._parse_completion(data)

    @staticmethod
    def _parse_completion(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract assistant text + usage from an OpenAI-shaped response."""
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(
                f"server returned no choices: {json.dumps(data)[:300]}"
            )
        message = choices[0].get("message") or {}
        text = message.get("content")
        if text is None:
            # Some servers use the legacy `text` field.
            text = choices[0].get("text")
        if text is None:
            raise ValueError(
                f"could not find completion text in response: {json.dumps(data)[:300]}"
            )
        usage = data.get("usage") or {}
        return text, usage

    @staticmethod
    def _resolve_concurrency(inputs: dict[str, Any], n_prompts: int) -> int:
        """Clamp the requested batch concurrency to a sane range.

        Never more than the number of prompts (no point), never more than the
        hard cap, never less than 1.
        """
        try:
            requested = int(inputs.get("concurrency", _DEFAULT_BATCH_CONCURRENCY))
        except (TypeError, ValueError):
            requested = _DEFAULT_BATCH_CONCURRENCY
        requested = max(1, min(requested, _MAX_BATCH_CONCURRENCY))
        return max(1, min(requested, n_prompts))

    def _generate(self, endpoint: str, inputs: dict[str, Any]) -> ToolResult:
        prompt = inputs.get("prompt")
        if not prompt or not str(prompt).strip():
            return ToolResult(
                success=False, error="kakeya_llm 'generate' requires a non-empty 'prompt'.",
            )
        text, usage = self._chat_once(endpoint, str(prompt), inputs)
        return ToolResult(
            success=True,
            data={
                "completion": text,
                "model": self._model(inputs),
                "usage": usage,
                "endpoint": endpoint,
                "operation": "generate",
            },
            model=self._model(inputs),
        )

    def _batch(self, endpoint: str, inputs: dict[str, Any]) -> ToolResult:
        prompts = inputs.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            return ToolResult(
                success=False,
                error="kakeya_llm 'batch' requires a non-empty 'prompts' list.",
            )
        if not all(isinstance(p, str) and p.strip() for p in prompts):
            return ToolResult(
                success=False,
                error="kakeya_llm 'batch' prompts must all be non-empty strings.",
            )

        # Fan the prompts out with `concurrency` workers (default 1). Against a
        # concurrency-capable backend (gRPC multi-tenant / CUDA) raising this
        # lets the SERVER's batched scheduler turn concurrency into throughput
        # (W2). Against the single-session HTTP shim, keep it at 1 (concurrent
        # requests there 500). Order is preserved by index; per-item errors are
        # isolated so one failure never aborts the batch.
        concurrency = self._resolve_concurrency(inputs, len(prompts))
        completions: list[Optional[str]] = [None] * len(prompts)
        errors: list[Optional[str]] = [None] * len(prompts)
        usages: list[dict[str, Any]] = [{} for _ in prompts]

        def _run_one(index: int, prompt: str) -> None:
            try:
                text, usage = self._chat_once(endpoint, prompt, inputs)
                completions[index] = text
                usages[index] = usage
            except Exception as exc:  # noqa: BLE001 - per-item isolation
                errors[index] = f"{type(exc).__name__}: {exc}"

        if concurrency == 1:
            for i, prompt in enumerate(prompts):
                _run_one(i, prompt)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(_run_one, i, prompt)
                    for i, prompt in enumerate(prompts)
                ]
                for future in futures:
                    future.result()  # _run_one swallows; this just joins

        total_usage: dict[str, int] = {}
        for usage in usages:
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    total_usage[key] = int(total_usage.get(key, 0) + value)

        succeeded = sum(1 for c in completions if c is not None)
        return ToolResult(
            success=succeeded > 0,
            data={
                "completions": completions,
                "errors": errors,
                "succeeded": succeeded,
                "failed": len(prompts) - succeeded,
                "model": self._model(inputs),
                "usage": total_usage,
                "endpoint": endpoint,
                "operation": "batch",
                "concurrency": concurrency,
            },
            error=(
                None
                if succeeded == len(prompts)
                else f"{len(prompts) - succeeded}/{len(prompts)} prompts failed"
            ),
            model=self._model(inputs),
        )
