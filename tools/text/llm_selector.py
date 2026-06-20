"""Capability-level text-generation selector.

Routes mechanical batch-text requests to whatever `text_generation` provider
is available, exactly like `tts_selector` does for TTS. Provider discovery is
automatic: any BaseTool with capability="text_generation" is picked up from the
registry, so adding a new local/cloud LLM provider needs no change here.

IMPORTANT (see docs/adr/0001-kakeya-llm-inference-integration.md): this family is
for OFFLOADING non-creative text chores (prompt expansion, translation, caption
variants). The host agent stays the creative intelligence. The selector exists so
pipelines can opportunistically use a local LLM when one is configured and fall
back gracefully (to a cloud provider, or to the agent) when none is.
"""

from __future__ import annotations

from typing import Any

from tools.base_tool import (
    BaseTool,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class LLMSelector(BaseTool):
    name = "llm_selector"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "text_generation"
    provider = "selector"
    stability = ToolStability.EXPERIMENTAL
    runtime = ToolRuntime.HYBRID
    agent_skills: list[str] = []

    capabilities = [
        "text_generation",
        "provider_selection",
    ]
    supports = {
        "user_preference_routing": True,
        "offline_fallback": True,
        "batch": True,
    }
    best_for = [
        "routing batch text chores to an available local LLM",
        "preflight text-generation provider selection",
    ]
    not_good_for = [
        "creative scripting (keep with the host agent)",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["generate", "batch", "health", "rank"],
                "default": "generate",
                "description": (
                    "'rank' returns scored provider rankings without generating; "
                    "everything else is forwarded to the selected provider."
                ),
            },
            "prompt": {"type": "string"},
            "prompts": {"type": "array", "items": {"type": "string"}},
            "system": {"type": "string"},
            "model": {"type": "string"},
            "max_tokens": {"type": "integer"},
            "temperature": {"type": "number"},
            "preferred_provider": {
                "type": "string",
                "default": "auto",
                "description": "Provider name or 'auto'. Valid values discovered at runtime.",
            },
            "allowed_providers": {"type": "array", "items": {"type": "string"}},
        },
    }

    def _providers(self) -> list[BaseTool]:
        """Auto-discover text_generation providers (excluding self)."""
        from tools.tool_registry import registry

        registry.ensure_discovered()
        return [
            t
            for t in registry.get_by_capability("text_generation")
            if t.name != self.name
        ]

    @property
    def fallback_tools(self) -> list[str]:
        return [t.name for t in self._providers()]

    @property
    def provider_matrix(self) -> dict[str, dict[str, str]]:
        matrix: dict[str, dict[str, str]] = {}
        for tool in self._providers():
            strength = ", ".join(tool.best_for) if tool.best_for else tool.name
            matrix[tool.provider] = {"tool": tool.name, "strength": strength}
        return matrix

    def get_status(self) -> ToolStatus:
        if any(t.get_status() == ToolStatus.AVAILABLE for t in self._providers()):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        tool = self._select(inputs)
        return tool.estimate_cost(inputs) if tool else 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        candidates = self._providers()
        if not candidates:
            return ToolResult(
                success=False,
                error=(
                    "No text_generation provider configured. Set KAKEYA_ENDPOINT to "
                    "use a local Kakeya server, or rely on the host agent for text."
                ),
            )

        if inputs.get("operation") == "rank":
            return self._rank(inputs, candidates)

        tool = self._select(inputs, candidates)
        if tool is None:
            # Providers are registered but none are currently available (e.g.
            # kakeya_llm exists but KAKEYA_ENDPOINT is unset, or allowed_providers
            # filtered everything out). Give the actionable setup hint rather
            # than a dead-end "no match".
            return ToolResult(
                success=False,
                error=(
                    "No text_generation provider is currently available. Set "
                    "KAKEYA_ENDPOINT to a running Kakeya server, or rely on the "
                    "host agent for text. Registered providers: "
                    + ", ".join(sorted(t.name for t in candidates))
                ),
            )

        result = tool.execute(inputs)
        if result.success:
            result.data.setdefault("selected_tool", tool.name)
            result.data["selected_provider"] = tool.provider
            result.data["alternatives_considered"] = [
                t.name
                for t in candidates
                if t.name != tool.name and t.get_status() == ToolStatus.AVAILABLE
            ]
        return result

    def _select(
        self,
        inputs: dict[str, Any],
        candidates: list[BaseTool] | None = None,
    ) -> BaseTool | None:
        if candidates is None:
            candidates = self._providers()

        allowed = set(inputs.get("allowed_providers") or [])
        if allowed:
            candidates = [t for t in candidates if t.provider in allowed]

        available = [t for t in candidates if t.get_status() == ToolStatus.AVAILABLE]
        if not available:
            return None

        preferred = inputs.get("preferred_provider", "auto")
        if preferred and preferred != "auto":
            for tool in available:
                if tool.provider == preferred:
                    return tool

        # Deterministic: first available by discovery order.
        return available[0]

    def _rank(
        self, inputs: dict[str, Any], candidates: list[BaseTool],
    ) -> ToolResult:
        from lib.scoring import normalize_task_context, rank_providers

        task_context = normalize_task_context(
            inputs.get("task_context", {}),
            prompt=inputs.get("prompt", "") or " ".join(inputs.get("prompts", []) or []),
            capability=self.capability,
            operation=inputs.get("operation", "generate"),
        )
        rankings = rank_providers(candidates, task_context)
        return ToolResult(
            success=True,
            data={
                "rankings": [r.to_dict() for r in rankings],
                "explanation": "\n".join(r.explain() for r in rankings[:5]),
            },
        )
