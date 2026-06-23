"""LLM client for the OpenMontage agent runtime.

The OpenMontage "agent" is normally the host LLM (Cursor/Claude). To run the pipeline
UNATTENDED behind the gateway's mode=agent, we need our own cloud-LLM reasoner. This is a
thin, dependency-free client (urllib only) over the providers' HTTP APIs, selected by env:

  AGENT_LLM = anthropic | openai | kakeya | stub   (auto-detected from keys if unset)
  ANTHROPIC_API_KEY / OPENAI_API_KEY / KAKEYA_ENDPOINT
  AGENT_LLM_MODEL  (optional override)

Creative reasoning (brief/script/scene_plan/edit) runs here; mechanical text can still use
the in-repo kakeya_llm tool. `complete_json` returns a parsed dict, retrying once on bad JSON.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Callable, Optional

_DEFAULT_MODELS = {
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o-mini",
    "kakeya": "kakeya-local",
}


class LLM:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None,
                 stub: Optional[Callable[[str, str], str]] = None):
        self.stub = stub
        self.provider = (provider or os.environ.get("AGENT_LLM", "")).strip().lower()
        if not self.provider:
            if stub is not None:
                self.provider = "stub"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
            elif os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.environ.get("KAKEYA_ENDPOINT"):
                self.provider = "kakeya"
            else:
                self.provider = "stub"
        self.model = model or os.environ.get("AGENT_LLM_MODEL", "") or _DEFAULT_MODELS.get(self.provider, "")

    # --- transports ---------------------------------------------------------
    def complete(self, system: str, user: str, max_tokens: int = 3000) -> str:
        if self.provider == "stub":
            if self.stub is None:
                raise RuntimeError("AGENT_LLM=stub but no stub callable provided")
            return self.stub(system, user)
        if self.provider == "anthropic":
            return self._anthropic(system, user, max_tokens)
        if self.provider == "openai":
            return self._openai(system, user, max_tokens)
        if self.provider == "kakeya":
            return self._openai(system, user, max_tokens, kakeya=True)
        raise RuntimeError(f"unknown AGENT_LLM provider: {self.provider}")

    def _post(self, url: str, headers: dict, body: dict, timeout: int = 180) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def _anthropic(self, system: str, user: str, max_tokens: int) -> str:
        out = self._post(
            "https://api.anthropic.com/v1/messages",
            {"Content-Type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
             "anthropic-version": "2023-06-01"},
            {"model": self.model, "max_tokens": max_tokens, "system": system,
             "messages": [{"role": "user", "content": user}]})
        return "".join(b.get("text", "") for b in out.get("content", []))

    def _openai(self, system: str, user: str, max_tokens: int, kakeya: bool = False) -> str:
        if kakeya:
            base = os.environ["KAKEYA_ENDPOINT"].rstrip("/")
            headers = {"Content-Type": "application/json"}
        else:
            base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        out = self._post(base + "/v1/chat/completions", headers,
                         {"model": self.model, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}]})
        return out["choices"][0]["message"]["content"]

    # --- JSON helper --------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> dict:
        # strip ```json fences, then take the outermost {...}
        text = re.sub(r"```(?:json)?", "", text).strip()
        start, depth = text.find("{"), 0
        if start < 0:
            raise ValueError("no JSON object in LLM output")
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
        raise ValueError("unbalanced JSON in LLM output")

    def complete_json(self, system: str, user: str, retries: int = 1) -> dict:
        last = None
        for attempt in range(retries + 1):
            txt = self.complete(system, user + ("\n\nReturn ONLY valid JSON." if attempt == 0
                                                else f"\n\nYour previous output failed to parse ({last}). "
                                                     "Return ONLY a single valid JSON object, no prose."))
            try:
                return self._extract_json(txt)
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise RuntimeError(f"LLM did not return valid JSON after {retries + 1} attempts: {last}")
