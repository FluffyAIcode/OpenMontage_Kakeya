"""LLM client for the OpenMontage agent runtime.

The OpenMontage "agent" is normally the host LLM (Cursor/Claude). To run the pipeline
UNATTENDED behind the gateway's mode=agent, we need our own cloud-LLM reasoner. This is a
thin, dependency-free client (urllib only) over the providers' HTTP APIs, selected by env:

  AGENT_LLM = anthropic | openai | kakeya | kakeya_grpc | stub  (auto-detected from keys if unset)
  ANTHROPIC_API_KEY / OPENAI_API_KEY / KAKEYA_ENDPOINT / KAKEYA_GRPC_ADDRESS
  AGENT_LLM_MODEL  (optional override)

Creative reasoning (brief/script/scene_plan/edit) runs here; mechanical text can still use
the in-repo kakeya_llm tool. `complete_json` returns a parsed dict, retrying once on bad JSON.

The `kakeya_grpc` provider talks DIRECTLY to a user-run Kakeya inference server over its native
gRPC RuntimeService (token-id level — the real stable surface; the HTTP shim is deprecated). It is
an EXTERNAL dependency contract: OpenMontage does NOT vendor the Kakeya SDK/proto or add
grpc/transformers as core deps — point `KAKEYA_REPO` at a checkout of the Kakeya engine and those
imports happen lazily, only when this provider is selected. Other providers stay urllib-only.
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
    "kakeya_grpc": "kakeya-local",
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
            elif os.environ.get("KAKEYA_GRPC_ADDRESS"):
                self.provider = "kakeya_grpc"
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
        if self.provider == "kakeya_grpc":
            return self._kakeya_grpc(system, user, max_tokens)
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

    def _kakeya_grpc(self, system: str, user: str, max_tokens: int) -> str:
        """Direct token-id-level generation over the Kakeya gRPC RuntimeService.

        External dependency contract: the Kakeya SDK/proto are NOT vendored — set KAKEYA_REPO to a
        checkout of the engine (it must contain a top-level `kakeya` package and `sdks/python`).
        Needs `grpcio` + `transformers` (and the tokenizer at KAKEYA_TOKENIZER_ID) on the host;
        these are imported lazily so they never become core OpenMontage deps."""
        import sys
        from pathlib import Path

        repo = os.environ.get("KAKEYA_REPO")
        if not repo:
            raise RuntimeError(
                "kakeya_grpc requires KAKEYA_REPO (path to a Kakeya inference-engine checkout) so "
                "its gRPC SDK can be imported. Also set KAKEYA_GRPC_ADDRESS (e.g. 127.0.0.1:51051) "
                "and KAKEYA_TOKENIZER_ID (HF/MLX tokenizer dir).")
        kakeya_root = Path(repo).expanduser()
        for p in (str(kakeya_root), str(kakeya_root / "sdks" / "python")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from kakeya import Client  # type: ignore  # noqa: E402
        from transformers import AutoTokenizer  # noqa: E402

        address = os.environ.get("KAKEYA_GRPC_ADDRESS", "127.0.0.1:51051")
        tokenizer_id = os.environ.get(
            "KAKEYA_TOKENIZER_ID", os.environ.get("KAKEYA_VERIFIER_ID", self.model))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        eos = tokenizer.eos_token_id
        eos_ids = [int(eos)] if eos is not None else []
        token_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=True, return_dict=False, enable_thinking=False)
        with Client(address) as client:
            session = client.create_session(eos_token_ids=eos_ids, client_label="openmontage-agent")
            try:
                session.append(token_ids)
                generated = list(session.generate(max_tokens=max_tokens))
            finally:
                try:
                    session.close()
                except Exception:  # noqa: BLE001
                    pass
        return tokenizer.decode(generated, skip_special_tokens=True)

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
