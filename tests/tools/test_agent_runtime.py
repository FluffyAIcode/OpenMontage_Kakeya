"""Offline test for the autonomous agent runtime (no LLM key, no GPU, no provider keys).

Uses a STUB LLM (emits valid brief/script/scene_plan artifacts) + AGENT_RUNTIME_FAKE_ASSETS=1
(ffmpeg color/sine fixtures) so the full pipeline plumbing — LLM planning -> schema validation ->
checkpoints -> per-scene assets -> audio mix -> ffmpeg compose -> final mp4 — is exercised
deterministically. Proves the harness works end-to-end before any cloud LLM / provider is attached.

Run:  pytest tests/tools/test_agent_runtime.py -q   (needs ffmpeg on PATH)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

pytest.importorskip("jsonschema")
if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    pytest.skip("ffmpeg/ffprobe not available", allow_module_level=True)


def _stub(system: str, user: str) -> str:
    s = system[:120].lower()  # match the prompt PREFIX, before appended skill text
    if "scene director" in s:
        return json.dumps({"version": "1.0", "scenes": [
            {"id": "sc1", "type": "generated", "description": "a red fox walking into a snowy forest, cinematic",
             "start_seconds": 0, "end_seconds": 5, "script_section_id": "s1"},
            {"id": "sc2", "type": "generated", "description": "fox pauses, soft winter light, slow motion",
             "start_seconds": 5, "end_seconds": 10, "script_section_id": "s2"}]})
    if "script director" in s:
        return json.dumps({"version": "1.0", "title": "Snowy Fox", "total_duration_seconds": 10,
                           "sections": [
                               {"id": "s1", "text": "A red fox steps into a snowy forest.",
                                "start_seconds": 0, "end_seconds": 5},
                               {"id": "s2", "text": "It pauses as soft winter light filters through.",
                                "start_seconds": 5, "end_seconds": 10}]})
    if "idea director" in s:
        return json.dumps({"version": "1.0", "title": "Snowy Fox", "hook": "A fox in the snow.",
                           "key_points": ["winter", "forest", "cinematic"], "tone": "calm",
                           "style": "cinematic", "target_platform": "youtube",
                           "target_duration_seconds": 10})
    return "{}"


def test_agent_runtime_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_FAKE_ASSETS", "1")
    monkeypatch.setenv("AGENT_CLIP_WIDTH", "320")   # tiny for speed
    monkeypatch.setenv("AGENT_CLIP_HEIGHT", "240")
    import importlib
    import services.agent_runtime.run as run_mod
    importlib.reload(run_mod)
    from services.agent_runtime.llm import LLM

    out = tmp_path / "final.mp4"
    run_mod.run("a red fox in a snowy forest", str(out), llm=LLM(stub=_stub),
                project_dir=tmp_path / "proj")

    assert out.exists() and out.stat().st_size > 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=codec_name,width,height", "-of", "default=noprint_wrappers=1", str(out)],
                           capture_output=True, text=True)
    assert "codec_name=h264" in probe.stdout, probe.stdout + probe.stderr
    # checkpoints written for the planning + asset + edit stages
    cks = list((tmp_path / "proj" / "pipeline").rglob("checkpoint_*.json"))
    names = {c.stem for c in cks}
    assert {"checkpoint_idea", "checkpoint_script", "checkpoint_scene_plan",
            "checkpoint_assets", "checkpoint_edit"} <= names, names


def test_llm_json_extraction():
    from services.agent_runtime.llm import LLM
    llm = LLM(stub=lambda s, u: 'prefix ```json\n{"a": 1, "b": {"c": 2}}\n``` suffix')
    assert llm.complete_json("sys", "user") == {"a": 1, "b": {"c": 2}}
