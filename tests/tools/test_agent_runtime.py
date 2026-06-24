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
    if "continuity director" in s:  # style/character bible (cross-scene consistency)
        return json.dumps({"subject": "a black-and-white Border Collie with a white blaze and amber eyes",
                            "style": "cinematic, soft cool winter light, 35mm film, muted palette"})
    if "plan reviewer" in s:  # governance review pass
        return json.dumps({"approved": True, "issues": [], "revised_prompts": []})
    if "video prompt director" in s:
        # echo the raw scene so the test can verify the DIRECTED prompt is what reaches the generator
        scene = user.split("\n", 1)[0].removeprefix("Scene: ").strip()
        return f"directed cinematic shot, {scene}, golden hour rim light, slow dolly-in, 35mm film"
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
    # prompt-director step ran: each scene got a DIRECTED prompt (enriched, != raw description)
    dp = json.loads((tmp_path / "proj" / "assets" / "director_prompts.json").read_text())
    assert len(dp) == 2, dp
    for entry in dp:
        assert entry["prompt"].startswith("directed cinematic shot, "), entry
        assert entry["prompt"] != entry["raw"], entry


def test_agent_runtime_plan_only(tmp_path):
    """--plan-only path: LLM planning + prompt-director, NO video render. Validates the directed
    prompts (Gemma + director) can be produced/inspected without any GPU/gateway/key."""
    import importlib
    import services.agent_runtime.run as run_mod
    importlib.reload(run_mod)
    from services.agent_runtime.llm import LLM

    res = run_mod.run("a red fox in a snowy forest", "", llm=LLM(stub=_stub),
                      project_dir=tmp_path / "proj", plan_only=True)
    dp_path = Path(res)
    assert dp_path.name == "director_prompts.json" and dp_path.exists()
    dp = json.loads(dp_path.read_text())
    assert len(dp) == 2 and all(e["prompt"].startswith("directed cinematic shot, ") for e in dp)
    # no clips/compose happened
    assert not list((tmp_path / "proj" / "assets").glob("scene_*.mp4"))
    # continuity bible + governance review artifacts were produced
    bible = json.loads((tmp_path / "proj" / "assets" / "style_bible.json").read_text())
    assert bible.get("subject") and bible.get("style")
    review = json.loads((tmp_path / "proj" / "assets" / "review.json").read_text())
    assert review.get("approved") is True


def test_llm_json_extraction():
    from services.agent_runtime.llm import LLM
    llm = LLM(stub=lambda s, u: 'prefix ```json\n{"a": 1, "b": {"c": 2}}\n``` suffix')
    assert llm.complete_json("sys", "user") == {"a": 1, "b": {"c": 2}}


def test_sanitize_script_drops_invalid_cues():
    """Guard the observed Gemma drift: an out-of-enum enhancement_cues.type (e.g. 'visual') and
    unknown section fields must be stripped so the strict script schema still validates."""
    import services.agent_runtime.run as run_mod
    from schemas.artifacts import validate_artifact
    script = {
        "version": "1.0", "title": "T", "total_duration_seconds": 10,
        "sections": [
            {"id": "s1", "text": "a", "start_seconds": 0, "end_seconds": 5, "bogus_field": 1,
             "enhancement_cues": [{"type": "visual", "description": "bad"},
                                  {"type": "broll", "description": "ok"}]},
            {"id": "s2", "text": "b", "start_seconds": 5, "end_seconds": 10,
             "enhancement_cues": [{"type": "nope", "description": "x"}]},
        ],
    }
    out = run_mod._sanitize_script(script)
    s1, s2 = out["sections"]
    assert "bogus_field" not in s1
    assert [c["type"] for c in s1["enhancement_cues"]] == ["broll"]  # 'visual' dropped
    assert "enhancement_cues" not in s2  # all-invalid -> removed
    validate_artifact("script", out)  # now passes the strict schema


def test_kakeya_grpc_requires_repo(monkeypatch):
    """kakeya_grpc is an external-dependency contract: without KAKEYA_REPO it must fail with a
    clear, actionable error (not an obscure ImportError) and never become a core dep."""
    from services.agent_runtime.llm import LLM
    monkeypatch.delenv("KAKEYA_REPO", raising=False)
    monkeypatch.setenv("KAKEYA_GRPC_ADDRESS", "127.0.0.1:51051")
    llm = LLM(provider="kakeya_grpc")
    assert llm.provider == "kakeya_grpc"
    with pytest.raises(RuntimeError, match="KAKEYA_REPO"):
        llm.complete("sys", "user")
