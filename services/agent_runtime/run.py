"""OpenMontage autonomous agent runtime (MVP) — `AGENT_RUNTIME_CMD` for the gateway's mode=agent.

OpenMontage is instruction-driven (the host LLM is normally the orchestrator). This module is a
minimal UNATTENDED orchestrator so the public gateway can run a real multi-stage pipeline from a
brief, instead of a bare text->video call:

    brief (LLM) -> script (LLM) -> scene_plan (LLM)
      -> assets: per-scene video via video_selector (auto-routes to the BEST available provider:
                 premium Seedance/Kling/Veo if configured, else the local WAN draft cluster),
                 + narration via tts_selector, + one music track via music_gen (when available)
      -> edit_decisions (deterministic cuts) -> compose via audio_mixer + video_compose (ffmpeg)
      -> final mp4 at --out.

Each canonical artifact is schema-validated and checkpointed (lib/checkpoint), exactly like the
governed pipeline. Creative decisions come from the LLM; only mechanical wiring is in code.

Honest scope: this is an MVP "auto" path, not the full governed animated-explainer pipeline
(no research/proposal/human-approval/reviewer governance). Output quality is bounded by the
configured providers — configure a premium video provider key to get production-grade clips;
with only the WAN cluster it is draft quality.

CLI:  python -m services.agent_runtime.run --prompt "..." --out /path/final.mp4
Env:  AGENT_LLM + key (see llm.py). AGENT_RUNTIME_FAKE_ASSETS=1 -> ffmpeg fixtures (offline tests).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from services.agent_runtime.llm import LLM  # noqa: E402

FAKE = os.environ.get("AGENT_RUNTIME_FAKE_ASSETS", "0").lower() in ("1", "true", "yes")
MAX_SCENES = int(os.environ.get("AGENT_MAX_SCENES", "5"))
CLIP_W = int(os.environ.get("AGENT_CLIP_WIDTH", "1280"))
CLIP_H = int(os.environ.get("AGENT_CLIP_HEIGHT", "720"))


def log(msg: str):
    print(f"[agent] {msg}", flush=True)


def _prepare_ffmpeg():
    """Ensure `ffmpeg`/`ffprobe` resolve on PATH for all tool subprocesses.

    Macs in this deployment have no system ffmpeg — the cluster uses imageio-ffmpeg's bundled
    binary. Symlink it as `ffmpeg` (and `ffprobe` if imageio-ffmpeg ships one) into a dir on PATH
    so video_compose/audio_mixer (which call bare `ffmpeg`) work without a brew install."""
    import shutil
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return
    bind = Path(os.environ.get("TMPDIR", "/tmp")) / "agent_ffmpeg_bin"
    bind.mkdir(parents=True, exist_ok=True)
    link = bind / "ffmpeg"  # imageio-ffmpeg bundles ffmpeg only (not ffprobe)
    if not link.exists():
        try:
            os.symlink(exe, link)
        except OSError:
            pass
    os.environ["PATH"] = f"{bind}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", exe)
    os.environ.setdefault("FFMPEG_BINARY", exe)


def _skill(path: str) -> str:
    """Best-effort director-skill text for LLM context (truncated). Intelligence lives in skills."""
    for cand in (REPO / "skills" / f"{path}.md", REPO / f"{path}.md"):
        try:
            return cand.read_text()[:6000]
        except OSError:
            continue
    return ""


def _skill_named(name: str) -> str:
    """Read a named Layer-3 skill (SKILL.md) from the skills trees (truncated)."""
    for base in (REPO / ".agents" / "skills", REPO / ".claude" / "skills"):
        cand = base / name / "SKILL.md"
        try:
            return cand.read_text()
        except OSError:
            continue
    return ""


def _resolve_video_skill(registry) -> str:
    """Layer-3 prompting guidance for the active video provider.

    OpenMontage already ships the "prompt director" knowledge as Layer-3 skills (ai-video-gen,
    ltx2, seedance-2-0, ...). The agent reads the skill(s) advertised by the video provider that
    `video_selector` routes to and feeds them to the LLM so per-scene prompts are written the way
    that model wants — instead of passing a bare scene description to the generator."""
    names: list[str] = []
    sel = registry.get("video_selector")
    if sel is not None:
        try:
            info = sel.get_info()
            # If the selector can name the provider it would route to, prefer that provider's skill.
            chosen = info.get("selected_tool") or info.get("provider")
            if chosen:
                prov = registry.get(chosen)
                if prov is not None:
                    names = list(getattr(prov, "agent_skills", []) or [])
            if not names:
                names = list(info.get("agent_skills", []) or [])
        except Exception:  # noqa: BLE001
            pass
    if not names:
        names = ["ai-video-gen"]
    txt = ""
    for n in names[:2]:
        s = _skill_named(n)
        if s:
            txt += f"\n# Provider guidance: {n}\n{s[:4500]}\n"
    return txt[:8000]


def direct_video_prompt(llm: LLM, scene_desc: str, brief: dict, skill_text: str) -> str:
    """LLM "prompt director": rewrite a scene into ONE rich, model-appropriate T2V prompt.

    Native T2V conditions generation via cross-attention over the text embeddings, so a longer,
    concrete, positive prompt (subject+action+setting+lighting+camera+style) yields a far stronger
    result than a bare scene description. Guidance comes from the provider's Layer-3 skill."""
    if not scene_desc:
        return scene_desc
    sys_p = (
        "You are OpenMontage's video prompt director. Rewrite the scene into ONE rich, concrete "
        "text-to-video prompt for a native text-to-video model. Include subject, action, setting, "
        "lighting, camera framing/movement, and style. Be vivid and specific, ~40-70 words, fully "
        "positive phrasing (describe what to SHOW, never what to avoid), no contradictions. "
        "Output ONLY the prompt text — no quotes, no JSON, no preamble." + skill_text
    )
    user = (f"Scene: {scene_desc}\nOverall tone: {brief.get('tone', '')}\n"
            f"Style: {brief.get('style', '')}\nTitle: {brief.get('title', '')}")
    try:
        out = llm.complete(sys_p, user)
    except Exception as exc:  # noqa: BLE001
        log(f"prompt-director fallback (raw scene): {exc}")
        return scene_desc
    out = " ".join((out or "").strip().strip('"').strip("`").split())
    return out[:600] or scene_desc


# --------------------------------------------------------------------------- #
# asset helpers (real providers via the registry, or ffmpeg fixtures for tests)
# --------------------------------------------------------------------------- #
def _ffmpeg_clip(out: str, seconds: float, color: str = "darkblue"):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"color=c={color}:s={CLIP_W}x{CLIP_H}:d={max(1, int(seconds))}:r=24",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", out], capture_output=True, check=True)


def _ffmpeg_sine(out: str, seconds: float):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=320:duration={max(1, int(seconds))}", "-ar", "44100", "-ac", "1", out],
                   capture_output=True, check=True)


def generate_clip(registry, prompt: str, out: str, seconds: float, idx: int) -> str:
    if FAKE:
        _ffmpeg_clip(out, seconds, ["darkblue", "darkgreen", "darkred", "purple", "teal"][idx % 5])
        return out
    sel = registry.get("video_selector")
    if sel is None:
        raise RuntimeError("no video_selector tool available")
    res = sel.execute({"prompt": prompt, "aspect_ratio": "16:9", "duration": str(int(seconds)),
                       "output_path": out})
    if not res.success:
        raise RuntimeError(f"video generation failed: {res.error}")
    return res.data.get("output") or res.data.get("output_path") or out


def generate_tts(registry, text: str, out: str) -> str | None:
    if FAKE:
        _ffmpeg_sine(out, max(2, len(text) // 15)); return out
    avail = [t for t in registry.get_by_capability("tts") if getattr(t, "name", "") != "tts_selector"
             and _tool_available(t)]
    if not avail:
        return None
    res = registry.get("tts_selector").execute({"text": text, "output_path": out})
    return (res.data.get("output") or out) if res.success else None


def generate_music(registry, prompt: str, out: str, seconds: float) -> str | None:
    if FAKE:
        _ffmpeg_sine(out, seconds); return out
    mg = registry.get("music_gen")
    if mg is None or not _tool_available(mg):
        return None
    res = mg.execute({"prompt": prompt, "duration_seconds": int(seconds), "output_path": out})
    return (res.data.get("output") or out) if res.success else None


def _tool_available(tool) -> bool:
    try:
        info = tool.get_info()
        return str(info.get("status", "")).lower() in ("available", "ready", "ok", "")
    except Exception:  # noqa: BLE001
        return True


# --------------------------------------------------------------------------- #
# LLM planning stages
# --------------------------------------------------------------------------- #
def plan_brief(llm: LLM, prompt: str) -> dict:
    sys_p = ("You are OpenMontage's idea director. Output a `brief` JSON artifact. "
             "Required keys: version='1.0', title, hook, key_points (array of 3-6 strings), "
             "tone, style, target_platform, target_duration_seconds (integer). " + _skill("pipelines/explainer/idea-director"))
    return llm.complete_json(sys_p, f"User brief: {prompt}")


def plan_script(llm: LLM, brief: dict) -> dict:
    dur = int(brief.get("target_duration_seconds", 20))
    sys_p = ("You are OpenMontage's script director. Output a `script` JSON artifact. Required: "
             "version='1.0', title, total_duration_seconds (integer), sections (array). Each section: "
             "id, text, start_seconds, end_seconds (contiguous, covering 0..total). Keep it concise. "
             + _skill("pipelines/explainer/script-director"))
    return llm.complete_json(sys_p, f"Brief:\n{json.dumps(brief)}\nTotal duration ~{dur}s.")


def plan_scenes(llm: LLM, script: dict) -> dict:
    sys_p = ("You are OpenMontage's scene director. Output a `scene_plan` JSON artifact. Required: "
             "version='1.0', scenes (array). Each scene: id, type (one of generated|text_card|"
             "animation|diagram), description (a vivid visual prompt for a video generator), "
             "start_seconds, end_seconds, script_section_id. One scene per script section, contiguous. "
             + _skill("pipelines/explainer/scene-director"))
    return llm.complete_json(sys_p, f"Script:\n{json.dumps(script)}")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run(prompt: str, out: str, llm: LLM | None = None, project_dir: Path | None = None) -> str:
    from lib.checkpoint import write_checkpoint
    from schemas.artifacts import validate_artifact
    from tools.tool_registry import registry

    _prepare_ffmpeg()
    registry.discover()
    llm = llm or LLM()
    log(f"LLM provider={llm.provider} model={llm.model or '(default)'}  fake_assets={FAKE}")
    pid = "agent_" + uuid.uuid4().hex[:8]
    proj = project_dir or (REPO / "projects" / "_agent" / pid)
    pdir = proj / "pipeline"
    assets = proj / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    # pipeline_type=None -> validate against ALL known stages (this MVP uses the canonical
    # idea->script->scene_plan->assets->edit->compose path, not a specific manifest's stage set).
    ck = lambda stage, art: write_checkpoint(pdir, pid, stage, "completed", artifacts=art,
                                              pipeline_type=None)  # noqa: E731

    # 1) brief -> 2) script -> 3) scene_plan (LLM, validated, checkpointed)
    log("planning brief...")
    brief = plan_brief(llm, prompt); brief.setdefault("version", "1.0")
    validate_artifact("brief", brief); ck("idea", {"brief": brief})
    log(f"brief: {brief.get('title')}")

    log("planning script...")
    script = plan_script(llm, brief); script.setdefault("version", "1.0")
    validate_artifact("script", script); ck("script", {"script": script})

    log("planning scenes...")
    scene_plan = plan_scenes(llm, script); scene_plan.setdefault("version", "1.0")
    scenes = scene_plan.get("scenes", [])[:MAX_SCENES]
    scene_plan["scenes"] = scenes
    validate_artifact("scene_plan", scene_plan); ck("scene_plan", {"scene_plan": scene_plan})
    log(f"{len(scenes)} scenes")

    # 4) assets: per-scene video (video_selector -> best provider) + narration; one music track
    sect_by_id = {s["id"]: s for s in script.get("sections", [])}
    vskill = _resolve_video_skill(registry)
    log(f"prompt-director: loaded provider guidance ({len(vskill)} chars)")
    asset_list, cuts, director_prompts = [], [], []
    for i, sc in enumerate(scenes):
        dur = max(2.0, float(sc.get("end_seconds", 0)) - float(sc.get("start_seconds", 0)) or 5.0)
        vid = str(assets / f"scene_{i}.mp4")
        raw = sc.get("description", prompt)
        vprompt = direct_video_prompt(llm, raw, brief, vskill)
        director_prompts.append({"scene_id": sc.get("id", f"sc{i}"), "raw": raw, "prompt": vprompt})
        log(f"scene {i}: directed prompt — {vprompt[:80]}")
        vid = generate_clip(registry, vprompt, vid, dur, i)
        asset_list.append({"id": f"v{i}", "type": "video", "path": vid, "source_tool": "video_selector",
                           "scene_id": sc.get("id", f"sc{i}")})
        cuts.append({"id": f"cut{i}", "source": vid, "in_seconds": 0.0, "out_seconds": dur, "speed": 1.0})
        # narration from the linked script section
        sec = sect_by_id.get(sc.get("script_section_id"))
        if sec and sec.get("text"):
            nar = generate_tts(registry, sec["text"], str(assets / f"nar_{i}.mp3"))
            if nar:
                asset_list.append({"id": f"n{i}", "type": "narration", "path": nar,
                                   "source_tool": "tts_selector", "scene_id": sc.get("id", f"sc{i}")})

    total = sum(c["out_seconds"] for c in cuts)
    music = generate_music(registry, f"background music, {brief.get('tone','')}",
                           str(assets / "music.mp3"), total)
    if music:
        asset_list.append({"id": "music", "type": "music", "path": music, "source_tool": "music_gen",
                           "scene_id": scenes[0].get("id", "sc0") if scenes else "sc0"})
    asset_manifest = {"version": "1.0", "assets": asset_list}
    validate_artifact("asset_manifest", asset_manifest); ck("assets", {"asset_manifest": asset_manifest})
    try:
        (assets / "director_prompts.json").write_text(json.dumps(director_prompts, indent=2))
    except OSError:
        pass

    # 5) edit_decisions (deterministic: clips in order, ffmpeg runtime)
    narrations = [a["path"] for a in asset_list if a["type"] == "narration"]
    edit = {"version": "1.0", "cuts": cuts, "render_runtime": "ffmpeg"}
    if music:
        edit["music"] = {"asset_id": "music", "volume": 0.2, "ducking": True}
    validate_artifact("edit_decisions", edit); ck("edit", {"edit_decisions": edit})

    # 6) compose: mix narration+music, then concat clips with audio
    from tools.video.video_compose import VideoCompose
    audio_path = None
    if narrations or music:
        audio_path = str(assets / "mix.wav")
        _build_audio(narrations, music, audio_path)
    log("composing final video (ffmpeg)...")
    res = VideoCompose().execute({
        "operation": "compose",
        "edit_decisions": {"cuts": [{"source": c["source"], "in_seconds": c["in_seconds"],
                                     "out_seconds": c["out_seconds"], "speed": 1.0} for c in cuts]},
        **({"audio_path": audio_path} if audio_path else {}),
        "output_path": out, "codec": "libx264", "crf": 23, "preset": "fast"})
    if not res.success or not Path(out).exists():
        raise RuntimeError(f"compose failed: {res.error}")
    render_report = {"version": "1.0", "outputs": [{"path": out, "format": "mp4", "codec": "h264",
                     "resolution": f"{CLIP_W}x{CLIP_H}", "duration_seconds": round(total, 2)}]}
    try:
        validate_artifact("render_report", render_report); ck("compose", {"render_report": render_report})
    except Exception as exc:  # noqa: BLE001
        log(f"render_report checkpoint skipped: {exc}")
    log(f"done -> {out}  ({brief.get('title')}, {len(scenes)} scenes, ~{total:.0f}s)")
    return out


def _build_audio(narrations: list[str], music: str | None, out: str):
    """Concat narration tracks, then duck music under them via the audio_mixer tool (ffmpeg)."""
    from tools.tool_registry import registry
    tmp = Path(out).with_suffix(".nar.wav")
    if narrations:
        lst = Path(out).with_suffix(".list.txt")
        lst.write_text("".join(f"file '{p}'\n" for p in narrations))
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), str(tmp)],
                       capture_output=True, check=True)
    mixer = registry.get("audio_mixer")
    if music and narrations and mixer is not None:
        r = mixer.execute({"operation": "duck",
                           "tracks": [{"path": str(tmp), "role": "speech"}, {"path": music, "role": "music"}],
                           "ducking": {"enabled": True, "music_volume_during_speech": 0.15},
                           "output_path": out})
        if r.success:
            return
    # fallback: narration-only (or music-only)
    src = str(tmp) if narrations else music
    subprocess.run(["ffmpeg", "-y", "-i", src, out], capture_output=True, check=True)


def check_llm() -> int:
    """Fast LLM-endpoint smoke test: one tiny round-trip. Verify the reasoner is reachable
    BEFORE committing to a full (slow) pipeline run. Returns a process exit code."""
    import time
    llm = LLM()
    log(f"LLM provider={llm.provider} model={llm.model or '(default)'}")
    if llm.provider == "stub":
        log("LLM check: provider=stub — no real endpoint configured (set AGENT_LLM + one of "
            "KAKEYA_GRPC_ADDRESS / KAKEYA_ENDPOINT / ANTHROPIC_API_KEY / OPENAI_API_KEY).")
        return 2
    t0 = time.time()
    try:
        out = llm.complete("You are a health check. Reply with exactly: OK",
                           "Reply with the single word OK.", max_tokens=16)
    except Exception as exc:  # noqa: BLE001
        log(f"LLM check FAILED ({type(exc).__name__}): {exc}")
        return 1
    dt = time.time() - t0
    log(f"LLM check OK in {dt:.1f}s — reply: {(out or '').strip()[:80]!r}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="brief/idea to drive the pipeline")
    ap.add_argument("--out", help="output mp4 path")
    ap.add_argument("--check-llm", action="store_true",
                    help="only verify the LLM endpoint round-trips, then exit (no GPU/pipeline)")
    args = ap.parse_args()
    if args.check_llm:
        raise SystemExit(check_llm())
    if not args.prompt or not args.out:
        ap.error("--prompt and --out are required (unless --check-llm)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run(args.prompt, args.out)


if __name__ == "__main__":
    main()
