# ADR 0016 — Autonomous agent runtime for the public gateway (Gap B, Phase 1)

**Status:** Accepted; MVP implemented + offline-validated. Live run pending a cloud LLM key.
**Date:** 2026-06-23
**Related:** ADR 0011/0013 (agent gateway, re-anchor), AGENT_GUIDE.md (instruction-driven model).

## 1. Problem (Gap B)

`agent.kakeya.ai` only did **bare text→video** through the WAN cluster. The gateway's `mode=agent`
requires an `AGENT_RUNTIME_CMD`, which was unattached (`agent_runtime=false`) — so none of
OpenMontage's pipeline/skills (script → scene_plan → assets via selectors → compose) ran. The user
asked to **integrate the OpenMontage agent** so the public service runs a real multi-stage pipeline.

## 2. Key finding

OpenMontage is **instruction-driven**: there is **no Python pipeline runner** — the *host LLM*
(Cursor/Claude) is normally the orchestrator, reading manifests + director skills and driving tools.
To run **unattended** behind the gateway, we must supply our own LLM reasoner. In-repo LLM access is
only `kakeya_llm`/`llm_selector` (mechanical text). So this ADR introduces a new **autonomous agent
runtime**.

## 3. Decision

New module `services/agent_runtime/` exposed as `AGENT_RUNTIME_CMD`
(`python -m services.agent_runtime.run --prompt … --out …`):

- **`llm.py`** — dependency-free (urllib) cloud-LLM client: `AGENT_LLM = anthropic|openai|kakeya|stub`
  (auto-detected from `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`KAKEYA_ENDPOINT`). `complete_json` parses
  + retries. Creative reasoning runs here; the host agent is no longer required at runtime.
- **`run.py`** — minimal unattended orchestrator that reuses the real machinery:
  - LLM produces **brief → script → scene_plan** (each schema-validated via `schemas.artifacts`
    and checkpointed via `lib/checkpoint`, exactly like the governed pipeline; director-skill text
    is loaded as LLM context — "intelligence in the skills").
  - assets: per-scene video via **`video_selector`** (auto-routes to the BEST available provider —
    premium Seedance/Kling/Veo if configured, else the local WAN draft cluster), narration via
    **`tts_selector`**, one music track via **`music_gen`** (skipped when unconfigured).
  - deterministic **edit_decisions** → **`audio_mixer`** + **`video_compose`** (ffmpeg) → final mp4.
- `AGENT_RUNTIME_FAKE_ASSETS=1` swaps providers for ffmpeg fixtures so the harness is CI-testable
  with no keys/GPU.

**Topology** (per the cost/stability recommendation): agent runtime runs on the **head Mac CPU**
(cloud LLM) next to the gateway; headless Mac stays a video worker; vast does heavy generation. The
agent uses a cloud LLM (light CPU/network) — it does **not** steal GPU from video.

## 4. Honest scope / limits

- This is an **MVP "auto" path**, not the full governed `animated-explainer` pipeline (no
  research/proposal, human-approval gates, reviewer governance, or per-stage director nuance).
- **Needs a cloud LLM key** to reason (none configured). Without it, `mode=agent` stays honestly
  "not attached".
- **Output quality is provider-bounded.** With only the WAN cluster, clips are draft quality.
  Configure a **premium video provider** (e.g. fal.ai/Seedance) and `video_selector` auto-upgrades —
  that is the real path to production-grade. TTS/music/images need their own keys.
- Not "all skills integrated": skills light up as their providers are configured + as more pipelines
  are wired. This is the first integration, not the finish line.

## 5. Validation

`tests/tools/test_agent_runtime.py` (offline, no keys/GPU): stub LLM emits valid brief/script/
scene_plan; `AGENT_RUNTIME_FAKE_ASSETS=1` → full plumbing (plan → validate → checkpoint → per-scene
assets → audio mix → ffmpeg compose) → **valid h264 mp4** + checkpoints for idea/script/scene_plan/
assets/edit. JSON-extraction unit test. 2 pass (23 with gateway+pipeline suites).

## 6. Next

- Deploy: set `AGENT_RUNTIME_CMD="<venv> -m services.agent_runtime.run"` + a cloud LLM key on the
  head; live-test `mode=agent` via `agent.kakeya.ai`.
- Configure premium video provider for production-grade clips (the quality answer from the
  draft-vs-premium discussion).
- Grow toward the governed pipeline (proposal/approval/reviewer, more director skills) and more
  skills (avatar, images, music) as keys are added.
