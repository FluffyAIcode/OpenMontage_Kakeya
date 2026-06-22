# ADR 0013 — Re-anchor: WAN cluster as a registered provider tool; localized OpenMontage behind Cloudflare

**Status:** Accepted; implementation in progress (keystone done).
**Date:** 2026-06-22
**Related:** ADR 0002 (video gateway), ADR 0006/0010 (distributed-WAN), ADR 0011 (agent gateway),
ADR 0012 (Mac high-res). Supersedes the "bare T2V endpoint" framing of the kakeya.ai service.

## 1. Problem (from the architecture review)

The integration had drifted into a standalone text→video stack *next to* OpenMontage rather than
*inside* its agent. A `kakeya.ai` caller got a raw WAN clip — no script/scene/music/compose/review.
The fix: make the WAN cluster a first-class **`video_generation` provider tool** the OpenMontage
agent + pipelines use, and run **OpenMontage itself locally** (on the Mac), exposed via Cloudflare.

Owner feedback adopted here:
- Localize OpenMontage; expose it through a **Cloudflare tunnel**.
- **No co-located CUDA**; attach a **vast** GPU **on-demand** only for the refine pass.
- The ~512 G Mac is **high-res capable** (ADR 0012), not drafts-only.
- Harden before public; **collapse the two video backends** into one local-video abstraction.

## 2. Re-anchor (keystone — implemented)

The WAN cluster is now reached through the existing provider seam, so the agent uses it like any
other provider:

```
OpenMontage agent / pipeline
   └─ video_selector  (capability routing, auto-discovers providers)
        └─ wan_video  (provider tool, capability="video_generation")
             └─ tools/video/_shared.py: generate_local_video()   ← ONE unified local-video seam
                  ├─ 1. distributed-WAN gRPC cluster   (WAN_WORKERS set)   ← NEW
                  │       • Mac-only  → DIRECT no-refine T2V at requested W×H
                  │       • + VAST_REFINE_WORKER → proposer(Mac)+refine(vast) on-demand
                  ├─ 2. warm video gateway (HTTP)       (VIDEO_INFER_ENDPOINT set)
                  └─ 3. in-process diffusers            (VIDEO_GEN_LOCAL_ENABLED=true)
```

- `wan_video` already declares `capability="video_generation"` → discoverable by `video_selector`
  → usable by every pipeline's asset stage. No pipeline changes needed; the agent picks it via the
  normal selector flow.
- **Collapse:** the distributed cluster, the warm HTTP gateway, and in-process diffusers are now
  three transports behind the **single** `generate_local_video()` seam — one local-video provider
  abstraction, not parallel stacks.
- **On-demand vast:** `VAST_REFINE_WORKER` appends a vast CUDA worker to `WAN_WORKERS` and switches
  the cluster from Mac-only DIRECT to proposer→refine. Unset → Mac-only. Attach/detach per need.
- Tests: `tests/tools/test_local_wan_provider.py` (Mac-only DIRECT + vast-attached refine, fake
  orchestrator). `local_generation_status()` reports AVAILABLE when `WAN_WORKERS` is set.

## 3. Localized OpenMontage behind Cloudflare

- OpenMontage (repo + agent + tools) runs **on the Mac**. The agent drives pipelines; the asset
  stage calls `wan_video` → local cluster. The `kakeya.ai` front door (ADR 0011 agent_gateway,
  `mode="agent"`) exposes the **agent/pipeline**, not a bare T2V call.
- Public access via **Cloudflare Tunnel** (`cloudflared`, outbound) — already documented
  (`services/agent_gateway/deploy/cloudflare.md`, `mac_all_in_one.sh`). The async job API keeps
  long Mac renders under Cloudflare's 100 s proxy limit.

## 4. Compute right-sizing (per ADR 0012)

- **Default:** Mac at Standard tier (14B @ 720p, few-step LoRA) — high-res, a few minutes, async.
- **On-demand vast** only when latency matters or for a refine polish — spin up, set
  `VAST_REFINE_WORKER`, run, tear down. No always-on / co-located CUDA.

## 5. Still to do (sequenced follow-ups, owner agreed)

1. **`mode="agent"` runtime.** Wire the gateway's agent mode to a real OpenMontage pipeline driver
   (needs an LLM-backed agent runtime on the Mac). Until then it's an honest "runtime not attached".
2. **Hardening before public:** durable job store (sqlite/redis, survives restart), launchd/systemd
   supervision + health-restart, rate limiting, per-key auth (not one static key).
3. **Parameterize the distributed canvas** so refine mode honors arbitrary W×H (today it emits a
   fixed 1472×768 canvas).
4. **Benchmark the 512 G Mac** and replace ADR 0012 estimates with measured latency.

## 6. Boundary (do not regress)

- The provider tool stays a thin dispatcher; orchestration/creative logic stays in the agent+skills.
- Never expose `kakeya.ai` publicly without auth + rate limiting (#2).
- vast is **ephemeral and on-demand**; nothing in the default path may assume it is present.
