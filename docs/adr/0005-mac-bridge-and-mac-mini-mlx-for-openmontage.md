# ADR 0005 — Kakeya "mac bridge" and using a Mac mini's MLX GPU for OpenMontage (objective evaluation)

- **Status:** Evaluation (no code change; documents feasibility + the correct path)
- **Date:** 2026-06-20
- **Deciders:** OpenMontage maintainers
- **Question:** Can the Kakeya repo's **mac bridge** be used (by this cloud agent) to
  connect to a local **Mac mini** and utilize its GPU for the integration?
- **Related:** ADR 0001 (Kakeya text path), ADR 0003 (Kakeya≠video), Kakeya
  `docs/design/mac-bridge-cloud-agent-access.md`

---

## 1. What the mac bridge actually is

Per Kakeya's own design doc, the implemented transport ("M1") is a **git-bus + GitHub
Actions self-hosted-runner** dispatch — **not** a GPU connection or a serving channel:

- The Mac mini is a **GitHub Actions self-hosted runner** (`kakeya-mac-m4`), **behind NAT,
  outbound-only** (long-polls GitHub). **No inbound path** by design (constraint C1).
- Agent **pushes a `mac-bridge/**` branch** + `.mac-bridge/request.json` → workflow
  `mac-bridge.yaml` triggers **on the Mac** → runs an **allowlisted preset** (fixed argv,
  no shell, typed/bounded params) → commits results back → agent polls via `gh`/`git fetch`.
- Allowlist = **MLX eval/bench/test** presets (`mlx-env-probe`, `mlx-backend-tests`,
  `k3-*` harness, `pytest-path`). Minutes-scale, async. Explicitly *"right for
  test/eval/bench, wrong for interactive"* — **not an inference-serving transport.**

## 2. Technical dealbreakers for "use it to utilize my Mac mini's GPU" (objective)

| # | Dealbreaker |
|---|-------------|
| D1 | **No inbound path to the Mac.** The bridge is git-relayed *because* the Mac is unreachable; this cloud agent cannot reach the user's local Mac (no SSH/creds/route). |
| D2 | **Requires user-side setup I can't perform.** The Mac must be **registered as a GitHub Actions self-hosted runner** on a repo I can push to, with the `mac-bridge.yaml` workflow + bridge code present, via `scripts/mac_bridge/setup_mac.sh`. That needs admin on the physical Mac **and** the repo — only the owner can do it. |
| D3 | **Wrong repo.** The bridge + workflow + runner registration live in the **Kakeya engine repo**, not `OpenMontage_Kakeya` (where this work happens). None of it exists here. |
| D4 | **CI/eval, not serving.** Even fully set up, the bridge dispatches MLX *test/bench* presets and returns commits — it does not serve live inference to OpenMontage. |
| D5 | **MLX serves LLM *text*, not WAN video.** The Mac's MLX runs Kakeya's LLM verifier/proposer (the text path, ADR 0001), not the CUDA diffusers video gateway. MLX multi-tenant is **serial-only** (README). |
| D6 | **WAN latency kills the data plane.** Doc §4.2: cloud↔desk RTT 30–150 ms/block makes token-level spec-decode across that boundary non-viable — *"proposer and verifier must share a LAN."* The Mac can be a **control/tool plane** node, never a low-latency data-plane peer to a cloud GPU. |

**Verdict:** this cloud agent **cannot** establish the connection or "use the Mac mini GPU"
via the bridge. The bridge is the right tool for a *different* job (dispatching Kakeya's MLX
eval/bench presets to the project's one Mac, for evidence), and it is **text/MLX-LLM**,
not video, and not serving.

## 3. The correct way to use a Mac mini's GPU for OpenMontage

**Not the git-bus bridge** — run Kakeya's **MLX server** on the Mac and point OpenMontage at
it. OpenMontage already supports this via ADR 0001's `kakeya_llm` + `KAKEYA_ENDPOINT` —
**no new OpenMontage code required.** It serves the `text_generation` chores (batch prompt
expansion, subtitle translation, caption variants) on the Mac's GPU.

```bash
# On the Mac mini (Apple Silicon), in the Kakeya repo — run by the OWNER:
bash scripts/setup_mac.sh
PYTHONPATH=.:sdks/python python3 scripts/serve.py \
    --backend mlx --verifier-id mlx-community/Qwen3-1.7B-4bit \
    --host 0.0.0.0 --port 8000
# Expose on the LAN, or join a Tailscale tailnet for off-LAN access.
```
```bash
# In OpenMontage .env (same LAN / tailnet as the Mac):
KAKEYA_ENDPOINT=http://<mac-mini-ip-or-tailnet-name>:8000
```

This is the same integration seam validated in ADR 0001 (and against a real Kakeya server in
the loop log). The Mac mini becomes the **local, private, zero-marginal-cost text backend**;
no torch/CUDA on the OpenMontage host, no cloud LLM cost.

### If the eval/bench bridge is specifically wanted

The owner sets up the Mac as a self-hosted runner + the `mac-bridge.yaml` workflow on a repo
the agent can push to, then an agent can dispatch MLX presets
(`scripts/mac_bridge/kakeya_mac.py run --preset mlx-env-probe`). That is **Kakeya-engine CI
evidence**, not OpenMontage serving — useful for validating MLX behavior, not for production
text generation.

## 4. Consequences / boundary

- **Control/tool plane over WAN, data plane on the LAN** (matches ADR 0003/0009): the Mac
  mini is a fine *local* MLX text server (§3) and a fine *eval* target via the bridge; it is
  not a cloud-coupled inference peer.
- **Video stays on CUDA** (ADR 0002/0004): the Mac mini cannot host the WAN/CogVideo gateway.
- **No OpenMontage code change**: §3 reuses the existing `kakeya_llm`/`KAKEYA_ENDPOINT` seam.

## 5. What I (cloud agent) did and did not do

- **Did:** read the bridge design + client (`kakeya_mac.py doctor` requires being inside the
  Kakeya repo with push rights to `mac-bridge/**` + a registered Mac runner — none of which
  exist for this OpenMontage workspace), and documented the objective feasibility + the
  correct serving path.
- **Did not:** fabricate a connection to a Mac I cannot reach, or vendor Kakeya-CI workflow
  files into OpenMontage. Per the no-fake guideline, the honest output is this evaluation +
  the runbook the owner runs on their Mac.
