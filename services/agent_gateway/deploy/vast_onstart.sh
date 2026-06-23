#!/usr/bin/env bash
# =============================================================================
# vast.ai ON-START SCRIPT — paste into the instance's "On-start Script" field.
# Prepares a freshly (re)created box so the head supervisor auto-adopts it as a
# distributed-WAN refiner / I2V long-form generator (ADR 0015 Phase 2c durability).
# Runs under the container init (survives SSH logout), unlike a login-shell launch.
# =============================================================================
set -e

# 1) authorize the head Mac so the supervisor can scp + hold the worker (substitute YOUR head pubkey)
mkdir -p ~/.ssh && chmod 700 ~/.ssh
HEAD_KEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHt78WgXxgiP0l6KQyiT2ioPMt70AjprV6vun+C1xzju fluffy314@fluffy314s-Mac-mini.local'
grep -qF "fluffy314s-Mac-mini" ~/.ssh/authorized_keys 2>/dev/null || printf '%s\n' "$HEAD_KEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 2) persistent model cache: keep the ~84G I2V-720P model across recreates.
#    Best: attach a vast VOLUME and point HF_HOME at it (or STOP instead of DESTROY to keep /workspace).
mkdir -p "${HF_HOME:-/workspace/.hf_home}"

mkdir -p /workspace/distwan
echo "[on-start] ready: head authorized, HF_HOME=${HF_HOME:-/workspace/.hf_home}."
echo "[on-start] The head supervisor (VAST_HOLD_WORKER=1) will scp the worker and hold it alive."
# Optional: pre-pull the I2V model now so the first long-form job isn't cold (uncomment + set token):
# HF_HOME="${HF_HOME:-/workspace/.hf_home}" python -c "from huggingface_hub import snapshot_download; \
#   snapshot_download('Wan-AI/Wan2.1-I2V-14B-720P-Diffusers')" || true
