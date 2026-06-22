#!/bin/bash
# Rotation-resilient cloudflared launcher for the gateway tunnel (head Mac).
# Substitute __TUNNEL_ID__ with your tunnel's UUID. Install at ~/run_cloudflared.sh and point
# the ai.kakeya.cloudflared LaunchAgent at it.
#
# Why a wrapper instead of an embedded --token in the plist: a Cloudflare dashboard "Refresh
# token" invalidates the old token and would silently break a plist that hard-codes it (this
# happened — see ADR 0001 Iteration 31). `cloudflared tunnel token <ID>` fetches the CURRENT
# token at startup (needs a valid ~/.cloudflared/cert.pem for the zone), so auto-start keeps
# working across rotations.
CF="$(command -v cloudflared || echo /opt/homebrew/bin/cloudflared)"
TUNNEL_ID="__TUNNEL_ID__"
TOKEN="$("$CF" tunnel token "$TUNNEL_ID" 2>/dev/null)"
if [ -z "$TOKEN" ]; then
  echo "$(date) ERROR: could not fetch token for $TUNNEL_ID (cert.pem missing/invalid?)" >&2
  sleep 10            # let launchd KeepAlive retry rather than hot-loop
  exit 1
fi
exec "$CF" tunnel --no-autoupdate --url http://localhost:8088 run --token "$TOKEN"
