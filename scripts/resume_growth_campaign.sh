#!/usr/bin/env bash
# Resume the full 10x growth campaign (ai-docs/gaps.md §16/§20) after it's been
# stopped for any reason (reboot, manual shutdown, another restart-to-deploy-a-fix
# like §21/§22's mid-campaign restarts). Self-contained: reads everything it needs
# from the campaign's own on-disk state (params.txt + metrics.jsonl) rather than
# anything session-specific, so it works even with no memory of prior sessions.
#
# Usage:
#   scripts/resume_growth_campaign.sh              # rebuilds somnus first, then resumes
#   SKIP_BUILD=1 scripts/resume_growth_campaign.sh  # skip the rebuild (binary already current)
#   DRY_RUN=1 scripts/resume_growth_campaign.sh     # print the computed invocation, don't run it
#
# To stop the campaign cleanly at any point:
#   pkill -f train_campaign.sh   # stop the wrapper so it doesn't launch a new session
#   pkill -TERM -f "target/release/somnus --file clips/h3h3"   # graceful shutdown of somnus itself
#   (state is snapshotted every sleep cycle, so at most a few seconds of
#   in-progress work is lost — safe to do any time)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CAMPAIGN_DIR="${CAMPAIGN_DIR:-memory/campaign_growth10x}"
PARAMS_FILE="${CAMPAIGN_DIR}/params.txt"
METRICS_FILE="${CAMPAIGN_DIR}/metrics.jsonl"
SKIP_BUILD="${SKIP_BUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "$PARAMS_FILE" ] || [ ! -f "$METRICS_FILE" ]; then
    echo "[resume] $PARAMS_FILE or $METRICS_FILE not found — nothing to resume." >&2
    echo "[resume] (this script only resumes an existing campaign; to start a new one, call scripts/train_campaign.sh directly)" >&2
    exit 1
fi

if pgrep -f train_campaign.sh >/dev/null 2>&1; then
    echo "[resume] train_campaign.sh is already running — nothing to do." >&2
    pgrep -af train_campaign.sh >&2
    exit 1
fi

# ── Read the most recent params block (params.txt is appended-to across restarts;
#    a plain key=value grep naturally picks up the last occurrence of each key). ──
get_param() {
    grep "^${1}=" "$PARAMS_FILE" | tail -1 | cut -d= -f2-
}
PREV_SESSION_START=$(get_param SESSION_START)
PREV_SESSIONS=$(get_param SESSIONS)
CYCLES=$(get_param CYCLES)
CAPACITY=$(get_param CAPACITY)
CLIPS=$(get_param CLIPS)
MEMORY_DIR=$(get_param MEMORY_DIR)
REPLAY_EPOCHS=$(get_param REPLAY_EPOCHS)
PRIOR_BETA=$(get_param PRIOR_BETA)

# ── Where the campaign was actually headed (stable across restarts — each prior
#    restart recomputed SESSIONS to preserve this same final session number). ──
TARGET_SESSION_END=$((PREV_SESSION_START + PREV_SESSIONS - 1))

# ── Where it actually got to, from the metrics themselves (authoritative — a
#    restart's params.txt block is written before that leg necessarily completes
#    a single cycle, so trust the last logged cycle, not the params file alone). ──
LAST_SESSION=$(tail -1 "$METRICS_FILE" | python3 -c "import json,sys; print(json.load(sys.stdin)['session'])")

NEW_SESSION_START=$((LAST_SESSION + 1))
NEW_SESSIONS=$((TARGET_SESSION_END - NEW_SESSION_START + 1))

if [ "$NEW_SESSIONS" -le 0 ]; then
    echo "[resume] campaign already reached its target session (${TARGET_SESSION_END}); nothing to resume."
    echo "[resume] run scripts/campaign_status.py to see final state."
    exit 0
fi

echo "[resume] last logged: session ${LAST_SESSION}"
echo "[resume] target session end: ${TARGET_SESSION_END} (from most recent params.txt block)"
echo "[resume] resuming: SESSION_START=${NEW_SESSION_START} SESSIONS=${NEW_SESSIONS} CYCLES=${CYCLES}"

if [ "$DRY_RUN" = "1" ]; then
    echo "[resume] DRY_RUN=1 set — not launching. Invocation would be:"
    echo "SESSION_START=${NEW_SESSION_START} SESSIONS=${NEW_SESSIONS} CYCLES=${CYCLES} CAPACITY=${CAPACITY} MEMORY_DIR=${MEMORY_DIR} CAMPAIGN_DIR=${CAMPAIGN_DIR} REPLAY_EPOCHS=${REPLAY_EPOCHS} PRIOR_BETA=${PRIOR_BETA} SKIP_BUILD=${SKIP_BUILD} CLIPS=\"...\" scripts/train_campaign.sh"
    exit 0
fi

if [ "$SKIP_BUILD" != "1" ]; then
    echo "[resume] building somnus (release)…"
    cargo build --release -p somnus
fi

export SESSION_START="$NEW_SESSION_START"
export SESSIONS="$NEW_SESSIONS"
export CYCLES
export CAPACITY
export MEMORY_DIR
export CAMPAIGN_DIR
export REPLAY_EPOCHS
export PRIOR_BETA
export CLIPS
export SKIP_BUILD=1   # already built above (or explicitly skipped) — don't build twice

exec scripts/train_campaign.sh
