#!/usr/bin/env bash
# Gracefully stop a running train_campaign.sh session (the growth campaign, or any
# other campaign launched the same way): stops the session-loop wrapper first (so
# it doesn't launch a new session right after this one exits), then sends SIGTERM
# to the somnus process itself so it can flush its current sleep-cycle snapshot before
# exiting. Safe to run any time — state is snapshotted every sleep cycle, so at
# most a few seconds of in-progress work is lost.
#
# Matches ANY train_campaign.sh wrapper and ANY somnus process invoked with
# --memory-dir (that's every campaign-launched somnus) — if more than one campaign is
# running at once this stops all of them. Fine for this repo's normal one-campaign-
# at-a-time usage; check the echoed PID list (or run --dry-run first) if unsure.
#
# Usage:
#   scripts/stop_graceful.sh                # stop whatever's running, wait up to 60s
#   scripts/stop_graceful.sh --wait 120      # wait up to 120s for clean exit before SIGKILL
#   scripts/stop_graceful.sh --dry-run       # show what would be stopped, don't touch it
#
# To resume afterward: scripts/resume_growth_campaign.sh

set -uo pipefail   # not -e: pgrep/kill returning "no match" (exit 1) is expected here

WAIT_SECS=60
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --wait) WAIT_SECS="${2:?--wait requires a number of seconds}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "[stop] unknown argument: $1" >&2; exit 2 ;;
    esac
done

WRAPPER_PIDS=$(pgrep -f "train_campaign\.sh" 2>/dev/null || true)
SOMNUS_PIDS=$(pgrep -f "target/release/somnus .*--memory-dir" 2>/dev/null || true)

if [ -z "$WRAPPER_PIDS" ] && [ -z "$SOMNUS_PIDS" ]; then
    echo "[stop] nothing running (no train_campaign.sh wrapper, no somnus process)."
    exit 0
fi

echo "[stop] wrapper PID(s): ${WRAPPER_PIDS:-none}"
echo "[stop] somnus PID(s):     ${SOMNUS_PIDS:-none}"

if [ "$DRY_RUN" = "1" ]; then
    echo "[stop] --dry-run set — not stopping anything."
    exit 0
fi

if [ -n "$WRAPPER_PIDS" ]; then
    echo "[stop] stopping train_campaign.sh wrapper(s)…"
    # shellcheck disable=SC2086
    kill $WRAPPER_PIDS 2>/dev/null || true
fi

if [ -n "$SOMNUS_PIDS" ]; then
    echo "[stop] sending SIGTERM to somnus process(es) for a clean shutdown…"
    # shellcheck disable=SC2086
    kill -TERM $SOMNUS_PIDS 2>/dev/null || true

    echo "[stop] waiting up to ${WAIT_SECS}s for clean exit…"
    waited=0
    still_alive=""
    while [ "$waited" -lt "$WAIT_SECS" ]; do
        still_alive=""
        for pid in $SOMNUS_PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                still_alive="$still_alive $pid"
            fi
        done
        if [ -z "$still_alive" ]; then
            echo "[stop] somnus exited cleanly."
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    if [ -n "$still_alive" ]; then
        echo "[stop] still running after ${WAIT_SECS}s:$still_alive — sending SIGKILL." >&2
        # shellcheck disable=SC2086
        kill -KILL $still_alive 2>/dev/null || true
    fi
fi

echo "[stop] done. Resume with scripts/resume_growth_campaign.sh"
