#!/usr/bin/env bash
# Continuous training campaign: run N sequential sessions, accumulating memory
# across all of them. Each session runs for CYCLES sleep cycles then exits.
# Metrics are appended to a single NDJSON file; eval_metrics.py can visualise
# the full trajectory once the campaign completes (or at any point mid-run).
#
# Usage:
#   scripts/train_campaign.sh [OPTIONS]
#
# Options (environment variables):
#   SESSIONS      Number of sessions to run  (default: 10)
#   CYCLES        Sleep cycles per session   (default: 3)
#   CAPACITY      Episodic buffer size; sleep triggers at 80% (default: 4096)
#                 Set small for fast validation, e.g. CAPACITY=64 → ~51 episodes/cycle
#   CLIPS         Space-separated clip paths (default: all four in clips/)
#   CAMPAIGN_DIR  Directory for metrics/logs (default: memory/campaign_<timestamp>)
#   MEMORY_DIR    Shared memory directory    (default: memory)
#   EXTRA_ARGS    Extra args passed to somnus   (default: none)
#
# Example:
#   SESSIONS=20 CYCLES=3 scripts/train_campaign.sh
#   SESSIONS=5 EXTRA_ARGS="--prune-fraction 0.5" scripts/train_campaign.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Parameters ────────────────────────────────────────────────────────────────
SESSIONS="${SESSIONS:-10}"
CYCLES="${CYCLES:-3}"
CAPACITY="${CAPACITY:-4096}"
MEMORY_DIR="${MEMORY_DIR:-memory}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-${MEMORY_DIR}/campaign_$(date +%Y%m%d_%H%M%S)}"
CLIPS="${CLIPS:-clips/talking_head.mp4 clips/urban.mp4 clips/animation.mp4 clips/nature.mp4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

METRICS_FILE="${CAMPAIGN_DIR}/metrics.jsonl"
LOG_DIR="${CAMPAIGN_DIR}/logs"

mkdir -p "$CAMPAIGN_DIR" "$LOG_DIR"

# ── Build once ────────────────────────────────────────────────────────────────
echo "[campaign] building somnus (release)…"
cargo build --release -p somnus 2>&1
SOMNUS="$REPO_ROOT/target/release/somnus"

# ── Record campaign parameters ────────────────────────────────────────────────
cat > "${CAMPAIGN_DIR}/params.txt" <<EOF
SESSIONS=${SESSIONS}
CYCLES=${CYCLES}
CAPACITY=${CAPACITY}
CLIPS=${CLIPS}
MEMORY_DIR=${MEMORY_DIR}
EXTRA_ARGS=${EXTRA_ARGS}
started=$(date -Iseconds)
EOF

echo "[campaign] starting: ${SESSIONS} sessions × ${CYCLES} cycles  (capacity=${CAPACITY})"
echo "[campaign] metrics → ${METRICS_FILE}"
echo "[campaign] logs    → ${LOG_DIR}/"
echo ""

# Build --file flags from CLIPS
FILE_FLAGS=""
for clip in $CLIPS; do
    FILE_FLAGS="$FILE_FLAGS --file $clip"
done

# ── Session loop ──────────────────────────────────────────────────────────────
for i in $(seq 1 "$SESSIONS"); do
    SESSION_LOG="${LOG_DIR}/session_${i}.log"
    echo "[campaign] ── session ${i}/${SESSIONS} ──────────────────────────────"

    # shellcheck disable=SC2086
    "$SOMNUS" \
        $FILE_FLAGS \
        --file-loop \
        --memory-dir "$MEMORY_DIR" \
        --language \
        --action \
        --prior-beta 0.1 \
        --settle-steps 3 \
        --working-memory-size 8 \
        --episodic-capacity "$CAPACITY" \
        --max-sleep-cycles "$CYCLES" \
        --replay-batch-size 64 \
        --metrics-file "$METRICS_FILE" \
        --log-file "$SESSION_LOG" \
        --session "$i" \
        $EXTRA_ARGS \
        2>>"$SESSION_LOG"

    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "[campaign] session ${i} exited with code ${EXIT} — check ${SESSION_LOG}"
        echo "[campaign] stopping campaign."
        exit $EXIT
    fi

    echo "[campaign] session ${i} complete."
done

# ── Post-campaign eval ────────────────────────────────────────────────────────
echo ""
echo "[campaign] all ${SESSIONS} sessions complete."
echo "[campaign] running eval harness…"

python3 scripts/eval_metrics.py \
    "$METRICS_FILE" \
    --out-dir "$CAMPAIGN_DIR"

echo "[campaign] charts written to ${CAMPAIGN_DIR}/"
echo "  language_trajectory.png"
echo "  growth_timeline.png"
echo "  per_region_mse.png"
echo "  sleep_triggers.png"
