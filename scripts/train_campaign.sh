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
#   SESSIONS         Number of sessions to run  (default: 10)
#   CYCLES           Sleep cycles per session   (default: 3)
#   CAPACITY         Episodic buffer size; sleep triggers at 80% (default: 8192)
#                    Set small for fast validation, e.g. CAPACITY=64 → ~51 episodes/cycle
#   CLIPS            Space-separated clip paths (default: all four in clips/)
#   CAMPAIGN_DIR     Directory for metrics/logs (default: memory/campaign_<timestamp>)
#   MEMORY_DIR       Shared memory directory    (default: memory)
#   ONNX_MODELS_DIR  Enable ORT encoder+binding via this dir (default: disabled)
#                    Writes/reads perception_encoder.onnx + perception_binding.onnx here.
#                    Do NOT also pass --onnx-models-dir in EXTRA_ARGS.
#   ONNX_SCRIPT      Override path to export_perception_graph.py (optional)
#   EXTRA_ARGS       Extra args passed to somnus   (default: none)
#
# Example:
#   SESSIONS=20 CYCLES=3 scripts/train_campaign.sh
#   SESSIONS=5 EXTRA_ARGS="--prune-fraction 0.5" scripts/train_campaign.sh
#   SESSIONS=10 ONNX_MODELS_DIR=memory/onnx_models scripts/train_campaign.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Parameters ────────────────────────────────────────────────────────────────
SESSIONS="${SESSIONS:-10}"
SESSION_START="${SESSION_START:-1}"    # first session number (for continuations)
CYCLES="${CYCLES:-3}"
CAPACITY="${CAPACITY:-8192}"
MEMORY_DIR="${MEMORY_DIR:-memory}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-${MEMORY_DIR}/campaign_$(date +%Y%m%d_%H%M%S)}"
CLIPS="${CLIPS:-clips/talking_head.mp4 clips/urban.mp4 clips/animation.mp4 clips/nature.mp4}"
# Optional text corpus for language injection via inject_text.py.
# Space-separated glob of .txt files. Leave empty to disable.
TEXT_CORPUS="${TEXT_CORPUS:-}"
TEXT_RATE="${TEXT_RATE:-5}"      # chunks per second fed to --language stdin
# Small non-zero base so prior cosines are always measured; S4 scales up to 0.05
# when predictions mature (ADR 0024). Set higher to override S4.
PRIOR_BETA="${PRIOR_BETA:-0.01}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# When set, enables ORT encoder+binding sessions (Phase 1 + 1.5).
# Maps to --onnx-models-dir; writes ONNX graphs relative to MEMORY_DIR by default.
ONNX_MODELS_DIR="${ONNX_MODELS_DIR:-}"
ONNX_SCRIPT="${ONNX_SCRIPT:-}"       # override path to export_perception_graph.py

METRICS_FILE="${CAMPAIGN_DIR}/metrics.jsonl"
LOG_DIR="${CAMPAIGN_DIR}/logs"

mkdir -p "$CAMPAIGN_DIR" "$LOG_DIR"

# ── Build once ────────────────────────────────────────────────────────────────
echo "[campaign] building somnus (release)…"
cargo build --release -p somnus 2>&1
SOMNUS="$REPO_ROOT/target/release/somnus"

# ── Record campaign parameters ────────────────────────────────────────────────
cat >> "${CAMPAIGN_DIR}/params.txt" <<EOF
SESSIONS=${SESSIONS}
SESSION_START=${SESSION_START}
CYCLES=${CYCLES}
CAPACITY=${CAPACITY}
CLIPS=${CLIPS}
MEMORY_DIR=${MEMORY_DIR}
ONNX_MODELS_DIR=${ONNX_MODELS_DIR}
ONNX_SCRIPT=${ONNX_SCRIPT}
PRIOR_BETA=${PRIOR_BETA}
EXTRA_ARGS=${EXTRA_ARGS}
started=$(date -Iseconds)
EOF

SESSION_END=$((SESSION_START + SESSIONS - 1))
echo "[campaign] starting: sessions ${SESSION_START}–${SESSION_END} × ${CYCLES} cycles  (capacity=${CAPACITY})"
echo "[campaign] metrics → ${METRICS_FILE}"
echo "[campaign] logs    → ${LOG_DIR}/"
echo ""

# Build --file flags from CLIPS
FILE_FLAGS=""
for clip in $CLIPS; do
    FILE_FLAGS="$FILE_FLAGS --file $clip"
done

# Build ORT flags from ONNX_MODELS_DIR / ONNX_SCRIPT.
# Guard against duplicate flags if user also passes them in EXTRA_ARGS.
ONNX_DIR_FLAG=""
if [ -n "$ONNX_MODELS_DIR" ] && [[ "$EXTRA_ARGS" != *"--onnx-models-dir"* ]]; then
    ONNX_DIR_FLAG="--onnx-models-dir ${ONNX_MODELS_DIR}"
    if [ -n "$ONNX_SCRIPT" ] && [[ "$EXTRA_ARGS" != *"--onnx-script"* ]]; then
        ONNX_DIR_FLAG="${ONNX_DIR_FLAG} --onnx-script ${ONNX_SCRIPT}"
    fi
fi

# ── Session loop ──────────────────────────────────────────────────────────────
for i in $(seq "$SESSION_START" "$SESSION_END"); do
    SESSION_LOG="${LOG_DIR}/session_${i}.log"
    echo "[campaign] ── session ${i}/${SESSION_END} ──────────────────────────────"

    # shellcheck disable=SC2086
    # Capture exit code via || to prevent set -e from triggering on expected
    # non-zero exits (e.g. exit 139 = known MIGraphX destructor race on gfx1201).
    EXIT=0
    if [ -n "$TEXT_CORPUS" ]; then
        # Pipe text corpus into somnus's stdin as language ingest events.
        # inject_text.py loops the corpus indefinitely; killed when somnus exits.
        # shellcheck disable=SC2086
        python3 "$REPO_ROOT/scripts/inject_text.py" $TEXT_CORPUS \
            --rate "$TEXT_RATE" --shuffle \
            2>>"$SESSION_LOG" \
        | "$SOMNUS" \
            $FILE_FLAGS \
            --file-loop \
            --memory-dir "$MEMORY_DIR" \
            --language \
            --action \
            --action-distill-only \
            --lang-use-lm \
            --prior-beta "$PRIOR_BETA" \
            --settle-steps 1 \
            --working-memory-size 8 \
            --episodic-capacity "$CAPACITY" \
            --max-sleep-cycles "$CYCLES" \
            --replay-batch-size 128 \
            --wake-reconstruct-interval 5 \
            --video-stride 2 \
            --recall-ef 16 \
            --max-warm-records 200000 \
            --metrics-file "$METRICS_FILE" \
            --log-file "$SESSION_LOG" \
            --session "$i" \
            $ONNX_DIR_FLAG \
            $EXTRA_ARGS \
            2>>"$SESSION_LOG" || EXIT=$?
    else
        "$SOMNUS" \
            $FILE_FLAGS \
            --file-loop \
            --memory-dir "$MEMORY_DIR" \
            --language \
            --action \
            --action-distill-only \
            --lang-use-lm \
            --prior-beta "$PRIOR_BETA" \
            --settle-steps 1 \
            --working-memory-size 8 \
            --episodic-capacity "$CAPACITY" \
            --max-sleep-cycles "$CYCLES" \
            --replay-batch-size 128 \
            --wake-reconstruct-interval 5 \
            --video-stride 2 \
            --recall-ef 16 \
            --max-warm-records 200000 \
            --metrics-file "$METRICS_FILE" \
            --log-file "$SESSION_LOG" \
            --session "$i" \
            $ONNX_DIR_FLAG \
            $EXTRA_ARGS \
            2>>"$SESSION_LOG" || EXIT=$?
    fi
    # Exit code 139 (SIGSEGV) is a known MIGraphX destructor race on gfx1201/RDNA4:
    # GPU encoder threads may be mid-kernel when process::exit(0) tears down the process.
    # All work is written to disk before this crash; treat it as success.
    # 139 = SIGSEGV from MIGraphX destructor race (known, safe to ignore).
    # 120 = pipefail picks up inject_text.py's SIGPIPE exit when somnus exits cleanly.
    if [ $EXIT -ne 0 ] && [ $EXIT -ne 139 ] && [ $EXIT -ne 120 ]; then
        echo "[campaign] session ${i} exited with code ${EXIT} — check ${SESSION_LOG}"
        echo "[campaign] stopping campaign."
        exit $EXIT
    fi

    echo "[campaign] session ${i} complete."
done

# ── Post-campaign eval ────────────────────────────────────────────────────────
echo ""
echo "[campaign] sessions ${SESSION_START}–${SESSION_END} complete."
echo "[campaign] running eval harness…"

python3 scripts/eval_metrics.py \
    "$METRICS_FILE" \
    --out-dir "$CAMPAIGN_DIR"

echo "[campaign] charts written to ${CAMPAIGN_DIR}/"
echo "  language_trajectory.png"
echo "  growth_timeline.png"
echo "  per_region_mse.png"
echo "  sleep_triggers.png"
echo "  timing_breakdown.png"
echo "  wake_timing.png"
