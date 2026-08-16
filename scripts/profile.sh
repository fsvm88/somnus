#!/usr/bin/env bash
# Profile a short somnus run using all three tools:
#   1. ORT Chrome-trace  — per-op timeline for Qwen LM (CPU) session
#   2. rocprof           — GPU kernel timeline for all MIGraphX sessions
#   3. cargo flamegraph  — CPU call-stack flamegraph via perf
#
# Usage:
#   scripts/profile.sh [OUTPUT_DIR]
#
# Environment:
#   CLIPS      space-separated clip paths (default: all four clips/)
#   CAPACITY   episodic buffer size (default: 64 — fast sleep trigger)
#
# Output:
#   <OUTPUT_DIR>/ort/qwen25_lm_*.json   ORT op timeline (Perfetto / chrome://tracing)
#   <OUTPUT_DIR>/rocprof/               GPU kernel CSV + JSON trace
#   <OUTPUT_DIR>/flamegraph.svg         CPU flamegraph (open in browser)
#   <OUTPUT_DIR>/metrics.jsonl          sleep-cycle metrics with wake timing
#
# What these profiles characterise:
#   All three runs are sleep-dominated (CAPACITY=64 triggers sleep quickly).
#   Wake is clock-bound by media file pacing — GPU and CPU are mostly idle
#   during wake. The profiles show the SLEEP CYCLE cost breakdown, which is
#   where compute actually concentrates.
#
# Reading the results:
#   ORT trace   → which Qwen LM ops dominate distillation precompute?
#   rocprof     → compare "total GPU kernel time" vs session wall time for
#                 CPU/GPU split; look for long gaps = CPU-bound stalls
#   flamegraph  → which Rust functions consume CPU cycles (burn matmul,
#                 HNSW insert/query, serde, homeostatic_decay, …)?

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${1:-profile_$(date +%Y%m%d_%H%M%S)}"
CAPACITY="${CAPACITY:-64}"
CLIPS="${CLIPS:-clips/talking_head.mp4 clips/urban.mp4 clips/animation.mp4 clips/nature.mp4}"

mkdir -p "$OUTPUT_DIR"
echo "[profile] output directory: $OUTPUT_DIR"
echo ""

# ── Build ──────────────────────────────────────────────────────────────────────
echo "[profile] building somnus (release)…"
cargo build --release -p somnus 2>&1
echo "[profile] building somnus (release-debug, for flamegraph)…"
cargo build --profile release-debug -p somnus 2>&1

SOMNUS_REL="$REPO_ROOT/target/release/somnus"
SOMNUS_DBG="$REPO_ROOT/target/release-debug/somnus"

# Shared somnus flags. Language + action + working memory all on by default
# (ADR 0047) — the Qwen LM path fires without needing to pass them explicitly.
FILE_FLAGS=""
for clip in $CLIPS; do FILE_FLAGS="$FILE_FLAGS --file $clip"; done

COMMON_ARGS="$FILE_FLAGS --file-loop \
  --episodic-capacity $CAPACITY \
  --max-sleep-cycles 1 \
  --lang-oversample 9 \
  --asr-min-chars 5"

# ── Pass 1: ORT CPU session profiling ─────────────────────────────────────────
echo ""
echo "=== Pass 1/3: ORT CPU session profile (Qwen LM) ==="
mkdir -p "$OUTPUT_DIR/ort" "$OUTPUT_DIR/mem_ort"
EXIT=0
# shellcheck disable=SC2086
"$SOMNUS_REL" $COMMON_ARGS \
  --ort-profile-dir "$OUTPUT_DIR/ort" \
  --memory-dir "$OUTPUT_DIR/mem_ort" \
  --metrics-file "$OUTPUT_DIR/metrics.jsonl" \
  --log-file "$OUTPUT_DIR/ort/session.log" \
  || EXIT=$?
# exit 139 = known MIGraphX gfx1201 destructor SIGSEGV; treated as success.
if [ $EXIT -ne 0 ] && [ $EXIT -ne 139 ]; then
  echo "[profile] ORT pass exited with $EXIT — check $OUTPUT_DIR/ort/session.log"
fi
ORT_FILES=$(ls "$OUTPUT_DIR/ort"/*.json 2>/dev/null | grep -v session.log | head -5)
if [ -n "$ORT_FILES" ]; then
  echo "  ORT traces written:"
  for f in $ORT_FILES; do echo "    $f"; done
  echo "  View at: https://ui.perfetto.dev  or  chrome://tracing (load JSON)"
else
  echo "  No ORT trace files found. Was --action used? Does the session fire before sleep?"
fi

# ── Pass 2: rocprof GPU kernel trace ─────────────────────────────────────────
echo ""
echo "=== Pass 2/3: rocprof GPU kernel trace ==="
ROCPROF=/opt/rocm/bin/rocprof
if [ ! -x "$ROCPROF" ]; then
  echo "  SKIP: rocprof not found at $ROCPROF"
else
  mkdir -p "$OUTPUT_DIR/rocprof" "$OUTPUT_DIR/mem_rocprof"
  EXIT=0
  # shellcheck disable=SC2086
  "$ROCPROF" \
    --sys-trace \
    -o "$OUTPUT_DIR/rocprof/profile.csv" \
    -- "$SOMNUS_REL" $COMMON_ARGS \
      --memory-dir "$OUTPUT_DIR/mem_rocprof" \
      --log-file "$OUTPUT_DIR/rocprof/session.log" \
    || EXIT=$?
  if [ $EXIT -ne 0 ] && [ $EXIT -ne 139 ]; then
    echo "[profile] rocprof pass exited with $EXIT — check $OUTPUT_DIR/rocprof/session.log"
  fi
  echo "  GPU kernel CSV + JSON written to $OUTPUT_DIR/rocprof/"
  echo "  Files:"
  ls "$OUTPUT_DIR/rocprof"/ | grep -v session.log | sed 's/^/    /'
  echo "  View JSON trace at: https://ui.perfetto.dev  or  chrome://tracing"
  echo "  Tip: compare sum(DurationNs) in profile.csv vs nrem_wall_ms in metrics.jsonl"
  echo "       to estimate the GPU-vs-CPU split."
fi

# ── Pass 3: CPU flamegraph ────────────────────────────────────────────────────
echo ""
echo "=== Pass 3/3: CPU flamegraph (perf + cargo-flamegraph) ==="

# Flamegraph needs perf_event_paranoid ≤ 1 (typically requires root or sysctl).
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
if [ "$PARANOID" = "unknown" ] || [ "$PARANOID" -gt 1 ] 2>/dev/null; then
  echo "  WARNING: kernel.perf_event_paranoid=$PARANOID — flamegraph may fail or be empty."
  echo "  Fix: sudo sysctl -w kernel.perf_event_paranoid=1"
  echo "       (or add 'kernel.perf_event_paranoid=1' to /etc/sysctl.conf for persistence)"
fi

if ! command -v cargo-flamegraph &>/dev/null; then
  echo "  SKIP: cargo-flamegraph not installed. Run: cargo install flamegraph"
else
  mkdir -p "$OUTPUT_DIR/mem_flame"
  EXIT=0
  # shellcheck disable=SC2086
  cargo flamegraph \
    --profile release-debug \
    --output "$OUTPUT_DIR/flamegraph.svg" \
    --bin somnus \
    -- $COMMON_ARGS \
      --memory-dir "$OUTPUT_DIR/mem_flame" \
      --log-file "$OUTPUT_DIR/flamegraph.log" \
    || EXIT=$?
  if [ $EXIT -ne 0 ] && [ $EXIT -ne 139 ]; then
    echo "[profile] flamegraph exited with $EXIT — check $OUTPUT_DIR/flamegraph.log"
  fi
  if [ -f "$OUTPUT_DIR/flamegraph.svg" ]; then
    echo "  Flamegraph: $OUTPUT_DIR/flamegraph.svg (open in browser)"
    echo "  Tip: search for 'ort' to isolate ONNX runtime frames; 'burn' for tensor math;"
    echo "       'hnsw' for warm-memory queries; 'homeostatic' for decay."
  fi
fi

# ── Cleanup temp memory dirs ──────────────────────────────────────────────────
rm -rf \
  "$OUTPUT_DIR/mem_ort" \
  "$OUTPUT_DIR/mem_rocprof" \
  "$OUTPUT_DIR/mem_flame" \
  2>/dev/null || true

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Profile run complete: $OUTPUT_DIR/ ==="
echo ""
echo "  ort/qwen25_lm_*.json  — ORT op-level timeline for Qwen LM session"
echo "  rocprof/              — GPU kernel timing for all MIGraphX models"
echo "  flamegraph.svg        — CPU call-stack (burn, HNSW, serde, tokio, …)"
echo "  metrics.jsonl         — sleep cycle metrics incl. wake-phase encode times"
echo ""
echo "  Interpreting CPU vs GPU split:"
echo "    1. Open metrics.jsonl: nrem_wall_ms is total sleep wall time."
echo "    2. In rocprof CSV: sum DurationNs for kernels in the sleep window."
echo "    3. Ratio = GPU_kernel_ns / (nrem_wall_ms * 1e6) — fraction GPU-bound."
echo "    4. Complement is CPU-bound; flamegraph shows which functions drive it."
