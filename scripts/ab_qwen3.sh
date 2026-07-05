#!/bin/bash
# A/B validation: Qwen2.5-0.5B (control) vs Qwen3-0.6B (treatment) — Item 5, redo
#
# Replaces the 2026-06-28 run, which was invalid: LM_HIDDEN was a single global
# constant (bumped to 1024 for Qwen3) while qwen25_lm.onnx was still Qwen2.5's real
# 896-wide table, so embed_flat() sliced it at the wrong stride for every token in
# both arms. On top of that, the treatment side never got a matching GPU-path export
# and instead swapped qwen3_lm_base.onnx in over qwen25_lm.onnx by filename, whose
# "restore" step silently failed and left qwen25_lm.onnx replaced by Qwen3 weights.
# See ai-docs/a_b_qwen3_findings.md and .claude/session-state.md for the full story.
#
# The redo uses two separate release binaries (LM_HIDDEN is a compile-time choice —
# see the `qwen3` Cargo feature) against two separate, correctly-exported model file
# sets, so there is no file-swapping and no way for the arms to cross-contaminate.

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

# The original A/B used 2 NREM cycles/session — likely too short to judge a bigger
# model fairly. Default to a longer cold-start here.
SESSIONS="${SESSIONS:-10}"
CYCLES="${CYCLES:-5}"

CONTROL_BIN="$REPO_ROOT/target/release/somnus-qwen25-control"
TREATMENT_BIN="$REPO_ROOT/target/release/somnus-qwen3-treatment"
CONTROL_DIR="$REPO_ROOT/memory/qwen25_ab_control"
TREATMENT_DIR="$REPO_ROOT/memory/qwen3_ab_treatment"

echo "=== Qwen3-0.6B A/B Validation redo (Item 5) ==="
echo ""
echo "Setup: Control=Qwen2.5-0.5B (qwen25_*, LM_HIDDEN=896), Treatment=Qwen3-0.6B (qwen3_*, LM_HIDDEN=1024)"
echo "Sessions: $SESSIONS × $CYCLES cycles each"
echo ""

for f in "$CONTROL_BIN" "$TREATMENT_BIN"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Build both arms first:"
        echo "  cargo build --release -p somnus --bin somnus && cp target/release/somnus $CONTROL_BIN"
        echo "  cargo build --release -p somnus --bin somnus --features qwen3 && cp target/release/somnus $TREATMENT_BIN"
        exit 1
    fi
done
for f in qwen25_gain.onnx qwen25_lm.onnx qwen25_embed_table.safetensors qwen25_tokenizer.json \
         qwen3_gain.onnx qwen3_lm.onnx qwen3_embed_table.safetensors qwen3_tokenizer.json; do
    if [ ! -f "$MODELS_DIR/$f" ]; then
        echo "ERROR: $MODELS_DIR/$f not found. Run: scripts/.venv/bin/python scripts/export_models.py"
        exit 1
    fi
done

mkdir -p "$CONTROL_DIR" "$TREATMENT_DIR"

echo "Running control (Qwen2.5-0.5B, cold start)..."
for i in $(seq 1 "$SESSIONS"); do
    "$CONTROL_BIN" \
        --file-loop \
        --file clips/talking_head.mp4 \
        --file clips/urban.mp4 \
        --file clips/animation.mp4 \
        --file clips/nature.mp4 \
        --language \
        --action \
        --action-distill-only \
        --lang-use-lm \
        --action-model-prefix qwen25 \
        --memory-dir "$CONTROL_DIR" \
        --session "$i" \
        --max-sleep-cycles "$CYCLES" \
        --metrics-file "$CONTROL_DIR/metrics.jsonl" \
        2>&1 | tail -10
done

echo ""
echo "Running treatment (Qwen3-0.6B, cold start)..."
for i in $(seq 1 "$SESSIONS"); do
    "$TREATMENT_BIN" \
        --file-loop \
        --file clips/talking_head.mp4 \
        --file clips/urban.mp4 \
        --file clips/animation.mp4 \
        --file clips/nature.mp4 \
        --language \
        --action \
        --action-distill-only \
        --lang-use-lm \
        --action-model-prefix qwen3 \
        --memory-dir "$TREATMENT_DIR" \
        --session "$i" \
        --max-sleep-cycles "$CYCLES" \
        --metrics-file "$TREATMENT_DIR/metrics.jsonl" \
        2>&1 | tail -10
done

echo ""
echo "=== Results ==="
echo "Control (Qwen2.5-0.5B):  $CONTROL_DIR/metrics.jsonl"
echo "Treatment (Qwen3-0.6B):  $TREATMENT_DIR/metrics.jsonl"
echo ""
echo "Compare mean_proj_loss, output_mse, p_lang across cycles (scripts/eval_metrics.py)."
