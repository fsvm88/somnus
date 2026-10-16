#!/bin/bash
# HISTORICAL: predates the 2026-08-15 standing A/B instruction (build+run both arms with
# fine-metrics, record deviations). Results are recorded in ai-docs/; to re-run, add
# --features fine-metrics to the build first.
# A/B validation: DINOv2-S (control) vs DINOv3-ViT-S (treatment)
# Item 6 — drop-in encoder swap parity test

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

SESSIONS="${SESSIONS:-5}"
CYCLES="${CYCLES:-2}"

CONTROL_DIR="$REPO_ROOT/memory/dinov3_ab_control"
TREATMENT_DIR="$REPO_ROOT/memory/dinov3_ab_treatment"

echo "=== DINOv3-ViT-S A/B Validation (Item 6) ==="
echo ""
echo "Setup: Control=DINOv2-S, Treatment=DINOv3-ViT-S"
echo "Sessions: $SESSIONS × $CYCLES cycles each"
echo ""

if [ ! -f "$MODELS_DIR/dinov3_small.onnx" ]; then
    echo "ERROR: dinov3_small.onnx not found. Run: .venv/bin/python scripts/export_models.py"
    exit 1
fi

mkdir -p "$CONTROL_DIR" "$TREATMENT_DIR"

# VideoEncoder::new() prefers dinov3_small.onnx when present (crates/somnus-encoding/src/video.rs),
# so the control run temporarily hides it to force the DINOv2 fallback path.
echo "Running control (DINOv2-S)..."
mv "$MODELS_DIR/dinov3_small.onnx" "$MODELS_DIR/.dinov3_small.onnx.bak"
for i in $(seq 1 "$SESSIONS"); do
    "$REPO_ROOT/target/release/somnus" \
        --file-loop \
        --file clips/talking_head.mp4 \
        --file clips/urban.mp4 \
        --file clips/animation.mp4 \
        --file clips/nature.mp4 \
        --language \
        --action \
        --action-distill-only \
        --lang-use-lm \
        --seed "${SEED:-42}" \
        --memory-dir "$CONTROL_DIR" \
        --session "$i" \
        --max-sleep-cycles "$CYCLES" \
        --metrics-file "$CONTROL_DIR/metrics.jsonl" \
        2>&1 | tail -10
done
mv "$MODELS_DIR/.dinov3_small.onnx.bak" "$MODELS_DIR/dinov3_small.onnx"

echo ""
echo "Running treatment (DINOv3-ViT-S)..."
for i in $(seq 1 "$SESSIONS"); do
    "$REPO_ROOT/target/release/somnus" \
        --file-loop \
        --file clips/talking_head.mp4 \
        --file clips/urban.mp4 \
        --file clips/animation.mp4 \
        --file clips/nature.mp4 \
        --language \
        --action \
        --action-distill-only \
        --lang-use-lm \
        --seed "${SEED:-42}" \
        --memory-dir "$TREATMENT_DIR" \
        --session "$i" \
        --max-sleep-cycles "$CYCLES" \
        --metrics-file "$TREATMENT_DIR/metrics.jsonl" \
        2>&1 | tail -10
done

echo ""
echo "=== Results ==="
echo "Control (DINOv2-S):   $CONTROL_DIR/metrics.jsonl"
echo "Treatment (DINOv3-S): $TREATMENT_DIR/metrics.jsonl"
echo ""
echo "✓ A/B complete. Compare warm_recall_l2, mean_output_mse across cycles."
