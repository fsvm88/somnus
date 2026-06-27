#!/bin/bash
# A/B validation: DINOv2-S (control) vs DINOv3-ViT-S (treatment)
# Item 6 — drop-in encoder swap parity test

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

SESSIONS="${SESSIONS:-5}"
CYCLES="${CYCLES:-2}"

echo "=== DINOv3-ViT-S A/B Validation ==="
echo ""
echo "Setup: Control=DINOv2-S, Treatment=DINOv3-ViT-S"
echo "Sessions: $SESSIONS × $CYCLES cycles"
echo ""

if [ ! -f "$MODELS_DIR/dinov3_small.onnx" ]; then
    echo "ERROR: dinov3_small.onnx not found. Run: .venv/bin/python scripts/export_models.py"
    exit 1
fi

# Control: temporarily hide dinov3
echo "Running control (DINOv2-S)..."
mv "$MODELS_DIR/dinov3_small.onnx" "$MODELS_DIR/.dinov3_small.onnx.bak"
"$REPO_ROOT/target/release/somnus" \
    --sessions "$SESSIONS" \
    --cycles "$CYCLES" \
    2>&1 | tail -50

# Restore dinov3 and hide dinov2
mv "$MODELS_DIR/.dinov3_small.onnx.bak" "$MODELS_DIR/dinov3_small.onnx"
mv "$MODELS_DIR/dinov2_small.onnx" "$MODELS_DIR/.dinov2_small.onnx.bak"

echo ""
echo "Running treatment (DINOv3-ViT-S)..."
"$REPO_ROOT/target/release/somnus" \
    --sessions "$SESSIONS" \
    --cycles "$CYCLES" \
    2>&1 | tail -50

# Restore dinov2
mv "$MODELS_DIR/.dinov2_small.onnx.bak" "$MODELS_DIR/dinov2_small.onnx"

echo ""
echo "✓ Complete — compare memory/{control,treatment} logs for parity"
