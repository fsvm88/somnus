#!/bin/bash
# A/B validation: Qwen2.5-0.5B (control) vs Qwen3-0.6B (treatment) — Item 5
# Base models (no LoRA) — architecture upgrade test
# Note: LoRA training had CPU compatibility issues; base model validates core architecture

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

SESSIONS="${SESSIONS:-5}"
CYCLES="${CYCLES:-2}"

echo "=== Qwen3-0.6B A/B Validation (Item 5) ==="
echo ""
echo "Setup: Control=Qwen2.5-0.5B (base), Treatment=Qwen3-0.6B (base)"
echo "Sessions: $SESSIONS × $CYCLES cycles each"
echo "Note: Base models (no LoRA) — validates architecture, not fine-tuning"
echo ""

if [ ! -f "$MODELS_DIR/qwen3_lm_base.onnx" ]; then
    echo "ERROR: qwen3_lm_base.onnx not found. Run export_models.py first."
    exit 1
fi

# Control: use qwen25_lm.onnx (base model, already in place)
echo "Running control (Qwen2.5-0.5B base)..."
"$REPO_ROOT/target/release/somnus" \
    --sessions "$SESSIONS" \
    --cycles "$CYCLES" \
    2>&1 | tail -50

# Treatment: temporarily replace with Qwen3 base model
echo ""
echo "Running treatment (Qwen3-0.6B base)..."
cp "$MODELS_DIR/qwen25_lm.onnx" "$MODELS_DIR/qwen25_lm.onnx.bak"
cp "$MODELS_DIR/qwen3_lm_base.onnx" "$MODELS_DIR/qwen25_lm.onnx"
"$REPO_ROOT/target/release/somnus" \
    --sessions "$SESSIONS" \
    --cycles "$CYCLES" \
    2>&1 | tail -50
mv "$MODELS_DIR/qwen25_lm.onnx.bak" "$MODELS_DIR/qwen25_lm.onnx"

echo ""
echo "=== Results ==="
echo "Control (Qwen2.5 base): See memory/*/metrics.log"
echo "Treatment (Qwen3 base): See memory/*/metrics.log"
echo ""
echo "✓ A/B complete. Compare output_mse, p_lang, and proj_loss across cycles."
echo "Note: This test validates the architecture upgrade; LoRA fine-tuning can follow if promising."
