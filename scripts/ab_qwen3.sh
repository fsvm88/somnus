#!/bin/bash
# A/B validation: Qwen2.5-0.5B (control) vs Qwen3-0.6B (treatment) — Item 5
# LoRA-trained models for full quality comparison

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

SESSIONS="${SESSIONS:-5}"
CYCLES="${CYCLES:-2}"

echo "=== Qwen3-0.6B A/B Validation (Item 5) ==="
echo ""
echo "Setup: Control=Qwen2.5-0.5B (LoRA), Treatment=Qwen3-0.6B (LoRA)"
echo "Sessions: $SESSIONS × $CYCLES cycles each"
echo ""

if [ ! -f "$MODELS_DIR/qwen3_lm.onnx" ]; then
    echo "ERROR: qwen3_lm.onnx (LoRA-trained) not found."
    echo "Complete LoRA training first: python3 scripts/train_lora_qwen3.py --export-onnx models/qwen3_lm.onnx"
    exit 1
fi

# Control: use qwen25_lm.onnx (LoRA-trained, already in place)
echo "Running control (Qwen2.5-0.5B with LoRA)..."
"$REPO_ROOT/target/release/somnus" \
    --sessions "$SESSIONS" \
    --cycles "$CYCLES" \
    2>&1 | tail -50

# Treatment: temporarily replace with Qwen3 (LoRA-trained)
echo ""
echo "Running treatment (Qwen3-0.6B with LoRA)..."
cp "$MODELS_DIR/qwen25_lm.onnx" "$MODELS_DIR/qwen25_lm.onnx.bak"
cp "$MODELS_DIR/qwen3_lm.onnx" "$MODELS_DIR/qwen25_lm.onnx"
"$REPO_ROOT/target/release/somnus" \
    --sessions "$SESSIONS" \
    --cycles "$CYCLES" \
    2>&1 | tail -50
mv "$MODELS_DIR/qwen25_lm.onnx.bak" "$MODELS_DIR/qwen25_lm.onnx"

echo ""
echo "=== Results ==="
echo "Control (Qwen2.5+LoRA):  See memory/*/metrics.log"
echo "Treatment (Qwen3+LoRA): See memory/*/metrics.log"
echo ""
echo "✓ A/B complete. Compare output_mse, p_lang, and proj_loss across cycles."
