#!/bin/bash
# Generate Qwen3 projection weights by running a 3-session campaign.
# Exports projection weights during the campaign.

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
# Use separate memory for Qwen3 (incompatible dims with Qwen2.5)
MEMORY_DIR="$REPO_ROOT/memory/qwen3_temp"
PROJ_WEIGHTS_DIR="$REPO_ROOT/memory/proj_weights_qwen3"

echo "=== Generating Qwen3 Projection Weights ==="
echo "Using temporary memory: $MEMORY_DIR"
echo ""

# Clean slate for Qwen3
rm -rf "$MEMORY_DIR"
mkdir -p "$MEMORY_DIR"

if [ ! -f "$MODELS_DIR/qwen3_lm_base.onnx" ]; then
    echo "ERROR: qwen3_lm_base.onnx not found"
    exit 1
fi

# Backup Qwen2.5 model
echo "Backing up Qwen2.5 model…"
cp "$MODELS_DIR/qwen25_lm.onnx" "$MODELS_DIR/qwen25_lm.onnx.bak"
[ -f "$MODELS_DIR/qwen25_lm.onnx.data" ] && cp "$MODELS_DIR/qwen25_lm.onnx.data" "$MODELS_DIR/qwen25_lm.onnx.data.bak" || true

# Swap to Qwen3
echo "Swapping to Qwen3…"
cp "$MODELS_DIR/qwen3_lm_base.onnx" "$MODELS_DIR/qwen25_lm.onnx"
[ -f "$MODELS_DIR/qwen3_lm_base.onnx.data" ] && cp "$MODELS_DIR/qwen3_lm_base.onnx.data" "$MODELS_DIR/qwen25_lm.onnx.data" || true

# Create projection weights directory
mkdir -p "$PROJ_WEIGHTS_DIR"

# Run 3-session campaign with projection weight export
echo ""
echo "Running 3-session campaign with Qwen3…"
SESSIONS=3 \
CYCLES=2 \
CAPACITY=2048 \
MEMORY_DIR="$MEMORY_DIR" \
EXTRA_ARGS="--export-proj-weights $PROJ_WEIGHTS_DIR" \
  "$REPO_ROOT/scripts/train_campaign.sh"

# Restore Qwen2.5
echo ""
echo "Restoring Qwen2.5…"
mv "$MODELS_DIR/qwen25_lm.onnx.bak" "$MODELS_DIR/qwen25_lm.onnx"
[ -f "$MODELS_DIR/qwen25_lm.onnx.data.bak" ] && mv "$MODELS_DIR/qwen25_lm.onnx.data.bak" "$MODELS_DIR/qwen25_lm.onnx.data" || true

echo ""
echo "✓ Projection weights generated:"
ls -lh "$PROJ_WEIGHTS_DIR"/
echo ""
echo "Next: Train LoRA with these weights:"
echo "  python3 scripts/train_lora_qwen3.py \\"
echo "    --training-data memory/lora_training_data.jsonl \\"
echo "    --proj-weights-dir $PROJ_WEIGHTS_DIR \\"
echo "    --output-dir memory/lora_output_qwen3"
