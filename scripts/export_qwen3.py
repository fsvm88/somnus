#!/usr/bin/env python3
"""Export Qwen3-0.6B ONNX models for somnus (LM forward + KV-cache)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print("=" * 60)
print("Exporting Qwen3-0.6B ONNX Models")
print("=" * 60)

model_id = "Qwen/Qwen3-0.6B"
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

print(f"\n[1/2] Loading model and tokenizer from {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)
print(f"✓ Model loaded. Hidden size: {model.config.hidden_size}")

# Export base model (for language region ingest)
print(f"\n[2/2] Exporting ONNX models...")
print("      Note: Full KV-cache export requires separate script (export_kv_cache.py)")

# Create a wrapper that exports the base model without LM head
class Qwen3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True
        )
        # Return last hidden state (for embedding generation)
        return outputs.hidden_states[-1]

wrapper = Qwen3Wrapper(model)

# Export with dynamic batch size and sequence length
print("  Exporting qwen3_lm_base.onnx (no LM head, dynamic seq)...")
try:
    torch.onnx.export(
        wrapper,
        (torch.randint(0, tokenizer.vocab_size, (1, 10)),),
        f"{output_dir}/qwen3_lm_base.onnx",
        input_names=["input_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "hidden_states": {0: "batch", 1: "sequence"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print(f"  ✓ Saved: {output_dir}/qwen3_lm_base.onnx")
except Exception as e:
    print(f"  ✗ Export failed: {e}")

print("\n" + "=" * 60)
print("Next steps:")
print("  1. Run: python3 scripts/export_kv_cache.py --model-id Qwen/Qwen3-0.6B --kv-size 2048")
print("  2. Re-run LoRA training: python3 scripts/train_lora.py")
print("  3. Re-quantize KV models: python3 scripts/quantize_kv.py --prefill")
print("=" * 60)
