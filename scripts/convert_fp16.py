#!/usr/bin/env python3
"""
Convert ONNX encoder models from FP32 to FP16.

keep_io_types=True: inputs/outputs stay FP32 (matching our Rust f32 arrays);
only internal weights and activations are cast to FP16. MIGraphX then runs the
matmuls in FP16 while the Rust side never needs to change its data layout.
"""
import sys
import os
import onnx
from onnxconverter_common import float16

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Encoders that run on every audio/video frame — primary targets.
# Qwen models are throttled (gain_interval, action queries only) so lower priority.
TARGETS = [
    "dinov2_small.onnx",
    "clap_audio_encoder.onnx",
    "whisper_tiny_encoder.onnx",
    "efficientat_mn10_encoder.onnx",  # gaps.md item 9, A/B alternative to clap_audio_encoder
]

for name in TARGETS:
    src = os.path.join(MODELS_DIR, name)
    stem, ext = os.path.splitext(name)
    dst = os.path.join(MODELS_DIR, stem + "_fp16" + ext)

    if not os.path.exists(src):
        print(f"SKIP {name} — not found")
        continue

    src_mb = os.path.getsize(src) / 1024 / 1024
    print(f"Converting {name} ({src_mb:.0f} MB) ...", flush=True)

    model = onnx.load(src)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, dst)

    dst_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"  -> {os.path.basename(dst)} ({dst_mb:.0f} MB, {100*dst_mb/src_mb:.0f}% of original)")

print("\nDone. Test with a short smoke run before committing.")
