# ML Runtimes

> **Contents:** Framework comparison — ort (inference, ✓ committed), burn (training, ✓ committed/NdArray backend), candle (not adopted), tch-rs (not adopted). Latency estimates, memory models, recommended split. Note: candle recommendation in this file was superseded; see updated "Recommended runtime architecture" section.
> **When to read:** Evaluating a new ML framework; understanding why burn was chosen over candle; checking ORT latency estimates.

Research note covering the tensor+inference+training runtime layer in Rust: `ort`, `candle`, `burn`, `tch-rs`. All numbers are **engineering estimates**; depend on GPU/driver/model topology. See `tech-stack.md` for the current candidate list.

---

## What we need from the runtime

| Mode | Operation | Requirement |
|---|---|---|
| Wake (online) | Frozen encoder inference | Low latency (<30ms/frame), no autograd |
| Wake (online) | Cognitive-layer forward pass | Low latency, minimal autograd |
| Sleep (batch) | Replay + semantic memory update | High throughput, full autograd, FP16/INT8 |
| Sleep (batch) | Growth: train new cascade units | Full autograd, small-batch iterative |
| Sleep (batch) | Compression: quantization/pruning | Quantization support, may be offline |

No single Rust ML framework covers all of this — likely answer is a split runtime: `ort` for frozen encoder inference, `candle`/`burn` for cognitive-layer training.

---

## Framework comparison

**`ort` (ONNX Runtime bindings).** Mature production inference engine, CUDA/TensorRT/DirectML backends. Best-in-class frozen-model latency (~3-5× faster than Python/PyTorch, 60-80% lower memory, per published case studies); native INT8/FP16; well-trodden `optimum`→ONNX→`ort` export path; graph-level optimizations. Inference-only, opaque graph (custom ops need ONNX tooling not Rust), dynamic shapes need careful export handling. Memory: model loaded once per session; Whisper-tiny FP16 ~75MB model + ~200-400MB activation peak (est.). Latency (est.): Whisper-tiny 30s chunk ~10-20ms, DINOv2-S/14 single frame ~3-6ms, ~1-3ms per-call overhead. **Rating: primary runtime for frozen encoder inference, not for cognitive-layer learning.**

**`candle` (HuggingFace).** Pure-Rust, inference-focused with optional training, CUDA via cublas/cudnn, Metal for Apple Silicon. Native Rust (small binary), good HF model coverage, native FP16/CUDA, active HF development, usable for small training loops (online updates, replay-driven weight updates). Training less mature than PyTorch (API gaps for complex schedules); ~1.5-2× slower inference than `ort` (est.); no built-in INT8. Memory: tracks a computation graph, ~2× forward-pass activations for backprop (checkpointing available), minimal overhead for small (<100M param) cognitive-layer networks. **Rating: good candidate for cognitive-layer training; also usable for encoder inference if `ort` integration is too complex early on.**

**`burn`.** Pure-Rust, backend-agnostic (CPU/ndarray, CUDA via CubeCL, Metal, WebGPU), type-safe shapes. Fully Rust (no C/C++ FFI), same model code across backends, best Rust-native training story (mature autograd/optimizers/loops), compile-time dimension safety, good for complex schedules. Younger ecosystem, fewer ported HF models, CubeCL CUDA path less battle-tested than cuBLAS/cuDNN, heavier compile times, no `ort`-comparable inference track record. Latency (est., rel. to candle): CubeCL CUDA within ~20-30% on common ops, not yet benchmarked for somnus's patterns. **Rating: viable cognitive-layer-training alternative to candle, especially if backend flexibility matters — higher risk given less production exposure.**

**`tch-rs` (libtorch bindings).** Full PyTorch op/kernel access, excellent training support, mature/widely used, native FP16/INT8/BFloat16. libtorch is a ~500MB+ native dependency (breaks cross-compilation, CI complexity), CUDA-version-matching required, thin non-idiomatic wrapper, poor ABI stability across libtorch versions. Memory model = PyTorch's exactly. **Rating: fallback of last resort when candle/burn lack a critical capability — the dependency cost is real, avoid unless justified.**

---

## Recommended runtime architecture

> **Actual choice (supersedes the recommendation below):** `burn` (NdArray/CPU) was adopted for cognitive-layer training, not `candle`. Wake inference later migrated to `ort`+MIGraphX (ADR 0023). Current split: `ort` for all inference (encoders + perception), `burn` for all training (NREM/wake_reconstruct/AdamW).

```
Frozen encoder inference (Whisper, CLAP, DINOv2)      → ort (lowest latency, ONNX quantization)
Perception inference during wake (k-WTA, binding)     → ort + MIGraphX (GPU, two-graph split; ADR 0023)
Sleep-cycle training (replay, growth, homeostasis)    → burn (NdArray/CPU, full autograd + AdamW)
Compression/quantization of semantic memory           → ort quantization tools (at ONNX export)
```

Keeps the hot path (encoders+perception during wake) on the fastest inference runtime while giving the sleep cycle full autograd.

---

## Budget analysis

Runtime overhead: `ort` ~50-100MB, `candle` ~50-150MB, both simultaneously ~150-250MB (est.) — of 14GB VRAM: runtime ~250MB, frozen encoders (moderate scenario) ~520MB, activation buffers (peak, all encoders) ~1-2GB, **remaining for cognitive layers+episodic memory: ~11-12GB.** Not a bottleneck.

---

## Open questions → future ADR

1. `ort` exclusively for encoder inference, or start with `candle` and switch only if latency becomes an issue?
2. `candle` vs. `burn` for cognitive-layer training — depends on Phase 3 sleep-cycle update loop complexity.
3. INT8 quantization of encoders via `ort` — worth it for Phase 1, or defer?

---

## Sources

- `ort` crate docs: ort.pyke.io
- candle: github.com/huggingface/candle
- burn: github.com/tracel-ai/burn
- "Building Sentence Transformers in Rust" — ort/candle/burn comparison: dev.to/mayu2008/...
- "Rust for Machine Learning: Burn vs Candle Framework Comparison" (2026): dasroot.net/posts/2026/04/...
- "ONNX Runtime in Rust: Running ML Models Efficiently" (2026): dasroot.net/posts/2026/03/...
- Published case study, financial services Python→Rust ONNX Runtime, 22ms→3.5ms (−84%): markaicode.com/rust-ml-inference-engines-2025/
