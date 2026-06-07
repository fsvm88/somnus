# ML Runtimes

> **Contents:** Framework comparison — ort (inference, ✓ committed), burn (training, ✓ committed/NdArray backend), candle (not adopted), tch-rs (not adopted). Latency estimates, memory models, recommended split. Note: candle recommendation in this file was superseded; see updated "Recommended runtime architecture" section.  
> **When to read:** Evaluating a new ML framework; understanding why burn was chosen over candle; checking ORT latency estimates.

Research note for somnus. Covers the tensor + inference + training runtime layer in Rust: `ort`, `candle`, `burn`, and `tch-rs`. All numbers are **engineering estimates**; exact figures depend on GPU, driver, and model topology.

See `tech-stack.md` for the current candidate list. The choice here drives how all frozen encoder inference and all cognitive-layer training is implemented.

---

## What we need from the runtime

somnus has two distinct runtime modes with different requirements:

| Mode              | Operation                           | Requirement                                    |
|-------------------|-------------------------------------|------------------------------------------------|
| **Wake (online)** | Frozen encoder inference             | Low latency (<30 ms/frame); no autograd needed |
| **Wake (online)** | Cognitive-layer forward pass         | Low latency; minimal autograd                  |
| **Sleep (batch)** | Replay + semantic memory update      | High throughput; full autograd; FP16/INT8      |
| **Sleep (batch)** | Growth: train new cascade units      | Full autograd; small-batch, iterative          |
| **Sleep (batch)** | Compression: quantization / pruning  | Quantization support; may be offline           |

No single Rust ML framework fully covers all of these today. The likely answer is a **split runtime**: `ort` for frozen encoder inference, `candle` or `burn` for cognitive-layer training.

---

## Framework comparison

### ort (ONNX Runtime bindings)

ONNX Runtime is a mature, production-grade inference engine with CUDA, TensorRT, and DirectML backends. The `ort` Rust crate provides idiomatic bindings.

**Strengths:**
- Best-in-class inference latency for frozen models — typically 3–5× faster than equivalent Python/PyTorch, with 60–80% lower memory usage (est. from published case studies).
- Native support for INT8 and FP16 quantized models — the right tool for running compressed encoders.
- All HuggingFace models can be exported to ONNX. Well-trodden path: `optimum` → ONNX → `ort`.
- Graph-level optimizations: constant folding, op fusion, redundant node elimination.
- No training — inference only.

**Weaknesses:**
- Cannot train or fine-tune. All learning must happen elsewhere.
- ONNX graph is opaque — adding custom ops or modifying the graph requires ONNX tooling, not Rust code.
- Dynamic shapes (variable-length audio) require careful handling at export time.

**Memory model:**
- Model loaded once into a session; CUDA memory allocated per-session.
- Activation memory: proportional to batch size × layer activations, released after forward pass.
- For Whisper-tiny FP16: ~75 MB model + ~200–400 MB activation peak during forward pass (est.).

**Latency (est.):**
- Whisper-tiny encoder, 30s chunk, GPU: ~10–20 ms
- DINOv2-S/14, single 224×224 frame, GPU: ~3–6 ms
- Overhead per-call (session dispatch + memory copies): ~1–3 ms

**Rating for somnus:** primary runtime for frozen encoder inference. Not suitable for cognitive-layer learning.

---

### candle (HuggingFace)

Pure-Rust ML framework focused on inference with optional training support. CUDA backend via `cublas`/`cudnn`; Metal backend for Apple Silicon.

**Strengths:**
- Native Rust — no large native dependencies, small binary size.
- Good coverage of HuggingFace model architectures (many models ported directly to Candle).
- Supports FP16 natively on CUDA.
- Active development by HuggingFace; API reasonably stable.
- Reasonable for small training loops: online updates, replay-driven weight updates.

**Weaknesses:**
- Training support is present but not as mature as PyTorch — expect API gaps for complex update schedules.
- Fewer optimization passes than ONNX Runtime; inference is ~1.5–2× slower than `ort` for identical models (est.).
- Limited documentation for training workflows.
- No built-in INT8 quantization.

**Memory model:**
- CUDA tensors managed explicitly; candle tracks a computation graph for autograd.
- Graph memory: roughly 2× forward-pass activations for backprop (activation checkpointing available).
- For cognitive-layer networks (expected to be small, <100M params during Phase 2–4): minimal overhead.

**Latency (est., relative to ort):**
- Encoder inference: ~1.5–2× slower than `ort`
- Small-network forward pass (<10M params): ~2–5 ms on GPU
- Small-network backward pass: ~2× forward pass cost

**Rating for somnus:** good candidate for cognitive-layer training (regional concept areas, binding area, cascade-correlation growth). May also serve for encoder inference if `ort` integration is too complex early in Phase 1.

---

### burn

Pure-Rust deep learning framework with a backend-agnostic design. Backends include CPU (ndarray), CUDA (via CubeCL), Metal, and WebGPU. Emphasizes type-safe shapes via Rust's type system.

**Strengths:**
- Fully Rust — no C/C++ FFI for core operations.
- Backend agnosticism: the same model code runs on CPU/CUDA/Metal/WASM.
- Best Rust-native training story — mature autograd, optimizer implementations, training loops.
- Type-level shape safety catches dimension mismatches at compile time.
- Good for complex training schedules (learning rate schedules, multi-loss, etc.).

**Weaknesses:**
- Younger ecosystem than candle; fewer pre-ported HuggingFace models.
- CubeCL-based CUDA backend is less battle-tested than cuBLAS/cuDNN paths — possible performance gaps on specific ops.
- Larger compile times; heavier crate.
- No production inference track record comparable to `ort`.

**Memory model:**
- Similar to candle; backend manages tensor allocation.
- CubeCL path has some extra overhead in kernel dispatch vs. hand-tuned cuBLAS.

**Latency (est., relative to candle):**
- CubeCL CUDA: within ~20–30% of candle on common ops; specific kernels may vary significantly.
- Not yet benchmarked for somnus's use patterns.

**Rating for somnus:** viable alternative to candle for cognitive-layer training, especially if backend flexibility (future non-CUDA targets) is valued. Higher risk than candle given less production exposure. May converge with candle in quality over time.

---

### tch-rs (libtorch bindings)

Rust bindings to PyTorch's C++ libtorch library.

**Strengths:**
- Full access to PyTorch's entire operation set and optimized kernels.
- Excellent training support — anything PyTorch can do, `tch-rs` can do.
- Stable, mature, widely used in production Rust ML.
- Native FP16, INT8, BFloat16 support via PyTorch.

**Weaknesses:**
- libtorch is a ~500 MB+ native library dependency. Breaks cross-compilation; adds CI complexity.
- PyTorch's CUDA ops are not automatically available on all systems — requires matching CUDA version.
- The Rust API is a thin wrapper; not idiomatic Rust — closer to writing PyTorch in Rust syntax.
- ABI stability between libtorch versions is poor.

**Memory model:**
- Follows PyTorch's memory model exactly (cached allocator, lazy tensor materialization).
- VRAM usage is identical to PyTorch equivalents.

**Rating for somnus:** fallback of last resort when candle/burn lack a critical capability (e.g., a specific optimized kernel or training technique). The dependency cost is real — avoid unless justified.

---

## Recommended runtime architecture

> **Actual choice (supersedes the recommendation below):** `burn` (NdArray/CPU) was adopted for cognitive-layer training, not `candle`. The wake inference path was later migrated to `ort` + MIGraphX (ADR 0023). The current split is: `ort` for all inference (encoders + perception), `burn` for all training (NREM/wake_reconstruct/AdamW).

```
Frozen encoder inference (Whisper, CLAP, DINOv2)
    → ort  (lowest latency, ONNX quantization)

Perception inference during wake (k-WTA, binding, cross_modal_proj)
    → ort + MIGraphX  (GPU, two-graph split; ADR 0023)

Sleep-cycle training (replay updates, growth, homeostatic downscaling)
    → burn  (NdArray backend, CPU; full autograd + AdamW)

Compression / quantization of semantic memory
    → ort quantization tools (at ONNX export)
```

This split keeps the hot path (encoders + perception during wake) on the fastest inference runtime, while giving the sleep cycle access to full autograd.

---

## Budget analysis

VRAM budget for runtimes:

- `ort` session overhead: ~50–100 MB (runtime allocations, kernel caches)
- `candle` overhead: ~50–150 MB (CUDA context, cublas handles)
- Running both simultaneously: ~150–250 MB total overhead (est.)

Of 14 GB VRAM available:
- Runtime overhead: ~250 MB
- Frozen encoders (moderate scenario): ~520 MB
- Activation buffers (peak, all encoders running): ~1–2 GB
- **Remaining for cognitive layers + episodic memory: ~11–12 GB**

Runtime overhead is not a bottleneck.

---

## Open questions → future ADR

1. Use `ort` exclusively for encoder inference, or start with `candle` and switch only if latency becomes an issue? `candle` has simpler integration.
2. `candle` vs. `burn` for the cognitive layer training path? Depends on how complex the sleep-cycle update loop becomes in Phase 3.
3. INT8 quantization of encoders via `ort` — worth the added complexity for Phase 1, or defer?

---

## Sources

- `ort` crate documentation: ort.pyke.io
- candle repository: github.com/huggingface/candle
- burn repository: github.com/tracel-ai/burn
- "Building Sentence Transformers in Rust" — DEV Community practical comparison of ort/candle/burn: dev.to/mayu2008/...
- "Rust for Machine Learning: Burn vs Candle Framework Comparison" (2026): dasroot.net/posts/2026/04/...
- "ONNX Runtime in Rust: Running ML Models Efficiently" (2026): dasroot.net/posts/2026/03/...
- Published case study: financial services firm, Python→Rust ONNX Runtime, 22 ms → 3.5 ms (−84%): markaicode.com/rust-ml-inference-engines-2025/
