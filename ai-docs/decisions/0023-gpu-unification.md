# 0023. GPU unification via ORT/MIGraphX for perception inference

- **Status:** accepted; Phase 0 + Phase 1 implemented (2026-06-07)
- **Date:** 2026-06-02

> **Decision:** Export perception inference (k-WTA, cross_modal_proj, BindingArea GRU) as two ONNX graphs (perception_encoder.onnx + perception_binding.onnx) run via ORT+MIGraphX; burn stays on CPU for training only. Duty-cycle topk/masking on CPU between graphs to avoid MIGraphX ScatterElements/INT64 bug on gfx1201.
> **Keywords:** ORT, MIGraphX, GPU, perception_encoder.onnx, perception_binding.onnx, two-graph split, burn, kwta_us, sym_us, ScatterElements bug, INT64 output, gfx1201, RDNA4, sync_ort_weights, enable_ort_inference

## Context

The system ran burn (NdArray, CPU) for all perception inference alongside ORT+MIGraphX for frozen encoders. Fine at N_UNITS=1024/batch=1 (perception hides behind the GPU encode path), but the two-framework split: (1) blocks GPU use for perception entirely — the wgpu attempt (5dbf9c9) failed because MIGraphX already owns the AMD device; (2) leaves ~9× VRAM bandwidth advantage unused; (3) becomes an increasingly large bottleneck as N_UNITS scales toward brain-inspired sizes (4K→64K units/region). The scaling analysis (`research/gpu-unification-design.md` §2) shows GPU breaks even with CPU at N_UNITS≈1,230 (batch-1) and wins 2.4× at current scale with micro-batch=8; at 4× scale (4K units), 2.2× batch-1, 6.5× with batching.

## Decision

Export the perception inference path (k-WTA layers, SAE decoders, cross_modal_proj, BindingArea, EMA output heads) as an ONNX graph, execute via ORT/MIGraphX alongside encoder sessions. burn stays CPU-only for all training (NREM replay, wake_reconstruct, AdamW). Weights sync burn→ORT graph at each sleep→wake boundary.

**Split:** GPU (ORT graph) — all frozen-during-wake inference (k-WTA linear forward, SAE decode, symbolic cross-modal projection, binding projections+cross-attention+GRU step, EMA head predict). CPU (burn) — all training ops, duty-cycle update, top-k selection/masking, growth/prune, snapshot I/O. Sync at sleep→wake: burn weights exported to ORT initializers, new session created. Full design: `research/gpu-unification-design.md`.

## Alternatives considered

- **wgpu backend for burn.** Tried (5dbf9c9) — AMD RX 9070 XT already claimed by MIGraphX ORT at first video frame (exit 101), not retryable without moving encoders off ORT.
- **Candle or tch for inference.** Would add a third GPU framework dependency, no better than ORT for this use case.
- **Keep CPU perception, optimize NdArray.** NdArray isn't BLAS-linked — linking BLAS would help at current scale but not at 4K+ scale where GPU dominates. Wrong direction for the scaling trajectory.
- **Do nothing.** Valid at current scale, wrong bet on the research trajectory.

## What would invalidate this decision

MIGraphX produces incorrect results for GRU/TopK/multi-head attention with no workaround. ORT session rebuild time at grown N_UNITS exceeds 10% of the 602s wake period. The research direction pivots away from region-scaling — at N_UNITS=1024 batch-1, CPU is faster; this work's cost is only justified by the scaling roadmap.
