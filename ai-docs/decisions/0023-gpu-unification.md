# 0023. GPU unification via ORT/MIGraphX for perception inference

- **Status:** proposed
- **Date:** 2026-06-02

## Context

The current system runs burn (NdArray, CPU) for all perception inference alongside ORT+MIGraphX for frozen encoders. At N_UNITS=1024 and batch=1 this is fine — perception hides behind the GPU encode path. The problem is that the two-framework split:

1. Blocks GPU use for perception entirely — wgpu attempt (commit 5dbf9c9) failed because MIGraphX already owns the AMD device.
2. Leaves ~9× VRAM bandwidth advantage unused.
3. Will become an increasingly large bottleneck as N_UNITS scales toward human-brain-inspired sizes (4K→64K units/region).

The scaling analysis (research/gpu-unification-design.md §2) shows GPU inference breaks even with CPU at N_UNITS≈1,230 (batch-1) and wins 2.4× at current scale with micro-batch=8. At 4× scale (4K units), GPU wins 2.2× batch-1 and 6.5× with batching.

## Decision

Export the perception inference path (k-WTA layers, SAE decoders, cross_modal_proj, BindingArea, EMA output heads) as an ONNX graph and execute it via ORT/MIGraphX alongside the encoder sessions. burn remains on CPU for all training (NREM replay, wake_reconstruct, AdamW). Weights sync from burn to the ORT graph at each sleep→wake boundary.

## Architecture split

- **GPU (ORT graph):** all frozen-during-wake inference ops — k-WTA linear forward, SAE decode, symbolic cross-modal projection, binding (projections + cross-attention + GRU step), EMA head predict.
- **CPU (burn):** all training ops, duty-cycle update, top-k selection and masking, growth/prune, snapshot I/O.
- **Sync:** at sleep→wake boundary — burn weights exported to ORT initializers, new session created.

Full design: research/gpu-unification-design.md.

## Alternatives considered

1. **wgpu backend for burn** — tried in commit 5dbf9c9; AMD RX 9070 XT already claimed by MIGraphX ORT at first video frame (exit 101). Not retryable without moving encoders off ORT.
2. **Candle or tch for inference** — would add a third GPU framework dependency; no better than ORT for this use case.
3. **Keep CPU perception, optimize NdArray** — NdArray is not BLAS-linked; linking BLAS would help at current scale but does not help at the 4K+ scale where GPU dominates. Wrong direction for a scaling trajectory.
4. **Do nothing** — valid at current scale, wrong bet on the research trajectory.

## What would invalidate this decision

- MIGraphX produces incorrect results for GRU, TopK, or multi-head attention and there is no available workaround.
- ORT session rebuild time at grown N_UNITS exceeds 10% of the 602s wake period.
- The research direction pivots away from region-scaling (e.g., toward algorithm changes that don't require larger N_UNITS). At N_UNITS=1024 batch-1, CPU is faster; the cost of this work is only justified by the scaling roadmap.
