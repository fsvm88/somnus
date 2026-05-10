# ADR 0008 — Cross-attention binding area

**Date:** 2026-05-10  
**Status:** Accepted

## Context

Phase 3 introduced a binding area that concatenated four N_UNITS-dimensional dense region activations and projected the result to EPISODE_DIM=512 via a single `linear_no_bias(4096, 512)` layer. This had two problems:

1. **No cross-regional interaction before projection.** The concat+linear can only learn fixed linear combinations of the four inputs. There is no mechanism for one region to weight or suppress another based on context — the kind of interaction that matters for multimodal disambiguation (e.g., a McGurk-style conflict between audio and visual signals).

2. **Parameter inefficiency.** The 4096×512 weight matrix has ~2M parameters that see all cross-regional combinations at once, with no inductive bias toward cross-modal relationships.

## Decision

Replace the concat+linear binding area with **single-head scaled dot-product self-attention over four region tokens**, followed by a flatten to EPISODE_DIM.

### Architecture

```
Input: 4 × SparseActivation (one per region)
  ↓  to_dense_n(N_UNITS) per region  →  4 × [N_UNITS]
  ↓  per-region linear_no_bias(N_UNITS → D_PROJ=128)  →  4 × [128]
  ↓  stack  →  [4, 128]
  ↓  Q = x @ W_q,  K = x @ W_k,  V = x @ W_v  →  3 × [4, 128]
  ↓  scores = (Q @ K^T) / sqrt(128)  →  [4, 4]
  ↓  attn = softmax(scores, dim=keys)  →  [4, 4]
  ↓  out = attn @ V  →  [4, 128]
  ↓  flatten  →  [512] = EPISODE_DIM
```

### Parameter count

| Component | Shape | Params |
|---|---|---|
| 4 region projections | 4 × [1024, 128] | 524,288 |
| W_q, W_k, W_v | 3 × [128, 128] | 49,152 |
| **Total** | | **~573K** |

vs. old concat+linear: 4096 × 512 = **2,097,152**. ~3.7× fewer parameters.

### D_PROJ = 128 choice

4 × D_PROJ = EPISODE_DIM (4 × 128 = 512), so the flattened attention output is exactly episode-vector sized — no final linear layer is needed. This is a clean constraint that also determines the scale of the per-region projections.

### Single-head attention

With only 4 tokens (one per region), multi-head attention over 4 tokens with typical head dimensions adds complexity without clear benefit. Single-head keeps the implementation simple; if future experiments show multi-head helps, it is an incremental change.

### Input clamping: to_dense_n(N_UNITS)

Region k-WTA layers grow beyond N_UNITS during sleep. The binding area deliberately clamps inputs to the original N_UNITS via `to_dense_n(N_UNITS)`. This keeps the binding area's input projections fixed-size regardless of regional growth, deferring the question of how to expose grown units to the binding area to Phase 6+.

## Consequences

- Episode vectors are now shaped by cross-regional attention weights. A region with all-zero activation will still receive signal from other regions via attention, but the attention row for that region will be diffuse (uniform over keys).
- The binding area now participates in sleep replay training (its Q/K/V and region projection weights are in the shared VarMap and updated by AdamW). In Phase 5 the binding decoder loss signal flows back through `sparse_z`, not through the binding forward pass, so the binding weights will primarily be shaped by the episode vector being reused downstream — this is sparse in Phase 5 and will be revisited in Phase 6.
- **Snapshot incompatibility:** old snapshots with `binding.weight` cannot be loaded by this version. `varmap.load` will log a warning and fall back to freshly-initialized binding weights. Regional k-WTA and SAE weights are unaffected.
- Binding area parameter count: 573K vs 2.1M (old). Total cognitive-layer parameters decrease slightly.

## Alternatives considered

- **Gated fusion:** each region multiplied by a learned scalar gate, then summed. Simpler but still no cross-regional context before aggregation.
- **Cross-attention (asymmetric):** one region queries others. More complex to wire since we'd need to designate a "query region". Self-attention over all four regions is symmetric and requires no such designation.
- **Multi-head attention:** straightforward extension; deferred since single-head is simpler and the token count (4) is small.
