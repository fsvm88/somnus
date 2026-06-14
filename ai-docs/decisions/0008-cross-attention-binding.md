---
status: accepted
date: 2026-05-10
---

# ADR 0008 — Cross-attention binding area

**Date:** 2026-05-10  
**Status:** Accepted

> **Decision:** 4-head scaled dot-product cross-attention over 5 region tokens + GRU temporal context (GRU_HIDDEN=256) → EPISODE_DIM=640 episode vector; working memory appended to K/V.  
> **Keywords:** BindingArea, cross-attention, GRU, GRU_HIDDEN=256, EPISODE_DIM=640, D_PROJ=128, working memory, region tokens, temporal context, try_bind

## Context

Phase 3 introduced a binding area that concatenated four N_UNITS-dimensional dense region activations and projected the result to EPISODE_DIM=512 via a single `linear_no_bias(4096, 512)` layer. This had two problems:

1. **No cross-regional interaction before projection.** The concat+linear can only learn fixed linear combinations of the four inputs. There is no mechanism for one region to weight or suppress another based on context — the kind of interaction that matters for multimodal disambiguation (e.g., a McGurk-style conflict between audio and visual signals).

2. **Parameter inefficiency.** The 4096×512 weight matrix has ~2M parameters that see all cross-regional combinations at once, with no inductive bias toward cross-modal relationships.

## Decision

Replace the concat+linear binding area with scaled dot-product self-attention over region tokens, followed by a flatten to EPISODE_DIM. Originally single-head; upgraded to 4-head (see Amendment below).

### Architecture (current — 5 regions, 4-head, GRU temporal context)

```
Input: 5 × SparseActivation (one per region)
  ↓  to_dense_n(N_UNITS) per region     →  5 × [N_UNITS]
  ↓  per-region linear_no_bias(N_UNITS → D_PROJ=128)  →  5 × [128]
  ↓  stack current tokens               →  x [5, 128]
  ↓  optional GRU over WM episode ring  →  ctx [5, 128]  (appended to K, V)
  ↓  Q = w_q(x);  K/V = w_k/w_v(x || ctx)
  ↓  split to N_HEADS=4 heads × D_HEAD=32 per head
  ↓  per-head: scores = (Q_h @ K_h^T) / sqrt(32)  →  softmax  →  out_h
  ↓  concat heads  →  [5, 128]
  ↓  flatten  →  [640] = EPISODE_DIM
```

The Language slot (index 4) is zero-padded when no text embedding is available; the binding area output is still valid in that case.

### Parameter count (current)

| Component | Shape | Params |
|---|---|---|
| 5 region projections | 5 × [1024, 128] | 655,360 |
| W_q, W_k, W_v | 3 × [128, 128] | 49,152 |
| GRU (temporal context) | input=640, hidden=256 | ~852K |
| temporal_proj | [256, 640] | 163,840 |
| **Total** | | **~1.72M** |

vs. old concat+linear: 4096 × 512 = **2,097,152**. Parameter count is now slightly lower despite the GRU; GRU is the dominant component.

### D_PROJ = 128 choice

5 × D_PROJ = EPISODE_DIM (5 × 128 = 640), so the flattened attention output is exactly episode-vector sized — no final linear layer needed.

### Multi-head attention

4-head attention (N_HEADS=4, D_HEAD=32) is implemented; see Amendment section for details. The original single-head design is superseded.

### Input clamping: to_dense_n(N_UNITS)

Region k-WTA layers grow beyond N_UNITS during sleep. The binding area deliberately clamps inputs to the original N_UNITS via `to_dense_n(N_UNITS)`. This keeps the binding area's input projections fixed-size regardless of regional growth, deferring the question of how to expose grown units to the binding area to Phase 6+.

## Consequences

- Episode vectors are now shaped by cross-regional attention weights. A region with all-zero activation will still receive signal from other regions via attention, but the attention row for that region will be diffuse (uniform over keys).
- The binding area now participates in sleep replay training (its Q/K/V and region projection weights are in the shared VarMap and updated by AdamW). In Phase 5 the binding decoder loss signal flows back through `sparse_z`, not through the binding forward pass, so the binding weights will primarily be shaped by the episode vector being reused downstream — this is sparse in Phase 5 and will be revisited in Phase 6.
- **Snapshot incompatibility:** old snapshots with `binding.weight` cannot be loaded by this version. `varmap.load` will log a warning and fall back to freshly-initialized binding weights. Regional k-WTA and SAE weights are unaffected.
- Binding area parameter count: 573K vs 2.1M (old). Total cognitive-layer parameters decrease slightly.

## Amendment — multi-head binding + language region + GRU (2026-05-30)

Three changes relative to the original Phase 3 decision, all reflected in the main body above:

1. **5 regions (Phase 6, ADR 0013).** Language region added (index 4, 896-dim Qwen2.5 embed). EPISODE_DIM grows from 512 (4×128) to 640 (5×128).

2. **4-head attention (2026-05-30).** W_q/W_k/W_v stay [D_PROJ, D_PROJ] = [128, 128] — same parameter count. Reshape: `[seq, 128] → [seq, 4, 32] → swap_dims(0,1) → [4, seq, 32]` (contiguous feature partition; wrong reshape order silently interleaves tokens across heads). Scale changes from 1/√128 to 1/√32. Test `different_head_slices_produce_different_outputs` discriminates correct partitioning from wrong reshape.

3. **GRU temporal context (architectural roundup).** Replaces the old WM-mean approach. A GRU (input=EPISODE_DIM=640, hidden=GRU_HIDDEN=256) reads the working-memory episode ring; final hidden state is projected ([256→640]) and reshaped to [5, D_PROJ] to augment K and V. Q is unchanged ([5, 128]), so output shape is still [5, 128]→[640]. `seq_k` is 5 (no context) or 10 (with WM context). Test `output_with_temporal_context_is_finite_and_differs` covers the seq_k=10 path.

## Alternatives considered

- **Gated fusion:** each region multiplied by a learned scalar gate, then summed. Simpler but still no cross-regional context before aggregation.
- **Cross-attention (asymmetric):** one region queries others. More complex to wire since we'd need to designate a "query region". Self-attention over all four regions is symmetric and requires no such designation.
- **Multi-head attention:** implemented (N_HEADS=4, see Amendment). Originally deferred; adopted once 5-region binding and GRU temporal context increased the representational demand.

## Amendment — grown-units binding exposure (2026-06-14)

Items 4+5 from the ADR scan deferred list, implemented once runs confirmed regions reach 1920 units (max observed across campaign logs).

### What changed

- `BindingArea` gains a `#[module(skip)] region_n: [usize; 5]` field, initialized to `[N_UNITS; 5]`.
- `forward()` uses `self.region_n[i]` instead of `N_UNITS` for `to_dense_n` and the input tensor shape, so grown units pass through to the projection rather than being silently dropped.
- `grow_region`, `grow_region_seeded`, and `prune_region` now also resize the corresponding binding projection weight (`[old_n, D_PROJ] → [new_n, D_PROJ]`, zero-initialized new rows) and update `region_n[i]`.
- `load_weights` reconstructs `region_n` from the actual projection weight shapes after loading (skip fields reset on deserialisation; weight dims are authoritative).

### EPISODE_DIM is unchanged

Each region projects `region_n[i] → D_PROJ=128` independently; the output is still `5 × 128 = 640 = EPISODE_DIM`. HNSW/cold-store/gist are unaffected. The original note in the deferred item about "episode vector dimension change" applied to the old concat+linear design and was stale.

### ORT path limitation

The compiled ONNX binding graph accepts fixed `[1, N_UNITS]` inputs and cannot grow with the burn model. `try_bind` continues to call `to_dense_n(N_UNITS)` when routing through ORT; grown units reach binding only via the burn path.

### Snapshot compatibility

Old snapshots (saved before this amendment) have `[1024, 128]` binding projection weights. When loaded into a grown processor, burn's recorder logs a shape-mismatch warning and falls back to the zero-initialized weights produced by `grow_region`. Functionality is preserved; projections retrain from zero.
