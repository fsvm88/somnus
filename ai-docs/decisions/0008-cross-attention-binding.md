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

Phase 3's binding area concatenated four N_UNITS-dim dense region activations and projected to EPISODE_DIM=512 via one `linear_no_bias(4096,512)`. Two problems: (1) no cross-regional interaction before projection — concat+linear can only learn fixed linear combinations, with no mechanism for one region to weight/suppress another based on context (the kind of interaction McGurk-style audio/visual conflict resolution needs); (2) parameter inefficiency — the 4096×512 matrix (~2M params) sees all cross-regional combinations at once with no inductive bias toward cross-modal relationships.

## Decision

Replace concat+linear with scaled dot-product self-attention over region tokens, then flatten to EPISODE_DIM. Originally single-head, upgraded to 4-head (see Amendment).

**Architecture** (current — 5 regions, 4-head, GRU temporal context):
```
5 × SparseActivation
  → to_dense_n(N_UNITS) per region → 5×[N_UNITS]
  → per-region linear_no_bias(N_UNITS→D_PROJ=128) → 5×[128]
  → stack current tokens → x[5,128]
  → optional GRU over WM episode ring → ctx[5,128] (appended to K,V)
  → Q=w_q(x); K/V=w_k/w_v(x||ctx)
  → split N_HEADS=4 × D_HEAD=32
  → per-head scores=(Q_h@K_h^T)/√32 → softmax → out_h
  → concat heads → [5,128] → flatten → [640]=EPISODE_DIM
```
Language slot (index 4) is zero-padded when no text embedding is available — output stays valid.

**Params:** 5 region projections `5×[1024,128]`=655,360; W_q/W_k/W_v `3×[128,128]`=49,152; GRU (input=640,hidden=256)≈852K; temporal_proj `[256,640]`=163,840 — **total ~1.72M**, vs. old concat+linear's 2,097,152 — slightly lower despite the GRU (GRU is now the dominant component).

**D_PROJ=128:** 5×D_PROJ=EPISODE_DIM (5×128=640) — the flattened attention output is exactly episode-vector sized, no final linear layer needed.

**Input clamping — `to_dense_n(N_UNITS)`:** region k-WTA layers grow beyond N_UNITS during sleep; binding deliberately clamps inputs to the original N_UNITS, keeping input projections fixed-size regardless of growth, deferring how to expose grown units to Phase 6+ (see the 2026-06-14 amendment — this was later addressed).

## Consequences

- Episode vectors are now shaped by cross-regional attention weights — an all-zero region still receives signal from others via attention (diffuse, uniform-over-keys row for that region).
- Binding participates in sleep replay training (Q/K/V and projections are in the shared VarMap, updated by AdamW) — in Phase 5 the decoder loss flows back through `sparse_z`, not the binding forward pass, so binding weights were expected to be shaped mainly by the episode vector's downstream reuse, sparse in Phase 5, "revisit in Phase 6." **Correction (2026-08-14, gaps.md §33):** that downstream signal never materialized — every training path consumes episodes as stored data (`Tensor::from_data`), no gradient ever reached binding weights, and the Phase 6 revisit never happened. Binding's ~1.72M params ran on random init and grown units' zero-init binding rows never learned. See gaps.md §33 (confirmed gap) — decided and implemented as ADR 0041.
- **Snapshot incompatibility:** old snapshots with `binding.weight` can't load — `varmap.load` warns and falls back to fresh-init binding weights; regional k-WTA/SAE weights unaffected.
- Binding param count 573K vs. 2.1M old — total cognitive-layer params decrease slightly.

## Amendment — multi-head binding + language region + GRU (2026-05-30)

1. **5 regions (Phase 6, ADR 0013):** language added (index 4, 896-dim Qwen2.5 embed), EPISODE_DIM 512(4×128)→640(5×128).
2. **4-head attention:** W_q/W_k/W_v stay `[128,128]` (same param count). Reshape `[seq,128]→[seq,4,32]→swap_dims(0,1)→[4,seq,32]` (contiguous feature partition — wrong reshape order silently interleaves tokens across heads). Scale 1/√128→1/√32. Test `different_head_slices_produce_different_outputs` discriminates correct partitioning from a wrong reshape.
3. **GRU temporal context:** replaces the old WM-mean approach. GRU (input=640, hidden=256) reads the WM episode ring; final hidden projects `[256→640]`, reshapes to `[5,D_PROJ]` to augment K/V. Q unchanged `[5,128]`, output stays `[5,128]→[640]`. `seq_k`=5 (no context) or 10 (WM context). Test `output_with_temporal_context_is_finite_and_differs` covers seq_k=10.

## Alternatives considered

- **Gated fusion.** Each region × learned scalar gate, summed — simpler, still no cross-regional context before aggregation.
- **Cross-attention (asymmetric).** One region queries others — more complex (needs a designated "query region"); symmetric self-attention over all regions needs no such designation.
- **Multi-head attention.** Implemented (N_HEADS=4, see Amendment) — originally deferred, adopted once 5-region binding + GRU temporal context raised representational demand.

## Amendment — grown-units binding exposure (2026-06-14)

Implemented once runs confirmed regions reaching 1920 units (max observed). `BindingArea` gains `#[module(skip)] region_n:[usize;5]` (init `[N_UNITS;5]`); `forward()` uses `region_n[i]` instead of `N_UNITS` for `to_dense_n` and the input shape, so grown units pass through instead of being silently dropped. `grow_region`/`grow_region_seeded`/`prune_region` also resize the binding projection weight (`[old_n,D_PROJ]→[new_n,D_PROJ]`, zero-init new rows) and update `region_n[i]`. `load_weights` reconstructs `region_n` from actual projection weight shapes after loading (skip fields reset on deserialisation, weight dims are authoritative).

**EPISODE_DIM unchanged** — each region still projects `region_n[i]→D_PROJ=128` independently, output still `5×128=640`. HNSW/cold-store/gist unaffected; the original deferred-item note about "episode vector dimension change" applied to the old concat+linear design and was stale.

**ORT path limitation:** the compiled ONNX binding graph accepts fixed `[1,N_UNITS]` inputs and can't grow with the burn model — `try_bind` keeps calling `to_dense_n(N_UNITS)` over ORT; grown units reach binding only via the burn path.

**Snapshot compatibility:** old snapshots (`[1024,128]` binding weights) loaded into a grown processor get a burn shape-mismatch warning and fall back to `grow_region`'s zero-initialized weights — functionality preserved, projections retrain from zero.
