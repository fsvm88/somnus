# ADR 0034 — KV-cache split prefill/decode generation (generation speedup)

**Date:** 2026-06-25
**Status:** Implemented

## Context

Generation was ~530ms/step because each decode step re-ran the full transformer over the growing prefix + all previously generated tokens — a 128-token response took 60-70s. Root cause: `qwen25_lm.onnx` is dynamic-shape, each step processes a longer sequence at O(n²) cost.

## Decision

Export two additional ONNX models from the LoRA-merged Qwen2.5-0.5B:

| Model | Shape | Purpose |
|---|---|---|
| `qwen25_lm_prefill_kv.onnx` | `[1,n,896]` dynamic | Run the prefix once; output logits + 24 key + 24 value tensors |
| `qwen25_lm_decode_kv.onnx` | `[1,1,896]` + static `[1,2,2048,64]`×48 | Single-token decode; input past KV, output logits + 24 new_key + 24 new_value |

**KV_SIZE=2048**, chosen from measured CPU latency — covers a ~400-token prefix + 1600 generated tokens, more than any realistic session.

| Approach | ms/step | 128 tokens |
|---|---|---|
| Old (growing sequence, no KV) | ~530ms | ~68s |
| KV decode, KV_SIZE=512 | 73ms | 9.4s |
| KV decode, KV_SIZE=1024 | 77ms | 9.9s |
| **KV decode, KV_SIZE=2048** | **68ms (ORT)** | **8.7s** |
| KV decode, KV_SIZE=4096 | 114ms | 14.6s |
| KV decode, KV_SIZE=16384 | 461ms | 59s (SLOWER!) |

Larger caches are slower on CPU because materialising GQA-expanded KV (`[1,14,N,64]`/layer) dominates — KV_SIZE=2048 is the sweet spot where memory-bandwidth cost is roughly proportional to actual content. **7.8× speedup** over the old path.

**ONNX export wrappers:** Prefill (`PrefillKVWrapper`) wraps `model.model`+`lm_head`, `use_cache=True`, extracts `out.past_key_values.layers[i].keys`/`.values` as outputs, traced `dynamo=False`. Decode (`DecodeKVWrapper`) is a custom layer-by-layer forward bypassing DynamicCache entirely, using `model.model.rotary_emb` (shared, attached to `model.model` not per-layer in transformers 5.12). Inputs: `inputs_embeds`, `position_ids`, `attention_mask`, 48 past-KV tensors; outputs: `logits`+24 `new_key_i`+24 `new_value_i`. Key discoveries: `attn.rotary_emb` doesn't exist per-layer in transformers 5.12 (use `model.model.rotary_emb`); GQA `N_KV_HEADS=2`/`N_HEADS=14`, expansion 7× (`repeat_kv`); `attn.scaling=1/√HEAD_DIM` is the correct per-layer scale.

**Attention mask convention:** decode takes `attention_mask[1,1,1,KV_SIZE+1]` — position `KV_SIZE` is the self-attention slot (always 0.0), positions 0..pos hold cached KV (0.0=attend, -1e4=unused).

**Rust (`action.rs`):** `KvCache` struct — 48 flat buffers (24 key, 24 value) of `[N_KV_HEADS*KV_SIZE*KV_HEAD_DIM]` f32s + a mask Vec. `prefill_kv_session`/`decode_kv_session` loaded if both ONNX files present. `generate_kv()` private method: prefill → scatter KV into buffers → greedy decode loop. KV inputs built as `Vec<(String, ValueRef<'_, DynValueTypeMarker>)>` via `.into_dyn()` (erases tensor element type for ORT's `IntoInputs`). Session `.take()`/re-placed for the borrow checker (same pattern as the projection optimizer step's `model.take()`). `generate()` falls back to old paths if KV models are absent or the prefix exceeds KV_SIZE.

**Numerical validation:** equivalence check (320-token prefix, 20 decode steps, LoRA-merged model): 18/20 token agreement (≥18 required) — the 2 divergent tokens occur where top-2 logits are very close, expected from floating-point order differences in GQA expansion vs. the reference forward.

**Export script:** `scripts/export_kv_cache.py --kv-size 2048` exports and validates both models.

## Alternatives considered

- **GPU (MIGraphX).** Previously measured 14× slower than CPU for static models on gfx1201. Not used.
- **Larger KV_SIZE (16384).** The original proposal — measured 461ms/step, actually slower than no-KV since GQA expansion materialises 2.2GB of KV data/step on CPU.
- **Smaller KV_SIZE (1024).** 77ms/step, nearly as fast, works for all current sessions — rejected since prefix could theoretically reach 400 tokens and responses can run long.

## Consequences

Two new ~1.9GB ONNX files (`qwen25_lm_prefill_kv.onnx`, `qwen25_lm_decode_kv.onnx`). `generate()` automatically uses the KV path when both are present. ~96MB of KV clones/decode step (48×2MB) — acceptable at 68ms/step. `generate_kv()` uses the `.take()`/restore pattern for clean borrow splits. If `PREFIX_LEN>KV_SIZE`, falls back to the old `lm_session` path (rare).
