# ADR 0034 — KV-cache split prefill/decode generation (generation speedup)

**Date:** 2026-06-25
**Status:** Implemented

## Context

Generation speed was ~530 ms/step because each decode step re-ran the full transformer over the growing prefix + all previously generated tokens. For a 128-token response, this totalled 60–70 s. The root cause: `qwen25_lm.onnx` is a dynamic-shape model — each step processes a longer sequence at O(n²) cost.

## Decision

Export two additional ONNX models from the LoRA-merged Qwen2.5-0.5B:

| Model | Shape | Purpose |
|---|---|---|
| `qwen25_lm_prefill_kv.onnx` | `[1, n, 896]` dynamic | Run the prefix once; output logits + 24 key + 24 value tensors |
| `qwen25_lm_decode_kv.onnx` | `[1, 1, 896]` + static `[1, 2, 2048, 64]` × 48 | Single-token decode; input past KV, output logits + 24 new_key + 24 new_value |

### KV_SIZE = 2048

Chosen based on measured CPU latency (see below). Covers a ~400-token prefix plus 1600 generated tokens — more than any realistic session.

**Measured CPU latency (ORT):**

| Approach | ms/step | 128 tokens |
|---|---|---|
| Old (growing sequence, no KV) | ~530 ms | ~68 s |
| KV decode, KV_SIZE = 512 | 73 ms | 9.4 s |
| KV decode, KV_SIZE = 1024 | 77 ms | 9.9 s |
| **KV decode, KV_SIZE = 2048** | **68 ms (ORT)** | **8.7 s** |
| KV decode, KV_SIZE = 4096 | 114 ms | 14.6 s |
| KV decode, KV_SIZE = 16384 | 461 ms | 59 s (SLOWER!) |

Larger caches are slower on CPU because materialising the GQA-expanded KV ([1,14,N,64] per layer) dominates. KV_SIZE=2048 hits the sweet spot where memory bandwidth cost is ~proportional to actual content (prefix + generated). **7.8× speedup** over the old path.

### ONNX export wrappers

**Prefill** (`PrefillKVWrapper`): wraps `model.model` + `lm_head`. Uses `use_cache=True` and extracts `out.past_key_values.layers[i].keys` / `.values` as outputs. Traced with `dynamo=False`.

**Decode** (`DecodeKVWrapper`): custom layer-by-layer forward that bypasses DynamicCache entirely. Uses `model.model.rotary_emb` (shared, attached to `model.model` not per-layer in transformers 5.12). Inputs: `inputs_embeds`, `position_ids`, `attention_mask`, and 48 past KV tensors. Outputs: `logits` + 24 `new_key_i` + 24 `new_value_i`.

Key discoveries during export:
- `attn.rotary_emb` does not exist per-layer in transformers 5.12; use `model.model.rotary_emb` instead
- GQA: `N_KV_HEADS=2`, `N_HEADS=14`, expansion factor 7× (via `repeat_kv`)
- `attn.scaling = 1/sqrt(HEAD_DIM)` is the correct per-layer scale (avoids recomputing)

### Attention mask convention

The decode model takes `attention_mask [1, 1, 1, KV_SIZE+1]`. Position `KV_SIZE` is the self-attention slot (always 0.0). Positions 0..pos hold cached KV (0.0 = attend, -1e4 = unused).

### Rust (action.rs)

- `KvCache` struct: 48 flat buffers (24 key, 24 value) of `[N_KV_HEADS * KV_SIZE * KV_HEAD_DIM]` f32s, plus a mask Vec.
- `prefill_kv_session` and `decode_kv_session`: loaded if both ONNX files are present.
- `generate_kv()` private method: prefill → scatter KV into buffers → greedy decode loop.
- KV inputs built as `Vec<(String, ValueRef<'_, DynValueTypeMarker>)>` — uses `.into_dyn()` to erase tensor element type for ORT's `IntoInputs` dispatch.
- Session taken with `.take()` / re-placed to satisfy Rust borrow checker (same pattern used by `model.take()` in the projection optimizer step).
- `generate()` falls back to old paths if KV models are absent or prefix exceeds KV_SIZE.

### Numerical validation

Equivalence check (320-token prefix, 20 decode steps, LoRA-merged model): 18/20 token agreement (≥18 required). The 2 divergent tokens occur at positions where the top-2 logits are very close; this is expected from floating-point order differences in GQA expansion vs the reference forward pass.

## Export script

`scripts/export_kv_cache.py` exports and validates both models. Run:
```
.venv/bin/python scripts/export_kv_cache.py --kv-size 2048
```

## Alternatives considered

**GPU (MIGraphX):** Previously measured 14× slower than CPU for static models on gfx1201 (RDNA 4). Not used.

**Larger KV_SIZE (16384):** The original proposal. Measured 461 ms/step — actually slower than no-KV because GQA expansion materialises 2.2 GB of KV data per step on CPU.

**Smaller KV_SIZE (1024):** 77 ms/step, nearly as fast. Would work for all current sessions; rejected because prefix could theoretically reach 400 tokens and responses can be long.

## Consequences

- Two new ~1.9 GB ONNX files: `models/qwen25_lm_prefill_kv.onnx`, `models/qwen25_lm_decode_kv.onnx`.
- `generate()` automatically uses KV path when both files are present.
- Memory allocation per decode step: ~96 MB of KV clones (48 × 2 MB). Acceptable at 68 ms/step total.
- `generate_kv()` uses `session.take()` / restore pattern to split borrows cleanly.
- If `PREFIX_LEN > KV_SIZE`, falls back to old `lm_session` path (rare edge case).
