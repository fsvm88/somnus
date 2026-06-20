# ADR 0031 — GPU generation via fixed-shape LM (qwen25_lm_fixed.onnx)

**Date:** 2026-06-20
**Status:** Accepted

## Context

`qwen25_lm.onnx` exports Qwen2.5-0.5B with a dynamic `seq_len` axis so the same model serves all sequence lengths during autoregressive generation. MIGraphX requires static shapes and rejects dynamic axes, so generation has been running entirely on CPU. Each decode step runs a full transformer forward pass at the current (growing) sequence length, no KV cache, making generation O(n) per token in the transformer and O(n²) total.

`qwen25_gain.onnx` and `qwen25_text_hidden.onnx` demonstrate the fix: static shapes make both MIGraphX-eligible and fast. The same technique applies to generation.

## Decision

Export `qwen25_lm_fixed.onnx`: the Qwen2.5-0.5B base transformer (no LM head) at fixed shape `[1, MAX_GEN_SEQ=1024, 896]`. Compute logits on CPU as `dot(hidden[seq_len-1], embed_table^T)`.

**Why no LM head in the GPU model:**  
Including logits in the ONNX graph would output `[1, 1024, 151936]` ≈ 620 MB per decode step across PCIe, which would dominate wall time. Keeping the LM head off-GPU avoids this.

**Why dot with embed_table works:**  
Qwen2.5-0.5B ties `lm_head.weight == embed_tokens.weight` (`config.tie_word_embeddings = True`). The embed table is already loaded in Rust as a flat `Vec<f32>` for the distillation path. Computing `logit[v] = dot(hidden, embed_table[v])` is O(vocab × hidden) per step (≈136M MACs), which Rust's auto-vectorization handles without BLAS.

**Why no `attention_mask` is needed:**  
Causal attention is triangular: position `i` attends only to `0..i`. Zero-padding at positions `seq_len..MAX_GEN_SEQ` cannot corrupt `hidden[seq_len-1]`. This is the same invariant used by `qwen25_text_hidden.onnx`.

**Sequence length budget (MAX_GEN_SEQ = 1024):**  
`recalled (3×5=15) + current (5) + history (256) + query (~50) + generated (200) ≈ 526`. 1024 gives 2× headroom without changing generation limits.

## Alternatives considered

**Export with dynamic seq_len and attention_mask:** The attention_mask preprocessing introduces ScatterElements or INT64 ops that MIGraphX rejects.

**Multiple fixed-size models (32/64/128/256/512):** Complex load logic, multiple GPU allocations. No clear win over a single generous fixed size.

**KV cache:** Correct fix for the O(n²) total-cost problem; would reduce each step from O(n) to O(1) in attention, and eliminate both the 3.7 MB buffer clone and the CPU logit cost. Blocked by the same dynamic-shape constraint (past_key_values are variable-length). Deferred.

## Consequences

- `qwen25_lm.onnx` (CPU, dynamic seq_len) is kept as fallback for generation when `qwen25_lm_fixed.onnx` is absent or `--gpu-generation` is not set.
- **Correctness confirmed (2026-06-21):** exact token match between GPU and CPU paths on same query + memory. Tied-embedding dot-product logit path is correct.
- **Timing result (2026-06-21, gfx1201 / RX 9070 XT):** GPU path is ~14× slower than CPU at typical session seq_lens (28–50 tokens): 2.1s/token GPU vs 0.14s/token CPU. Confirmed per-step cost (not deferred compilation) by issuing two queries in the same session — both uniformly slow. Root cause not fully characterised: O(1024²) attention waste is insufficient to explain the gap (at 45 TFLOPS theoretical, even a full 1024-token forward should take ~20ms, not 2s), so the dominant factor is likely per-call overhead or immature MIGraphX support for gfx1201 (RDNA 4, early 2025 ROCm). KV cache would reduce compute, but if the bottleneck is overhead-bound it may not help. Re-evaluate when ROCm matures for gfx1201.
- **Opt-in via `--gpu-generation`:** the GPU path is disabled by default. When the flag is absent, `qwen25_lm_fixed.onnx` is not loaded at all (no startup cost). Pass `--gpu-generation` explicitly to activate.
- If any region exceeds `MAX_GEN_SEQ` tokens during a very long session, generation silently stops early (break). With current budgets this cannot happen.
- GainWorker note: when task 2 (MLP projection) lands, `gain_from_episode()` and `proj_weights_flat()` must be updated — they reimagine the projection as a single matmul and will silently produce wrong gains if the MLP is not also reflected there.
