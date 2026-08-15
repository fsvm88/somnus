# ADR 0031 — GPU generation via fixed-shape LM (qwen25_lm_fixed.onnx)

**Date:** 2026-06-20
**Status:** Accepted

## Context

`qwen25_lm.onnx` exports Qwen2.5-0.5B with a dynamic `seq_len` axis so one model serves all lengths during autoregressive generation. MIGraphX requires static shapes and rejects dynamic axes, so generation ran entirely on CPU — each decode step a full transformer forward at the current (growing) sequence length, no KV cache, O(n)/token in the transformer and O(n²) total. `qwen25_gain.onnx`/`qwen25_text_hidden.onnx` demonstrated the fix: static shapes make both MIGraphX-eligible and fast — the same technique applies to generation.

## Decision

Export `qwen25_lm_fixed.onnx`: the Qwen2.5-0.5B base transformer (no LM head) at fixed shape `[1,MAX_GEN_SEQ=1024,896]`. Compute logits on CPU as `dot(hidden[seq_len-1], embed_table^T)`.

**Why no LM head in the GPU model:** including logits in the graph would output `[1,1024,151936]`≈620MB/decode step across PCIe, dominating wall time. Keeping the LM head off-GPU avoids this. **Why dot-with-embed-table works:** Qwen2.5-0.5B ties `lm_head.weight==embed_tokens.weight` (`tie_word_embeddings=True`); the embed table is already loaded in Rust as a flat `Vec<f32>` for the distillation path — `logit[v]=dot(hidden,embed_table[v])` is O(vocab×hidden)/step (≈136M MACs), which Rust's auto-vectorization handles without BLAS. **Why no `attention_mask` needed:** causal attention is triangular (position i attends only to 0..i) — zero-padding at `seq_len..MAX_GEN_SEQ` can't corrupt `hidden[seq_len-1]`, the same invariant `qwen25_text_hidden.onnx` already uses. **Sequence budget (MAX_GEN_SEQ=1024):** `recalled(3×5=15)+current(5)+history(256)+query(~50)+generated(200)≈526` — 1024 gives 2× headroom without changing generation limits.

## Alternatives considered

- **Dynamic seq_len + attention_mask export.** Attention_mask preprocessing introduces ScatterElements/INT64 ops MIGraphX rejects.
- **Multiple fixed-size models (32/64/128/256/512).** Complex load logic, multiple GPU allocations — no clear win over one generous fixed size.
- **KV cache.** The correct fix for the O(n²) total cost — would cut each step to O(1) attention and eliminate the buffer clone + CPU logit cost. Blocked by the same dynamic-shape constraint (past_key_values are variable-length). Deferred (done later — ADR 0034).

## Consequences

`qwen25_lm.onnx` (CPU, dynamic seq_len) kept as fallback when `qwen25_lm_fixed.onnx` is absent or `--gpu-generation` isn't set. **Correctness confirmed (2026-06-21):** exact token match between GPU and CPU paths on the same query+memory — the tied-embedding dot-product logit path is correct. **Timing (2026-06-21, gfx1201/RX 9070 XT):** GPU path is ~14× *slower* than CPU at typical session seq_lens (28-50 tokens) — 2.1s/token GPU vs. 0.14s/token CPU, confirmed per-step (not deferred compilation) via two queries in one session, both uniformly slow. Root cause not fully characterised — O(1024²) attention waste is insufficient to explain the gap (at 45 TFLOPS theoretical, a full 1024-token forward should take ~20ms, not 2s) — likely per-call overhead or immature MIGraphX support for gfx1201 (RDNA 4, early-2025 ROCm); a KV cache would reduce compute but may not help if the bottleneck is overhead-bound. Re-evaluate when ROCm matures for gfx1201. **Opt-in via `--gpu-generation`:** disabled by default, `qwen25_lm_fixed.onnx` not loaded at all without the flag (no startup cost). If a session exceeds `MAX_GEN_SEQ`, generation silently stops early — can't happen at current budgets. GainWorker note: when the MLP projection (ADR 0032) lands, `gain_from_episode()`/`proj_weights_flat()` must be updated in step, or they'll silently produce wrong gains.
