# ADR 0029 — Multi-turn conversation (R2)

**Date:** 2026-06-18
**Status:** Accepted

## Context

Each query was fully independent — the LLM had no access to what was said one turn ago. R2 adds a sliding conversation history so the system can refer back to earlier turns within a wake session.

## Decision

Maintain a ring buffer of `(query_token_ids, response_token_ids)` pairs on `ActionDecoder`. Each `generate()` call embeds buffered turns and prepends them between episode context and the current query:

```
[recalled_0_tokens | ... | recalled_k_tokens | current_tokens | hist_turn_0 | ... | hist_turn_N | query_tokens]
```

**Token budget:** `CONV_HISTORY_BUDGET=256` tokens of history per call — oldest turns evicted at push-time (not embed-time) so the buffer stays within budget, bounding memory across a long wake session. **Truncation:** oldest-first order, first-fit — a turn is included if its tokens fit the remaining budget, else skipped. No summarisation, appropriate at this stage. **EOS stripping:** trailing `EOS_TOKEN_ID` stripped from `response_ids` before storage so they're never embedded mid-context (the tokenizer's `decode(..,true)` already strips them from text, but raw IDs need explicit handling). **Lifecycle:** cleared on sleep entry (`clear_conversation()` before `run_sleep_cycle`) — wake-local by design, matching biological working-memory scope. **All three call sites** (pre-sleep drain, main event loop, the `--query` one-shot) push turns — the one-shot fires before the main loop so its turn is immediately available, clearing at the first sleep cycle. **Observability:** `generate()` logs `history_turns`/`history_tokens`/`recalled`/`seq_len` before each call, verifying history grows across turns and resets after sleep.

## Alternatives considered

- **Summarise oldest turns instead of dropping.** Preserves more info but needs another LM call or offline summary path — deferred, not worth the complexity while generation quality itself isn't saturated.
- **Separate history budget per context block.** Chose one simple combined check (256 history + recall bounded separately by `--recall-k`, default 3×5=15 tokens, well within limits) over per-block budgets.
- **Store embeddings instead of IDs.** Would save re-embedding cost — rejected, IDs are ~200× smaller to store and re-embedding is cheap (CPU table lookup via `embed_flat`).

## What would invalidate this decision

History budget too small (generation clearly context-limited) → raise `CONV_HISTORY_BUDGET` or switch to summarisation. Multi-session continuity required → remove the sleep-clear or persist the buffer (not a current goal).
