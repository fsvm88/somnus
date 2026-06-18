# ADR 0029 — Multi-turn conversation (R2)

**Date:** 2026-06-18
**Status:** Accepted

## Context

Each query was fully independent. The LLM had no access to what was said one turn ago. R2
adds a sliding conversation history so the system can refer back to earlier turns within
a wake session.

## Decision

Maintain a ring buffer of `(query_token_ids, response_token_ids)` pairs on `ActionDecoder`.
On each `generate()` call, embed the buffered turns and prepend them to the input sequence
between the episode context and the current query.

**Sequence layout (R1 + R2):**
```
[recalled_0_tokens | ... | recalled_k_tokens | current_tokens | hist_turn_0 | ... | hist_turn_N | query_tokens]
```

**Token budget:** `CONV_HISTORY_BUDGET = 256` tokens of history per generation call.
Oldest turns are evicted at push-time (not at embed-time) so the buffer is always within
budget and memory is bounded across a long wake session.

**Truncation policy:** oldest-first. Turns are included in the order they were added; if a
turn's tokens fit within the remaining budget it is included, otherwise skipped (first-fit
oldest-first). No summarisation — the simplicity is appropriate at this stage.

**EOS stripping:** trailing `EOS_TOKEN_ID` tokens are stripped from `response_ids` before
storage so they are never embedded mid-context (the tokenizer's `decode(.., true)` already
strips them from the text, but the raw IDs need explicit handling).

**Buffer lifecycle:** cleared on sleep entry (`clear_conversation()` called before
`run_sleep_cycle`). Conversation history is wake-local by design — matching the
biological working-memory scope.

**All three call sites** (pre-sleep drain ~line 1476, main event loop ~line 2305, and the
`--query` one-shot ~line 2648) push turns. The one-shot fires before the main loop so its
turn is available immediately; it clears at the first sleep cycle.

**Observability:** `generate()` emits a `tracing::info!` with `history_turns`,
`history_tokens`, `recalled`, and `seq_len` before each call, making it easy to verify
that history grows across turns and resets after sleep.

## Alternatives considered

- **Summarise oldest turns instead of dropping them.** Would preserve more information but
  requires another LM call or an offline summary path. Deferred — not worth the complexity
  at this stage when generation quality itself is not saturated.
- **Separate history budget per context block.** Fixed 256 tokens for history, 128 for
  recall. Chose a simple combined check; recall tokens are already bounded by `--recall-k`
  (default 3 × 5 region tokens = 15 tokens, well within limits).
- **Store embeddings instead of IDs.** Saves re-embedding cost at generation time. Rejected
  — IDs are ~200× smaller to store, and re-embedding is cheap (CPU table lookup via
  `embed_flat`).

## What would invalidate this decision

- History budget too small: generation quality clearly limited by truncated context. Raise
  `CONV_HISTORY_BUDGET` or switch to a summarisation strategy.
- Multi-session continuity required: remove the sleep-clear or persist the buffer. Not a
  current goal.
