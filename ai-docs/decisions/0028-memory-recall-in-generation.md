# ADR 0028 — Memory recall in generation (R1)

**Status:** Accepted
**Date:** 2026-06-18

## Context

Generation (stdin query → Qwen2.5 autoregressive decode) only saw 5 region tokens projected from the *current* episode vector — retrieved memories never reached the LLM. The recollection layer queries warm HNSW during wake (for reconstruction gain) but that path was entirely separate from generation. R1 closes this: on a query, retrieve semantically related past episodes from warm memory and prepend their projected region tokens as additional context before the query tokens.

## Decision

**Mechanism:** on a stdin query, take the current episode vector (last hot-buffer entry) as the HNSW cue → query `RecollectionLayer::query_warm(cue, k+1)` (k=recall count, default 3; k+1 accounts for the cue episode's self-match in the index) → filter out the entry matching the most recent hot-buffer episode id, keep at most k → for each recalled record, `ActionDecoder::project_flat()` → `[N_REGIONS×LM_HIDDEN]` tokens → prepend in ascending similarity order (furthest first, closest last — nearest the query tokens in the LLM's causal window). Final layout: `[far_recall_tokens | ... | near_recall_tokens | query_tokens | generated]`. `generate()` gains `recalled_episodes: &[Vec<f32>]` (empty=no recall, existing behavior). `k=3` default, `--recall-k` CLI override, 0 disables recall entirely.

## Alternatives considered

- **Query-text cue** (the theoretically correct choice — retrieved memories relevant to what was *asked*, not last *perceived*). Rejected for now: `ingest_language` requires regions 0-3 primed (`try_bind` gate), failing in text-only interactive sessions; a stateless text→cue path needs factoring out a read-only k-WTA+partial-binding forward, a significant refactor. In the campaign context (audio/video streaming), the last episode was just produced by recent media + possibly the preceding text line — a reasonable semantic proxy. **Deliberate deviation, documented here:** recall retrieves "episodes similar to what I just perceived," not "what the user asked" — correlated in campaign evaluation, would diverge in a purely conversational deployment. A future `perception.language_cue_vector()` helper (stateless k-WTA + zero-padded binding) would enable the correct cue without state side effects.
- **k ordering.** Closest-last (nearest the query tokens) lets the LLM attend most strongly to the most relevant memory, since attention bias decays with distance in causal LMs.
- **Empty-warm guard.** Warm with fewer than k+1 records skips recall, falls back to current behavior.

## Consequences

Generation prompt grows by up to `k×N_REGIONS=3×5=15` tokens/query — negligible against Qwen2.5-0.5B's 32k context on CPU. `generate()` API changes: callers must pass `recalled_episodes` (empty slice for unchanged behavior). `--recall-k 0` disables R1 entirely, reproducing pre-R1 baselines. The cue-source deviation (current episode vs. query-text episode) should be revisited once `try_bind` can fire on language-only input, or a stateless cue helper exists.
