# ADR 0028 — Memory recall in generation (R1)

**Status:** Accepted  
**Date:** 2026-06-18

## Context

Generation (stdin query → Qwen2.5 autoregressive decode) currently sees only 5 region tokens projected from the *current* episode vector. Retrieved memories never reach the LLM. The recollection layer queries warm HNSW during wake (for reconstruction gain) but that path is completely separate from the generation path.

R1 closes this gap: on a query, retrieve semantically related past episodes from warm memory and prepend their projected region tokens as additional context before the query tokens.

## Decision

**Mechanism:**
1. On a stdin query, take the current episode vector (last hot-buffer entry) as the HNSW cue.
2. Query `RecollectionLayer::query_warm(cue, k+1)` where k is the recall count (default 3). Fetch k+1 to account for the cue episode being in the index (self-match).
3. Filter out the entry whose id matches the most recent hot-buffer episode id, and keep at most k results.
4. For each recalled record, call `ActionDecoder::project_flat()` to produce `[N_REGIONS × LM_HIDDEN]` tokens.
5. Prepend recalled tokens in ascending similarity order (furthest first, closest last — nearest to the query tokens in the LLM's causal window).
6. Final input layout: `[far_recall_tokens | ... | near_recall_tokens | query_tokens | generated]`.

`generate()` gains a `recalled_episodes: &[Vec<f32>]` parameter (empty slice = no recall, existing behaviour).

**k = 3 default.** `--recall-k` CLI flag allows override. 0 disables recall entirely.

## Alternatives considered

**Query-text cue (committed plan step 1):** Encode the query text through the language k-WTA to get a true query-driven episode vector. This is the theoretically correct choice (retrieved memories are relevant to what the user *asked*, not what the system last perceived). However:
- `ingest_language` requires regions 0–3 to be primed (`try_bind` gate) — fails in text-only interactive sessions.
- A stateless text→cue path would require factoring out a read-only k-WTA + partial-binding forward pass, which is a significant refactor.
- In the campaign context (audio/video streaming), the last episode was just produced by recent media + possibly the preceding text line, so it is a reasonable semantic proxy for the query subject.

**Deliberate deviation:** The cue is the current episode rather than a query-text-derived episode. This is documented here. The consequence is that recall retrieves "episodes similar to what I just perceived" rather than "episodes similar to what the user asked." For campaign evaluation these are correlated; for a purely conversational deployment they would diverge. A future `perception.language_cue_vector()` helper (stateless k-WTA + zero-padded binding) would enable the correct cue without state side effects.

**k ordering — closest last vs closest first:** Closest last (nearest to query tokens in the causal window) allows the LLM to attend most strongly to the most relevant memory, since attention bias decays with distance in causal LMs.

**Empty-warm guard:** When warm has fewer than 2 records (< k+1), recall is skipped and generation falls back to current behaviour.

## Consequences

- Generation prompt grows by up to `k × N_REGIONS = 3 × 5 = 15` tokens per query. With Qwen2.5-0.5B's 32k context limit and CPU inference, this is negligible cost.
- `generate()` API changes: callers must pass `recalled_episodes` (pass empty slice for unchanged behaviour).
- `--recall-k 0` disables R1 entirely; can be used to reproduce pre-R1 generation baselines.
- The cue-source deviation (current episode vs. query-text episode) should be revisited once `try_bind` can fire on language-only input, or once a stateless cue helper exists.
