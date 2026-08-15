# ADR 0015 — Phase 8: Generation Evaluation & Diagnostics

**Status:** accepted
**Date:** 2026-05-17

> **Decision:** Replace MSE projection loss with cosine alignment loss (prevents gradient explosion across sessions); add repetition penalty 1.3 for generation; add per-session eval diagnostics.
> **Keywords:** cosine alignment loss, MSE replacement, repetition penalty, generation, evaluation, projection training, gradient explosion

## Context

Phase 7's auto-grounding (ADR 0014) automatically sets ASR transcriptions as `text_target` on every language-ingest episode, ~135-166 trainable records/cycle. Phase 7 calibration (2 sessions × 3 cycles) confirmed the projection learns (proj_loss 0.099→0.022 within session 1), with one anomaly: S2C1 proj_loss spiked to 1160 before recovering to 0.078 by S2C3. Phase 8 goals: evaluate generation quality at the current training level, add diagnostics for the spike, fix degenerate greedy decoding.

## What was built

**`--query TEXT` flag** (one-shot evaluation): after the first episode records, `generate()` runs once with the given text, printed to stdout as `{"type":"startup_query_response","query":…,"response":…}` — enables non-interactive evaluation without a FIFO/interactive session.

**`proj_loss_max` diagnostic:** added to `SleepStats`/NDJSON, tracks max per-step projection loss during NREM alongside the existing mean — distinguishes sustained distribution shift (mean≈max, every step high, needs a training epoch) from catastrophic steps (mean≪max, a few spikes, needs optimizer tuning).

**Projection weight norm logging:** after NREM, logs `proj_weight_norm` (Frobenius norm) — expected to grow steadily during training; a spike between sessions would correlate with the S2C1 anomaly.

**Repetition penalty in `generate()`:** greedy decoding without it produces degenerate loops (all of Q1-Q3 looped). Added a standard penalty — each previously-generated token's logit divided by 1.3 before argmax. 1.3 is a typical default, made a CLI arg once generation is semantically meaningful enough to tune further.

## Generation evaluation results (pre-repetition-penalty, ~700 training examples)

All three queries used the trained projection (proj_loss≈0.078 from S2C3), fired on the first episode (no sleep, projection restored from snapshot):

| Query | Response summary |
|---|---|
| "what is being said?" | Echo loop: "What is being said? What is being said? …" |
| "describe what you observe" | Loop: ". What you observe is the result of the process of observation. …" |
| "hello" | ". The author of the book is a man named John. John is a man. …" (loop, different content) |

**Interpretation:** Q1 — LM attends to query tokens, ignores the region prefix, pure echo. Q2 — paraphrases "observe"→"observation," some query influence, no clip signal. Q3 — "John is a man" is neither the query nor a known clip theme, likely region-prefix-influenced LM prior — suggests the prefix does *something*, just not yet semantically grounded. All three loop: greedy decoding without repetition penalty always degenerates at this scale. **Expected** at ~700 examples with language prior_cosine ~0.05 (vs. 0.9 for trained regions) — the projection hasn't converged; the session-state estimate (30+ sessions for convergence) held up.

## Decision

1. **Repetition penalty 1.3** — permanent fix, loops are always wrong, penalizing repetition only helps. Not yet tuned but prevents degenerate output.
2. **More training is the primary lever for semantic quality** — no optimizer changes made for the S2C1 spike. `proj_loss_max` clarifies next session whether it was sustained or catastrophic; the advisor noted AdamW self-normalises per-parameter so loss-scaling is ineffective — a true fix would need splitting backward/step and scaling gradients before the step, if needed.
3. **`--query` flag is permanent infrastructure** — useful for cross-session regression checking.

## What to watch next

S3C1 `proj_loss_max` vs. `mean_proj_loss` (max≈mean → sustained; max≫mean → a few catastrophic steps). Language prior_cosine trajectory (should reach 0.1-0.2 within 10 sessions). Post-repetition-penalty output quality (should be qualitatively better even if still incoherent).
