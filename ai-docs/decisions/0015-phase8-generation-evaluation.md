# ADR 0015 — Phase 8: Generation Evaluation & Diagnostics

**Status:** accepted  
**Date:** 2026-05-17

> **Decision:** Replace MSE projection loss with cosine alignment loss (prevents gradient explosion across sessions); add repetition penalty 1.3 for generation; add per-session eval diagnostics.  
> **Keywords:** cosine alignment loss, MSE replacement, repetition penalty, generation, evaluation, projection training, gradient explosion

## Context

Phase 7 delivered auto-grounding (ADR 0014): ASR transcriptions are now automatically set as
`text_target` on every language-ingest episode, giving ~135–166 trainable records per cycle.
Phase 7 calibration (2 sessions × 3 cycles) confirmed the projection learns (proj_loss 0.099 →
0.022 within session 1), with one anomaly: S2C1 proj_loss spiked to 1160 before recovering to
0.078 by S2C3.

Phase 8 goals: evaluate generation quality at the current training level, add diagnostics for
the spike anomaly, and fix degenerate greedy decoding.

## What was built

### `--query TEXT` flag (one-shot evaluation)

Added `--query TEXT` CLI arg: after the first episode is recorded during a run, `generate()` is
called once with the given text and the result is printed to stdout as
`{"type":"startup_query_response","query":…,"response":…}`. This enables non-interactive
evaluation without a FIFO or interactive session.

### `proj_loss_max` diagnostic

Added `proj_loss_max: Option<f32>` to `SleepStats` and the NDJSON metrics line. Tracks the
max per-step projection loss during NREM alongside the existing mean. Differentiates:
- **Sustained distribution shift**: mean ≈ max (every step is high → training epoch needed)
- **Catastrophic steps**: mean << max (a few steps spike, rest normal → optimizer tuning needed)

### Projection weight norm logging

After NREM completes, logs `proj_weight_norm` (Frobenius norm of the projection weight matrix).
Baseline: expected to grow steadily during training; if it spikes between sessions this
correlates with the S2C1 loss anomaly.

### Repetition penalty in `generate()`

Greedy decoding without repetition penalty produces degenerate loops (evaluation Q1–Q3 all
looped without it). Added a standard repetition penalty: logit for each previously generated
token is divided by 1.3 before argmax. Value 1.3 is a typical default; can be made a CLI arg
once generation is semantically meaningful enough to tune.

## Generation evaluation results (pre-repetition-penalty, ~700 training examples)

All three queries were run using the trained projection (proj_loss ≈ 0.078 from S2C3).
The `--query` fires on the first episode (no sleep; the projection is restored from snapshot).

| Query | Response summary |
|-------|-----------------|
| "what is being said?" | Echo loop: "What is being said? What is being said? …" |
| "describe what you observe" | Loop: ". What you observe is the result of the process of observation. …" |
| "hello" | ". The author of the book is a man named John. John is a man. …" (loop, different content) |

**Interpretation:**
- Query 1: LM attends to query tokens, ignores region prefix → pure echo.
- Query 2: LM paraphrases "observe" into "observation" — some query influence, no clip signal.
- Query 3: "John is a man" is neither the query nor a known clip theme — likely LM prior
  influenced by the region prefix. Suggests the prefix is doing *something*, just not
  semantically grounded yet.
- All three loop: greedy decoding without repetition penalty always degenerates at this scale.

**Expected**: at ~700 training examples with language prior_cosine ~0.05 (vs 0.9 for trained
regions), the projection has not converged. The session-state projection (30+ sessions on this
clip for convergence) was accurate.

## Decision

1. **Repetition penalty 1.3** — accepted as a permanent fix. Loops are always wrong; penalizing
   repetition only helps. The value 1.3 is not yet tuned but prevents degenerate output.

2. **More training is the primary lever for semantic quality.** No optimizer changes were made
   to address the S2C1 spike. The `proj_loss_max` field will clarify on the next session whether
   the spike was sustained (distribution shift) or catastrophic (a few bad steps). The advisor
   correctly noted that AdamW self-normalises per-parameter, so loss-scaling is ineffective; true
   fix (if needed) would require splitting backward/step and scaling gradients before step.

3. **`--query` flag is permanent infrastructure.** Useful for regression checking across sessions.

## What to watch next

- S3C1 `proj_loss_max` vs `mean_proj_loss`: if max ≈ mean, spike is sustained; if max >> mean,
  spike is a few catastrophic steps.
- Language prior_cosine trajectory: should reach 0.1–0.2 within 10 sessions.
- Post-repetition-penalty output: qualitatively better even if still incoherent.
