# ADR 0022 — Instrumentation and metrics expansion

**Status:** accepted  
**Date:** 2026-05-31

> **Decision:** Expand per-cycle NDJSON metrics to include ep_salience accumulation, wake_reconstruct throttle counters, blend_count, modality balance, NREM coverage, and per-region prior_cosine means.  
> **Keywords:** ep_salience, wake_reconstruct_counter, blend_count, asr_count, modality_balance, NREM coverage, prior_cosine_mean, metrics.jsonl, NDJSON

## Context

The per-cycle NDJSON metrics (ADR 0016) provided good visibility into NREM/REM
phases but had several dark spots that made wake-cycle calibration essentially
blind:

- `ep_salience` (warm-recall L2 distance) was computed for every episode and
  used as a gate for `wake_reconstruct_step`, but was never accumulated.
  The result: `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD = 0.8` could not be verified
  from the metrics file alone.
- `wake_reconstruct_counter` (eligible episodes) was reset without being emitted.
  The fired count (`wake_output_mse_n`) was emitted but the total attempted and
  the throttle-passed intermediate were not, making it impossible to distinguish
  "throttle blocked" from "salience gate blocked".
- The `blended` flag (episode merged with top-1 warm recall) was logged at
  `info!` per episode but never aggregated. `RECALL_BLEND_THRESHOLD = 0.5`
  was unverifiable from NDJSON.
- `wake_episodes` lumped all modalities together; modality imbalance was invisible.
- `recall.warm.len()` and `recall.cold.len()` were already accessible and
  logged every metrics tick, but absent from the per-cycle NDJSON.
- NREM replay coverage (`traces_replayed / n_traces`) was computable only if
  you knew the oversampling factor; `nrem_budget_traces` and `nrem_unique_traces`
  were not exposed.

## Decision

Add the following to the per-cycle NDJSON record and to `SleepStats`:

### Wake-cycle salience stats (A)

| Field | Type | Meaning |
|---|---|---|
| `mean_salience_wake` | `f64 \| null` | Mean warm-recall L2 across wake episodes with genuine recall (sentinel `f32::MAX` excluded). `null` when warm was empty all cycle. |
| `min_salience_wake` | `f32 \| null` | Minimum warm-recall L2 (closest recall seen). `null` when warm empty. |
| `high_salience_episodes` | `usize` | Episodes with genuine recall dist > `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD` (i.e. novel content that would trigger reconstruction if throttle passes). Sentinels excluded. |

### Wake reconstruct gate breakdown (B)

| Field | Type | Meaning |
|---|---|---|
| `wake_reconstruct_attempts` | `usize` | Episodes eligible for reconstruction (had `out.episode` and `out.regional_inputs`). |
| `wake_reconstruct_throttle_passed` | `usize` | Subset where the interval throttle (`is_multiple_of`) passed. |
| `wake_reconstruct_fired` | `usize` | Subset where both throttle AND salience gate passed (MSE was computed). |

Three-counter breakdown: `fired ≤ throttle_passed ≤ attempts`. Comparing them
isolates which gate is the binding constraint.

### Recollection blend count (C)

| Field | Type | Meaning |
|---|---|---|
| `blend_count` | `usize` | Episodes whose stored vector was blended with the top-1 warm recall (dist < `RECALL_BLEND_THRESHOLD`). |

### Memory store sizes (D)

| Field | Type | Meaning |
|---|---|---|
| `warm_len` | `usize` | Records in the warm HNSW at sleep time. |
| `cold_len` | `usize` | Records in the cold archive at sleep time. |

These were already logged every metrics tick but absent from NDJSON, making
growth trends invisible without grepping logs.

### Per-modality episode counts (E)

| Field | Type | Meaning |
|---|---|---|
| `episodes_speech` | `usize` | Episodes tagged `IngestSource::Speech`. |
| `episodes_env` | `usize` | Episodes tagged `IngestSource::Env`. |
| `episodes_visual` | `usize` | Episodes tagged `IngestSource::Visual`. |
| `episodes_language` | `usize` | Episodes tagged `IngestSource::Language`. |
| `episodes_mixed` | `usize` | All remaining (Mixed + Unknown). |

Counts key off the stored `IngestSource` derived from which regional inputs
were non-empty, NOT the receiver channel. This correctly handles `Mixed` episodes
that arrive on the audio path but activate multiple regions.

### NREM replay coverage (F)

Added to `SleepStats` and emitted in NDJSON:

| Field | Type | Meaning |
|---|---|---|
| `nrem_budget_traces` | `usize` | Total replay list length (unique + oversampled copies). |
| `nrem_unique_traces` | `usize` | Unique episodes in the buffer before language oversampling. |

Coverage = `traces_replayed / nrem_budget_traces`. Combined with `nrem_exit_reason`,
this shows whether early exits (loss target, time, salience interrupt) skipped
unique traces that sit after the oversampled prefix.

## Implementation notes

- `record_episode` now returns `Option<RecordedEpisode> { salience, blended, source }`
  instead of bare `f32`. `None` means no episode was produced (early return);
  `salience = f32::MAX` inside a `Some` means warm was empty (no recall possible).
- A shared `accumulate_salience` helper consolidates the per-episode accumulation
  at all three call sites (speech/env audio, video, language).
- Sentinel filtering uses `< f32::MAX / 2.0` (matching the pattern in `sleep.rs`
  NREM LR computation) so empty-warm episodes don't corrupt salience means.
- `ingest_language_line` return type changed from tuple to `LangIngestResult`
  struct to carry the additional fields cleanly.
- JSON recursion limit bumped to 256 (`#![recursion_limit = "256"]` in `main.rs`)
  to accommodate the expanded `serde_json::json!` macro invocation.

## Fields no longer dark after this change

`WAKE_RECONSTRUCT_SALIENCE_THRESHOLD`, `RECALL_BLEND_THRESHOLD`, warm/cold store
growth, modality balance, and NREM replay coverage are all now directly observable
in the NDJSON without log parsing.

## What would invalidate this decision

If `record_episode` is split or inlined, the shared accumulation helper should
follow. If the NDJSON schema is restructured (e.g. nested objects per phase), the
flat field names here should migrate accordingly; update ADR 0016 and this ADR
together.
