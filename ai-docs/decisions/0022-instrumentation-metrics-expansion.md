# ADR 0022 — Instrumentation and metrics expansion

**Status:** accepted
**Date:** 2026-05-31

> **Decision:** Expand per-cycle NDJSON metrics to include ep_salience accumulation, wake_reconstruct throttle counters, blend_count, modality balance, NREM coverage, and per-region prior_cosine means.
> **Keywords:** ep_salience, wake_reconstruct_counter, blend_count, asr_count, modality_balance, NREM coverage, prior_cosine_mean, metrics.jsonl, NDJSON

## Context

The per-cycle NDJSON (ADR 0016) gave good NREM/REM visibility but left wake-cycle calibration essentially blind: `ep_salience` (warm-recall L2, gates `wake_reconstruct_step`) was computed per episode but never accumulated, so `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD=0.8` was unverifiable from the metrics file. `wake_reconstruct_counter` (eligible episodes) reset without being emitted — the fired count (`wake_output_mse_n`) was logged but the attempted total and throttle-passed intermediate weren't, so "throttle blocked" vs. "salience gate blocked" was indistinguishable. The `blended` flag was logged per-episode at `info!` but never aggregated, leaving `RECALL_BLEND_THRESHOLD=0.5` unverifiable. `wake_episodes` lumped all modalities, hiding imbalance. `recall.warm.len()`/`recall.cold.len()` were logged every metrics tick but absent from per-cycle NDJSON. NREM replay coverage (`traces_replayed/n_traces`) needed knowing the oversampling factor by hand; `nrem_budget_traces`/`nrem_unique_traces` weren't exposed.

## Decision

Add to the per-cycle NDJSON and `SleepStats`:

**Wake-cycle salience stats (A):** `mean_salience_wake` (f64|null — mean warm-recall L2 across episodes with genuine recall, sentinel `f32::MAX` excluded, null if warm empty all cycle), `min_salience_wake` (f32|null — closest recall seen), `high_salience_episodes` (usize — episodes with genuine recall dist > threshold, i.e. novel content that would trigger reconstruction if throttle passes).

**Wake reconstruct gate breakdown (B):** `wake_reconstruct_attempts` (eligible episodes, had `out.episode`+`out.regional_inputs`), `wake_reconstruct_throttle_passed` (subset passing the interval throttle), `wake_reconstruct_fired` (subset passing both throttle and salience gate, MSE computed). Three-counter chain `fired≤throttle_passed≤attempts` isolates the binding constraint.

**Recollection blend count (C):** `blend_count` (episodes blended with top-1 warm recall, dist<`RECALL_BLEND_THRESHOLD`).

**Memory store sizes (D):** `warm_len`, `cold_len` — already logged per metrics tick but missing from NDJSON, making growth trends invisible without log-grepping.

**Per-modality episode counts (E):** `episodes_speech`/`episodes_env`/`episodes_visual`/`episodes_language` (keyed off the stored `IngestSource`, not the receiver channel — correctly handles Mixed episodes arriving on the audio path that activate multiple regions), `episodes_mixed` (Mixed+Unknown).

**NREM replay coverage (F):** `nrem_budget_traces` (total replay list length, unique+oversampled), `nrem_unique_traces` (unique episodes before language oversampling). Coverage = `traces_replayed/nrem_budget_traces`; combined with `nrem_exit_reason`, shows whether early exits skipped unique traces sitting after the oversampled prefix.

## Implementation notes

`record_episode` now returns `Option<RecordedEpisode>{salience, blended, source}` instead of bare f32 — `None`=no episode produced, `salience=f32::MAX` inside `Some`=warm was empty. A shared `accumulate_salience` helper consolidates per-episode accumulation across all three call sites (speech/env audio, video, language). Sentinel filtering uses `< f32::MAX/2.0` (matching `sleep.rs`'s NREM LR computation) so empty-warm episodes don't corrupt means. `ingest_language_line`'s return type changed tuple→`LangIngestResult` struct to carry the new fields cleanly. JSON recursion limit bumped to 256 (`#![recursion_limit="256"]`) for the expanded `json!` invocation.

## Fields no longer dark after this change

`WAKE_RECONSTRUCT_SALIENCE_THRESHOLD`, `RECALL_BLEND_THRESHOLD`, warm/cold store growth, modality balance, and NREM replay coverage are all directly observable in NDJSON without log parsing.

## What would invalidate this decision

If `record_episode` is split or inlined, the shared accumulation helper should follow. If the NDJSON schema is restructured (nested objects per phase), migrate these flat field names accordingly and update ADR 0016 alongside this ADR.
