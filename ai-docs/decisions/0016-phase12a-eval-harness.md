# ADR 0016 — Phase 12a: Metrics eval harness

**Status:** accepted
**Date:** 2026-05-18

> **Decision:** eval_metrics.py reads metrics.jsonl and emits 4 charts (language trajectory, growth timeline, per-region MSE, sleep triggers); train_campaign.sh runs it automatically.
> **Keywords:** eval_metrics.py, metrics.jsonl, charts, language_trajectory, growth_timeline, per_region_mse, sleep_triggers, train_campaign.sh, session labels

## Context

Every sleep cycle writes a JSON line to `memory/metrics_*.jsonl` (`prior_cosine_mean[5]`, `mean_loss`, `mean_output_mse`, `per_region_mse[5]`, `warm_recall_l2`, `gist_recall_rate`, `grew`, `regions_grown`, `region_sizes`, `sleep_trigger`, `nrem_exit_reason`, `synthetics_produced`, `mean_real_inference_mse`, `mean_synthetic_output_mse`, more). No tooling turned these into answers to four recurring questions: is the language projection improving (prior_cosine[4] trajectory)? when/where does growth fire? is per-region output_mse converging or diverging? what triggers sleep and how often? Without tooling, every investigation needed ad-hoc `jq`/manual Python — the harness pays for itself immediately.

## Decision

`scripts/eval_metrics.py`: reads one or more `metrics_*.jsonl` files, emits four outputs in one invocation — **language trajectory** (per-cycle `prior_cosine_mean[4]` line, overlaid with the other four regions for context); **growth timeline** (scatter of `grew=true` cycles, annotated with `regions_grown`/post-sleep `region_sizes`); **per-region output_mse** (grouped line chart, `per_region_mse[0..4]` across cycles); **sleep trigger distribution** (bar chart of `sleep_trigger` counts — `high_water`/`schedule`/`perf_degradation` — secondary axis for `nrem_exit_reason`).

Requirements: accepts one or more JSONL paths (glob-friendly); sorts chronologically (line order within files, files in argument order); writes 4 PNGs to the first input's directory or `--out-dir`; `matplotlib`+`numpy` only (no pandas/seaborn); skips lines missing a field rather than crashing. No Rust binary needed — a read-only analysis tool, doesn't touch the live system.

## Alternatives considered

- **Grafana/InfluxDB.** Over-engineered for a single-developer project, adds an always-on service dependency.
- **Jupyter notebook.** Harder to run non-interactively, outputs not reproducible from a single command.
- **Generic dashboard (W&B, MLflow).** External service, needs an account and network — overkill for four charts.

## Exit criterion

All four charts generated from a single invocation on any `metrics_*.jsonl` file.

## What would invalidate this decision

A substantial metrics-schema change (new field names, structural refactor) needs the script updated — expected and acceptable.

**2026-05-31 update (ADR 0022):** 14 new fields added — `mean_salience_wake`, `min_salience_wake`, `high_salience_episodes`, `blend_count`, `wake_reconstruct_attempts`, `wake_reconstruct_throttle_passed`, `wake_reconstruct_fired`, `episodes_speech`, `episodes_env`, `episodes_visual`, `episodes_language`, `episodes_mixed`, `warm_len`, `cold_len` — plus `nrem_budget_traces`/`nrem_unique_traces` in `SleepStats`. All use `null` (not absent) when not applicable, matching the existing convention.
