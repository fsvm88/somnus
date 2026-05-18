# ADR 0016 — Phase 12a: Metrics eval harness

**Status:** accepted  
**Date:** 2026-05-18

## Context

Every sleep cycle writes a line of JSON to `memory/metrics_*.jsonl` containing
`prior_cosine_mean[5]`, `mean_loss`, `mean_output_mse`, `per_region_mse[5]`,
`warm_recall_l2`, `gist_recall_rate`, `grew`, `regions_grown`, `region_sizes`,
`sleep_trigger`, `nrem_exit_reason`, `synthetics_produced`, `mean_real_inference_mse`,
`mean_synthetic_output_mse`, and more.  There is currently no tooling to turn these
records into answers to the four recurring analysis questions:

1. Is the language projection improving over time (prior_cosine[4] trajectory)?
2. When and where does growth fire (growth event timeline by region)?
3. Is per-region output_mse converging or diverging across sleep cycles?
4. What triggers sleep and how often (trigger distribution)?

Without tooling, every investigation requires ad-hoc `jq` or manual Python.
The harness pays for itself immediately — it will be used to evaluate all subsequent phases.

## Decision

Write a Python script `scripts/eval_metrics.py` that reads one or more `metrics_*.jsonl`
files and emits four outputs from a single invocation:

1. **Language trajectory** — per-cycle line chart of `prior_cosine_mean[4]` (language
   region), overlaid with the other four region cosines for context.
2. **Growth timeline** — scatter of cycles where `grew=true`, annotated with
   `regions_grown` and the post-sleep `region_sizes`.
3. **Per-region output_mse** — stacked/grouped line chart of `per_region_mse[0..4]`
   across all cycles.
4. **Sleep trigger distribution** — bar chart of `sleep_trigger` counts
   (`high_water`, `schedule`, `perf_degradation`); secondary axis for `nrem_exit_reason`.

The script must:
- Accept one or more JSONL paths as positional arguments (glob-friendly).
- Sort cycles chronologically (by line order within files, files in argument order).
- Write four PNG files to the same directory as the first input file, or to `--out-dir`.
- Require only `matplotlib` and `numpy` (no pandas; no seaborn).
- Skip lines that are missing a field rather than crashing.

No Rust binary is needed. The harness is a read-only analysis tool and does not
touch the live system.

## Alternatives considered

- **Grafana / InfluxDB** — over-engineered for a single-developer research project;
  adds an always-on service dependency.
- **Jupyter notebook** — harder to run non-interactively; outputs not reproducible
  from a single command.
- **Generic dashboard (Weights & Biases, MLflow)** — external service; requires
  account and network; overkill for four charts.

## Exit criterion

All four charts generated from a single invocation on any `metrics_*.jsonl` file.

## What would invalidate this decision

If the metrics schema changes substantially (new field names, structural refactor),
the script must be updated. This is expected and acceptable.
