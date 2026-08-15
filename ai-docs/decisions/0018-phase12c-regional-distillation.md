---
status: accepted
date: 2026-05-18
deferred:
  - id: activation-frequency-pruning
    what: Track per-unit activation frequency during NREM replay; remove least-activated grown units instead of lowest-L1-norm
    when: When magnitude-based pruning shows clear false-positives (removing structurally important but low-magnitude units)
  - id: knowledge-distillation-compression
    what: Train a fresh smaller fixed-size encoder to match a grown encoder's outputs, then replace the grown model
    when: If grown regions exceed 2× baseline unit count and magnitude pruning degrades recall quality
---

# ADR 0018 — Phase 12c: Regional unit pruning (distillation)

**Status:** accepted
**Date:** 2026-05-18

> **Decision:** Prune grown units (index ≥ N_UNITS=1024) by L1 norm of encoder weight row when gist_recall_rate ≥ 0.8; default prune_fraction=0.5, prune_min_grown=64.
> **Keywords:** pruning, grow_region, prune_region, L1 norm, gist_recall_rate, prune_fraction, prune_min_grown, region_sizes

## Context

`grow_region` adds units on gist-recall failure during REM, unbounded — a repeatedly-growing region accumulates units indefinitely, raising forward-pass cost and memory. `to_dense_n(N_UNITS=1024)` clamping meant grown units above 1024 didn't contribute to the episode vector (binding input always N_UNITS-wide) — they participate in k-WTA winner selection and SAE reconstruction, but downstream effect was limited, creating a safe pruning scope: units above baseline could be removed without breaking snapshot compatibility or episode-vector dimensionality.

## Decision

**Scope:** only prune `grow_region`-added units above N_UNITS=1024, never below — preserves fixed indices 0-1023 in the safetensors file. **Selection:** magnitude-based — L1 norm of each candidate unit's (index≥1024) encoder weight row, remove the smallest-norm fraction (correlates with activation frequency/downstream influence). **Procedure per region:** read encoder weight rows ≥1024 → rank by L1 norm ascending, mark bottom `prune_fraction` → build a new `Var` with pruned rows removed → rebuild the SAE decoder correspondingly → rebuild the output head (He-init new rows, output dim changed) → reset that region's optimizer state (stale moment buffers after reshape) → update `region_sizes[i]`.

**Hyperparameters:** `--prune-fraction` (default 0.5, fraction of grown units removed/trigger), `--prune-min-grown` (default 64, minimum growth before pruning eligibility — avoids thrashing on small growth steps). Runs during REM, after growth check, before creative recombination. **Trigger:** `region_sizes[i] > N_UNITS+prune_min_grown` AND `gist_recall_rate≥0.8` (familiar content, safe to compress); never prunes while actively growing (`gist_recall_rate<0.5`).

**Eval dependency:** needs Phase 12a (eval harness) to confirm `per_region_mse` doesn't regress post-prune — implement 12a first, use it for the exit-criterion baseline.

## Alternatives considered

- **Activation-frequency pruning.** More accurate than magnitude but needs extra replay-time bookkeeping. Deferred — magnitude is simpler and sufficient for a first pass.
- **Knowledge distillation to a smaller fixed model.** Much higher implementation cost, not needed while grown-unit counts are small.
- **Prune below 1024.** Would break snapshot compatibility and episode-vector dimensionality — not considered.

## Exit criterion

After pruning 50% of a region's grown units, `per_region_mse` stays within 5% of the pre-prune baseline across the next sleep cycle.

## What would invalidate this decision

Routinely needing >2× baseline unit count for good gist recall would make magnitude pruning hurt performance, needing a smarter criterion or acceptance of the larger size. Removing `to_dense_n` clamping in a future phase requires revisiting the pruning scope.

## Amendment — 2026-06-14: safe-pruning assumption invalidated by items 4+5 (ef4682d)

ADR 0005/0008 items 4+5 removed the `to_dense_n(N_UNITS)` clamp from binding's forward pass — binding now uses `to_dense_n(region_n[i])`, so grown units fully participate in episode vectors and warm/cold retrieval. The "safe pruning scope" argument is now stale: pruning a grown unit changes the encoding function mid-campaign — episodes already stored were produced under the pre-prune encoder, future ones under post-prune, and the HNSW index has no record of the shift.

**Consequence:** `--prune-fraction` stays disabled by default (0.0) until a longer campaign establishes whether mid-run pruning degrades recall or is tolerable. Before enabling in production, run an A/B (pruning on vs. off) comparing `warm_recall_l2`/`gist_recall_rate`/`per_region_mse` trajectories across sessions. **(Done: ADR 0026 — FAIL, pruning stays disabled.)**
