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

`grow_region` adds units to a region's k-WTA encoder and SAE decoder when gist-level
recall fails during REM.  Growth is unbounded: a region that grows repeatedly will
accumulate units indefinitely, increasing both forward-pass cost and memory footprint.

Clamping in `to_dense_n(N_UNITS=1024)` means that grown units above 1024 **do not
contribute to the episode vector** (binding input is always N_UNITS-wide).  They
participate in k-WTA winner selection (more winners can come from the grown tail) and
in SAE reconstruction, but their downstream effect is limited.

This creates a safe pruning scope: units above the 1024 baseline are candidates for
removal without breaking snapshot compatibility or altering the episode vector
dimensionality.

## Decision

### Pruning scope

Only prune units added by `grow_region` above the N_UNITS=1024 baseline.  Never prune
below 1024.  This preserves snapshot compatibility: weights for units 0–1023 have fixed
indices in the safetensors file.

### Selection criterion

Magnitude-based pruning: for each candidate unit (index ≥ 1024), compute the L1 norm
of its encoder weight row.  Remove the fraction with the smallest norms.  L1 norm
correlates with activation frequency and downstream influence.

### Pruning procedure (per region)

1. Read `encoder.weight` rows for indices ≥ 1024.
2. Rank by L1 norm ascending; mark bottom `prune_fraction` for removal.
3. Build a new `Var` tensor with pruned rows removed.
4. Rebuild SAE decoder correspondingly (same row indices).
5. Rebuild output head (decoder output dim changes; re-init with He init for new rows).
6. Reset optimizer state for that region (moment buffers are stale after reshape).
7. Update `region_sizes[i]`.

### Hyperparameters

- `--prune-fraction f32` (default 0.5): fraction of grown units to remove per trigger.
- `--prune-min-grown usize` (default 64): only prune a region if it has grown by at
  least this many units; avoids thrashing on small growth steps.
- Pruning runs during REM, after growth check, before creative recombination.

### Trigger condition

Prune a region when:
- `region_sizes[i] > N_UNITS + prune_min_grown`, AND
- gist_recall_rate ≥ 0.8 (familiar content — system is not currently struggling,
  so safe to compress).

If the system is actively growing (gist_recall_rate < 0.5), do not prune.

### Eval signal dependency

This phase depends on Phase 12a (eval harness) to confirm that `per_region_mse` does
not regress after pruning.  Implement 12a first; use it to set the exit criterion
baseline.

## Alternatives considered

- **Activation-frequency pruning** — track per-unit activation counts during NREM
  replay; remove least-activated.  More accurate than magnitude but requires extra
  bookkeeping during replay.  Deferred; magnitude is simpler and sufficient for a first
  pass.
- **Knowledge distillation to a smaller fixed model** — train a fresh smaller encoder
  to match the grown one's outputs.  Much higher implementation cost; not needed while
  the grown unit count is small.
- **Prune below 1024** — would break snapshot compatibility and change episode vector
  dimensionality.  Not considered.

## Exit criterion

After pruning 50% of grown units in a region, `per_region_mse` (measured by eval
harness) stays within 5% of the pre-prune baseline across the next sleep cycle.

## What would invalidate this decision

If the system routinely needs >2× the baseline unit count to achieve good gist recall,
magnitude pruning will hurt performance and a smarter selection criterion (or accepting
the larger size) is needed.
If `to_dense_n` clamping is removed in a future phase, pruning scope must be revisited.

## Amendment — 2026-06-14: safe-pruning assumption invalidated by items 4+5 (ef4682d)

ADR 0005/0008 items 4+5 (commit ef4682d) removed the `to_dense_n(N_UNITS)` clamp from
the binding forward pass. Binding now uses `to_dense_n(region_n[i])`, so grown units
fully participate in episode vectors and warm/cold memory retrieval.

The "safe pruning scope" argument above is therefore stale. Pruning a grown unit now
changes the encoding function mid-campaign: episodes already stored in warm/cold were
produced under the pre-prune encoder; future episodes will be produced under the
post-prune encoder. The HNSW index has no record of this shift.

**Consequence:** `--prune-fraction` remains disabled by default (0.0) until a longer
campaign establishes whether pruning mid-run degrades recall quality or is tolerable.
Before enabling pruning in production campaigns, run an A/B with pruning on vs. off
and compare `warm_recall_l2`, `gist_recall_rate`, and `per_region_mse` trajectories
across sessions.
