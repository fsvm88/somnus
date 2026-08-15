# ADR 0044 — DER++: Stale-Output Blocking (Adopted) and Auto-λ Loss Weighting (Rejected)

**Date:** 2026-07-08
**Status:** Mixed. Stale-output blocking — Adopted, implemented (65ceaf7). Auto-λ — Closed, FAIL, opt-in flag kept as a test fixture only, default off.

## Context

Kobayashi 2025 (arXiv:2504.20932) proposes two refinements to DER++-style replay: (1) skip buffered replay samples whose stored (distilled) outputs have drifted too far from what the current model would produce, since training against a stale target under distribution shift can push the model the wrong way; (2) weight the replay loss term adaptively (auto-λ) instead of a fixed coefficient, so replay strength self-tunes to how noisy/large the replay loss currently is.

`somnus`'s NREM replay loop (`crates/somnus/src/sleep.rs`) already combines a buffered-sample loss (`sae_loss`) with a live wake-episode loss (`wake_loss`) each replay step; before this work the combination was an unweighted sum, i.e. implicit λ=1. Both ideas were investigated together under the same gaps.md item and are reported together here since they share the same source paper, the same code path, and were implemented back-to-back.

## Decision

### 1. Stale-output blocking — Adopted

Before training on a buffered replay sample, `PerceptionProcessor::infer_episode_outputs()` runs a fresh per-batch inference pass to get the model's *current* regional outputs for that sample, and compares them via cosine similarity against the outputs stored at write time. Samples whose mean cosine similarity across regions falls below `STALE_COSINE_THRESHOLD` (0.90) are skipped — treated as stale traces whose stored target no longer reflects a distribution the current model should be pulled toward. Survivors are trained/projected as before.

Implementation: `crates/somnus/src/sleep.rs` (NREM loop, threshold check ~line 646; skip counter increment ~line 649), backed by `PerceptionProcessor::infer_episode_outputs()` in `crates/somnus-perception/src/lib.rs`. Tracked via `SleepStats::stale_traces_skipped` (`crates/somnus/src/sleep.rs:197`), threaded through the stats struct and returned each NREM cycle.

This shipped unconditionally (no feature flag) — it's a filtering step, not a tunable hyperparameter, and the paper's motivating failure mode (training against drifted stale targets) is a real risk in somnus's continual-replay setting.

### 2. Auto-λ loss weighting — Rejected

`--der-auto-lambda` (f2fd185), opt-in, default off. When enabled, `PerceptionProcessor` tracks a Welford-lite EWMA (α=0.05) of the mean and mean-square of `wake_loss` (`der_lambda_stats: Option<(f32, f32)>`, `crates/somnus-perception/src/lib.rs:338`), derives a variance estimate, and weights `wake_loss` by `1/variance` (clamped to `[0.1, 10.0]`) before summing with `sae_loss` (`crates/somnus-perception/src/lib.rs:1652-1668`).

**A/B result (2026-07-08):** control (fixed λ=1, unweighted sum) averaged loss 0.0487; treatment (auto-λ) averaged 0.0847 — **+73.8% worse**. Treatment regressed on all 10/10 measured cycles, with per-cycle regressions ranging +16% to +937% and no reversals in either direction. Likely cause: with only α=0.05 EWMA smoothing and no warm-up/minimum-sample floor, early variance estimates are themselves noisy, and `1/variance` amplifies that noise into destabilizing weight spikes rather than the intended stabilization.

**Verdict: FAIL.** Not adopted. The `--der-auto-lambda` flag and `set_der_auto_lambda()` remain in the codebase opt-in/default-false, kept solely so `der_auto_lambda_stays_finite_and_loss_decreases` (`crates/somnus-perception/tests/phase5_smoke.rs:149`) can exercise the code path as a numerical-stability (finiteness) guard — not as an endorsed feature.

## Alternatives considered

- **Fixed λ ≠ 1 (manual tuning):** not attempted — the paper's premise is that a fixed coefficient is suboptimal across the run; hand-tuning one value defeats that motivation and wasn't judged worth a separate A/B given the auto-λ result already failed for a more fundamental reason (variance-estimator noise, not "the wrong fixed value").
- **No replay-loss weighting (status quo, implicit λ=1):** this is what "control" already was, and it won the A/B — so it remains the default combination going forward.
- **Windowed EWMA or minimum-sample floor before trusting the variance estimate:** identified as the likely fix but not implemented or tested — see "What would invalidate this."

## Consequences

- Stale-output blocking adds one extra forward-inference pass per buffered replay sample before training (already-budgeted GPU/CPU cost of a normal inference call), plus a cosine comparison across regions. No new failure modes beyond "a sample gets skipped" — visible directly via `stale_traces_skipped`.
- Auto-λ's rejection means replay-loss weighting stays static (λ=1) for the foreseeable future; DER++ replay strength does not self-tune to loss variance in somnus.
- The auto-λ code path (flag, EWMA state, weighting math) stays in the tree as dead-by-default machinery purely to keep a finiteness regression test meaningful. It is not wired into any production default and should not be enabled outside that test without redoing the estimator.

## What would invalidate this

- **Stale-output blocking:** if `stale_traces_skipped` is observed firing pathologically high (most/all buffered samples deemed stale) under normal campaign conditions, the 0.90 cosine threshold may be too strict for the actual drift rate and would need recalibration — not yet observed, no live evidence against the current threshold.
- **Auto-λ:** explicitly *not planned to be revisited* without a different variance estimator — specifically a windowed EWMA (bounded history instead of unbounded exponential decay) or a minimum-sample floor (don't trust `1/variance` until N samples have been observed) that would damp the early-noise-driven weight spikes implicated in the A/B failure. Absent one of those two changes, re-running the same estimator is expected to reproduce the same regression.

## Metrics

Core tier: `SleepStats::stale_traces_skipped` is already wired and firing every NREM cycle (counts samples skipped for staleness) — this is the change's primary always-on observability and needs no further work. No new failure-mode or degradation counters were introduced by stale-output blocking beyond this skip counter, since a skip is the intended behavior, not a fallback.

Auto-λ's A/B (control vs. treatment average/per-cycle loss, 10/10 cycles) was run and reported above; the feature was not adopted, so no Core-tier metric was added for it in production — the only surviving instrumentation is the existing finiteness-guard test, which is correctness-testing infrastructure, not a metrics-tier concern.

## Source

`ai-docs/engineering-log.md §3` ("Algorithm: DER++ auto-λ + stale-output blocking").
