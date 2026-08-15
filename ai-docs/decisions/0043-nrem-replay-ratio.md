# 0043. NREM replay ratio raised to N=4

- **Status:** accepted, shipped as default.
- **Date:** 2026-07-06

## Context

arXiv:2505.12512 (May 2025) reports that a replay ratio of 4-8× during consolidation improves memory retention with no observed degradation. somnus's NREM phase already replays stored traces through the binding area (`--nrem-replay-epochs`, prior default 1, i.e. one pass per sleep cycle); this ADR tests raising it.

A prior A/B (2026-06-27) had attempted this comparison but was invalid: `train_campaign.sh` hardcoded `--nrem-replay-epochs 4` regardless of what `EXTRA_ARGS` specified, so both arms silently ran at N=4 — the control was never N=1. Fixed by introducing a `REPLAY_EPOCHS` env var (default 1, matching the CLI default at the time) that both campaign arms set explicitly, removing the hardcode.

## Decision

Raise the default NREM replay ratio from N=1 to **N=4**. Both the CLI default (`--nrem-replay-epochs`, `crates/somnus/src/lib.rs`) and `scripts/train_campaign.sh`'s `REPLAY_EPOCHS` default were flipped 1→4.

## Result (2026-07-06 A/B, corrected)

5 sessions × 2 cycles, fresh cold memory both arms, h3h3/shorts corpus.

`mean_output_mse`: control (N=1) 0.2379 vs. treatment (N=4) 0.0693 — **-70.9%**, consistent direction, no reversals, all 5 regions improve monotonically in both arms.

**N=2 arm (cost/benefit knee):**

| N | avg mean_output_mse | Δ vs N=1 | wall-clock Δ |
|---|---|---|---|
| 1 | 0.2379 | — | — |
| 2 | 0.1211 | -49.1% | +7.6% |
| 4 | 0.0693 | -70.9% | +16.6% |

## Alternatives considered

1. **N=2.** Measured, real, and more cost-efficient on a per-unit basis: ~6.5% error reduction per 1% wall-clock, vs. N=4's ~4.3% per 1%. Not chosen. This is a judgment call, not a clear-cut result — N=4's larger *absolute* wall-clock tax (+16.6% vs +7.6%) was judged small enough to pay for, given the campaign's total NREM time is a modest fraction of overall wall-clock and the absolute error reduction (-70.9% vs -49.1%) is substantially larger. A researcher prioritizing wall-clock budget over consolidation quality would reasonably pick N=2 instead.
2. **N=8** (upper end of the paper's reported 4-8× range). Not tested — not run as part of this A/B, so no somnus-specific evidence either way. Left as a candidate if N=4's gains are later found insufficient.
3. **Keep N=1 (status quo).** Rejected outright: the corrected A/B shows a large, consistent, monotonic improvement at N=4 with no observed degradation, matching the source paper's claim.

## Consequences

- NREM phase wall-clock increases ~16.6% per sleep cycle by default.
- All `train_campaign.sh`-launched campaigns and any direct `somnus` invocation without `--nrem-replay-epochs` now replay each buffer 4× per cycle instead of 1×.
- The 2026-06-27 A/B result is void (compared N=4 vs N=4); this ADR's 2026-07-06 corrected run is the only valid comparison on record.

## What would invalidate this

- A campaign at larger scale (more regions grown, longer session count) shows the -70.9% mean_output_mse gain doesn't hold or reverses — re-run the A/B at that scale before trusting N=4 there.
- Wall-clock becomes a binding constraint (e.g. campaign throughput becomes the bottleneck) — revisit N=2 using the cost-efficiency numbers above, or reduce N.
- N=8 is tried and beats N=4 on cost-adjusted terms — would supersede this default.

## Metrics

No new metric was needed. This is a training-hyperparameter change on an already-instrumented path: `mean_output_mse` (existing per-region reporting) was the sole evaluation metric for the A/B, and wall-clock was read from existing campaign timing output. Both are Core-tier and already fire on every campaign run; nothing about NREM replay count required new instrumentation to evaluate or to monitor going forward.

## Source

engineering-log.md §2.
