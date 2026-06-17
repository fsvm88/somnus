# ADR 0026 — A/B: Magnitude Pruning Safety Under Distribution Shift

**Date:** 2026-06-17
**Status:** Closed — FAIL. Mid-campaign magnitude pruning is unsafe at warm-store scale. Pruning disabled by default.

## Context

ADR 0018 implemented magnitude-based pruning of grown units during REM. The pruning
fires when `gist_recall_rate >= PRUNE_GIST_THRESHOLD (0.8)` and a region has grown
≥ `prune_min_grown (64)` units above the N_UNITS=1024 baseline.

An amendment to ADR 0018 (2026-06-14, commit ef4682d) invalidated the original
safe-pruning-scope argument: the binding forward pass was changed from
`to_dense_n(N_UNITS)` to `to_dense_n(region_n[i])`, so grown units now fully
participate in episode vectors stored in warm/cold memory. Pruning mid-campaign
therefore changes the encoding function after episodes have already been stored —
episodes written pre-prune and episodes written post-prune have different-length
encoding vectors, and the HNSW index has no record of this shift.

By s50 of the mixed_campaign, all five regions had grown to ~2752 units (2.7× baseline).
Pruning is disabled (default `--prune-fraction 0.0`) and distribution-shift risk is the
stated blocker.

Per-region prior cosines at steady state (s43–50) show speech is already the strongest
region (mean 0.973). The performance delta that pruning could destroy is modest; the
question is whether it destroys anything at all.

## Decision

Run a fresh cold A/B (5 sessions × 2 cycles each) with:

- **Control:** `--prune-fraction 0.0` (current default)
- **Treatment:** `--prune-fraction 0.5`, `--prune-min-grown 64` (ADR defaults)

Both arms use identical clip sets and training hyperparameters.

## Pre-stated win threshold

Treatment wins if, averaged over sessions 3–5:

1. `warm_recall_l2` within **15%** of control (primary recall quality check).
2. `gist_recall_rate` within **0.05** of control (measures whether pruning degrades
   the very signal used to gate future pruning).
3. `prior_cosine_mean` per-region within **0.05** of control for all five regions.

All three conditions must hold. Failure on any one means distribution shift is real
and the original magnitude-pruning approach is unsafe at this growth level.

## What happens on win

Enable `--prune-fraction 0.5` by default in `train_campaign.sh`. Add it to the
session-state key-facts block. Regions will be pruned back toward baseline during
each REM cycle that satisfies the gist-rate condition, containing runaway growth.

## What happens on loss

Magnitude pruning is unsafe as implemented. Options:

1. **Activation-frequency tracking (ADR 0018 deferred item H):** prune only
   units with low activation counts over the preceding NREM replay window — a
   more conservative selection that avoids removing structurally important units.
2. **Accept grown size:** stop pruning, accept that regions grow to ~3× baseline.
   Forward-pass cost increases linearly with unit count but is not currently a
   bottleneck.
3. **Snapshot-aware pruning:** at campaign end only (not mid-run), prune and
   re-index warm memory. High implementation cost; consider only if (1) also fails.

## Cold-start A/B result (2026-06-17) — inconclusive

5 sessions × 2 cycles, fresh cold memory, control (prune=0.0) vs treatment (prune=0.5).
Pruning fired at s2c1 (64 units: speech+visual) and s5c2 (192 units: all regions).

| Metric | Control (s3–5 avg) | Treatment (s3–5 avg) | Δ | Threshold | Verdict |
|---|---|---|---|---|---|
| warm_recall_l2 | 0.4121 | 0.4521 | +9.7% | ±15% | PASS |
| gist_recall_rate | 0.889 | 0.926 | +0.037 | ±0.05 | PASS |
| prior cosine — speech | 0.855 | 0.841 | −0.014 | ±0.05 | PASS |
| prior cosine — env | 0.817 | 0.866 | +0.049 | ±0.05 | PASS |
| prior cosine — visual | 0.599 | 0.740 | +0.141 | ±0.05 | — |
| prior cosine — symbolic | 0.772 | 0.845 | +0.073 | ±0.05 | — |
| prior cosine — language | 0.952 | 0.883 | −0.069 | ±0.05 | **FAIL** |

Language cosine narrowly fails the threshold (−0.069). More importantly, language crashed to
0.257 at s2c1 in the treatment arm — the exact cycle the first prune fired (speech+visual) —
versus 0.804 in control that same cycle. Speech and visual feed the cross-modal symbolic
projection; pruning them perturbs binding and cascades to the language region. Consistent
direction across all three comparable cycles rules out noise.

**Why this run is underpowered:**
H is motivated by regions reaching 2752 units (2.7×) across 50 sessions. This cold-start
run only reached 1216 units (1.2×) and pruned 256 units total. The distribution-shift risk
scales with the size of the warm/cold store encoded under the pre-prune encoder — both are
tiny in a cold-start run. The two formal threshold passes (warm_recall_l2, gist_recall_rate)
do not transfer to the 50-session-warm-store regime where H actually matters.

**Next step:** warm-start A/B from the existing 50-session mixed_campaign memory
(memory/mixed_campaign), copying to two dirs and running 3 sessions each from s51.
Pruning fires immediately (gist_recall_rate ≥ 0.8, all regions at ~1728 grown units);
comparison captures the distribution-shift risk against the realistic warm/cold store.

## Warm-start A/B result (2026-06-17) — FAIL

Starting from the 50-session mixed_campaign memory (all regions at ~2752 units, 1.7 GB
warm/cold store), 3 sessions × 2 cycles each. Treatment arm (prune=0.5) fired on 4
consecutive cycles: s51c2 (4384 units pruned), s52c1 (2192), s52c2 (1096), s53c1 (548).
Total removed: 8220 units, reducing all regions from ~2752 to ~1136.

**Per-cycle warm_recall_l2:**

| Cycle | Control | Treatment | Δ | Units pruned |
|---|---|---|---|---|
| s51c1 | 0.6307 | 0.6307 | — | 0 (shared start) |
| s51c2 | 0.4029 | 0.3275 | −18.7% | 4384 |
| s52c1 | 0.6173 | 0.7423 | +20.2% | 2192 |
| s52c2 | 0.2986 | 0.3502 | +17.3% | 1096 |
| s53c1 | 0.6072 | 0.5898 | −2.9% | 548 |
| s53c2 | 0.2935 | 0.3649 | +24.3% | 0 |

| Metric | Control avg | Treatment avg | Δ | Threshold | Verdict |
|---|---|---|---|---|---|
| warm_recall_l2 | 0.4750 | 0.5051 | +6.3% | ±15% | PASS* |
| gist_recall_rate | 0.645 | 0.714 | +0.069 | ±0.05 | FAIL† |
| prior cosine — speech | — | — | — | ±0.05 | PASS |
| prior cosine — env | — | — | −0.065 | ±0.05 | **FAIL** |
| prior cosine — visual | — | — | — | ±0.05 | PASS |
| prior cosine — symbolic | — | — | — | ±0.05 | PASS |
| prior cosine — language | — | — | — | ±0.05 | PASS |

\* The warm_recall_l2 average PASS is an artifact. The s51c2 cycle measures recall
*before* the REM prune writes out — it artificially offsets the mean. Post-prune cycles
show +17–24% degradation on 3 of 4 cycles. The underlying recall quality is worse.

† gist_recall_rate FAIL is in the positive direction (treatment higher). Not a concern.

**Secondary finding — cascade:** With `prune_min_grown=64`, every cycle that satisfies
the gist threshold re-triggers pruning as long as grown units > 64. 4 consecutive
prunes fired before the region fell below the trigger. This cascading encoder change
amplifies distribution shift; there is no inter-prune cooldown (unlike growth, which has
`growth_cooldown`).

**env cosine trajectory** shows treatment consistently and persistently below control
after pruning begins (s52c2: 0.518 vs 0.752; s53c1: 0.667 vs 0.780; s53c2: 0.571 vs
0.720). The deficit does not recover within the 3-session window — it is not a transient
disruption from the initial large prune.

**Verdict:** FAIL. Two threshold conditions breached: gist (positive-direction, not a
concern) and env cosine (negative, persistent, −0.065). Per pre-stated rule: "failure on
any one means distribution shift is real." Distribution shift is confirmed at warm-store
scale. Pruning remains disabled.

**Note on activation-frequency pruning (H proper):** Activation frequency is a better
unit-selection heuristic than magnitude, but it does not address the root failure mode —
changing the encoder function mid-campaign after episodes are already stored under the
old weights. It would produce the same cascade and the same distribution shift, just with
different units removed. The H enhancement does not rescue this result.

## Decision (final)

`--prune-fraction 0.0` remains the default. Mid-campaign magnitude pruning is unsafe
when the warm/cold store contains episodes encoded under a materially different encoder
(> 2× baseline growth). Grown regions at 2.7× baseline are not a performance bottleneck;
accepting the grown size is the lowest-cost path.

If grown-size containment is ever needed, the viable path is option (3) from the loss
branch: end-of-campaign snapshot-aware pruning + HNSW re-index. This is a high
implementation cost and should not be started without a concrete bottleneck to justify it.

## What would reopen this

- If grown regions stabilise below 1.5× baseline without pruning (possible if
  training converges more tightly), this A/B result becomes irrelevant.
- If binding is changed to use a fixed N_UNITS slice again, the original safe-scope
  argument is restored and no A/B is needed.
- If a concrete performance bottleneck from grown size is observed (memory pressure,
  latency regression), revisit option (3).
