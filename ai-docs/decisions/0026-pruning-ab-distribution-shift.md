# ADR 0026 — A/B: Magnitude Pruning Safety Under Distribution Shift

**Date:** 2026-06-17
**Status:** Closed — FAIL. Mid-campaign magnitude pruning is unsafe at warm-store scale. Pruning disabled by default.

## Context

ADR 0018 implemented magnitude-based pruning of grown units during REM — fires when `gist_recall_rate >= PRUNE_GIST_THRESHOLD (0.8)` and a region has grown ≥`prune_min_grown (64)` units above the N_UNITS=1024 baseline.

An ADR 0018 amendment (2026-06-14, ef4682d) invalidated the original safe-pruning-scope argument: binding's forward pass changed from `to_dense_n(N_UNITS)` to `to_dense_n(region_n[i])`, so grown units now fully participate in stored episode vectors. Pruning mid-campaign therefore changes the encoding function *after* episodes are already stored — pre-prune and post-prune episodes have different-length encoding vectors, and the HNSW index has no record of the shift.

By s50 of the mixed_campaign, all five regions had grown to ~2752 units (2.7× baseline). Pruning is disabled (`--prune-fraction 0.0` default) with distribution-shift risk as the stated blocker. Per-region prior cosines at steady state (s43-50) show speech already strongest (mean 0.973) — the performance delta pruning could destroy is modest; the question is whether it destroys anything at all.

## Decision

Fresh cold A/B (5 sessions × 2 cycles): control `--prune-fraction 0.0` vs. treatment `--prune-fraction 0.5 --prune-min-grown 64` (ADR defaults), identical clips/hyperparameters both arms.

## Pre-stated win threshold

Averaged over sessions 3-5, treatment wins only if **all three** hold: `warm_recall_l2` within 15% of control; `gist_recall_rate` within 0.05; `prior_cosine_mean` per-region within 0.05 for all five regions. Failure on any one means distribution shift is real and magnitude pruning as implemented is unsafe at this growth level.

## What happens on win/loss

**Win:** enable `--prune-fraction 0.5` by default in `train_campaign.sh`, add to session-state key-facts — regions pruned back toward baseline each qualifying REM cycle, containing runaway growth. **Loss:** options are (1) activation-frequency tracking (ADR 0018 deferred item H) — prune only low-activation-count units, more conservative selection; (2) accept grown size — stop pruning, forward-pass cost grows linearly with units but isn't currently a bottleneck; (3) snapshot-aware pruning — prune + re-index warm memory at campaign end only, high implementation cost, only if (1) also fails.

## Cold-start A/B result (2026-06-17) — inconclusive

5×2 cycles, fresh cold memory. Pruning fired s2c1 (64 units, speech+visual) and s5c2 (192 units, all regions).

| Metric | Control (s3-5 avg) | Treatment (s3-5 avg) | Δ | Threshold | Verdict |
|---|---|---|---|---|---|
| warm_recall_l2 | 0.4121 | 0.4521 | +9.7% | ±15% | PASS |
| gist_recall_rate | 0.889 | 0.926 | +0.037 | ±0.05 | PASS |
| prior cosine — speech | 0.855 | 0.841 | −0.014 | ±0.05 | PASS |
| prior cosine — env | 0.817 | 0.866 | +0.049 | ±0.05 | PASS |
| prior cosine — visual | 0.599 | 0.740 | +0.141 | ±0.05 | — |
| prior cosine — symbolic | 0.772 | 0.845 | +0.073 | ±0.05 | — |
| prior cosine — language | 0.952 | 0.883 | −0.069 | ±0.05 | **FAIL** |

Language narrowly fails (−0.069) — more importantly, it crashed to 0.257 at s2c1 in treatment (the exact cycle the first prune fired, speech+visual) vs. 0.804 in control the same cycle. Speech/visual feed the cross-modal symbolic projection; pruning them perturbs binding and cascades to language. Consistent direction across all three comparable cycles rules out noise.

**Why underpowered:** item H is motivated by regions reaching 2752 units (2.7×) over 50 sessions; this cold-start run only reached 1216 units (1.2×), pruned 256 total. Distribution-shift risk scales with the size of the warm/cold store encoded pre-prune — both tiny here. The two formal passes don't transfer to the 50-session-warm-store regime where H actually matters.

**Next step:** warm-start A/B from the existing 50-session mixed_campaign memory, copied to two dirs, 3 sessions each from s51 — pruning fires immediately (gist≥0.8, all regions ~1728 grown units), captures the realistic warm/cold-store risk.

## Warm-start A/B result (2026-06-17) — FAIL

From the 50-session mixed_campaign memory (all regions ~2752 units, 1.7GB warm/cold store), 3×2 cycles. Treatment fired on 4 consecutive cycles: s51c2 (4384 units pruned), s52c1 (2192), s52c2 (1096), s53c1 (548) — 8220 units removed total, all regions 2752→~1136.

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

\* Artifact — s51c2 measures recall *before* the REM prune writes out, offsetting the mean; 3 of 4 post-prune cycles show +17-24% degradation. Underlying recall quality is worse than the average suggests. † gist FAIL is in the positive direction (treatment higher) — not a concern.

**Secondary finding — cascade:** with `prune_min_grown=64`, every cycle satisfying the gist threshold re-triggers pruning as long as grown units>64 — 4 consecutive prunes fired before falling below trigger. No inter-prune cooldown (unlike growth's `growth_cooldown`) — this amplifies distribution shift.

**env cosine** stays consistently and persistently below control after pruning begins (s52c2: 0.518 vs 0.752; s53c1: 0.667 vs 0.780; s53c2: 0.571 vs 0.720) — no recovery within the 3-session window, not a transient disruption from the initial large prune.

**Verdict: FAIL.** Two threshold conditions breached (gist positive-direction, not a concern; env cosine negative and persistent, −0.065). Per the pre-stated rule, distribution shift is confirmed at warm-store scale — pruning stays disabled.

**Note on activation-frequency pruning (item H proper):** a better unit-selection heuristic than magnitude, but doesn't address the root failure mode — changing the encoder function mid-campaign after episodes are already stored under the old weights. Would produce the same cascade and distribution shift with different units removed; doesn't rescue this result.

## Decision (final)

`--prune-fraction 0.0` remains default. Mid-campaign magnitude pruning is unsafe once the warm/cold store contains episodes encoded under a materially different encoder (>2× baseline growth). Grown regions at 2.7× baseline aren't a performance bottleneck — accepting the grown size is the lowest-cost path. If containment is ever needed, option (3) (end-of-campaign snapshot-aware pruning + HNSW re-index) is the viable path — high implementation cost, don't start without a concrete bottleneck.

## What would reopen this

- Grown regions stabilise below 1.5× baseline without pruning (tighter convergence) — this A/B becomes irrelevant.
- Binding reverts to a fixed N_UNITS slice — restores the original safe-scope argument, no A/B needed.
- A concrete performance bottleneck from grown size appears (memory pressure, latency regression) — revisit option (3).
