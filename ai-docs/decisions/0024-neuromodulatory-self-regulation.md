---
status: accepted
date: 2026-06-14
---

# ADR 0024 — Neuromodulatory self-regulation: derived signals replace fixed thresholds

**Status:** accepted
**Date:** 2026-06-14

> **Decision:** Replace fixed-threshold CLI knobs with internal controllers driven by signals derived from the system's own runtime state. Two layers — signal design (what to measure) and control law (how to regulate) — staged by signal cleanliness. Phase 1: adaptive salience gate + settle-rate modulation. Phase 2: wakefulness pressure homeostat. Growth controller deferred behind a documented prerequisite.
> **Keywords:** neuromodulation, homeostasis, EWMA controller, salience gate, settle-rate, wakefulness pressure, prior-beta, growth controller, within-session gist, self-regulation, bootstrap prior

---

## Context

The system has accumulated ~40 CLI flags, most of them one-time calibration choices that calcified: `--sleep-trigger-fraction 0.8` (chosen once for a 4096-capacity setup, never revisited), `--settle-gate 0.85` (arbitrary, unknown if right post-convergence), `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD=0.8` (hardcoded, unverifiable against real novelty), `--growth-cooldown 3` (a band-aid over contaminated signal, see below), `--prior-beta 0.0` (manually enabled once "converged," no signal tells you when).

**This is a control-systems argument, not a biological one:** a fixed threshold is a constant-gain controller with no feedback; an EWMA-based threshold is a first-order adaptive controller with known stability properties. The neuromodulator/arousal/homeostasis vocabulary is a mnemonic, not a design warrant.

**Two layers must stay separate:** (1) **signal design** — does this measurement track what we care about, is it bounded/low-variance/defined at cold start/available in-loop; (2) **control law** — given a clean signal, the decision rule, setpoint, deadband, stability guard. Conflating them caused the growth runaway: `gist_recall_rate` was a plausible controller input but not a clean signal — it collapsed from cross-session NREM embedding drift, independent of whether the region needed capacity.

**Long-clip campaign (2026-06-13), 5×3 cycles, CAPACITY=4096, 49k warm records:** c3 gist declined monotonically 0.714→0.746→0.634→0.36→0.22; c2 collapsed by s4-s5 (0.0, 0.0). Tracks index size + cumulative drift, not capacity need — confirms `gist_recall_rate` is not a clean growth signal (ADR 0023's cooldown rate-limits the damage, doesn't fix the measurement).

---

## Three-bucket classification of all flags

Not every knob is a regulation candidate — forcing a learning signal onto a structural/perf knob just adds noise.

**Bucket R — regulation candidates** (optimal value changes with system state): `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD` (const, 0.8, gates wake_reconstruct firing), `--nrem-salience-threshold` (0, NREM interrupt on pending novel embeddings), `--sleep-trigger-fraction` (0.8, sleep onset), `--max-wake-secs` (600, safety valve), `--settle-gate` (0.85, settle firing), `--settle-every` (1, settle throttle), `--prior-beta` (0.01, prior blend base; S4 scales to 0.05), `--perf-sleep-threshold` (0.0, early sleep on EWMA drop), `--growth-cooldown`+`GIST_GROWTH_THRESHOLD` (3/0.85, region growth).

**Bucket P — bootstrap priors** (set once at session start, become the controller's cold-start estimate once live): `--episodic-capacity` (4096), `--working-memory-size` (0), `--settle-steps` (0), `--settle-beta` (0.5), `--replay-batch-size` (128), `--language-oversample` (0), `--nrem-min-traces` (0), `--nrem-loss-target` (0.0).

**Bucket S-query — startup-queryable priors** (seeded from environment at startup, don't regulate mid-session):

| Flag | Default | Startup query | Status |
|---|---|---|---|
| `--max-warm-records` | 0 (auto) | `free_ram_bytes×0.35/2500` bytes/record via sysinfo | Done (S5, 88b4080) |
| `--recall-ef` | 16 | calibrated at scale, benchmark deferred | Done (S5, 88b4080) |
| `--max-cold-records` | 0 | `free_disk_bytes×0.20/COLD_RECORD_BYTES(661)` | Done (P1a, 7964eb3) |

**S5 note:** `0` now means "auto-query" (was "unlimited") — no flag rename, the `--init-` prefix scheme was skipped (CLAUDE.md: no backcompat shims for research code). `sysinfo 0.38` was already a transitive dep.

**Bucket S-safety — safety valves** (adaptive ceiling, fixed floor): S6 (88b4080) widens `--max-wake-secs`'s ceiling to `max(600, 3×wake_ewma_mu)` after N≥3 cycles, α=0.3/cycle, persisted `snapshot/wake_ewma.bin`. `--max-sleep-secs`: `max(300, 3×mean_sleep_secs_rolling)`, 5-cycle rolling window (P1b, 7964eb3). `--max-cycles`: permanent manual (test harness). **S6 validation:** inert in fast-clip runs (S3 sleeps at ~120 episodes/<30s wake → `3×wake_ewma_mu≈90s<600s floor`); activates in long-clip runs approaching the floor (602s schedule trigger observed).

**Bucket S-flags — feature flags** (all default `true` — historical `false` defaults were staged-rollout accidents, flipped in 6d42f2a): `--language`, `--action`, `--wake-recon-async`, `--action-gain`, `--migraphx`. `somnus --inputs <file>` now works with no subsystem flags.

**Bucket S-throttle — deferred, no clean signal yet:** `--wake-reconstruct-interval` (5 → done, S7: decrements toward 1 on wake-MSE-EWMA stagnation <1% improvement/3 cycles, 7964eb3). `--gain-update-every` (20, potential signal: `gain[i]` variance — deferred). `--top-k-recall` (5, potential signal: recall diversity ratio).

**Bucket S-ops — permanent manual** (hardware/user-intent, no learning signal substitutes): `--video-stride` (1, content-rate tradeoff), `--replay-batch-size` (128, BLAS batch size), `--max-tokens` (128, user intent), `--language-oversample` (0, curriculum), `--nrem-min-traces` (0, safety floor), `--asr-min-chars` (20, noise filter), plus paths/session-id/ort-profiling/inputs.

---

## Decision: implementation sequence

Signal cleanliness governs sequencing, not knob priority.

| Phase | Controller | Signal clean? | Implements |
|---|---|---|---|
| 1a | Adaptive salience gate | Yes — L2 distance, well-observed | Replaces `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD` + `--nrem-salience-threshold` |
| 1b | Settle-rate modulation | Yes — prior_cosine already tracked | Replaces `--settle-gate` + (later) `--settle-every` |
| 2 | Wakefulness pressure homeostat | Mostly — safeties needed | Replaces `--sleep-trigger-fraction` + `--max-wake-secs` |
| 2b | Prior-beta auto-enable | Yes — prior_cosine mean | Replaces `--prior-beta` manual enable |
| 2c | Growth controller | Yes (c2+); noisy at c1 | `within_session_gist_rate` primary, cross-session fallback at c1, cooldown retained |
| S5 | Startup-queryable priors | Stateless per-startup | `--max-warm-records 0` → auto; `--recall-ef` default 16 |
| S6 | Adaptive wake ceiling | Stateful EWMA/cycle | `max_wake=max(floor, 3×wake_ewma_mu)` after N≥3 |

---

## Signal and controller specifications

**Controller principles for all signals:** cold-start = current CLI default until `N_WARMUP` samples exist (controller-off ≡ today's behavior, a free A/B baseline); every state variable goes into per-cycle NDJSON, no dark state; the replaced knob's CLI flag keeps seeding the controller's cold-start prior, not deleted.

### S1 — Adaptive salience gate (1a)

**Replaces:** `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD=0.8` (const) + the salience-count component of `--nrem-salience-threshold`. **Controls:** whether an episode's warm-recall L2 is novel enough to trigger `wake_reconstruct_step` / count as pending during NREM.

**Signal:** `ep_salience` (warm-recall L2, already in `mean_salience_wake`/`min_salience_wake`, ADR 0022) — per-episode, bounded below at 0, sentinel `f32::MAX` already filtered, well-defined at cold start (empty warm → all novel).

```
μ_s[t] = α·ep_salience[t] + (1-α)·μ_s[t-1]     α=0.01
σ_s[t] = α·|ep_salience[t]-μ_s[t]| + (1-α)·σ_s[t-1]
τ_s[t] = clamp(μ_s[t] + 0.5·σ_s[t], 0.3, 3.0)
```

k=0.5 targets ~top-third novelty. **Range:** μ_s 0.8-1.5, σ_s 0.2-0.6, τ_s 0.9-2.1 typical. **Cold-start:** τ_s=0.8 until N=100. `--nrem-salience-threshold`'s count gate now uses τ_s instead of a fixed L2 threshold; the count itself stays manual (throughput knob). Persisted `snapshot/sal_ewma.bin`. NDJSON: `sal_ewma_mean`, `sal_ewma_std`, `sal_threshold`. **Invalidates if:** modality mix shifts the salience distribution dramatically (e.g. pure-language near-zero L2) — monitor `sal_threshold` vs. modality breakdown, stratify if needed.

### S2 — Settle-rate modulation (1b)

**Replaces:** `--settle-gate 0.85` + eventually `--settle-every N`. **Controls:** how often the predictive-coding settle correction fires (~2-3ms/ep) — should fire more when uncertain, less when accurate.

**Signal:** `prior_cosine` (output-head prediction vs. actual input, already in `prior_cosine_mean`) — bounded [-1,1], every episode.

```
C[t] = α·prior_cosine_global[t] + (1-α)·C[t-1]     α=0.005
fire when prior_cosine_global < C[t] - 0.1
```

**Range:** C[t] 0.3-0.7 post-convergence; Δ=0.1 fires ~30-50% when unconverged, <10% when converged. **Cold-start:** C[t]=0.85 until N=200. **Settle-every follow-on:** C[t]>0.6 for ≥3 cycles → double settle-every (max 4); C[t]<0.4 → reset to 1. NDJSON: `settle_cosine_ewma`, `settle_gate_effective`, `settle_every_effective`. **Invalidates if:** prior_cosine has high per-region variance under a global mean (it does — c1 low, c3 high) — stratify by region if one sits >0.2 below global.

### S3 — Wakefulness pressure homeostat (2)

**Replaces:** `--sleep-trigger-fraction 0.8`, `--max-wake-secs 600`. **Current mechanism** (fires on volume/time, not learning): sleep at 80% capacity fill or 600s hard limit.

**Signal:** accumulator P(t)∈[0,∞). Per-episode: `P += novelty_weight(ep_salience)/baseline_rate` where `novelty_weight(s)=max(0, s-τ_s)`. Per-NREM-cycle dissipation: `P -= consolidation_quality(gist_recall_rate, nrem_loss_ratio) ∈ [0,0.5]`. Sleep fires at `P≥1.0`.

**Setpoint:** ~3000-4000 episodes/sleep for typical varied-novelty wake. **Cold-start:** capacity-fill trigger (`--init-sleep-trigger-fraction`, default 0.8) until N=500 wake episodes; P accumulates in parallel so the controller is ready at warmup end. **Stability guards:** hard min 120 episodes before sleep; `--max-wake-secs` kept as a safety valve; P clamped [0,3.0]. **`--perf-sleep-threshold` interaction:** becomes a P impulse (`+0.5` capped) on prior_cosine EWMA drop >0.2, renamed `--perf-sleep-pressure-boost`. NDJSON: `wakefulness_pressure`, `wake_trigger`, `wake_episodes_until_pressure_sleep`. **Invalidates if:** novelty_weight miscalibrated (too-frequent short sleeps) or NREM dissipation too weak (contaminated gist) — `baseline_rate` stays tunable; capacity trigger stays active in parallel as fallback.

### S4 — Prior-beta scale-up controller (2b)

**Replaces:** `--prior-beta 0.0` manual enable. **Controls:** top-down prior blend weight β.

**Amendment (2026-06-14, chicken-and-egg fix):** original design toggled 0.0→0.05, but `blend_with_prior` returns `None` cosine when `prior_beta≤0.0`, so the ring buffer never filled at cold start. Fixed with a small non-zero base (`--prior-beta 0.01`): cosines always measured, blend negligible early (1% noise), S4 scales to 0.05 once mature.

**Signal:** `prior_cosine_mean` (global, rolling), 3-cycle ring buffer:
```
scale up (β=0.05):   all 3 ring values > 0.65
scale down (β=base):  any ring value  < 0.40
```
Hysteresis gap [0.40,0.65] prevents oscillation; only engages when base<0.05. **Cold-start:** β=0.01, cosines measured from episode 1. NDJSON: `prior_cosine_mean`, `prior_beta_effective`. **Invalidates if:** prior_cosine stays <0.65 indefinitely for architectural/content reasons (not convergence) — expose `--prior-beta-threshold` override; reduce base if it measurably hurts learning.

### Growth controller — within-session signal (2c)

**Prerequisite resolved (7214d16):** `SemanticStore::records_since(session_start_secs)` gives a clean within-session warm subset; `within_session_gist_rate` emitted every cycle. Necessary because the long-clip campaign showed cross-session gist declining independent of actual capacity need — regulating on it would never stop growing.

```
quality_poor = match (within_session_gist_rate, gist_recall_rate) {
    (Some(ws), _)   => ws < GIST_GROWTH_THRESHOLD,          // c2+: clean
    (None, Some(_)) => gist_poor,                            // c1: noisy fallback
    (None, None)    => per_region_mse[i] > GROWTH_THRESHOLD, // S1C1: no warm yet
}
```

`within_session_gist_rate` is always `None` at c1 of every session. Pruning gate switched to `.or()` the same way (`None`→no pruning, conservative). Pressure dissipation in main.rs uses the same preference. **Cooldown retained** rather than the originally-specified plasticity depletion/recovery model — c1 stays noisy, the cooldown backstops it, and the plasticity model adds complexity with no benefit until c1 itself is solved (needs session-tagged episodes or cross-cycle within-session accumulation). NDJSON: `gist_recall_rate`, `within_session_gist_rate`.

---

## Full mapping table

| Knob (default) | Gates | Signal | Controller | Setpoint | Range | Cold-start | Guard | NDJSON | Phase |
|---|---|---|---|---|---|---|---|---|---|
| `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD=0.8` | wake_reconstruct firing | L2 salience EWMA | τ=μ+0.5σ, α=0.01 | ~top-40th pctile | μ∈[0.5,2.0], τ∈[0.3,3.0] | 0.8 until N=100 | clamp+sentinel | `sal_ewma_mean/std`, `sal_threshold` | 1a |
| `--nrem-salience-threshold 0` | NREM interrupt count | τ_s from S1 | count L2>τ_s, interrupt>N | — | — | 0 (off) | N manual | `nrem_interrupt_count` | 1a |
| `--settle-gate 0.85` | settle firing | prior_cosine EWMA C[t] | fire if <C[t]-0.1, α=0.005 | below-average eps | C∈[0,1], 0.3-0.7 typ | 0.85 until N=200 | Δ floor 0.05 | `settle_cosine_ewma`, `settle_gate_effective` | 1b |
| `--settle-every 1` | settle throttle | C[t] trend | +1 if C>0.6×3cyc; reset if C<0.4 | auto-scale | [1,4] | 1 | step, max 4 | `settle_every_effective` | 1b |
| `--sleep-trigger-fraction 0.8` | sleep onset | P novelty accumulator | +=novelty/baseline; sleep P≥1.0 | ~3500 eps/sleep | P∈[0,3.0] | capacity-0.8 until N=500 | min 120 eps | `wakefulness_pressure`, `wake_trigger` | 2 |
| `--max-wake-secs 600` | sleep onset (safety) | wake-duration EWMA | max(floor,3×μ), α=0.3/cyc | widens with wakes | floor=600s | floor until N≥3 | floor inviolable | `wake_ewma_mean`, `max_wake_secs_effective` | S6 |
| `--perf-sleep-threshold 0.0` | early sleep | P impulse on cosine collapse | drop>0.2→P+=0.5 | — | [0,0.5] | 0.0 (off) | capped 0.5/cyc | `perf_sleep_boost` | 2 |
| `--prior-beta 0.01` | prior blend weight | 3-cycle cosine ring | up 0.05 if all>0.65; down if any<0.40 | hysteresis [0.40,0.65] | [0,1] | 0.01 base | hysteresis | `prior_beta_effective`, `prior_cosine_mean` | 2b |
| `--growth-cooldown 3`+threshold | region growth | ws primary, gist@c1, MSE@S1C1 | ws<0.85→grow; cooldown backstop | — | ws∈[0,1], None@c1 | cooldown=3 | None→no-prune@c1 | `gist_recall_rate`, `within_session_gist_rate` | 2c |

---

## Implementation notes

1. **Knob renaming:** replaced `--foo` → `--init-foo` (same default), old flag kept as deprecated alias one release, then removed — makes explicit it seeds the controller, not the threshold.
2. **Persistence:** each controller's EWMA state in `snapshot/<controller>.bin`, same u64-LE pattern as `growth_cooldown.bin`; multiple EWMAs as a flat `[f32;N]`.
3. **A/B validation:** each Phase-1 controller validated with a small fixed-seed run (2×3) comparing on vs. off (`--init-foo` only) — `settle_every_effective`, `sal_threshold`, `mean_output_mse_wake` trajectory.
4. **Observability discipline:** every new state variable in per-cycle NDJSON before an ADR item counts as implemented — no dark state (ADR 0022).
5. **Biology is analogy.** "Wakefulness pressure" is a leaky-integrator homeostat with a known setpoint and stability bounds — the control-systems justification is sufficient; no neuronal-fidelity claim is made.

## What would invalidate this decision

- EWMA controllers stabilizing slower than fixed thresholds on short clips (<500 eps/session) — cold-start fallback prevents regression, but Phase-1 controllers may need stronger init.
- prior_cosine staying persistently low across all domains (not convergence-dependent) — S2/S4 would have no dynamic range; fix the cosine signal first.
- S3's `baseline_rate` hard to calibrate across clip types (novelty varies 10×) — may need per-clip-type priors; fall back to capacity trigger.
- If gist stays contaminated after the within-session fix, the growth controller needs a different capacity signal (e.g. per-region MSE trend) once `--growth-cooldown`/`GIST_GROWTH_THRESHOLD` are due for retirement.

---

## Amendment — S4 chicken-and-egg fix + full validation (2026-06-14)

**Bug found during confirmatory run:** original S4 gated on `prior_beta==0.0` and started β at 0; `blend_with_prior` returns `None` cosine whenever `prior_beta≤0.0`, so the ring never filled and S2's settle EWMA (which also tracks cosines) froze at cold-start indefinitely.

**Fix:** `--prior-beta` default 0.0→0.01 (self-calibrating: negligible blend early, directional once predictions improve — no `blend_with_prior` change needed). S4 guard changed to `prior_beta<0.05`; on scale-down β resets to base, not 0.0. `train_campaign.sh`'s `PRIOR_BETA` env var defaults 0.01.

**Confirmatory run (2s×3c, fresh memory, CAPACITY=4096):**

| cycle | session | trigger | wake_eps | sal_threshold | settle_gate_dynamic | wakefulness_pressure | prior_beta_effective | prior_cosine (r0–r4) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | high_water | 3276 | 0.800 | 0.638 | 0.000 | 0.010 | [0.517,0.603,0.496,0.680,0.292] |
| 2 | 1 | pressure | 500 | 1.023 | 0.576 | 2.796 | 0.010 | [0.653,0.736,0.506,0.699,0.209] |
| 3 | 1 | pressure | 120 | 0.825 | 0.618 | 2.611 | 0.010 | [null,null,0.667,0.836,null] |
| 1 | 2 | pressure | 287 | 0.345 | 0.697 | 0.732 | 0.010 | [0.713,0.889,0.620,0.843,0.328] |
| 2 | 2 | pressure | 120 | 0.424 | 0.567 | 2.504 | 0.010 | [0.733,0.935,0.526,0.739,null] |
| 3 | 2 | pressure | 120 | 0.519 | 0.593 | 2.645 | **0.050** | [null,null,0.628,0.770,null] |

S1 ✅ `sal_threshold` adapts off 0.8 from cycle 2. S2 ✅ `settle_gate_dynamic` tracks ~0.10 below `settle_ewma_mean`. S3 ✅ pressure triggers sleep from cycle 2, 120-ep min guard active. S4 ✅ `prior_beta_effective` scales to 0.05 at S2C3 (3 consecutive cycles, active-region cosines >0.65).

---

## Amendment — Growth controller implemented (2026-06-14)

**Within-session probe (7214d16):** `SemanticStore::records_since()` filters warm by timestamp; both k-means centroids and NN search use only within-session records, so cross-session drift can't contaminate either. `within_session_gist_rate` always `None` at c1.

**Validation run (5s×3c, CAPACITY=4096, fresh memory):**

| s | c | gist_recall | ws_gist | grew | notes |
|---|---|---|---|---|---|
| 1 | 1 | None | None | **Y** | S1C1: MSE-driven (warm empty) |
| 1 | 2 | 0.646 | 0.555 | — | cooldown |
| 1 | 3 | 0.930 | 0.969 | — | cooldown |
| 2 | 1 | 0.717 | None | — | cooldown |
| 2 | 2 | 0.852 | 0.781 | — | cooldown |
| 2 | 3 | 0.508 | 0.805 | **Y** | both <0.85; growth real |
| 3 | 1 | 0.961 | None | — | gist high; none needed |
| 3 | 2 | 0.953 | 0.875 | — | |
| 3 | 3 | 0.750 | 0.883 | — | |
| 4 | 1 | 0.798 | None | **Y** | c1 fallback; MSE elevated |
| 4 | 2 | 0.844 | 0.852 | — | cooldown |
| 4 | 3 | 0.984 | 1.000 | — | cooldown |
| 5 | 1 | 0.781 | None | — | cooldown |
| 5 | 2 | 0.797 | **0.906** | **Y (old)** | ← false positive eliminated by ws |
| 5 | 3 | 0.938 | 0.867 | — | |

**S5C2:** cross-session gist=0.797 (<0.85) would have triggered growth on all 5 regions; within-session gist=0.906 correctly reads quality as good — with ws-primary this event is suppressed. **S2C3:** gist=0.508, ws=0.805, both agree growth needed. **S4C1:** c1 fallback fires (ws=None), grew — per-region MSE was legitimately elevated (speech 0.435, visual 0.385), cooldown absorbs the residual c1 risk.

**Change (e30916c):** `sleep.rs`'s `quality_poor` matches `(within_session_gist_rate, gist_recall_rate)`; `prune_ok` uses `.or()` (None→no pruning at c1); `main.rs`'s pressure dissipation uses `ws.or(gist).unwrap_or(0.0)`. Cooldown retained; plasticity model deferred pending a c1 signal fix.

---

## Amendment — S5 startup priors + S6 adaptive wake ceiling (2026-06-14, 88b4080)

**S5:** `--max-warm-records 0` now auto-queries: `sysinfo::System::available_memory()` → `(free_bytes×0.35/2500.0) as usize`, clamped `[1_000, 1_000_000]`, logged as `max_warm_records`/`max_warm_auto`. `--recall-ef` default 64→16 (ADR 0022 campaign confirmed 16 at scale; synthetic benchmark deferred as redundant). `sysinfo="0.38"` added as a direct dep (already transitive, zero lockfile churn). No `--init-` rename — `0`="auto" is a clean semantic change, no backcompat shim per CLAUDE.md. **Stateless** — re-evaluated every process start, no persistence. **Amendment (2026-08-16, ADR 0047):** `--max-warm-records`'s own default was changed 0→200000, so this auto-derive formula is no longer exercised unless `0` is passed explicitly — every real campaign had hardcoded 200,000 since three days after this landed (`f891e21`, 2026-06-17) and never actually used the auto path in practice; the completed 10x campaign ended at 151,725 warm records, comfortably under that hardcoded cap.

**S6:** per-cycle wake-duration EWMA (`wake_ewma_mu`, `wake_ewma_n`), persisted `snapshot/wake_ewma.bin` (12-byte pattern, like `settle_ewma.bin`). α=0.3/cycle (not per-episode); cold-start assigns (not blends) from the first cycle's measured duration. `max_wake = max(floor, 3×wake_ewma_mu)` after N≥3, floor=`--max-wake-secs`. NDJSON: `wake_ewma_mean`, `max_wake_secs_effective`. **Stateful** across sessions via snapshot. **Known inertness:** fast runs (S3 sleeps at ~120 eps, 10-30s wake) keep `3×μ≈30-90s<600s floor` — ceiling only moves in long-clip runs approaching the floor (~602s schedule-triggered sleep observed).

---

## Amendment — P1a cold cap, P1b sleep cap, S7 wake-interval throttle (2026-06-14, 7964eb3)

**P1a (`--max-cold-records 0`→auto):** `COLD_RECORD_BYTES=661` (8+8+4+640+1). At startup, `sysinfo::Disks` finds the longest-matching mount for `memory_dir`, reads `available_space()`, computes `(free_bytes×0.20/661.0) as usize`, clamped `[10_000, 10_000_000]`. Disk-lookup failure (btrfs subvolumes, network FS) → warn and fall back to `0` (unlimited) rather than over-capping. NDJSON: `max_cold_records_effective`.

**P1b (`--max-sleep-secs 0`→rolling):** rolling 5-cycle window of sleep duration (`nrem_wall_ms+rem_wall_ms+spill_wall_ms+snapshot_wall_ms`), computed pre-sleep-call. Empty window (c1) → `effective=0` (no cap); else `max(300, 3×mean_sleep_secs)`. Manual `--max-sleep-secs>0` bypasses entirely. NDJSON: `sleep_max_secs_effective`. **Validated (2S×3C, fast clips):** c1 correctly gets 0 (no history); c2/c3 get 300 (short sleeps → 3×mean hits the floor).

**S7 (wake-interval throttle):** signal `mean_output_mse_wake`, EWMA α=0.3. Stagnation = <1% relative improvement for 3 consecutive cycles → `effective_wake_interval` decrements toward 1 (reconstruct more often). Improvement resuming resets the stagnation counter; interval stays at its current value (one-way pressure downward only, no recovery toward the user default).

```rust
s7_mse_ewma = S7_MSE_ALPHA * mean_mse_wake + (1.0 - S7_MSE_ALPHA) * s7_mse_ewma;
if s7_ewma_n >= 2 {
    let improvement = (prev_ewma - s7_mse_ewma) / prev_ewma;
    if improvement < 0.01 { s7_stagnant += 1 } else { s7_stagnant = 0 }
    if s7_stagnant >= 3 && effective_wake_interval > 1 { effective_wake_interval -= 1 }
}
```

NDJSON: `wake_interval_effective`, `s7_mse_ewma`.
