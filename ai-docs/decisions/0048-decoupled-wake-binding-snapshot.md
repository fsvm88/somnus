# 0048. Decoupled wake-time binding snapshot (§33 candidate (e), scoped after confirming the real root cause)

- **Status:** implemented 2026-09-13 (`wake_binding` field, `sync_wake_binding`, growth/prune
  lockstep, all wake-time call sites, ORT export, snapshot save/load, `--binding-wake-lag-cycles`
  CLI flag, 2 new unit tests, `cargo fmt`/`clippy -D warnings`/full test suite all clean).
  **Independently re-reviewed 2026-09-13** via a second model (local Lemonade/gpt-oss-20b through
  `opencode`, dispatched per this project's director-mode convention), against the full commit
  diff split into 6 self-contained chunks (metrics registry, the burn-clone-quirk regression test,
  the CLI flag/sleep-loop gating, the two new unit tests, and the `somnus-perception/src/lib.rs`
  changes split in two) — 26/26 checks came back clean, no issues found, independently confirming
  the self-review. **Piloted 2026-10-17** — see "Pilot results" at the end (mixed: fixes the settle-gate cascade, does not beat the headline `mean_output_mse` criterion).
- **Date:** 2026-09-13
- **Correction 2026-10-16 (found during the burn 0.22 migration, ADR 0049):** the "plain value copy
  of the eight trained tensors" left `wake_binding`'s GRU gates and `temporal_proj` independently
  random-initialised (the doc comment assumed neither copy trains them, but also that they started
  equal; they did not). A restored session therefore ran its wake path on fresh random temporal
  weights, differing from the saved `binding` and from run to run. Fixed by
  `BindingArea::copy_temporal_from` (construction + every `sync_wake_binding`). Any ADR 0048 pilot
  must run on a build with this fix. **Second correction (same day):** every restored run's first
  wake phase used an ORT binding session exported from random weights (built before the snapshot
  restore; fixed in ADR 0049), which also affects all earlier §33 pilots that restored a snapshot.
  And the lag counter restarts at 0 on every process start, while a 0.22 snapshot now persists the
  lagged `wake_binding` itself.

## Implementation findings (2026-09-13, during the build — corrections to the plan above)

Two real issues surfaced only by writing and testing the code, not by review:

1. **`binding.clone()` does NOT seed `wake_binding` correctly.** Empirically confirmed (regression
   test `binding::burn_linear_clone_quirk::linear_clone_does_not_preserve_materialized_values`):
   burn's derived `Module`/`Linear` `Clone` re-runs each `Param`'s lazy random initializer
   independently per clone — it does not copy the already-materialized weight value the way a bare
   `Tensor::clone()` does. The originally-planned `let wake_binding = binding.clone();` therefore
   gave `wake_binding` **different random initial weights** than `binding`, not an exact copy.
   Fixed in `PerceptionModel::new()` by explicitly copying each of the 8 trained tensors' `.val()`
   into a fresh `Param::initialized(...)` — the same pattern `sync_wake_binding` uses at runtime,
   just inlined at construction time.
2. **The sync gate must not be conditioned on `--binding-replay`.** `homeostatic_decay`
   (`crates/somnus/src/sleep.rs:896`) mutates `binding`'s trained tensors (region_0..4/w_q/w_k/w_v)
   every sleep cycle **unconditionally**, regardless of the `--binding-replay` flag. Gating
   `sync_wake_binding()` on `args.binding_replay` (as first implemented) would leave `wake_binding`
   frozen at its initial values forever whenever `--binding-replay` is off — the default/production
   configuration. Fixed: the cadence gate (`sleep_cycle_count.is_multiple_of(lag_cycles)`) applies
   unconditionally; `--binding-wake-lag-cycles 1` (the default) reproduces the pre-ADR-0048
   always-immediate behavior in both configurations.

Also found and fixed one more call site the original scoping missed: `binding_only` (`:3146` at
scoping time) reads `gru_hidden` right after its `forward()` call to return it to the caller — this
must read `wake_binding.gru_hidden` (the copy `forward()` was actually called on), not
`binding.gru_hidden` (which the GRU never touches). Caught by re-grepping every `.gru_hidden`
reference after the call-site swap, not anticipated during scoping.

## Context

`gaps.md` §33 / ADR 0041 confirmed that BindingArea's ~1.72M params never received a gradient
before ADR 0041, and implemented decision (b) — a live-episode NREM pass, `binding_replay_step_batch`
— to fix it. Six pilots since (`engineering-log.md` §35/§36/§44/§45/§46/§47) all show the same
`mean_output_mse` regression (+445% to +587%) regardless of update dosage, loss scale, LR scale, or
whether the output heads are excluded from the optimizer step (candidate (d) — froze the heads,
regression persisted with **no gradient path** for it to occur, disproving a head/gradient cause).

§47's confirming diagnostic (`--force-cpu-inference`, structurally removing ORT) proved the
mechanism is backend-independent: **the single problem is that `binding_replay_step_batch` trains
the one live copy of `BindingArea` (`PerceptionModel.binding`), and the very next wake phase's
`try_bind`/`settle_episode`/`binding_only` read from that same copy (directly on burn, or via an
ORT export that's just a copy of the same values) to form episodes that get stored.** Every sleep
cycle, the binder that forms and stores episodes is a newer generation than the one the stored
episodes already in warm/cold were formed by; `fused_replay_wake_step_batch`'s head training then
straddles both generations at once. This was ADR 0041's own predicted risk ("Episode-space drift
across binder generations"), now confirmed as the primary driver of every pilot's headline number,
not a secondary effect.

Grepped confirmed: there is exactly one `BindingArea<TrainB>` instance
(`PerceptionModel::binding`, `crates/somnus-perception/src/lib.rs:220`). `binding_replay_step_batch`
(same file, `:1184`) trains it via `forward_tracked`. `try_bind` (`:887`), `settle_episode`
(`:3117`), and `binding_only` (`:3038`) all read `self.model.binding` (or its ORT-exported mirror)
at wake time. `grow_region`/`grow_region_seeded` mutate the same tensors in place. There is no
existing seam between "the copy that trains" and "the copy that infers" — this ADR adds one.

## Decision

**Add a second, lagged copy of the currently-trained binding weights — `wake_binding` — that wake-time
inference and ORT export read from, hard-copied from the live training copy only every
`--binding-wake-lag-cycles` sleep cycles (default 1, i.e. today's always-immediate behavior, so
existing runs are unaffected unless the flag is raised).**

Scope: only the tensors `binding_replay_step_batch` actually trains — `region_0..region_4`
(per-region projections) and `w_q`/`w_k`/`w_v`. The GRU and `temporal_proj` are deliberately
excluded from live training by ADR 0041's own `None`-context pass and stay untouched here too; if
candidate (c) (temporal prediction loss) is later adopted and starts training the GRU, this ADR's
scope extends to cover it at that point, not before.

**Mechanism:**

1. `PerceptionModel` gains `wake_binding: BindingArea<TrainB>` (or a plain inference-only variant —
   see Alternatives), initialized as a clone of `binding` at model construction/load.
2. `binding_replay_step_batch` is unchanged — it keeps training `model.binding` every NREM cycle,
   at whatever dosage/step-count settings are already in place.
3. A new `sync_wake_binding()` copies `region_0..region_4`/`w_q`/`w_k`/`w_v`'s values from
   `binding` into `wake_binding` (plain tensor copy, no gradient tracking needed on the target).
   The driver loop's existing `sleep_cycle_count` counter (`crates/somnus/src/lib.rs:1347`,
   incremented at `:2150`) is the natural gate: call `sync_wake_binding()` right alongside the
   existing `rebuild_ort_session` call (`:2451-2469`) when `sleep_cycle_count % lag_cycles == 0`.
4. `try_bind`, `settle_episode`, `binding_only`, and `export_weights_for_onnx`/`rebuild_ort_session`
   all switch from reading `model.binding` to reading `model.wake_binding`. This is the critical
   correctness point: burn's wake path and the ORT-exported graph must read the *same* generation,
   or we reintroduce a burn-vs-ORT split identical in spirit to the one §47 just ruled out.
5. `grow_region`/`grow_region_seeded` apply their shape change (append rows/columns) to **both**
   `binding` and `wake_binding` immediately, unconditionally — growth is a structural change, not a
   value update, and the two copies must never disagree on shape. Only weight *values* lag; shapes
   never do.
6. New CLI flag `--binding-wake-lag-cycles <u32>` (default 1), meaningful only when
   `--binding-replay` is set. `lag=1` reproduces today's behavior exactly (sync every cycle) —
   this is the control-arm equivalent for the upcoming pilot's sweep.

## Implementation notes (found during review, 2026-09-13 — before any code was written)

A pre-implementation review of the actual call sites turned up gaps in the mechanism as first
scoped above. These are corrections, not changes of direction:

1. **`prune_region_inner` was missing from the growth-lockstep list.** There are three structural
   mutators of `model.binding`'s per-region tensors, not two: `grow_region_inner` (:2095),
   `grow_region_seeded_inner` (:2187), and `prune_region_inner` (:2338) — all three splice rows into
   or out of `region_0..4`. All three must apply the identical splice to `wake_binding` in the same
   call, unconditionally (never gated by the lag counter), or the two copies' shapes diverge and the
   next `wake_binding.forward()` panics on a dimension mismatch against the (already-resized)
   encoder output it's fed.
2. **`export_weights_for_onnx_inner` (:2774) must read every binding field — including
   `temporal_gru`/`temporal_proj`, currently untrained by any path — from `wake_binding`, not
   `model.binding`.** ORT's exported graph must be internally self-consistent; simplest correct rule
   is "the ORT graph always mirrors `wake_binding` in full," not "the trained tensors come from one
   copy and the untrained ones from another."
3. **`binding_proj_row_norms` (:1835) stays pointed at `model.binding`, unchanged.** It's ADR 0041's
   own pilot-verification probe into training progress (do grown rows move under
   `binding_replay_step_batch`), not a wake-time read — this ADR must not touch it.
4. **`homeostatic_decay` (:2058) needs no change.** It mutates `model.binding`'s trained tensors
   (region_0..4/w_q/w_k/w_v) in place, same as `binding_replay_step_batch` does — `sync_wake_binding`
   copying whatever `binding` currently holds already carries decay's effect along for free, exactly
   the batching behavior this ADR wants. No special-casing needed.
5. **`wake_binding` must be seeded as `binding.clone()` at construction, not an independent
   `BindingArea::new(device)` call.** `PerceptionModel::new()` (:233) currently does `binding:
   BindingArea::new(device)`; a second independent `BindingArea::new(device)` call for
   `wake_binding` would draw different random initial weights (two separate RNG draws), causing a
   spurious day-0 divergence between the two copies before any training has happened. Construct
   `binding` first, then set `wake_binding` to a clone of it (burn `Module` types implement `Clone`;
   `PerceptionModel` itself is already cloned elsewhere, e.g. `save_weights` at :2691).
6. **Snapshot save/load:** `wake_binding` is a normal (non-`#[module(skip)]`) field, so burn's
   `BinFileRecorder` serializes/deserializes it automatically alongside every other field — no
   special handling needed for the weight *values*. But `region_n` is `#[module(skip)]` on
   `BindingArea` and is **not** persisted; `load_weights` (:2702) already has a manual
   reconstruction block for `loaded.binding.region_n` from the region weight shapes (:2716-2722) —
   the identical block is needed for `loaded.wake_binding.region_n`. The load-failure fallback path
   (constructs a fresh `PerceptionModel::new()` and re-grows it, :2731 onward) also needs
   `wake_binding` grown in lockstep there, same as point 1 above.
7. **`wake_binding` becomes the sole stateful copy.** Wake-time `forward()` (not `forward_tracked`)
   caches `gru_hidden` across calls for the GRU temporal-context branch. After `try_bind`/
   `settle_episode`/`binding_only` switch to `wake_binding.forward(...)`, `wake_binding` owns that
   cache exclusively; `model.binding`'s own `gru_hidden`/`temporal_gru`/`temporal_proj` become
   permanently inert (never forward()'d, never trained, by any path, before or after this ADR —
   that's unchanged; candidate (c) is what would eventually train the GRU). Harmless, but worth
   naming so it isn't mistaken for a bug in review.

## Alternatives considered

- **EMA-smoothed wake copy** (`wake_binding = (1-α)·wake_binding + α·binding` every cycle) instead
  of a periodic hard copy: rejected as the first thing to try. It still lets *some* drift through
  every single cycle rather than batching it into discrete, measurable promotion events, which
  makes it harder to isolate "does spacing out promotions help" as a clean yes/no from the pilot.
  Worth a follow-up ADR if the hard-copy lag helps but produces visible step-discontinuities in
  `mean_output_mse` at sync boundaries.
- **Throttle training frequency instead of adding a second copy** (call `binding_replay_step_batch`
  only every N cycles, no `wake_binding` at all): simpler — no new state, no new call-site
  plumbing — but conflates two different things: how often the binder *trains* vs. how often wake
  *sees* a new generation. A future need (e.g. more frequent training for faster convergence, but
  still-infrequent promotion for episode-storage stability) can't be expressed under this
  alternative. Rejected in favor of the explicit two-copy design after discussing tradeoffs with
  the user.
- **Validated/gated promotion** (only copy `binding` → `wake_binding` if a held-out proxy metric
  doesn't regress): the principled long-term answer, but needs a cheap held-out evaluation signal
  that doesn't yet exist, and adds real complexity (what data, what threshold, what happens on a
  failed promotion — skip forever? retry next cycle?). Deferred; revisit only if the periodic
  hard-copy lag in this ADR is not sufficient on its own.
- **Reduce warm-store eviction rate instead of controlling promotion** (let more old-generation
  episodes linger so heads see a more stable mixture): rejected — this is exactly the "warm-store
  turnover" mitigation ADR 0041 already named as tolerable-but-passive; it does not address the
  cause, only dilutes the symptom, and ties the fix to unrelated memory-capacity tuning.

## Consequences

- **Cost:** one extra `BindingArea` weight copy (~1.72M params × 4 bytes ≈ 6.9MB) plus one
  `Tensor::clone`/copy operation on `sync_wake_binding()` calls — negligible relative to a sleep
  cycle's existing cost.
- **New drift-controlling knob, not a drift-eliminating one:** stored episodes will still show
  generational drift when a sync happens — this ADR controls *how often* that generational jump
  occurs, not whether it occurs at all. If `mean_output_mse` still regresses at every lag setting
  tried, that's a real result (points toward validated/gated promotion or reconsidering decision
  (b) itself), not a failed pilot to retry with different numbers.
- **Interacts with growth:** growth events force both copies to reshape in lockstep regardless of
  lag setting, so a lag setting doesn't delay a region-growth-triggered `wake_binding` shape change
  — only value updates lag.
- **Interacts with ORT:** `rebuild_ort_session`'s existing "fires after every sleep cycle" behavior
  (§46) is unchanged, but it now exports `wake_binding`'s (lagged) values instead of `binding`'s
  (live) values — so ORT re-export cadence and binding-promotion cadence become independently
  controllable, closing the conflation §46 originally (and incorrectly) attributed causal weight to.

## Metrics

Per the standing metrics policy (`CLAUDE.md`):

- **New Core counter `BindingWakeSyncCount`**: fires once per `sync_wake_binding()` call — lets a
  production run confirm the lag cadence is actually taking effect, the same discipline ADR 0041
  used for `binding_row_norms`/unit tests.
- **New Core gauge `BindingWakeDivergenceL2`**: L2 distance between `binding` and `wake_binding`
  (summed across the 8 trained tensors) measured immediately *before* each sync — quantifies how
  much the live copy drifted during the lag window just closed. Directly informs whether a given
  `--binding-wake-lag-cycles` value is "enough" drift-batching or still too frequent, without
  needing a full pilot re-run to eyeball it.
- **Existing coverage named:** `mean_output_mse`, `gist_recall_rate`, `warm_recall_l2`,
  `settle_gate_dynamic`/`sal_threshold`/`settle_iters_sum` (the §36/§47 cascade signature) are
  already Core/Fine-tier covered and are exactly what the validation plan below reads.

## Validation plan (before adoption)

**Longer pilot than every prior §33 pilot.** All six pilots so far ran 10 cycles; ADR 0041 itself
argues warm-store turnover (~30 cycles at the 200k cap) is the real drift-absorption mechanism, so
a 10-cycle window may be too short to observe whether `mean_output_mse` reaches a new steady state
after a binder-generation change, versus permanently regressing. Run **30-50 cycles** this time,
and report the metric as a **trend** (e.g. mean over cycles 1-10 vs. 11-25 vs. 26-50), not just one
aggregate mean — the previous pilot scripts' single-number aggregate cannot distinguish "transient
adaptation cost after each promotion" from "permanently broken."

**Sweep, one variable per arm per project discipline:**

| arm | `--binding-wake-lag-cycles` | purpose |
|---|---|---|
| control | binding-replay off | pre-existing baseline, as in every prior pilot |
| baseline-lag | 1 | today's always-immediate behavior, reproduced under the new code path — must match §35/§47's magnitude to prove this ADR's plumbing didn't silently change anything before testing the actual lever |
| lag-10 | 10 | roughly the step-count-cap order of magnitude tried (unsuccessfully) in §36, now applied to promotion frequency instead of training frequency |
| lag-30 | 30 | matches ADR 0041's own warm-turnover-window citation |

**Adopt if:** `mean_output_mse`'s cycles-26-50 mean at lag-10 or lag-30 is materially below
baseline-lag's cycles-26-50 mean (i.e., the system reaches a *better* steady state, not just a
slower path to the same one), without `gist_recall_rate`/`warm_recall_l2` regressing further and
without reproducing §47's settle-gate collapse at a similar or worse magnitude.

**Reject / iterate if:** all three lag settings converge to the same steady-state band — that would
mean promotion *frequency* isn't the lever either, and the next candidate is validated/gated
promotion (this ADR's "Alternatives considered" deferred option) or revisiting decision (b) itself.

## What would invalidate this

- If `BindingWakeDivergenceL2` shows negligible drift even at `lag-30` (i.e., `binding_replay`'s own
  updates are already small enough that lag doesn't matter) — would mean the lag lever has nothing
  to work with, and the real fix has to act on update magnitude or promotion validation instead.
- If growth events during the pilot force enough lockstep reshapes that the "lag" is effectively
  never longer than the growth cadence — would mean this design needs a smarter growth interaction
  (e.g. resetting the lag counter, or growth-triggered syncs counting toward `BindingWakeSyncCount`
  separately) before a real read is possible.

## Pilot results (2026-10-17, `scripts/ab_binding_wake_lag.sh`)

4 arms x 40 cycles, one binary (HEAD `56925da`, fine-metrics), same restored snapshot, `SEED=42`, two arms
concurrently. Control = no `--binding-replay`; lag1/lag10/lag30 = `--binding-replay --binding-wake-lag-cycles N`.
Artifacts: `memory/ab_binding_wake_lag/` (gitignored; `analysis.txt` has every table). **Single seed, no noise
baseline for this pilot**; the R1 pilot's control-vs-control spread on `mean_output_mse` was ~14%.
Means by cycle bin (1-10 / 11-25 / 26-40):

| metric | control | lag1 | lag10 | lag30 |
|---|---|---|---|---|
| `mean_output_mse` | .0168 / .0160 / .0234 | .0560 / .0408 / .0225 | .0204 / .0386 / .0389 | .0169 / .0226 / .0215 |
| `mean_output_mse_wake` | .046 / .046 / .045 | .051 / .069 / .107 | .042 / .031 / .028 | .044 / .047 / .036 |
| `settle_gate_dynamic` | .80 / .82 / .80 | .30 / .28 / .26 | .81 / .74 / .77 | .79 / .81 / .63 |
| `settle_iters_sum` | 12 / 13 / 15 | 37 / 41 / 56 | 12 / 19 / 16 | 14 / 14 / 15 |
| `gist_recall_rate` | .60 / .71 / .70 | .44 / .44 / .53 | .68 / .39 / .70 | .58 / .73 / .61 |
| `warm_recall_l2` | .51 / .46 / .45 | .71 / .71 / .72 | .48 / .63 / .50 | .47 / .46 / .58 |

- **Plumbing check passed:** lag1 reproduces the known failure (the §47 settle-gate collapse: gate .26-.30 vs .80,
  settle iterations 3-4x, wake MSE rising 0.05 -> 0.11, `warm_recall_l2` .71). `binding_wake_sync` fired 40/40/4/2
  times for control/lag1/lag10/lag30, as designed.
- **The lag lever works on the cascade:** lag10 and lag30 have no settle-gate collapse, normal settle iterations,
  and wake MSE at or below control; `gist_recall_rate` and `warm_recall_l2` recover to control-like values (lag10 had a
  bad middle bin, 11-25, on both).
- **But the ADR's stated adopt criterion is NOT met:** cycles 26-40 `mean_output_mse` vs lag1: lag30 -4.3% (within
  noise, not "materially below"), lag10 +73% (worse). Control reaches the same band as lag30 (.0234 vs .0215), i.e.
  lag-N restores baseline behaviour; it does not beat it. lag1's `mean_output_mse` is far worse early (.056) but
  converges to .0225 by 26-40 while its wake metrics keep degrading, so this metric alone hides the damage.
- **Not shown by this pilot:** any benefit of `binding_replay` itself (lag30 ~ control). The pilot measured removal of
  harm, not gain. Growth cadence differed per arm (8-10 growth events, final sizes 10139-10295), a confounder; lag10's
  late-bin `mean_output_mse` excess is unexplained and could be noise or a real per-lag effect (needs a second seed).
- `BindingWakeDivergenceL2` was recorded (fine-metrics counters, summed) but I did not analyze how much drift each lag
  window accumulated; that remains the open check on the "what would invalidate this" first bullet.

**Status:** not adopted as a default. Next questions: a second seed (or a control-vs-control baseline) for lag10 vs
lag30; whether `binding_replay` yields any benefit worth the lag (needs a metric that rewards it); then the
validated/gated promotion alternative if not.

### Second seed (SEED=43, 2026-10-17, same script/binary + duty-cycle metrics; `memory/ab_binding_wake_lag_s43/`)

Replication of the seed-42 pilot. What replicated and what did not:

- **lag1 reproduces the cascade** (settle gate .69 -> .52 -> .19 by bins; `warm_recall_l2` .59 -> .70; region-2 MSE 0.0095 vs
  0.0050 control in cycles 26-40). Its `mean_output_mse` is huge early (.120 in cycles 1-10) and converges to control's
  band by 26-40 (.0199 vs .0196), same as seed 42. So that metric converges while the health metrics keep degrading.
- **lag10 is worse than control on `mean_output_mse` in both seeds**, cycles 26-40: .0389 vs .0234 (seed 42), .0353 vs
  .0196 (seed 43), i.e. +66% / +80%. This replicated, so it is probably real, not noise; cause unknown. Its wake MSE is the
  best of all arms in both seeds (.028 / .025) and it never collapses the settle gate.
- **lag30 does NOT hold up: the lag only defers the shock.** Per-cycle settle gate for lag30 sits at ~0.80 until the
  cycle-30 promotion, then collapses: seed 42 .80 -> .51 -> .24 -> .47 -> .68; seed 43 .79 -> **0.0, 0.03, 0.11** -> .48
  -> .08. A promotion after 30 cycles of accumulated drift reproduces the §47 collapse at the moment it happens. lag10
  (promotions every 10 cycles) shows only mild dips (.61-.80) in both seeds. The seed-42 table's "lag30 is fine" reading
  was an artifact of the bins: it measured mostly the pre-promotion window.
- **Adopt criterion: still not met** (lag30 vs lag1, cycles 26-40: -4.3% seed 42, +26.0% seed 43; lag10 +73% / +77%).
- **Revised reading:** what matters is the size of the generation jump at promotion, not (only) its frequency; small, frequent
  promotions (lag10) are safe for the settle gate but cost `mean_output_mse`; one big late promotion (lag30) is the
  worst of both. This points at the ADR's deferred alternative (validated/gated promotion, or interpolating `wake_binding`
  toward `binding` instead of hard-copying) rather than at a larger lag. Not decided; not implemented.
- **Control (no binding replay) is as good as or better than every binding_replay arm** on the sleep/health metrics in
  both seeds. The pilots still show no gain from binding_replay itself.

**New observation (duty-cycle metrics, all four arms, both within the run):** `kwta_duty_max` reaches **~1.0 by cycle 10
and stays there** in every arm, including control (cycle 1: ~0.5). At least one k-WTA unit is active on essentially every
ingest, so boosting (`BOOST_STRENGTH` 1.0 x (target - dc), target ~0.0195) cannot suppress it, and one of the ~193 active
slots per region is permanently taken. `kwta_duty_dev_mean` is stable at ~0.016-0.024 (target dc 0.0195), i.e. the average
unit is as far from target as the target itself. This is independent of binding_replay and of the lag. Which region/unit,
and whether it predates R1 (the seed-42 pilots lack the metric), is not yet known. Candidate causes to check: a grown
unit seeded with a large-norm weight (`grow_region_seeded`), or dead-by-design k-WTA dynamics with a weak boost.
