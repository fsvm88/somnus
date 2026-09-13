# 0041. Live-episode binding training in NREM (close the §33 never-trained-binding gap)

- **Status:** implemented; isolated A/B pilot run 2026-08-15 (engineering-log.md §35) — mechanism validated
  and the §33 fix engages in production state, but the predicted episode-drift recall regression
  materialized and the NREM cost displaced ~30% of replay. **Not adopted.** Adoption path queued
  (engineering-log.md §35 "Adoption path"): step budget (`--binding-replay-steps`) → longer repeat pilot
  (≥20 cycles, n≥2) → adopt if drift contained. **Correction recorded:** the named mitigation
  "rebuild warm from cold at switchover" does not work as written — cold records store only
  episode vectors, not regional inputs, so they can't be re-formed through a new binder; the
  real absorption mechanism is warm-store turnover (~30 cycles) plus a slower binder change rate.
  Flag stays opt-in (`--binding-replay`, default off) until adoption.
  **2026-08-15 (later): adoption-path step budget tried and FAILED (engineering-log.md §36).**
  `--binding-replay-steps N` (default 1) caps `binding_replay_step_batch` calls per sleep cycle;
  the repeat pilot at N=1 (20 cycles, n=1) cut NREM wall-clock (−24.8%, the cap works as a cost
  lever) but made quality *worse* than the unbounded §35 pilot: ~10× same-cycle `mean_output_mse`
  on cycles where the (now rarer, larger) step fires, plus a new failure mode — 6/20 treatment
  cycles replayed zero traces, a cascade through the salience/settle homeostatic controllers
  reacting to the degraded output. The §33 fix still engages under the cap (grown rows 30–78%
  nonzero, confirmed via `binding_row_norms`), so the mechanism isn't broken — but step-count
  capping decoupled cost from drift instead of taming both as hypothesized. **Step-count capping
  abandoned as the lever.**
  **2026-08-15 (later still): candidate (a), loss-scale reduction, tried — result INVALIDATED, corrected
  2026-08-16 (engineering-log.md §44).**
  New `--binding-replay-loss-scale <f32>` flag (default 1.0) scaled the live-episode loss tensor
  immediately before `.backward()`. Pilot at scale=0.1 (step cap left at its default of 1) produced
  `mean_output_mse` +440.6% vs. control. **This was not a valid dosage test:**
  `binding_replay_step_batch` steps an `AdamW` optimizer, which normalizes each parameter update by
  its running second-moment estimate, so scaling the *loss* tensor before backward cancels out
  almost entirely before it reaches the parameters. Read correctly, the +440.6% aggregate is
  statistically the same as the N=1 pilot's own aggregate (+445%, engineering-log.md §36) — evidence
  the scale had no real effect, not evidence that a weaker update makes things worse. The previously
  recorded "roughly 4x the quality damage" compared this pilot's aggregate against §36's
  same-cycle-on-fired-cycles figure (~10x), a different measurement — not a valid basis for that
  claim. **Loss-scale is retired as the mechanism.** `--binding-replay-loss-scale` is renamed
  `--binding-replay-lr-scale` and now scales the learning rate passed to `optimizer.step()`
  directly, which does change the update magnitude — the real dosage test candidate (a) was meant to
  be. Not yet piloted (`scripts/ab_binding_replay_lr_scale.sh`). Two further candidates remain
  untried: fuse into the existing `fused_replay_wake_step_batch` step rather than a separate
  optimizer pass; decouple the salience/settle cascade directly (note `sal_gate`/`settle_gate_val`,
  `crates/somnus/src/lib.rs:2905-2922`, are homeostatic controllers correctly reacting to genuinely
  degraded output — as literally described this candidate is symptom suppression, not a fix, and
  any pilot needs `mean_output_mse`/`per_region_mse` as gating criteria, not just cascade metrics).
  **2026-08-16: candidate (a) re-run with the real LR-scale dosage lever — FAILED, same magnitude
  as full-strength (engineering-log.md §45).** `--binding-replay-lr-scale 0.1` (a genuine 10×
  weaker optimizer step this time) produced `mean_output_mse` +444.7% — statistically
  indistinguishable from §36's unscaled +445% and the invalidated §44 result. Failure shape
  differs (per-region movement now genuinely mixed, 2 worse/3 better; `gist_recall_rate`/
  `warm_recall_l2` barely moved instead of collapsing) but the headline number doesn't improve at
  10× lower update magnitude. **Candidate (a) is now closed in both variants** (loss-scale
  invalidated as a no-op; LR-scale tested for real and failed). This is now solid evidence — not
  the flawed inference §44 rested on — that the damage isn't a dosage effect: **(c) decouple the
  salience/settle cascade is the favored remaining candidate**, with (b) fuse-into-existing-step
  still untried and lower priority (doesn't address the cascade).
  **2026-08-16: new candidate (d) found while scoping (c) — shared-head interference, implemented
  and unit-tested, pilot pending.** Reading the code (not just this ADR's own prose) against
  `fused_replay_wake_step_batch` found two things: this ADR's "Gradients reach... the encoders
  (through `z`)" claim was wrong (corrected in "Mechanism" below — `SparseActivation.values` are
  materialized before the binding forward, severing the graph; encoders don't train here); and the
  production `mean_output_mse` metric is the head-reconstruction MSE on **stored** episodes
  (`sleep.rs`'s `fused_replay_wake_step_batch` return value), while this pass trains the *same*
  five head weights on **live** episodes in the same NREM batch through the same optimizer — a
  plausible dosage-independent cause of the regression, which would explain why both variants of
  candidate (a) failed identically regardless of update magnitude. New
  `--binding-replay-freeze-heads` flag excludes the heads' own weights from this step's
  `optimizer.step()` (gradient still flows through them into binding). Verified by a dedicated
  unit test (`binding_replay_freeze_heads_trains_binding_not_heads`). **(d) is now the favored
  candidate for the next pilot** — it targets `mean_output_mse` directly and is code-grounded,
  vs. (c) which targets a wake-side symptom (the cascade) that this hypothesis says isn't even the
  cause. Pilot script: `scripts/ab_binding_replay_freeze_heads.sh`.
  **2026-08-16 (later still): candidate (d), freeze heads, tried and FAILED — but reveals the
  likely real root cause (engineering-log.md §46).** `mean_output_mse` +586.9%, same 445-587% band
  as every prior pilot. This is stronger than a failed pilot: with heads frozen and encoders never
  in binding-replay's loss graph, there is **no gradient path** for the treatment arm to move
  `mean_output_mse` — yet it moved. Found by reading `crates/somnus/src/lib.rs:2451-2469`:
  `rebuild_ort_session` fires after **every** sleep cycle once ORT is active, not only on growth as
  this ADR's Consequences section implied. At 10x-growth scale ORT was active from cycle 1 in both
  arms. Real channel: binding weights change → re-export picks them up → next wake phase forms
  episodes through the new binder → those episodes get **stored** as new traces → later cycles'
  `fused_replay_wake_step_batch` trains heads against these drifted stored episodes, which is
  exactly what `mean_output_mse` measures. Explains dosage invariance (both variants of candidate
  a), why freezing heads didn't help (d), and reproduces §35's original episode-space-drift finding
  directly (`gist_recall_rate` −41.9%, `warm_recall_l2` +38.4% this pilot). **Not yet confirmed by
  an isolating pilot** (disable/delay ORT rebuild during a binding-replay run, or force the
  CPU/burn wake path) — leading hypothesis, not proven. **Blocks (b) and (c) as previously scoped**
  — neither touches the re-export cadence this hypothesis implicates. No path chosen; next step is
  the confirming diagnostic, not another optimizer/loss variant.
  **2026-09-13 — confirming diagnostic run, and the ORT-specific framing above is WRONG
  (engineering-log.md §47).** Repeated the candidate-(d) pilot with `--force-cpu-inference` on both
  arms — `rebuild_ort_session` never runs under this flag (`onnx_dir` is never resolved), so the
  re-export channel is structurally absent, confirmed by `ort_binding_forward` firing 0 times in
  both arms. `mean_output_mse` was still +462.0% — the same 445-587% band as every ORT-active pilot.
  **ORT re-export is not the cause; the mechanism is backend-independent.** Whatever serves wake-time
  binding inference (ORT or burn) uses live weights that changed during the immediately preceding
  NREM cycle; those episodes get stored; later cycles train heads against the now-drifted
  stored-episode distribution — burn delivers the identical drift with no export step at all. This
  is §35's original "episode-space drift" finding, now positively confirmed rather than inferred.
  **(b) and (c) are no longer blocked by an ORT-specific reasoning** (there is no ORT-specific
  cadence to touch), but neither addresses the now-confirmed real constraint either — both still
  update binding weights mid-campaign and let the very next wake phase consume the change
  unmediated. A candidate that would: decouple wake-time binding inference from the NREM-training
  copy of the weights (e.g. an EMA/lagged snapshot for episode formation). Not scoped or piloted.
  Needs a decision from the user.
- **Date:** 2026-08-15

## Context

`gaps.md` §33 (2026-08-14) confirmed a long-standing gap first anticipated by ADR 0008 itself:
**no training path in this codebase has ever sent a gradient into the binding area.** All training
(`replay_step*`, `wake_reconstruct_step*`, `fused_replay_wake_step_batch`, ADR 0012 proj_loss)
consumes episodes as stored data (`Tensor::from_data`), so the graph stops at the episode boundary.
ADR 0008's "the binding weights will primarily be shaped by the episode vector being reused
downstream — will be revisited in Phase 6" never materialized: the downstream reuse never carried
gradients, and the Phase 6 revisit never happened.

Concrete consequences, both verified in §33:

1. The binding's ~1.72M parameters (5 region projections, W_q/W_k/W_v, GRU, temporal_proj) run on
   their random init, permanently.
2. `grow_region`/`grow_region_seeded` append **zero-initialized rows** to the binding projection
   (`crates/somnus-perception/src/lib.rs:1851-1868`, comment "gradients will shape them over time").
   Since no gradients ever flow, **grown units' binding rows stay exactly zero forever — grown
   units are invisible to episode formation.** The 10x growth campaign completed 2026-08-15
   (engineering-log.md §34) at `[9821]×5`, so 8,797 of 9,821 units per region (89.6%) — the campaign's entire
   product — contribute to SAE reconstruction but nothing to episodes, memory, or recall.

The campaign's completion removes the sequencing blocker §33 named (no mid-campaign dynamics
change to protect). The user chose option (b) — live-episode head training — with option (c)
(temporal prediction loss) as an explicitly sequenced follow-up, not a combined change (one new
loss term per pilot for A/B attribution; see "Alternatives considered").

## Decision

Add an opt-in NREM training pass in which episodes are **recomputed live through the binding
area** from the stored regional inputs and the existing output heads are trained against the
regional targets on those live episodes — so gradients flow into the binding weights.

Mechanism, per NREM batch (traces already carry `regional: [Vec<f32>; 5]`):

1. Per-region encoder `forward_tracked` → k-WTA mask → `sparse_z` — exactly the tracked forwards
   `replay_step_batch` already runs for the SAE reconstruction loss (reused, not duplicated).
2. **New:** feed the five `SparseActivation`s through a new graph-preserving binding forward
   (`BindingArea::forward_tracked`, see contract changes below) → live episode tensor
   `[1, EPISODE_DIM]` per trace, batched to `[B, EPISODE_DIM]`.
3. **New:** run the existing five output heads (`speech_head`…`language_head`, the same heads
   ADR 0012's proj_loss trains) on the live episodes; MSE against the traces' stored regional
   inputs — the identical target `wake_reconstruct_step_batch` already uses, so the loss semantics
   are unchanged; only the episode's provenance changes (live-through-binding instead of stored).
4. Sum the five heads' losses and run one `backward()` + `optimizer.step()` — **a separate step
   from the SAE reconstruction term**, not accumulated into it (`binding_replay_step_batch` is a
   distinct method with its own optimizer call; see "Contract changes" below for why).

Gradients reach: the active rows of each region's binding projection (including grown units' rows
the first time they win k-WTA selection — this is the fix for consequence 2 above), W_q/W_k/W_v,
and the heads. **The encoders do NOT receive gradients from this pass, despite an earlier version
of this document claiming otherwise (corrected 2026-08-16):** phase 1's `SparseActivation.values`
are materialized to plain `Vec<f32>` (`z_batch.into_data().convert::<f32>().to_vec()`,
`crates/somnus-perception/src/lib.rs`) before entering the binding forward, which severs the autodiff
graph at that boundary — `BindingArea::project_regions` rebuilds a fresh leaf tensor from those
values (`binding.rs`'s `vals_t = Tensor::from_data(...)`). Encoder columns are trained only by the
SAE reconstruction term in the same NREM batch, exactly as the in-code doc comment on
`binding_replay_step_batch` already stated (the discrepancy was between this ADR and the code, not
within the code itself). **The GRU (`temporal_gru`/`temporal_proj`) is also NOT trained by this
pass** — see "Contract changes" for why, and option (c) under "Alternatives" for the sequenced
follow-up that covers it.

**2026-08-16 finding, relevant to the four failed pilots (engineering-log.md §35/§36/§44/§45):**
the five output heads are shared, unguarded state — this pass and `fused_replay_wake_step_batch`
both call `model.speech_head.forward(...)` (etc.) and both write through the same
`self.optimizer`, but on two different episode sources: this pass uses **live** episodes
(recomputed through the currently-training binder), the fused path uses **stored** episodes (the
production `mean_output_mse` metric is measured on the fused path's stored-episode task —
`sleep.rs`'s `fused_replay_wake_step_batch` return value). Training the same head weights toward
two different input distributions in the same NREM batch is a plausible direct cause of the
dosage-invariant `mean_output_mse` regression seen across all four pilots so far — worth testing
directly (freeze the heads during this pass; see gaps.md §33) before concluding (per the
"Alternatives" candidate (c)) that the fix has to be on the wake-side cascade instead.

**Gating:** new CLI flag `--binding-replay` (default off). Off = byte-identical to today's NREM
(control arm); on = the new pass runs (treatment arm). No other behavior changes.

### Contract changes

- **`BindingArea::forward_tracked(&mut self, acts: &[&SparseActivation; 5], device) -> Tensor<B, 2>`**
  (new): stateless binding forward returning the `[1, EPISODE_DIM]` episode *tensor* without
  materializing to `Vec<f32>`. Two reasons this can't just call the existing `forward`:
  1. `forward_inner` ends in `.into_data().to_vec()` (`binding.rs:213-219`) — materializing
     severs the autodiff graph, so no gradient could flow through an `EpisodeVector` no matter
     how it was produced.
  2. `forward` with `new_episode: Some(_)` mutates the cached `gru_hidden` (`binding.rs:178-185`),
     which the wake ingest path relies on; a training pass over unordered replay traces must not
     touch it. With `new_episode: None` the existing path is already stateless (the GRU branch is
     skipped entirely), so `forward_tracked` is implemented as the `None`-context computation
     returning the tensor pre-flatten. The existing `forward` is refactored to share this inner
     computation (its observable behavior unchanged).
- **`PerceptionProcessor::binding_replay_step_batch(&mut self, batch: &[&[Vec<f32>; 5]])`** (new):
  the step described above. Deliberately a separate method + separate optimizer step rather than
  fusing into `fused_replay_wake_step_batch`: keeps the existing fused path byte-identical for the
  control arm, and fusion is a cheap optimization to do only if the pilot wins (one fewer AdamW
  pass per batch, ~150ms at batch=128, per that function's own doc comment).
- **`sleep.rs` NREM loop:** when `--binding-replay` is set, call `binding_replay_step_batch` on
  the same trace batches the existing replay consumes.

## Alternatives considered

- **(a) Accept the fixed random binder, document the limitation** (§33 option a): free, but gives
  up exactly what ADR 0008 was designed to provide and leaves 89.6% of the grown core invisible to
  episodes/memory/recall. Rejected — the growth campaign's product is mostly unused downstream
  while this stands.
- **(c) Temporal prediction loss** (episode_t → predict episode_{t+1}, §33 option c): the
  GRU-specific signal. Chosen as the *follow-up*, not a combined change, for three reasons:
  (1) one new loss term per pilot is this project's A/B discipline — combined terms can't be
  attributed or individually rolled back; (2) (c) needs temporally-adjacent episode pairs, which
  replay traces don't currently carry (real plumbing: pair construction or next-episode links);
  (3) two new loss terms at once double the destabilization surface on a healthy trained state.
  Note (b) and (c) are complementary by construction, not redundant: (b)'s `None`-context pass
  doesn't put the GRU in the graph at all, so (c) is the only path that trains the temporal
  components.
- **(b)+(c) combined in one ADR/pilot**: the user's initial question; sequenced instead, per the
  three reasons above. If (b)'s pilot lands, (c) gets its own small ADR reusing (b)'s
  live-episode machinery.
- **Train the binding against stored episodes directly** (e.g. a loss pulling live episodes toward
  stored ones): rejected — it would teach the binder to imitate its own untrained output, freezing
  the very behavior being fixed.
- **Fusing the new term into `fused_replay_wake_step_batch` immediately**: rejected for the pilot
  phase (control-arm byte-identity + rollback isolation); revisit as an optimization if adopted.

## Consequences

- **Training dynamics change for any run with the flag on** — that's the point, and why adoption
  is gated on the isolated A/B pilot below; production `memory/` is untouched until the pilot
  wins.
- **Episode-space drift across binder generations:** all episodes currently in the warm/cold
  tiers were produced by the untrained binder; once binding trains, newly-formed episodes drift
  away from stored ones, and the warm HNSW store mixes both generations. Judged tolerable for the
  pilot: the warm store turns over (`--max-warm-records` eviction, currently 151,725/200,000),
   recall is L2-distance-based so gradual drift degrades gracefully, and the pilot measures the
   degradation directly (`warm_recall_l2`, `gist_recall_rate`). If drift shows up as recall
   regression in the pilot, mitigation becomes part of the adoption plan. **(2026-08-15 update:
   drift did show up — engineering-log.md §35 — and the mitigation named here does not work as written:
   cold records store only episode vectors, not regional inputs, so a rebuild from cold can't
   re-form episodes through a new binder. The real absorption mechanisms are warm-store turnover
   (~30 cycles at the 200k cap) and a step budget slowing the binder's change rate; see the
   §35 queued adoption path.)**
- **ORT inference picks up trained binding weights at the next re-export** (`rebuild_ort_session`
  on growth/restart — already part of the lifecycle; `export_weights_for_onnx` already exports all
  22 tensors including the binding's). Between training and re-export, wake inference runs the
  stale binder — same lag that already exists for every other weight class after growth.
  **(2026-09-13 correction, engineering-log.md §47): this bullet's original framing treated ORT
  re-export as the specific channel by which trained binding weights reach wake-time inference —
  it isn't specific to ORT. `rebuild_ort_session` in fact fires after *every* sleep cycle once ORT
  is active, not just on growth/restart (§46), and a confirming pilot with `--force-cpu-inference`
  (ORT structurally disabled) showed the identical `mean_output_mse` regression band (§47) — burn's
  wake path always evaluates the live weights directly, with no re-export lag at all, and still
  drifts identically. The real, backend-independent mechanism: whatever serves wake-time binding
  inference uses weights that changed during the immediately preceding NREM cycle, and the episodes
  formed then get stored and later corrupt head training.**
- **Cost:** one extra binding forward per trace per NREM batch (burn path, sparse-gather — the
  §25/§30 fast path, since training always runs on burn/CPU), plus the head forwards already
  shared with the proj_loss semantics. Expected ≪ cycle wall-clock; measured live by the new Core
  metric below, not assumed.

## Metrics

Per the standing metrics-discipline policy (CLAUDE.md) and the 2026-08-15 standing instruction
that A/B runs always use full debugging metrics and record deviations:

- **Existing coverage named:** the enclosing NREM phase is already Core-timed; `BindingForward`
  (Core) covers the wake-path binding call; `binding_forward` (Fine) exists; `wake_worker_train_step`
  and the fused-replay sub-phase metrics (Fine) cover the adjacent training paths the new step
  sits next to.
- **New Core:** `NremBindingReplay` — wall-clock of `binding_replay_step_batch` (operation
  granularity, same shape as the other training-step metrics). Fires every NREM batch when the
  flag is on; stays at zero when off (which is itself the control-arm signal).
- **New Fine:** `binding_replay_loss` — `counted!` accumulator holding the summed loss
  scaled ×1e6 (the registry has one accumulator per metric, so the step count rides
  `NremBindingReplay`; mean loss/step = binding_replay_loss.count ÷ nrem_binding_replay.count
  ÷ 1e6). The A/B-comparison-specific signal, exactly the Fine tier's stated purpose.
- **No new failure counter:** the new path is the same tensor operations as existing tracked
  paths (encoder forwards, sparse gather, attention, head forwards) with no I/O, no ORT, and no
  new invariant that can break independently; the one genuinely new surface — `forward_tracked`'s
  graph continuity — is pinned by a unit test (episode tensor backward reaches a binding weight's
  grad) rather than a runtime counter.
- **A/B pilot runs both arms with `fine-metrics` enabled and the write-up records deviations** —
  arm-vs-arm divergence beyond the headline metrics plus any deviation from baseline behavior
  (unexpected counter firings, metric regressions, health anomalies), per the standing instruction.

## Validation plan (before production adoption)

1. **Unit/integration tests — DONE 2026-08-15:** `forward_tracked` ≡ `forward(None)` numerically
   and stateless (`binding.rs::forward_tracked_matches_forward_none_and_is_stateless`); backward
   through `forward_tracked` moves grown units' zero-init binding-projection rows off zero and
   leaves the GRU temporal components untouched
   (`lib.rs::binding_replay_tests::binding_replay_trains_grown_binding_rows_when_active`);
   existing `forward` behavior unchanged (all pre-existing binding tests pass untouched).
   **Test-placement note (matters for reading item 2):** the grown-row gradient test lives
   in-crate, not in `tests/`, because grown units do not win k-WTA in any synthetic single-input
   setup on a test-sized step budget — `grow_region` inits their encoder columns at
   Normal(0, 0.01), far below the Kaiming-scale base columns, and the duty-cycle boost
   (≤ k/n ≈ 0.02, growing ~0.001/step) cannot bridge that z-score gap; grown units break into the
   winner set only on production timescales, via homeostatic rotation over many cycles of a
   diverse corpus. The in-crate test therefore forces grown units active via encoder-column
   surgery to pin the gradient path itself; the public-API integration test
   (`burn_path_contracts.rs::binding_replay_trains_only_active_binding_rows`) pins the control
   arm — never-active grown rows stay exactly zero (no dense-scatter leak), loss still falls.
   **Pilot implication:** in the item-2 A/B, grown-row movement (`binding_proj_row_norms` on the
   grown slice) is only expected in regions/cycles where grown units actually win k-WTA; a
   treatment that learns on base rows but shows no grown-row movement is not automatically inert —
   check winner composition before concluding.
2. **Isolated A/B pilot from the trained production state** — **DONE 2026-08-15, writeup gaps.md
   §35** (not fresh memory — fresh memory has no grown units, so it couldn't test the actual fix):
   copied `memory/` (weights + tiers) to two scratch dirs, ran control (`--binding-replay` off) vs
   treatment (on) for the same clip corpus and 10 cycles, both built and run with `fine-metrics`.
   Headline metrics: `mean_output_mse`, `per_region_mse`, `gist_recall_rate`/
   `within_session_gist_rate`, `warm_recall_l2`, plus the new `binding_replay_loss` and
   `NremBindingReplay` cost. Deviations recorded per the standing instruction. **Outcome: binding
   learned (loss −67.5%) and the §33 fix engages (47–96% of grown rows nonzero in 10 cycles vs 0 in
   control), but the predicted episode-drift consequence materialized (cross-session gist −53% and
   progressively rising warm_recall_l2 while within-session gist improved) and the NREM cost
   (+34.5%) displaced ~30% of replay traces — not a clean pass.** The pilot also falsified the
   validation-item-1 worry that grown units might not win k-WTA in production state: 791 cycles of
   duty-cycle homeostasis plus corpus diversity rotate them in from the first cycles.
3. Only after a winning pilot: enable by default (or adopt as the new control), with the
   episode-drift monitoring above kept on for the first production leg. **Blocked by item 2's
   outcome — the adoption path now requires the warm-store rebuild-from-cold mitigation and cost
   reduction first, then a repeat pilot (engineering-log.md §35 verdict).**
4. **Cost/drift-reduction step — implemented 2026-08-15, repeat pilot FAILED (engineering-log.md §36):** new
   `--binding-replay-steps N` CLI flag (default 1) caps `binding_replay_step_batch` calls per
   sleep cycle via a step-budget counter in the NREM loop
   (`sleep.rs::binding_replay_step_budget_remaining`); `N=0` disables the pass even with
   `--binding-replay` set. No new metric needed — `NremBindingReplay`/`binding_replay_loss`
   already cover the (now capped) call count and cost. **Repeat pilot at N=1 (20 cycles, n=1):
   wall-clock goal met (NREM −24.8%) but quality got worse, not better** — ~10× same-cycle
   `mean_output_mse` on cycles where the step fires, plus a new failure mode (6/20 cycles
   trace-starved via a salience/settle homeostatic cascade reacting to the degraded output). §33
   fix still engages under the cap (grown rows 30–78% nonzero). **Verdict: step-count capping
   decouples cost from drift instead of taming both — abandoned as the lever.** Not adopted.
   Candidates queued for the next attempt: reduce step magnitude (LR/loss-scale) instead of count;
   fuse into the existing fused-replay step instead of a separate optimizer pass; decouple the
   salience/settle cascade directly. No path chosen yet.
5. **Candidate (a), loss-scale reduction — implemented, result INVALIDATED (engineering-log.md
   §44, corrected 2026-08-16):** new `--binding-replay-loss-scale <f32>` flag (default 1.0, no
   behavior change) scaled the live-episode loss immediately before `.backward()`. Pilot at
   scale=0.1 (step cap left at its default of N=1) produced `mean_output_mse` +440.6% vs.
   control. **Not a valid dosage test:** the step's optimizer is `AdamW`, which normalizes each
   update by its running second-moment estimate, so a loss-tensor scale cancels out almost
   entirely before reaching the parameters. The +440.6% aggregate is statistically the same as
   §36's own aggregate (+445%) — the scale had no measurable effect, not a "4×" worsening (that
   figure compared this pilot's aggregate against §36's same-cycle-on-fired-cycles number, a
   different measurement). **Loss-scale retired as the lever.** Fixed 2026-08-16: the flag is
   renamed `--binding-replay-lr-scale` and now scales the learning rate passed to
   `optimizer.step()` directly (`crates/somnus-perception/src/lib.rs`), which does change the update
   magnitude. Not yet piloted (`scripts/ab_binding_replay_lr_scale.sh`). Two further candidates
   remain untried: fuse into the existing `fused_replay_wake_step_batch` step; decouple the
   salience/settle cascade directly (note: `sal_gate`/`settle_gate_val` are homeostatic
   controllers correctly reacting to genuinely degraded output, so as literally described this is
   symptom suppression — any pilot needs quality metrics as gating criteria, not just cascade
   metrics).
6. **Candidate (a) re-run with real LR-scale dosage — implemented and FAILED (engineering-log.md
   §45, 2026-08-16):** `--binding-replay-lr-scale 0.1` (step cap at default N=1), the actual
   dosage test item 5 was meant to be. Produced `mean_output_mse` +444.7% — statistically the same
   as §36's unscaled +445% and the invalidated §44 result, despite a genuine 10× weaker update.
   Failure shape differs from both prior pilots (per-region movement mixed rather than uniformly
   worse; `gist_recall_rate`/`warm_recall_l2` barely moved instead of collapsing), but the
   headline number doesn't improve at lower magnitude. **Candidate (a) is now closed in both
   variants.** This is solid evidence — not the flawed inference item 5 previously rested on —
   that the damage is not a dosage effect.
7. **Candidate (d), freeze the heads during binding-replay — implemented and unit-tested, pilot
   pending (2026-08-16):** found while scoping a (c) pilot, by reading `binding_replay_step_batch`
   against `fused_replay_wake_step_batch` rather than trusting this ADR's own "Mechanism" prose.
   The production `mean_output_mse` metric is the head-reconstruction MSE on **stored** episodes
   (`sleep.rs`'s `fused_replay_wake_step_batch` return value); this pass trains the same five head
   weights on **live** episodes, in the same NREM batch, through the same shared optimizer — a
   dosage-independent mechanism that would explain why both variants of candidate (a) failed
   identically at 10× different update magnitudes. New `--binding-replay-freeze-heads` flag
   (default false) removes the heads' own gradients from `GradientsParams` before
   `optimizer.step()` — Burn skips the update only for params with no registered gradient, so
   gradient still flows *through* the heads into the live episode tensor and on into the binding
   weights, but the heads stop moving under this step (they're still updated by the fused path).
   Verified: `binding_replay_freeze_heads_trains_binding_not_heads` (`crates/somnus-perception/src/lib.rs`)
   pins loss falling + binding rows moving + head weights staying byte-identical. **(d) is now
   favored for the next pilot** over (c) — it targets `mean_output_mse` directly via a
   code-grounded mechanism, where (c) targets a wake-side symptom (the salience/settle cascade)
   this hypothesis says isn't the actual cause. Pilot script: `scripts/ab_binding_replay_freeze_heads.sh`,
   not yet run. No path chosen until that pilot lands — per this project's working-with-user norm
   for exploratory/architectural calls, the user chose to scope this before running it.
8. **Candidate (d) piloted — FAILED, but the failure reveals the likely real root cause
   (engineering-log.md §46, 2026-08-16):** `mean_output_mse` +586.9%, same band as every prior
   pilot. Since heads were frozen and encoders are never in binding-replay's loss graph, there was
   no gradient path for the treatment arm to move this metric — the contradiction traced to
   `rebuild_ort_session` firing after every sleep cycle (not only on growth), so a drifting binder
   propagates into wake-time episode formation and then into later cycles' stored-episode training
   independent of any gradient-level mechanism. See status header above for the full chain. **Not
   yet confirmed by an isolating pilot** — the natural next step is disabling/delaying ORT rebuild
   during a binding-replay run (or forcing the CPU/burn wake path) to see if `mean_output_mse`
   recovers, rather than trying (b) or (c), neither of which touches this channel.

## What would invalidate this

- The pilot shows recall/quality regression from episode-space drift that warm-store turnover
  doesn't absorb (the consequence flagged above turns out real at this scale).
- The new term is inert: binding_replay_loss doesn't fall, or grown units' binding rows stay
  ~zero after many cycles (would indicate the heads can route around the binding — e.g. learn to
  ignore the episode's binding-shaped components — in which case the loss design, not just the
  gating, is wrong).
- The NREM wall-clock cost of the extra pass is materially larger than the ≪-cycle estimate at
  9.8k-unit scale (the Core metric exists to catch exactly this; if it fires large, the batched
  fused variant becomes a prerequisite, not an optimization).
- A future change makes replay traces' stored regional inputs unavailable or stale relative to
  current encoder state in a way the existing DER++ staleness filter doesn't catch (the new term
  reuses the same targets, so it inherits their validity assumptions).
