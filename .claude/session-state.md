# Session state — somnus

> Crash-resume journal. Keeps the last 3 checkpoints (N, N-1, N-2) only — not a durable history.
> Everything else (decisions, research, campaign/pilot numbers, backlog triggers) is filed under
> `ai-docs/`; start at `ai-docs/README.md`. When adding a new checkpoint, drop the oldest.

> **✅ CHECKPOINT (clean), 2026-09-13 (latest) — §33 candidate (e) implemented: decoupled
> wake-time binding snapshot (ADR 0048). Not yet piloted.**
> - **Scoped, reviewed, then implemented directly** (not delegated to opencode/Lemonade — its
>   first real attempt on this file was destructive, deleted an existing struct field, then blew
>   its 65280-token context window on the very next step; reverted via `git checkout` and built by
>   hand per the director-mode escape valve).
> - **What changed:** `PerceptionModel` gains `wake_binding: BindingArea<TrainB>`. New
>   `sync_wake_binding()` hard-copies the 8 live-trained tensors (region_0..4, w_q/w_k/w_v) from
>   `binding` into `wake_binding`. All wake-time call sites (`try_bind`/`settle_episode`/
>   `binding_only`, plus `export_weights_for_onnx`) now read `wake_binding`, never `binding`
>   directly. `grow_region`/`grow_region_seeded`/`prune_region` reshape both copies in lockstep
>   (shape never lags, only trained values do). New CLI flag `--binding-wake-lag-cycles` (default
>   1 = today's always-immediate behavior). 2 new unit tests (`wake_binding_stays_frozen_until_synced`,
>   `growth_and_prune_keep_binding_and_wake_binding_in_lockstep`) plus a regression pin for a real
>   burn quirk found during the build.
> - **Two implementation-time findings, not caught by review** (both fixed, documented in ADR
>   0048's "Implementation findings"): (1) `binding.clone()` does NOT preserve materialized weight
>   values — burn's derived `Linear`/`Module` `Clone` re-runs each `Param`'s lazy random
>   initializer independently per clone (confirmed empirically, now a regression test); fixed by
>   explicitly copying each trained tensor's `.val()` at construction, same pattern as
>   `sync_wake_binding`. (2) The sync gate must NOT be conditioned on `--binding-replay`:
>   `homeostatic_decay` (`sleep.rs:896`) mutates the trained binding tensors every cycle
>   unconditionally, so gating on the flag would leave `wake_binding` frozen forever on the
>   default/production path (binding-replay off). Also caught: `binding_only`'s `gru_hidden`
>   readback needed to switch to `wake_binding` too (missed in the original scoping pass, found by
>   re-grepping every `.gru_hidden` site after the call-site swap).
> - **Verified:** `cargo fmt --check`, `cargo clippy --all-targets -D warnings`, and the full test
>   suite all clean in both feature configs (default and `fine-metrics`) across `somnus-perception`
>   and `somnus`. Live smoke run (`--binding-replay --binding-wake-lag-cycles 2`, small clip set)
>   completed 2 full sleep cycles with no errors.
> - **Not done: the actual validation pilot.** ADR 0048's plan (4-arm sweep: control, lag=1
>   baseline, lag=10, lag=30, over 30-50 cycles, `mean_output_mse` reported as a cycle-range trend
>   rather than one aggregate mean) has not been run — needs a decision from the user on when to
>   launch it, since it's a real campaign-affecting pilot.
> - **All work committed** — `gaps.md` §33, ADR 0048, and this checkpoint updated in the same
>   session; code changes committed separately per the usual code/docs split.
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-09-13 — §33: confirming diagnostic run, ORT re-export
> hypothesis DISCONFIRMED; real root cause is backend-independent live-weight drift, now confirmed.**
> - **Sixth pilot (`engineering-log.md` §47):** repeated the candidate-(d) freeze-heads pilot
>   (`scripts/ab_binding_replay_freeze_heads_cpu.sh`, new script) with `--force-cpu-inference` added
>   to both arms, structurally removing ORT (`rebuild_ort_session` never runs; confirmed
>   `ort_binding_forward` fired 0 times in both arms). `mean_output_mse` still +462.0% — same
>   445-587% band as every prior pilot with ORT active. **ORT re-export is not the cause.** The
>   mechanism is backend-independent: whatever serves wake-time binding inference (ORT or burn) uses
>   live weights that changed during the immediately preceding NREM cycle; those episodes get
>   stored; later cycles train heads against the now-drifted stored-episode distribution. This is
>   §35's original "episode-space drift" finding, now positively confirmed rather than inferred from
>   head/gradient elimination.
> - **New deviation:** treatment arm's salience/settle gate degenerates much harder than §46's
>   comparable pilot — `settle_gate_dynamic` collapses to exactly 0.0 by cycle 10, `sal_threshold`
>   exceeds its designed [0,1] range. Not explained; flagged per standing A/B deviation policy, not
>   investigated further.
> - **Docs corrected same session:** `gaps.md` §33 (verdict + priority table), `engineering-log.md`
>   §47 (new entry), ADR 0041 (both the changelog and its stale "ORT inference picks up trained
>   weights" Consequences bullet) — the ORT-specific framing in all three was actively wrong, not
>   just unconfirmed, and is now corrected in place rather than left to accumulate as a stale lead.
> - **Open — root cause confirmed, no path chosen.** (a) and (d) closed as failed leads. (b) and (c)
>   are no longer blocked by ORT-specific reasoning (there is no ORT-specific cadence), but neither
>   addresses the now-confirmed real constraint (live weight updates immediately consumed by the next
>   wake phase). A candidate that would: decouple wake-time binding inference from the NREM-training
>   copy of the weights (e.g. an EMA/lagged snapshot for episode formation, reconciled on a slower
>   cadence). **Not scoped or piloted — needs a decision from the user before more pilots run.**
> - **All docs committed as of this checkpoint.**
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-08-16/17 — §33: candidates (a) and (d) both closed as
> failed; (d)'s failure revealed the likely real root cause (ORT re-export drift), not yet
> confirmed. (b) and (c) both stalled pending that confirmation.**
> - **LR-scale dosage pilot (candidate (a) re-run, `engineering-log.md` §45):** corrected §44's
>   loss-scale flaw (`AdamW` normalizes away a loss-tensor scale — that pilot was a no-op).
>   Renamed `--binding-replay-loss-scale` → `--binding-replay-lr-scale`, scales the LR passed to
>   `optimizer.step()` directly. Piloted for real at scale=0.1: `mean_output_mse` still +444.7% —
>   genuinely dosage-invariant. **Candidate (a) closed, both variants.**
> - **Freeze-heads pilot (candidate (d), `engineering-log.md` §46):** scoping a (c) pilot surfaced
>   a sharper hypothesis — the production `mean_output_mse` metric is the head-reconstruction MSE
>   on *stored* episodes, while `binding_replay_step_batch` trains the same heads on *live*
>   episodes through the same optimizer. Implemented `--binding-replay-freeze-heads` (excludes head
>   grads from the optimizer step; pinned by a dedicated unit test before piloting). **Piloted —
>   FAILED: `mean_output_mse` +586.9%, same 445-587% band as every prior pilot.** This is stronger
>   than a failed pilot: with heads frozen and encoders never in binding-replay's loss graph, there
>   is no gradient path for the treatment arm to move this metric at all — yet it moved.
> - **Root cause found by reading `crates/somnus/src/lib.rs:2451-2469` (not by more pilots):
>   `rebuild_ort_session` fires after every sleep cycle once ORT is active**, not only on growth as
>   ADR 0041 previously implied — true from cycle 1 at this campaign's 10x-growth scale. Real
>   channel: binding weights change → re-export picks them up → next wake phase forms episodes
>   through the new binder → those get **stored** as new traces → later cycles train heads against
>   these drifted stored episodes, which is exactly what `mean_output_mse` measures. Explains
>   dosage invariance in both (a) variants, why freezing heads (d) didn't help, and reproduces
>   §35's original episode-drift finding directly. **Not yet confirmed by an isolating pilot**
>   (disable/delay ORT rebuild during a binding-replay run, or force the CPU/burn wake path) —
>   leading hypothesis, not proven.
> - **Blocks (b) and (c) as previously scoped** — neither touches the ORT re-export cadence this
>   hypothesis implicates. **No path chosen. Next step on resume: propose the confirming diagnostic
>   to the user** (isolate whether ORT re-export drift is really the channel) before running any
>   more optimizer/loss variant pilots on §33.
> - **Scripts cleanup (unrelated, done same session):** removed `ab_lora.sh`/`train_lora.py`/
>   `ab_q1_text_hidden.sh`/`export_qwen3.py` (dead, referenced Qwen2.5 files that no longer exist
>   anywhere); fixed `profile.sh`/`quantize_kv.py`'s stale Qwen2.5 paths and constants; fixed a real
>   `NameError`-on-first-run bug in `train_lora_qwen3.py` (caught only because the script has never
>   actually been run, gaps.md §39).
> - **All work is committed as of this checkpoint** — `gaps.md` §33, `engineering-log.md` §46, and
>   ADR 0041 all updated with the §46 finding in the same commit as this checkpoint.
