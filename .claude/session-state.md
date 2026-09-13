# Session state — somnus

> Crash-resume journal. Keeps the last 3 checkpoints (N, N-1, N-2) only — not a durable history.
> Everything else (decisions, research, campaign/pilot numbers, backlog triggers) is filed under
> `ai-docs/`; start at `ai-docs/README.md`. When adding a new checkpoint, drop the oldest.

> **✅ CHECKPOINT (clean), 2026-09-13 (latest) — §33: confirming diagnostic run, ORT re-export
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
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-08-16 — gaps.md triage: §41 (change-gated video encoder)
> closed as a rejected null-result, no other open items needed action.**
> - User asked to close open gaps; scoped down to triage-only (no new pilots, no campaign-affecting
>   runs). Reviewed every open item in `gaps.md`: §4/§7/§10/§11/§15/§23/§40/§42 already carry real
>   revisit triggers and needed no change; §33 and §39 need actual implementation/campaign decisions
>   the user explicitly deferred this round.
> - **§41 verdict: reject, close as null-result.** A/B (`engineering-log.md` §43) showed
>   threshold=0.05 skips 93.8% of video-encode calls but costs +21.8% `mean_output_mse` (+10.8%
>   visual-region), with no stated compute bottleneck to justify the tradeoff. Moved from `gaps.md`
>   into `engineering-log.md` §43 (updated in place, same section, now marked REJECTED); removed
>   from `gaps.md`'s open list and priority table. `--video-change-threshold` stays default-off
>   (0.0), flag stays opt-in for future revisit if wake-path compute ever becomes the bottleneck.
> - **This file's scope narrowed** (this checkpoint): trimmed to a 3-checkpoint crash-resume
>   journal per explicit user request. Moved the old "Key files" pointers into `ai-docs/README.md`'s
>   new Tooling section, and the "Deferred backlog" trigger table into `gaps.md`'s new
>   "Deferred backlog" section; `CLAUDE.md` updated to point at `gaps.md` instead of this file.
>   Everything in the old "Historical" checkpoint log was already covered by `engineering-log.md`/
>   ADRs (confirmed by the 2026-08-15 §37 audit) or self-documenting in code/scripts, with two
>   exceptions folded into `engineering-log.md` §34: the `setsid` background-launch lesson (added)
>   and the pre-commit-hook fix (already self-documented in `scripts/setup-hooks.sh`'s own comment,
>   no separate entry needed).
