# Session state — somnus

> Crash-resume journal. Keeps the last 3 checkpoints (N, N-1, N-2) only — not a durable history.
> Everything else (decisions, research, campaign/pilot numbers, backlog triggers) is filed under
> `ai-docs/`; start at `ai-docs/README.md`. When adding a new checkpoint, drop the oldest.

> **🔄 CHECKPOINT (background pilot running), 2026-08-16 (latest) — §33 candidate (a) closed for
> real via LR-scale dosage; candidate (d) (freeze heads) scoped, implemented, unit-tested, and
> mid-pilot at session end.**
> - **LR-scale dosage pilot (candidate (a) re-run, `engineering-log.md` §45):** corrected the §44
>   loss-scale pilot's flaw — `binding_replay_step_batch` steps `AdamW`, which normalizes updates
>   by their second-moment estimate, so scaling the *loss* before backward (§44) was a no-op; §44's
>   +440.6% and §36's unscaled +445% were statistically the same result, not "4× worse" as
>   previously recorded. Renamed `--binding-replay-loss-scale` → `--binding-replay-lr-scale`, now
>   scales the LR passed to `optimizer.step()` directly. Piloted for real at scale=0.1: still
>   `mean_output_mse` +444.7% — genuinely dosage-invariant this time. **Candidate (a) is closed in
>   both variants.**
> - **Scoping a (c) pilot (decouple the salience/settle cascade) surfaced a sharper hypothesis
>   instead — candidate (d).** Found via code reading (not ADR prose): ADR 0041's "gradients reach
>   the encoders" claim was wrong (materialized `Vec<f32>` severs the graph — corrected in the
>   ADR); and the production `mean_output_mse` metric is the head-reconstruction MSE on *stored*
>   episodes (`fused_replay_wake_step_batch`), while `binding_replay_step_batch` trains the *same*
>   five head weights on *live* episodes in the same NREM batch through the same optimizer — a
>   dosage-independent interference mechanism that explains why both (a) variants failed
>   identically. Implemented `--binding-replay-freeze-heads` (excludes head grads from the
>   optimizer step; gradient still flows through them into binding) and pinned it with a dedicated
>   unit test (`binding_replay_freeze_heads_trains_binding_not_heads`) before piloting.
> - **Pilot launched, NOT YET COMPLETE at session end:** `scripts/ab_binding_replay_freeze_heads.sh`
>   running in the background (control arm was in progress at last check; monitor task `bqfrlx3d3`
>   watching `/tmp/claude-1000/-home-falcon-projects-fsvm88-somnus/943cc060-cdf3-40c9-bc25-56a29e65b413/scratchpad/ab_binding_replay_freeze_heads.log`).
>   **On resume: check that log file for the finished comparison table before doing anything else
>   on §33** — if the session ended before it completed, the pilot process may still be running
>   unattended or may have finished with results never written up. Verdict not yet known.
> - **Candidate (b) (fuse binding-replay into `fused_replay_wake_step_batch`'s existing optimizer
>   step) was scoped/discussed but not implemented** — reframed from a pure cost optimization into
>   a real second test of the interference hypothesis (joint gradient vs. sequential Adam steps on
>   shared head weights). User was mid-decision on whether to implement/pilot it now or wait for
>   (d)'s result when the session paused for shutdown. **Ask the user this question on resume if
>   (d)'s pilot has landed and no further instruction was given.**
> - **Scripts cleanup (unrelated, done same session):** removed `ab_lora.sh`/`train_lora.py`/
>   `ab_q1_text_hidden.sh`/`export_qwen3.py` (dead, referenced Qwen2.5 files that no longer exist
>   anywhere); fixed `profile.sh`/`quantize_kv.py`'s stale Qwen2.5 paths and constants; fixed a real
>   `NameError`-on-first-run bug in `train_lora_qwen3.py` (`_Qwen25LMWrapper` call site never
>   renamed when the class was renamed to `_LMWrapper`) — caught only because the script has never
>   actually been run (gaps.md §39).
> - **All work through this point is committed** (`6810cf6` is HEAD). The only outstanding thing is
>   the still-running pilot process and its eventual write-up.
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
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-08-16 — docs cleanup: split the resolved-items archive into ADRs +
> a trimmed engineering log, per explicit user request ("go through all of it, document reasons why
> in ADRs, remove it from the changelog where already documented").**
> - **Renamed** `gaps-resolved.md` → `engineering-log.md` (66 cross-file §N citations updated
>   mechanically, no renumbering). New header declares its real scope: bug fixes, pilot/A-B numeric
>   results, campaign narratives — not decision rationale.
> - **Synced 7 existing ADRs** (0034/0036/0037/0038/0039/0040/0041) against the log before trimming
>   anything — found ADR 0034 never mentioned Qwen3 at all (the KV-cache fast path silently fell
>   back to the slow path for a month after ADR 0036 made Qwen3 the sole LM); added the missing
>   outcome. ADR 0041 was missing its own §44 Validation-plan entry.
> - **Wrote 5 new ADRs** (0042 BatchTopK SAE, 0043 replay ratio N=4, 0044 DER++, 0045 DINOv3
>   encoder, 0046 audio-env-encoder candidates) for decision-shaped log sections that had real
>   alternatives-considered material but no ADR. Each verified current code against the log's claims
>   rather than trusting it — caught one real staleness (§1 said `--batch-topk` defaults false; code
>   has it flipped true post-A/B).
> - **`research/gpu-unification-design.md`:** found one real staleness — §4's Binding-graph
>   description still described the pre-ADR-0040 dense-matmul contract (`sp_i[1,N_UNITS]` inputs),
>   superseded 2026-08-12 by the Gather-based sparse rewrite. Fixed inline, with a pointer to ADR
>   0040. Also flagged two long-open questions (§13) that got overtaken by events without being
>   revisited: D_PROJ scaling policy (still fixed at 128 after a 10x campaign that was supposed to
>   trigger revisiting it) and EMA-head-sync Option B's N_UNITS≥4096 trigger (crossed ~2.4×, not
>   acted on). Phases 0-4 status lines were all still accurate — not touched.
>
