# Session state — somnus

> Crash-resume journal. Keeps the last 3 checkpoints (N, N-1, N-2) only — not a durable history.
> Everything else (decisions, research, campaign/pilot numbers, backlog triggers) is filed under
> `ai-docs/`; start at `ai-docs/README.md`. When adding a new checkpoint, drop the oldest.

> **✅ CHECKPOINT (clean), 2026-08-16 (latest) — gaps.md triage: §41 (change-gated video encoder)
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
> ---
>
> **✅ CHECKPOINT (clean), 2026-08-15 — ADR 0041's third adoption attempt, candidate (a)
> loss-scale reduction, tried and FAILED WORSE than both prior pilots. Detail: engineering-log.md
> §44, ADR 0041 status + Validation-plan item 5.**
> - **New `--binding-replay-loss-scale <f32>` flag** (default 1.0, no behavior change): scales the
>   live-episode loss tensor immediately before `.backward()`. Treatment: scale=0.1, step cap left
>   at its §36 default of N=1 — isolates the scale variable, not a combination.
> - **Result: worse than both §35 (unbounded) and §36 (step-count cap alone).** `mean_output_mse`
>   +440.6% (every region worse, +25% to +81%), `traces_replayed` −49.2%, the same salience/settle
>   cascade signature as §36 (`settle_iters_sum`/`settle_trigger_count` both +39%) but ~4× the
>   quality damage at 10× *lower* gradient magnitude — the opposite of the theory being tested.
> - **Verdict: loss-scale reduction abandoned as the lever**, same conclusion as §36 reached about
>   step count. Reading across all three failed pilots: the salience/settle cascade looks like a
>   structural consequence of live-episode training itself, not a dosage effect tunable via step
>   count or loss magnitude — **candidate (c) (decouple the cascade directly) is now favored** over
>   (b) (fuse into the existing optimizer step). **No path chosen yet — ask the user before writing
>   more code here.**
> - **Docs:** ADR 0041 status line + new Validation-plan item 5; `gaps.md` §33 updated (candidate
>   (a) tried, (b)/(c) queued).
