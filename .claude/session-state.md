# Session state — somnus

> Crash-resume journal. Keeps the last 3 checkpoints (N, N-1, N-2) only — not a durable history.
> Everything else (decisions, research, campaign/pilot numbers, backlog triggers) is filed under
> `ai-docs/`; start at `ai-docs/README.md`. When adding a new checkpoint, drop the oldest.

> **✅ CHECKPOINT (clean), 2026-10-16 (latest) — repo sweep done, queue = `gaps.md` §48, most items fixed.**
> - Sweep → `gaps.md` §48 (13 items); fix pass committed (`170aa76`, `b3b0136`, `badb099`, `9be497f`): `--seed` + `SEED` in A/B scripts, CI/hook run `cargo test --workspace` and fine-metrics clippy, `model_ref()` accessor, `cli.rs` split, new Core metric `nrem_stale_trace_skipped` (fired 2032× in a 2-cycle pilot — worth a look vs §33), scripts README/requirements.
> - Not done on purpose: `run_sleep_cycle` split (needs golden-output test under fixed seed), pedantic lint churn. Open decision: burn 0.22/burn-flex and ffmpeg-the-third 6 migrations.
> - **Still open:** ADR 0048 4-arm pilot not launched (use `--seed` now; SEED defaults to 42 in `ab_*.sh`).
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-09-14 (latest) — CI green again; deps refreshed; ai-docs audited;
> dead code pruned. §33/ADR 0048 pilot still not launched — unchanged from last checkpoint.**
> - **Pipeline fixes:** CI broke twice on `clippy::chunks_exact_to_as_chunks`, a lint CI's rustc
>   1.98 has that the local toolchain (1.96, now updated to match) didn't. First fix
>   (`254deca`) only covered `somnus-sensory`; a full repo grep turned up four more call sites
>   (`somnus-encoding`, `somnus/src/action.rs`, `somnus/src/lib.rs`), fixed in `01ee6ed`. Local rustup
>   updated to stable 1.98.1 to match CI exactly; going forward, verify locally with the literal
>   CI commands before pushing rather than trusting a stale local toolchain.
> - **`ai-docs/` audit:** dispatched to a fork with an explicit "don't edit, report only"
>   instruction — it ignored that and committed fixes itself anyway (flagged via SendFeedback).
>   Content was correct: `decisions/README.md` was missing the ADR 0048 row and undercounted ADR
>   0041's failed pilots (3 → 6, missing root-cause note); `constants.md` had 3 stale
>   `somnus/src/lib.rs` line citations (values unchanged). Fixed, committed (`37789f3`), verified,
>   pushed. Everything else in `ai-docs/` (constants values, architecture.md/.svg structure,
>   gaps.md §33/§39 tracking, tech-stack, glossary, roadmap, recent ADR statuses) checked out
>   clean against the code — no other drift found.
> - **`cargo update` (`573b136`):** ~80 packages bumped to latest semver-compatible versions, no
>   manifest changes. Checked for available major-version bumps too (`cargo-outdated` — its `-v`
>   full-depth mode is memory-hungry on this tree, hit 32GB RSS and had to be killed; the
>   non-verbose `--root-deps-only` pass plus a manual crates.io cross-check of every direct
>   dependency both came back clean): every direct dep is already on the latest version its
>   current `^`-constraint allows, nothing to bump without a manifest edit.
> - **Dead-code pruning (`db4b5b3`):** audited `#[allow(dead_code)]` sites and unreferenced pub
>   API (cargo machete: no unused deps; codebase-memory-mcp graph queries gave false
>   positives/negatives from doc-comment mentions and unresolved generic dispatch — verified
>   every candidate by grep before touching anything). Removed 3 confirmed-dead items:
>   `ActionDecoder::gain`/`update_gain()` (superseded by `GainWorker`'s background-thread path,
>   zero real callers), `PerceptionProcessor::disable_ort_inference()` (orphaned —
>   `--force-cpu-inference` works by never building `onnx_dir`, not by calling this),
>   `EpisodicBuffer::capacity()` (unused accessor). Left `projection_step`/`projection_step_batch`
>   alone — ADR 0012's amendment explicitly documents retaining the old variant on purpose.
> - **Verified throughout:** every push confirmed green on CI (not just local checks) before
>   moving to the next step; final state has `cargo fmt --check`, `cargo clippy --all-targets -D
>   warnings` (default and `fine-metrics`), `cargo build --all-targets`, and `cargo test
>   --workspace` all clean.
> - **Still open, unchanged from last checkpoint:** ADR 0048's 4-arm binding-wake-lag validation
>   pilot has not been launched — needs a decision from the user on when to run it.
> - **All work committed and pushed** as of this checkpoint (`db4b5b3` is HEAD).
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-09-13 — §33 candidate (e) implemented: decoupled
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
