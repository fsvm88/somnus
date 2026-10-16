# Session state — somnus

> Crash-resume journal. Keeps the last 3 checkpoints (N, N-1, N-2) only — not a durable history.
> Everything else (decisions, research, campaign/pilot numbers, backlog triggers) is filed under
> `ai-docs/`; start at `ai-docs/README.md`. When adding a new checkpoint, drop the oldest.

> **✅ CHECKPOINT (clean), 2026-10-16 (latest) — Q11 done: burn 0.22 + burn-flex + safetensors (ADR 0049).**
> - Merged to `main` from branch `burn-0.22`. Device-based API, flex with rayon requested via a direct
>   `burn-flex` dep, weights now `*.safetensors` (burn-store). Parity vs 0.21 on real converted weights:
>   0 diffs at rtol 1e-3 (`scripts/compare_golden.py`, `ai-docs/research/burn-021-golden.json`).
> - **Three pre-existing bugs found and fixed on the way:** (1) every pre-ADR-0048 snapshot was
>   unloadable on `main`; (2) `wake_binding` GRU/temporal_proj never synced from `binding`; (3) the ORT
>   binding session was exported from random weights BEFORE the snapshot restore, so every restored
>   run's first wake phase inferred on random weights (cycle-1 per_region_mse .075 -> .002 after the fix).
>   All earlier restored §33 pilots are affected by (3).
> - Converted snapshots written next to the originals: `/data/somnus/memory/snapshot/` and
>   `ab_binding_replay_freeze_heads_cpu/control/snapshot/` only; other `ab_*` dirs still need
>   `tooling/convert-weights-pre0048` (branch). Originals untouched.
> - Queue: gaps §48 Q14 (size rayon's pool vs ORT; `encode_video` +28%/call observed). ffmpeg-the-third
>   not migrated (ffmpeg-next 9.0.0 is current). CI now also installs `ffmpeg` (media_file_it needs the CLI) -
>   not yet confirmed green on the runner. Nothing pushed.
> - **Still open:** ADR 0048 4-arm pilot, now worth running on this build (restore bug fixed).
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-10-16 — repo sweep done, queue = `gaps.md` §48, most items fixed.**
> - Sweep → `gaps.md` §48 (13 items); fix pass committed (`170aa76`, `b3b0136`, `badb099`, `9be497f`): `--seed` + `SEED` in A/B scripts, CI/hook run `cargo test --workspace` and fine-metrics clippy, `model_ref()` accessor, `cli.rs` split, new Core metric `nrem_stale_trace_skipped` (fired 2032× in a 2-cycle pilot — worth a look vs §33), scripts README/requirements.
> - Not done on purpose: `run_sleep_cycle` split (needs golden-output test under fixed seed), pedantic lint churn. Open decision: burn 0.22/burn-flex and ffmpeg-the-third 6 migrations.
> - **Still open:** ADR 0048 4-arm pilot not launched (use `--seed` now; SEED defaults to 42 in `ab_*.sh`).
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-09-14 — CI green again; deps refreshed; ai-docs audited;
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
