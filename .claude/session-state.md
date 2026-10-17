# Session state — somnus

> Crash-resume journal. Keeps the last 3 checkpoints (N, N-1, N-2) only — not a durable history.
> Everything else (decisions, research, campaign/pilot numbers, backlog triggers) is filed under
> `ai-docs/`; start at `ai-docs/README.md`. When adding a new checkpoint, drop the oldest.

> **✅ CHECKPOINT (clean), 2026-10-17 (latest, end of day) — resume here.**
> - **State:** all work committed on `main`, nothing pushed, working tree clean. Gates green (fmt, clippy default +
>   fine-metrics, `cargo test --workspace`). Branches `tooling/convert-weights-pre0048` and `tooling/golden-021`
>   hold the burn-migration tooling (see ADR 0049).
> - **Done today (details in `ai-docs/gaps.md` §48/§49, ADRs 0049/0050):** burn 0.22 + flex + safetensors; atomic
>   snapshot writes; ORT built once post-restore (+ the restore bug); thread budgets (wake wall 448 -> 336 s);
>   `run_sleep_cycle` 1205 -> 260 lines and `run()` 2006 -> 405 (loop ~285) with `startup.rs`/`pipeline.rs`/
>   `wake_state.rs`/`wake.rs`; KV-cache generation fixed for Qwen3 (R19, verified token-identical to the reference).
> - **R1 UPDATE (later 2026-10-17): option (a) implemented and committed** (both REM scoring passes use `forward_no_dc`; regression test added, fails on old code; workspace tests green). **The 2-arm A/B is still TODO** (deferred: user's GPU/RAM workload was running). All 10 `ab_*` snapshot arms converted to safetensors 2026-10-17 (originals kept; main-side load not yet exercised). R4 DONE too (batched REM scoring, ~5x in a synthetic bench; see gaps.md R4). A/B pilot RUN 2026-10-17 (results in gaps.md R1): no regression, REM scoring ~2.5-3.7x faster, region-4 MSE +217% (tiny abs) to watch; `memory/ab_rem_scoring/control-worktree` still to remove (`git worktree remove`).
> - (Original R1 note, now resolved in code:) `measure_mse` and `collect_high_residual_inputs` (REM scoring passes) call
>   `KWtaLayer::forward`, which advances k-WTA duty cycles (~16k spurious steps/cycle at 3.3k traces, several time
>   constants at alpha=1/1000). History: `measure_mse` is from 2026-05-10; `forward_no_dc` came 2026-05-11 (ADR 0011,
>   settling only); nothing documents the REM side effect, so it reads as accidental. Options: (a) switch both to
>   `forward_no_dc`, run a 2-arm A/B on a campaign snapshot (fine-metrics on; compare mean_output_mse, per-region MSE,
>   growth events, gist rates, duty-cycle stats; record deviations); (b) keep it, document as intended in an ADR note.
>   **R4** (batch the REM scoring matmul: ~11.6 s + ~12 s/cycle) is queued behind it.
> - **Still queued (user said keep):** ADR 0048 4-arm pilot (now runnable on a build with the restore fixes; use `--seed`,
>   `SEED` defaults to 42 in `ab_*.sh`), R8 (k-WTA alloc/dedupe), R10 (cold append batch, not worthwhile today),
>   moving the video encoder off MIGraphX (CPU EP measured 23 ms vs 54 ms isolated), other `ab_*` snapshot dirs still
>   need conversion to safetensors (tool on `tooling/convert-weights-pre0048`).
> - Habits to keep: run `cargo test` before committing (hook only runs fmt+clippy); after mechanical renames, diff the
>   string-literal sets (two silent regressions were caught that way this session).
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-10-17 (latest) — codebase review fix pass (gaps.md §49), ADR 0050.**
> - Done: atomic snapshot writes (R2), single ORT build post-restore (R3), `ingest_region` + accessors (R11),
>   `TensorF32Ext` (R12), `run_sleep_cycle` 1205 -> 260 lines with `SleepParams`/`SleepSignals` and an end-to-end
>   characterization test (R13/R14), concurrent decoder init 32 -> 16 s (R6), automatic thread budgets (ADR 0050:
>   wake wall 448 -> 336 s), LTO profile measured <=5% so opt-in only (R9), R7 benchmark rejected the hypothesis.
> - **Awaiting user:** R1 (do REM scoring passes `measure_mse`/`collect_high_residual_inputs` mutate k-WTA duty
>   cycles on purpose? history says accidental; switching to `forward_no_dc` changes REM numerics, needs an A/B).
>   R4 (batch REM scoring) is queued behind it.
> - `run()` 2006 -> 405 lines (loop ~285): `startup.rs`, `pipeline.rs`, `wake_state.rs`, `wake.rs`; metrics line key set pinned by a test. R19 (`generate_kv` panic on the Qwen3 KV models) found and FIXED: layout is now derived from the sessions; KV output verified token-identical to the reference path.
>   Two mechanical-rename bugs were caught by review (st.* prefixes leaked into tracing field names and into the metrics JSON keys);
>   string-literal sets verified equal to the pre-refactor tree. 
> - Always run `cargo test` before committing (the pre-commit hook only runs fmt + clippy; one commit this session
>   landed with failing startup tests and was fixed in the next).
> - **Still open:** ADR 0048 4-arm pilot. Nothing pushed.
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-10-16 — Q11 done: burn 0.22 + burn-flex + safetensors (ADR 0049).**
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
