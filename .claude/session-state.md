# Session state — somnus

> Crash-resume journal. Keeps the last 3 checkpoints (N, N-1, N-2) only — not a durable history.
> Everything else (decisions, research, campaign/pilot numbers, backlog triggers) is filed under
> `ai-docs/`; start at `ai-docs/README.md`. When adding a new checkpoint, drop the oldest.

> **✅ CHECKPOINT (clean), 2026-10-17 (latest) — codebase review fix pass (gaps.md §49), ADR 0050.**
> - Done: atomic snapshot writes (R2), single ORT build post-restore (R3), `ingest_region` + accessors (R11),
>   `TensorF32Ext` (R12), `run_sleep_cycle` 1205 -> 260 lines with `SleepParams`/`SleepSignals` and an end-to-end
>   characterization test (R13/R14), concurrent decoder init 32 -> 16 s (R6), automatic thread budgets (ADR 0050:
>   wake wall 448 -> 336 s), LTO profile measured <=5% so opt-in only (R9), R7 benchmark rejected the hypothesis.
> - **Awaiting user:** R1 (do REM scoring passes `measure_mse`/`collect_high_residual_inputs` mutate k-WTA duty
>   cycles on purpose? history says accidental; switching to `forward_no_dc` changes REM numerics, needs an A/B).
>   R4 (batch REM scoring) is queued behind it.
> - Wake main loop of `run()` split (~1000 -> ~290 lines): `WakeState`, `wake.rs`, `startup.rs`; metrics line key set pinned by a test.
>   Two mechanical-rename bugs were caught by review (st.* prefixes leaked into tracing field names and into the metrics JSON keys);
>   string-literal sets verified equal to the pre-refactor tree. Remaining ~950 lines of `run()` are linear setup.
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
>
> **✅ CHECKPOINT (clean), 2026-10-16 — repo sweep done, queue = `gaps.md` §48, most items fixed.**
> - Sweep → `gaps.md` §48 (13 items); fix pass committed (`170aa76`, `b3b0136`, `badb099`, `9be497f`): `--seed` + `SEED` in A/B scripts, CI/hook run `cargo test --workspace` and fine-metrics clippy, `model_ref()` accessor, `cli.rs` split, new Core metric `nrem_stale_trace_skipped` (fired 2032× in a 2-cycle pilot — worth a look vs §33), scripts README/requirements.
> - Not done on purpose: `run_sleep_cycle` split (needs golden-output test under fixed seed), pedantic lint churn. Open decision: burn 0.22/burn-flex and ffmpeg-the-third 6 migrations.
> - **Still open:** ADR 0048 4-arm pilot not launched (use `--seed` now; SEED defaults to 42 in `ab_*.sh`).

>
> ---
