# Session state — somnus

> Crash-resume journal. Keeps the last 3 checkpoints (N, N-1, N-2) only — not a durable history.
> Everything else (decisions, research, campaign/pilot numbers, backlog triggers) is filed under
> `ai-docs/`; start at `ai-docs/README.md`. When adding a new checkpoint, drop the oldest.

> **✅ CHECKPOINT (clean), 2026-10-17 late evening — SHUTDOWN checkpoint, resume here.**
> - **State:** everything committed on `main` (HEAD `5861ba1`), nothing pushed, working tree clean. Gates were green at the
>   last full run (fmt, clippy default + fine-metrics, `cargo test --workspace`). Nothing is running: the last A/B
>   (`memory/ab_boost2`, boost strength 2, 20 cycles) was **killed ~2 min in on shutdown, no result**; its dir is scratch and
>   can be deleted. `memory/` is gitignored (pilot artifacts: `ab_rem_scoring`, `ab_binding_wake_lag[_s43]`, `ab_interp[_s43]`,
>   `ab_boost5`, `ab_persist_duty`, `ab_grown_init_scale`, `ab_video_encoder_cpu`, `diag_*`; all re-derivable, results are in docs).
> - **Done today (all documented in `ai-docs/gaps.md` §40/§49/§50 and ADRs 0025/0039/0041/0048):**
>   R1 fixed (REM scoring uses `forward_no_dc`; A/B: no regression) + R4 batched REM scoring (~2.5-3.7x faster in the live
>   A/B) + R8 partly + R10 `ColdArchive::append_batch`; all 10 `ab_*` snapshot arms converted to safetensors;
>   **§40 Whisper "GetElementType" failure SOLVED = shutdown race** (process::exit tears ORT down under the ASR thread's
>   in-flight decode; 27/27 failures within 0.5 s after the exit log line; fix `AsrShutdown::stop_and_wait`, verified on 6 runs);
>   new Core metrics `kwta_duty_dev_mean`, `kwta_duty_max` and per-region `kwta_duty_max_*`; per-region `kwta duty-cycle report` log
>   line (base vs grown slot share, mean |z|); `--video-encoder-cpu` (rejected by A/B: +20% cycle wall); `scripts/requirements.txt`.
> - **Experiment knobs added (all default = unchanged behaviour; each A/B'd at n=1 or 2 seeds):**
>   `--grown-unit-init-scale` (inconclusive: affects ~1.6% of units), `--persist-duty-cycles` (neutral; dominance survives restart;
>   `duty_cycles.bin` is now always written), `--kwta-boost-strength` (5 = HARMFUL, MSE x5-9; **2 untested**),
>   `--binding-wake-interp` (0.1 removes the settle-gate collapse in 2 seeds, wake MSE -36..-50% vs no-replay, but
>   `mean_output_mse` worse in both seeds; not adopted). Generic runner `scripts/ab_flag_pair.sh` (`CONTROL_ARGS`, `TREATMENT_ARGS`,
>   `SESSIONS`, `CYCLES`, `SEED`, `PILOT_ROOT`, `PATCH_TREATMENT_CMD`; ~18 GiB RSS per arm, arms in parallel).
> - **Prepared, not run:** `scripts/rescale_grown_units.py` (offline: rescales grown units' encoder columns to a target norm,
>   verified on a copy) for the direct test of the grown-unit init hypothesis. Run it as e.g.
>   `PILOT_ROOT=memory/ab_rescale PATCH_TREATMENT_CMD='python3 scripts/rescale_grown_units.py "$ARM_DIR/snapshot/weights.safetensors" 0.5'
>   TREATMENT_ARGS='--grown-unit-init-scale 0.01' SESSIONS=1 CYCLES=20 bash scripts/ab_flag_pair.sh` (needs a fine-metrics release
>   build; ~36 GiB free RAM + the GPU). Caveat: decoder rows of grown units stay ~0, so reconstruction may dip first.
> - **Open, needs the user (growth/boost design, gaps §50):** a few large-norm original units fire on every ingest while grown units
>   take most active slots with 10-240x smaller activations; boost 5 fixes the saturation but destroys reconstruction (so the dominant
>   units carry signal). ADR 0025's ReLU+L1 reopen trigger is recorded as met (no change). ADR 0041/0048: binding replay shelved;
>   next questions are a yardstick other than `mean_output_mse` for a lagged binder, and whether replay helps at all.
> - **Still open, not user-gated:** boost 2 A/B; rescale A/B above; language-region MSE (+217% in the R1 pilot, tiny absolute) to
>   watch in the next campaign; §40 residual hazard: other background threads running ORT at exit (gain/wake worker) not observed failing.
> - Habits to keep: `cargo test` before committing (the hook only runs fmt+clippy); never `pkill -f` a pattern that appears in the same
>   command line (it kills the shell: use `ps ... | grep -E "[p]attern"` and kill by PID); do not run a workspace test run next to an A/B
>   (CPU contention overran a tool timeout); check `free -g` before launching arms (two arms = ~36-46 GiB).
> - Resume prompt: "resume" -> read this file, then `ai-docs/gaps.md` §50 and ADR 0048 "Pilot results"/"Follow-up" sections.
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-10-17 (earlier) — codebase review fix pass (gaps.md §49), ADR 0050.**
> - Atomic snapshot writes (R2), single ORT build post-restore (R3), `ingest_region` + accessors (R11), `TensorF32Ext` (R12),
>   `run_sleep_cycle` 1205 -> 260 lines and `run()` 2006 -> 405 (`startup.rs`/`pipeline.rs`/`wake_state.rs`/`wake.rs`; R13/R14 with an
>   end-to-end characterization test), concurrent decoder init 32 -> 16 s (R6; now 3.4 s warm), automatic thread budgets (ADR 0050:
>   wake wall 448 -> 336 s), LTO opt-in only (R9), R7 hypothesis rejected, R19 KV-cache generation fixed for Qwen3.
> - Two mechanical-rename bugs were caught by diffing string-literal sets (st.* prefixes leaked into tracing/metrics keys).
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-10-16 — Q11 done: burn 0.22 + burn-flex + safetensors (ADR 0049).**
> - Merged to `main`. Device-based API, flex with rayon via a direct `burn-flex` dep, weights `*.safetensors` (burn-store). Parity vs
>   0.21 on real converted weights: 0 diffs at rtol 1e-3 (`scripts/compare_golden.py`, `ai-docs/research/burn-021-golden.json`).
> - Three pre-existing bugs fixed on the way: every pre-ADR-0048 snapshot was unloadable on `main`; `wake_binding` GRU/temporal_proj
>   never synced; the ORT binding session was exported from random weights before the snapshot restore (all earlier restored §33 pilots
>   affected). Branches `tooling/convert-weights-pre0048` and `tooling/golden-021` hold the migration tooling.
> - ffmpeg-the-third not migrated (ffmpeg-next 9.0.0 is current). CI also installs `ffmpeg`; not yet confirmed green on the runner.
