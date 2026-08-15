# Session state — somnus

> **✅ CHECKPOINT (clean), 2026-08-15 (latest) — pre-commit hook restored; ADR 0041 adoption-path
> step 1 (`--binding-replay-steps N`) implemented and repeat-piloted — FAIL, worse than the
> original pilot. Detail: gaps.md §36, ADR 0041 status.**
> - **Pre-commit hook:** CI was red on the last 2 pushes (`cargo fmt --check` failure —
>   `binding_row_norms.rs` committed unformatted; no hook had ever caught this — `.git/hooks` isn't
>   versioned, so there was nothing to "re-setup" from). Fixed formatting; added tracked
>   `scripts/hooks/pre-commit` (fmt --check + clippy, mirrors `ci.yml`) + `scripts/setup-hooks.sh`
>   to install it. CI green (`ba642b2`). **Re-run `scripts/setup-hooks.sh` if hooks ever vanish
>   again** (fresh clone, `.git` recreated).
> - **`--binding-replay-steps N` (default 1), `a1ea230`:** step-budget counter in the NREM loop
>   (`sleep.rs::binding_replay_step_budget_remaining`, tested) caps `binding_replay_step_batch`
>   calls/cycle. Full test/clippy/fmt clean, both feature states verified, CI green.
> - **Repeat pilot at N=1 (`ab_binding_replay.sh`, CYCLES=20, n=1/arm, fine-metrics both arms) —
>   FAILS, worse than §35:** NREM wall-clock did improve (−24.8%, vs §35's +34.5% — cap works as a
>   pure cost lever). But quality got *worse*: on the 13/20 cycles where the (rarer, larger) step
>   fired, `mean_output_mse` ran ~10× control same-cycle; new failure mode — 6/20 treatment cycles
>   replayed **zero** traces (episodic buffer starved into NREM), traced to a salience/settle
>   homeostatic cascade reacting to the degraded output (`sal_threshold` +39%, `settle_gate_dynamic`
>   −44%, `settle_iters_sum` +154%). §33 fix still engages under the cap (`binding_row_norms`:
>   control 0/9,089 grown rows nonzero, treatment 30–78%). **Verdict: step-count capping decouples
>   cost from drift instead of taming both — abandoned as the lever.** n=1, but a 10× same-cycle MSE
>   gap and 6-cycle trace starvation are far past plausible noise.
> - **No path chosen for the next attempt** — three candidates queued (gaps.md §36/ADR 0041),
>   undecided: (a) reduce step *magnitude* (LR/loss-scale) instead of count, (b) fuse into the
>   existing `fused_replay_wake_step_batch` step instead of a separate optimizer pass, (c) decouple
>   the salience/settle cascade directly. **Ask the user before writing more code here.**
> - **Docs:** gaps.md §36 (new) + §33/§35 rows corrected; ADR 0041 status + validation item 4
>   updated. Pilot artifacts (`memory/ab_binding_replay/`, 2.9G) `rm -rf`'d after writeup —
>   production `memory/` untouched throughout (pilot always uses a separate scratch copy).
> - **Pushed:** `ba642b2`, `a1ea230`, `0dcf023` — all CI-green.
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-08-15 — ADR 0041 isolated A/B pilot run, written up as gaps.md §35.
> Verdict: mechanism validated, §33 fix engages in production state, predicted episode-drift
> materialized → NOT adopted.**
> - **Setup:** `scripts/ab_binding_replay.sh` (new) — production state copied per arm (snapshot+
>   warm+cold+onnx, btrfs reflink), 1 session × 10 cycles, campaign corpus, both arms fine-metrics,
>   sequential for clean wall-clock.
> - **Headline — §33 fix engages in production state:** control grown rows stay 0/8,901 in all
>   regions; treatment **47–96% nonzero after 10 cycles**. Grown units DO win k-WTA in the trained
>   state — 791 cycles of duty-cycle homeostasis (overused base units carry negative boost) plus
>   corpus diversity rotate them in from the first cycles.
> - **Mechanism:** `binding_replay_loss` 0.141→0.046 (−67.5%); `nrem_binding_replay` 28 steps @ mean
>   1.44s/step; control fired neither; 0 NaN; clean exits.
> - **Predicted consequence materialized:** warm store mixes 151k untrained-binder episodes with new
>   trained-binder ones → `warm_recall_l2` rises progressively in treatment (0.44→0.89) while control
>   improves (1.29→0.31–0.44); cross-session gist −53% and falling; within-session gist +13.3% (same-
>   generation recall fine) — the mixed-generation-store fingerprint, exactly the ADR's flagged risk.
>   **Cost:** +34.5% NREM wall-clock, displaced ~30% of replay and 64% of blends — the ≪-cycle
>   estimate didn't hold; a cheaper/fused step variant becomes an adoption prerequisite per the
>   ADR's own invalidation clause. Other deviations: growth divergence 10,032 vs 9,925, salience/
>   settle/wake-MSE movements; n=1 per arm.
> - **Adoption path (ADR + §35):** (1) warm-store rebuild-from-cold at switchover, (2) cost
>   reduction (fused step/fewer binding steps), (3) repeat pilot showing drift contained. Flag stays
>   opt-in.
> - **Also this session:** fixed a `BindingReplayLoss` doc bug — `counted!` accumulates into `count`
>   only, so the documented "count=steps, total=loss" semantics were impossible; doc now says count
>   accumulates summed scaled loss, step count rides `NremBindingReplay`. Comparison script fixed
>   accordingly.
> - **Committed + pushed:** `35a2346` (pilot infra) + writeup commit (gaps.md §35, ADR 0041, somnus-
>   metrics doc fix). Production `memory/` untouched.
> - **Queued next (superseded by the entry above — see it for what actually happened):** step
>   budget (`--binding-replay-steps N`, start N=1) → repeat pilot (≥20 cycles, n≥2) → adopt only on
>   a clean repeat. **Key correction found while queueing:** the ADR's "rebuild warm from cold"
>   mitigation doesn't work as written — `EpisodeRecord` (SMNCOLD3) stores only episode vectors, not
>   regional inputs, so episodes can't be re-formed through a new binder from cold. Real absorption
>   = warm-store turnover (~30 cycles at the 200k cap) + the step budget slowing the binder's change
>   rate; SMNCOLD4-with-regionals only if those prove insufficient (~5× cold bloat).
>
> ---
>
> **✅ CHECKPOINT (clean), 2026-08-15 — resumed after a hard machine lockup; ADR 0041 implemented,
> verified, committed, pushed.** The locked session had: taken the §33 decision (option b),
> committed ADR 0041 + the standing A/B-metrics instruction (`7a63d09`), implemented the whole
> feature, and was mid-debug of a failing grown-rows test when the machine died. This session
> finished that step.
> - **Root cause of the failing test — not a gradient bug:** grown units' encoder columns init at
>   Normal(0, 0.01) vs Kaiming-scale base columns, so grown units never win k-WTA in a synthetic
>   single-input setup (duty-cycle boost ≤k/n≈0.02, growing ~0.001/step, can't bridge the z-gap).
>   Gradient flow itself is sound: burn's `select` backward scatters correctly, `Param::val()`
>   preserves the graph, and a new in-crate test proves grown rows learn once grown units are
>   active. **In production, grown units break into the winner set via homeostatic rotation over
>   many cycles of a diverse corpus** — recorded in ADR 0041's validation plan.
> - **Tests landed:** `binding_replay_trains_grown_binding_rows_when_active` (in-crate, forces
>   grown units active via encoder-column surgery) and `binding_replay_trains_only_active_binding_
>   rows` (public-API control: never-active grown rows stay exactly zero).
> - **Feature summary (`4f7aa1f`):** `BindingArea::forward_tracked` (graph-preserving stateless
>   None-context forward; `forward` refactored onto shared helpers, behavior unchanged),
>   `PerceptionProcessor::binding_replay_step_batch` (+`binding_proj_row_norms` inspection),
>   `--binding-replay` CLI flag (default off), new Core `NremBindingReplay` + Fine
>   `BindingReplayLoss` metrics (`Metric::COUNT` 100→102).
> - **Also fixed (`1bc8af1`):** `ort_grown_region` encoder test under `mn10_env` — the completed
>   campaign re-exported `perception_encoder.onnx` at baseline ENV_DIM=512, unmatchable by a
>   960-dim build; skip logic now covers that mismatch.
> - **Verified:** fmt/clippy/285 tests clean, both feature states green, `somnus --help` shows the
>   flag. ADR 0041 → "implemented & verified (validation item 1)".
> - **Pushed:** `origin/main` synced at `1bc8af1`.
>
> ---
>
> **✅ SHUTDOWN (clean), 2026-08-15. The 10x growth campaign is COMPLETE.** Resumed the interrupted
> 2026-08-14 session: committed its leftover docs (`836083a`), then reconstructed and finished the
> campaign run.
> - **Interrupted state:** session 80 (`SESSIONS=1 CYCLES=15`) had been killed mid-cycle-8 by a hard
>   shutdown. On-disk state snapshot-consistent at `[9645]×5`, 15 units short of 9,660.
>   `resume_growth_campaign.sh` correctly refused (session-budget exhausted, not unit-target-based).
> - **Completion run:** first relaunch (session 81) ran 1 cycle then got killed by a launcher-side
>   process-group kill on a 120s timeout — **operational note: background launches must use `setsid`
>   (+ `</dev/null`, output to a log file) or they die with the launching command.** Relaunched
>   detached as session 82 — all 15 cycles ran clean. Growth: 9645→9675 (c5, **target crossed**)→
>   9769 (c8)→9821 (c12).
> - **Final state: `[9821]×5` = 101.7% of the 9,660 target.** Campaign lifetime: 791 cycles, 188
>   growth events (mean batch 46.8, cadence 1/4.21), 82 sessions, 2026-07-12→2026-08-15. Health: 0
>   panics, 0 segfault restarts, 0 ORT fallbacks (final session), 7 `misc_warning` lifetime.
>   `mean_output_mse` 0.0430→0.0112 (−74.0%) first-vs-last-10; warm 151,725/cold 158,534 records.
> - **Docs:** gaps.md §16 complete, §20 pointer, new §34 + priority row; architecture.md updated.
>   SVG untouched (depicts the n=1024/k=20 baseline, not production campaign state).
> - **Next-session entry point:** campaign completion unblocks gaps.md §33 (BindingArea weights
>   never trained) — ask the user. Whisper-ASR-panic root cause also queued (§5). Standing threads
>   unchanged: ADR 0038 prereq 3 (trigger-gated), deferred items A-N.
> - **Pushed:** everything, `origin/main` in sync.
>
> ---
>
> **⏸️ SHUTDOWN (clean), 2026-08-14.** Test-coverage push round 2 — closed the two remaining thin
> spots §31 named: `somnus-perception` burn-path training functions, `somnus-memory` I/O paths. **27 new
> tests, workspace 281 passing.**
> - **`somnus-memory` (35→47):** `snapshot.rs` (3→9, round-trip + malformed-fallback pins for every
>   small state file); `recollection.rs` (0→3, `promote()` semantics, unknown-id no-op, ordering);
>   `cold.rs` (13→14, `iter_from` seek); `semantic.rs` (8→10, insert skip-at-cap, `records_since`).
> - **`somnus-perception` (+15, new `tests/burn_path_contracts.rs`):** `measure_mse` monotonic-decrease
>   pin (the growth-gate signal); bias-free head linearity; `head_weights_flat` round-trip;
>   save/load weights exact round-trip + `.sizes` sidecar; `grow_region_seeded` bookkeeping;
>   `collect_high_residual_inputs`; duty cycles; `encoder_z_only`/`binding_only` dims/determinism.
> - **No production code changed** — one dev-dep added (`tempfile`, pinned). Verified: fmt/clippy/
>   281 tests clean, both feature states re-verified.
> - **Docs:** gaps.md §32 + priority row.
> - **Same-day continuation:** (1) investigated the §25 note that binding weights never train —
>   **CONFIRMED REAL, gaps.md §33** — all training paths consume episodes as stored data; ADR 0008's
>   "revisit in Phase 6" never happened; grown units' zero-init binding rows never learn (8,797/9,821
>   units/region invisible to episodes). Options a/b/c listed, decision pending until campaign
>   finishes. (2) Pushed local commits. (3) Cold-archive btrfs fallback — ok for now. (4) Whisper ASR
>   panic — queued in gaps.md §5.
> - **Not pushed at shutdown:** this session's commits were local-only; check `git log
>   origin/main..HEAD` before assuming push state.
>
> ---
>
> **⏸️ SHUTDOWN (clean), 2026-08-13 (second session).** Completed the remaining advisor-ranked test
> targets. **54 new tests, workspace 254 passing.**
> - **`somnus-encoding` (0→41 unit tests):** 38 inline + `tests/encode_models.rs` (3 model-gated
>   integration tests, skip-don't-fail pattern). Coverage: `resample.rs` (rate-ratio tracking, DC
>   passthrough, 440Hz preservation, chunk-length contract); `mel.rs` internals (windows, padding,
>   HTK/Slaney/Kaldi scales, filterbanks, exact hand-computed cases); `image.rs` (YUYV conversion,
>   Bgr8/Yuyv422 paths — previously 0 coverage, ImageNet normalisation, bilinear resize incl.
>   hand-computed `align_corners=False` case).
> - **Two small production changes the resample tests forced:** rubato 5 silently broke the
>   documented contract (same drift class as §17/§22/§24) — `SincResampler::new` now rejects zero
>   rates, `process` enforces `input.len()==chunk_frames`. Behavior on every production path
>   unchanged. Full writeup: gaps.md §31.
> - **`whisper_asr.rs` (0→8, somnus lib 91→99):** argmax edge cases; special-token-ID drift pin
>   (SOT/EOT/EN/TRANSCRIBE/NO_TIMESTAMPS map to expected strings); `max_tokens=0` short-circuit;
>   determinism on zeros and on real encoder output. Note: Whisper EOT is id 50257, "<endoftext>"
>   (decodes to two newlines, hence the trim in decode_greedy).
> - **`somnus-metrics` (3→8, 10 with fine-metrics):** exhaustive enum invariants over all 100 `Metric`
>   variants (COUNT==ALL.len(), index/name uniqueness, snake_case well-formedness) + proptest
>   record/snapshot/reset round-trip. `proptest` added as dev-dep.
> - **Verified:** fmt/clippy/254 tests clean, both feature states, `somnus --help` smoke run.
> - **Docs:** gaps.md §31; `tech-stack.md` staleness fixed (rubato candidate→committed v5,
>   ffmpeg-next 8→9, rayon/clap candidate→committed, dead candle declarations removed).
> - **Next-session entry point:** test push has no remaining advisor-ranked targets. Otherwise
>   nothing queued as a bug. Growth campaign at 98.9% of 10x target.
> - **Not pushed:** commits local-only (`88be92e`, `451e8dd`, `5670a70` + docs commit); last push to
>   origin was `4115bf9`.
>
> ---
>
> **⏸️ SHUTDOWN (clean), 2026-08-13.** Continuation into "test every function properly."
> - **Dependency-sweep wrap-up:** advisor review of the prior session's 45 new `somnus/main.rs` tests
>   caught two issues, fixed (`f98e670`): `current_process_rss_bytes` test now skips (not fails) on
>   a 0 RSS reading; `forward_frames` "drops oldest when full" test rewritten to model real call
>   sites correctly (drain as a clone of dst's own receiver).
> - **`somnus` split into `src/lib.rs` + thin `src/main.rs`** (`01581f6`, explicit user choice): all
>   logic moved into `lib.rs` as `pub async fn run()`; `main.rs` is a 4-line entry point. Unblocks
>   `cargo test -p somnus --lib`. No behavior change, verified via full build/clippy/fmt/test + real
>   `somnus --help` run.
> - **8 pure functions extracted from `run()`'s sleep-trigger/EWMA/S4 logic, unit tested**
>   (`66c23a0`): `parse_prior_beta_regions`, `decide_should_sleep`, `compute_effective_sleep_max_
>   secs`, `update_wake_ewma`/`recompute_max_wake_secs`, `dissipate_wakefulness_pressure`,
>   `global_prior_cosine`, `s4_ring_all_high`. 27 new tests (somnus lib 64→91).
> - **Most of the remaining ~2000 lines in `run()` is I/O/setup wiring, not pure logic** —
>   diminishing returns. User chose to stop rather than continue into `somnus-encoding`/
>   `whisper_asr.rs`/`somnus-metrics` or dig deeper into `run()`.
> - **Next-session priority (advisor-ranked):** (1) `somnus-encoding` 0 unit tests incl. `resample.rs`;
>   (2) `whisper_asr.rs` 0 tests, on every request's critical path; (3) `somnus-metrics` 3-5 tests for a
>   67-variant enum.
> - **Not committed:** nothing outstanding; three commits above local-only; last origin push
>   `3d785d8`.
>
> ---
>
> **⏸️ SHUTDOWN (clean), 2026-08-12.** Implemented AND deployed the sparse-gather binding-graph
> rewrite live. Committed `19e0f52`, `3949b52`, `ad69560` (ADR 0040, gaps.md §30).
> - **Deployed live, confirmed via `OrtRunBinding`:** 2-cycle verification leg (session 79, budget
>   already exhausted at 98.3%). Zero ORT fallback, clean rebuilds. **40-59% reduction in
>   `OrtRunBinding`'s real per-call cost** (2.15-2.83ms→1.17-1.30ms), including one cycle catching a
>   live growth event (9,493→9,549) with correct re-export.
> - **Campaign now:** session 79, `[9549]×5`, **98.9% to target**, clean exit, zero panics/
>   segfaults.
> - **Two unrelated pre-existing environment blockers fixed:** (1) `ffmpeg-next` 8 didn't build
>   against installed ffmpeg 9.0 — bumped to `"9"` (not downgrading ffmpeg, per user instruction).
>   (2) `models/` had silently diverged from `/data/somnus/models` — it's git-tracked in part
>   (README.md, qwen3_tokenizer.json are real files); first attempt (full symlink) was wrong and
>   self-corrected — restored as a real directory, symlinked only the individual gitignored
>   artifacts.
> - **Next-session entry point:** binding rewrite fully done. Campaign at session-budget ceiling
>   short of unit target (98.9%) — same standing question as always.
> - **MIGraphX probe: PASS.** `Gather` fed by a runtime int64 index, at campaign scale (n=9,660,
>   k=190), max abs diff 1.49e-07.
> - **Numerically verified exact** (~1e-8 vs. dense numpy reference), both CPU and MIGraphX EP,
>   baseline and campaign scale. Found and worked around a MIGraphX compiler bug (`Reshape`+matmul
>   crashed `simplify_reshapes`), unrelated to the Gather question.
> - **Measured win** via a 3rd arm on `bench_binding_projection.py`: **54-59% reduction in
>   `OrtRunBinding`'s per-call cost** at campaign scale (0.85ms→0.35-0.39ms).
> - **gaps.md §28/§29 and ADR 0038 corrected** — both mischaracterized `OrtRunBinding` as
>   "fixed-size, region-count-independent"; the dense projection this replaces was actually 75-76%
>   of that graph's cost.
> - **Rust contract change:** `SparseActivation::to_exact_k` (new), `run_binding` now takes
>   `&[SparseActivation; 5]` directly. `k_active` travels via `weights_manifest.json` from Rust.
> - **Verified:** `somnus-perception` build/test/clippy/fmt clean. `ort_grown_region.rs` rewritten for
>   the new contract, both integration tests skip correctly against pre-rewrite ONNX artifacts.
>
> ---
>
> **⏸️ SHUTDOWN (clean, user-requested), 2026-08-11.** User wants to scale further/faster on the
> *same* hardware. Redirected through corrections, landed on a validated queued next step.
> - **Corrected two wrong premises:** (1) both `run_encoder`/`run_binding` are already single fused
>   ORT calls across all 5 regions — no "5 separate dispatches" to batch. (2) The binding graph is
>   **not** region-size-independent — `sp_i @ W_region_i` scales with `n_units` and is the dominant
>   cost at campaign scale, not GRU+attention. Corrected §28/ADR 0038's mischaracterization.
> - **Read all of `ai-docs/`** before committing to a direction. Verdict on porting §25's sparse-
>   gather fix (burn/CPU, 11.4x) into the ONNX binding graph: no hard blocker, two edge cases in
>   scope (`lang_zero` placeholder feeds k=0; a dense→sparse re-derivation path can yield fewer than
>   k_active nonzeros on a tie). `k` must travel via `weights_manifest.json`, never re-derived in
>   Python (this class of bug has hit the repo 3× — §17, §22, §24).
> - **Built and ran a real benchmark** (`scripts/bench_binding_projection.py`) before implementing:
>   real `build_binding_graph` vs. a variant with the 5 region-projection matmuls removed, timed on
>   real MIGraphX. **Result at campaign scale (n_units=9,660): the dense projection is 75-76% of the
>   binding graph's per-call cost** (memory-bandwidth-bound — each call reads the full ~4.9MB×5
>   weight matrix regardless of input sparsity) — reverses an earlier FLOP-based estimate
>   (~24%/5-10%). **Conclusion: the rewrite is a real, large, validated win.**
> - **Queued (not started, "put it in queue"):** (1) run the MIGraphX op probe on `Gather`; (2) if
>   clean, implement the rewrite + thread k/indices through the manifest + change `run_binding`'s
>   Rust signature; (3) verify via the existing `OrtRunBinding` metric + numerical parity (~1e-6);
>   (4) write an ADR + correct §28/ADR 0038.
> - **Committed:** `scripts/bench_binding_projection.py` only.
>
> ---
>
> **⏸️ SHUTDOWN (clean, user-requested), 2026-08-04.** No code changes (docs-only session-state
> update).
> - **Housekeeping found (not this session):** two commits landed outside any recorded session —
>   `215b7b5` (git-filter-repo history purge) and `2c423dd` (migrated memory/clips/data/models to
>   `/data/somnus/`). Found the `memory`/`clips`/`data` symlinks missing from the working tree —
>   recreated them; campaign data intact (timestamps matched the 2026-07-16 shutdown state).
> - **Campaign explicitly not resumed** — safely resumable via `scripts/resume_growth_campaign.sh`
>   (state: session 78/130, cycle 10, 98.3% to target, unchanged since 2026-07-16).
> - **Reviewed a Cloudflare blog post** on their inference stack (FP8 KV-cache, INT4 weights,
>   multi-tenant MoE cache-integrity) — assessed not applicable (somnus has no comparable multi-tenant
>   KV-cache contention). No action.
> - **Discussed further k-WTA scoring optimization.** Three options: (1) quantize scoring weights —
>   cheap, but §28 found this matmul isn't the dominant cost; (2) hierarchical/coarse-to-fine WTA
>   (ADR 0038's target) — real upside, blocked on its own prereq 3, trigger not fired; (3) batch/
>   fuse the 5 regions' ORT calls — orthogonal lever. No decision; asked what's motivating "faster."
> - **Clarification received same session:** goal is scaling on the *same* hardware (hardware
>   ceiling, not more hardware). Reprioritized: (1) do first — quantization + batching/fusing the
>   ORT *binding* graph (GRU+attention), since §28 found binding is the dominant cost, pure
>   engineering, no architecture change; (2) second, longer-horizon — hierarchical/coarse-to-fine
>   WTA, changes the actual scaling law, still prereq-3-blocked. **Proposed next step (not agreed):**
>   benchmark the ORT binding graph with the same fine-metrics treatment §28 gave the scoring
>   matmul. Session ended before confirmation.
>
> ---
>
> **⏸️ SHUTDOWN (clean, user-requested), 2026-07-16.** Everything committed and pushed (`3b06375`).
> **§28/§29, resolving 2 of ADR 0038's 3 prerequisites:**
> - **Prereq 1 (§28):** the exported ONNX encoder graph has the identical `O(d_input×n_units)` dense
>   matmul as `burn`, no sparsity applied before scoring. Live cost (from existing metrics, ~9,400-
>   9,493 units/region): ~1.28ms/call, 1,585 calls — small in absolute terms, not even the dominant
>   ORT cost (fixed-size `OrtRunBinding` costs 2-3× more/call).
> - **Prereq 2 (§29):** no roadmap item exists past the current 10x/honeybee-order campaign;
>   combined with prereq 1, hierarchical WTA isn't worth building for the campaign actually running
>   right now.
> - **Correction caught by advisor:** initial instinct was "closed — not needed" — overstates the
>   finding (no roadmap naming mouse-order ≠ honeybee-order is a decided ceiling; the project's own
>   brain-scale-comparison doc frames eventual scale-gap closure via exactly this technique). **Kept
>   the ADR `proposed`/deferred** with an explicit trigger (campaign target moves past honeybee-
>   order, or encoder/binding cost balance shifts). **Do not mark this ADR closed on §28/§29 alone.**
> - No code changes (pure metrics analysis).
>
> ---
>
> **⏸️ SHUTDOWN (clean, user-requested), 2026-07-15.** Everything committed/pushed (`5607eb7`). ADR
> 0038's first prerequisite got real numbers as a side effect of metrics work: `kwta_score_matmul`
> 2.15s/640 calls (~3.4ms/call), `measure_mse` 4.21s/128 calls at ~148k warm/9,493-unit scale. Other
> two prerequisites untouched.
> - **§27:** `SleepWarmRebuild` metric surfaced `SemanticStore::rebuild_from_recent` costing 46-51s/
>   cycle (~70% of total) at ~148k warm records — `hnsw_rs` has no delete API, forcing a full serial
>   rebuild every cycle once the warm cap hits. Fixed: switched to `hnsw_rs::parallel_insert_slice`
>   (rayon). Same graph/eviction semantics, spread across cores. **Verified on real production data**
>   (copy of the campaign's actual cold archive, 148,447 records, `git stash`/`pop` old-vs-new):
>   47.6s→2.85-2.94s, **16.7× speedup**. New `crates/somnus-memory/examples/bench_warm_rebuild.rs`
>   (committed) is a reusable regression harness.
> - **Growth campaign:** stopped, session 78/130, cycle 10, `[9493]×5` = **98.3% to target**, warm
>   148,447/cold 155,256.
>
> ---
>
> **⏸️ SHUTDOWN (clean, user-requested), 2026-07-14.** Working tree clean; `25d8174` + `b5b5849`
> (base + full-coverage somnus-metrics, research note, ADR 0038) committed this session.
>
> **`somnus-metrics` full-coverage expansion** (on top of ADR 0039's base two-tier system): user asked
> to instrument "every stupid step... evaluate it against an A/B run." 3-way parallel inventory over
> somnus-perception/somnus-memory/somnus-sensory+main.rs, instrumented essentially all remaining uninstrumented
> data-handling ops (operation granularity; hot loops stay covered by an enclosing phase metric).
> `Metric` enum **24→67 variants**: somnus-perception +25 (binding forward, ORT run_encoder/run_binding,
> fused-replay sub-phases, ORT lifecycle, growth/prune, REM self-probes, EMA sub-steps,
> refresh_autodiff_state, wake_worker::train_step), somnus-memory +10 (cold open/truncate/evict,
> semantic warm-load, recollection promote, snapshot save/load), somnus-sensory +1 (MediaFileDecode),
> somnus binary +7 (action-decoder init/generate, gain-worker, ASR decode, audio resample, language
> embedding, record_episode). Two patterns documented in ADR 0039: private-`_inner`-method split,
> immediately-invoked-closure wrap. Verified incrementally (rebuild/retest after every conversion);
> live pilot recorded 40 distinct metrics simultaneously, cross-thread ones confirmed safe, untriggered
> conditions correctly stayed at zero. Docs: ADR 0039 (Update + Implementation-patterns sections),
> gaps.md §26.
>
> **Base conversion (superseded by the above, kept for the discovery trail):** ADR 0039, two tiers
> sharing one `Registry` — Core (`timed_core!`/`counted_core!`, unconditional, the "always reported"
> tier) and Fine (`timed!`/`counted!`, behind `fine-metrics`, bare passthrough when disabled). All
> `somnus-perception` ingest phases, all `sleep.rs` wall-clock phases (8 via macro block-wrap, 3 via
> direct `METRICS.record()` for structurally-entangled spans), all `somnus-encoding` encoder timers, all
> `main.rs` sites converted. Found/fixed two duplicate-measurement bugs (main.rs was re-timing
> encoder calls already timed by somnus-encoding). Verified: build/clippy/test clean both feature
> states, two isolated pilot runs cross-checked old-field vs. registry values (all within truncation
> noise), confirmed `kwta_score_matmul` dominates `kwta_select_topk` ~124-165× at baseline scale
> (bearing on ADR 0038, not yet at campaign scale). `somnus-memory` deliberately left unconverted (no
> pre-existing hand-rolled timers there).
>
> **Research note + ADR 0038:** new section in `sparse-coding.md` confirming the encoder scoring
> matmul (`KWtaLayer::forward`) is inherently `O(input_dim×n_units)` regardless of `k_active` —
> output sparsity doesn't help, every candidate must be scored before top-k. Surveyed three
> mechanisms: local/blocked connectivity (MoE-style), hierarchical/coarse-to-fine WTA, approximate
> top-k sampling. **ADR 0038 (`proposed`):** picked hierarchical/coarse-to-fine WTA — preserves exact
> top-k within blocks, degrades gracefully, may reuse HNSW machinery for the coarse stage. Three
> prerequisites: (1) benchmark actual scoring cost both paths, (2) confirm this matters at honeybee
> vs. next-order-of-magnitude scale, (3) numerical validation that block-local competition doesn't
> break growth/duty-cycle/binding assumptions. Implementation not started.
>
> ---
>
> ## Historical (pre-2026-07-14, superseded by the campaign narrative above — compressed timeline)
>
> - **2026-07-13, §25 sparse-aware binding projection:** `BindingArea::forward` materialized a dense
>   `[1,n_units]` vector from a ~98%-zero k-WTA activation. Replaced with sparse gather + small
>   matmul, numerically identical, **11.4× speedup** isolated benchmark (n=8,655, k=169). Deployed
>   live (session 72). **Correction (2026-07-14):** dormant on ORT-active campaigns — `perc_bind_us`
>   times `try_bind`, which dispatches to `ort.run_binding`, not `burn`'s path, past the §22
>   crossover. Kept deployed (strictly better, costs nothing dormant) for pre-crossover/CPU-only/
>   ORT-fallback cases. Also noticed: no training function calls `BindingArea::forward` — its
>   weights receive no gradient from any path (this became §33/ADR 0041 later).
> - **2026-07-13, §24 two ORT bugs fixed and confirmed live:** (1) binding-side `to_dense_n(N_UNITS)`
>   hardcoded baseline 1024 instead of actual grown size, both in `lib.rs` and `onnx_session.rs` —
>   fixed to derive from actual data (`8718099`). (2) encoder-side `ENC_DIMS` hardcoded language
>   input at 896, stale since ADR 0036 raised LANG_HIDDEN to 1024 — fixed to derive from live
>   constants (`debd6c4`). Both regression-tested. Live confirmation: zero ORT failures across
>   ~8,000 log lines; `perception_total_ms` 2,158ms/1,457ms (sessions 60/61) vs. 250,099ms fully-
>   broken vs. ~6,900-7,200ms pre-ORT baseline — **confirmed ~3-4× real speedup**, retracting ADR
>   0037's original "~2.1-2.4×" (measured on the broken path).
> - **2026-07-12/13, campaign restarts (5 total, all clean, state snapshotted every cycle):**
>   Restart 1 (§21) — `SemanticStore::dump()` leaked a full HNSW dump every cycle forever (461
>   orphaned pairs, 11G) because reload permanently disables in-place overwrite; fixed by clearing
>   stale dumps before writing (11G→192M). Restart 2 (§22/ADR 0037) — implemented auto-enable ORT
>   inference (`should_use_ort_inference`) past the CPU/GPU crossover (~1,230 units/region);
>   surfaced two pre-existing bugs in `export_perception_graph.py` (stale LANG_HIDDEN, hardcoded
>   N_UNITS/K_ACTIVE) fixed via `infer_dims()`. Restart 3/4 — deployed §24's two bug fixes above.
>   Restart 5 — deployed §25's sparse binding projection.
> - **Cold-archive note (2026-07-13):** `max_cold_records_effective`=0 (unlimited) all campaign —
>   `sysinfo` can't match the btrfs mount, falls back to no cap per ADR 0024's documented behavior
>   (not a bug). Growing correctly regardless (append-only ledger by design).
> - **2026-07-12, §17/§18/§19 — three growth-mechanism fixes, all tested, all committed:**
>   §17 K_ACTIVE now scales with region growth (`scaled_k(n_units)` preserves baseline ratio;
>   K_ACTIVE=20 was hardcoded, sparsity would've fallen ~10× over the campaign). §18 optimizer reset
>   scoped to the grown region only (was wiping AdamW momentum for the entire model on every growth/
>   prune event — burn-optim state is already keyed by ParamId, so the full reset was pure waste;
>   deleted). §19 novelty-proportional growth batch size (GROWTH_BATCH_MIN=16/MAX=128 replacing a
>   flat 64, scaled by how far the trigger missed threshold via `gist_novelty`/`mse_novelty`).
> - **2026-07-12, growth pilot 3/"v2":** validated §17-19 live — mechanism confirmed working (14
>   growth events, 4-7 cycle cadence, `k_active` tracked growth correctly, `per_region_mse` 0.0136→
>   0.0048). **Recalibration:** mean batch 35.6/event (nearly half the old flat 64, since most
>   triggers land at moderate not severe novelty) → **10x campaign re-estimated at ~9.3-11.5h**
>   (was ~3.9h), ~2.4-3× longer — not a regression, §19 correctly making growth severity-proportional.
> - **2026-07-11, expanded-corpus growth pilot + podcast corpus + doc audit:** added
>   `scripts/download_podcasts.sh` (10 shows, 10-90min filter) — 56 episodes/5.6GB in
>   `clips/podcasts/`. Pilot 2 (145-clip combined corpus, 75 cycles): **no plateau** — cadence/
>   quality indistinguishable from pilot 1 (16 events, `per_region_mse` 0.0135→0.0039, final sizes
>   uniform `[2048]×5`) — confirms growth is gate-limited (3-cycle cooldown), not signal-limited.
>   Full documentation audit (delegated staleness sweep, applied directly): fixed stale open
>   questions in architecture.md, renumbered gaps.md's priority table (`#` had drifted from `##`
>   since ~item 7), restored session-state.md's lost deferred-backlog trigger table from git history
>   (`03eb9b5`), added missing Phase 14 to roadmap.md, added missing ADR callouts to README.md.
> - **2026-07-09, growth-rate pilot campaign (10x scaling, honeybee-order target ~54,000 units,
>   corrected from an earlier mouse-order miscalibration):** growth mechanism already fully
>   implemented (cascade-correlation-style, ADR 0024 §2c gate, 3-cycle cooldown, GROWTH_BATCH=64) —
>   no new code. 75-cycle pilot: 16 growth events (~1/4.7 cycles, gate-limited not signal-limited),
>   final sizes `[2048,1984,1984,1984,1984]`, no plateau, `per_region_mse` improved alongside growth,
>   21.8s/cycle avg, 1.7GB disk. **Extrapolation: ~633 cycles ≈ ~3.9h for 10x** (later revised up by
>   §19, see pilot 3 above).
> - **2026-07-09, M2D-CLAP (item 8) — NOT PURSUED, no code:** feasibility research only — no ONNX
>   export precedent anywhere (dynamic positional-embedding interpolation, nnAudio STFT frontend —
>   same fragility class mn10 was chosen to avoid), non-commercial license. +3 mAP claim judged not
>   worth the export/license risk given item 9's precedent (paper-benchmark quality and somnus-pipeline
>   quality diverge). Docs-only, gaps.md §8.
> - **2026-07-09, EfficientAT mn10 (item 9) — ❌ REJECTED:** A/B (5 sessions × 2 cycles, fresh memory
>   both arms): control (CLAP-HTSAT) avg MSE 0.011966 vs. treatment (mn10) 0.030588, **+155.6%
>   worse**, 9/10 cycles regressed, no reversals. Default env encoder stays CLAP-HTSAT.
>   Implementation kept as an opt-in `mn10_env` Cargo feature (clean, parity-tested, may be useful
>   for deployment-size later). gaps.md §9.
> - **2026-07-08, DER++ auto-λ Phase 3 — implemented then A/B REJECTED:** EWMA 1/variance weighting
>   for `sae_loss`+`wake_loss` combination (`--der-auto-lambda`, default false). A/B: control avg
>   0.0487 vs. treatment 0.0847, **+73.8% worse**, 10/10 cycles regressed, no reversals — likely
>   early-EWMA-variance noise causing destabilizing weight spikes. Not adopted; flag stays opt-in,
>   default false. Lesson recorded: prefer Bash `run_in_background` over manual `nohup ... &` — a
>   stray un-killed background process once collided with a clean rerun writing to the same path.
> - **2026-07-06:** change-gated video encoder + SVD null-result shipped (`9bba42b`, `b410ad7`).
>   High replay ratio (N=1/2/4) validated, **N=4 adopted** (`d5ce2ab`, `f0fa9c6`): -70.9% output_mse
>   for +16.6% wall-clock vs N=1; N=2 more efficient (-49.1%/+7.6%) but not chosen. gaps.md §2.
> - **EfficientAT mn10 implementation detail (2026-07-08, now A/B'd above):** new `Mn10Mel` mel
>   frontend (EfficientAT's `AugmentMelSTFT`, Kaldi-style filterbank, parity-tested at 1e-4). Export
>   script bakes in only the MobileNetV3 backbone (no mel frontend — avoids reintroducing an STFT op
>   into the graph). New `mn10_env` Cargo feature swaps REGION_DIMS[1] 512→960 and all cross-modal
>   offset math. **Region-1 MSE is not comparable across arms** (different target distributions,
>   CLAP contrastive vs. mn10 raw log-mel) — verdict uses the mean of the other 4 regions instead.
>
> **Earlier baseline (pre-2026-07-06), for reference:**
> - DINOv3-ViT-S adopted (drop-in 384-dim replacement for DINOv2-S): visual-region MSE -95.6%,
>   `warm_recall_l2` +61.5%, no confound. `ai-docs/a_b_dinov3_findings.md`, gaps.md §6.
> - Qwen3-0.6B adopted as sole action-decoder LM (ADR 0036, supersedes ADR 0035); Qwen2.5 support
>   fully removed. ~22% lower `mean_proj_loss`, coherent generation vs. Qwen2.5's token salad.
>   `ai-docs/a_b_qwen3_findings.md`, gaps.md §5.
> - Whisper ASR panic (intermittent `decode_greedy` "GetElementType is not implemented") mitigated
>   not root-caused — `catch_unwind` stops it from killing the ASR thread. Unverified whether it
>   fully recovers or downgrades to a per-cycle WARN. gaps.md §5.

## Key files

- `ai-docs/gaps.md` — full backlog with effort/impact/status table (read first for "what's next")
- `ai-docs/a_b_dinov3_findings.md`, `ai-docs/a_b_qwen3_findings.md` — A/B evidence trails
- `scripts/ab_dinov3.sh`, `scripts/ab_qwen3.sh`, `scripts/ab_qwen3_longclips.sh`,
  `scripts/ab_binding_replay.sh` — A/B harnesses (isolated `--memory-dir` per arm, NDJSON metrics)
- `scripts/setup-hooks.sh` — installs `.git/hooks/pre-commit` (fmt + clippy, mirrors CI); rerun
  after a fresh clone or if `.git/hooks` is ever recreated

## Deferred backlog (do not raise unprompted)

CLAUDE.md's "Deferred architectural items" points here for trigger conditions. `gaps.md`'s priority
table covers everything not listed here.

| # | Source | Item | Trigger |
|---|--------|------|---------|
| A | ADR 0006 | Cold archive O(1) promotion index | `promote()` called AND cold > 500k records |
| C | ADR 0006 | Per-region sparse activation storage | Replay analysis shows episode vectors insufficient for diagnostics; no sparse-code consumer currently exists |
| E | ADR 0009 | Separate optimizer for output heads (independent LR schedule) | Conflicting gradient directions between output heads and k-WTA observed |
| F | ADR 0010 | Post-k-WTA prior injection | Pre-k-WTA injection shows poor `prior_cosine` uptake |
| I | ADR 0018 | Knowledge distillation compression | Grown regions exceed 2× baseline |
| J | ADR 0003 | Whisper-tiny → Whisper-base / WavLM-base (audio quality) | — (Tier 3, revisit opportunistically) |
| K | ADR 0003 | DINOv2-S → SigLIP-B (language-aligned visual embeddings, Phase 7+) | — (superseded in spirit by DINOv3-ViT-S, gaps.md §6; SigLIP2-ViT-B/16 tracked separately as §7) |
| L | ADR 0012 | LoRA fine-tuning of Qwen for episode-vector prefix | — (Tier 3) — **confirmed still open 2026-08-15 (gaps.md §37):** done 2026-06-25 for Qwen2.5 (ADR 0033), orphaned when Qwen2.5 was dropped (ADR 0036); `train_lora_qwen3.py` exists (committed 2026-06-27, same session as the abandoned KV-cache-decode half) but was never run — `qwen3_lm.onnx` is the corrected-export base model, not a LoRA merge |
| M | ADR 0012 | MLP projection (640→5×896) instead of linear | **RESOLVED 2026-08-15 (gaps.md §37) — not actually open.** Confirmed via code: `--linear-proj` defaults false, the MLP (ADR 0032) is the default production path, dimension-parameterized by `LM_HIDDEN` (now 1024/Qwen3) — works unmodified. Remove from any future copy of this table. |
| N | ADR 0014 | Visual/audio region grounding (captions, env sound labels) | — (Tier 3) |

Items B, D, G, H (language last-token pooling, full differentiable SAE, learned per-region gate,
activation-frequency pruning) are not in CLAUDE.md's current list — resolved/superseded/dropped at
some point after the 2026-06-14 scan; check `gaps.md` and ADR history if one becomes relevant.
