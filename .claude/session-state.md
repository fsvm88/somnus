# Session state — somnus

> **⏸️ SHUTDOWN (clean, user-requested), 2026-07-16.** Working tree clean, everything committed
> and pushed (commit below). **Next-session entry point: ADR 0038 prerequisite 3** (numerical
> validation plan — does block-local competition break growth/duty-cycle/binding assumptions) is
> the only prerequisite left open, but it's now gated behind a trigger that hasn't fired (see
> below) — **not** a live next step to just pick up. If nothing else is queued, the other live
> option is resuming the growth campaign (98.3% to its 10x target, see further below).
>
> **This session's work — §28/§29, `ai-docs/gaps.md`, resolving 2 of ADR 0038's 3 prerequisites:**
> user asked to work through ADR 0038's prerequisites in order.
> - **Prereq 1 (§28):** read `scripts/export_perception_graph.py` — the exported ONNX encoder graph
>   has the identical `O(d_input × n_units)` dense matmul as the `burn` path, no sparsity/gating
>   applied before scoring, so switching to ORT didn't change the scaling law. But live cost,
>   pulled from the already-existing `OrtRunEncoder`/`OrtRunBinding` Core metrics in the production
>   campaign's `memory/campaign_growth10x/metrics.jsonl` (~9,400-9,493 units/region), is small in
>   absolute terms (~1.28ms/call mean, 1,585 calls) and not even the dominant ORT cost — the
>   fixed-size `OrtRunBinding` graph (GRU+attention, independent of region size) costs 2-3× more per
>   call at this scale.
> - **Prereq 2 (§29):** checked `gaps.md` §16 (states the campaign's own goal is honeybee-order,
>   explicitly *not* mouse-order) and grepped all of `ai-docs/` for any roadmap item past the
>   current 10x campaign — none exists. Combined with prereq 1: hierarchical WTA isn't worth
>   building for the campaign this project is actually running, *right now*.
> - **Important correction caught by advisor before finalizing:** initial instinct was to mark the
>   ADR "closed — not needed." Advisor flagged this overstates the finding — "no roadmap names
>   mouse-order yet" is not the same claim as "honeybee-order is a decided ceiling," and
>   `brain-scale-comparison.md` frames the whole project around eventually closing the scale gap via
>   exactly this kind of algorithmic-efficiency technique. **Kept the ADR `proposed`/deferred
>   instead**, added an explicit, nameable trigger (growth campaign's target scale moves past
>   honeybee-order, or the encoder/binding cost balance shifts back toward the encoder mattering) —
>   same deferred-with-trigger pattern CLAUDE.md already uses for items A-N, not an open-ended
>   "revisit sometime." **Do not mark this ADR closed on the strength of §28/§29 alone** — that
>   requires someone affirmatively deciding honeybee-order is the ceiling, which hasn't happened.
> - No code changes (pure analysis using metrics that already existed) — consistent with the
>   metrics-discipline policy's "if a change doesn't touch any measurable data-handling operation,
>   say so explicitly" clause. `ai-docs/decisions/0038-sparse-encoder-scoring.md` updated in place
>   (status line, prerequisites section, "what would invalidate this" section) rather than leaving
>   the resolved prerequisites to drift from the ADR text.
>
> ---
>
> **⏸️ SHUTDOWN (clean, user-requested), 2026-07-15.** Working tree clean, everything committed
> and pushed (`5607eb7`). It was `proposed`/not-started as of the
> 2026-07-14 shutdown below; this session's metrics work incidentally produced real numbers for
> ADR 0038's first prerequisite ("benchmark the actual scoring cost on both `burn` and the
> ORT-exported path") — `kwta_score_matmul` measured at 2.15s/640 calls (~3.4ms/call) and
> `measure_mse` at 4.21s/128 calls in a live cycle at ~148k warm records / 9,493-unit regions (see
> §27 writeup below for how these were pulled from `memory/campaign_growth10x/metrics.jsonl`). Its
> other two prerequisites (confirm this matters at honeybee scale vs. next-order-of-magnitude;
> numerical validation that block-local competition doesn't break growth/duty-cycle/binding
> assumptions) are NOT yet addressed — this session did no new work toward them, just surfaced a
> number for the first prerequisite as a side effect of unrelated metrics analysis. Full ADR text:
> `ai-docs/decisions/0038-sparse-encoder-scoring.md`.
>
> **This session's actual work — §27, `ai-docs/gaps.md`:** the new `SleepWarmRebuild` Core metric
> (from the prior session's full-coverage `somnus-metrics` pass) surfaced that
> `SemanticStore::rebuild_from_recent` (`crates/somnus-memory/src/semantic.rs`) was costing 46-51s per
> sleep cycle at ~148k warm records — ~70% of total cycle wall-clock — because `hnsw_rs` has no
> delete API, forcing a full serial-insert rebuild every cycle once the warm cap is hit (every cycle,
> per the §21 fix). Fixed by switching to `hnsw_rs::parallel_insert_slice` (rayon, already a
> workspace dep) — same graph, same eviction semantics, spread across cores. **Verified on real
> production data**, not just a clean build: benchmarked the exact instrumented function against a
> copy of the campaign's actual cold archive (148,447 records), `git stash`/`pop` to flip old/new
> code on identical data — 47.6s → 2.85-2.94s, **16.7× speedup**. New
> `crates/somnus-memory/examples/bench_warm_rebuild.rs` (committed) is a reusable regression-check
> harness for this metric going forward, not a throwaway. Build/clippy/test clean both feature
> states. Committed + pushed as `5607eb7`. Full writeup: `ai-docs/gaps.md` §27.
>
> **Growth campaign status:** stopped (clean, no process running), session 78/130, cycle 10,
> region sizes `[9493]×5` — **98.3% to the 10x target** (~9,660 units/region), up from 94.7% at the
> 2026-07-14 shutdown (it was resumed and run sessions 76-78 as part of validating the
> full-coverage metrics pass live, then stopped cleanly again). `warm_len=148447`,
> `cold_len=155256`. Safe to resume anytime via `scripts/resume_growth_campaign.sh` — it will pick
> up the §27 rebuild fix automatically on its default rebuild-first behavior. Resuming it is the
> other live option alongside ADR 0038, but the user explicitly named ADR 0038 as next up.
>
> ---
>
> **⏸️ SHUTDOWN (clean, user-requested), 2026-07-14.** Working tree clean, everything committed
> (`25d8174` full-coverage somnus-metrics expansion, `b5b5849` base somnus-metrics + research note + ADR
> 0038 — both this session). No `somnus` process running, no stray pilot/scratch directories (all
> `memory/*_pilot*` dirs used during this session's validation runs were `rm -rf`'d after use).
> `memory/` (the production growth campaign) untouched this entire session — still safely stopped
> from before, resumable anytime via `scripts/resume_growth_campaign.sh`, session 75/131, 94.7% to
> the 10x target, per the shutdown record further below. **Next-session entry points:** ADR 0038
> (hierarchical k-WTA scoring, `proposed`, not started — now unblockable via a `--features
> fine-metrics` build against the real campaign instead of a synthetic benchmark) or resuming the
> growth campaign itself. Nothing else queued.
>
> **Last updated:** 2026-07-14 (final, full-coverage pass)
> **✅ somnus-metrics: full-coverage expansion complete, on top of the unified two-tier system (ADR
> 0039).** After the base conversion (below) landed, user asked to go further still — "make
> somnus-metrics cover all code... reach into every stupid step and evaluate it against an A/B run."
> Ran a 3-way parallel inventory (Explore agents over somnus-perception/somnus-memory/somnus-sensory+main.rs)
> of every remaining uninstrumented data-handling operation, then instrumented essentially all of
> it (operation-granularity only — per-element hot loops stay covered in aggregate by an enclosing
> phase metric, per the crate's own guardrail). `Metric` enum grew **24 → 67 variants**:
> `somnus-perception` +25 (binding forward, ORT run_encoder/run_binding, fused-replay's 3 internal
> sub-phases, ORT lifecycle, growth/prune, all REM self-probe functions, EMA sub-steps,
> refresh_autodiff_state, the async wake_worker::train_step), `somnus-memory` +10 (cold archive
> open/truncate/evict, semantic-store warm-load, recollection promote, snapshot save/load — closes
> the "zero coverage" gap noted in the original ADR 0039), `somnus-sensory` +1 (one aggregate
> `MediaFileDecode` metric per file), `somnus` binary +7 (action-decoder init/generate, gain-worker
> compute, ASR decode, audio resampling, language embedding, record_episode). Two new patterns
> documented in ADR 0039: **private-`_inner`-method split** (default for functions with early
> return/`?`/`bail!`) and **immediately-invoked-closure wrap** (used once, somnus-sensory's decode
> thread, which had no separate method to split into). Both preserve the original body
> byte-for-byte aside from the wrapper.
>
> **Verified incrementally** — rebuilt/retested after every single function conversion, not just
> at the end — `cargo build/clippy/test --release --workspace` clean in both feature states
> throughout. Final live pilot (2 sessions, isolated scratch memory dir, cleaned up after) recorded
> **40 distinct metrics simultaneously** in one small run, including both cross-thread ones
> (`wake_worker_train_step`, `gain_compute` — confirmed plain atomics are safe to write from any
> thread) firing correctly, and metrics tied to untriggered conditions (growth, `--action`,
> cold-archive eviction) correctly staying at zero rather than silently missing. Docs updated:
> `ai-docs/decisions/0039-unified-metrics-instrumentation.md` (new "Update" section + "Implementation
> patterns" section), `ai-docs/gaps.md` §26 (follow-up writeup + priority-table row bumped to
> Med–High impact). **Committed to git** — see commit history, nothing left uncommitted.
>
> ---
>
> **Prior writeup (base conversion, now extended by the above, kept for the discovery trail):**
> unified two-tier metrics instrumentation, fully implemented and converted: ADR 0039,
> `crates/somnus-metrics`. Extends the earlier same-session instrumentation work (see superseded
> writeup a few lines below, kept for the discovery trail) after the user asked to go further:
> convert *every* existing hand-rolled timer to one shared system ("one way of doing metrics")
> and treat the metrics this codebase has always reported as the always-on default tier.
>
> **Design:** two tiers sharing one `Registry`. **Core** (`timed_core!`/`counted_core!`) —
> unconditional, always recording, the "big item" tier (exactly what was always reported, now via
> one macro instead of copy-pasted `Instant::now()`/`.elapsed()` boilerplate). **Fine**
> (`timed!`/`counted!`) — gated behind `fine-metrics` (default off), textually a bare passthrough
> when disabled. `Metric` enum mirrors `RegionId::as_index()`'s dense-array-over-enum pattern (24
> variants). `fine-metrics` forwarded from `somnus-perception`/`somnus-memory`/`somnus-sensory`/
> `somnus-encoding`/`somnus`, same shape as the existing `mn10_env` precedent.
>
> **Full conversion done:** all 6 `somnus-perception` ingest phases (single `timed_core!` call each,
> replacing the earlier dual-write pilot pattern), all ~11 `somnus/src/sleep.rs` wall-clock phases (8
> via the macro block-wrap idiom; 3 structurally-entangled spans — the full NREM loop, the two
> non-contiguous REM sub-sections, the whole-cycle total — via a direct
> `somnus_metrics::METRICS.record()` call instead, a documented exception in ADR 0039 since their
> locals escape scope for use hundreds of lines later), all 3 `somnus-encoding` encoder timers
> (`encode_speech`/`encode_env`/video `encode`), and all 4 remaining `somnus/src/main.rs` sites
> (3× `MainPerceptionIngest`, 1× `MainWarmQuery`). **Found and fixed two real duplicate-measurement
> bugs** in the process: `main.rs` was independently re-timing the same `encode_speech`/
> `encode_env`/video-encode calls `somnus-encoding` already timed and returned via `encode_ms` — now
> logs that field directly instead of measuring twice. All pre-existing top-level JSON fields
> (`perc_kwta_us`, `nrem_wall_ms`, `speech_encode_total_ms`, etc.) still populated identically, so
> `campaign_status.py`/`eval_metrics.py` need no changes; new `"fine_metrics"` key carries the full
> registry snapshot.
>
> **Verified:** `cargo build/clippy/test --release --workspace` clean in both feature states
> (including 5 `somnus-metrics` unit tests, up from 3). Two isolated-scratch-dir pilot runs (cleaned
> up after) confirmed correct NDJSON output both before and after the full conversion; the second
> run cross-checked old-field vs. new-registry values across every converted site (all within
> expected truncation-order noise — e.g. `perc_kwta_us` 92,976 vs. registry-derived 93,004 — not a
> bug) and confirmed `kwta_score_matmul` dominates `kwta_select_topk` by ~124–165× at tiny baseline
> scale, live evidence bearing on ADR 0038's still-open question (not yet at the campaign's actual
> ~9,660-unit scale).
>
> **Deliberately not done:** `somnus-memory` has zero converted call sites — it never had hand-rolled
> timers (no I/O path was ever individually timed), so there's nothing to convert there; adding
> instrumentation is future opportunistic work, not part of this pass. Full design writeup:
> `ai-docs/decisions/0039-unified-metrics-instrumentation.md`. gaps.md §26 has the summary; plan
> file `/home/falcon/.claude/plans/wild-imagining-thompson.md` ("Plan A") has the blow-by-blow.
>
> **ADR 0038 (hierarchical k-WTA encoder scoring) remains status `proposed`, implementation not
> started** — unaffected by this instrumentation work; still the queued next architectural
> question. Next natural step (user-driven): run a `fine-metrics` build against the real growth
> campaign to get `kwta_score_matmul` numbers at actual ~9,660-unit scale, which is what ADR 0038's
> Phase 0 needs.
>
> **This entire session's work — research note, ADR 0038, ADR 0039 + `somnus-metrics` crate + full
> conversion — is committed to git as of this writeup's timestamp** (see commit history; nothing
> left uncommitted).
>
> ---
>
> **Superseded pilot-stage writeup (kept for the discovery trail, not the current state):** the
> instrumentation work started as a smaller pilot — new crate, dual-write pattern alongside the
> existing `perf_*_us` fields, only `somnus-perception`'s ingest phases touched. The user then asked
> to finish the job (convert everything, one system) — see the fully-converted state described
> above, which supersedes every claim below about what's "not yet done."
>
> **✅ Research note written: k-WTA encoder-side scoring-cost scaling.** The queued-next item from
> the §25 shutdown writeup below — user chose to start the research note this session rather than
> resume the campaign. Added a new section to `ai-docs/research/sparse-coding.md`
> ("Encoder-side scaling: why sparsity doesn't save the O(input_dim × n_units) scoring cost")
> confirming the scoring matmul in `KWtaLayer::forward` (`crates/somnus-perception/src/region.rs`) is
> inherently linear in `n_units` regardless of `k_active`/`scaled_k` (output sparsity doesn't help —
> every candidate unit must be scored before top-k can pick winners), and surveying three candidate
> mechanisms: local/blocked connectivity (MoE-style routing), hierarchical/coarse-to-fine WTA
> (two-stage scoring, maybe reusable HNSW-style indexing), approximate top-k sampling. Cross-
> referenced from `gaps.md` §25. **Docs-only, no code changes, nothing to build/test.** Three
> prerequisites identified before any implementation: (1) benchmark the actual scoring cost on both
> `burn` and the ORT-exported path (no isolated measurement exists yet), (2) confirm this matters at
> honeybee scale vs. being a next-order-of-magnitude concern, (3) scope as an ADR (touches growth/
> duty-cycle/binding assumptions). **Not committed yet this session** — `ai-docs/gaps.md` and
> `ai-docs/research/sparse-coding.md` both modified, working tree otherwise clean.
>
> **✅ ADR 0038 written 2026-07-14 (same session, follow-on to the research note above):**
> `ai-docs/decisions/0038-sparse-encoder-scoring.md`, status `proposed`. Picks **hierarchical/
> coarse-to-fine WTA** as the target mechanism for the encoder scoring-cost problem — over MoE-style
> learned routing (better biological analogue but a bad router reintroduces the dead-unit problem)
> and approximate top-k sampling (cheapest but breaks the exact-top-k guarantee the growth
> mechanism's residual-correlation logic depends on). Reasoning: preserves exact top-k within
> selected blocks, degrades gracefully if the coarse scorer is imprecise, and may be able to reuse
> existing HNSW machinery (`crates/somnus-memory`) for the coarse stage instead of a bespoke gating
> network. **Implementation explicitly not started** — same three prerequisites as the research note
> (benchmark actual scoring cost on both `burn` and ORT-exported paths; confirm this matters at
> honeybee scale vs. being a next-order-of-magnitude concern; numerical validation that block-local
> competition doesn't break growth/duty-cycle/binding assumptions), now tracked as the ADR's own
> prerequisite list. Cross-referenced from `gaps.md` §25 and `sparse-coding.md`; added to
> `decisions/README.md` index. Docs-only, no code changes.
>
> **Growth campaign status unchanged from the shutdown below** — still stopped at session 75/131,
> 94.7% to target, safe to resume anytime via `scripts/resume_growth_campaign.sh`. Not touched this
> session.
> **⏸️ STOPPED (clean, user-requested shutdown).** Campaign was at session 75/131, cycle 6, region
> sizes 9,151/9,660 ≈ **94.7%** — very close to the 10x target. Stopped via
> `scripts/stop_graceful.sh`, confirmed clean exit, no data lost. `CAMPAIGN_DIR=memory/campaign_growth10x`,
> `CLIPS`=full 145-file h3h3+shorts+podcasts corpus. **No known open bugs; safe to resume anytime**
> via `scripts/resume_growth_campaign.sh` (self-contained, computes the correct resume point from
> the campaign's own on-disk state).
>
> **✅ gaps.md §25 — sparse-aware binding projection, deployed 2026-07-13, with an honest
> correction added 2026-07-14.** Raised while discussing what would genuinely help scale toward
> mouse-order brain size rather than just running more GPU cycles: `BindingArea::forward`
> (`crates/somnus-perception/src/binding.rs`) was materializing a dense `[1, n_units]` vector from each
> region's ~98%-zero k-WTA activation and running a full dense matmul on it. Replaced with a sparse
> `select(0, active_indices)` gather + small matmul — numerically identical (new test
> `sparse_projection_matches_dense_reference` proves it), all pre-existing `BindingArea` tests pass
> unchanged, **measured 11.4× speedup** in an isolated benchmark at the campaign's actual region
> size (n=8,655, k=169). Deployed live at session 72 (stop/rebuild/resume, same safe pattern as
> every other restart this campaign).
>
> **Correction (2026-07-14):** checked `perception_total_ms`/`nrem_wall_ms` before vs. after
> deploying — no visible change. Traced why: `perc_bind_us` (the profiling metric that motivated
> this fix) times the whole `try_bind` call, which — since ORT is active past the §22 crossover —
> dispatches to `ort.run_binding` (ONNX/MIGraphX), not `burn`'s `BindingArea::forward` at all. The
> fix is real and correctly deployed, but it's currently **dormant** on this specific campaign: the
> only call sites that hit it (non-ORT `try_bind` branch, ORT-failure fallback in `settle_episode`,
> the `binding_only` validation helper) don't fire while ORT is healthy. It'll matter for
> pre-crossover campaigns, CPU-only/no-MIGraphX hardware, and as an ORT-failure safety net — keep
> it deployed (strictly better, costs nothing when dormant) — but don't attribute any of this
> campaign's remaining wall-clock to it. Separately noticed while tracing this: no training function
> (`replay_step*`, `wake_reconstruct_step*`) calls `BindingArea::forward` either, so its weights
> (`region_projs`, `w_q`/`w_k`/`w_v`) currently receive no gradient updates from any path in this
> codebase — un-investigated, flagged for later, not fixed here. Full writeup: `ai-docs/gaps.md` §25.
>
> **Queued next (not started, by design):** the harder, deeper lever from the same discussion — the
> k-WTA *encoder* itself is inherently O(input_dim × n_units) since every candidate unit must be
> scored before top-k can pick winners, and can't be sparsified the way the binding projection was.
> Getting real headroom toward mouse-order scale (not just bigger `N_UNITS`) likely needs local/
> blocked connectivity or hierarchical/approximate WTA — a genuine research question, to be written
> up as a research note before any implementation. User confirmed this stays queued, not started.
>
> **✅ gaps.md §24 FULLY FIXED AND CONFIRMED LIVE (2026-07-13).** Two independent bugs, both in the
> ORT call path, found back to back:
> 1. **Binding-side:** `to_dense_n(N_UNITS)` at 5 call sites in `crates/somnus-perception/src/lib.rs`
>    hardcoded the baseline `N_UNITS=1024` instead of each region's actual grown size;
>    `onnx_session.rs`'s `run_binding` independently hardcoded `N_UNITS` for its ONNX tensor shapes.
>    Fixed: both now derive size from the actual data (`SparseActivation.n_units`, each vector's
>    `.len()`). Commit `8718099`.
> 2. **Encoder-side** (found resuming the campaign to verify fix #1 live): `onnx_session.rs`'s
>    `ENC_DIMS` hardcoded the language encoder input at 896, stale since ADR 0036 raised
>    `LANG_HIDDEN` to 1024 — same drift class as §22's already-fixed Python-side bug, just never
>    fixed on this Rust caller. Fixed: `ENC_DIMS` now derives from live `ENV_DIM`/`LANG_HIDDEN`
>    constants. Commit `debd6c4`.
>
> Both fixes have regression tests (`crates/somnus-perception/tests/ort_grown_region.rs`) confirmed to
> fail pre-fix (exact errors seen in the campaign log) and pass post-fix.
>
> **Live confirmation (sessions 60-61, both fixes deployed):** **zero** `ORT encoder failed` and
> **zero** `ORT binding failed` occurrences across ~8,000 log lines. `perception_total_ms` mean:
> 2,158ms (session 60), 1,457ms (session 61) — vs. 250,099ms fully-broken (session 57) and
> ~6,900-7,200ms pre-ORT-crossover baseline (sessions 50-52). **Real confirmed speedup: ~3-4× over
> CPU baseline.** This supersedes and retracts ADR 0037's original "~2.1×/2.4×" claim (measured
> entirely on the broken path). §24 and ADR 0037 are both fully closed out — no further action
> pending on this item. Full writeup: `ai-docs/gaps.md` §24, ADR 0037.
>
> **To resume (if stopped again):** `scripts/resume_growth_campaign.sh` — self-contained, reads
> `SESSION_START`/`SESSIONS`/`CYCLES`/`CLIPS`/etc. from the campaign's own `params.txt` +
> `metrics.jsonl` (not from this file or any session-specific state), computes the correct resume
> point automatically, rebuilds `somnus` first by default (`SKIP_BUILD=1` to skip), refuses to
> double-launch. `DRY_RUN=1` to preview. **To stop:** `scripts/stop_graceful.sh` (`--dry-run` to
> preview, `--wait N` to change the SIGTERM timeout — state is snapshotted every cycle, so this
> loses at most a few seconds).
>
> **What's live and correct:** §17/§18/§19 (K_ACTIVE scaling, optimizer-reset scoping,
> novelty-proportional batch), §21 (fixed a warm-store HNSW dump leak — was leaking ~24-26MB/cycle
> unboundedly, also silently broke cross-restart index reload), §22/ADR 0037's auto-switch
> mechanism plus both §24 fixes (confirmed above). `memory/` was wiped completely at campaign start
> per the user's explicit call (all production state + old A/B/pilot scratch dirs, ~17GB,
> gitignored/untracked, nothing lost that wasn't reproducible) — regrowing from scratch.
>
> **Cold-archive note (2026-07-13, user question):** `max_cold_records_effective` has been `0`
> (unlimited) the whole campaign — `sysinfo`'s disk lookup can't match `memory/`'s mount point
> (likely the btrfs subvolume under `/mainpool`), so it falls back to no cap per ADR 0024's
> documented behavior (not a bug). Cold archive (`memory/episodes.cold`) is growing correctly and
> steadily every cycle regardless (94MB / 141,765 records as of session 67, exactly matching
> `cold_len`) — it's an append-only full ledger by design (every episode goes to both cold and warm
> every cycle), not a warm-eviction target, so it was never expected to jump discontinuously.
>
> **Restart 1** (~cycle 486, gaps.md §21): `SemanticStore::dump()` leaked a full HNSW index dump
> every sleep cycle forever (461 orphaned pairs, 11G) because `load_warm()`'s reload path
> permanently disables `hnsw_rs`'s in-place overwrite; fixed by clearing stale dumps before
> writing, which also restores the plain-name path `load_warm()` needs to find anything on
> restart. Cleaned up the leak (`memory/warm/` 11G → 192M), resumed `SESSION_START=50 SESSIONS=82`,
> metrics/logs continuous across the restart (486→487, no gap).
>
> **Restart 2** (~cycle 542, gaps.md §22/ADR 0037): user asked whether `onnx=off` had a runtime
> cost — pulling real numbers from `metrics.jsonl` showed `perception_total_ms` climbing steadily
> (9.1s→16.1s as regions grew 1,986→6,043 units), already over half the cycle's wall-clock, and
> past ADR 0023's documented CPU/GPU crossover (~1,230 units/region). Implemented auto-enable
> (`should_use_ort_inference`, unit-tested) instead of requiring the old `--onnx-models-dir` flag.
> This surfaced two pre-existing, never-before-exercised bugs in `scripts/export_perception_graph.py`
> (stale hardcoded `LANG_HIDDEN=896` since ADR 0036 made it 1024, and hardcoded `N_UNITS`/`K_ACTIVE`
> that could never survive region growth) — fixed by deriving all dims from the actual exported
> weight shapes (`infer_dims()`) instead of a second hand-maintained copy. Resumed
> `SESSION_START=56 SESSIONS=76`; log confirmed `ORT binding session active`. **Then found §24's two
> bugs (above) — now both fixed and confirmed live.** User also proposed eliminating the Python
> export step entirely (native Rust ONNX graph construction) — sound long-term direction, scoped as
> gaps.md §23, deliberately not started (no mature Rust ONNX-graph-builder crate exists; real
> rewrite effort, not a fast-follow).
>
> **Restart 3** (session 57→58, gaps.md §24 bug #1 fix): stopped cleanly to deploy the binding-side
> N_UNITS fix. Resumed at session 58, confirmed zero `ORT binding failed` but found bug #2
> (`ORT encoder failed` on `language_enc`, 100% of calls).
>
> **Restart 4** (session 59→60, gaps.md §24 bug #2 fix): stopped cleanly to deploy the encoder-side
> LANG_HIDDEN fix. Resumed at session 60 — both bugs confirmed fixed, campaign running clean since.
>
> **Restart 5** (session 71→72, gaps.md §25): stopped cleanly to deploy the sparse binding-projection
> fix. Resumed at session 72 — confirmed clean (zero ORT fallback), though see the §25 correction
> above re: this fix's actual live impact.
>
> **Shutdown** (session 75, this writeup): user asked to save everything and shut down cleanly.
> Docs brought current (this file, `gaps.md` §25 correction), all code changes already committed,
> campaign stopped via `scripts/stop_graceful.sh` — confirmed clean exit.
>
> **Next-session action:** none queued as a bug — campaign is healthy, 94.7% to target, safe to
> resume anytime with `scripts/resume_growth_campaign.sh`. The one open design thread is the
> queued-next encoder-sparsity research note (see above) — user-driven, not started.
>
> Full backstory: `ai-docs/gaps.md` §16/§19/§20/§21/§22/§23/§24/§25, ADR 0037.

## Historical (superseded by the in-progress campaign above, kept for context)

> **Last updated:** 2026-07-11
> **Branch:** `main`. Committed (through 2026-07-09): change-gated video encoder (`9bba42b`),
> SVD null-result (`b410ad7`), train_campaign.sh bugfix + replay-ratio A/B (`d5ce2ab`), N=4
> replay-ratio default (`f0fa9c6`), items queued (`709442b`), DER++ auto-λ implementation
> (`f2fd185`, gaps.md §3), DER++ auto-λ A/B reject (`72d1503`), EfficientAT-mn10 implementation +
> A/B result (gaps.md §9), M2D-CLAP scoped and not pursued (gaps.md §8, docs-only).
> **Not yet committed (2026-07-11 session):** `scripts/download_podcasts.sh` (new — pulls diverse
> podcast episodes into `clips/podcasts/`, 56 episodes/5.6GB across 10 shows), growth-pilot 2
> results (expanded-corpus characterization, `memory/growth_pilot_expanded/` — scratch, untracked),
> and a documentation audit/correction pass across `architecture.md`, `gaps.md`,
> `growth-mechanisms.md`, `roadmap.md`, `README.md`, and this file (stale open-questions,
> priority-table numbering drift, a lost deferred-items trigger table, missing Phase 14 section).
> **Status: gaps.md §16 (growth-scaling) CHARACTERIZED — three 75-cycle pilots (10-clip loop, then
> 145-clip diverse corpus, then a post-fix rerun of the same 145-clip corpus) all show gate-limited
> growth cadence, no memorization-plateau risk.**
> **Also done (2026-07-12, not yet committed): three architectural fixes found during pre-campaign
> review, gaps.md §17/§18/§19 — K_ACTIVE now scales with region growth, optimizer reset scoped to
> the grown region only (not the whole model), and growth batch size is now novelty-proportional
> instead of a flat constant. Validated with a third growth pilot (`memory/growth_pilot_v2/`,
> untracked) — mechanism confirmed working (k_active tracked region size correctly; batches varied
> 17–77 with cadence unchanged from pilot 1/2). But that pilot revealed the average batch (35.6/
> event) is well below the old flat 64, because most triggers in this corpus land at moderate, not
> severe, novelty — so **the full 10x campaign is now re-estimated at ~9.3–11.5h (was ~3.9h),
> ~2.4-3x longer than the pre-§19 number.** Full writeup: `ai-docs/gaps.md` §16 pilot 3, §19.**

## Just completed (2026-07-11) — expanded-corpus growth pilot + podcast corpus + doc audit

Continuation of the growth-scaling pilot below: user wanted the memorization-plateau caveat closed
before committing to the full 10x campaign, which meant a more diverse corpus than pilot 1's
10-clip loop. Added `scripts/download_podcasts.sh` (new, modeled on `download_h3h3.sh`) — pulls
episodes from 10 distinct podcast shows (Lex Fridman, Joe Rogan, Huberman Lab, Tim Ferriss, Diary
of a CEO, Armchair Expert, Conan O'Brien, This American Life, The Daily, Freakonomics Radio),
10–90 min duration filter, `N_PER`/`M_SEARCH` configurable. Ran twice (1 episode/show, then 10x
more): landed 56 episodes / 5.6GB in `clips/podcasts/` (Lex Fridman/Huberman Lab/Tim Ferriss
under-represented — their typical episode length exceeds the 90-min filter ceiling on most search
candidates, not a download failure).

Ran pilot 2: identical protocol to pilot 1 (`SESSIONS=15 CYCLES=5 CAPACITY=8192`, 75 cycles) against
the combined 145-clip corpus (h3h3 + shorts + podcasts), isolated `memory/growth_pilot_expanded/`.
**Result: no plateau — cadence and quality trend indistinguishable from pilot 1** (16 growth
events, same gate-limited cadence, `per_region_mse` improved ~0.0135→~0.0039 first-10-vs-last-10
cycles, final region sizes uniformly `[2048]×5`). Confirms growth is gate-limited (bounded by the
3-cycle cooldown), not corpus/signal-limited — the memorization-plateau risk flagged after pilot 1
did not materialize. Full numbers: `ai-docs/gaps.md` §16, `ai-docs/research/growth-mechanisms.md`
"Empirical validation" section (both written up this session).

Then did a full documentation audit (user request: "recheck all docs and make sure they are
correct") — delegated a read-only staleness sweep to a subagent, applied its findings directly:
- `architecture.md`: "Growth mechanism" and "Compression strategy" open questions were stale (both
  long since resolved — ADR 0024 §2c and INT8+PQ respectively); struck through with references.
  "Sleep-cycle time budget" narrowed (ADR 0024's S1–S4 controllers resolved most of what it asked).
  Implementation-state paragraph updated with everything since 2026-07-05.
- `gaps.md`: priority table's `#` column had drifted from the actual `##` section numbers since
  around item 7 (table said "❌7 DER++" but that's actually §3; "10 SigLIP2" is actually §7; etc.) —
  renumbered to match, added missing §10/§12 rows, added new §16 row for the growth pilots.
- `.claude/session-state.md` (this file): the "Deferred backlog" section had decayed into a
  circular pointer to `CLAUDE.md`, which itself points back here — the actual trigger-condition
  table had been silently pruned in an earlier edit. Restored from git history (commit `03eb9b5`).
  Flagged L/M (LoRA fine-tuning, MLP projection) as unverified against Qwen3 — they were marked
  done for Qwen2.5, which was fully removed by ADR 0036; not re-checked here.
- `roadmap.md`: had no Phase 14 section despite `architecture.md`/`gaps.md` referencing "Phase 14
  (GPU unification)" as complete. Added it (ADR 0023, GPU unification via ORT/MIGraphX).
- `README.md`: "Living archives" ADR callout list was missing ADR 0024 (large, multi-phase) and
  ADR 0036 (Qwen3 adoption) — added both.

Nothing here required code changes — pure documentation correction plus the two new artifacts
(podcast corpus, pilot 2 metrics). Not yet committed.

## Just completed (2026-07-09, later) — growth-rate pilot campaign (10x growable-core scaling)

User asked to scale somnus's own growable core (region k-WTA units + GRU + binding, currently
~5,400 units total, ~480,000× smaller than the human brain / roughly fruit-fly order of
magnitude) 10x, toward honeybee-order scale (~54,000 units) rather than mouse (~71M neurons,
a ~400x jump — corrected an earlier miscalibration in-conversation). Explicitly *not* the
frozen-encoder scale-up (Qwen3/Whisper/DINOv3/CLAP) — those are "easy, do later."

Growth mechanism (cascade-correlation-style, `grow_region_seeded`, ADR 0024 §2c
`within_session_gist_rate` gate, 3-cycle cooldown, `GROWTH_BATCH=64`) was already fully
implemented and live — no new code needed. Ran a 75-cycle characterization pilot
(`SESSIONS=15 CYCLES=5 CAPACITY=8192`, h3h3/shorts corpus, isolated `memory/growth_pilot/`,
not the main `memory/`) purely to measure real growth cadence, since nobody had run enough
cycles to know it empirically.

**Result:** 16 growth events in 75 cycles (~1 event per 4.7 cycles, close to the 3-cycle
cooldown floor — growth is essentially gate-limited, not signal-limited). Final region sizes
`[2048, 1984, 1984, 1984, 1984]` (region 0 grown 2×, others ~1.94× from the n=1024 baseline).
No plateau observed over the full pilot — one quiet session (s9) was immediately followed by
resumed growth at s10c1. `per_region_mse` *improved* alongside growth (e.g. region 0:
0.127→0.007–0.017), no quality regression from growing. Avg 21.8s/cycle wall-clock (dominated
by short pressure-triggered ~120-episode cycles after the first buffer-fill cycle); 1.7GB disk
for the whole pilot (816GB free on the volume — no capacity concern).

**Extrapolation to 10x (~9,660 units/region, ~135 growth events needed):** at the observed
rate, ~633 cycles ≈ **~3.9 hours wall-clock** for a full campaign — far more tractable than
the pessimistic first estimate (hundreds of cycles was right, but wall-clock per cycle is much
cheaper than the growth-mechanisms.md research note's aspirational cascade-correlation
estimate, since the actual implementation is a weight-tensor resize, not a candidate-training
loop). Caveats not yet tested at this scale: (1) whether the 10-clip looping corpus causes a
memorization plateau past 75 cycles (no sign of it yet, but untested); (2) whether per-cycle
compute cost grows non-trivially as region matrices get ~10× larger (linear in n_units, likely
modest but unverified).

**Not yet done:** the actual full-scale campaign to reach 10x. Discussed with user but not
launched — next step is user-driven (scope: same corpus repeated vs. expanded corpus to
pressure-test the memorization-plateau risk before committing ~4h of unattended compute).
`memory/growth_pilot/` is scratch pilot output, not committed/tracked.

## Just completed (2026-07-09) — M2D-CLAP (gaps.md item 8) — ⏸️ NOT PURSUED, no code written

User asked to try a deferred item; picked M2D-CLAP (item 8) — the more credible audio-encoder
upgrade candidate vs. the already-rejected mn10. Before implementing, ran feasibility research
(no code): weights are available with no gate, but (a) no ONNX export precedent exists anywhere
for M2D or M2D-CLAP — real risk of hitting a dead end at export, driven by dynamic
positional-embedding interpolation and an `nnAudio` STFT-based mel frontend (the same STFT-op
fragility class mn10 was chosen specifically to avoid), and (b) the license is NTT's custom
non-commercial "Evaluation" agreement, not MIT/Apache like everything else in the tree.

Weighed against that risk: the paper's +3 mAP AudioSet-tagging claim is a modest number on a
different task than somnus's actual metric (`per_region_mse` through somnus's own pipeline), and
item 9's A/B is a concrete precedent in this exact slot that paper-benchmark quality and
somnus-pipeline quality diverge (mn10 had no quality claim at all and still regressed MSE +155.6%
while a benchmark-adjacent recall metric moved the other way). User's call: **not worth the
implementation/export/license risk for the expected effect size. Documented in `gaps.md` §8,
no code written, nothing to revert.**

## Just completed (2026-07-09) — EfficientAT mn10 (gaps.md item 9) — ❌ REJECTED

Resumed the A/B campaign left mid-run at the end of the previous session (implementation was
already complete and smoke-tested). Ran clean (`rm -rf memory/ab_efficientat_mn10` first, then
`bash scripts/ab_efficientat_mn10.sh`, 5 sessions × 2 cycles per arm, fresh memory both arms,
h3h3/shorts corpus).

**Result: FAIL.** Verdict is based on the mean MSE of the four regions with identical
architecture in both arms (speech/visual/symbolic — region-1/env excluded, see script header
for why CLAP vs. mn10 env-MSE isn't a valid cross-arm comparison): control (CLAP-HTSAT) avg
0.011966 vs. treatment (mn10) avg 0.030588, **+155.6%, worse**, regressed in 9/10 cycles, no
reversals. `warm_recall_l2`/`gist_recall_rate` moved in mn10's favor but don't override a 2.5×
MSE regression on the shared representation — likely a scale-distortion artifact rather than a
genuine improvement.

**Decision:** reject: default env encoder stays CLAP-HTSAT. Implementation kept as an opt-in,
off-by-default Cargo feature (`mn10_env`) rather than deleted — it's clean and parity-tested,
and may be worth revisiting for a different reason (e.g. deployment-size constraints) later.
Full writeup: `ai-docs/gaps.md` §9. Build/clippy/test pass clean both with and without the
feature; committed together with the implementation in one shot (was never committed before
since the A/B hadn't concluded).

## Just completed (2026-07-12) — three growth-mechanism architectural fixes (gaps.md §17/§18/§19)

Found and fixed in order, while reviewing the growth mechanism ahead of the full 10x campaign.
All implemented, tested, and documented this session; **not yet committed**.

- **`gaps.md` §17 — K_ACTIVE scaling with region growth.** `K_ACTIVE=20` was a hardcoded const
  never adjusted by `grow_region`/`grow_region_seeded`/`prune_region` — sparsity ratio would have
  fallen ~10x (2%→0.2%) over the full 10x campaign. Fixed: `KWtaLayer` gained a per-layer
  `k_active` field, new `region::scaled_k(n_units)` preserves the baseline ratio linearly, wired
  into all three growth/prune call sites and all k-WTA selection code paths. Also fixed four
  ORT-bootstrap placeholder call sites and `binding_only` that hardcoded `N_UNITS`/`K_ACTIVE`
  instead of the layer's live size (latent bugs for an already-grown region after snapshot
  reload). New tests: `scaled_k_preserves_baseline_ratio`, `kwta_respects_grown_k_active`.
- **`gaps.md` §18 — scope optimizer reset to the grown region only.** Every growth/prune event
  was calling `self.optimizer = AdamWConfig::new().init(...)`, wiping Adam momentum for the
  *entire* model, not just the touched region. Investigated burn-optim's source
  (`OptimizerAdaptor`, `HashMap<ParamId, AdaptorRecord>`) before changing anything: state is
  already keyed by `ParamId`, and the resized tensors already get fresh `ParamId::new()` — so the
  full reset was pure waste. Fix: deleted the reset line from all three call sites; nothing else
  needed to change. Confirmed unrelated to (and doesn't regress) the separate
  `refresh_autodiff_state` mechanism, which resets the optimizer for a different, legitimate
  reason (NREM graph-accumulation slowdown) and is untouched.
- **`gaps.md` §19 — novelty-proportional growth batch size** (user's proposal, judged sound).
  Flat `GROWTH_BATCH=64` replaced with `GROWTH_BATCH_MIN=16`/`GROWTH_BATCH_MAX=128` and three
  pure functions (`gist_novelty`, `mse_novelty`, `growth_batch_size`) that scale the batch by how
  far the trigger signal missed its threshold — a region deep in recall failure now grows more
  per event than one that barely tripped the gate, instead of every triggered region getting the
  same fixed lump regardless of severity. New tests: `growth_batch_size_spans_min_to_max`,
  `growth_batch_size_clamps_out_of_range_novelty`, `gist_novelty_zero_at_threshold_...`,
  `mse_novelty_zero_at_threshold_...`, `worse_recall_failure_grows_a_bigger_batch`.

All three: build/clippy/fmt clean, full `somnus`/`somnus-perception` test suite passes (including
`phase5_smoke.rs`'s `grow_then_bind_produces_finite_episode`, `grow_then_replay_finite`,
`prune_region_reduces_grown_units`). Full writeups: `ai-docs/gaps.md` §17–19.

## Just completed (2026-07-12, later) — growth pilot 3/"v2": validated §17/§18/§19 live, recalibrated 10x campaign estimate

Reran the exact pilot 2 protocol (`SESSIONS=15 CYCLES=5 CAPACITY=8192`, same 145-clip h3h3+shorts+
podcasts corpus) against the post-fix build, fresh isolated `memory/growth_pilot_v2/` (untracked).
75 cycles, 39m49s wall-clock, 5 benign MIGraphX-exit-race segfaults (same known issue, auto-
recovered by `train_campaign.sh` as always).

**Mechanism confirmed working correctly:** 14 growth events, same 4–7 cycle gate-limited cadence
as pilot 1/2; `per_region_mse` improved 0.0136→0.0048 (comparable to pilot 2's 0.0135→0.0039);
`k_active` tracked region growth as designed (final `region_sizes=[1523]×5`, `scaled_k(1523)=30`
vs. baseline `scaled_k(1024)=20`, ratio preserved); batch sizes varied sensibly with novelty
(17–77 across observed novelty 0.01–0.54).

**But: this surfaced an important recalibration.** Mean batch across the 14 events was 35.6 —
nearly half the old flat 64 — because most triggers in this corpus land at moderate, not severe,
novelty. Total capacity added over 75 cycles (499 units/region) was noticeably less than the old
flat-batch mechanism would have produced for the same 14 events (896 units/region). Extrapolating
this pilot's observed mean batch (35.6/event) and cadence (1/5.36 cycles) to the 10x target
(~9,660 units/region) now projects **~242 growth events, ~1,298 cycles, ~9.3–11.5h wall-clock** —
roughly **2.4-3x longer than the pre-§19 estimate of ~3.9h/~633 cycles.** This isn't a regression;
it's §19 correctly making growth proportional to actual severity, in a setup where severity is
usually moderate. Full numbers: `ai-docs/gaps.md` §16 (pilot 3) and §19 (validation note).

**Not yet committed:** all of §17/§18/§19's code changes plus this pilot's docs updates.

**Next step is user-driven:** commit these changes, then either accept the longer (~9.3–11.5h)
campaign time and launch it, or revisit §19's `GROWTH_BATCH_MAX`/curve shape first if ~3-4h is a
hard time budget.

## Historical: EfficientAT mn10 implementation notes (2026-07-08, now committed and A/B'd — see above)

**What was implemented:**
- `crates/somnus-encoding/src/preprocess/mel.rs`: new `Mn10Mel` — EfficientAT's `AugmentMelSTFT`
  mel frontend (pre-emphasis, centered/padded Hann window, Kaldi-style mel filterbank —
  genuinely different triangle construction from the existing HTK/Slaney code, not a
  parameter variant, see `kaldi_mel_filterbank_f64`'s doc comment) ported to Rust and
  parity-tested against the PyTorch reference (`crates/somnus-encoding/tests/mel_parity.rs`,
  `mn10_zeros_parity`/`mn10_sine440_parity`, both pass at 1e-4 tolerance). Fixture files in
  `crates/somnus-encoding/tests/fixtures/mn10_*` are new, untracked, need `git add`.
- `scripts/export_efficientat.py`: exports the MobileNetV3 backbone only (no mel frontend —
  deliberately, baking `AugmentMelSTFT` in works numerically but reintroduces an STFT op,
  defeating mn10's whole MIGraphX-compat rationale; see script docstring). Verified
  numerically against PyTorch (rel L2 ~9e-7) and confirmed zero STFT/DFT ops in the graph.
  Needs its own venv (`scripts/.venv-efficientat/`, gitignored, torch 2.13/torchvision 0.28 —
  incompatible with `scripts/.venv`'s pinned torch 2.11.0, so it's isolated rather than
  risking the shared venv). Exported model already on disk:
  `models/efficientat_mn10_encoder{,_fp16}.onnx` (gitignored, not committed, regenerable via
  `scripts/.venv-efficientat/bin/python scripts/export_efficientat.py &&
  scripts/.venv-efficientat/bin/python scripts/convert_fp16.py`).
- New Cargo feature `mn10_env` (in `somnus-encoding`, `somnus-perception`, forwarded by `somnus`):
  swaps `REGION_DIMS[1]` 512→960, `AudioEncoder`'s env session/mel/sample-rate (CLAP 48kHz
  vs mn10 32kHz), and all the cross-modal-concat offset math in
  `crates/somnus-perception/src/lib.rs` (now derived from `ENV_DIM`/`CM_OFF_*` consts instead of
  hardcoded 512/896/1280 literals). Builds, tests, and clippy pass clean **both** with and
  without the feature. End-to-end smoke-tested manually (single session, small capacity,
  `--features mn10_env` binary) — ran a full sleep cycle with `env embedding dim=960`, no
  crashes, log confirms correct dim.
- `scripts/train_campaign.sh` gained an additive `SKIP_BUILD` env var (default `0`, no effect
  on any other script) so the A/B script can build each arm itself (with/without
  `--features mn10_env`) without the campaign script's own unconditional `cargo build`
  clobbering it.
- `scripts/ab_efficientat_mn10.sh`: the A/B harness. **Important asymmetry vs. every prior
  A/B in this repo:** region-1 (env) MSE is NOT a valid comparison between arms — CLAP and
  mn10 have different target distributions (contrastive/unit-norm vs. raw pooled-conv
  log-mel-scale), so `per_region_mse[1]` moving is a scale artifact, not a quality signal.
  The verdict is computed from the mean of the OTHER four regions' MSE (indices 0,2,3,4 —
  identical architecture both arms, genuinely comparable) plus `warm_recall_l2` /
  `gist_recall_rate`. Region-1 MSE is still printed for reference only.

## Just completed (2026-07-08)

Ran the A/B that gaps.md §3 flagged as its own next step: `--der-auto-lambda` (EWMA
1/variance weighting) vs. default unweighted (λ=1) sum in `fused_replay_wake_step_batch`. New
script `scripts/ab_der_auto_lambda.sh` (mirrors `ab_replay_ratio.sh`: 5 sessions × 2 cycles, fresh
memory per arm, h3h3/shorts corpus, control then treatment). **Result: FAIL.** `mean_output_mse`
control avg 0.0487 vs. treatment avg 0.0847 (**+73.8%**, worse), regressed in all 10/10 cycles
(range +16% to +937%), no reversals — decisive, not noise. Likely cause: EWMA variance estimate
too small/noisy early, causing large `1/variance` weight spikes that destabilize training rather
than damp a noisy term. Decision: do not adopt; `--der-auto-lambda` remains opt-in, defaults to
`false`, no code changes needed. Full writeup: `ai-docs/gaps.md` §3.

Note for next session: while the A/B ran, a first attempt (backgrounded via `nohup ... &`) didn't
actually die when its launching shell exited as expected — it kept running and collided with a
second, cleanly-restarted attempt writing to the same `memory/ab_der_auto_lambda/control` path
before the stale one was caught and killed. The result above is from the clean single-process run
only (verified no PID collision before trusting the numbers). Lesson: prefer the Bash tool's own
`run_in_background` over manual `nohup ... &`, and always verify only one process is writing to a
given output path before trusting metrics from a backgrounded training run.

## Just completed (2026-07-06)

Both action items from `ai-docs/research/brain-scale-comparison.md`'s "Summary judgment" are done
(`9bba42b` change-gated video encoder, `b410ad7` SVD null result). High replay ratio (item 2)
validated and adopted at N=4 (`d5ce2ab`, `f0fa9c6`) — see `ai-docs/gaps.md` §2 for the N=1/2/4
comparison table.

**gaps.md §3, DER++ auto-λ Phase 3 — implemented and committed (`f2fd185`).** Stale-output
blocking (Phase 2) was already shipped (commit 65ceaf7); this adds the auto-λ half.
`fused_replay_wake_step_batch` (`crates/somnus-perception/src/lib.rs`) previously combined
`sae_loss`+`wake_loss` as an unweighted sum. New opt-in behavior (`--der-auto-lambda`, default
`false`): tracks a running EWMA mean/mean-square of `wake_loss` (α=0.05, seeded from the first
sample to avoid a cold-start zero artifact), derives its variance, and weights `wake_loss` by
`1/variance` (clamped `[0.1, 10.0]`) — homoscedastic uncertainty weighting instead of a hand-tuned
constant. New unit test `der_auto_lambda_stays_finite_and_loss_decreases`
(`crates/somnus-perception/tests/phase5_smoke.rs`) confirms the weighted path stays finite and loss
still decreases over repeated passes. Build/clippy/test all pass; manually smoke-tested via CLI
(`--der-auto-lambda`, no crash).

## Queued for next session

- **gaps.md §9 — EfficientAT-mn10, resume the A/B.** Implementation is done (see "In
  progress" section above for full detail) — this is a resume, not a fresh start. Exact
  steps: `rm -rf memory/ab_efficientat_mn10` then `bash scripts/ab_efficientat_mn10.sh`,
  then document + commit. Was going to be "Low effort" per the original gaps.md estimate;
  turned out Medium (cold-start region-dim change, new Rust mel port, isolated venv) — same
  pattern as §3 before it. Effort actually spent is sunk; only the A/B run + writeup remain.

## Current state

- **DINOv3-ViT-S**: adopted and active (drop-in 384-dim replacement for DINOv2-S). Validated via
  A/B: visual-region MSE -95.6%, `warm_recall_l2` +61.5%, no confound in other regions.
  Details: `ai-docs/a_b_dinov3_findings.md`, `ai-docs/gaps.md` §6.
- **Qwen3-0.6B**: adopted as the sole action-decoder LM (ADR 0036); Qwen2.5 support fully removed
  (`qwen3`/`qwen3_hidden` Cargo features deleted, `LM_HIDDEN`/`LANG_HIDDEN`/`LANG_DIM`/
  `EOS_TOKEN_ID` hardcoded to Qwen3's values, `--action-model-prefix` defaults to `"qwen3"`).
  Confirmed via a speech-dense A/B (~22% lower `mean_proj_loss`, consistent, no reversals) plus a
  direct `generate()` comparison (coherent English vs. Qwen2.5's incoherent token salad).
  Details: `ai-docs/a_b_qwen3_findings.md`, `ai-docs/gaps.md` §5, ADR 0036 (supersedes ADR 0035).
- **Whisper ASR panic**: an intermittent `decode_greedy` ORT panic ("GetElementType is not
  implemented") is mitigated (not root-caused) — `catch_unwind` around the decode call in
  `crates/somnus/src/main.rs` stops it from silently killing the ASR thread for the rest of the
  session. Unverified whether this fully recovers or just downgrades to a per-cycle WARN if the
  underlying ORT `Session` is left poisoned. Caveat detail: `ai-docs/gaps.md` §5.
- All docs in sync: `ai-docs/architecture.md` + `.svg` (cross-modal dim fixed at 1280+1024=2304),
  `ai-docs/gaps.md`, ADR 0036. Build/clippy/tests pass
  (`cargo build/clippy/test --release -p somnus -p somnus-perception`).
- Housekeeping since: removed accidentally-committed model/build cruft from git (models_control/,
  models_lora_ab_control/, stray logs) and hardened `.gitignore`; corrected a stale "queued" line
  in `gaps.md` (candle deps were already removed from the workspace).

**Next step is user-driven, not queued.** Candidates already scoped in `ai-docs/gaps.md`'s
priority table: high replay ratio A/B (item 2, trivial), DER++ auto-λ (item 3 phase 3), or any
deferred encoder/LLM item. A longer single-arm Qwen3 soak test was offered and explicitly declined
by the user 2026-07-05 (take 3 already confirms the adoption; Qwen2.5 no longer exists in the tree
to re-A/B against).

## Key files

- `ai-docs/gaps.md` — full backlog with effort/impact/status table (read this first for "what's next")
- `ai-docs/a_b_dinov3_findings.md`, `ai-docs/a_b_qwen3_findings.md` — A/B evidence trails
- `scripts/ab_dinov3.sh`, `scripts/ab_qwen3.sh`, `scripts/ab_qwen3_longclips.sh` — A/B harnesses (isolated `--memory-dir` per arm, `--metrics-file` NDJSON)
- `memory/dinov3_ab_{control,treatment}/`, `memory/qwen{25,3}_ab_*/` — raw A/B metrics

## Deferred backlog (do not raise unprompted)

CLAUDE.md's "Deferred architectural items" points here for trigger conditions. That table was lost
in an earlier prune (silently collapsed to a pointer with nothing behind it) — restored 2026-07-11
from git history (commit `03eb9b5`, 2026-06-14 ADR scan). `gaps.md`'s priority table covers
everything not listed here.

| # | Source | Item | Trigger |
|---|--------|------|---------|
| A | ADR 0006 | Cold archive O(1) promotion index | `promote()` called AND cold > 500k records |
| C | ADR 0006 | Per-region sparse activation storage | Replay analysis shows episode vectors insufficient for diagnostics; no sparse-code consumer currently exists |
| E | ADR 0009 | Separate optimizer for output heads (independent LR schedule) | Conflicting gradient directions between output heads and k-WTA observed |
| F | ADR 0010 | Post-k-WTA prior injection | Pre-k-WTA injection shows poor `prior_cosine` uptake |
| I | ADR 0018 | Knowledge distillation compression | Grown regions exceed 2× baseline |
| J | ADR 0003 | Whisper-tiny → Whisper-base / WavLM-base (audio quality) | — (Tier 3, revisit opportunistically) |
| K | ADR 0003 | DINOv2-S → SigLIP-B (language-aligned visual embeddings, Phase 7+) | — (superseded in spirit by DINOv3-ViT-S adoption, gaps.md §6 — SigLIP2-ViT-B/16 tracked separately as gaps.md §7) |
| L | ADR 0012 | LoRA fine-tuning of Qwen2.5 for episode-vector prefix | — (Tier 3) — **caveat:** this was marked done 2026-06-25 for Qwen2.5, then Qwen2.5 was fully removed from the tree (ADR 0036, 2026-07-05). Not re-verified whether an equivalent LoRA path exists/is warranted for Qwen3 — check `crates/somnus/src/sleep.rs`/`main.rs` LoRA-related code before assuming this is either done or not-yet-started. |
| M | ADR 0012 | MLP projection (640→5×896) instead of linear | — (Tier 3) — **caveat:** same as L, marked done 2026-06-21 against Qwen2.5's 896-dim output; not re-verified against Qwen3's dimensions post-ADR-0036. |
| N | ADR 0014 | Visual/audio region grounding (captions, env sound labels) | — (Tier 3) |

Original table also had items B, D, G, H (language last-token pooling, full differentiable SAE,
learned per-region gate, activation-frequency pruning) — those are not in CLAUDE.md's current
deferred list, meaning they were resolved/superseded/dropped at some point after the 2026-06-14
scan. Not re-investigated here; if one becomes relevant, check `gaps.md` and ADR history before
assuming it's still open.
