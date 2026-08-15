# Engineering log (archive)

> **Contents:** Bug fixes, pilot/A-B numeric results, campaign narratives, and other non-decision history from resolved `gaps.md` items. **Decisions live in `ai-docs/decisions/` (ADRs) — where an ADR exists for an item, the ADR is the canonical record of *why*; this file is not it.** Sections with an ADR are trimmed to a pointer plus whatever numeric/debugging detail doesn't fit an ADR's format (raw A/B tables, root-cause trails, per-cycle deviations). Sections with no ADR (bug fixes, test-coverage pushes, campaign milestones, mechanisms not yet defaulted) keep their full write-up here, since this file is their only record.
> **When to read:** Looking up what a pilot actually measured, how a bug was root-caused, or campaign history. For *why a decision was made*, read the ADR first.
> **Numbering:** section numbers (`§N`) are unchanged from their original assignment in `gaps.md` — external docs cite them as `engineering-log.md §N`. Do not renumber.
> **Promoted remainders:** a few closed-header sections had an open tail that didn't belong in an archive. Those tails were split out and now live in `gaps.md` under new numbers — noted inline at the point of the split (§5 → §38/§39/§40; §13 → §41; §25/§28/§29 → §42; §37 → collapsed into §38/§39).

---

## 0. Critical: bincode security advisory ✅ RESOLVED

**RUSTSEC-2025-0141.** Migrated bincode 2.x (archived) → bitcode 0.6 (maintained). Commit 57c87a1. `crates/somnus-memory/src/semantic.rs` updated; WARM_VERSION 1→2. Old bincode sidecars trigger graceful cold rebuild (existing fallback). All tests pass.

---

## 1. Algorithm: BatchTopK SAE ✅ DEPLOYED — see ADR 0042

Decision, alternatives, and current-code verification: `decisions/0042-batchtopk-sae.md`. Landed in two commits — `81b64c7` (opt-in, default false) then `0eca914` (flipped both defaults to true after the A/B passed). A/B: commit `0eca914`'s message records PASS, +3.6% MSE within pre-stated tolerance (the log previously cited commit `25d7496` for this result; that hash no longer resolves in repo history — `0eca914` is the surviving record).

---

## 2. Algorithm: High replay ratio ✅ ADOPTED (N=4, 2026-07-06) — see ADR 0043

Decision, full N=1/2/4 table, and the N=2 cost-efficiency trade-off: `decisions/0043-nrem-replay-ratio.md`. **Bug found first:** `train_campaign.sh` hardcoded `--nrem-replay-epochs 4` regardless of `EXTRA_ARGS`, so the prior A/B (2026-06-27) compared N=4 vs N=4 — meaningless. Fixed via a `REPLAY_EPOCHS` env var, set explicitly per arm, before the real A/B in the ADR ran.

---

## 3. Algorithm: DER++ auto-λ + stale-output blocking — see ADR 0044

Both sub-decisions (stale-output blocking adopted; auto-λ rejected after A/B, +73.8% worse, 10/10 cycles regressed) and full rationale: `decisions/0044-derpp-stale-blocking-and-auto-lambda.md`.

---

## 5. LLM: Qwen3-0.6B ✅ ADOPTED (2026-07-05, ADR 0036) — Qwen2.5 dropped

Decision, consequences, and final generation-quality confirmation: `decisions/0036-adopt-qwen3-drop-qwen25.md`. Full A/B detail: `a_b_qwen3_findings.md`.

**Path to adoption (bug trail, not in the ADR):** an initial A/B (2026-06-28) was invalid — a hardcoded 896-wide embed table sliced at stride 1024, corrupting embeddings both arms. Fixing it surfaced two more bugs: Qwen3 loaded into the wrong model class, and ~6 more hardcoded language dims disagreeing. Re-run on the stock 4-clip corpus was inconclusive (too little speech); re-run on speech-dense `clips/long/` gave the ~22% `mean_proj_loss` win the ADR reports.

> **Open remainders split out 2026-08-15:** the adoption above is fully decided and closed. Three things it surfaced stayed open and now live in `gaps.md`: KV-cache split generation incomplete for Qwen3 (§38), LoRA fine-tuning never run for Qwen3 (§39), Whisper `decode_greedy` panic root-cause undiagnosed (§40).

---

## 6. Visual encoder: DINOv3-ViT-S ✅ VALIDATED & ADOPTED (2026-07-05) — see ADR 0045

Decision, the MIGraphX misdiagnosis (a CWD-relative external-data path bug misread as an attention-graph incompatibility), the shape-handling fix, and the full A/B table: `decisions/0045-dinov3-visual-encoder.md`. Full A/B write-up: `a_b_dinov3_findings.md`.

---

## 8. Audio env encoder: M2D-CLAP ⏸️ NOT PURSUED (2026-07-09) — see ADR 0046

## 9. Audio env encoder: EfficientAT-mn10 ❌ REJECTED (2026-07-09) — see ADR 0046

Both candidates, the feasibility/licensing analysis, the full A/B table, and the paper-benchmark-vs-pipeline-quality divergence point: `decisions/0046-audio-env-encoder-candidates.md`.

---

## 12. Null result: episodic binding survey

All 2025-2026 binding improvements outperforming 4-head cross-attention + GRU use Mamba/SSM variants (arXiv:2601.21648, arXiv:2509.00677), all requiring `selective_scan_fn` — no ONNX equivalent, unsupported by MIGraphX. **Conclusion:** current binding is state-of-art under the ORT/MIGraphX constraint. Not pursued. One extractable idea, unvalidated: a scalar learned gate per modality per GRU step (sigmoid+multiply).

---

## 13. Change-gated video encoder inference ✅ DEPLOYED (2026-07-06)

Mirrors the biggest efficiency mechanism real neural tissue uses (spikes only on change). `VideoFrame::mean_abs_diff` (`somnus-sensory/src/frame.rs`) computes normalized mean-abs-diff between consecutive candidate frames (returns 1.0/"changed" on length mismatch — covers Mjpeg/resolution changes). The video encoder task skips `enc.encode()` below `--video-change-threshold`; a `--video-change-max-skip` heartbeat (default 30) forces periodic re-encode. Default threshold 0.0 (disabled, opt-in). Audio encoders left ungated (VAD-style silence detection is a different problem, out of scope).

> **Open remainder split out 2026-08-15:** the mechanism deployment above is closed. It was never A/B'd for real wake-path savings (skip rate / downstream binding effect) — that measurement gap now lives in `gaps.md` §41.

---

## 14. Null result: SVD low-rank check (2026-07-06)

Checked whether somnus's 21 growable weight matrices (region encoders, cross-modal projection, binding projections, Q/K/V, GRU, temporal projection) have learned low effective rank (would make LoRA-style compression near-free). New bin `svd_rank_check` dumps every 2-D matrix; `scripts/svd_rank_check.py` (numpy) runs SVD per matrix. Checked against the most-trained checkpoint on disk (`[1664,1600,1600,1600,1600]`).

**Result: null.** Rank needed for 95% energy sits at 60-82% of full rank (mean 0.772) across all 21 matrices — too close to `min(n,m)` for `(n+m)·r` low-rank parameterization to undercut `n·m`. Effectively full-rank. **Do not pursue.** Caveat: single checkpoint; re-check if a matrix's dims grow much larger relative to EPISODE_DIM/D_PROJ, though the consistency across all 21 (no outliers) makes a different result unlikely.

---

## 16. Growth-scaling pilot campaigns (10x growable-core) ✅ COMPLETE (2026-08-15, see §34)

**Goal:** scale the growable core (~5,400 units baseline) 10x toward honeybee-order (~54,000 units, ~9,660/region), not mouse-order (~71M, ~400x). Growth mechanism (`grow_region_seeded`, ADR 0024 §2c) already implemented — this item characterizes cadence/cost before committing unattended compute.

**Pilot 1 (2026-07-09):** 75 cycles, tight 10-clip looping corpus. 16 growth events (~1/4.7 cycles, gate-limited by the 3-cycle cooldown), no plateau. Extrapolated: ~135 events, ~633 cycles, **~3.9h** at ~21.8s/cycle.

**Pilot 2 (2026-07-11):** added `download_podcasts.sh` (10 shows, 10-90min filter, 56 episodes/5.6GB) to close the memorization-plateau caveat with a 145-clip diverse corpus. **Result: no plateau**, indistinguishable from pilot 1 (16 events, `per_region_mse` 0.0135→0.0039, final `[2048]×5`). Confirms gate-limited, not corpus-limited.

**Pilot 3/"v2" (2026-07-12):** rerun after §17/§18/§19 fixes. 14 events (4-7 cycle cadence, unchanged), `per_region_mse` 0.0136→0.0048, `k_active` scaling confirmed live (`scaled_k(1523)=30` vs baseline `scaled_k(1024)=20`, ratio preserved). **Recalibration:** mean batch 35.6/event (nearly half the old flat 64 — §19 correctly makes growth severity-proportional, and this corpus mostly produces moderate novelty). Re-estimated: **~242 events, ~1,298 cycles, ~9.3-11.5h** — 2.4-3× the pre-§19 estimate.

**Done (2026-08-15):** full campaign (launched 2026-07-12, §20) reached `[9821]×5` (101.7% of 9,660) after 791 cycles / 188 events, roughly half the pilot-3-projected budget (real mean batch 46.8 vs projected 35.6). All pilots ran in isolated scratch dirs, never the main `memory/`. Full writeup: §34.

---

## 17. K_ACTIVE scaling with region growth ✅ DONE (2026-07-12)

**Problem:** `K_ACTIVE=20` was a hardcoded const, untouched by growth — sparsity would have fallen 2%→0.2% (10×) at the 10x target, untested by either pilot (both only reached ~2× baseline).

**Fix:** `KWtaLayer` gained a per-layer `k_active` field; `region::scaled_k(n_units) = round(n_units·K_ACTIVE/N_UNITS)` preserves the baseline ratio linearly (chosen over sub-linear scaling — simplest, reversible). `grow_region`/`grow_region_seeded`/`prune_region` all call it; all k-WTA selection sites read `self.k_active`. Also fixed 4 ORT-bootstrap placeholder sites + `binding_only` that hardcoded N_UNITS/K_ACTIVE instead of live size (latent bugs for a grown region after snapshot reload). New tests: `scaled_k_preserves_baseline_ratio`, `kwta_respects_grown_k_active`. Full suite passes.

**Not empirically A/B'd** — the fix corrects a design/implementation mismatch a priori; no pilot rerun to confirm it helps (vs. just being correct) at multi-x growth. Judged closed: the fix directly matches the encoder's documented sparsity contract and has run cleanly through the entire 10x campaign (§34) with no observed harm.

---

## 18. Scope optimizer reset to grown region only ✅ DONE (2026-07-12)

**Problem:** growth/prune reinitialized AdamW for the *entire* model, discarding every other region's valid momentum ~135× projected over the campaign.

**Investigated burn-optim's source** before changing anything: momentum is a `HashMap<ParamId, AdaptorRecord>`; a param missing from the map (as the fresh `ParamId::new()` growth/pruning already assigns to resized tensors) automatically starts fresh — no full reset needed.

**Fix:** deleted the full-reset line from all three call sites; nothing else changed. Confirmed unrelated to `refresh_autodiff_state` (a separate, legitimate optimizer-reset mechanism for NREM graph-accumulation slowdown — untouched). Full suite passes. **Not empirically A/B'd** — follows directly from source, but no pilot measured whether preserved momentum improves convergence in practice (no visible harm from the old behavior either). Judged closed on the same basis as §17.

---

## 19. Novelty-proportional growth batch size ✅ DONE (2026-07-12)

**Problem (user-raised):** flat `GROWTH_BATCH=64` added per growth event regardless of how novel the triggering signal actually was — relative growth rate necessarily decayed over a long campaign against an ever-growing denominator.

**Fix:** `GROWTH_BATCH_MIN=16`/`MAX=128` (old 64 sits mid-range) + three pure functions: `gist_novelty(rate, threshold) = clamp((threshold-rate)/threshold, 0, 1)`, `mse_novelty` (mirror for the MSE-fallback trigger), `growth_batch_size(novelty)` (linear interpolation MIN→MAX). The growth loop computes `novelty` alongside the existing trigger match. Seed cap bumped `2×GROWTH_BATCH` → `2×GROWTH_BATCH_MAX`. New tests: `growth_batch_size_spans_min_to_max`, `_clamps_out_of_range_novelty`, `gist_novelty_zero_at_threshold...`, `mse_novelty_zero_at_threshold...`, `worse_recall_failure_grows_a_bigger_batch`. **Deliberately no EWMA smoothing** — `within_session_gist_rate` is already an average with inherent per-cycle stability; add smoothing only if a pilot shows oscillation.

**Empirically validated (§16 pilot 3):** batches varied sensibly (17-77, tracking novelty 0.01-0.54) — mechanism works. But total capacity added (499 units/region) was less than flat-64 would have produced for the same 14 events (896), because most triggers land at moderate novelty. Not a bug — correctly severity-proportional — but means the 10x campaign needed ~2.4-3× more cycles than previously estimated (§16).

---

## 20. Full 10x campaign launch + `campaign_status.py` + `eval_metrics.py` region-name fix (2026-07-12)

> **Completed 2026-08-15** — `[9821]×5` (101.7% of 9,660). See §34.

`memory/` wiped completely first (user's explicit call, ~17GB, all gitignored/reproducible) so the core regrows on the post-§17/§18/§19 mechanism. Launched via `train_campaign.sh` (`SESSIONS=130 CYCLES=10 CAPACITY=8192`, 145-clip corpus, `CAMPAIGN_DIR=memory/campaign_growth10x`), unattended.

**Early rate beat pilot 3:** at ~440 cycles, mean batch ~43.4 (pilot-3: 35.6), cadence ~1/4.25 (pilot-3: 1/5.36) — projecting ~6.1h vs. the 9.3-11.5h estimate (provisional; untested whether it held once low-hanging novelty was consumed later).

**New tool `campaign_status.py`:** one-command status (auto-detects newest `memory/campaign_*`) — process state, cycle/session progress+ETA, per-region growth, cadence/batch, ETA to a configurable unit target (default 9,660), quality trend, recall metrics, timing breakdown, disk usage, health check (stale-log, panic scan). Colored green/yellow/red grading against pilot-3 baselines (`EXPECTED_*` constants). No non-stdlib deps.

**Bug fixed while writing it:** `eval_metrics.py`'s `REGION_NAMES` had indices 0/2 swapped vs. the real Rust order (`RegionId::as_index`: 0=AudioSpeech, 1=AudioEnv, 2=Visual, 3=Symbolic, 4=Language) — every growth-timeline/per-region-MSE chart ever generated had mislabeled speech↔visual (env/symbolic/language unaffected). Fixed, verified against live data.

---

## 21. Warm-store HNSW dump leak + broken cross-restart persistence ✅ FIXED (2026-07-12)

**Found answering "why is `warm` 11G but `episodes.cold` 41M for the same 65,798 records?"** `memory/warm/` had **461 separate HNSW dump pairs** (~24-26MB each) — only one ever live, the rest pure leak.

**Root cause (`hnsw_rs` 0.3.4):** `SemanticStore::dump()` calls `file_dump` every cycle, which only overwrites the fixed `index.hnsw.{data,graph}` name in place when `!datamap_opt`. `HnswIo::load_hnsw` (used on process start) hardcodes `datamap_opt: true` on every reloaded index regardless of whether real mmap was requested — so after the very first dump of a process's life, every subsequent dump (this process and all future ones) hits the "can't overwrite" branch and picks a random `index-<N>` suffix. Nothing ever deleted the superseded files → ~24-26MB leak per cycle, unboundedly.

**Second consequence:** `load_warm()` only ever looks for the plain `index.hnsw.{data,graph}` name — since real dumps after the first always land under a random name, warm-index reload has silently failed on every restart since this mechanism existed, always falling back to a full cold-archive rebuild (not a correctness bug — cold is authoritative — but wasted work every restart).

**Fix:** `dump()` now deletes any existing `index*.hnsw.{data,graph}` before calling `file_dump` — bounds disk usage to one live pair, and restores plain-name persistence so `load_warm()` works again. Verified live: one pair after dump, next restart loads it successfully (safe on Linux even if a process had it mmap'd — inode stays alive via the existing fd until unmapped).

**Applied live to the §16/§20 campaign:** stopped cleanly (SIGTERM, snapshot-per-cycle so minimal loss), fix built/tested (35/35 `somnus-memory` tests), 461 orphaned pairs cleaned (`memory/warm/` 11G→192M), resumed `SESSION_START=50 SESSIONS=82` — metrics continuous, no gap. Confirmed next dump/restart cycle worked. Ended at 362M (live index + sidecar only).

**Also fixed two `campaign_status.py` bugs found via the restart:** (1) crash on JSON `null` metric fields (`dict.get` doesn't catch explicit `null`) — added a `gv()` helper. (2) `params.txt` is append-only across restarts, so `SESSIONS×CYCLES` alone understated the whole-campaign target after a resume — fixed to `(SESSION_START+SESSIONS-1)×CYCLES`.

---

## 22. Auto-enable ORT/MIGraphX perception inference past the CPU/GPU crossover ✅ DONE (2026-07-12) — see ADR 0037

Decision, the two prerequisite bugs fixed in `export_perception_graph.py`, and the full correction trail (the initial "~2.1×" measurement was invalid — see §24): `decisions/0037-auto-onnx-perception-inference.md`, which already carries this correction inline. **Deferred:** eliminating the Python export step via native Rust ONNX construction — no mature crate exists; tracked as `gaps.md` §23 (still open).

---

## 24. ORT sparse→dense conversion hardcodes baseline N_UNITS ✅ FIXED (2026-07-13)

This bug (and its fix) is the correction referenced inline in `decisions/0037-auto-onnx-perception-inference.md` — that ADR has the final confirmed ~3-4× speedup number; this section keeps the root-cause/debugging trail.

**Found before a routine shutdown check:** `perception_total_ms` had spiked to 188,000-196,000ms/cycle (vs §22's ~6,500-7,700ms), recall metrics collapsing. Log grep: **102,810 occurrences** of "ORT encoder/binding failed, falling back to burn" across session 57's entire runtime — present since session 56 (ORT's very first activation), not a later regression.

**Root cause:** `lib.rs`'s ORT call path converts sparse activations to dense via `sp.to_dense_n(N_UNITS)` using the **hardcoded baseline constant** (1024), not each region's actual (grown) size — so every call past baseline size failed dimension checks, was caught, and silently fell back to burn/CPU, paying both the ORT round-trip overhead *and* the full burn computation. This invalidates §22's reported speedup (both its verification run and the campaign were already past baseline). Same drift class as §17's K_ACTIVE bug and §22's Python-side N_UNITS bug — a second, non-derived copy of a per-region size.

**Fix:** all 5 `to_dense_n(N_UNITS)` call sites now pass each `SparseActivation`'s own `n_units`; `lang_zero` fallbacks read the live layer's `n_units`; `onnx_session.rs`'s `run_binding` derives its `Array2` shapes from each `sparse_z[i]`'s own length instead of the constant. New regression test `ort_grown_region.rs` loads the real leftover campaign ONNX graph (n_units=6886), confirmed to fail pre-fix (`git stash` check) and pass post-fix. Full build/clippy/test clean.

**Live verification (resumed session 58):** binding-path fix worked immediately (zero `ORT binding failed`, sessions 58-60) but surfaced a **second independent bug**: `ORT encoder failed` on `language_enc`, 100% of calls — `onnx_session.rs`'s `ENC_DIMS` hardcoded the language input at the pre-ADR-0036 value 896 instead of 1024, same drift class, never fixed on the Rust side. Fixed by deriving `ENC_DIMS` from live constants; new regression test `ort_encoder_accepts_current_lang_hidden_dim` confirmed the fail→pass transition.

**Impact even before the encoder fix:** `perception_total_ms` mean dropped 250,099ms (session 57, both bugs live) → ~1,900ms (sessions 58-59, binding fixed only) — already below the pre-ORT-crossover CPU baseline (~6,900-7,200ms).

**Fully confirmed (sessions 60-61, both fixes live):** zero ORT encoder/binding failures across ~8,000 log lines. `perception_total_ms` mean 2,158ms/1,457ms vs. session 57's 250,099ms and the pre-crossover 6,900-7,200ms baseline. **Conclusion: ~3-4× real speedup**, superseding and retracting ADR 0037's invalid "~2.1×/2.4×" (measured entirely on the silent-fallback path). §24/ADR 0037 fully closed.

---

## 25. Sparse-aware binding projection ✅ DEPLOYED (2026-07-13)

**Motivation:** k-WTA activations are already sparse in representation (~2% via `scaled_k`), but `BindingArea::forward` still materialized a dense `[1,n_units]` vector and ran a full dense `[n_units,D_PROJ]` matmul on it — ~98% wasted multiplications. Live profiling flagged `perc_bind_us`/`perc_settle_us` as the two largest region-size-dependent cost buckets.

**Fix:** `select(0, active_indices)` gather (`[k,D_PROJ]`) + small `[1,k]×[k,D_PROJ]` matmul — numerically identical (no bias term to account for), autodiff-tracked (standard sparse-gradient `select`). New test `sparse_projection_matches_dense_reference` proves equivalence; all 5 pre-existing `BindingArea` tests pass unchanged. **Measured 11.4× speedup** at live campaign scale (n=8,655, k=169: 532µs→47µs/call); the ratio holds at any future scale since `scaled_k` keeps k/n constant.

**Correction (2026-07-14) — currently dormant on the live campaign.** Sessions 72-75 showed no `perception_total_ms`/`nrem_wall_ms` drop. `perc_bind_us` times the whole `try_bind` call, which — once ORT is active past the §22 crossover — dispatches to `ort.run_binding`, not `burn`'s path at all. `BindingArea::forward` only fires from `try_bind`'s non-ORT branch, `settle_episode`'s ORT-failure fallback, and the `binding_only` validation helper — none active on this healthy-ORT campaign. **Also found while tracing this:** no training function calls `BindingArea::forward` either — its weights receive no gradient from any path in the codebase. Investigated 2026-08-14, became `gaps.md` §33 (still open — the central open item).

**Net effect:** real, verified, 11.4× win wherever `burn`'s binding path actually runs (pre-crossover, CPU-only/no-MIGraphX, ORT-failure fallback) — kept deployed (strictly better, free when dormant), but not moving this campaign's current wall-clock. Lesson: check which backend owns a profiled call site before attributing cost to `burn`.

**Queued next (from the same discussion):** the k-WTA encoder itself is inherently O(input_dim×n_units) and can't be sparsified the same way — the real lever for approaching mouse-order scale. Written up as a research note (`sparse-coding.md` §"Encoder-side scaling") + **ADR 0038** (`proposed`), which picks hierarchical/coarse-to-fine WTA over MoE-routing and approximate sampling (preserves exact top-k within blocks, degrades gracefully, may reuse HNSW for the coarse stage).

> **Open remainder split out 2026-08-15:** ADR 0038's prerequisites 1 and 2 (§28, §29 below) both answered; prerequisite 3 (numerical validation that block-local competition doesn't break growth/duty-cycle/binding assumptions) was never started and stays open, along with §29's scale-based revisit trigger — both now live in `gaps.md` §42.

---

## §26 — Unified two-tier metrics instrumentation (`somnus-metrics`, ADR 0039)

Raised while scoping ADR 0038's Phase 0 (no shared timing mechanism existed — every call site hand-rolled `Instant::now()`/`.elapsed()`). Decision, full coverage-expansion history (`Metric` 24→102 variants across 5 follow-up rounds), and the two-tier Core/Fine policy that's now standing project policy: `decisions/0039-unified-metrics-instrumentation.md` — its update blocks already carry this section's detail verbatim, this pointer avoids maintaining two copies.

---

## §27 — Parallelized warm-store HNSW rebuild

The new `SleepWarmRebuild` Core metric (§26) immediately surfaced a large invisible cost: once the warm cap is hit (every cycle, per §21), `rebuild_from_recent` does a full from-scratch serial HNSW rebuild. Live data: **46-51s/cycle at ~145-148k warm records — ~70% of total cycle wall-clock.**

`hnsw_rs` 0.3.4 has no delete API (insert-only), so a full rebuild is the only way to enforce "most recent N" eviction without a crate swap or a tombstone scheme (not pursued). Cheap fix: `hnsw_rs::parallel_insert_slice` (rayon, already a workspace dep) replaces the serial loop — same graph/eviction semantics, spread across cores.

**Verified on real production data:** benchmarked through the real metric wrapper against a copy of the campaign's actual cold archive (148,447 records), `git stash`/`pop` old-vs-new: 47.6s (matches live campaign) → 2.85-2.94s — **~16.7× speedup**. New `somnus-memory/examples/bench_warm_rebuild.rs` is a reusable regression harness. Build/clippy/test clean. Deployed via `resume_growth_campaign.sh`'s rebuild-first default; ran through the rest of the §34 campaign.

---

## §28 — ADR 0038 prerequisite 1: exported encoder graph + live scoring cost

Full detail (measured `OrtRunEncoder`/`OrtRunBinding` costs, the §30 correction to the "region-count-independent" framing) now lives in `decisions/0038-sparse-encoder-scoring.md`'s own "Prerequisites" section — that ADR already quotes this section verbatim. See `gaps.md` §42 for prerequisite 3 (still open).

## §29 — ADR 0038 prerequisite 2: does this matter at the targeted scale?

Same as §28 — full detail (resolves negative at honeybee-order scale, but the ADR stays `proposed`/deferred rather than closed, with an explicit revisit trigger) lives in `decisions/0038-sparse-encoder-scoring.md`.

---

## §30 — Sparse-gather binding projection (ONNX/MIGraphX), ADR 0040 — correction to §28/§29

**Correction:** `build_binding_graph` also ran each region's `sp_i[1,n_units]@W_region_i[n_units,D_PROJ]` — a dense matmul that *does* scale with n_units and was the dominant cost inside `OrtRunBinding` (75-76% at campaign scale), not the GRU+attention §28/§29 attributed it to.

Full write-up (sparse-gather rewrite, MIGraphX probe/compiler-bug workaround, numerical parity, measured 54-59% synthetic / 40-59% live-production win, Rust contract change, deployment trail including two unrelated environment blockers fixed along the way): `decisions/0040-sparse-gather-binding-projection.md` — comprehensive, this section would only duplicate it.

---

## §31 — Test-coverage push: rubato-5 contract drift + drift-pin pattern (2026-08-13)

Closed the last three advisor-ranked targets: `somnus-encoding` (0 unit tests), `whisper_asr.rs` (0 tests, on every request's critical path), `somnus-metrics` (3-5 tests for a 100-variant enum). **54 new tests** (somnus-encoding 0→41 incl. 3 model-gated integration tests; somnus lib 91→99; somnus-metrics 3→8/10). Workspace 254 passing.

**Finding 1 — rubato 5 silently broke `SincResampler`'s contract.** The 3→5 major-version bump (verified only by manual smoke run) let `Async` accept input longer than chunk size, and `new_sinc` accept degenerate ratios (in_rate=0) — same silent-drift class as §17/§22/§24, across a dependency boundary. Fixed: `new` rejects zero rates, `process` rejects wrong-length input. No production behavior change (real callers always pass valid data); guards exist so a future bug fails loudly. Pinned by 7 tests (frequency preservation, DC passthrough, identity-ratio, rate tracking).

**Finding 2 — the "drift pin" pattern.** `whisper_asr.rs` hardcodes 5 Whisper special-token IDs that must match `models/whisper_tokenizer.json` — new test asserts the match against the real on-disk artifact, not a copy of the constant. Generalizes: any hardcoded constant that must match an on-disk artifact gets a test against that artifact.

**Metrics discipline:** tests + 2 error-path guards only, no data-handling behavior change — `AudioResample` (Core) already covers the path. No new metrics needed. **Remaining thin spots (unflagged):** somnus-perception burn-path training, somnus-memory I/O — queued list otherwise empty.

## §32 — Test-coverage push round 2: burn-path contracts + memory I/O (2026-08-14)

Closed §31's two remaining thin spots. **27 new tests** (somnus-memory 35→47; somnus-perception +15 new `tests/burn_path_contracts.rs`). Workspace 281 passing, both feature states re-verified (tests are dimension-generic via `REGION_DIMS`).

**somnus-memory (12 tests):** `snapshot.rs` (3→9) round-trip + malformed-fallback pins for every small restart state file (`region_sizes.bin`, `growth_cooldown.bin`, the 3 ADR 0024 EWMA files, each asserting its documented default seed); `recollection.rs` (0→3) `promote()` semantics; `cold.rs` (13→14) `iter_from` seek correctness (§27's hot path); `semantic.rs` (8→10) insert skip-at-cap, `records_since` inclusive filter.

**somnus-perception (15 tests):** `measure_mse` strictly decreases with replay training (the growth-gate signal); head linearity pins (bias-free `Linear`); save/load exact round-trip incl. `.sizes` sidecar, grown-model restore, failed-load fallback; `grow_region_seeded` bookkeeping; `collect_high_residual_inputs`; duty cycles; `encoder_z_only`/`binding_only` dims/determinism/grown sizes.

**No production code changed** (one dev-dep: `tempfile`). No new metrics — nothing touching data-handling changed. **State after this round: no remaining named thin spots**; the two untestable-in-CI surfaces stay skip-don't-fail model-gated.

---

## §34 — The 10x growth campaign is complete (2026-08-15)

**Target reached:** all 5 regions at **9,821 units = 101.7%** of the 9,660 goal, 791 logged cycles / 188 growth events (mean batch 46.8, cadence 1/4.21) over 82 sessions, `[1024]×5`→honeybee-scale — roughly half the pilot-3-projected cycle budget (real batches ran larger than the pilot's 35.6).

**Final leg:** the prior session's last-expected run (session 80) was interrupted by a machine shutdown mid-cycle-8 at `[9645]×5` (15 short). One aborted relaunch (session 81, killed by a launcher process-group timeout, harmless/snapshot-consistent), then session 82 ran clean to completion: 9645→9675(c5, target crossed)→9769(c8)→9821(c12). ONNX artifacts re-exported at final size by the session's own rebuild.

**Health:** 0 panics, 0 segfault restarts, 0 ORT fallbacks (final session), 7 `misc_warning` lifetime. Final-cycle timing at 9,821/region: perception 1.6s, NREM 7.4s, REM 9.4s, snapshot 2.9s, spill 5.4s.

**Quality:** `mean_output_mse` first-10 vs last-10: 0.0430→0.0112 (**−74.0%**); latest per-region MSE 0.0001-0.0019. `gist_recall_rate` 0.562, `within_session_gist_rate` 0.820, `warm_recall_l2` 0.381. Warm 151,725 (816M) / cold 158,534 (101M) records. Eval harness ran automatically on exit — all charts regenerated.

**Unblocks:** `gaps.md` §33's binding-weights decision (options a/b/c), explicitly sequenced behind campaign completion.

---

## §35 — ADR 0041 isolated A/B pilot: mechanism validated, §33 fix engages, drift materializes (2026-08-15)

Mechanism, verdict, and the corrected mitigation analysis are in `decisions/0041-live-episode-binding-training.md`'s status line and Validation-plan item 2 — not repeated here. **Setup:** `ab_binding_replay.sh`, trained production state copied per arm (reflink), 1 session × 10 cycles, campaign corpus, both arms `fine-metrics`.

**Numeric record (deviations, per the standing A/B instruction):** growth diverged (10,032 treatment vs 9,925 control), `mean_salience_wake` +34%, `settle_delta_mean` +77%, `mean_output_mse_wake` −21.5% (treatment better during wake), NREM `mean_output_mse` +46.1%, `mean_proj_loss` +24.8%. `warm_recall_l2` 0.44→0.89 treatment vs 1.29→0.31-0.44 control; cross-session `gist_recall_rate` collapses to 0.23-0.45 treatment vs 0.90-0.99 control; `within_session_gist_rate` +13.3%. `binding_replay_loss` 0.141→0.046 (−67.5%); `nrem_binding_replay` 28 steps @ mean 1.44s/step (+34.5% NREM wall-clock, `traces_replayed` −30%, `blend_count` −64%). Control: grown_nz = 0/8,901 all regions; treatment: 47-96% of grown rows nonzero (grown_max 0.04-0.14 vs base mean ~0.19). n=1/arm, 0 NaN, both clean exits.

## §36 — ADR 0041 repeat pilot at `--binding-replay-steps 1`: wall-clock improved, quality got worse (2026-08-15)

Verdict and interpretation are in `decisions/0041-live-episode-binding-training.md`'s status line and Validation-plan item 4. **Setup:** same harness, post-§35 production state, CYCLES=20 (vs §35's 10), n=1/arm, both `fine-metrics`, treatment steps capped at N=1.

**Numeric record:** `nrem_wall_ms` −24.8% (control 6,784ms/cycle vs treatment 5,099ms). On the 13/20 cycles where the step fired, `mean_output_mse` ran ~10× control same-cycle (e.g. cycle 3: 0.0098 vs 0.314). 6/20 treatment cycles replayed zero traces (vs control's steady 460-512) — traced to a salience/settle cascade (`sal_threshold` +39%, `settle_gate_dynamic` −44%, `settle_iters_sum` +154%, `prior_beta_effective` −41%). Aggregate: `mean_output_mse` +445%, `mean_real_inference_mse` +655%, `gist_recall_rate` −32.5%, `per_region_mse` +20-159% all regions, `blend_count` −52%, `traces_replayed` −58%, `within_session_gist_rate` +6.6%. `binding_row_norms`: control 0/9,089 grown rows nonzero; treatment 30-78% nonzero, `grown_max` 0.029-0.057.

---

## §37 — Documentation consistency audit: docs vs. code, via codebase-memory-mcp index (2026-08-15)

**Scope:** user requested a full pass over `ai-docs/` for inconsistencies/gaps, cross-checked against the actual codebase (indexed via `codebase-memory-mcp`, 154,553 nodes/165,461 edges), with any findings written up here and doc/code architecture soundness re-evaluated. This followed the same-day mega-compression pass (session-state.md/ai-docs/ cut ~51%) — this audit is the correctness check on both the compression and the underlying docs it compressed.

**Method:** indexed the repo; spot-checked every hardcoded constant/dimension the docs assert (`N_UNITS`, `K_ACTIVE`, `EPISODE_DIM`, `D_PROJ`, `GRU_HIDDEN`, `LANG_HIDDEN`, `REGION_DIMS`, `Metric::COUNT`, CLI flag defaults, Cargo feature lists) against the actual `crates/*/src/*.rs` definitions; diffed each ADR's own status line against `decisions/README.md`'s index; checked `CLAUDE.md`'s deferred-items list and `session-state.md`'s trigger table against current code and prior ADR outcomes; verified file-existence claims (scripts, symlinks, model artifacts) on disk; ran `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` to confirm the health claims implicit throughout the docs still hold (both clean, 0 failures).

**Result: constants, dimensions, feature flags, and file-existence claims all checked out clean** — `N_UNITS=1024`, `K_ACTIVE=20`, `EPISODE_DIM=640`, `D_PROJ=128`, `GRU_HIDDEN=256`, `LANG_HIDDEN=1024`, `REGION_DIMS=[384,ENV_DIM,384,384,LANG_HIDDEN]`, `RegionId` (5 variants), `Metric::COUNT=102` (100→102 from ADR 0041, confirmed in code, correctly reflected in session-state.md though never stated in gaps.md/ADR 0041 themselves — minor completeness gap, not an error), the `qwen3` Cargo feature's removal (ADR 0036), `mn10_env`/`fine-metrics` features, `region_sizes.bin` on disk (`[9821]×5`, matches §34 exactly), all `memory`/`clips`/`data`/`models` symlinks, `scripts/setup-hooks.sh`/`scripts/hooks/pre-commit` (this session's additions), and the `catch_unwind` Whisper-panic mitigation (item 5) all matched what the docs claim. No fabricated numbers found anywhere in the compression pass.

**Four real findings, two fixed in place, two structural (moved out to `gaps.md` §38/§39 rather than duplicated here):**

### §37.1 — `decisions/README.md` index drifted from two ADRs' own status lines (fixed)

ADR 0023's own header says "accepted; Phase 0 + Phase 1 implemented (2026-06-07)" but the index table listed it as `proposed`. ADR 0041's own header says "implemented; ... **Not adopted**" (and, as of §36, failed a second pilot) but the index table also listed it as `proposed`. Both corrected in the index. Neither ADR file itself was wrong — only the index's status column had gone stale, presumably never updated when those ADRs' status lines were later amended in place.

### §37.2 — `CLAUDE.md`'s deferred-items list has one flatly wrong entry (fixed)

Item **M (MLP projection)** was listed as deferred. It is not: ADR 0032 implemented the 640→1024→5×LM_HIDDEN MLP and it has been the **default production path** since 2026-06-21 — confirmed via code, not inference: `--linear-proj` (`crates/somnus/src/lib.rs`) defaults to `false`, and the MLP's `fc1`/`fc2` layers (`crates/somnus/src/action.rs`) are parameterized by the `LM_HIDDEN` constant, which is now 1024 (Qwen3) — the same code that shipped for Qwen2.5 works unmodified for Qwen3, no re-implementation needed. `session-state.md`'s deferred-backlog table had carried a hedge ("not re-verified against Qwen3") pending exactly this check — now resolved, caveat removed, both docs corrected. **Item M should not reappear in any future deferred-items list.**

### §37.3 / §37.4 — Qwen3 KV-cache and LoRA gaps

Both findings (KV-cache split generation not active for Qwen3; LoRA fine-tuning never run for Qwen3) moved to `gaps.md` as standalone open items — §38 and §39 respectively — rather than staying duplicated here. See those entries for full detail (checksums, blocked-commit trail, `ActionDecoder::new`'s activation check).

### Architecture/design soundness re-evaluation

The two orphaned-optimization gaps above (§37.3/§37.4, now `gaps.md` §38/§39) are real but non-critical — cost/quality regressions on a research system with no latency SLA, not correctness bugs. The one gap that **is** architecturally significant, and was already the subject of two failed adoption attempts this same day (§35, §36), remains **`gaps.md` §33: in the default build (`--binding-replay` off), ~89.6% of each region's units — the entire product of the 10x growth campaign — still have zero-initialized, never-trained binding-projection rows, invisible to episode formation.** The growth campaign's SAE-side win is real and measured (`mean_output_mse` −74% first-vs-last-10 cycles, §34), but the binding/episodic payoff of that growth is still mostly unrealized in the deployed (flag-off) configuration, and the two pilots run today to fix this (§35, §36) both failed for different reasons (episode drift; then, after capping step count, worse instability via a salience/settle cascade). No architectural flaw was found beyond what §33/§35/§36 already document — the mechanism itself (live-episode training through `forward_tracked`) is correct and proven to work; the open problem is tuning its blast radius on a live, mixed-generation warm store, which is a genuinely hard problem the docs already characterize accurately. No new mitigation is proposed here; this is a confirmation that the existing gap description is accurate and current, not stale.

**Metrics discipline:** this audit is documentation/build-configuration analysis, no code changed, no new data-handling operation introduced — no new metrics needed, stated explicitly per the standing policy.

---

## §38 — Qwen3 KV-cache generation speedup (ADR 0034) ✅ ACTIVE (2026-08-15)

ADR 0034 now carries this outcome inline (its original write-up was Qwen2.5-only and never mentioned Qwen3 until this fix). This section keeps the full root-cause/debugging trail.

**Was:** `models/qwen3_lm_decode_kv.onnx` never existed; `ActionDecoder::new` fell through to the slow O(n²) `lm_session` path for every `generate()` call since Qwen3 became the sole LM (ADR 0036). Reported blocker: a `torch.export` shape-inference issue in `scripts/export_kv_cache_qwen3.py`'s decode-side export, never revisited after a 2026-06-27 commit noted it and moved on.

**Root cause was not what the blocker note said.** The script already had `dynamo=False` set (the documented fix for the reported issue) — attempting the export fresh hit a *different*, previously-undiagnosed error: `RuntimeError: shape '[1, 1, 16, 64]' is invalid for input of size 2048`. Qwen3's `head_dim` (128, confirmed via `AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")`) is decoupled from `hidden_size/num_attention_heads` (which would compute to 64) — a GQA/architecture detail Qwen2.5 doesn't have, so the script's `HEAD_DIM=64` constant (correct for Qwen2.5, copied over for Qwen3) was silently wrong. Qwen3 also has per-head QK-RMSNorm (`q_norm`/`k_norm` on the `head_dim` axis, `Qwen3Attention.forward`) which Qwen2.5 has no equivalent of and the decode wrapper's manual reimplementation was missing entirely — not a crash, but a numerically-wrong export had the shape bug been papered over without also adding this.

**Three fixes landed in `scripts/export_kv_cache_qwen3.py`:**
1. `HEAD_DIM = 128` (was 64), plus `attn.q_norm(...)`/`attn.k_norm(...)` applied on the `head_dim` axis before rotary, matching `Qwen3Attention.forward` exactly.
2. `attn_out` reshape before `o_proj` fixed to `N_HEADS*HEAD_DIM` (2048), not `LM_HIDDEN` (1024) — they only coincide for Qwen2.5.
3. A second bug found via the equivalence check (below): the decode wrapper's hand-rolled MLP had `gate_proj(x) * silu(up_proj(x))` — SwiGLU's gate/up order backwards vs. the real `Qwen3MLP.forward`'s `silu(gate_proj(x)) * up_proj(x)`. Fixed by delegating to the real `layer.mlp(hidden)` module instead of reimplementing it by hand (matching how `export_kv_cache.py`, the working Qwen2.5 script, already avoided this class of bug).

Also replaced `F.scaled_dot_product_attention` in the decode wrapper with the same manual matmul+softmax attention `export_kv_cache.py` uses (SDPA's export path is the one structurally new element vs. the script that already exports cleanly — pre-emptive, not itself diagnosed as a failure).

**Verification bug found and fixed too:** the script's `verify_equivalence` (ported from `export_kv_cache.py`) compared `zip(ref_tokens, kv_toks)` index-for-index, but by construction `kv_toks[i]` is the prediction for position `prefix_len+1+i` while `ref_tokens[i]` is the prediction for position `prefix_len+i` — an inherent off-by-one. Before the two model bugs above were fixed, this made the check silently unable to distinguish "wrong by one position" (a real bug) from "wrong outputs" (also a real bug) — both scored near-zero. Fixed by comparing `ref_tokens[1:]` against `kv_toks[:-1]`. This same off-by-one likely exists in `export_kv_cache.py` (identical code, never touched here — Qwen2.5 is deprecated post-ADR-0036, out of scope) — flagging in case that script is ever revived.

**Final verification: 19/19 token agreement** (prefix=320, 20 greedy decode steps), prefill logits exact match (`max_abs_diff=0.00e+00`). Exported, consolidated external data into a single `<name>.onnx.data` companion file (the legacy TorchScript exporter defaults to ~200 loose per-initializer files past a size threshold — cleaned up to match the existing `prefill_kv.onnx` convention), moved into `/data/somnus/models/` with the standard symlink-into-`models/` pattern. **Confirmed live**: `somnus --action` now logs `"loading KV-cache FP32 models — fast generate() path active (ADR 0034)"` instead of the slow-path fallback.

Environment: `uv venv .venv && uv pip install torch onnx onnxruntime peft transformers accelerate` (CPU-only, matching ADR 0034's own finding that GPU/MIGraphX is slower for these static-shape KV models). `torch_dtype` deprecation warnings are cosmetic (transformers 5.15 wants `dtype=`), left as-is.

**Not done:** INT8 quantization of the new decode model (`qwen3_lm_decode_kv_int8.onnx`, the higher-priority `Q-B` path in `ActionDecoder::new`) — FP32 is active and correct, INT8 would be a further speedup, not attempted this pass.

---

## §43 — Change-gated video encoder A/B (gaps.md §41): measured, 93.8% skip rate, quality tradeoff concentrated in the visual region — no default adopted

**Pilot:** `scripts/ab_video_change_threshold.sh` (new, follows the `ab_binding_replay.sh` template — copied trained `campaign_growth10x` state, sequential control/treatment, `--features fine-metrics`, 10 cycles). Control: no flag (`--video-change-threshold 0.0`, disabled, matches production default). Treatment: `--video-change-threshold 0.05`.

**Skip rate — the headline §41 question:** control encoded 772/772 frames (0% skip, as expected with the gate disabled); treatment encoded 753/12,107 frames, **93.8% skipped**. The h3h3/podcasts/shorts corpus used for the growth campaign is mostly slow-changing (talking-head/interview footage) at `--video-stride 2` sampling, so `mean_abs_diff` rarely crosses 0.05 frame-to-frame — a much higher skip rate than might be assumed from the threshold's magnitude alone.

**Downstream quality — mixed, not a clean win or loss:**

| metric | control | treatment | delta |
|---|---|---|---|
| mean_output_mse | 0.010370 | 0.012633 | **+21.8%** |
| mean_output_mse_wake | 0.052496 | 0.050438 | −3.9% |
| gist_recall_rate | 0.735365 | 0.798353 | +8.6% |
| warm_recall_l2 | 0.539064 | 0.460980 | −14.5% (lower is better) |
| per_region_mse.Visual (index 2) | 0.001577 | 0.001747 | **+10.8%** |
| nrem_wall_ms | 6539 | 6386 | −2.3% |
| rem_wall_ms | 9359 | 8524 | −8.9% |

Other four regions' `per_region_mse` moved ≤4% either direction. The pattern is coherent: the one region whose *input* is directly gated (Visual) degrades, while several session-level recall/gist metrics improve slightly — plausibly because fewer, more decorrelated video ingests reduce redundant near-duplicate episodes competing in the same warm-store neighborhood, the same mechanism `engineering-log.md`'s BatchTopK/replay-ratio wins traded on, though not confirmed here. `mean_output_mse`'s +21.8% is the one metric that reads as a real regression rather than noise, and it's an aggregate across all five regions, not just Visual.

**Deviations recorded (standing A/B instruction) beyond the headline:** `snapshot_episodic_bytes_written` +53% and `synthetics_produced` +53% in the treatment arm — both moved together, consistent with each other, but not obviously explained by the change-gate mechanism itself (video-frame skipping shouldn't directly change episodic snapshot volume or REM synthesis count). Flagging as unexplained, not investigated further this pass — worth checking before adopting a default threshold, in case the change-gate is perturbing episode timing/salience in a way that has a second-order effect on synthesis.

**Not adopted — no decision on a default threshold.** Real wake-path compute savings are large (93.8% fewer `enc.encode()` calls at threshold=0.05) and the NREM/REM wall-clock deltas point the same direction, but the quality cost isn't zero (visual-region MSE +10.8%, aggregate MSE +21.8%) and the two unexplained deviations above haven't been chased down. A single 10-cycle pilot at one threshold value is enough to answer "does this do anything" (yes) but not "what's the right threshold" or "is the visual-region cost acceptable" — those need either a threshold sweep or an explicit product/research-priority call this pilot doesn't make on its own. `--video-change-threshold` stays default-off (0.0) in production.

---

## §44 — ADR 0041 third pilot: candidate (a), loss-scale reduction — FAIL, worse than either prior pilot (2026-08-15)

Verdict and interpretation are in `decisions/0041-live-episode-binding-training.md`'s Validation-plan item 5. **Setup:** `scripts/ab_binding_replay_loss_scale.sh` (`ab_binding_replay.sh` template). New `--binding-replay-loss-scale <f32>` flag (default 1.0) scales the loss tensor via `loss.mul_scalar(loss_scale)` immediately before `.backward()` (the `BindingReplayLoss` metric still records the *unscaled* value). Treatment: `--binding-replay --binding-replay-loss-scale 0.1`, step cap at default N=1. Control: no flag. 10 cycles, copied trained state, `fine-metrics`.

**Numeric record:**

| metric | control | treatment (scale=0.1) | delta |
|---|---|---|---|
| mean_output_mse | 0.013483 | 0.072891 | **+440.6%** |
| per_region_mse (all 5 regions) | — | — | +25.4% to **+80.8%**, every region worse |
| traces_replayed | 431.2 | 219.0 | −49.2% |
| settle_iters_sum / settle_trigger_count | — | — | +39% both |
| nrem_wall_ms | 6091 | 4917.7 | −19.3% (the only clearly "good" number) |

`nrem_binding_replay` fired 6/10 cycles; control-arm leak check clean (zero firings).

---
