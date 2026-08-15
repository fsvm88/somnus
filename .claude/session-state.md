# Session state — somnus

> **✅ CHECKPOINT (clean), 2026-08-15 (latest) — ADR 0041's third adoption attempt, candidate (a)
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
>   outcome. ADR 0041 was missing its own §44 Validation-plan entry (see checkpoint above).
> - **Wrote 5 new ADRs** (0042 BatchTopK SAE, 0043 replay ratio N=4, 0044 DER++, 0045 DINOv3
>   encoder, 0046 audio-env-encoder candidates) for decision-shaped log sections that had real
>   alternatives-considered material but no ADR. Each verified current code against the log's claims
>   rather than trusting it — caught one real staleness (§1 said `--batch-topk` defaults false; code
>   has it flipped true post-A/B).
> - **Trimmed the log** to pointers wherever an ADR now owns the rationale, keeping every bug-hunt
>   trail and numeric/deviation table not captured in an ADR. 493→375 lines.
> - **This file:** fixed the same staleness bug found in the engineering log — this file's "latest"
>   checkpoint was one pilot behind (missing §44, see above). Compressed all checkpoints from
>   2026-07-14 through 2026-08-15/§36 into the Historical section below (all fully covered by
>   engineering-log.md/ADRs now); kept only the two genuinely current checkpoints above in full.
> - **`a_b_dinov3_findings.md` / `a_b_qwen3_findings.md`:** checked for staleness — both are raw A/B
>   evidence appendices (data, not decision rationale), exactly what ADRs 0045/0036 already point to
>   for "full detail." No overlap or drift found; left untouched.
> - **`research/gpu-unification-design.md`:** found one real staleness — §4's Binding-graph
>   description still described the pre-ADR-0040 dense-matmul contract (`sp_i[1,N_UNITS]` inputs),
>   superseded 2026-08-12 by the Gather-based sparse rewrite. Fixed inline, with a pointer to ADR
>   0040. Also flagged two long-open questions (§13) that got overtaken by events without being
>   revisited: D_PROJ scaling policy (still fixed at 128 after a 10x campaign that was supposed to
>   trigger revisiting it) and EMA-head-sync Option B's N_UNITS≥4096 trigger (crossed ~2.4×, not
>   acted on). Phases 0-4 status lines were all still accurate — not touched.
> - **Docs cleanup queue (from the prior turn) now complete.**

## Historical (pre-2026-08-15, superseded by the checkpoints above — compressed timeline)

- **2026-08-15, §36 (ADR 0041 repeat pilot, step budget N=1):** wall-clock improved (−24.8%) but
  quality got worse — ~10× same-cycle MSE on cycles where the step fired, 6/20 cycles trace-starved
  via a salience/settle cascade. Step-count capping abandoned as the lever. Same session: pre-commit
  hook was broken (`.git/hooks` isn't versioned, nothing had ever caught `cargo fmt --check`
  failures) — fixed, added tracked `scripts/hooks/pre-commit` + `scripts/setup-hooks.sh` (**rerun
  the latter if hooks ever vanish again**, e.g. fresh clone).
- **2026-08-15, §35 (ADR 0041 isolated pilot, first attempt):** mechanism validated, §33 fix engages
  in production state (47-96% of grown rows nonzero after 10 cycles), but the predicted
  episode-space drift materialized and NREM cost was +34.5%, not ≪-cycle as estimated. Not adopted.
  Also fixed a `BindingReplayLoss` metrics-doc bug (documented count/total semantics were
  impossible given how `counted!` actually accumulates).
- **2026-08-15, ADR 0041 implementation** (resumed after a machine lockup): landed
  `BindingArea::forward_tracked`, `PerceptionProcessor::binding_replay_step_batch`,
  `--binding-replay` flag, 2 new metrics (`Metric::COUNT` 100→102). Root-caused a failing test as
  not a gradient bug — grown units don't win k-WTA in a synthetic single-input setup (duty-cycle
  boost can't bridge the init z-score gap); verified via encoder-column surgery. Full detail folded
  into ADR 0041's Validation-plan item 1.
- **2026-08-15, 10x growth campaign complete:** `[9821]×5` = 101.7% of the 9,660 target, 791
  cycles/188 growth events/82 sessions, 2026-07-12→2026-08-15. `mean_output_mse` −74.0%
  first-vs-last-10. Operational note from the completion run: **background launches must use
  `setsid` (+`</dev/null`, log to file) or they die with the launching command** — a launcher-side
  process-group kill wasted one relaunch attempt.
- **2026-08-14, test-coverage round 2:** closed somnus-perception/somnus-memory thin spots, 27 new tests.
  Same session confirmed `gaps.md` §33 (binding weights never trained) is real, not a false alarm.
- **2026-08-13 (second session), test-coverage round 1:** somnus-encoding/whisper_asr/somnus-metrics, 54
  new tests. Found and fixed a rubato-5 contract-drift bug along the way (same silent-drift class as
  §17/§22/§24).
- **2026-08-13, dependency-sweep wrap-up + `somnus` lib/main split:** `somnus` split into `lib.rs` (all
  logic) + a 4-line `main.rs`, unblocking `cargo test -p somnus --lib`; 8 pure functions extracted from
  `run()`'s sleep-trigger/EWMA logic and unit-tested. User chose to stop rather than go deeper into
  `run()` — most of the remaining ~2000 lines is I/O/setup wiring, diminishing returns.
- **2026-08-12, sparse-gather binding rewrite (ADR 0040) deployed live:** 40-59% `OrtRunBinding`
  reduction confirmed on real production traffic, including one cycle catching a live growth event
  with correct re-export. Two unrelated environment blockers fixed along the way (ffmpeg-next/ffmpeg
  9.0 version mismatch; `models/` had silently diverged from its symlink target).
- **2026-08-11, redirect toward the ONNX binding rewrite:** corrected two wrong premises — encoder
  and binding calls are already single fused ORT calls (no "5 separate dispatches" to batch), and
  the binding graph is not region-size-independent. Benchmarked before implementing: the dense
  region-projection matmul is 75-76% of the binding graph's per-call cost at campaign scale.
- **2026-08-04, housekeeping, no code changes:** recreated `memory`/`clips`/`data` symlinks that had
  gone missing outside any recorded session (campaign data intact). Discussed k-WTA scoring
  optimization options, no decision reached — clarified mid-discussion that the actual goal is
  same-hardware scaling, not more hardware.
- **2026-07-16, ADR 0038 prerequisites 1/2 resolved:** encoder-graph scoring cost is small and not
  dominant at current scale; no roadmap item exists past the current honeybee-order campaign.
  Advisor-caught correction: kept the ADR `proposed`/deferred rather than closed — "no roadmap names
  mouse-order" isn't the same claim as "honeybee-order is a decided ceiling."
- **2026-07-15, `somnus-metrics` first real numbers + §27 HNSW rebuild fix:** the new `SleepWarmRebuild`
  metric surfaced a 46-51s/cycle serial HNSW rebuild (~70% of cycle wall-clock) once the warm cap
  hit; switched to `hnsw_rs::parallel_insert_slice`, verified 16.7× speedup on real production data.
- **2026-07-14, `somnus-metrics` two-tier system (ADR 0039) + full-coverage expansion + ADR 0038:**
  built the crate, converted every hand-rolled timer, then a 3-way parallel inventory took `Metric`
  24→67 variants. Same session: research note on the encoder-scoring matmul's inherent
  `O(input_dim×n_units)` cost, three mechanisms surveyed, ADR 0038 written (`proposed`,
  hierarchical/coarse-to-fine WTA).
- **2026-07-13, §25 sparse-aware binding projection:** `BindingArea::forward` materialized a dense
  `[1,n_units]` vector from a ~98%-zero k-WTA activation. Replaced with sparse gather + small
  matmul, numerically identical, **11.4× speedup** isolated benchmark (n=8,655, k=169). Deployed
  live (session 72). **Correction (2026-07-14):** dormant on ORT-active campaigns — `perc_bind_us`
  times `try_bind`, which dispatches to `ort.run_binding`, not `burn`'s path, past the §22
  crossover. Kept deployed (strictly better, costs nothing dormant) for pre-crossover/CPU-only/
  ORT-fallback cases. Also noticed: no training function calls `BindingArea::forward` — its
  weights receive no gradient from any path (this became §33/ADR 0041 later).
- **2026-07-13, §24 two ORT bugs fixed and confirmed live:** (1) binding-side `to_dense_n(N_UNITS)`
  hardcoded baseline 1024 instead of actual grown size, both in `lib.rs` and `onnx_session.rs` —
  fixed to derive from actual data (`8718099`). (2) encoder-side `ENC_DIMS` hardcoded language
  input at 896, stale since ADR 0036 raised LANG_HIDDEN to 1024 — fixed to derive from live
  constants (`debd6c4`). Both regression-tested. Live confirmation: zero ORT failures across
  ~8,000 log lines; `perception_total_ms` 2,158ms/1,457ms (sessions 60/61) vs. 250,099ms fully-
  broken vs. ~6,900-7,200ms pre-ORT baseline — **confirmed ~3-4× real speedup**, retracting ADR
  0037's original "~2.1-2.4×" (measured on the broken path).
- **2026-07-12/13, campaign restarts (5 total, all clean, state snapshotted every cycle):**
  Restart 1 (§21) — `SemanticStore::dump()` leaked a full HNSW dump every cycle forever (461
  orphaned pairs, 11G) because reload permanently disables in-place overwrite; fixed by clearing
  stale dumps before writing (11G→192M). Restart 2 (§22/ADR 0037) — implemented auto-enable ORT
  inference (`should_use_ort_inference`) past the CPU/GPU crossover (~1,230 units/region);
  surfaced two pre-existing bugs in `export_perception_graph.py` (stale LANG_HIDDEN, hardcoded
  N_UNITS/K_ACTIVE) fixed via `infer_dims()`. Restart 3/4 — deployed §24's two bug fixes above.
  Restart 5 — deployed §25's sparse binding projection.
- **Cold-archive note (2026-07-13):** `max_cold_records_effective`=0 (unlimited) all campaign —
  `sysinfo` can't match the btrfs mount, falls back to no cap per ADR 0024's documented behavior
  (not a bug). Growing correctly regardless (append-only ledger by design).
- **2026-07-12, §17/§18/§19 — three growth-mechanism fixes, all tested, all committed:**
  §17 K_ACTIVE now scales with region growth (`scaled_k(n_units)` preserves baseline ratio;
  K_ACTIVE=20 was hardcoded, sparsity would've fallen ~10× over the campaign). §18 optimizer reset
  scoped to the grown region only (was wiping AdamW momentum for the entire model on every growth/
  prune event — burn-optim state is already keyed by ParamId, so the full reset was pure waste;
  deleted). §19 novelty-proportional growth batch size (GROWTH_BATCH_MIN=16/MAX=128 replacing a
  flat 64, scaled by how far the trigger missed threshold via `gist_novelty`/`mse_novelty`).
- **2026-07-12, growth pilot 3/"v2":** validated §17-19 live — mechanism confirmed working (14
  growth events, 4-7 cycle cadence, `k_active` tracked growth correctly, `per_region_mse` 0.0136→
  0.0048). **Recalibration:** mean batch 35.6/event (nearly half the old flat 64, since most
  triggers land at moderate not severe novelty) → **10x campaign re-estimated at ~9.3-11.5h**
  (was ~3.9h), ~2.4-3× longer — not a regression, §19 correctly making growth severity-proportional.
- **2026-07-11, expanded-corpus growth pilot + podcast corpus + doc audit:** added
  `scripts/download_podcasts.sh` (10 shows, 10-90min filter) — 56 episodes/5.6GB in
  `clips/podcasts/`. Pilot 2 (145-clip combined corpus, 75 cycles): **no plateau** — cadence/
  quality indistinguishable from pilot 1 (16 events, `per_region_mse` 0.0135→0.0039, final sizes
  uniform `[2048]×5`) — confirms growth is gate-limited (3-cycle cooldown), not signal-limited.
  Full documentation audit (delegated staleness sweep, applied directly): fixed stale open
  questions in architecture.md, renumbered gaps.md's priority table (`#` had drifted from `##`
  since ~item 7), restored session-state.md's lost deferred-backlog trigger table from git history
  (`03eb9b5`), added missing Phase 14 to roadmap.md, added missing ADR callouts to README.md.
- **2026-07-09, growth-rate pilot campaign (10x scaling, honeybee-order target ~54,000 units,
  corrected from an earlier mouse-order miscalibration):** growth mechanism already fully
  implemented (cascade-correlation-style, ADR 0024 §2c gate, 3-cycle cooldown, GROWTH_BATCH=64) —
  no new code. 75-cycle pilot: 16 growth events (~1/4.7 cycles, gate-limited not signal-limited),
  final sizes `[2048,1984,1984,1984,1984]`, no plateau, `per_region_mse` improved alongside growth,
  21.8s/cycle avg, 1.7GB disk. **Extrapolation: ~633 cycles ≈ ~3.9h for 10x** (later revised up by
  §19, see pilot 3 above).
- **2026-07-09, M2D-CLAP (item 8) — NOT PURSUED, no code:** feasibility research only — no ONNX
  export precedent anywhere (dynamic positional-embedding interpolation, nnAudio STFT frontend —
  same fragility class mn10 was chosen to avoid), non-commercial license. +3 mAP claim judged not
  worth the export/license risk given item 9's precedent (paper-benchmark quality and somnus-pipeline
  quality diverge). Docs-only, engineering-log.md §8, ADR 0046.
- **2026-07-09, EfficientAT mn10 (item 9) — ❌ REJECTED:** A/B (5 sessions × 2 cycles, fresh memory
  both arms): control (CLAP-HTSAT) avg MSE 0.011966 vs. treatment (mn10) 0.030588, **+155.6%
  worse**, 9/10 cycles regressed, no reversals. Default env encoder stays CLAP-HTSAT.
  Implementation kept as an opt-in `mn10_env` Cargo feature (clean, parity-tested, may be useful
  for deployment-size later). engineering-log.md §9, ADR 0046.
- **2026-07-08, DER++ auto-λ Phase 3 — implemented then A/B REJECTED:** EWMA 1/variance weighting
  for `sae_loss`+`wake_loss` combination (`--der-auto-lambda`, default false). A/B: control avg
  0.0487 vs. treatment 0.0847, **+73.8% worse**, 10/10 cycles regressed, no reversals — likely
  early-EWMA-variance noise causing destabilizing weight spikes. Not adopted; flag stays opt-in,
  default false. ADR 0044. Lesson recorded: prefer Bash `run_in_background` over manual
  `nohup ... &` — a stray un-killed background process once collided with a clean rerun writing to
  the same path.
- **2026-07-06:** change-gated video encoder + SVD null-result shipped (`9bba42b`, `b410ad7`).
  High replay ratio (N=1/2/4) validated, **N=4 adopted** (`d5ce2ab`, `f0fa9c6`): -70.9% output_mse
  for +16.6% wall-clock vs N=1; N=2 more efficient (-49.1%/+7.6%) but not chosen. engineering-log.md
  §2, ADR 0043.
- **EfficientAT mn10 implementation detail (2026-07-08, now A/B'd above):** new `Mn10Mel` mel
  frontend (EfficientAT's `AugmentMelSTFT`, Kaldi-style filterbank, parity-tested at 1e-4). Export
  script bakes in only the MobileNetV3 backbone (no mel frontend — avoids reintroducing an STFT op
  into the graph). New `mn10_env` Cargo feature swaps REGION_DIMS[1] 512→960 and all cross-modal
  offset math. **Region-1 MSE is not comparable across arms** (different target distributions,
  CLAP contrastive vs. mn10 raw log-mel) — verdict uses the mean of the other 4 regions instead.

**Earlier baseline (pre-2026-07-06), for reference:**
- DINOv3-ViT-S adopted (drop-in 384-dim replacement for DINOv2-S): visual-region MSE -95.6%,
  `warm_recall_l2` +61.5%, no confound. `ai-docs/a_b_dinov3_findings.md`, engineering-log.md §6,
  ADR 0045.
- Qwen3-0.6B adopted as sole action-decoder LM (ADR 0036, supersedes ADR 0035); Qwen2.5 support
  fully removed. ~22% lower `mean_proj_loss`, coherent generation vs. Qwen2.5's token salad.
  `ai-docs/a_b_qwen3_findings.md`, engineering-log.md §5.
- Whisper ASR panic (intermittent `decode_greedy` "GetElementType is not implemented") mitigated
  not root-caused — `catch_unwind` stops it from killing the ASR thread. Unverified whether it
  fully recovers or downgrades to a per-cycle WARN. gaps.md §40 (open).

## Key files

- `ai-docs/gaps.md` — full backlog with effort/impact/status table (read first for "what's next")
- `ai-docs/decisions/` — ADRs, the canonical record of *why*; `ai-docs/engineering-log.md` — bug
  fixes, pilot numbers, campaign history (trimmed to pointers wherever an ADR covers the decision)
- `ai-docs/a_b_dinov3_findings.md`, `ai-docs/a_b_qwen3_findings.md` — A/B evidence trails
- `scripts/ab_dinov3.sh`, `scripts/ab_qwen3.sh`, `scripts/ab_qwen3_longclips.sh`,
  `scripts/ab_binding_replay.sh`, `scripts/ab_binding_replay_loss_scale.sh`,
  `scripts/ab_video_change_threshold.sh` — A/B harnesses (isolated `--memory-dir` per arm, NDJSON
  metrics)
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
| K | ADR 0003 | DINOv2-S → SigLIP-B (language-aligned visual embeddings, Phase 7+) | — (superseded in spirit by DINOv3-ViT-S, engineering-log.md §6/ADR 0045; SigLIP2-ViT-B/16 tracked separately as §7) |
| L | ADR 0012 | LoRA fine-tuning of Qwen for episode-vector prefix | — (Tier 3) — **confirmed still open 2026-08-15 (engineering-log.md §37):** done 2026-06-25 for Qwen2.5 (ADR 0033), orphaned when Qwen2.5 was dropped (ADR 0036); `train_lora_qwen3.py` exists (committed 2026-06-27, same session as the abandoned KV-cache-decode half) but was never run — `qwen3_lm.onnx` is the corrected-export base model, not a LoRA merge |
| N | ADR 0014 | Visual/audio region grounding (captions, env sound labels) | — (Tier 3) |

Items B, D, G, H (language last-token pooling, full differentiable SAE, learned per-region gate,
activation-frequency pruning) are not in CLAUDE.md's current list — resolved/superseded/dropped at
some point after the 2026-06-14 scan; check `gaps.md` and ADR history if one becomes relevant.
