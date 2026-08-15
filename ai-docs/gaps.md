# Gaps, candidates, and open work

> **Contents:** Output of the 2026-06-27 review round — security advisories, algorithm improvement candidates, encoder/LLM upgrade candidates, crate hygiene. Each item has enough detail to implement or reject without re-researching.
> **When to read:** Starting new work, picking the next item to try.
> **Status:** All items unimplemented as of 2026-06-27 unless noted. Add ✅ and ADR reference when an item ships.

---

## 0. Critical: bincode security advisory ✅ RESOLVED

**RUSTSEC-2025-0141.** Migrated bincode 2.x (archived) → bitcode 0.6 (maintained). Commit 57c87a1. `crates/somnus-memory/src/semantic.rs` updated; WARM_VERSION 1→2. Old bincode sidecars trigger graceful cold rebuild (existing fallback). All tests pass.

---

## 1. Algorithm: BatchTopK SAE ✅ DEPLOYED

**Source:** Bussmann et al., arXiv:2412.06410, Dec 2024. A/B PASS (commit 25d7496); default in all NREM training.

**What:** Batch-level quantile threshold instead of per-sample hard top-k — one scalar τ per batch so the *average* active-feature count equals k; individual samples may use k±δ. Pure L2 reconstruction loss, no L1 (our L1 SAE experiment, ADR 0025 λ=0.1, hurt MSE +27% and was closed — BatchTopK is structurally different, not a retry).

**Implementation:** `KWtaLayer::forward_tracked_batch(batch_topk: bool)`. Snapshot duty-cycle → boosted values → single τ via `select_nth_unstable` on the flattened batch → apply to all. Inference unchanged (per-sample top-k). `--batch-topk` flag, default false.

---

## 2. Algorithm: High replay ratio ✅ ADOPTED (N=4, 2026-07-06)

**Source:** arXiv:2505.12512, May 2025 — replay ratio 4-8× improves consolidation with no degradation.

**Bug found first:** `train_campaign.sh` hardcoded `--nrem-replay-epochs 4` regardless of `EXTRA_ARGS`, so the prior A/B (2026-06-27) compared N=4 vs N=4 — meaningless. Fixed via a `REPLAY_EPOCHS` env var (default 1, matching the CLI default), set explicitly per arm.

**Result:** 5 sessions × 2 cycles, fresh memory both arms, h3h3/shorts corpus. `mean_output_mse` control 0.2379 vs N=4 0.0693 (**-70.9%**), consistent, no reversals, all 5 regions improve monotonically both arms.

**N=2 arm (cost/benefit knee):**

| N | avg mean_output_mse | Δ vs N=1 | wall-clock Δ |
|---|---|---|---|
| 1 | 0.2379 | — | — |
| 2 | 0.1211 | -49.1% | +7.6% |
| 4 | 0.0693 | -70.9% | +16.6% |

N=2 is more cost-efficient (~6.5% error reduction/1% wall-clock vs N=4's ~4.3%), but **N=4 adopted** — the absolute wall-clock tax (+16.6%) was judged small enough to pay for the larger absolute gain. CLI default and `train_campaign.sh` both flipped 1→4.

---

## 3. Algorithm: DER++ auto-λ + stale-output blocking

**Source:** Kobayashi 2025, arXiv:2504.20932.

1. **Stale-output blocking ✅ IMPLEMENTED** (65ceaf7): per-batch inference + cosine check (threshold 0.90) skips buffered samples whose stored outputs drifted under distribution shift, before training. Tracked in `SleepStats.stale_traces_skipped`.
2. **Auto-λ ❌ REJECTED** (2026-07-08, opt-in only, default off): `fused_replay_wake_step_batch` combined `sae_loss`+`wake_loss` as an unweighted sum (implicit λ=1). `--der-auto-lambda` (f2fd185) instead tracks an EWMA (α=0.05) mean/variance of `wake_loss` and weights it by `1/variance` (clamped [0.1, 10.0]). **A/B FAIL:** control avg 0.0487 vs treatment 0.0847 (**+73.8%**), regressed 10/10 cycles (+16% to +937%), no reversals — likely early-EWMA-variance noise causing destabilizing weight spikes. Not adopted; test `der_auto_lambda_stays_finite_and_loss_decreases` stays as a finiteness guard only. Not planned to revisit without a different variance estimator (windowed, or a min-sample floor).

**Implementation:** `PerceptionProcessor.infer_episode_outputs()` infers current predictions; NREM loop compares vs. stored regionals via cosine, filters, trains/projects the survivors.

---

## 4. Algorithm: LISTA Top-K unrolled encoder

**Source:** arXiv:2511.10575, Nov 2025. 3-5 unrolled ISTA iterations (learned step size + straight-through Top-K gate) instead of single-shot k-WTA — converges to sparser, better-reconstructed codes at the same k. All-5-regions change. Each step is matmul+bias+TopK+multiply — standard ONNX ops, static graph, no CUDA. **Effort: Medium.** Deferred.

---

## 5. LLM: Qwen3-0.6B ✅ ADOPTED (2026-07-05, ADR 0036) — Qwen2.5 dropped

**Source:** Qwen team, April 2025, `Qwen/Qwen3-0.6B`. 600M params, hidden 1024 (Qwen2.5-0.5B: 896), MMLU 70.0 vs ~47.

Sole action-decoder LM; `LM_HIDDEN`/`LANG_HIDDEN`/`LANG_DIM`/`EOS_TOKEN_ID` hardcoded to Qwen3, cross-modal projection fixed at 2304. Detail: `a_b_qwen3_findings.md`, ADR 0036.

**Path to adoption:** an initial A/B (2026-06-28) was invalid (a hardcoded 896-wide embed table sliced at stride 1024, corrupting embeddings both arms). Fixing it surfaced two more bugs (Qwen3 loaded into the wrong model class; ~6 more hardcoded language dims disagreeing). Re-run on stock 4-clip corpus was inconclusive (too little speech). Re-run on speech-dense `clips/long/`: Qwen3 beat Qwen2.5 on `mean_proj_loss` by ~22%, no reversals; a direct `generate()` comparison confirmed coherent English vs. Qwen2.5's token garbage. Adopted on both.

**Remaining (non-critical), corrected 2026-08-15 (gaps.md §37) — this note was stale, written before ADR 0033/0034 shipped both things for Qwen2.5:** LoRA fine-tuning (ADR 0033) and KV-cache split generation (ADR 0034) both shipped and worked for Qwen2.5, but neither carried over to Qwen3 when ADR 0036 dropped Qwen2.5 — **both are currently incomplete for the model actually in production.** LoRA: `train_lora_qwen3.py` exists (committed 2026-06-27) but was never run — `qwen3_lm.onnx` is an uncorrected-then-fixed base export, not LoRA-tuned. KV-cache: only `qwen3_lm_prefill_kv.onnx` was exported (same commit); `qwen3_lm_decode_kv.onnx` is missing, so `ActionDecoder::new`'s both-files check fails and every `generate()` call falls back to the slow O(n²) CPU path — the 7.8× ADR 0034 speedup is not active. Neither gap is blocking (generation works, just slower/less grounded than it could be); full detail in §37. Whisper `decode_greedy` intermittent ORT panic ("GetElementType is not implemented") — ✅ mitigated 2026-07-05 via `catch_unwind` in the ASR thread; **unverified** whether the underlying `Session` gets poisoned (degrading to "WARN + drop every cycle" instead of full recovery). Root cause undiagnosed, **queued 2026-08-14**: first instrument to settle the poisoning question, then capture the exact triggering input (intermittent, input-correlated) before theorizing about ORT internals.

**Metrics per cycle:** `output_mse`, `p_lang`, `proj_loss`, regional activation stats, episode counts.

---

## 6. Visual encoder: DINOv3-ViT-S ✅ VALIDATED & ADOPTED (2026-07-05)

**Source:** Meta, Aug 2025, arXiv:2508.10104. ~21M params (= DINOv2-S), output dim 384 (true drop-in). ImageNet linear probe 81.40% vs DINOv2-S 81.1%.

**MIGraphX fix:** the earlier "CPU backend can't compile DINOv3's attention graph" conclusion was wrong — misread error. Root cause: `torch.onnx.export` wrote external-data format (weights in a sibling `.data` file); the MIGraphX EP resolves that path relative to **process CWD**, not the model's directory, so running outside `models/` failed with an opaque error. Fixed: `export_models.py`'s `export_dinov3()` re-packs with embedded weights (matches DINOv2's layout), no CWD dependency.

**Shape difference (not a bug):** DINOv3-ViT-S/16 outputs `[1,201,384]` (CLS + 4 register tokens + 196 patches) vs DINOv2-S/14's `[1,257,384]` (CLS + 256 patches). `VideoEncoder` now tracks `prefix_tokens`/`total_tokens` per model instead of hardcoding "skip 1, pool 256."

**A/B (2026-07-05, `a_b_dinov3_findings.md`):** visual-region MSE -95.6% (2.183→0.097), `warm_recall_l2` +61.5%, no change to unrelated regions — clean, large, correctly-localized win. Already the active encoder.

---

## 7. Visual encoder: SigLIP2-ViT-B/16

**Source:** Google DeepMind, Feb 2025, arXiv:2502.14786. 86M params, ImageNet zero-shot 79.1% vs SigLIP 76.7%. Language-grounded (CLIP-style) vs DINOv2's supervised-probe-only features. 768-dim (up from 384) — cold-starts visual region k-WTA input + binding projection. MIGraphX likely fine (standard ViT attention). This is deferred item K (Phase 7+) — raise when cross-modal grounding (item N) becomes a target. **Effort: Medium.**

---

## 8. Audio env encoder: M2D-CLAP ⏸️ NOT PURSUED (2026-07-09)

**Source:** NTT, March 2025, arXiv:2503.22104, `nttcslab/m2d`. ~86M (ViT-B), AudioSet linear-probe mAP 49.0 vs CLAP-HTSAT ~45-46 (+3 mAP). 768-dim (up from 512).

**Feasibility research only, decision not pursued:** weights available, no gate. But no ONNX export precedent exists anywhere (dynamic positional-embedding interpolation, `nnAudio` STFT frontend — same fragility class mn10/item 9 was chosen to avoid); license is NTT's non-commercial custom "Evaluation" agreement, not MIT/Apache. Expected gain small relative to that risk — item 9's A/B is a concrete precedent that paper-benchmark quality and somnus-pipeline quality diverge here (mn10 had no quality claim and still regressed MSE +155.6% while a benchmark-adjacent metric moved the "wrong" way). Not implemented. Revisit only if an ONNX export appears elsewhere or CLAP-HTSAT becomes a bottleneck worth the risk.

---

## 9. Audio env encoder: EfficientAT-mn10 ❌ REJECTED (2026-07-09)

**Source:** `fschmid56/EfficientAT`. ~5-20M params (MobileNetV3, width_mult=1.0), pure CNN, no attention/ScatterElements.

**Implementation (kept, opt-in):** `Mn10Mel` (`somnus-encoding/src/preprocess/mel.rs`) — Rust port of `AugmentMelSTFT`'s Kaldi-style filterbank, parity-tested at 1e-4. `export_efficientat.py` exports only the MobileNetV3 backbone (mel frontend excluded to avoid reintroducing an STFT op), verified rel L2 ~9e-7, zero STFT ops. New opt-in `mn10_env` Cargo feature: `REGION_DIMS[1]` 512→960, env encoder session/mel/rate (CLAP 48kHz vs mn10 32kHz).

**A/B (5 sessions × 2 cycles, fresh memory, h3h3/shorts):** region-1 MSE isn't a valid cross-arm comparison (different target distributions — CLAP contrastive/unit-norm vs mn10 raw log-mel). Verdict from the other 4 regions (identical architecture both arms): control (CLAP-HTSAT) avg 0.011966 vs treatment (mn10) 0.030588, **+155.6% worse**, 9/10 cycles regressed, no reversals. `warm_recall_l2`/`gist_recall_rate` moved in mn10's favor (+52.6%/+30.7%) but don't override a 2.5× MSE regression — likely a scale-distortion artifact.

**Decision: REJECT.** Default stays CLAP-HTSAT. `mn10_env` kept as opt-in (clean, parity-tested, may suit deployment-size constraints later, or if M2D-CLAP also fails).

---

## 10. Language encoder: dedicated text embedding model

Current input: Qwen2.5-0.5B embed table (896-dim mean-pooled). Key constraint: `--lang-use-lm`'s LM hidden states are the training signal behind the 4× p_lang improvement — switching the input encoder and switching the LM dim must move together.

| Model | Params | Dim | ONNX | MTEB | Notes |
|---|---|---|---|---|---|
| nomic-embed-text-v1.5 | 137M | 768 (Matryoshka) | confirmed | 62.39 | 8192-token context |
| modernbert-embed-base | 149M | 768 (Matryoshka→256) | confirmed (raw) | — | ModernBERT backbone |
| BGE-small-en-v1.5 | 33M | 384 | confirmed | low-60s | smallest |

**Recommended:** only migrate alongside a future LM upgrade (Qwen3's embed table is 1024-dim, avoiding a mismatched intermediate state). Deferred.

---

## 11. LLM: Gemma 3 1B IT

**Source:** Google DeepMind, March 2025, `onnx-community/gemma-3-1b-it-ONNX`. 1B, better benchmarks than Qwen2.5-0.5B. **Caveat:** community ONNX uses onnxruntime-genai's opaque KV-cache, not ADR 0034's static prefill/decode split — reproducing that split is high effort; verify ScatterElements absence before committing to GPU. **Effort: Medium-High.** Lower priority than Qwen3. Deferred.

---

## 12. Null result: episodic binding survey

All 2025-2026 binding improvements outperforming 4-head cross-attention + GRU use Mamba/SSM variants (arXiv:2601.21648, arXiv:2509.00677), all requiring `selective_scan_fn` — no ONNX equivalent, unsupported by MIGraphX. **Conclusion:** current binding is state-of-art under the ORT/MIGraphX constraint. Not pursued. One extractable idea, unvalidated: a scalar learned gate per modality per GRU step (sigmoid+multiply).

---

## 13. Change-gated video encoder inference ✅ DEPLOYED (2026-07-06)

Mirrors the biggest efficiency mechanism real neural tissue uses (spikes only on change). `VideoFrame::mean_abs_diff` (`somnus-sensory/src/frame.rs`) computes normalized mean-abs-diff between consecutive candidate frames (returns 1.0/"changed" on length mismatch — covers Mjpeg/resolution changes). The video encoder task skips `enc.encode()` below `--video-change-threshold`; a `--video-change-max-skip` heartbeat (default 30) forces periodic re-encode. Default threshold 0.0 (disabled, opt-in). Audio encoders left ungated (VAD-style silence detection is a different problem, out of scope). Not yet A/B'd for real wake-path savings — mechanism deployed and unit-tested, skip rate/downstream binding effect unmeasured.

---

## 14. Null result: SVD low-rank check (2026-07-06)

Checked whether somnus's 21 growable weight matrices (region encoders, cross-modal projection, binding projections, Q/K/V, GRU, temporal projection) have learned low effective rank (would make LoRA-style compression near-free). New bin `svd_rank_check` dumps every 2-D matrix; `scripts/svd_rank_check.py` (numpy) runs SVD per matrix. Checked against the most-trained checkpoint on disk (`[1664,1600,1600,1600,1600]`).

**Result: null.** Rank needed for 95% energy sits at 60-82% of full rank (mean 0.772) across all 21 matrices — too close to `min(n,m)` for `(n+m)·r` low-rank parameterization to undercut `n·m`. Effectively full-rank. **Do not pursue.** Caveat: single checkpoint; re-check if a matrix's dims grow much larger relative to EPISODE_DIM/D_PROJ, though the consistency across all 21 (no outliers) makes a different result unlikely.

---

## 15. Crate hygiene

- **Dead deps:** ✅ done — candle-core/candle-nn already removed.
- **usearch 2.25.3** (vs `hnsw_rs`): built-in INT8/INT4 quantization, AVX-512/NEON SIMD, concurrent inserts, recall@10 at parity with hnswlib/FAISS. Migration cost: rewrite `somnus-memory`'s HNSW layer. Raise when warm-tier footprint becomes a concern.
- **burn-ndarray**: deprecation path toward `burn-flex` in 0.22 (~late summer 2026). Watch, no action yet.
- **ffmpeg-the-third**: actively maintained `ffmpeg-next` fork, may have surpassed it. Check next time the media pipeline is touched.

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

**Not empirically A/B'd** — the fix corrects a design/implementation mismatch a priori; no pilot rerun to confirm it helps (vs. just being correct) at multi-x growth.

---

## 18. Scope optimizer reset to grown region only ✅ DONE (2026-07-12)

**Problem:** growth/prune reinitialized AdamW for the *entire* model, discarding every other region's valid momentum ~135× projected over the campaign.

**Investigated burn-optim's source** before changing anything: momentum is a `HashMap<ParamId, AdaptorRecord>`; a param missing from the map (as the fresh `ParamId::new()` growth/pruning already assigns to resized tensors) automatically starts fresh — no full reset needed.

**Fix:** deleted the full-reset line from all three call sites; nothing else changed. Confirmed unrelated to `refresh_autodiff_state` (a separate, legitimate optimizer-reset mechanism for NREM graph-accumulation slowdown — untouched). Full suite passes. **Not empirically A/B'd** — follows directly from source, but no pilot measured whether preserved momentum improves convergence in practice (no visible harm from the old behavior either).

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

## 22. Auto-enable ORT/MIGraphX perception inference past the CPU/GPU crossover ✅ DONE (2026-07-12)

**User question** ("onnx=off — any impact?") led to real numbers from `metrics.jsonl`: `perception_total_ms` climbing with region size (9.1s at 1,986 units → 16.1s at 6,043), already over half of cycle wall-clock, past ADR 0023's ~1,230-units/region crossover — the opt-in `--onnx-models-dir` flag was costing real wall-clock for no reason.

**Decision (ADR 0037):** auto-enable ORT/MIGraphX once any region crosses `--onnx-crossover-units` (default 1,230) instead of requiring the flag. One-directional, sticky failure handling (WARN once, stay CPU for process life), `--force-cpu-inference` opt-out. New tested `should_use_ort_inference()`.

**Surfaced two pre-existing bugs in `export_perception_graph.py`** (neither caused by this change, just never exercised past the crossover since Qwen3): stale hardcoded language dim (896 vs Qwen3's 1024), and hardcoded N_UNITS/K_ACTIVE that couldn't survive growth; `--validate-numpy`'s synthetic generator never created `W_cross_bias`. Fixed by deriving dims from actual exported weight shapes (`infer_dims()`) and adding the missing bias generator. Verified: `--validate-numpy` passes (ep=6.64e-09), a real export against a campaign-weight copy (6,397 units/region) loaded through MIGraphX, `perception_total_ms=7,735` vs. the live CPU-only campaign's 16,051 at comparable size — **~2.1× faster** (later invalidated, see §24 correction below).

**Deferred:** eliminating the Python export step via native Rust ONNX construction — no mature crate exists, real hand-rolled-protobuf effort. Tracked as §23.

> **⚠️ CORRECTION (§24): the "~2.1×/2.4×" speedup above is invalid** — `to_dense_n(N_UNITS)` used the hardcoded baseline constant, not each region's grown size, so every ORT call silently failed and fell back to burn/CPU for the campaign's entire ORT-active history. The verification run and the campaign's own post-restart cycles were almost certainly both hitting silent fallback — no genuine accelerated ORT call was ever observed at this point. Fix and real numbers: §24.

---

## 23. Future: native Rust ONNX graph construction 🔮 SCOPED, NOT STARTED

Raised 2026-07-12 while fixing §22's export-script bugs. Would eliminate the Python3+onnx+onnxruntime subprocess round-trip and solve dynamic shapes for free (Rust already knows every shape exactly). **Why not now:** no mature "build ONNX graphs from Rust" crate — would mean hand-constructing `ModelProto`/`GraphProto`/`NodeProto` replicating everything the Python script does (GRU cell weight transpose, multi-head attention, the MIGraphX ScatterElements two-graph-split workaround). Real correctness risk, genuinely open-ended — a dedicated session, not a fast-follow. **Wins if done:** removes the Python runtime dependency, the per-cycle subprocess+export overhead, and the weights-manifest.json round-trip. **Trigger:** none set — raise when export-script maintenance recurs or profiling shows the subprocess overhead is itself meaningful.

---

## 24. ORT sparse→dense conversion hardcodes baseline N_UNITS ✅ FIXED (2026-07-13)

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

**Correction (2026-07-14) — currently dormant on the live campaign.** Sessions 72-75 showed no `perception_total_ms`/`nrem_wall_ms` drop. `perc_bind_us` times the whole `try_bind` call, which — once ORT is active past the §22 crossover — dispatches to `ort.run_binding`, not `burn`'s path at all. `BindingArea::forward` only fires from `try_bind`'s non-ORT branch, `settle_episode`'s ORT-failure fallback, and the `binding_only` validation helper — none active on this healthy-ORT campaign. **Also found while tracing this:** no training function calls `BindingArea::forward` either — its weights receive no gradient from any path in the codebase. Investigated 2026-08-14, became §33.

**Net effect:** real, verified, 11.4× win wherever `burn`'s binding path actually runs (pre-crossover, CPU-only/no-MIGraphX, ORT-failure fallback) — kept deployed (strictly better, free when dormant), but not moving this campaign's current wall-clock. Lesson: check which backend owns a profiled call site before attributing cost to `burn`.

**Queued next (from the same discussion):** the k-WTA encoder itself is inherently O(input_dim×n_units) and can't be sparsified the same way — the real lever for approaching mouse-order scale. Written up as a research note (`sparse-coding.md` §"Encoder-side scaling") + **ADR 0038** (`proposed`), which picks hierarchical/coarse-to-fine WTA over MoE-routing and approximate sampling (preserves exact top-k within blocks, degrades gracefully, may reuse HNSW for the coarse stage). Three prerequisites before implementation: (1) benchmark actual scoring cost both `burn`/ORT paths, (2) confirm this matters at honeybee vs. next-order-of-magnitude scale, (3) numerical validation that block-local competition doesn't break growth/duty-cycle/binding assumptions. Not started.

---

## §26 — Unified two-tier metrics instrumentation (`somnus-metrics`, ADR 0039)

Raised while scoping ADR 0038's Phase 0. No shared timing mechanism existed — every call site hand-rolled `Instant::now()`/`.elapsed()`.

1. **Fine-grained gated counters:** new crate `somnus-metrics` — `Metric` enum (dense-array-over-enum, mirrors `RegionId::as_index()`), atomic `Registry`, `timed!`/`counted!` macros behind `fine-metrics` (default off, zero-cost when disabled — bare passthrough). Piloted on `somnus-perception`'s 6 ingest phases + new `KwtaScoreMatmul`/`KwtaSelectTopK` sub-breakdown counters (reopens ADR 0038's benchmarking prerequisite via live traffic).
2. **Full conversion — "one way of doing metrics."** Added an unconditional Core tier (`timed_core!`/`counted_core!`) and converted every hand-rolled site: `somnus-perception` (6 phases), `sleep.rs` (11 phases — 8 via macro block-wrap, 3 structurally-entangled spans via direct `METRICS.record()`, documented exception), `somnus-encoding` (3 encoder timers), `main.rs` (4 sites). Found/fixed 2 duplicate-measurement bugs (main.rs re-timing calls `somnus-encoding` already timed). All pre-existing JSON fields unchanged; new `"fine_metrics"` key carries the registry snapshot.

**Verified:** both feature states clean (5 new `somnus-metrics` tests). Two pilot runs cross-checked old-field vs. registry values (within truncation noise), confirmed `kwta_score_matmul` dominates `kwta_select_topk` ~124-165× at baseline scale (early evidence for ADR 0038, not yet campaign scale).

**Follow-up (same session): full-coverage expansion.** 3-way parallel inventory of every remaining uninstrumented op → `Metric` 24→67 variants: somnus-perception +25 (binding forward, ORT run_encoder/run_binding, fused-replay sub-phases, ORT lifecycle, growth/prune, REM self-probes, EMA sub-steps, refresh_autodiff_state, wake_worker::train_step), somnus-memory +10 (cold open/truncate/evict, semantic warm-load, recollection promote, snapshot save/load), somnus-sensory +1 (aggregate `MediaFileDecode`), somnus binary +7 (action-decoder init/generate, gain-worker, ASR decode, audio resample, language embedding, record_episode). Two new patterns (ADR 0039): private-`_inner`-method split, immediately-invoked-closure wrap. Verified incrementally; live pilot recorded 40 metrics simultaneously, cross-thread ones safe, untriggered conditions correctly stayed zero.

**Follow-up (2026-07-15): value/count metrics.** All 67 were duration-based — added 7 `counted_core!`/`counted!` sites (COUNT 67→74): `region_n_units`/`region_k_active` (per growth/prune event), `cold_archive_evicted_records`, `video_frames_encoded`/`_skipped`, `refresh_language_cache_run`/`_skipped`. Verified; 4/7 fired in a 2-session pilot, 3 conditional ones correctly absent.

**Follow-up (2026-07-15): operational health.** GPU/ORT degradation, resource exhaustion, failure visibility ahead of a long unattended run. COUNT 74→99: `ort_encoder_fallback`/`ort_binding_fallback`/`ort_session_init_failed`/`ort_session_rebuild_failed`; `process_rss_bytes`/`memory_dir_disk_free_bytes` (per-cycle sampling); `growth_batch_size`; 13 grouped failure counters covering every recurring warn!/error! site (`perception_ingest_error` flagged highest-value — means real data loss); disk I/O byte counters (`semantic_store_dump`/`_bytes` found `dump()` itself untimed — fixed in passing). The one gap `somnus-metrics` can't capture (the MIGraphX destructor-race segfault, exit 139, happens during teardown after metrics write) now recorded shell-side: `train_campaign.sh` appends `segfault_restarts.jsonl`, displayed by `campaign_status.py`. Verified; live pilot confirmed 7 new metrics with plausible values (RSS ~11GB, disk-free ~916GB).

**Follow-up (2026-07-15): correctness audit.** Verified prior rounds sound; fixed 2 gaps: added `training_nan_detected` (no prior NaN/Inf check before writing loss/MSE to JSON, COUNT 99→100); added `summarize_failure_counters()` to `campaign_status.py`'s health section (round-2 counters were recorded but never displayed). Verified both feature states + a hand-crafted fixture.

---

## §27 — Parallelized warm-store HNSW rebuild

The new `SleepWarmRebuild` Core metric (§26) immediately surfaced a large invisible cost: once the warm cap is hit (every cycle, per §21), `rebuild_from_recent` does a full from-scratch serial HNSW rebuild. Live data: **46-51s/cycle at ~145-148k warm records — ~70% of total cycle wall-clock.**

`hnsw_rs` 0.3.4 has no delete API (insert-only), so a full rebuild is the only way to enforce "most recent N" eviction without a crate swap or a tombstone scheme (not pursued). Cheap fix: `hnsw_rs::parallel_insert_slice` (rayon, already a workspace dep) replaces the serial loop — same graph/eviction semantics, spread across cores.

**Verified on real production data:** benchmarked through the real metric wrapper against a copy of the campaign's actual cold archive (148,447 records), `git stash`/`pop` old-vs-new: 47.6s (matches live campaign) → 2.85-2.94s — **~16.7× speedup**. New `somnus-memory/examples/bench_warm_rebuild.rs` is a reusable regression harness. Build/clippy/test clean. Not yet redeployed as of this writeup — next campaign resume picks it up automatically via `resume_growth_campaign.sh`'s rebuild-first default.

---

## §28 — ADR 0038 prerequisite 1: exported encoder graph + live scoring cost

**The exported ONNX encoder graph has the identical dense cost** — one `MatMul` per region, no top-k/gating before it, same `O(d_input×n_units)` as `burn`'s path. Switching backends didn't change the scaling law.

**But live cost is small and not dominant.** `OrtRunEncoder` (Core metric): ~0.85-1.8ms/call (mean ~1.28ms, 1,585 calls) at ~9,400-9,493 units/region. `OrtRunBinding` costs 2-3× more per call at the same scale. Encoder+binding together are 28-73% of `perception_total_ms` (wide variance likely from ASR/video decode contention). **Correction (§30, 2026-08-12):** the "region-count-independent GRU+attention" framing undercounted — a dense region-projection matmul was actually 75-76% of `OrtRunBinding`'s cost, since fixed (54-59% reduction).

**Bearing on ADR 0038:** evidence *for* "ORT/GPU already absorbs wasted FLOPs" at current scale — 1.28ms doesn't need rescuing in a ~1-3s wake loop. Doesn't resolve whether that holds after another 10x (a mouse-order jump would scale the encoder ~10x to ~10-18ms/call while binding stays flat). Don't implement hierarchical WTA on this alone; prerequisite 2 is next. No code changes.

---

## §29 — ADR 0038 prerequisite 2: does this matter at the targeted scale?

**Stated target is honeybee-order, not mouse-order** (§16: ~9,660 units/region, explicitly not the ~400x mouse jump). No roadmap item exists past the current campaign. Combined with §28 (encoder cost small, not dominant at honeybee scale): **prerequisite 2 resolves negative — hierarchical WTA isn't worth building for the campaign actually running.**

**"Doesn't matter yet," not "wrong direction" — ADR stays `proposed`, not closed** (advisor-caught correction): "no roadmap names mouse-order" ≠ "honeybee-order is a decided ceiling" — `brain-scale-comparison.md` frames the whole project around eventually closing the scale gap via exactly this kind of technique. **Trigger for revisiting:** campaign target moves past honeybee-order, or the encoder/binding cost balance shifts back toward the encoder mattering. Prerequisite 3 (numerical validation) stays open regardless, needed before any implementation. No code changes.

---

## §30 — Sparse-gather binding projection (ONNX/MIGraphX), ADR 0040 — correction to §28/§29

**Correction:** `build_binding_graph` also ran each region's `sp_i[1,n_units]@W_region_i[n_units,D_PROJ]` — a dense matmul that *does* scale with n_units and (per the benchmark below) was the dominant cost inside `OrtRunBinding`, not the GRU+attention §28/§29 attributed it to.

**Implemented the sparse-gather rewrite** (porting §25's burn/CPU fix to ONNX/MIGraphX): `Gather(W_region_i, active_idx_i, axis=0)` + `[1,k]@[k,D_PROJ]` matmul, mathematically identical, reads ~k/n_units of the weight matrix instead of all of it.

**MIGraphX probe: PASS** — `Gather` fed by a runtime int64 index at campaign scale (n=9,660,k=190), max abs diff 1.49e-07. **Found during implementation:** a `Reshape`+matmul combo crashed MIGraphX's `simplify_reshapes` pass (a MIGraphX compiler bug, unrelated to the Gather question) — fixed by declaring the `[1,k]` input directly, no Reshape node.

**Numerical parity: exact** (~1e-8 vs. the deliberately-unrewritten dense `numpy_binding` reference), both CPU and MIGraphX EP, baseline and campaign scale.

**Measured win** (`bench_binding_projection.py`, real MIGraphX EP, n_units=9,660, k_active=189, two reruns): full (dense) 0.856/0.853ms/call → gather 0.393/0.350ms/call — **54-59% reduction** (2.2-2.4× speedup), stable across reruns. This also answers §29's trigger condition (binding got cheaper) — if anything pushes further away from hierarchical-WTA urgency.

**Rust contract change:** `SparseActivation::to_exact_k(k)` (new, pads/truncates to exactly k — covers the `lang_zero` and dense→sparse-tie edge cases). `run_binding` now takes `&[SparseActivation; 5]` directly; a per-region `k_active: [usize;5]` loads once from `weights_manifest.json` (never re-derived from n_units in Rust or Python — the §17/§22/§24 drift class). All 3 `run_binding` call sites in `lib.rs` updated; `binding_only` (burn-only, never ORT) untouched.

**Verified:** affected-crate build/test/clippy/fmt clean (whole-workspace build blocked by an unrelated pre-existing ffmpeg-next mismatch). `ort_grown_region.rs` rewritten for the new fixed-k contract; both integration tests correctly skip against the pre-rewrite on-disk artifact.

**Deployed live same session.** 2-cycle verification leg (session 79): clean rebuild, zero ORT fallback, real **40-59% reduction** in `OrtRunBinding` per-call cost on production traffic (2.15-2.83ms→1.17-1.30ms), including one cycle catching a live growth event with correct re-export. Two unrelated environment blockers fixed along the way (ffmpeg-next/system-ffmpeg version mismatch; `models/` diverged from its symlink target) — neither caused by this change. Full detail: ADR 0040.

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

## §33 — BindingArea weights never trained — confirmed; decided (b), implemented, pilots FAIL (2026-08-14/15)

**Confirmed 2026-08-14:** the binding area's ~1.72M params (5 region projections, W_q/W_k/W_v, GRU, temporal_proj) have never received a gradient — since ADR 0008. All training paths (`replay_step*`, `wake_reconstruct_step*`, `fused_replay_wake_step_batch`, ADR 0012 proj_loss) consume episodes as stored `Tensor::from_data` — the graph stops at the episode boundary; only inference call sites (`try_bind`, `settle_episode`, `binding_only`) ever call `binding.forward`. ADR 0008 anticipated this ("will be revisited in Phase 6") — the revisit never happened; corrected in place.

**Concrete consequence:** `grow_region`/`grow_region_seeded` append zero-init rows to the binding projection with a "gradients will shape them over time" comment — since no gradients ever flow, grown units' binding rows stay exactly zero forever. At 9.3× growth, ~89% of each region's units contribute to SAE reconstruction but nothing to episodes/memory/recall. Campaign stayed healthy because encoders/decoders/heads adapt around the fixed random binder — `per_region_mse` doesn't gate on binding.

**Options:** (a) accept/document (free, gives up ADR 0008's whole point). **(b) live-episode head training** — recompute episodes live via burn binding during NREM, train heads against regional targets on them, gradients flow into binding. (c) temporal prediction loss (GRU-specific, complements b).

**Decision (2026-08-15): (b)**, with (c) sequenced as a separate follow-up. **ADR 0041**: opt-in `--binding-replay` NREM pass; deliberately does NOT train the GRU (the `None`-context pass excludes it — that's (c)'s job).

**Implemented & verified (`4f7aa1f`):** `forward_tracked` ≡ `forward(None)`, stateless; grown units' zero-init rows move once active (in-crate test forces selection — grown units don't win k-WTA in any *synthetic* setup, fresh-init 0.01-scale encoder columns vs. Kaiming base, but break in via homeostatic rotation on production timescales); public-API test pins never-active rows staying exactly zero.

**Isolated A/B pilot (§35, 2026-08-15):** mechanism validated — §33 fix engages in production state (47-96% of grown rows nonzero in 10 cycles vs. 0 control) — but the predicted episode-drift recall regression materialized and NREM cost rose 34.5%. **Not adopted.**

**Repeat pilot at step-budget cap N=1 (§36, 2026-08-15):** wall-clock did improve (−24.8%) but quality got *worse* — 10× same-cycle MSE on fired cycles, plus a new salience/settle cascade starving 6/20 cycles of replay traces. **Also not adopted.** Step-count capping abandoned as the lever; three untried candidates queued (step magnitude, fusing into the existing optimizer step, decoupling the salience/settle cascade) — **no path chosen, needs a decision.**

---

## §34 — The 10x growth campaign is complete (2026-08-15)

**Target reached:** all 5 regions at **9,821 units = 101.7%** of the 9,660 goal, 791 logged cycles / 188 growth events (mean batch 46.8, cadence 1/4.21) over 82 sessions, `[1024]×5`→honeybee-scale — roughly half the pilot-3-projected cycle budget (real batches ran larger than the pilot's 35.6).

**Final leg:** the prior session's last-expected run (session 80) was interrupted by a machine shutdown mid-cycle-8 at `[9645]×5` (15 short). One aborted relaunch (session 81, killed by a launcher process-group timeout, harmless/snapshot-consistent), then session 82 ran clean to completion: 9645→9675(c5, target crossed)→9769(c8)→9821(c12). ONNX artifacts re-exported at final size by the session's own rebuild.

**Health:** 0 panics, 0 segfault restarts, 0 ORT fallbacks (final session), 7 `misc_warning` lifetime. Final-cycle timing at 9,821/region: perception 1.6s, NREM 7.4s, REM 9.4s, snapshot 2.9s, spill 5.4s.

**Quality:** `mean_output_mse` first-10 vs last-10: 0.0430→0.0112 (**−74.0%**); latest per-region MSE 0.0001-0.0019. `gist_recall_rate` 0.562, `within_session_gist_rate` 0.820, `warm_recall_l2` 0.381. Warm 151,725 (816M) / cold 158,534 (101M) records. Eval harness ran automatically on exit — all charts regenerated.

**Unblocks:** §33's binding-weights decision (options a/b/c), explicitly sequenced behind campaign completion.

---

## §35 — ADR 0041 isolated A/B pilot: mechanism validated, §33 fix engages, drift materializes (2026-08-15)

**Setup** (`ab_binding_replay.sh`): trained production state copied per arm (reflink), 1 session × 10 cycles, campaign corpus, both arms `fine-metrics`, sequential. Control: no flag. Treatment: `--binding-replay`.

**§33 fix engages in production state.** Grown units DO win k-WTA there (unlike synthetic setups — 791 cycles of duty-cycle homeostasis plus corpus diversity rotate them in). Control: grown_nz = 0/8,901 all regions. Treatment: **47-96% of grown rows nonzero** (grown_max 0.04-0.14 vs base mean ~0.19) — the campaign's entire grown product started entering episodes within 10 cycles.

**Mechanism:** `binding_replay_loss` 0.141→0.046 (−67.5%); `nrem_binding_replay` 28 steps @ mean 1.44s/step; control fired neither; 0 NaN; both clean exits.

**Predicted consequence materialized — episode-space drift.** `warm_recall_l2` rises progressively in treatment (0.44→0.89) while control improves (1.29→0.31-0.44); cross-session `gist_recall_rate` collapses to 0.23-0.45 (falling) while control reaches 0.90-0.99; `within_session_gist_rate` improves +13.3% (same-generation recall fine) — the mixed-generation-store fingerprint ADR 0041 named as its flagged risk, not a broken mechanism. The named mitigation (warm-store rebuild from cold) can't be tested in 10 cycles (eviction only starts ~cycle 8 at this ingest rate).

**Cost: the ≪-cycle estimate didn't hold.** 1.44s×~2.8 steps/cycle = +34.5% NREM wall-clock, displacing `traces_replayed` −30%, `blend_count` −64%. Per the ADR's own invalidation clause, a cheaper/fused step variant becomes an adoption prerequisite, not an optimization.

**Other deviations:** growth diverged (10,032 treatment vs 9,925 control), `mean_salience_wake` +34%, `settle_delta_mean` +77%, `mean_output_mse_wake` −21.5% (treatment better during wake), NREM `mean_output_mse` +46.1%, `mean_proj_loss` +24.8%. n=1/arm.

**Verdict:** binding learns ✓ (§33 fix engages beyond the ask), no-regression ✗ (recall regresses via the predicted drift channel). **Not adopted.**

**Correction to the named mitigation:** cold archive (`EpisodeRecord`, SMNCOLD3) stores only the 640-dim episode vector, not regional inputs — "rebuild warm from cold" can only re-index old-generation vectors, not re-form episodes through the new binder. It doesn't actually mitigate the mixing. Real options: (a) accept the one-time turnover tax (~30 cycles at the 200k cap; drift shrinks as the binder converges — loss already fell 67.5% in 10 cycles), (b) a SMNCOLD4 format storing regional inputs (~5× cold bloat, only if (a) proves insufficient), (c) slow the binder's change rate via a step budget.

**Adoption path (queued):** (1) step budget `--binding-replay-steps N`, start N=1. (2) repeat pilot ≥20 cycles, n≥2 if possible; pass = binding learns + grown rows move + cross-session gist stable/recovering + displacement contained. (3) adopt only on a clean repeat, drift monitoring on for the first production leg.

**Step 1 done 2026-08-15** — see §36 for the repeat pilot, which **failed**.

## §36 — ADR 0041 repeat pilot at `--binding-replay-steps 1`: wall-clock improved, quality got worse (2026-08-15)

**Setup:** same harness, post-§35 production state, CYCLES=20 (vs §35's 10), n=1/arm, both `fine-metrics`. Treatment steps now capped at N=1 (was unbounded, ~2.8/cycle in §35).

**Wall-clock goal met:** `nrem_wall_ms` **−24.8%** (control 6,784ms/cycle vs treatment 5,099ms) — a real reversal of §35's +34.5%.

**But quality got worse, not better, decomposed per-cycle:**
1. On the 13/20 cycles where the (rarer, larger) step fired, `mean_output_mse` ran ~**10× control** same-cycle (e.g. cycle 3: 0.0098 vs 0.314). Fewer, bigger steps were more disruptive than §35's smaller, frequent ones, not gentler.
2. **New failure mode:** 6/20 treatment cycles replayed **zero** traces (vs control's steady 460-512), traced to a salience/settle homeostatic cascade reacting to degraded output (`sal_threshold` +39%, `settle_gate_dynamic` −44%, `settle_iters_sum` +154%, `prior_beta_effective` −41%) — not visible at this magnitude in §35.
3. Aggregate: `mean_output_mse` +445%, `mean_real_inference_mse` +655%, `gist_recall_rate` −32.5% (comparable order to §35's −53%), `per_region_mse` +20-159% all regions, `blend_count` −52%, `traces_replayed` −58% (worse than §35's −30%). `within_session_gist_rate` still +6.6% (weaker than §35's +13.3%) — drift-not-breakage read holds.

**§33 fix still engages under the cap** (checked directly via `binding_row_norms`, not inferred): control 0/9,089 grown rows nonzero; treatment 30-78% nonzero, `grown_max` 0.029-0.057 (lower than §35's 0.04-0.14, consistent with less total training).

**Verdict — FAIL, worse than §35.** "Binding learning"✓, "grown rows moving"✓, but "gist stable-or-recovering"✗ and "displacement contained"✗ (both worse than §35). n=1/arm, but a 10× same-cycle MSE gap and 6-cycle trace starvation are far past plausible noise.

**What changes:** the hypothesis that capping step *count* would proportionally tame cost and drift was wrong — it decoupled them, trading cost for concentrated-step instability. **Step-count capping abandoned as the lever.** Three untried candidates, no decision yet: (a) reduce step *magnitude* (LR/loss-scale on the term) instead of count, (b) fuse into `fused_replay_wake_step_batch`'s existing optimizer step instead of a separate pass, (c) decouple the salience/settle cascade directly (a control-side clamp on wake-triggering reacting to NREM-quality swings).

---

## §37 — Documentation consistency audit: docs vs. code, via codebase-memory-mcp index (2026-08-15)

**Scope:** user requested a full pass over `ai-docs/` for inconsistencies/gaps, cross-checked against the actual codebase (indexed via `codebase-memory-mcp`, 154,553 nodes/165,461 edges), with any findings written up here and doc/code architecture soundness re-evaluated. This followed the same-day mega-compression pass (session-state.md/ai-docs/ cut ~51%) — this audit is the correctness check on both the compression and the underlying docs it compressed.

**Method:** indexed the repo; spot-checked every hardcoded constant/dimension the docs assert (`N_UNITS`, `K_ACTIVE`, `EPISODE_DIM`, `D_PROJ`, `GRU_HIDDEN`, `LANG_HIDDEN`, `REGION_DIMS`, `Metric::COUNT`, CLI flag defaults, Cargo feature lists) against the actual `crates/*/src/*.rs` definitions; diffed each ADR's own status line against `decisions/README.md`'s index; checked `CLAUDE.md`'s deferred-items list and `session-state.md`'s trigger table against current code and prior ADR outcomes; verified file-existence claims (scripts, symlinks, model artifacts) on disk; ran `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` to confirm the health claims implicit throughout the docs still hold (both clean, 0 failures).

**Result: constants, dimensions, feature flags, and file-existence claims all checked out clean** — `N_UNITS=1024`, `K_ACTIVE=20`, `EPISODE_DIM=640`, `D_PROJ=128`, `GRU_HIDDEN=256`, `LANG_HIDDEN=1024`, `REGION_DIMS=[384,ENV_DIM,384,384,LANG_HIDDEN]`, `RegionId` (5 variants), `Metric::COUNT=102` (100→102 from ADR 0041, confirmed in code, correctly reflected in session-state.md though never stated in gaps.md/ADR 0041 themselves — minor completeness gap, not an error), the `qwen3` Cargo feature's removal (ADR 0036), `mn10_env`/`fine-metrics` features, `region_sizes.bin` on disk (`[9821]×5`, matches §34 exactly), all `memory`/`clips`/`data`/`models` symlinks, `scripts/setup-hooks.sh`/`scripts/hooks/pre-commit` (this session's additions), and the `catch_unwind` Whisper-panic mitigation (gaps.md §5) all matched what the docs claim. No fabricated numbers found anywhere in the compression pass.

**Four real findings, all fixed in place except the two structural ones (§37.3/§37.4, which need a decision, not a doc edit):**

### §37.1 — `decisions/README.md` index drifted from two ADRs' own status lines (fixed)

ADR 0023's own header says "accepted; Phase 0 + Phase 1 implemented (2026-06-07)" but the index table listed it as `proposed`. ADR 0041's own header says "implemented; ... **Not adopted**" (and, as of §36, failed a second pilot) but the index table also listed it as `proposed`. Both corrected in the index. Neither ADR file itself was wrong — only the index's status column had gone stale, presumably never updated when those ADRs' status lines were later amended in place.

### §37.2 — `CLAUDE.md`'s deferred-items list has one flatly wrong entry (fixed)

Item **M (MLP projection)** was listed as deferred. It is not: ADR 0032 implemented the 640→1024→5×LM_HIDDEN MLP and it has been the **default production path** since 2026-06-21 — confirmed via code, not inference: `--linear-proj` (`crates/somnus/src/lib.rs`) defaults to `false`, and the MLP's `fc1`/`fc2` layers (`crates/somnus/src/action.rs`) are parameterized by the `LM_HIDDEN` constant, which is now 1024 (Qwen3) — the same code that shipped for Qwen2.5 works unmodified for Qwen3, no re-implementation needed. `session-state.md`'s deferred-backlog table had carried a hedge ("not re-verified against Qwen3") pending exactly this check — now resolved, caveat removed, both docs corrected. **Item M should not reappear in any future deferred-items list.**

### §37.3 — Qwen3 KV-cache generation speedup (ADR 0034) is not active in production (undocumented, needs a decision)

ADR 0034 shipped a 7.8× generation speedup for Qwen2.5 (530ms/step → 68ms/step) via a split prefill/decode ONNX pair with static-shape KV buffers. That pair was never fully re-exported for Qwen3: `models/qwen3_lm_prefill_kv.onnx` exists (committed 2026-06-27, part of a commit explicitly titled "Item 5 partial — Qwen3-0.6B infrastructure setup"), but **`qwen3_lm_decode_kv.onnx` does not exist** — checked directly on disk (`/data/somnus/models/`, no such file, no such symlink). `ActionDecoder::new` (`crates/somnus/src/action.rs:362-376`) requires *both* the prefill and decode files (or both INT8 variants) to be present before it activates the fast path; with only one present, it falls through to `tracing::info!("KV-cache models absent — generate() using slow lm_session path")`. **Every `generate()` call since Qwen3 became the sole LM (ADR 0036, 2026-07-05) has been running on the slow O(n²) CPU path** — not a silent failure, just a silent non-upgrade nobody re-checked after the Qwen3 swap. Zero mentions of this anywhere in `architecture.md`, `ADR 0034`, `ADR 0036`, or `gaps.md` prior to this audit (fixed in gaps.md item 5, see above). **Not blocking** — generation still works, just ~7.8× slower per call than the architecture is capable of. Fix is mechanical (re-run `scripts/export_kv_cache_qwen3.py`'s decode-side export, blocked per its own 2026-06-27 commit message by a `torch.export` shape-inference issue that was never revisited) — not attempted here, flagging for a decision on priority.

### §37.4 — Qwen3 LoRA fine-tuning (ADR 0033's natural follow-on) was never run (undocumented, needs a decision)

Same orphaned-infrastructure story as §37.3, same source commit (2026-06-27): `scripts/train_lora_qwen3.py` exists, ported from ADR 0033's Qwen2.5 script, but its own commit message says it's blocked ("current blocker: projection weights from Qwen2.5 don't match Qwen3 dims... regenerate projection weights from Qwen3 model, future campaign run"). That campaign run never happened — `memory/lora_output/` has never existed in git history (`git log --all` on the path returns nothing), and a checksum/size/date comparison of `qwen3_lm.onnx` (4,517,046 bytes, 2026-07-05) vs. `qwen3_lm_base.onnx` (5,248,472 bytes, 2026-06-27) shows two genuinely different files with the later one's date matching the ADR 0035/0036 Qwen3-architecture-class-fix session, not a LoRA merge timestamp. **Production Qwen3 generation is running on frozen pretrained weights, never fine-tuned on somnus's own episode-prefix distribution** — this is exactly the "L" deferred item, but `CLAUDE.md`'s one-line description ("LoRA fine-tuning") didn't previously distinguish "never started" from "started, half-built, then orphaned by an unrelated architecture swap." `CLAUDE.md` and `session-state.md` both corrected to state this precisely (see §37.2's edits, same commits).

### Architecture/design soundness re-evaluation

The two orphaned-optimization gaps above (§37.3/§37.4) are real but non-critical — cost/quality regressions on a research system with no latency SLA, not correctness bugs. The one gap that **is** architecturally significant, and was already the subject of two failed adoption attempts this same day (§35, §36), remains **§33: in the default build (`--binding-replay` off), ~89.6% of each region's units — the entire product of the 10x growth campaign — still have zero-initialized, never-trained binding-projection rows, invisible to episode formation.** The growth campaign's SAE-side win is real and measured (`mean_output_mse` −74% first-vs-last-10 cycles, §34), but the binding/episodic payoff of that growth is still mostly unrealized in the deployed (flag-off) configuration, and the two pilots run today to fix this (§35, §36) both failed for different reasons (episode drift; then, after capping step count, worse instability via a salience/settle cascade). No architectural flaw was found beyond what §33/§35/§36 already document — the mechanism itself (live-episode training through `forward_tracked`) is correct and proven to work; the open problem is tuning its blast radius on a live, mixed-generation warm store, which is a genuinely hard problem the docs already characterize accurately. No new mitigation is proposed here; this is a confirmation that the existing gap description is accurate and current, not stale.

**Metrics discipline:** this audit is documentation/build-configuration analysis, no code changed, no new data-handling operation introduced — no new metrics needed, stated explicitly per the standing policy.

---

> **Note:** the `#` column matches the `##` section number above (`§N`), not chronological/priority order. Re-fetch rather than trust a cached copy if numbering looks off (corrected once, 2026-07-11).

| # | Item | Effort | Impact | Status |
|---|---|---|---|---|
| ✅ §0 | **bincode migration** (→ bitcode) | Low–Med | Security advisory | Deployed (57c87a1) |
| ✅ §1 | **BatchTopK SAE** | Low | Sparse coding quality | Deployed (25d7496) |
| ✅ §2 | **High replay ratio A/B** (N=1/2/4) | Trivial | Consolidation quality | Adopted 2026-07-06 — N=4 default (-70.9% output_mse/+16.6% wall-clock); N=2 more efficient (-49.1%/+7.6%) but not chosen |
| ❌ §3 | **DER++ auto-λ (Phase 3)** | Low–Med | Replay consolidation | Stale-output blocking shipped earlier; auto-λ rejected 2026-07-08 — A/B FAIL (+73.8% mean_output_mse), opt-in only |
| §4 | **LISTA unrolled encoder** | Med | Sparse coding quality | Deferred |
| ✅ §5 | **Qwen3-0.6B upgrade** | High | Generation quality + language region | Adopted 2026-07-05 (ADR 0036); Qwen2.5 dropped. **KV-cache (ADR 0034) and LoRA (ADR 0033) both incomplete for Qwen3 — see §37** |
| ✅ §6 | **DINOv3-ViT-S ONNX export test** | Low | Visual region (drop-in) | Deployed (validated 2026-07-05) |
| §7 | **SigLIP2-ViT-B/16** | Med | Visual (Phase 7+ deferred) | Deferred |
| ⏸️ §8 | **M2D-CLAP ONNX export test** | Med | Audio env encoder (+3 mAP) | Not pursued 2026-07-09 — small gain vs. untested export + restrictive license |
| ❌ §9 | **EfficientAT-mn10 ONNX export test** | Low→Med | Audio env encoder | Rejected 2026-07-09 — A/B FAIL (+155.6% MSE), opt-in `mn10_env` feature kept |
| §10 | **Dedicated text embedding model** | Med | Language encoder | Deferred |
| §11 | **Gemma 3 1B IT** | Med–High | Generation (lower priority than Qwen3) | Deferred |
| §12 | **Episodic binding survey** | — | Binding area | Null result — 4-head cross-attention + GRU is SOTA under the ORT/MIGraphX constraint; not pursued |
| ✅ §13 | **Change-gated video encoder** | Low | Wake-path compute (unmeasured savings) | Deployed 2026-07-06, opt-in |
| ❌ §14 | **SVD low-rank check** | Low | Capacity efficiency | Null result 2026-07-06 — near-full-rank, not pursued |
| §15 | **Crate hygiene** (usearch, burn-ndarray watch, ffmpeg-the-third) | Low–Med | HNSW footprint / maintenance | Dead deps removed; rest deferred until raised |
| ✅ §16 | **Growth-scaling pilot campaigns** | Med | Growable-core capacity | Characterized 2026-07-09/11/12 (3 pilots); campaign **completed 2026-08-15 at `[9821]×5` (101.7%)** — §20/§34 |
| ✅ §17 | **K_ACTIVE scaling with region growth** | Low–Med | Sparse coding quality at scale | Deployed 2026-07-12 |
| ✅ §18 | **Scope optimizer reset to grown region only** | Low | Convergence stability during growth | Deployed 2026-07-12 |
| ✅ §19 | **Novelty-proportional growth batch size** | Med | Growth-mechanism correctness | Deployed 2026-07-12 |
| ✅ §20 | **Full 10x campaign launch + tooling** | Low (tooling) | Ops visibility / chart correctness | Deployed 2026-07-12 |
| ✅ §21 | **Warm-store HNSW dump leak + broken persistence** | Low | Disk usage / warm-index reload correctness | Fixed 2026-07-12 |
| ✅ §22 | **Auto-enable ORT/MIGraphX past CPU/GPU crossover** (ADR 0037) | Med | Wake-path perception wall-clock | Deployed 2026-07-12; real speedup blocked by §24, confirmed **~3-4×** live 2026-07-13 |
| 🔮 §23 | **Native Rust ONNX graph construction** | High | Removes Python runtime dependency | Scoped, not started |
| ✅ §24 | **ORT sparse→dense N_UNITS bug + encoder dim drift** | Low | Correctness/perf — ORT silently no-oped | Fixed 2026-07-12/13, confirmed live: zero fallback, ~3-4× real speedup |
| ✅ §25 | **Sparse-aware binding projection** | Low | 11.4× win for `burn`'s path, dormant on ORT-active campaign | Deployed 2026-07-13; ADR 0038 (encoder scoring) scoped, not started |
| ✅ §26 | **Unified two-tier metrics** (`somnus-metrics`, ADR 0039) | Med–High | Ops visibility — 100-variant `Metric` enum | Implemented+expanded 2026-07-14/15, audited |
| ✅ §27 | **Parallelized warm-store HNSW rebuild** | Low | Sleep-cycle wall-clock (~70% of cycle time) | Deployed 2026-07-15 — 16.7× speedup |
| 🔍 §28 | **ADR 0038 prereq 1: encoder graph + live scoring cost** | — | Informs hierarchical WTA priority | Answered 2026-07-15 — small, not dominant at current scale; prereqs 2/3 open |
| 🔍 §29 | **ADR 0038 prereq 2: matters at target scale?** | — | Informs hierarchical WTA priority | Answered 2026-07-15 — no, not at honeybee-order; ADR stays deferred with a trigger, not closed; prereq 3 open |
| ✅ §30 | **Sparse-gather binding projection** (ADR 0040) | Med | Wake-path `OrtRunBinding` — dense projection was 75-76% of its cost | Deployed live 2026-08-12: 40-59% reduction, §28/§29 corrected |
| ✅ §31 | **Test push: rubato-5 drift + drift-pin pattern** | Low | Correctness hardening — 54 new tests | Done 2026-08-13; queued list empty |
| ✅ §32 | **Test push round 2: burn-path contracts + memory I/O** | Low | Correctness hardening — 27 new tests | Done 2026-08-14; no remaining thin spots |
| 🔍 §33 | **BindingArea weights never trained** | Med–High | Episode quality / temporal context / grown-unit participation | Decision (b), ADR 0041 implemented; **§35 pilot: fix engages, drift+cost — not adopted; §36 step-budget cap: wall-clock better but quality WORSE — also not adopted**; step-count capping abandoned, no path chosen |
| ✅ §34 | **10x growth campaign complete** — `[9821]×5` = 101.7% | — | Growable-core capacity milestone | Done 2026-08-15; unblocks §33 |
| 🔍 §35 | **ADR 0041 isolated A/B pilot** (10 cycles/arm) | — | Validates live-episode binding training | Done 2026-08-15 — mechanism validated, drift materialized, +34.5% cost; adoption path queued |
| 🔍 §36 | **ADR 0041 repeat pilot at N=1** (20 cycles/arm) | — | Tests whether step-count capping tames §35's cost/drift | Done 2026-08-15 — **FAIL, worse than §35**; step-count capping abandoned, magnitude/fusion/cascade-decoupling candidates queued |
| 🔍 §37 | **Documentation consistency audit** (codebase-memory-mcp index vs. all ai-docs/) | Med | Doc/code drift correctness | Done 2026-08-15 — constants/dims/flags all verified correct; fixed 2 stale ADR-index statuses, 1 wrong CLAUDE.md deferred entry (M, done not deferred); found 2 undocumented gaps (Qwen3 KV-cache decode half + LoRA never run, both orphaned since 2026-06-27) — not blocking, no path chosen yet |
