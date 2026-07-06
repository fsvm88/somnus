# Gaps, candidates, and open work

> **Contents:** Output of the 2026-06-27 review round — security advisories, algorithm improvement candidates, encoder/LLM upgrade candidates, crate hygiene. Each item has enough detail to implement or reject without re-researching.  
> **When to read:** Starting new work, picking the next item to try.  
> **Status:** All items unimplemented as of 2026-06-27 unless noted. Add ✅ and ADR reference when an item ships.

---

## 0. Critical: bincode security advisory ✅ RESOLVED

**RUSTSEC-2025-0141.** Migrated from archived `bincode` 2.x to `bitcode` 0.6 (actively maintained).

**Migration:** Commit 57c87a1. `crates/somnus-memory/src/semantic.rs` updated; WARM_VERSION bumped 1→2. Old bincode sidecars trigger graceful cold rebuild (existing fallback pattern). All tests pass.

---

## 1. Algorithm: BatchTopK SAE ✅ DEPLOYED

**Source:** Bussmann et al., arXiv:2412.06410, Dec 2024.

**Status:** A/B validation PASS (commit 25d7496). Default behavior in all NREM training.

**What:** Replaces per-sample hard top-k (k-WTA) with a batch-level quantile threshold: one scalar threshold is computed per batch such that the *average* number of active features across the batch equals k. Individual samples may use k±δ features as their activation distribution demands. Uses pure **L2 reconstruction loss** — no L1 penalty.

**Why relevant:** Our L1 SAE experiment (ADR 0025, λ=0.1) hurt MSE by +27% and was closed. BatchTopK is a structurally different approach — it avoids L1 entirely and shows improvements over per-sample TopK on GPT-2 Small and Gemma 2 2B benchmarks at matched average sparsity.

**Implementation:** `KWtaLayer::forward_tracked_batch(batch_topk: bool)` parameter. Snapshot duty-cycle at entry, compute boosted batch values, find single threshold τ via `select_nth_unstable` on flattened array, apply to all activations. Duty-cycle update order-independent (snapshot→threshold→decay). Inference unchanged (per-sample top-k in ONNX). `--batch-topk` flag added; default false.

**A/B:** `scripts/ab_batch_topk.sh`. Primary metric: output_mse (same metric L1 experiment regressed on). Secondary: per-region cosines. Control: per-sample (default). Treatment: --batch-topk.

---

## 2. Algorithm: High replay ratio ✅ VALIDATED (2026-07-06) — recommend flipping default

**Source:** "Scalable Strategies for Continual Learning with Replay," arXiv:2505.12512, May 2025.

**What:** Increasing the replay ratio (buffered episodes replayed per new episode seen) to 4–8× improves consolidation quality with no degradation, orthogonal to buffer selection strategy. If NREM currently does one pass through all episodes in the buffer per sleep cycle, doing 4 passes is the change.

**Which part of somnus:** NREM replay loop in `sleep.rs`. `--nrem-replay-epochs N` (default 1) already existed as a flag before this A/B.

**Bug found and fixed first:** `scripts/train_campaign.sh` (the shared harness behind every A/B script, including this one) hardcoded `--nrem-replay-epochs 4` unconditionally, regardless of `EXTRA_ARGS`. A prior run of this exact A/B (dated 2026-06-27, `memory/ab_replay_ratio_stale_20260627_buggy_n4_both_arms/`) therefore compared N=4 against N=4, not N=1 vs N=4 — meaningless, which is presumably why item 2 was never marked resolved despite a script existing. Fixed by parameterizing `REPLAY_EPOCHS` (env var, default 1, matching the CLI's own default) in `train_campaign.sh`, and having `ab_replay_ratio.sh` set it explicitly per arm instead of via `EXTRA_ARGS` (avoids relying on clap's last-occurrence-wins behavior for a repeated flag). That old stale data also predates the DINOv3/Qwen3 adoption, so it wouldn't have been usable regardless.

**Result: real, decisive PASS.** 5 sessions × 2 cycles, fresh memory both arms, h3h3/shorts clip corpus (`scripts/ab_replay_ratio.sh`). `mean_output_mse`: control avg 0.2379, treatment (N=4) avg 0.0693 (**-70.9%**), consistent across every session with valid data (a few control cycles logged `mean_output_mse=0.0` — an artifact of the training-step-skip path when a batch is fully filtered as stale/empty, not a real zero, correctly excluded by the analysis). Per-cycle deltas where both arms have real data: **-83% to -88%**, no reversals. Per-region MSE breakdown confirms it's not one region compensating for another — all 5 regions improve monotonically across the campaign in both arms, treatment strictly below control at every point (language region is already near-floor in both, ~0.00097, as expected — nothing left to improve there).

**Third arm added — N=2, to find the cost/benefit knee.** Same corpus/sessions/cycles, fresh memory, `REPLAY_EPOCHS=2`. Real NREM wall-clock (`nrem_wall_ms`, summed over the whole campaign) and total per-cycle wall-clock (perception + NREM + REM + snapshot + spill) both measured directly rather than assumed:

| N | avg `mean_output_mse` | Δ vs N=1 | total wall-clock Δ vs N=1 |
|---|---|---|---|
| 1 | 0.2379 | — | — |
| 2 | 0.1211 | **-49.1%** | **+7.6%** |
| 4 | 0.0693 | -70.9% | +16.6% |

Real diminishing returns: N=1→2 buys 49% of the total error reduction for well under half the wall-clock cost of N=1→4; the second half of the gain (49%→71%) costs almost as much wall-clock again. Per-1%-of-cost, N=2 is the more efficient point (~6.5% error reduction per 1% wall-clock vs N=4's ~4.3%). N=2's per-region breakdown is equally clean — every region sits strictly between the N=1 and N=4 values at matching session/cycle points, improving monotonically, no reversals.

**Effort:** Trivial (one multiplier), already implemented. A/B: `scripts/ab_replay_ratio.sh` (N=1 vs N=4); N=2 arm run ad hoc via `train_campaign.sh REPLAY_EPOCHS=2` directly (not yet folded into the script as a third arm).

**Decision (2026-07-06): N=4 adopted as the default.** N=2 was the more efficiency-optimal point (better error-reduction-per-cost ratio), but the absolute wall-clock tax at N=4 is still small (+16.6% total cycle time) and the user chose to pay it for the larger absolute quality gain, on the view that ~17% is recoverable elsewhere. CLI default (`crates/somnus/src/main.rs`, `nrem_replay_epochs`) and `train_campaign.sh`'s `REPLAY_EPOCHS` default both flipped 1→4. `scripts/ab_replay_ratio.sh` is unaffected (sets `REPLAY_EPOCHS` explicitly per arm already).

---

## 3. Algorithm: DER++ auto-λ + stale-output blocking

**Source:** Kobayashi 2025, arXiv:2504.20932, published Frontiers in AI 2026.

**What:** Two additions to replay-based consolidation:
1. **Stale-output blocking:** ✅ IMPLEMENTED (commit 65ceaf7)
   - Detects when buffered sample's stored outputs have drifted under distribution shift
   - Per-batch inference + cosine similarity check (threshold: 0.90)
   - Skips stale samples before training, preventing outdated targets from hurting consolidation
   - Tracked in SleepStats.stale_traces_skipped for monitoring

2. **Auto-λ:** ⏳ DEFERRED TO PHASE 3
   - Auto-adapts loss weight λ using running variance estimate
   - Build on Phase 2 staleness implementation
   - Eliminates hand-tuning λ between MSE-on-past-outputs and current-task loss

**Which part of somnus:** NREM consolidation replay loop.

**Phase 2 Implementation:** `PerceptionProcessor.infer_episode_outputs()` infers current model predictions; NREM batch loop compares vs stored regionals via cosine. Filtered batch trained and projected.

**Phase 3 effort:** Low–Medium. Auto-λ is a running variance update (~10 lines) — implement once Phase 2 validated via campaign.

---

## 4. Algorithm: LISTA Top-K unrolled encoder

**Source:** "Semi-Unified Sparse Dictionary Learning with Learnable Top-K LISTA and FISTA Encoders," arXiv:2511.10575, Nov 2025.

**What:** Replaces single-shot k-WTA with 3–5 unrolled ISTA iterations, each with a learned step size and Top-K gate using a straight-through estimator. Multi-step soft competition converges to sparser, better-reconstructed codes than single-step k-WTA at the same k.

**Which part of somnus:** All 5 sparse-coding regions — replaces the per-region k-WTA layer with a small learned iterative module.

**Constraints:** Each unrolled step is matmul + bias + TopK mask + elementwise multiply — all standard ONNX ops. Static depth (fixed number of steps) means a static ONNX graph. No CUDA required.

**Effort:** Medium. Implement 3–5 LISTA layers per region as a learned burn module; retrain; re-export `perception_encoder.onnx`.

---

## 5. LLM: Qwen3-0.6B ✅ ADOPTED (2026-07-05, ADR 0036) — Qwen2.5 dropped

**Source:** Qwen team, April 2025. HF repo: `Qwen/Qwen3-0.6B`. 600M params, hidden_size 1024
(Qwen2.5-0.5B: 896), MMLU 70.0 vs ~47.

**Outcome:** Qwen3-0.6B is now the sole action-decoder LM; Qwen2.5 support (and the floated
"Option D" Qwen2.5-0.5B-Instruct) is dropped. `LM_HIDDEN`/`LANG_HIDDEN`/`LANG_DIM`/`EOS_TOKEN_ID`
hardcoded to Qwen3's values; cross-modal projection input fixed at 2304 (was a 2176/2304 toggle).
Full evidence trail: `ai-docs/a_b_qwen3_findings.md`, ADR 0036.

**Path to adoption (2026-06-28 → 2026-07-05):** an initial A/B (2026-06-28) was later found
invalid — a hardcoded 896-wide embed table was sliced at stride 1024, corrupting token embeddings
in both arms. Fixing that surfaced two more real bugs (Qwen3 loaded into the wrong model class;
~6 more hardcoded language-region dims in `somnus-perception` disagreeing with each other). Once fixed
and re-run on the stock 4-clip corpus, the result was inconclusive (too little speech for
lang-ingest to fire more than ~once/session) — not "Qwen3 loses." Re-run on the speech-dense
`clips/long/` corpus with identical exposure both arms: Qwen3 beat Qwen2.5 on `mean_proj_loss` by
~22%, consistently, no reversals across the campaign. A direct `generate()` comparison confirmed
this is real, user-visible: Qwen3 produced coherent English, Qwen2.5 incoherent token garbage.
Adopted on the strength of both results. Full blow-by-blow (useful mainly as an A/B-methodology
reference) in `ai-docs/a_b_qwen3_findings.md`.

**Aside, found not fixed:** an intermittent `WhisperDecoder::decode_greedy` ORT panic
("GetElementType is not implemented"), unrelated to this work — see the Whisper item below.

**Remaining (non-critical):**
- LoRA training: blocked on CPU (PyTorch/HF compatibility). Not worth pursuing unless distillation
  training proves insufficient.
- KV-cache export: optional optimization, base model functional without it.
- Whisper `decode_greedy` intermittent ORT panic — ✅ mitigated 2026-07-05: the ASR thread
  (`crates/somnus/src/main.rs`) wraps the call in `catch_unwind` so it no longer kills the whole
  thread. **Unverified:** if the underlying ORT panic poisons the `Session`, every later call on
  it may also panic — degrading to "logs a WARN and drops every cycle" rather than full recovery.
  Root cause still undiagnosed.

**Metrics collected per cycle:** `output_mse` (reconstruction error), `p_lang` (language region
activation probability), `proj_loss` (action decoder projection loss), regional activation stats,
episode counts.

---

## 6. Visual encoder: DINOv3-ViT-S (drop-in candidate) ✅ VALIDATED & ADOPTED (2026-07-05)

**Source:** Meta, August 2025, arXiv:2508.10104.

**What:** Distilled from a 7B ViT teacher. ~21M params — same as DINOv2-S. Output dim: **384 — identical to current DINOv2-S/14 (true drop-in)**. ImageNet-1k linear probe: 81.40% vs DINOv2-S 81.1% (marginal improvement).

**Status:** Working on the MIGraphX GPU EP (AMD RX 9070 XT / gfx1201). The earlier "GPU-only, MIGraphX CPU backend can't compile DINOv3's attention graph" conclusion (2026-06-28) was **wrong** — that write-up was based on a misread error. Root cause and fix:

- `torch.onnx.export` wrote `dinov3_small.onnx` in external-data format (weights in a sibling `.data` file), unlike `dinov2_small.onnx` which embeds weights inline. Nothing in the export code explains the difference; not investigated further since the fix below makes it moot.
- The MIGraphX EP resolves that external-data reference relative to the **process's CWD**, not the model file's directory. Running from anywhere except `models/` produced `generic_read_file: Failure opening file: dinov3_small.onnx.data`, surfaced by ORT as the more opaque `MGXKernel_graph_main_graph_..." Failed to call function`.
- Confirmed live: `experiments/ort-smoke` loaded and ran DINOv3 correctly on MIGraphX from `models/` as CWD, and failed with the exact error above from any other CWD.
- **Fix:** `scripts/export_models.py`'s `export_dinov3()` now re-packs the model with `onnx.load(..., load_external_data=True)` / `onnx.save(..., save_as_external_data=False)`, embedding all weights into the single `.onnx` file (matching DINOv2's layout) so there's no CWD dependency.

**Separate real finding (not a bug, a shape difference):** DINOv3-ViT-S/16 outputs `[1, 201, 384]` — CLS + 4 register tokens + 196 patches (14×14 @ patch16) — vs DINOv2-S/14's `[1, 257, 384]` (CLS + 256 patches, 16×16 @ patch14). Token order is `[CLS, register×4, patch...]` (`transformers/models/dinov3_vit/modeling_dinov3_vit.py`, `torch.cat([cls_token, register_tokens, patch_embeddings])`). `crates/somnus-encoding/src/video.rs` (`VideoEncoder`) now tracks `prefix_tokens`/`total_tokens` per model and skips the correct prefix before mean-pooling — previously it hardcoded "skip 1 token, pool 256," which is DINOv2-only.

**Status now:** DINOv3 is a real drop-in (identical 384-dim output). The A/B run in `ai-docs/a_b_qwen3_findings.md` used DINOv2 for both arms due to this now-fixed bug. A proper DINOv3-vs-DINOv2 A/B was run 2026-07-05 (`ai-docs/a_b_dinov3_findings.md`): visual-region reconstruction MSE dropped ~95.6% (2.183 → 0.097) and memory retrieval quality (warm_recall_l2) improved +61.5%, with no change to unrelated regions (speech/env/language) — a clean, large, correctly-localized win. DINOv3 is already the active encoder (`VideoEncoder::new` prefers it whenever present); no further rollout action needed.

---

## 7. Visual encoder: SigLIP2-ViT-B/16

**Source:** Google DeepMind, February 2025, arXiv:2502.14786. HF ONNX: confirmed via transformers/Optimum. NVIDIA TAO Toolkit also ships ONNX.

**What:** 86M params. ImageNet zero-shot top-1: 79.1% vs SigLIP's 76.7%. Key advantage over DINOv2-S: language-grounded features (CLIP-style). DINOv2's 81.1% is a supervised linear probe — not directly comparable. SigLIP2 adds zero-shot transfer that DINOv2 lacks entirely.

**Dimension change:** 768-dim (up from 384). Requires:
- Visual region k-WTA input changes: 384→768 (cold-start for visual region)
- Binding area visual token projection dim changes

**MIGraphX:** Likely compatible — standard ViT multi-head attention, no ScatterElements expected.

**This is item K (SigLIP-B, deferred to Phase 7+).** No trigger condition is newly met; raise when cross-modal grounding becomes a research target (item N).

**Effort:** Medium. ONNX export, dimension migration, A/B on visual region cosine.

---

## 8. Audio env encoder: M2D-CLAP

**Source:** NTT, March 2025, arXiv:2503.22104. Weights: `nttcslab/m2d` (GitHub).

**What:** ~86M params (ViT-B backbone). Two-stage pretraining: masked audio modeling + CLAP contrastive alignment. AudioSet linear probe mAP: **49.0** vs CLAP-HTSAT-unfused ~45–46 mAP (+3 mAP improvement).

**Dimension change:** 768-dim (up from CLAP-HTSAT's 512). Requires audio-env region input adaptation.

**ONNX status:** Needs testing. ViT-B architecture; masked modeling pretraining sometimes leaves unusual graph ops — medium export risk.

**MIGraphX:** Uncertain (ViT attention ops require GPU op verification); expect to fall back to CPU as with current CLAP.

**Effort:** Medium (export + validation + A/B). Verify ONNX export before committing.

---

## 9. Audio env encoder: EfficientAT-mn10 (low-risk alternative to M2D-CLAP)

**Source:** `fschmid56/EfficientAT` (GitHub, actively maintained 2025).

**What:** ~5–20M params (MobileNetV3, width_mult=1.0). Pure CNN — no attention ops, no ScatterElements. AudioSet pre-trained.

**ONNX status:** Very low risk — `torch.onnx.export` should be trivial for MobileNetV3.

**Dimension:** ~960 (MobileNetV3 penultimate feature — verify before committing; projection update required vs CLAP-HTSAT's 512).

**Quality evidence:** Weaker than M2D-CLAP (no direct mAP vs CLAP-HTSAT head-to-head found). Main argument: tiny param count, guaranteed ONNX/MIGraphX compatibility.

**Effort:** Low (export + A/B). Best tried before M2D-CLAP given its lower export risk.

---

## 10. Language encoder: dedicated text embedding model

Current language region input: Qwen2.5-0.5B embed table (896-dim mean-pooled). A dedicated text encoder would be better, but **the interaction with `--lang-use-lm` is the key constraint** — the LM hidden states are the training signal that gave us the 4× p_lang improvement. Switching the input encoder changes the region's input dim; switching the LM changes the training signal dim. Both need to move together for a clean migration.

**Two confirmed ONNX options:**

| Model | Params | Dim | ONNX | MTEB | Notes |
|---|---|---|---|---|---|
| **nomic-embed-text-v1.5** | 137M | 768 (Matryoshka-reducible) | confirmed | 62.39 | Supports 8192-token context |
| **modernbert-embed-base** | 149M | 768 (Matryoshka to 256) | confirmed (raw export; `--optimize` flag not yet supported) | — | Based on ModernBERT backbone |
| **BGE-small-en-v1.5** | 33M | 384 | confirmed | low-to-mid 60s | Smallest option, existing release |

**Recommended path:** Only migrate the language encoder as part of the Qwen3-0.6B upgrade (item 5). Qwen3's embed table is 1024-dim; the LM hidden states training signal switches to 1024-dim at the same time. This avoids a mismatched-dimension intermediate state.

---

## 11. LLM: Gemma 3 1B IT

**Source:** Google DeepMind, March 2025. Community ONNX: `onnx-community/gemma-3-1b-it-ONNX`.

**What:** 1B instruction-tuned. Community ONNX release via onnxruntime-genai. Better benchmark coverage than Qwen2.5-0.5B.

**Critical caveat:** The community ONNX uses onnxruntime-genai's opaque KV-cache management, not the static prefill/decode split from ADR 0034. Reproducing ADR 0034's static split for Gemma 3 is high effort. Also: **verify ScatterElements is absent from the decode graph before committing to GPU path**.

**Effort:** Medium (onnxruntime-genai CPU route) to High (reproducing ADR 0034 static KV split). Lower priority than Qwen3-0.6B given less certain ONNX compatibility story.

---

## 12. Null result: episodic binding

Surveyed 2025–2026 binding improvements. All papers that outperform 4-head cross-attention + GRU use Mamba/SSM variants (CAF-Mamba arXiv:2601.21648; Cross-State Fusion Mamba arXiv:2509.00677) — all require the `selective_scan_fn` CUDA kernel, which has no ONNX standard equivalent and is not supported by MIGraphX.

**Conclusion:** Current 4-head cross-attention + GRU binding is state-of-art under the ORT/MIGraphX constraint. No actionable improvement found.

One extractable idea: a scalar learned gate per modality per GRU step (pure sigmoid + multiply), bolted onto the existing binding. This is incremental and unvalidated.

---

## 13. Change-gated video encoder inference ✅ DEPLOYED (2026-07-06)

Grew out of `ai-docs/research/brain-scale-comparison.md`'s capacity-efficiency survey: skip
encoder inference on near-static input rather than running it on a fixed clock, mirroring the
biggest efficiency mechanism real neural tissue uses (spikes only on change).

**Implementation:** `VideoFrame::mean_abs_diff` (`crates/somnus-sensory/src/frame.rs`) computes mean
absolute per-byte difference between consecutive candidate frames, normalized to `[0, 1]`;
returns `1.0` (treated as changed) on length mismatch (covers `Mjpeg`/resolution changes, where
byte-level comparison isn't meaningful). The video encoder task
(`crates/somnus/src/main.rs`, video encoder `spawn_blocking`) compares each stride-selected frame
against the last *encoded* frame and skips `enc.encode()` (no embedding emitted, same as the
existing stride-skip path) when the diff falls below `--video-change-threshold`. A
`--video-change-max-skip` heartbeat (default 30) forces a re-encode periodically so a fully static
scene doesn't stop emitting embeddings entirely. Default threshold is `0.0` (disabled) —
opt-in, preserves prior behavior exactly.

Audio encoders (Whisper/CLAP) were left ungated — silence detection is a different, less
contained problem (VAD-style, not a simple byte diff) and wasn't in scope for this pass.

Not yet A/B'd for wake-path CPU savings on a real capture session — the mechanism is deployed and
tested (unit tests on `mean_abs_diff`, full `cargo test`/`clippy` pass) but the actual skip rate
and any downstream effect on temporal binding quality is unmeasured. Do this before assuming a
speedup.

---

## 14. Null result: low-rank (SVD) factorization of region/binding weights (2026-07-06)

Item 2 of `ai-docs/research/brain-scale-comparison.md`'s capacity-efficiency survey: checked
whether somnus's own growable region/binding weight matrices (k-WTA region encoders, cross-modal
projection, binding-area region projections, attention Q/K/V, GRU gates, temporal projection —
21 matrices total, none of the frozen encoders/LM) have learned a low effective rank, which would
make LoRA-style factorization near-free compression.

**Method:** new standalone bin `cargo run --release -p somnus-perception --bin svd_rank_check --
<memory_dir> <out_dir>` pre-grows a fresh `PerceptionProcessor` to a saved checkpoint's region
sizes, loads its trained `weights.bin`, and dumps every 2-D weight matrix as raw f32 + a JSON
manifest (reuses the existing `export_weights_for_onnx` path). `scripts/svd_rank_check.py`
(numpy, no scipy/torch needed) then runs `np.linalg.svd` per matrix and reports the rank needed
to capture 90/95/99% of singular-value energy. Checked against `memory/qwen3_ab_treatment/`, the
most-trained checkpoint on disk (grown to `[1664,1600,1600,1600,1600]` from a Qwen3 A/B campaign).

**Result: null. No matrix showed fast singular-value decay.** Rank needed for 95% of energy
(`r95`) sits at 60–82% of full rank across all 21 matrices (mean `r95/full_rank` = 0.772); the
"compression" a rank-`r95` factorization (`U·Vᵀ`, `(n+m)·r` params) would give is ≈1.0x or worse
versus the dense matrix, because `r95` is too close to `min(n, m)` for the `(n+m)·r` low-rank
parameterization to undercut `n·m`. These weights are effectively full-rank at their current
size/training. **Low-rank factorization is not free here — do not pursue it.**

**Caveats:** single checkpoint, one point in the growth/training arc — a matrix could still
become effectively low-rank as regions grow much larger relative to `EPISODE_DIM`/`D_PROJ`
(rank is capped at `min(n, m)`, and several of these matrices have one dimension fixed at 128/256/
384/640 regardless of region growth), so re-check if that ratio changes substantially. Not
re-verified against a second checkpoint, but the near-full-rank result was consistent enough
across all 21 matrices (no outliers) that a second checkpoint is unlikely to change the
conclusion.

---

## 15. Crate hygiene

### Remove dead deps
✅ Done — `candle-core`/`candle-nn` are no longer in `[workspace.dependencies]` (removed prior to
2026-07-05; this line was stale, the item had already been resolved).

### HNSW alternative: usearch
`usearch 2.25.3` (Unum Cloud, May 2026, actively maintained). Key advantage over `hnsw_rs`: built-in INT8/INT4 vector quantization inside the index — warm-tier episode embeddings could be stored compressed without a separate quantization pass. Also: AVX-512/AVX2/NEON SIMD, thread-safe concurrent insertions, user-defined distance metrics.

Migration cost: rewrite the HNSW interaction layer in `crates/somnus-memory`. Recall@10 claimed at parity with hnswlib/FAISS under matched hyperparameters. Raise when warm memory footprint becomes a concern.

### burn-ndarray deprecation watch
`burn 0.21` notes that `burn-ndarray` is on a deprecation path toward `burn-flex` eager CPU backend. Watch for breaking changes in 0.22 (expected late summer 2026). Currently no change needed.

### ffmpeg-the-third
`ffmpeg-the-third` (`shssoichiro/ffmpeg-the-third`) is an actively maintained fork of `ffmpeg-next` that may have surpassed the primary crate in maintenance velocity. Worth checking when next touching the media pipeline.

---

## Priority order (as of 2026-06-28)

| # | Item | Effort | Impact | Status |
|---|---|---|---|---|
| ✅ 0 | **bincode migration** (→ bitcode) | Low–Med | Security advisory | Deployed (57c87a1) |
| ✅ 1 | **BatchTopK SAE** | Low | Sparse coding quality | Deployed (25d7496) |
| ✅ 2 | **High replay ratio A/B** (N=1/2/4) | Trivial | Consolidation quality | Adopted 2026-07-06 — N=4 default (-70.9% output_mse/+16.6% wall-clock); N=2 was more efficient (-49.1%/+7.6%) but not chosen |
| ✅ 4 | **Remove dead candle deps** | Trivial | Cleanliness | Already done (stale entry, corrected 2026-07-05) |
| 5 | **Qwen3-0.6B upgrade** | High | Generation quality + language region | ✅ Adopted 2026-07-05 (ADR 0036); Qwen2.5 dropped. Gen-quality confirmed (take 3) |
| ✅ 6 | **DINOv3-ViT-S ONNX export test** | Low | Visual region (drop-in) | Deployed (validated 2026-07-05, ai-docs/a_b_dinov3_findings.md) |
| 7 | **DER++ stale-output blocking** | Low–Med | Replay consolidation | Deferred |
| 8 | **EfficientAT-mn10 ONNX export test** | Low | Audio env encoder | Deferred |
| 9 | **M2D-CLAP ONNX export test** | Med | Audio env encoder (+3 mAP) | Deferred |
| 10 | **SigLIP2-ViT-B/16** | Med | Visual (Phase 7+ deferred) | Deferred |
| 11 | **LISTA unrolled encoder** | Med | Sparse coding quality | Deferred |
| 12 | **Gemma 3 1B IT** | Med–High | Generation (lower priority than Qwen3) | Deferred |
| 13 | **usearch migration** | Med | HNSW footprint | Deferred |
| ✅ 14 | **Change-gated video encoder** | Low | Wake-path compute (unmeasured savings) | Deployed 2026-07-06, opt-in via `--video-change-threshold` |
| ❌ 15 | **SVD low-rank check (region/binding weights)** | Low | Capacity efficiency | Null result 2026-07-06 — near-full-rank, not pursued |
