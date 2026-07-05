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

## 2. Algorithm: High replay ratio

**Source:** "Scalable Strategies for Continual Learning with Replay," arXiv:2505.12512, May 2025.

**What:** Increasing the replay ratio (buffered episodes replayed per new episode seen) to 4–8× improves consolidation quality with no degradation, orthogonal to buffer selection strategy. If NREM currently does one pass through all episodes in the buffer per sleep cycle, doing 4 passes is the change.

**Which part of somnus:** NREM replay loop in `sleep.rs`.

**Constraints:** Hyperparameter only. No model or ONNX changes.

**Effort:** Trivial (one multiplier). Add `--nrem-replay-epochs N` (default 1) flag. A/B with N=4 on a short campaign.

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

## 5. LLM: Qwen3-0.6B — pipeline fixed, A/B inconclusive (needs a richer speech corpus)

**Source:** Qwen team, April 2025. HF repo: `Qwen/Qwen3-0.6B`.

**What:** 600M parameter LM. MMLU 70.0 vs Qwen2.5-0.5B ~47 — large quality gap. Outperforms Qwen2.5 base models ≤7B on over half of benchmarks, especially STEM and reasoning.

**Architecture:** hidden_size 1024 (up from 896), num_layers 28 (up from 24), num_attention_heads 16 (up from 14), num_key_value_heads 8 (up from 2).

**Completed (commits d0f92c5 → bcf1480):**
- ✅ Base model exported: `qwen3_lm_base.onnx` (2.3 GB)
- ✅ Projection weights: w1.bin (2.5MB, 640×1024), w2.bin (20MB, 1024×5120)
- ✅ Qwen3 embed table: `qwen3_embed_table.safetensors` (622.3 MB, 151k vocab × 1024-dim)
- ✅ Qwen3 tokenizer: `qwen3_tokenizer.json` (11.4 MB)
- ✅ Infrastructure scripts: `export_kv_cache_qwen3.py`, `train_lora_qwen3.py`, `ab_qwen3.sh`
- ✅ **CRITICAL FIX:** Language region matmul dimension mismatch
  - Root cause: `LANG_DIM` hardcoded to 896 (Qwen2.5), k-WTA expected 1024
  - Fix: Updated `LANG_DIM: 896 → 1024`, exported Qwen3 embed table
  - Verification: Text ingestion tested and working (no panic)
- ⚠️ **A/B Validation:** ran 2026-06-28, but found INVALID on 2026-07-05.
  - Reported result: Qwen3 "underperforms" on warm_recall_l2 (0.550 → 0.498, -9.6%)
  - **This number is not trustworthy.** Root cause found 2026-07-05: `LM_HIDDEN` was bumped
    896→1024 (commit `6335b42`, the day before this A/B ran) but `ActionDecoder::new` always
    loads `qwen25_embed_table.safetensors` (896-wide) by hardcoded filename. `embed_flat()`
    (`action.rs:1207`) slices that table at stride `LM_HIDDEN=1024` — every token embedding for
    *both* arms was silently misaligned/corrupted, not a fair test of either architecture.
    Qwen3 also never got matching GPU-path exports (gain/text_hidden/lm_fixed), so it ran on
    the CPU fallback via `ab_qwen3.sh`'s file-swap trick, whose "restore" step also silently
    failed — `qwen25_lm.onnx` on disk is still, right now, actually the Qwen3 model.
  - Analysis (superseded conclusion, corrected root-cause note added): `ai-docs/a_b_qwen3_findings.md`
  - Full incident writeup: `.claude/session-state.md`
- ✅ **Redo (2026-07-05):** pipeline fully fixed — genuine Qwen2.5 baseline restored, Qwen3 given
  a complete matching export (via the correct `Qwen3ForCausalLM`, not `Qwen2ForCausalLM`), model-file
  prefix parameterized (no more hardcoded `qwen25_*`/auto-detect footguns), `LM_HIDDEN`/`LANG_DIM`/
  `EOS_TOKEN_ID` made real compile-time choices via a `qwen3` Cargo feature (two release binaries:
  `somnus-qwen25-control`, `somnus-qwen3-treatment`). Found and fixed a third real bug along the way: ~6
  more hardcoded 896/1024 literals scattered through `somnus-perception` for the language region.
  - **A/B result: inconclusive, not "Qwen3 loses."** Ran 10 sessions × 5 cycles/arm cold-started on
    the standard 4-clip corpus. Language-region MSE is statistically indistinguishable between arms
    (0.0011 vs 0.0010) and `mean_proj_loss` has almost no samples (0/50 vs 1/50) — the stock clips
    have too little actual speech for ASR/lang-ingest to fire more than ~once per 5-cycle session.
    Large gaps in aggregate `mean_output_mse`/`warm_recall_l2` are noise from independently-seeded
    non-LM regions (speech/env/visual/symbolic), not a Qwen3 effect — see per-region breakdown in
    `ai-docs/a_b_qwen3_findings.md`.
  - Full results + recommended next corpus (`clips/long/` or direct generation-quality test):
    `ai-docs/a_b_qwen3_findings.md`.
  - Aside: found an unrelated, pre-existing intermittent panic in `WhisperDecoder::decode_greedy`
    (ORT "GetElementType is not implemented" escalating past its usual caught-WARN path) — not
    fixed, flagged for separate triage.
  - Status: **Infra done. Comparison needs a speech-heavy corpus or a direct generation-quality
    test to actually answer the Qwen3-vs-Qwen2.5 question** — not queued as blocked, just not the
    next automatic step.
- ✅ **Redo take 2 (2026-07-05):** re-ran on a speech-dense `clips/long/` corpus (6 lecture/podcast
  clips, `scripts/ab_qwen3_longclips.sh`) instead of the sparse-speech 4-clip set. Same 5 sessions
  × 4 cycles/arm. This time both arms got identical exposure (73 `lang_ingest` events, ~40
  `decoder_trainable` events each) and `mean_proj_loss` had 12/20 non-null samples per arm — a real
  comparison this time.
  - **Result: Qwen3 wins on `mean_proj_loss`, consistently.** Treatment sits below control at every
    comparable point across the whole campaign (session 1: 0.00190 vs 0.00223 → session 5: 0.00120
    vs 0.00155, a ~22% final gap), no reversals. `per_region_mse[language]` is directionally
    consistent (0.0010 vs 0.0011). `warm_recall_l2` came back close this time (0.48 vs 0.47),
    supporting take 1's read that its earlier gap there was seed noise, not signal.
  - **Verdict: worth adopting Qwen3-0.6B** for the action-decoder LM, on the strength of this result
    — recommend confirming with a longer run and/or a direct generation-quality test (drop
    `--action-distill-only`) before flipping the default build. Full writeup:
    `ai-docs/a_b_qwen3_findings.md`.
  - The Whisper ASR panic recurred more often on this denser corpus (~1 crash/5 sessions control,
    ~3/5 treatment) — still no data loss (20/20 cycles present both arms), still unrelated to this
    work, but now more worth fixing given how often it's showing up.

**Remaining (non-critical):**
- ⚠️ LoRA training: Blocked on CPU (PyTorch/HF compatibility)
  - Not worth pursuing until the A/B redo above establishes a real base-model signal
- ⚠️ KV-cache export: Optional optimization
  - Base model functional without it

**Metrics being collected (per cycle):**
- `output_mse` — reconstruction error
- `p_lang` — language region activation probability  
- `proj_loss` — action decoder projection loss
- Regional activation stats, episode counts

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

## 13. Crate hygiene

### Remove dead deps
`candle-core = "0.10"` and `candle-nn = "0.10"` are declared in `[workspace.dependencies]` but no crate depends on them. Safe to remove the workspace declarations.

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
| 2 | **High replay ratio A/B** (N=4) | Trivial | Consolidation quality | Queued |
| 4 | **Remove dead candle deps** | Trivial | Cleanliness | Queued |
| 5 | **Qwen3-0.6B upgrade** | High | Generation quality + language region | A/B favors Qwen3 on proj_loss (clips/long redo, 2026-07-05) — worth adopting, confirm w/ longer run |
| ✅ 6 | **DINOv3-ViT-S ONNX export test** | Low | Visual region (drop-in) | Deployed (validated 2026-07-05, ai-docs/a_b_dinov3_findings.md) |
| 7 | **DER++ stale-output blocking** | Low–Med | Replay consolidation | Deferred |
| 8 | **EfficientAT-mn10 ONNX export test** | Low | Audio env encoder | Deferred |
| 9 | **M2D-CLAP ONNX export test** | Med | Audio env encoder (+3 mAP) | Deferred |
| 10 | **SigLIP2-ViT-B/16** | Med | Visual (Phase 7+ deferred) | Deferred |
| 11 | **LISTA unrolled encoder** | Med | Sparse coding quality | Deferred |
| 12 | **Gemma 3 1B IT** | Med–High | Generation (lower priority than Qwen3) | Deferred |
| 13 | **usearch migration** | Med | HNSW footprint | Deferred |
