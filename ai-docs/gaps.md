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
1. **Auto-λ:** auto-adapts the loss weight λ between the MSE-on-past-outputs term and the current-task loss term using a running variance estimate — eliminates hand-tuning λ.
2. **Stale-output blocking:** detects when a buffered sample's stored output has become stale under distribution shift (cosine divergence check) and skips replaying it, preventing cases where replaying outdated targets hurts consolidation.

**Which part of somnus:** NREM consolidation replay loop.

**Constraints:** Pure Rust math (array indexing, scalar statistics, cosine distance). No external deps needed.

**Effort:** Low–Medium. Auto-λ is a running variance update (~10 lines); stale blocking is a cosine check per buffered sample before replay (~15 lines).

---

## 4. Algorithm: LISTA Top-K unrolled encoder

**Source:** "Semi-Unified Sparse Dictionary Learning with Learnable Top-K LISTA and FISTA Encoders," arXiv:2511.10575, Nov 2025.

**What:** Replaces single-shot k-WTA with 3–5 unrolled ISTA iterations, each with a learned step size and Top-K gate using a straight-through estimator. Multi-step soft competition converges to sparser, better-reconstructed codes than single-step k-WTA at the same k.

**Which part of somnus:** All 5 sparse-coding regions — replaces the per-region k-WTA layer with a small learned iterative module.

**Constraints:** Each unrolled step is matmul + bias + TopK mask + elementwise multiply — all standard ONNX ops. Static depth (fixed number of steps) means a static ONNX graph. No CUDA required.

**Effort:** Medium. Implement 3–5 LISTA layers per region as a learned burn module; retrain; re-export `perception_encoder.onnx`.

---

## 5. LLM: Qwen3-0.6B

**Source:** Qwen team, April 2025. HF ONNX: `onnx-community/Qwen3-0.6B-ONNX` (confirmed working, used in Transformers.js WebGPU demos).

**What:** 600M parameter LM. MMLU 70.0 vs Qwen2.5-0.5B ~47 — large quality gap. Outperforms Qwen2.5 base models ≤7B on over half of benchmarks, especially STEM and reasoning.

**Dimension change:** hidden_size 1024 (up from 896). This cascades:
- MLP projection output changes: 640→1024→5×**1024** (currently 5×896)
- Language region k-WTA input changes: 1024 (from 896)
- KV-cache ONNX models need re-export (`scripts/export_kv_cache.py`)
- LoRA fine-tuning (ADR 0033) needs to be re-run on Qwen3 architecture

**GPU path:** CPU only (same constraint as Qwen2.5 — dynamic seq_len blocks MIGraphX). GPU generation path remains 14× slower on RDNA4, so default stays CPU.

**MIGraphX for generation:** Risky. KV-cache generation commonly introduces ScatterElements for KV scatter — inspect exported graph before committing to GPU path.

**Effort:** High (projection update, LoRA rerun, KV re-export, A/B). Best treated as a cohesive "LM upgrade cycle" rather than a quick swap.

---

## 6. Visual encoder: DINOv3-ViT-S (drop-in candidate)

**Source:** Meta, August 2025, arXiv:2508.10104.

**What:** Distilled from a 7B ViT teacher. ~21M params — same as DINOv2-S. Output dim: **384 — identical to current DINOv2-S/14 (true drop-in)**. ImageNet-1k linear probe: 81.40% vs DINOv2-S 81.1% (marginal improvement).

**ONNX status:** Needs testing. Architecture is identical to DINOv2 family, which exports cleanly; no known blockers. Low-risk experiment.

**MIGraphX:** Likely compatible — same op set as DINOv2.

**Effort:** Low if ONNX export works. Re-export, validate parity on a short campaign, A/B.

**Note:** Marginal quality gain (~0.3%). Worth testing precisely because it's a true drop-in — no projection changes, no cold-start, no snapshot incompatibility.

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
| 5 | **Qwen3-0.6B upgrade** | High | Generation quality + language region | Deferred |
| 6 | **DINOv3-ViT-S ONNX export test** | Low | Visual region (drop-in) | Deferred |
| 7 | **DER++ stale-output blocking** | Low–Med | Replay consolidation | Deferred |
| 8 | **EfficientAT-mn10 ONNX export test** | Low | Audio env encoder | Deferred |
| 9 | **M2D-CLAP ONNX export test** | Med | Audio env encoder (+3 mAP) | Deferred |
| 10 | **SigLIP2-ViT-B/16** | Med | Visual (Phase 7+ deferred) | Deferred |
| 11 | **LISTA unrolled encoder** | Med | Sparse coding quality | Deferred |
| 12 | **Gemma 3 1B IT** | Med–High | Generation (lower priority than Qwen3) | Deferred |
| 13 | **usearch migration** | Med | HNSW footprint | Deferred |
