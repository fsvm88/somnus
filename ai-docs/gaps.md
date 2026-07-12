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

2. **Auto-λ:** ❌ REJECTED (2026-07-08, A/B FAIL) — kept as opt-in-only, off by default
   - `PerceptionProcessor.fused_replay_wake_step_batch` (`crates/somnus-perception/src/lib.rs`)
     combined `sae_loss` (current-batch SAE reconstruction) and `wake_loss` (output-head MSE
     against stale stored outputs) as an unweighted sum — an implicit, hand-tuned λ=1.
     Implemented (2026-07-06, commit `f2fd185`) an opt-in `--der-auto-lambda` alternative:
     tracks a running EWMA mean/mean-square of `wake_loss` (α=0.05, seeded from the first
     sample), derives its variance, and weights `wake_loss` by `1/variance` (clamped
     `[0.1, 10.0]`) — homoscedastic uncertainty weighting instead of the hand-tuned constant.
   - **A/B result (2026-07-08, `scripts/ab_der_auto_lambda.sh`):** 5 sessions × 2 cycles,
     fresh memory both arms, h3h3/shorts corpus. `mean_output_mse`: control (unweighted) avg
     0.0487, treatment (`--der-auto-lambda`) avg 0.0847 (**+73.8%**, worse). Regression held
     in all 10/10 cycles, no reversals (range +16% to +937% per cycle) — a consistent, decisive
     failure, not noise. Likely cause: the EWMA variance estimate is too small/noisy early in
     a cycle, producing large `1/variance` weight spikes that destabilize consolidation instead
     of damping a noisy term as intended.
   - **Decision:** do not adopt. `--der-auto-lambda` remains an opt-in flag defaulting to
     `false`; no code revert needed. Unit test `der_auto_lambda_stays_finite_and_loss_decreases`
     (`crates/somnus-perception/tests/phase5_smoke.rs`) stays as a finite/does-not-diverge guard
     on the code path, not an endorsement of using it.
   - Not planned to revisit unless a different variance estimator (e.g. windowed instead of
     EWMA, or a floor on the sample count before weighting kicks in) is proposed.

**Which part of somnus:** NREM consolidation replay loop (`crates/somnus/src/sleep.rs` calls into
`PerceptionProcessor::fused_replay_wake_step_batch`, `crates/somnus-perception/src/lib.rs`).

**Phase 2 Implementation:** `PerceptionProcessor.infer_episode_outputs()` infers current model predictions; NREM batch loop compares vs stored regionals via cosine. Filtered batch trained and projected.

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

## 8. Audio env encoder: M2D-CLAP ⏸️ NOT PURSUED (2026-07-09) — small expected gain, too much risk

**Source:** NTT, March 2025, arXiv:2503.22104. Weights: `nttcslab/m2d` (GitHub).

**What:** ~86M params (ViT-B backbone). Two-stage pretraining: masked audio modeling + CLAP contrastive alignment. AudioSet linear probe mAP: **49.0** vs CLAP-HTSAT-unfused ~45–46 mAP (+3 mAP improvement).

**Dimension change:** 768-dim (up from CLAP-HTSAT's 512). Requires audio-env region input adaptation.

**ONNX status:** Needs testing. ViT-B architecture; masked modeling pretraining sometimes leaves unusual graph ops — medium export risk.

**MIGraphX:** Uncertain (ViT attention ops require GPU op verification); expect to fall back to CPU as with current CLAP.

**Effort:** Medium (export + validation + A/B). Verify ONNX export before committing.

**Feasibility research (2026-07-09), decision: not pursued.** Before implementing, scoped out
export/licensing risk (no code written):
- **Weights**: available with no gate, plain GitHub release ZIPs — this part was fine.
- **No ONNX export precedent found** for M2D or M2D-CLAP anywhere (no issues/forks/posts).
  Real risk of a dead end before even reaching an A/B: dynamic positional-embedding
  interpolation, and an `nnAudio` STFT-based mel frontend (the same STFT-op fragility class
  that EfficientAT-mn10, item 9, was specifically chosen to avoid).
- **License is non-standard**: NTT's custom "Evaluation" agreement (not MIT/Apache) —
  non-commercial, no redistribution, NTT can terminate at will, Japan-law arbitration.
  Plausibly fine for personal research use but a meaningfully different bar than everything
  else in the tree.
- **Expected gain is small relative to that risk.** The +3 mAP figure is on AudioSet
  linear-probe tagging, not on somnus's actual metric (`per_region_mse` through somnus's own
  projection/binding/replay pipeline). Item 9's A/B is a concrete cautionary precedent in
  this exact slot: EfficientAT-mn10 had no claimed quality edge, and its downstream MSE
  still moved sharply worse while a benchmark-adjacent number (recall) moved the "wrong"
  direction — i.e. paper-benchmark quality and somnus-pipeline quality demonstrably diverge here.
  A modest +3 mAP claim is unlikely to survive that translation any better.
- **Decision:** not worth the implementation/export/licensing risk for the expected effect
  size. Not implemented, no code exists. Revisit only if a documented ONNX export of
  M2D-CLAP appears elsewhere, or if CLAP-HTSAT's quality becomes a bottleneck worth a
  higher-risk swing.

---

## 9. Audio env encoder: EfficientAT-mn10 ❌ REJECTED (2026-07-09)

**Source:** `fschmid56/EfficientAT` (GitHub, actively maintained 2025).

**What:** ~5–20M params (MobileNetV3, width_mult=1.0). Pure CNN — no attention ops, no ScatterElements. AudioSet pre-trained.

**Implementation (all done, uncommitted-then-committed together with this result):**
- `Mn10Mel` in `crates/somnus-encoding/src/preprocess/mel.rs` — Rust port of EfficientAT's `AugmentMelSTFT` (Kaldi-style mel filterbank, genuinely different construction from the existing HTK/Slaney code). Parity-tested against the PyTorch reference at 1e-4 tolerance (`crates/somnus-encoding/tests/mel_parity.rs`).
- `scripts/export_efficientat.py` exports the MobileNetV3 backbone only (mel frontend deliberately excluded — baking it in reintroduces an STFT op, defeating mn10's MIGraphX-compat rationale). Verified numerically against PyTorch (rel L2 ~9e-7), zero STFT/DFT ops in the graph.
- New opt-in Cargo feature `mn10_env` (`somnus-encoding`, `somnus-perception`, `somnus`): swaps `REGION_DIMS[1]` 512→960 and the env encoder's session/mel/sample-rate (CLAP 48kHz vs mn10 32kHz).

**Dimension:** confirmed 960 (MobileNetV3 penultimate feature, pre-pooling-head).

**A/B result (`scripts/ab_efficientat_mn10.sh`, 5 sessions × 2 cycles, fresh memory per arm, h3h3/shorts corpus):** region-1 (env) MSE is not a valid cross-arm comparison — CLAP (contrastive, unit-norm) and mn10 (raw pooled-conv, log-mel-scale) have different target distributions, so that number is a scale artifact, not a quality signal. Verdict is based on the mean MSE of the other four regions (speech/visual/symbolic — identical architecture both arms, genuinely comparable):

- control (CLAP-HTSAT) avg = 0.011966, treatment (mn10) avg = 0.030588, **delta +155.6%, worse**
- Regressed in essentially every cycle (9/10), no reversals — decisive, not noise.
- `warm_recall_l2` and `gist_recall_rate` both moved in mn10's favor (+52.6%, +30.7%) but these don't override a 2.5× MSE regression on the shared representation; likely an artifact of the different env-region scale distorting recall geometry rather than genuine improvement.

**Decision: REJECT.** mn10 significantly degrades the shared episode representation despite its ONNX/MIGraphX-compat appeal. `mn10_env` remains an opt-in, off-by-default Cargo feature (not deleted — the implementation is clean, parity-tested, and may be worth revisiting if M2D-CLAP (item 8) also fails and a smaller/cheaper env encoder becomes worth the quality tradeoff for a different reason, e.g. deployment constraints). Default env encoder stays CLAP-HTSAT.

**Quality evidence (pre-A/B):** was already flagged weaker than M2D-CLAP (no direct mAP vs CLAP-HTSAT head-to-head found) — the A/B result is consistent with that prior.

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

## 16. Growth-scaling pilot campaigns (10x growable-core scaling) 🔬 CHARACTERIZED, full campaign not yet run

**Goal:** scale somnus's own growable core (5× k-WTA regions + GRU + binding, `~5,400` units baseline
per `brain-scale-comparison.md`) 10x, toward honeybee-order scale (~54,000 units, ~9,660
units/region) rather than mouse-order (~71M neurons, a ~400x jump). The growth mechanism itself
(`grow_region_seeded`, ADR 0024 §2c) was already implemented and live — this item is about
characterizing its real-world cadence/cost before spending unattended compute on the full run, not
about implementing anything new.

**Pilot 1 (2026-07-09):** 75-cycle run (`SESSIONS=15 CYCLES=5 CAPACITY=8192`) on a tight 10-clip
looping corpus (5× `clips/h3h3/`, 5× `clips/shorts/`). 16 growth events, ~1 per 4.7 cycles
(essentially gate-limited by the 3-cycle cooldown, not signal-limited), `per_region_mse` improved
alongside growth, no plateau. Extrapolated to full 10x scale: ~135 growth events needed, ~633
cycles, **~3.9 h wall-clock** at the observed ~21.8 s/cycle rate — far cheaper than the growth-
mechanisms.md research note's pre-implementation cascade-correlation cost estimate, since the
shipped mechanism is a seeded weight-tensor resize, not a candidate-training loop.

**Open caveat after pilot 1:** the 10-clip corpus repeats every ~10 clips per session: could the
observed growth/quality trend be a memorization artifact of seeing the same 10 clips over and over,
rather than genuine capacity pressure? Untested at 75-cycle scale.

**New corpus added to close that caveat:** `scripts/download_podcasts.sh` (new script, modeled on
`download_h3h3.sh`) — pulls one-to-many episodes per show across 10 distinct podcasts (Lex Fridman,
Joe Rogan, Huberman Lab, Tim Ferriss, Diary of a CEO, Armchair Expert, Conan O'Brien, This American
Life, The Daily, Freakonomics Radio), 10–90 min duration filter. Yielded 56 episodes / 5.6 GB in
`clips/podcasts/` (Lex Fridman/Huberman Lab/Tim Ferriss under-represented — their typical episode
length exceeds the 90-min filter ceiling on most candidates, not a download failure).

**Pilot 2 (2026-07-11):** identical protocol against the expanded 145-clip corpus (h3h3 + shorts +
podcasts, 14.5x pilot 1's corpus size, far less repetition). **Result: no plateau, cadence and
quality trend indistinguishable from pilot 1** — 16 growth events, same gate-limited cadence
(4–7 cycle gaps), `per_region_mse` mean dropped ~0.0135→~0.0039 first-10-vs-last-10 cycles, final
region sizes uniformly `[2048]×5` (vs. pilot 1's uneven `[2048,1984,1984,1984,1984]`). ~19% slower
wall-clock per cycle (25.9 s vs 21.8 s), plausibly from decoding larger/more varied source files,
not from the mechanism itself. Full writeup: `ai-docs/research/growth-mechanisms.md` "Empirical
validation" section.

**Conclusion: the memorization-plateau risk did not materialize — growth is gate-limited, not
corpus-limited.** This was the last open question before committing to the full campaign at the
time (before §17/§18/§19 below existed).

**Pilot 3 / "v2" (2026-07-12): rerun after the §17/§18/§19 architectural fixes.** Identical
protocol to pilot 2 (`SESSIONS=15 CYCLES=5 CAPACITY=8192`, same 145-clip corpus), fresh isolated
`memory/growth_pilot_v2/`, to validate the fixes in a live run before committing to the full
campaign. **Result: 14 growth events (cadence 4–7 cycles, unchanged from pilot 1/2), quality trend
intact** (`per_region_mse` mean 0.0136→0.0048 first-10-vs-last-10 cycles, comparable to pilot 2's
0.0135→0.0039), **k_active scaling confirmed live** (final `region_sizes = [1523]×5`;
`scaled_k(1523) = 30` vs baseline `scaled_k(1024) = 20`, ratio preserved), no crashes attributable
to the changes (5 segfaults, all the pre-existing/documented MIGraphX exit-race, auto-recovered by
`train_campaign.sh` same as prior pilots), total wall-clock 39m49s (~31.9s/cycle avg, vs. pilot 2's
25.9s/cycle — plausibly the extra segfault-restart overhead, not the architectural changes).

**Important recalibration: batch sizes were much smaller than the old flat 64 in practice.**
Novelty values across the 14 events ranged 0.01–0.54 (only one event exceeded 0.3), giving batches
of 17–77 with a **mean of 35.6/event** — nearly half the old flat batch. Total capacity added
over the 75 cycles was 499 units/region (final size 1523) vs. pilot 1/2's 896 units/region
(64 × 14 events, final ~1984–2048) had the old flat batch been used for the same 14 events. This
is §19 working exactly as designed (severity-proportional growth) — but it means **the previous
10x campaign extrapolation (~633 cycles, ~3.9h) assumed the old flat-64 batch and is now stale**.
Recalibrating from this pilot's observed mean batch (35.6/event) and cadence (1 event/5.36 cycles):
reaching the 10x target (~9,660 units/region, ~8,636 units still needed) now projects to **~242
growth events, ~1,298 cycles, ~9.3–11.5h wall-clock** (range spans this pilot's 25.9–31.9s/cycle
observed rates) — roughly **2.4–3x longer than the pre-§19 estimate**, not because growth is
slower or less healthy, but because it's now correctly proportional to actual novelty, and this
corpus/setup mostly produces moderate (not severe) novelty per trigger.

**Not yet done:** the actual full-scale campaign to reach 10x, now estimated at ~9.3–11.5h
unattended (was ~3.9h pre-§19). All three pilots ran in isolated scratch memory dirs
(`memory/growth_pilot/`, `memory/growth_pilot_expanded/`, `memory/growth_pilot_v2/` — untracked,
not the main `memory/`); the full campaign has not been launched. Next step is user-driven: accept
the longer campaign time, or revisit §19's `GROWTH_BATCH_MAX`/curve shape if ~3-4h is a hard
budget constraint.

---

## 17. K_ACTIVE scaling with region growth ✅ DONE (2026-07-12)

**Problem, found 2026-07-12 while reviewing the mechanism ahead of the full 10x campaign:**
`K_ACTIVE = 20` (`crates/somnus-perception/src/region.rs:11`) is a hardcoded const. Neither
`grow_region` nor `grow_region_seeded` (`crates/somnus-perception/src/lib.rs`) touched it — only
`n_units` grew. `growth-mechanisms.md`'s own design note ("k must increase slightly to maintain
the same sparsity ratio") flagged this as needed but it was never implemented.

At baseline (n=1024) sparsity is 20/1024 ≈ 2%. At the 10x target (~9,660 units/region) it would have
been 20/9660 ≈ 0.2% — a ~10x drop in relative sparsity, untested by either characterization pilot
(both only grew regions to ~2x baseline).

**Fix implemented:** `KWtaLayer` gained a per-layer `k_active` field (was a global const lookup);
new free function `region::scaled_k(n_units)` computes `round(n_units * K_ACTIVE / N_UNITS)`
(floored at 1) to preserve the baseline ratio linearly — the simplest option, chosen over
sub-linear scaling since there's no evidence yet that concept complexity should track region size
sub-linearly, and linear is trivially reversible if a pilot shows otherwise. `grow_region`,
`grow_region_seeded`, and `prune_region` all call `scaled_k` after touching `n_units`, so k tracks
region size symmetrically on both growth and pruning. All internal k-WTA selection call sites
(`select_top_k`, `select_top_k_no_dc`, `forward_tracked_batch`, both batched and per-sample paths)
now read `self.k_active` instead of the const. Also fixed four ORT-bootstrap placeholder call
sites and `binding_only` that hardcoded `N_UNITS`/`K_ACTIVE` instead of the layer's live size —
latent bugs that would have produced a wrong-sized placeholder activation for an already-grown
region after a snapshot reload.

New tests: `scaled_k_preserves_baseline_ratio`, `kwta_respects_grown_k_active`
(`crates/somnus-perception/src/region.rs`). Full existing suite (`phase5_smoke.rs`, including
`grow_then_bind_produces_finite_episode`, `grow_then_replay_finite`,
`prune_region_reduces_grown_units`) passes unchanged. Build/clippy/test clean.

**Not yet empirically validated:** no new pilot run to confirm sparsity-preserving k improves
(or at least doesn't harm) `per_region_mse` at multi-x growth — the fix is a priori-motivated
correction of a mismatch between design intent and implementation, not yet A/B'd against the old
fixed-k behavior at scale. Worth a pilot before/instead of the full 10x campaign if the user wants
empirical confirmation.

---

## 18. Scope optimizer reset to the grown region only ✅ DONE (2026-07-12)

**Problem, found 2026-07-12:** `grow_region`/`grow_region_seeded`/`prune_region` all ended with
`self.optimizer = AdamWConfig::new().init(...)` — this reinitialized Adam's momentum/variance state
for the *entire* model (all 5 regions, GRU, binding), not just the grown/pruned region's new params.
Every other region/component already had valid optimizer state (same `ParamId`) that was being
discarded unnecessarily, ~135 times projected over the full 10x campaign.

**Investigated burn-optim's actual mechanism** (`burn_optim::optim::simple::adaptor::
OptimizerAdaptor`, `~/.cargo/registry/.../burn-optim-0.21.0/src/optim/simple/adaptor.rs`) before
changing anything: its momentum state is a `HashMap<ParamId, AdaptorRecord>`, looked up per-param
on every `.step()` call. When a param's `ParamId` isn't in the map (as happens automatically for
the fresh `ParamId::new()` that growth/pruning already assigns to the resized encoder/decoder/
binding-proj tensors), the optimizer treats it as brand new and starts fresh momentum for *that
param only* — no full reset needed to achieve that. Every other param's id is unchanged, so it
keeps its accumulated momentum for free.

**Fix implemented:** deleted the `self.optimizer = AdamWConfig::new().init(...)` line from
`grow_region`, `grow_region_seeded`, and `prune_region` — nothing else changed, since the desired
per-param-fresh-state behavior already falls out of burn's ParamId-keyed HashMap once the full
reset stops overriding it. Confirmed this is unrelated to (and doesn't regress) the separate
`refresh_autodiff_state` mechanism (`crates/somnus-perception/src/lib.rs`), which resets the optimizer
for a different, legitimate reason (a 100-500x NREM slowdown from AutodiffServer graph
accumulation, fixed by re-keying *every* param's ParamId, not just a grown region's) — that
mechanism is untouched and still runs on its own schedule from `sleep.rs`/`main.rs`.

Full `phase5_smoke.rs` suite (including `grow_then_bind_produces_finite_episode`,
`grow_then_replay_finite`, `prune_region_reduces_grown_units`) passes unchanged. Build/clippy clean.

**Not empirically A/B'd:** the fix follows directly from reading burn-optim's source (not just an
assumption), but no pilot was rerun to measure whether preserved momentum actually improves
convergence speed/stability around growth events in practice — the old behavior didn't show visible
harm in the existing pilots either, so there's no regression baseline to compare against.

---

## 19. Novelty-proportional growth batch size ✅ DONE (2026-07-12)

**Problem, raised by user 2026-07-12:** growth batch size (`GROWTH_BATCH = 64`,
`crates/somnus/src/sleep.rs:28`) was a flat constant added every time the gate (
`within_session_gist_rate` failure + 3-cycle cooldown) fired. The *trigger* already reflected a
novelty-like signal (recall-failure rate), but the *amount* added per event didn't scale with how
much unexplained/novel information was actually present — relative growth rate necessarily decayed
over a long campaign simply because a fixed absolute batch was added to an ever-growing denominator,
independent of whether the information stream flowing in was still just as novel.

**Fix implemented:** `crates/somnus/src/sleep.rs` gained `GROWTH_BATCH_MIN=16`/`GROWTH_BATCH_MAX=128`
(the old 64 sits roughly in the middle, so near-threshold triggers land close to the old default)
and three pure, unit-tested functions:
- `gist_novelty(recall_rate, threshold)` — novelty ratio for the gist-based trigger:
  `(threshold - rate) / threshold`, clamped `[0, 1]`. 0 at the pass/fail boundary, 1 at total
  recall failure.
- `mse_novelty(mse, threshold)` — mirror for the MSE-fallback trigger (S1C1, before any gist
  signal exists): `(mse - threshold) / threshold`, same clamp.
- `growth_batch_size(novelty)` — linear interpolation from `GROWTH_BATCH_MIN` (novelty 0) to
  `GROWTH_BATCH_MAX` (novelty 1). Linear was chosen as the simplest starting point, not because
  it's known to be the right response shape — easy to swap for a nonlinear curve if a pilot shows
  otherwise.

The growth loop's three-way signal match (within-session gist → cross-session gist → MSE
fallback) now computes `novelty` alongside the existing `quality_poor` boolean from the same
branch, so each region's batch size reflects whichever signal actually gated its growth. The
cascade-correlation seed cap (`collect_high_residual_inputs`) was bumped from `2 × GROWTH_BATCH`
to `2 × GROWTH_BATCH_MAX` to cover the worst case (every triggered region hitting max novelty) —
seeds cycle via modulo when fewer are available than the batch, so this is a memory bound, not a
hard requirement.

New tests: `growth_batch_size_spans_min_to_max`, `growth_batch_size_clamps_out_of_range_novelty`,
`gist_novelty_zero_at_threshold_one_at_zero_recall`, `mse_novelty_zero_at_threshold_scales_upward`,
`worse_recall_failure_grows_a_bigger_batch` (`crates/somnus/src/sleep.rs`). Full `somnus`/`somnus-perception`
suite passes, build/clippy/fmt clean.

**Deliberately not done:** EWMA smoothing of the novelty signal across cycles. `within_session_gist_rate`
is already an average over every probed episode in the current within-session warm subset (not a
single noisy sample), so it has some inherent stability per-cycle; cross-cycle smoothing was left
out to keep the first version simple. If a pilot shows batch size oscillating unhelpfully between
consecutive growth events, that's the first thing to add.

**Empirically validated 2026-07-12** (§16's pilot 3 / "v2" rerun, same protocol/corpus as pilot 2):
batch sizes did vary sensibly across events (17–77, tracking novelty 0.01–0.54) — the mechanism
works as designed. But total capacity added over 75 cycles was *smaller* than the flat-64 baseline
would have produced for the same 14 events (499 vs. 896 units/region), because most triggers in
this corpus/setup land at moderate, not severe, novelty. That's not a bug in this fix — it's the
flat-64 baseline having (accidentally) been more generous than the actual observed severity
warranted — but it does mean the 10x campaign now needs ~2.4-3x more cycles/wall-clock than
previously estimated. See §16 pilot 3 for full numbers and the revised campaign-time estimate
(~9.3–11.5h, was ~3.9h).

---

## 20. Full 10x growth campaign launched; `campaign_status.py`; `eval_metrics.py` region-name bug fix (2026-07-12)

**§16's full campaign is now running** (not just pilots). Per the user's explicit call, `memory/`
was wiped completely first (all production state + every old A/B/pilot scratch dir — ~17GB, all
gitignored/untracked, nothing lost that wasn't reproducible), so the growable core is regrowing
from scratch on the post-§17/§18/§19 mechanism. Launched via `scripts/train_campaign.sh`
(`SESSIONS=130 CYCLES=10 CAPACITY=8192`, 1300 target cycles, full 145-clip h3h3+shorts+podcasts
corpus, `CAMPAIGN_DIR=memory/campaign_growth10x`), running unattended in the background.

**First observed rate is beating the pilot-3 baseline, not matching it.** At ~440 cycles in:
mean batch ~43.4 units/event (pilot-3: 35.6) and cadence ~1/4.25 cycles (pilot-3: 1/5.36) — both
faster than expected, projecting a **~6.1h total campaign time** against the pilot-3 estimate of
9.3–11.5h. Plausible cause: a from-scratch corpus produces more severe novelty per trigger early
on than pilot 3's partially-trained starting state did — untested whether this holds once the
low-hanging novelty is consumed later in the run. Treat the faster rate as provisional until the
campaign is further along or complete.

**New tool: `scripts/campaign_status.py`.** One-command status report for any `train_campaign.sh`
run (auto-detects the newest `memory/campaign_*` dir, or takes one explicitly) — replaces manually
recomputing progress/growth/ETA from `metrics.jsonl` by hand every check-in. Reports: process
state, cycle progress + ETA to session target, current-session progress + ETA, per-region growth
sizes/deltas, growth-event cadence/batch size, an ETA to a configurable unit target
(`--target-units`, defaults to the 9,660 10x goal), quality trend (early-vs-late MSE window),
recall metrics, latest-cycle timing breakdown, disk usage per memory subdir + free space, and a
health check (stale-log detection, panic scan across session logs). Colored output (light-blue
labels, light-green values; auto-disabled when not a TTY, or via `--no-color`/`NO_COLOR`) grades
the pace-sensitive numbers — rate, growth batch/cadence, per-region growth-so-far, projected total
campaign time, quality trend — green/yellow/red against the pilot-3 baseline (`EXPECTED_*`
constants at the top of the script, sourced from this doc's §16/§19), so "ahead of schedule" vs.
"behind schedule" is visible at a glance instead of requiring the reader to do the arithmetic.
No non-stdlib dependencies.

**Bug found and fixed while writing that script:** `scripts/eval_metrics.py`'s
`REGION_NAMES = ["visual", "env", "speech", "symbolic", "language"]` had indices 0 and 2 swapped
relative to the actual Rust emission order (`RegionId::as_index`,
`crates/somnus-perception/src/region.rs`: 0=AudioSpeech, 1=AudioEnv, 2=Visual, 3=Symbolic,
4=Language — confirmed against `PerceptionState::region_sizes()` and `sleep.rs`'s
`per_region_mse` construction). Every growth-timeline and per-region-MSE chart this project has
ever generated has been mislabeling speech as visual and vice versa (env/symbolic/language were
unaffected — they happened to already be in the right slots). Fixed; verified the eval harness
still runs clean against live campaign data with the corrected labels.

---

## 21. Warm-store HNSW dump leak + broken cross-restart persistence ✅ FIXED (2026-07-12)

**Found while answering a user question about disk usage** ("why is `warm` 11G but `episodes.cold`
41M, when both hold the same 65,798 records?"). Answer led to a real, previously-unknown bug, not
an expected size difference — `episodes.cold` is a single append-only file (`ColdArchive`,
`crates/somnus-memory/src/cold.rs`) that grows cleanly with record count. `memory/warm/` had **461
separate HNSW index dump pairs** (`index-<N>.hnsw.data`/`.graph`, ~24-26MB each) — only one of
which was ever the live index; the rest were pure leak.

**Root cause, traced into `hnsw_rs` 0.3.4:** `SemanticStore::dump()`
(`crates/somnus-memory/src/semantic.rs`) calls `self.index.file_dump(warm_dir, "index")` on every
sleep cycle. `Hnsw::file_dump` only overwrites the fixed `index.hnsw.{data,graph}` name in place
when `!self.datamap_opt`. `HnswIo::load_hnsw` (used by `load_warm()` on process start) hardcodes
`datamap_opt: true` on the returned `Hnsw` **unconditionally**, regardless of whether real mmap
was actually requested (`ReloadOptions::default()` — what this code uses — has `datamap: false`,
so no real mmap ever backs these reads; the flag is a spurious carry-over from the crate's reload
path). Net effect: the very first dump each process makes writes the plain `index.hnsw.data`
correctly; every dump for the rest of that process's life (and every dump in every subsequent
process, since the reloaded index is *always* marked `datamap_opt: true`) instead falls into
`DumpInit::new`'s "can't overwrite" branch, which finds the plain name already taken and picks a
**random 0–9999 suffix** — `index-<N>` where `N` has no meaning (not a record count, not a
timestamp). Nothing in `somnus`'s code ever deleted the superseded files, so warm-store disk usage
grew ~24-26MB **every single sleep cycle**, unboundedly, for the life of any long-running process.

**Second, related consequence — not just a leak, but broken persistence:** `SemanticStore::load_warm()`
only ever looks for the fixed `index.hnsw.data`/`.graph` names. Since real dumps after the first
one always land under a random name, `load_warm()` has silently failed to find *any* saved warm
index on every restart after the first reload, for the entire time this mechanism has existed —
falling back to `"warm dump absent or stale — rebuilding from cold archive"` every time. Not a
correctness bug (cold archive is authoritative, so no data was lost — the rebuild is just wasted
work every restart) but it meant the warm-index persistence path has effectively never fired.

**Fix:** `SemanticStore::dump()` now deletes any existing `index*.hnsw.{data,graph}` in `warm_dir`
before calling `file_dump`. This has two effects: (1) bounds warm-store disk usage to one live
pair regardless of `hnsw_rs`'s internal overwrite decision, and (2) since `DumpInit::new`'s
collision check only picks a random suffix when the plain-named file *already exists*, clearing it
first means every dump now lands under the plain `index.hnsw.data` name again — restoring the
cross-restart persistence path `load_warm()` was always meant to use. Verified live: after the
fix, `memory/warm/` holds exactly one `index.hnsw.data` pair after a dump, and the *next* restart
successfully finds and loads it (no more "rebuilding from cold archive" fallback each time).
On Linux, deleting a file that a process still has mmap'd (if one ever were active) is safe — the
inode stays alive via the existing file descriptor/mapping until it's unmapped; this only affects
what `ls`/`du` see, not correctness.

**Applied live to the running §16/§20 growth campaign:** stopped the campaign cleanly (SIGTERM;
state is snapshotted every cycle, so at most a few seconds of in-progress work was lost), built
and verified the fix (`cargo build/clippy/test -p somnus-memory`, 35/35 tests pass), manually cleaned
up the pre-existing 461 orphaned dump pairs (`memory/warm/` 11G → 192M), and resumed via
`SESSION_START=50 SESSIONS=82` against the same `CAMPAIGN_DIR` (metrics/logs are append-only, so
the trajectory is continuous — cycle 486 to 487 crosses the restart with no gap). Confirmed via a
`Monitor` wait that the very next dump landed under the plain name and the restart after it
reloaded successfully. `memory/warm/` is now 362M (mostly the live index + records sidecar, not
leaked dumps) vs. 11G before the fix.

**Also fixed two `campaign_status.py` bugs surfaced by the restart, since this is exactly the kind
of thing the tool exists to catch:** (1) a crash (`TypeError` on `None.__format__`) when a metric
field is present but JSON `null` early in a session (e.g. `within_session_gist_rate` before the
first gist probe) — `dict.get(key, default)` doesn't catch an explicit `null`, only a missing key;
added a `gv()` helper that treats both the same. (2) `train_campaign.sh`'s `params.txt` is
appended-to (not overwritten) across restarts, so after a `SESSION_START`/`SESSIONS` resume, using
`SESSIONS × CYCLES` alone for the progress target understated the original whole-campaign target
(reported 820 instead of the intended 1310). Fixed by computing the target as
`(SESSION_START + SESSIONS - 1) × CYCLES` instead, which stays correct across any number of
stop/resume cycles as long as `CYCLES` is unchanged.

---

## Priority order (as of 2026-07-11)

> **Note:** the `#` column matches the `##` section number above (`§N`), not chronological/priority
> order. Table numbering used to drift from the section headers — corrected 2026-07-11; if you're
> reading an old cached copy of this table, re-fetch rather than trust remembered numbers.

| # | Item | Effort | Impact | Status |
|---|---|---|---|---|
| ✅ §0 | **bincode migration** (→ bitcode) | Low–Med | Security advisory | Deployed (57c87a1) |
| ✅ §1 | **BatchTopK SAE** | Low | Sparse coding quality | Deployed (25d7496) |
| ✅ §2 | **High replay ratio A/B** (N=1/2/4) | Trivial | Consolidation quality | Adopted 2026-07-06 — N=4 default (-70.9% output_mse/+16.6% wall-clock); N=2 was more efficient (-49.1%/+7.6%) but not chosen |
| ❌ §3 | **DER++ auto-λ (Phase 3)** | Low–Med | Replay consolidation | Stale-output blocking (Phase 2) shipped earlier; auto-λ half rejected 2026-07-08 — A/B FAIL (+73.8% mean_output_mse), opt-in only |
| §4 | **LISTA unrolled encoder** | Med | Sparse coding quality | Deferred |
| ✅ §5 | **Qwen3-0.6B upgrade** | High | Generation quality + language region | Adopted 2026-07-05 (ADR 0036); Qwen2.5 dropped. Gen-quality confirmed (take 3) |
| ✅ §6 | **DINOv3-ViT-S ONNX export test** | Low | Visual region (drop-in) | Deployed (validated 2026-07-05, ai-docs/a_b_dinov3_findings.md) |
| §7 | **SigLIP2-ViT-B/16** | Med | Visual (Phase 7+ deferred) | Deferred |
| ⏸️ §8 | **M2D-CLAP ONNX export test** | Med | Audio env encoder (+3 mAP) | Not pursued 2026-07-09 — small expected gain vs. untested ONNX export + restrictive license |
| ❌ §9 | **EfficientAT-mn10 ONNX export test** | Low→Med | Audio env encoder | Rejected 2026-07-09 — A/B FAIL (+155.6% MSE on comparable regions), opt-in-only `mn10_env` feature kept |
| §10 | **Dedicated text embedding model** | Med | Language encoder | Deferred |
| §11 | **Gemma 3 1B IT** | Med–High | Generation (lower priority than Qwen3) | Deferred |
| §12 | **Episodic binding survey** | — | Binding area | Null result — 4-head cross-attention + GRU is state-of-art under the ORT/MIGraphX constraint (SSM/Mamba alternatives need an unsupported CUDA kernel); not pursued |
| ✅ §13 | **Change-gated video encoder** | Low | Wake-path compute (unmeasured savings) | Deployed 2026-07-06, opt-in via `--video-change-threshold` |
| ❌ §14 | **SVD low-rank check (region/binding weights)** | Low | Capacity efficiency | Null result 2026-07-06 — near-full-rank, not pursued |
| §15 | **Crate hygiene** (usearch migration, burn-ndarray watch, ffmpeg-the-third) | Low–Med | HNSW footprint / maintenance | Dead deps removal already done; usearch/ffmpeg-the-third deferred until raised by a concrete need |
| 🔄 §16 | **Growth-scaling pilot campaigns** (10x growable-core scaling) | Med (pilots done; full campaign launched 2026-07-12, running unattended) | Growable-core capacity | Characterized 2026-07-09/11/12 (3 pilots, no plateau risk found); full 10x campaign launched 2026-07-12 from wiped `memory/`, tracked live via `scripts/campaign_status.py` — see §20 |
| ✅ §17 | **K_ACTIVE scaling with region growth** | Low–Med | Sparse coding quality at scale | Deployed 2026-07-12 — see §17 below |
| ✅ §18 | **Scope optimizer reset to grown region only** | Low | Convergence stability during growth | Deployed 2026-07-12 — see §18 below |
| ✅ §19 | **Novelty-proportional growth batch size** | Med | Growth-mechanism correctness | Deployed 2026-07-12 — see §19 below |
| ✅ §20 | **Full 10x campaign launch + `campaign_status.py` + `eval_metrics.py` region-name fix** | Low (tooling) | Ops visibility / chart correctness | Deployed 2026-07-12 — see §20 below |
| ✅ §21 | **Warm-store HNSW dump leak + broken cross-restart persistence** | Low | Disk usage (unbounded growth) / warm-index reload correctness | Fixed 2026-07-12 — see §21 below |
