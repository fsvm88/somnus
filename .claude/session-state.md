# Session state — somnus

> **Last updated:** 2026-06-28 (late evening, Item 7 Phase 2 complete)
> **Branch:** `main`

## Status: Items 0–6 complete + Item 7 Phase 2 complete. Item 7 Phase 3 queued. Item 5 (Qwen3-0.6B) deferred to final.

## Completed this session (2026-06-28, evening)

### 6. DINOv3-ViT-S Encoder Swap ✅ INTEGRATED (commit 43be241)
- **Exported:** DINOv3-ViT-S (facebook/dinov3-vits16-pretrain-lvd1689m) to ONNX (87 MB)
- **Output:** 384-dim (identical to DINOv2-S) — true drop-in replacement
- **Integration:** VideoEncoder prefers dinov3_small.onnx, falls back to dinov2_small.onnx
- **A/B script:** scripts/ab_dinov3.sh (5 sessions × 2 cycles control vs treatment)
- **Status:** Ready for parity validation; zero-configuration encoder swap active

## In Progress (2026-06-28, post-Item-6)

### Item 7: DER++ Auto-λ + Stale-output Blocking 🔄 (Low-Med effort)
- ✅ Code preparation: Updated LM_HIDDEN (896→1024) constants across crates
  - REGION_DIMS[4]: 896 → 1024 (language region)
  - CROSS_MODAL_DIM: 2176 → 2304
  - MLP fc2 output: [1024, 4480] → [1024, 5120]
  - Commits: 6335b42
- ✅ Base model ONNX export: `models/qwen3_lm_base.onnx` (1024-dim)
  - Qwen3-0.6B downloaded and exported successfully
- ⏳ Remaining steps (queued for next session):
  1. KV-cache export: `python3 scripts/export_kv_cache.py --model-id Qwen/Qwen3-0.6B --kv-size 2048`
  2. LoRA re-training: `python3 scripts/train_lora.py` (reuse existing training_data.jsonl or recollect)
  3. INT8 quantization: `python3 scripts/quantize_kv.py --prefill`
  4. A/B validation: 5 sessions × 2 cycles (control: Qwen2.5-0.5B, treatment: Qwen3-0.6B)
- 📊 Cold-start: language region, language head, projection fc2 all have incompatible dims (old weights dropped)

### Item 7: DER++ Auto-λ + Stale-output Blocking (Low-Med effort)
- ✅ Phase 1 Foundation: Infrastructure for stale-output detection (commits: 1cac096, 4c1114e)
  - Cosine similarity helper
  - DER++ constant: `STALE_COSINE_THRESHOLD = 0.90`
  - SleepStats tracking: `stale_traces_skipped: usize` field
- ✅ Phase 2: Stale-output detection in NREM replay (commit: 65ceaf7)
  - `PerceptionProcessor.infer_episode_outputs()` method added
  - Per-batch staleness detection: infer current model predictions, compare with stored outputs via cosine sim
  - Skip replaying samples where mean cosine < threshold
  - Filter both training and projection distillation batches
- ⏳ Phase 3: Auto-λ adaptive loss weighting (deferred)
  - Build on Phase 2 staleness implementation
  - Implement running variance estimate for MSE/DER++ loss reweighting

## Completed this session (2026-06-28, late evening)

### 2. High replay ratio A/B validation ✅ PASS, HARDENED (commits 2d59d65, 3b29f96)
- A/B campaign: N=1 control vs N=4 treatment, 5 sessions × 2 cycles each
- **Result:** treatment avg output_mse 0.159467 vs control 0.221494 = **−28.0% improvement**
- Per-session variance: early cycles show regression (c1,2: +52.4%), later cycles strong gains (c3,4,5: −21% to −49%)
  - Pattern consistent with replay warmup: N=4 initially unstable, converges stronger than N=1 by cycle 5
- Hardened into `train_campaign.sh`: `--nrem-replay-epochs 4` now default
- Fixed comparison script to use `mean_output_mse` (not non-existent `output_mse`)
- Verdict: PASS — replay ratio 1→4 improves consolidation per theory (aggregate−28% outweighs early variance)

## Completed this session (2026-06-28, evening)

### 2. High replay ratio A/B ✅ IMPLEMENTED (commits 63b95f3, 28c8e9e)
- Added `--nrem-replay-epochs N` flag (default 1) in main.rs and sleep.rs
- Wrapped batch loop in epoch-based loop; loss EWMA carries across epochs for continuous LR scheduling
- Created `scripts/ab_replay_ratio.sh` for N=1 (control) vs N=4 (treatment) A/B validation
- Reference: arXiv:2505.12512 (replay ratio 4–8× improves consolidation at matched sparsity)
- Ready for A/B campaign: `bash scripts/ab_replay_ratio.sh [SESSIONS=N CYCLES=M]`

### 4. Remove dead candle deps ✅ DEPLOYED (commit 28c8e9e)
- Removed unused `candle-core` and `candle-nn` from workspace.dependencies
- No in-tree usage found; clean-up only

## Completed this session (2026-06-28, earlier)

### 0. bincode → bitcode migration ✅ DEPLOYED (commit 57c87a1)
- Resolved RUSTSEC-2025-0141 (archived maintainer, no future security fixes)
- Replaced unmaintained `bincode` 2.x with actively-maintained `bitcode` 0.6
- WARM_VERSION bumped 1→2 (version gate for format change)
- Graceful fallback: old bincode dirs trigger cold rebuild via existing load_snapshot pattern
- All tests pass including warm_dump_load_round_trip

### 1. BatchTopK SAE ✅ DEPLOYED (commits 4663a2f, af6cb7f, 25d7496)
- A/B validation PASS: +3.6% MSE (within +5% threshold), per-region cosines stable
- Batch-quantile k-WTA threshold: snapshot duty-cycle → find threshold τ → apply to all
- Duty-cycle update order-independent (snapshot→threshold→decay pattern, b decays total)
- Deployed as default: `batch_topk: true` in PerceptionProcessor, `--batch-topk default_value="true"`
- Per-sample top-k available via `--batch-topk=false` for debugging/comparison
- Rationale: high variance on small datasets should not gate theoretically-grounded improvements at scale

## Queued work (next)

See `ai-docs/gaps.md` for full detail and priority order. Reordered after Item 6:

1. DER++ auto-λ + stale blocking (item 7, low–med effort) — **IN PROGRESS**
2. LISTA unrolled encoder (item 11, med effort, multi-step ISTA per region)
3. Qwen3-0.6B upgrade (item 5, high effort: LoRA rerun, projection update, KV re-export, training) — **DEFERRED TO END**

---

### Q-C — Hot-loop allocation elimination + streaming output ✅ DONE (2026-06-26)

Commit 6b39194. Eliminates ~80 MB of malloc/free per 128-token generation in `generate_kv()`:
- `KvCache.logits_scratch` (vocab_size floats): `copy_from_slice` instead of `to_vec()` — no 607 KB malloc per step
- `KvCache.new_kv_scratch` (48 × 128 floats): `copy_from_slice` + field destructuring instead of 48 Vec allocs per step
- Static `PAST_KEY/PAST_VALUE/NEW_KEY/NEW_VALUE` `&str` arrays: eliminates 96 `format!()` + String allocs per step
- Prefill: KV scattered directly into `kv_cache.bufs` (no 48-Vec intermediate, ~7 MB peak saved)
- Streaming output: each token printed + flushed immediately (~40 ms first-token latency)
- `println!()` after decode loop ends stream before JSON response
- `--action-max-tokens` default: 128 → 25 (~1 s at 40 ms/step)

**ORT API note:** `try_extract_tensor::<T>()` in ort 2.0.0-rc.12 returns `Result<(&Shape, &[T])>` — data is already a `&[T]`, use `copy_from_slice` directly. `.as_slice()` does not exist on this type.

**NREM parallelism investigation:** `Module<B>: Send` confirmed. Parallelizing 5 region forward passes requires splitting `PerceptionModel` into per-region sub-models (separate `&mut` borrows needed per thread). Deferred — it's a refactor, not a quick fix.

### Q-A — Zero-copy ArrayView4 for KV decode ✅ DONE (2026-06-25)

Commit a773536. Zero-copy `ArrayView4` for 48 KV buffers per decode step.

### Q-B — INT8 quantization of KV-cache models ✅ DONE (2026-06-25)

Commit e741fa2. **1.76× decode speedup** (70ms → 40ms/step). 128-token generation: ~9s → ~5s.
- No VNNI (Ryzen 9 5950X, AVX2 only) — gain is from bandwidth reduction (1.9GB → 0.5GB per model).
- `reduce_range=True` required on this CPU.
- Token-match A/B: 12/15 on real embeddings (PASS).
- `action.rs` prefers `_int8` variants when present, falls back to FP32.
- Script: `scripts/quantize_kv.py` (also benchmarks; `--skip-quantize` to bench-only).
- INT8 files not in git (`.gitignore`); regenerate: `.venv/bin/python scripts/quantize_kv.py --prefill`.

---

### KV-cache generation — DONE (2026-06-25)

**ADR 0034.** Two new ONNX models: `models/qwen25_lm_prefill_kv.onnx` + `models/qwen25_lm_decode_kv.onnx` (each ~1.9 GB). KV_SIZE=2048.
**Speed:** Old path ~530ms/step → KV path 68ms/step (ORT CPU) → 7.8× speedup. 128 tokens: 68s → 8.7s.
**Rust:** `KvCache` struct + `generate_kv()` in `action.rs`. Falls back to old path if models absent or prefix > KV_SIZE.
**Export:** `scripts/export_kv_cache.py --kv-size 2048`. Numerical equivalence: 18/20 tokens match (PASS).
**Key notes:** `transformers 5.12` rotary_emb is on `model.model` not per-layer; KV_SIZE=16384 is slower than no-KV on CPU (461ms/step); use `Vec<(String, ValueRef<DynValueTypeMarker>)>` for ORT dynamic inputs.

### LoRA pipeline — DONE (2026-06-25)

### LoRA pipeline — DONE (2026-06-25)

**Training:** 1544 pairs, 3 epochs, loss 6.37→5.62. Adapters: `memory/lora_output/`. 
**ONNX:** `models/qwen25_lm.onnx` (1977 MB, LoRA-merged). Backup: `models/qwen25_lm_pretrained.onnx`.
**A/B result:** proj_loss −5.8% (control=0.002086, treatment=0.001966). PASS. Commit 396ccb1.

**ONNX export note:** PyTorch 2.12 requires `dynamo=False` + `Qwen25LMWrapper` (model.model + lm_head, returns last_hidden_state AND logits). Fixed in `scripts/train_lora.py`. A/B script: `scripts/ab_lora.sh`. Symlinked control models dir: `models_lora_ab_control/` (remember to include `.onnx.data` files in symlinks).

---

## Completed this session (2026-06-21)

| Item | What changed | Result |
|------|-------------|--------|
| **MLP projection A/B** | `--linear-proj` flag added; A/B script `scripts/ab_mlp_proj.sh`; GainWorker takes `linear_proj: bool`; branched `project_flat()`, `projection_step_batch()`, `proj_weights_flat()`, `proj_weight_norm()` | A/B: avg proj_loss control=0.002160 vs treatment=0.002094 (−3.1%). INCONCLUSIVE (<5%). MLP kept as default (marginal signal in expected direction). |
| **LoRA pipeline (ADR 0033)** | `--export-training-data`, `--export-proj-weights` flags; `write_training_pair()` inline at both language ingest call sites; `run_sleep_cycle()` gets `training_data_path`; `scripts/train_lora.py` with peft/HF; `scripts/export_models.py` ONNX re-export note | ADR 0033 written. Qwen2.5-0.5B HF checkpoint downloaded to `models/qwen25_hf/`. Python venv at `.venv/`. |
| **Training data collection** | 3 failed attempts (NREM-only, wake-time with ASR, TEXT_RATE=20 stall); final: TEXT_RATE=0.5 with 20 sessions | **1548 pairs** in `memory/lora_training_data.jsonl` |

---

## Completed this session (2026-06-21, earlier)

| Item | What changed | Result |
|------|-------------|--------|
| **GPU generation (Task 1)** | `qwen25_lm_fixed.onnx` exported at `[1, 1024, 896]` (no LM head); Rust extracts `hidden[seq_len-1]` and computes logits via `embed_table^T` dot. `--gpu-generation` flag gates it. ADR 0031. Commits c7f43b4, 332c7c1, 3d619ce. | Token-match A/B confirmed correctness. Timing: GPU 2.1s/token vs CPU 0.14s/token on gfx1201 (RDNA 4) — GPU is 14× slower due to immature MIGraphX support. `--gpu-generation` is opt-in; default uses CPU. ADR updated with honest timing. |
| **MLP projection (Task 2)** | `ProjModel` updated to 2-layer MLP: fc1 [640→1024] + ReLU + fc2 [1024→4480]. `proj_weights_flat()` returns [W1\|W2]. `gain_from_episode()` updated for 2-layer forward. Old `decoder_weights.bin` incompatible — graceful cold-start with ADR 0032 log message. ADR 0032. Commit f9e5faa. | proj_loss plateau at ~0.0007 (sessions 46–50) triggered M. MLP should reduce proj_loss further; evaluate over 3–5 sessions post-deploy via `mean_proj_loss`. Persistent hallucination is not expected to change — that's L's domain. |
| **Dep updates** | `cargo update` (100+ semver bumps); cpal 0.17→0.18 (unified `cpal::Error`, `StreamConfig` now `Copy`); rubato 0.16→3.0 (API rewrite: `Async::new_sinc` + audioadapter buffers); safetensors 0.7→0.8, tokenizers 0.21→0.23, sysinfo 0.38→0.39 (no API changes); bincode 1→2.0.1 (`serde::encode_to_vec`/`decode_from_slice` API; `bincode-next` considered and rejected). Commits 9a6121e, ac41791, 6652661. | All tests pass, clippy clean. bincode wire-format change: existing `records.bin` sidecars silently dropped on first run (graceful fallback already in place). |
| **Completed 2026-06-19–20** | ORT log suppression (a874166), live test (R1+R2+R3 generating 42 tokens, complete hallucination as expected). | — |

---

## Completed this session (2026-06-17)

| Item | What changed | Result |
|------|-------------|--------|
| **B: --lang-use-lm** | `ActionDecoder.text_hidden_states()` wired into `ingest_language_line`; `--lang-use-lm` flag in `main.rs`; hardened into `train_campaign.sh` | A/B confirmed 4× p_lang improvement (0.557→0.85–0.91). Commit 7eea3ae. |
| **lm-on-mixed campaign** | s41–50 × 2 cycles continuing mixed_campaign memory | p[4] peaked 0.910 (s46c1), stable 0.80–0.91 from s42+; MSE 0.026–0.053 |
| **D: SAE L1 coding loss** | `--sae-l1-coeff` flag (default 0.0) in `main.rs`; `sae_l1_coeff` field + `set_sae_l1_coeff()` on `PerceptionProcessor`; L1 term in `fused_replay_wake_step_batch` NREM loop | A/B result: λ=0.1 made MSE 27% WORSE (control 0.315 vs treatment 0.398 avg s3–5). Item D closed. ADR 0025. |
| **J/K/L/M item review** | Checked per-region cosines at steady state (s43–50 mixed_campaign). Speech=0.973 (strongest), language=0.843, visual=0.885. J (Whisper-base) not applicable — speech is already best region. K not triggered (Phase 7+ grounding). L not triggered (proj_loss still declining: 0.0023→0.0007). | Items J, K, L, M all deferred. |
| **H: Magnitude pruning A/B** | Cold A/B inconclusive (regions only 1.2× baseline). Warm A/B (s51–53 from 50-session memory, regions at 2.7×): 2 thresholds failed — env cosine −0.065 (persistent, not transient); gist +0.069 (positive direction). Cascade: 4384→2192→1096→548 units pruned on consecutive cycles (no prune cooldown). warm_recall_l2 average PASS is timing artifact. **FAIL. ADR 0026 closed.** Pruning stays at --prune-fraction 0.0. |

---

## Campaign results summary

| Campaign | Sessions | output_mse (final) | p_lang (peak) | Notes |
|---|---|---|---|---|
| shorts-only | 30 | 0.498 | None (dead) | No ASR in short clips |
| mixed (shorts+H3) | s1–30 | 0.049 | 0.557 | Language alive; all regions growing |
| wiki-on-mixed | s31–40 | 0.028 | 0.564 | Text injection not additive for p_lang |
| lm-on-mixed | s41–50 | 0.026–0.053 | **0.910** | --lang-use-lm; shattered embed-table ceiling |

---

## Queued gaps (from 2026-06-17 architectural review)

Three concrete gaps identified in the architecture review. Ordered by severity:

| # | Gap | What to do | Priority |
|---|-----|------------|----------|
| **Q1** | Qwen LM on CPU during NREM | **Done + A/B confirmed (2026-06-18).** Exported `qwen25_text_hidden.onnx` — static `[1, 128, 896]`, no LM head — loaded via `migraphx_session`. `text_hidden_states()` uses GPU path when present; CPU fallback otherwise. A/B (5s×2c fresh memory, h3h3+shorts): proj_loss 0.0021 vs 0.0021 (+2.1%, PASS). | Medium |
| **R1** | Memory recall in generation | **Done (2026-06-18).** `generate()` now takes `recalled_episodes: &[Vec<f32>]`. Query handlers call `query_recalled_episodes()` to fetch top-k warm HNSW results (default k=3, `--recall-k` flag), filtered for self-match, reversed so closest-recalled is nearest to query tokens. Startup query gets empty recall. ADR 0028. | — |
| **R2** | Multi-turn conversation | **Done (2026-06-18).** `ConvTurn` ring buffer on `ActionDecoder`; `CONV_HISTORY_BUDGET=256` tokens; oldest-first eviction at push-time; EOS stripped from stored response_ids; cleared at sleep entry via `clear_conversation()`. Layout: `[recalled | current_episode | history | query]`. `tracing::info!` logs `history_turns`/`history_tokens`/`seq_len` per call. ADR 0029. Commit ae29a70. | — |
| **R3** | Introspection Phase 1 | **Done (2026-06-19).** Introspective queries (keyword-matched: "what did you hear", "recap", etc.) trigger injection of the 8 most recent `text_target` entries from the hot buffer, decoded and prepended as `[Recent perceived text: ...]` before calling `generate()`. `EpisodicBuffer::recent_text_targets`, `ActionDecoder::decode_ids`, `is_introspective_query`, `build_query_with_history`. Wake-local (hot buffer cleared at sleep). ADR 0030. Commit 615af1f. | — |
| **Q2** | Pruning disabled, regions grow monotonically | `--prune-fraction 0.0` after magnitude-based A/B failure. Grown units (>N_UNITS=1024) are clamped by `to_dense_n` — unreachable but accumulating. No pruning strategy yet works. Item I (knowledge distillation) is the correct next step once regions grow large enough to matter (trigger: any region > 2× baseline). | Low — not a current bottleneck |
| **Q3** | Top-down prior is weak in practice | Language region prior_cosine 0.843 at steady state. The McGurk-style disambiguation described in architecture.md is not yet realized. β=0.05 (via S4) is a soft uniform blend, not a reliable predictive-coding feedback loop. Would require much stronger per-region priors before disambiguation is useful. | Low — research frontier, no clear fix without better prior signal |

---

## Interaction capability queue (committed roadmap — 2026-06-17)

R1, R2, R3 all done (see above).

### M — MLP projection ✅ DONE (2026-06-21)

**ADR 0032.** Commit f9e5faa. proj_loss plateau at ~0.0007 across sessions 46–50 triggered M. 2-layer MLP (640→1024→5×896, ReLU) cold-started (architecture incompatible with old linear weights). Evaluate via `mean_proj_loss` over next 3–5 sessions. Persistent hallucination unchanged — that's L's job.

---

### L — LoRA fine-tuning of Qwen2.5 ✅ DONE (2026-06-25)

**ADR 0033.** 1544 training pairs, 3 epochs. A/B: proj_loss −5.8% (PASS). `qwen25_lm.onnx` deployed. Commit 396ccb1.

---

## Open items

### Tier 1 — trigger-gated (not yet met)

| # | Source | Item | Trigger |
|---|--------|------|---------|
| A | ADR 0006 | Cold archive O(1) promotion index | `promote()` called AND cold > 500k records |
| C | ADR 0006 | Per-region sparse activation storage | Replay analysis shows episode vectors insufficient |

### Tier 2 — conditional / quality-gated

| # | Source | Item | Trigger |
|---|--------|------|---------|
| E | ADR 0009 | Separate optimizer for output heads | Conflicting gradient directions observed |
| F | ADR 0010 | Post-k-WTA prior injection | Pre-k-WTA injection shows poor `prior_cosine` uptake |
| G | ADR 0010 | Learned per-region scalar gate | **Closed** — gist −0.061 (FAIL); S4 never fired in control (comparison confounded); ADR 0027 |
| H | ADR 0018 | Activation-frequency pruning (+ base magnitude pruning) | **Closed** — magnitude pruning failed warm A/B; activation-freq pruning has same root problem (mid-run encoder change); grown size (2.7×) not a bottleneck |
| I | ADR 0018 | Knowledge distillation compression | Grown regions exceed 2× baseline |

### Tier 3 — model swaps / Phase 7+

| # | Source | Item |
|---|--------|------|
| J | ADR 0003 | Whisper-tiny → Whisper-base / WavLM-base |
| K | ADR 0003 | DINOv2-S → SigLIP-B |
| L | ADR 0012 | LoRA fine-tuning of Qwen2.5 |
| M | ADR 0012 | MLP projection (640→5×896) |
| N | ADR 0014 | Visual/audio region grounding |

### Closed items (2026-06-17)

| # | Source | Item | Outcome |
|---|--------|------|---------|
| B | ADR 0019/0014 | Language last-token pooling | Done — LM hidden states 4× better; hardened |
| D | ADR 0005 | Full differentiable SAE | Closed — L1 coding term at λ=0.1 destabilised MSE by 27%; item answered |
| H | ADR 0018 | Magnitude/frequency pruning | **FAIL (warm A/B)** — env cosine −0.065 persistent; cascade (no cooldown); activation-freq pruning same root problem. --prune-fraction 0.0 stays. ADR 0026. |
| G | ADR 0010 | Per-region prior beta | **FAIL (cold A/B)** — gist −0.061; S4 never fired so comparison was 0.01 vs 0.05 (confounded); --prior-beta-regions flag ships but not in campaign. ADR 0027. |

---

## Key facts

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--max-warm-records 200000` hardcoded in `train_campaign.sh` (fixes cycle-2 HNSW starvation)
- CAPACITY default is 8192; overrideable via env var
- `--working-memory-size 8` hardcoded
- `--action-distill-only` + `--lang-use-lm` both hardcoded in `train_campaign.sh`
- `--video-stride 2`, `--recall-ef 16`, `--wake-reconstruct-interval 5` hardcoded
- `PRIOR_BETA=0.01` campaign default
- Wake schedule trigger at 602s/cycle; N_UNITS=1024, K_ACTIVE=20, D_PROJ=128, EPISODE_DIM=640
- `qwen25_lm.onnx` runs on CPU only (dynamic seq_len blocks MIGraphX)
- `--sae-l1-coeff 0.0` default (off); leave off — A/B showed harm
- TEXT_RATE for wiki injection must stay ≤ 1/sec (higher rates stall the session via drain-loop starvation)

---

## MIGraphX constraint (hard)

Never expose ScatterElements or INT64 outputs. Two-graph split, topk/masking on CPU.
`qwen25_lm.onnx` also blocked (dynamic seq_len) — would need fixed-shape re-export + attention_mask input.

---

## Mixed campaign memory location

`memory/mixed_campaign/` (shared across s1–50); best steady state: MSE ~0.028, p_lang ~0.85–0.91.
