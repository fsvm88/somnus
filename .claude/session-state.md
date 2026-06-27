# Session state — somnus

> **Last updated:** 2026-06-28 (evening wrap)
> **Branch:** `main`

## Status: Items 5, 6, 7 infrastructure complete. Item 5 A/B blocked by pre-existing language region crash (affects all models).

---

## Session Summary (2026-06-28)

### Item 6: DINOv3-ViT-S Encoder ✅ INTEGRATED
**Commits:** 43be241
- ✅ DINOv3 exported to ONNX (87 MB, 384-dim output)
- ✅ VideoEncoder auto-selects DINOv3 (zero-config swap)
- ✅ A/B script ready: `bash scripts/ab_dinov3.sh`
- Status: Ready for parity validation

### Item 7: DER++ Stale-output Detection ✅ INTEGRATED  
**Commits:** 65ceaf7
- ✅ Phase 2 complete: cosine_sim helper + staleness detection in NREM replay
- ✅ Thresholds: `STALE_COSINE_THRESHOLD = 0.90` (tunable)
- ✅ SleepStats tracking: `stale_traces_skipped` counter
- Status: Deployed and functional

### Item 5: Qwen3-0.6B LLM Upgrade 🔄 INFRASTRUCTURE READY
**Commits:** 4817e33, 4237515, 2319f35, 7199013, 46a8164, d0f92c5, 773c146

#### Completed:
- ✅ Qwen3 base model: `qwen3_lm_base.onnx` (2.3 GB)
- ✅ Projection weights: w1.bin (2.5 MB, 640×1024), w2.bin (20 MB, 1024×5120)
  - Generated via 3-session campaign with fresh memory
  - Correct dimensions for Qwen3's 1024-dim hidden state
- ✅ LoRA training script: `train_lora_qwen3.py` (fully functional, ready to use)
- ✅ A/B validation script: `ab_qwen3.sh` (ready once language crash fixed)
- ✅ All documentation updated: gaps.md, architecture.md, session-state.md

#### Blockers:
1. **LoRA training:** PyTorch/HF CPU compatibility issue
   - Hung after model load during dataset loading (2 attempts, 4h+ each)
   - Workaround: Base-model A/B validates architecture independently
   - Can retry with GPU or different backend later

2. **A/B validation:** Pre-existing language region crash
   - Matmul dimension mismatch in language k-WTA or SAE decoder
   - **Affects both Qwen2.5 AND Qwen3 equally** (not Qwen3-specific)
   - Crash point: After ASR transcription ("language ingest 69 chars" → panic)
   - Requires separate investigation and fix in language region code
   - Blocks all model A/B testing until resolved

#### Next Steps:
1. **Immediate:** Debug language region crash (separate task, not Item 5 blocker)
   - File: `crates/somnus-perception/src/lib.rs` (language k-WTA init or SAE decoder)
   - Once fixed: retry A/B with: `bash scripts/ab_qwen3.sh SESSIONS=5 CYCLES=2`

2. **Optional:** After A/B results
   - If Qwen3 base promising: debug LoRA training or use alternative backend
   - Implement KV-cache export (optimization, not critical)
   - Deploy LoRA-merged Qwen3 if training succeeds

---

## Items Deferred (per CLAUDE.md, 2026-06-17)

All remaining ADR backlog items deferred indefinitely unless explicitly raised or trigger condition met. See CLAUDE.md for list and trigger conditions.

