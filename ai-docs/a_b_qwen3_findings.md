# A/B Validation: Qwen3-0.6B vs Qwen2.5-0.5B

**Date:** 2026-06-28  
**Duration:** ~2 cycles per session (5 sessions each model)  
**Visual encoder:** DINOv2-S (fallback; DINOv3 disabled due to MIGraphX incompatibility)

## Results

### Memory Retrieval Quality (warm_recall_l2)

| Phase | Qwen2.5-0.5B | Qwen3-0.6B | Delta |
|-------|--------------|-----------|-------|
| Cycle 1 | 0.541932 | 0.545247 | +0.61% |
| Cycle 2 | 0.558813 | 0.449861 | **-19.5%** |
| **Average** | **0.550373** | **0.497554** | **-9.60%** ⚠️ |

**Interpretation:** warm_recall_l2 measures episodic memory L2 retrieval error (lower encoding distance → higher score). Qwen3 shows **worse retrieval performance** on the second cycle, suggesting:
- Embedding space may be less effective for memory queries
- LANG_DIM (896→1024) change may affect binding operations
- Qwen3's larger embedding (1024-dim) may need more data to converge

### Cross-Modal Gist Entropy
- Control: 2.3219 (stable across cycles)
- Treatment: 2.3219 (stable across cycles)
- **Verdict:** No degradation; binding/cross-modal integration unaffected

### Language Ingestion
- Both models successfully ingest language (lang_ingest=1 on cycle 2)
- No crashes or dimension mismatches
- Qwen3 projection weights loaded correctly

## Known Limitations

1. **Short validation window:** 2 sleep cycles per session is minimal. Convergence may require longer runs.
2. **Partial metrics:** Only warm_recall_l2 extracted from tail output. Missing:
   - output_mse (reconstruction error)
   - p_lang (language region activation)
   - proj_loss (projection head loss)
3. **Visual encoder variance:** DINOv2 fallback may mask Qwen3-specific improvements
   - Original plan was DINOv3 → DINOv3 comparison
   - DINOv3 disabled (ORT/MIGraphX incompatibility)
4. **Session interruption:** Previous sessions killed mid-run; fresh data only from this restart

## Recommendations

### Immediate (if Qwen3 is priority)
1. **Re-run with longer cycles:** 5+ NREM cycles per session to allow convergence
2. **Parse full cold archives:** Extract output_mse, p_lang, proj_loss for comprehensive comparison
3. **LoRA fine-tuning:** Current test is base models; LoRA training may show Qwen3 advantages

### Infrastructure
1. **DINOv3/ORT issue:** Escalate MIGraphX CPU backend incompatibility
   - May require GPU execution or different ORT backend
   - Or use alternative visual encoder (SigLIP-B, item K)
2. **Metrics logging:** Add explicit metrics.jsonl output to A/B scripts (not just tail)
3. **Multi-session tracking:** Record per-session metrics to detect convergence patterns

### Next A/B Candidate
If Qwen3 underperforms, consider:
- **Item 2:** High replay ratio (simple hyperparameter, proven benefit)
- **Item 4:** LISTA Top-K encoder (algorithmic improvement)
- **Item 6:** DINOv3 swap (once visual encoder issue resolved)

## Files

- Control results: `memory/qwen25_ab_control/`
- Treatment results: `memory/qwen3_ab_treatment/`
- Cold archives: `episodes.cold` (3.1–3.2 MB each, binary format)
- Script: `scripts/ab_qwen3.sh`
