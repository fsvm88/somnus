# Session state — somnus

> **Last updated:** 2026-05-30
> **Last commit on main:** b40a895 — Gap 5+6: multi-head binding (4×32) + LM hidden-state distillation target
> **Branch:** `main` (clean, not yet pushed to Codeberg)

## Status: Medium gap queue closed — ready for first smoke-test campaign

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| Gap 1 | 1769bdd | REM synthetics pushed to episodic after clear; blended regionals; source-pair L2 as salience |
| Gap 2 | 1769bdd | `record_episode` returns salience; `wake_reconstruct_step` gated on salience > 2.0 |
| Gap 3 | 1769bdd | `nrem_lr` takes `batch_salience`; novel traces get up to 2× LR; sentinel f32::MAX → 0.5 novelty |
| Gap 4 | DROPPED | Last-token pooling rests on false premise: static embed table; mean-pool is better |
| Gap 5 | b40a895 | 4-head binding (N_HEADS=4, D_HEAD=32); same W_q/W_k/W_v params; scale 1/√32; discriminating test added |
| Gap 6 | b40a895 | Projection trains toward LM contextualised hidden states (not static embed-table mean); precomputed before NREM loop; no new ONNX export needed |

---

## Next step: smoke-test run

```bash
SESSIONS=1 CYCLES=1 CAPACITY=512 \
MEMORY_DIR=memory/smoke_$(date +%H%M) \
EXTRA_ARGS="--lang-oversample 9 --asr-min-chars 5 --working-memory-size 8 --max-warm-records 200000" \
scripts/train_campaign.sh
```

**Watch for:**
- No NaN/inf — first run with multi-head binding + distillation target
- `decoder_trainable_records > 0` and `mean_proj_loss` finite (Gap 6 wired)
- `synthetics_produced > 0` (Gap 1)
- `grew=true` on novel content (Gap B threshold)
- `mean_output_mse_wake` reasonable (not exploded)

After smoke test passes, run the full campaign:

```bash
SESSIONS=10 CYCLES=5 CAPACITY=8192 \
MEMORY_DIR=memory/campaign_post_roundup \
EXTRA_ARGS="--lang-oversample 9 --asr-min-chars 5 --working-memory-size 8 --max-warm-records 200000" \
scripts/train_campaign.sh
```

---

## Remaining gap queue

### Hard, needs ADR first

7. **Systematic forgetting** — no mechanism to evict outdated/contradictory cold records. Long-run viability gap. One path: track cold record access frequency; evict least-recently-retrieved above a capacity threshold.

8. **REM concept-level probe** — current gist probe is episodic (k-means on episode vectors). Doesn't measure concept stability across perceptual contexts.

---

## Key facts

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--asr-min-chars 5` for noisy-audio clips
- `--max-warm-records 200000` recommended for long runs
- Wake schedule trigger at 602s/cycle — compute speedups have zero wall-clock effect
- EMA_ALPHA = 0.99 for output head inference copy
- CROSS_MODAL_DIM = 2176 (speech+env+visual+language → symbolic)
- GRU_HIDDEN = 256 in binding area; N_HEADS=4, D_HEAD=32 in binding attention
- All existing memory snapshots are **incompatible** — always use fresh --memory-dir
- WAKE_RECONSTRUCT_SALIENCE_THRESHOLD = 2.0 (below same-content p95 of 4.1)
- LR_SALIENCE_CAP = 10.0, LR_SALIENCE_MAX_SCALE = 2.0 (max NREM LR = 2e-3)
- `qwen25_lm.onnx` already outputs `last_hidden_state` (was there from original export, just not read)
- Projection training: `projection_step_batch` (old, embed-table mean) marked dead_code; `projection_step_batch_distill` is the live path
- Gap 6 distillation adds one lm_session forward per language-tagged trace at NREM start (precomputed, not on hot path)
