# Session state — somnus

> **Last updated:** 2026-05-30
> **Last commit on main:** 1769bdd — Gap fixes: synthetic NREM replay, salience-gated wake reconstruct, salience → NREM LR
> **Branch:** `main` (clean, not yet pushed to Codeberg)

## Status: Gap queue complete — ready for first post-roundup campaign

---

## Completed this session

Gap queue items 1–3 committed (ADR 0019):

| Item | Commit | What changed |
|------|--------|--------------|
| 1 | 1769bdd | REM synthetics pushed to episodic after clear; blended regionals; source-pair L2 as salience |
| 2 | 1769bdd | `record_episode` returns salience; `wake_reconstruct_step` gated on salience > 2.0 |
| 3 | 1769bdd | `nrem_lr` takes `batch_salience`; novel traces get up to 2× LR; sentinel f32::MAX → 0.5 novelty |
| 4 | DROPPED | Last-token pooling rests on false premise: static embed table, not Qwen forward pass; mean-pool is better |

---

## Next campaign (run this first)

```bash
SESSIONS=10 CYCLES=5 CAPACITY=8192 \
MEMORY_DIR=memory/campaign_post_roundup \
EXTRA_ARGS="--lang-oversample 9 --asr-min-chars 5 --working-memory-size 8 --max-warm-records 200000" \
scripts/train_campaign.sh
```

**Watch for:**
- `grew=true` in metrics — should fire on novel content (gap B fixed threshold)
- `prior_cosine_mean[3]` (symbolic) diverging from `[0]` (speech) — cross-modal learning signal
- `synthetics_produced > 0` per cycle (gap 1 working)
- `mean_output_mse_wake` lower than pre-roundup baseline (gap 2 effect: fewer familiar-frame grads)
- Any NaN/inf (GRU adds ~852k params; first campaign with new binding + salience LR)

---

## Remaining gap queue

### Medium effort

5. **Multi-head binding** — 4 heads × 32-dim each, output still 640-dim. Lets binding attend to different cross-modal aspects simultaneously.

6. **Action decoder coherence signal** — during NREM, run inference-mode generation on sampled past episodes, measure perplexity against Qwen prior; high-perplexity = incoherent, drives projection training.

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
- GRU_HIDDEN = 256 in binding area
- All existing memory snapshots are **incompatible** — always use fresh --memory-dir
- WAKE_RECONSTRUCT_SALIENCE_THRESHOLD = 2.0 (below same-content p95 of 4.1)
- LR_SALIENCE_CAP = 10.0, LR_SALIENCE_MAX_SCALE = 2.0 (max NREM LR = 2e-3)
- Language last-token pooling is NOT low-effort: requires Qwen LM forward pass (medium effort, no ADR yet)

---

## Key findings this session

- `synthesize_episode` returns episode vec only (not regionals) — blended regionals computed separately in sleep.rs
- `record_episode` now called before `wake_reconstruct_step` at all wake call sites (audio, video, language)
- Pre-existing `clippy::needless_range_loop` in `settle_episode` fixed as part of this commit
