# Session state — somnus

> **Last updated:** 2026-05-29
> **Last commit on main:** 719c3db — Housekeeping: gitignore runtime artifacts
> **Branch:** `main` (clean, pushed to Codeberg)

## Status: Ready for new campaign + architectural gap queue identified

---

## Completed this session

All architectural improvements from the roundup committed and pushed:

| Task | Commit | What changed |
|------|--------|--------------|
| B | af9d4d6 | Growth threshold 0.5→0.85; removed broken warm_recall_poor gate |
| C | af9d4d6 | Episodes L2-normalised before HNSW — better cosine-equivalent recall |
| E | af9d4d6 | EpisodeTrace.salience; NREM replays novel traces first |
| F | a6940aa | Symbolic: cross_modal_proj 2176→384; convergence layer not speech copy |
| D | 93ba871 | EmaOutputHeads α=0.99; inference decoupled from autodiff training path |
| G | 98c5775 | --max-warm-records; rebuild_from_recent during sleep |
| A | 958a18c | GRU(640→256) temporal binding; wm_seq replaces wm_mean_vec |
| H | DROPPED | Wake time-boxed; wgpu has zero wall-clock benefit |
| cleanup | 719c3db | .gitignore covers memory/, data/, experiments byproducts; 750k line removal |

memory/ trimmed from ~81 GB to ~400 MB. All pushed.

---

## Next campaign (run this first tomorrow)

```bash
SESSIONS=10 CYCLES=5 CAPACITY=8192 \
MEMORY_DIR=memory/campaign_post_roundup \
EXTRA_ARGS="--lang-oversample 9 --asr-min-chars 5 --working-memory-size 8 --max-warm-records 200000" \
scripts/train_campaign.sh
```

**Watch for:**
- `grew=true` in metrics (B fixed threshold — should fire on novel content)
- `prior_cosine_mean[3]` (symbolic) diverging from `[0]` (speech) — signals cross-modal learning
- Temporal GRU stability in early sessions (may be noisy first 2–3 cycles)
- Any NaN/inf (GRU adds ~852k params, first campaign with new binding)

---

## Architectural gaps — prioritised queue for next session

### Low effort, high value (do first)

1. **Synthetic NREM replay** — REM produces synthetics, measures them, then discards. Wire them into the episodic buffer so the next NREM cycle trains on them. Infrastructure exists; it's just plumbing.

2. **Salience-gated wake reconstruction** — skip `wake_reconstruct_step` when salience is below a threshold (familiar frame, low novelty). Focuses gradient signal. The salience score is already computed at encode time.

3. **Neuromodulation → NREM LR** — wire the per-trace salience score into `nrem_lr()`. High-salience (novel) traces should replay at higher LR. Currently LR schedule ignores novelty entirely.

4. **Language last-token pooling** — change language embedder from mean-pool to last-token of Qwen2.5. Short ASR fragments (5–30 chars) get worse mean-pool representations than last-token (causal LM has full context at last position).

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

---

## Key findings from this session

- Wake is time-boxed at 602s/cycle regardless of per-frame compute cost
- wake_reconstruct is 17ms (not 315ms — 315ms was multi-thread log noise)
- interval=4 experiment: freed compute → more frames → more NREM → +3–5 min sessions (net zero)
- Growth never fired because GIST_GROWTH_THRESHOLD=0.5 + broken warm_recall_poor gate
- Symbolic was a dead-end copy of speech; now a cross-modal convergence layer
