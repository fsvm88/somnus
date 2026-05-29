# Session state — somnus

> **Last updated:** 2026-05-29
> **Last commit on main:** 958a18c — A: Temporal binding via GRU over episode sequence
> **Branch:** `main`

## Status: Full architectural roundup complete — ready for new campaign

---

## All changes committed this session

| Task | Commit | What changed |
|------|--------|--------------|
| B | af9d4d6 | Growth threshold 0.5→0.85; removed broken warm_recall_poor gate |
| C | af9d4d6 | Episodes L2-normalised before HNSW — better cosine-equivalent recall |
| E | af9d4d6 | EpisodeTrace.salience; NREM replays novel traces first |
| F | a6940aa | Symbolic: cross_modal_proj 2176→384; convergence layer not speech copy |
| D | 93ba871 | EmaOutputHeads α=0.99; inference decoupled from autodiff training path |
| G | 98c5775 | --max-warm-records; rebuild_from_recent during sleep |
| H | DROPPED | Wake time-boxed; wgpu has zero wall-clock benefit |
| A | 958a18c | GRU(640→256) temporal binding; wm_seq replaces wm_mean_vec |

---

## Key architectural state

- **Episode space changed** (C + A + F): existing snapshots are INCOMPATIBLE
- **Start fresh `--memory-dir`** for the next campaign
- **EPISODE_DIM = 640** unchanged — memory/serialisation interfaces stable
- **New params in binding**: GRU ~688k + temporal_proj ~164k ≈ 852k total
- **Growth should now fire** at gist_recall < 0.85 (was 0.5, rarely hit)
- **Symbolic now a convergence layer**: cross-modal context, not speech echo
- **EMA heads**: inference runs on NdArray copy (no autodiff), training on live model

---

## Next campaign recommendation

```bash
SESSIONS=10 CYCLES=5 CAPACITY=8192 \
MEMORY_DIR=memory/campaign_post_roundup \
EXTRA_ARGS="--lang-oversample 9 --asr-min-chars 5 --working-memory-size 8 --max-warm-records 200000" \
scripts/train_campaign.sh
```

Watch for:
- `grew=true` firing in metrics (B fixed threshold)
- `prior_cosine_mean[3]` (symbolic) behaving differently from `[0]` (speech)
- `gist_recall_rate` trajectory vs old campaigns
- Any instability from GRU temporal context (first sessions may be noisier)

---

## Key facts

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--asr-min-chars 5` for noisy-audio clips
- `--max-warm-records 200000` for long runs (~500 MB HNSW cap)
- Wake schedule trigger at 602s/cycle — not fill-based; compute speedups have no wall-clock effect
- H was dropped: wake_reconstruct is 17ms (not 315ms); interval=4 experiment proved time-box dominates
- EMA_ALPHA = 0.99
- CROSS_MODAL_DIM = 2176 (speech+env+visual+language inputs to symbolic)
- GRU_HIDDEN = 256
