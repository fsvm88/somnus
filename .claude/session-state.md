# Session state — somnus

> **Last updated:** 2026-05-30
> **Last commit on main:** 5a8e515 — Recalibrate thresholds for L2-normalised episode vectors
> **Branch:** `main` (clean, not yet pushed to Codeberg)

## Status: Smoke test passed — calibration fixed — ready for Gap 7 + 8

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| Gap 5 | b40a895 | 4-head multi-head binding (N_HEADS=4, D_HEAD=32) |
| Gap 6 | b40a895 | LM hidden-state distillation target for projection training |
| Instrumentation | b952520 | Sleep phase timings (nrem/rem/spill/snapshot/distill ms), wake encode times (speech/env/video/gain), lm_hidden_computed; timing_breakdown.png chart |
| Tests (+16) | b952520 | nrem_lr schedule, kmeans, seq_k=10 binding path, replay loss decrease, wake_reconstruct_finite |
| Tests (+11) | 11f2e22 | all_five_ingest_paths, prior_blending, settling, homeostatic_decay, synthesize_episode, prune_region, text_target_survives_push, clear, query_with_dist, sample_values, rebuild_from_recent |
| Profiling | 0d62427 | scripts/profile.sh (ORT traces + rocprof + flamegraph), release-debug profile, wake timing in metrics, ActionDecoder::end_profiling |
| Docs | 7c85f2c | ADRs 0007–0012, architecture.md, roadmap.md synced to current code |
| wake_timing chart | 8025106 | wake_timing.png: stacked encode+gain totals + per-episode averages |
| Calibration fixes | 5a8e515 | 4 thresholds recalibrated for L2-normalised unit vectors (see below) |

### Calibration fixes (critical — all from smoke test)

All four constants were calibrated before L2-normalisation was added to `record_episode`.
Max L2 distance for unit vectors = 2.0; old values assumed ~5–10x larger un-normalised magnitudes.

| Constant | Was | Now | Effect of old value |
|---|---|---|---|
| `REM_SYNTH_MIN_L2` | 2.0 | 1.0 | synthetics never fired |
| `LR_SALIENCE_CAP` | 10.0 | 2.0 | salience LR boost was ≤20% of intended |
| `RECALL_BLEND_THRESHOLD` | 5.5 | 0.5 | always blended (> max L2), injected cross-content noise |
| `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD` | 2.0 | 0.8 | reconstruction died after session 1 |

### Smoke test results (1 session, 1 cycle, CAPACITY=512)

- No NaN/inf ✓
- grew=True, regions_grown=[0,2,3] (AudioSpeech, Visual, Symbolic) ✓
- lm_hidden_computed=27, decoder_trainable=270 (27 unique × 10 oversample) ✓
- mean_proj_loss=0.00225 (very low, consistent with few unique traces) ✓
- synthetics_produced=0 (was the REM_SYNTH_MIN_L2 bug — now fixed) ✓
- distill_precompute_ms=1455ms (~54ms/trace for 27 lm_session forwards)
- nrem_wall_ms=4393ms, rem_wall_ms=1444ms, spill_wall_ms=125ms
- speech_encode_total_ms=28041, video_encode_total_ms=39092, gain_update_total_ms=1797

### Arg fix: `--working-memory-size` duplicate
`--working-memory-size 8` is **hardcoded in train_campaign.sh** — do NOT include it in `EXTRA_ARGS` or you get "argument cannot be used multiple times".

---

## Next: smoke test with fixed calibration (run this first)

```bash
SESSIONS=1 CYCLES=1 CAPACITY=512 \
MEMORY_DIR=memory/smoke_$(date +%H%M) \
EXTRA_ARGS="--lang-oversample 9 --asr-min-chars 5 --max-warm-records 200000" \
scripts/train_campaign.sh
```

Watch for: `synthetics_produced > 0` (validates REM_SYNTH_MIN_L2 fix).

---

## Gap queue: next tasks

### Gap 7 — Systematic forgetting (medium now, ADR needed)

**Design:** Add `--max-cold-records N` CLI arg (default 0 = disabled). During each sleep
spill, if `cold.len() > N`, truncate cold archive to the N most recent records and rebuild
warm from it. Cold archive records are fixed-size (COLD_DIM=640 bytes + 20 header = 660B),
so truncation = rewrite file tail-first.

**Implementation path:**
1. `ColdArchive::truncate_to_recent(n: usize)` — reads all, rewrites keeping last N
2. Sleep.rs: after warm-HNSW-size check, if `max_cold_records > 0 && recall.cold.len() > max_cold_records`, truncate + log
3. `--max-cold-records` CLI arg in main.rs
4. ADR 0020

**Effort:** ~100 lines Rust + ADR. No new dependencies.

### Gap 8 — REM concept-level probe (hard, needs ADR + data collection)

**Problem:** current gist probe measures episodic clustering (k-means on episode vectors).
Doesn't measure whether the same concept is represented consistently when encountered
via different senses (audio vs visual vs text).

**Proposed ADR approach:** Add an `ingest_source` field to `EpisodeRecord` tracking which
modality produced the episode (speech/env/visual/language or mixed). Run cross-modal
gist probe: for each gist cluster, measure the Shannon entropy of the `ingest_source`
distribution. High entropy = diverse sources in cluster = cross-modal grounding. Low
entropy = unimodal clusters = modalities not binding to shared concepts.

Emit `cross_modal_gist_entropy` per cycle. This is a diagnostic metric first; growth
or pruning decisions can be added once we understand the distribution.

**Effort:** medium-hard. Requires:
1. `ingest_source: u8` added to `EpisodeRecord` (schema change, cold archive incompatible)
2. EpisodeTrace stores source tag through to sleep
3. Gist probe extended to compute per-cluster source entropy
4. ADR 0021

---

## Key facts (updated)

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--working-memory-size 8` hardcoded in `train_campaign.sh` — do NOT add to EXTRA_ARGS
- `--asr-min-chars 5` for noisy-audio clips
- `--max-warm-records 200000` recommended for long runs
- Wake schedule trigger at 602s/cycle — compute speedups have zero wall-clock effect
- EMA_ALPHA = 0.99 for output head inference copy
- CROSS_MODAL_DIM = 2176 (speech+env+visual+language → symbolic)
- GRU_HIDDEN = 256 in binding area; N_HEADS=4, D_HEAD=32
- All existing memory snapshots are **incompatible** — always use fresh --memory-dir
- WAKE_RECONSTRUCT_SALIENCE_THRESHOLD = 0.8 (recalibrated from 2.0)
- LR_SALIENCE_CAP = 2.0 (recalibrated from 10.0)
- RECALL_BLEND_THRESHOLD = 0.5 (recalibrated from 5.5)
- REM_SYNTH_MIN_L2 = 1.0 (recalibrated from 2.0)
- LR_SALIENCE_MAX_SCALE = 2.0
- distill_precompute_ms ~54ms/trace for Qwen lm_session on CPU (27 traces = ~1.5s overhead per NREM cycle)
