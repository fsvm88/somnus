# Session state — somnus

> **Last updated:** 2026-06-01
> **Last commit on main:** 19f789d — Default wake_reconstruct_interval to 5; add to campaign script
> **Branch:** `main` (pushed to Codeberg)

## Status: wake_reconstruct_interval=5 landed. Ready for longer campaign.

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| Bug 1 | 8a6fd66 | Symbolic region (3): `IngestSource::Visual` → `Mixed` (session fix), then `Mixed` → `Unknown` (audit fix) |
| Bug 2 | 8a6fd66 | Snapshot `ingest_source` persisted: SMNHOT01→SMNHOT02, field added to save/load, `push_with_id` signature extended |
| Audit pass 1 | 7247ff6 | `usearch`→`hnsw_rs` in tech-stack.md + ADR 0002; ADR index 0019–0021 added |
| Audit pass 2 | 3d641b4 | ADR 0021 region 3 mapping; ADR 0019 threshold values (WAKE_RECONSTRUCT 2.0→0.8, LR_SALIENCE_CAP 10.0→2.0); ADR 0020 RECORD_BYTES 532→661; sleep.rs growth-gate comment; snapshot.rs format diagram |
| Audit pass 3 | ef8d765 | pq.rs INT8 record size 660→661 + ingest_source note; ADR 0017 warm magic `SMNWARM`→`SMNW` |
| Instrumentation | 589adfb | 14 new NDJSON fields (salience, blend, modality counts, warm/cold sizes, NREM coverage) + ADR 0022 |
| Perf B | eaace08 | Deduplicate LM hidden-state precomp: HashMap cache by token-ids, 2.5× fewer ORT calls (11 unique / 27 total) |
| Perf A | e123911 | HNSW_EF_C 400→100: spill_wall_ms 133→77ms (42% faster) |
| Perf C | a8d9fe7 | Settle loop stats in PerceptionOutput + NDJSON (settle_trigger_count, settle_iters_sum) |
| Settle delta | 105f2a5 | Track settle_delta per episode; smoke test showed delta_mean=0.46 vs SETTLE_EPS=4.0 |
| settle_steps fix | 20e9dee | settle_steps 3→1 in campaign script (was wasting 3× compute for 1 real iteration) |
| Eval harness | c439fbc | 4 new panels: memory_growth, recall_quality, settle_stats, modality_balance |
| HNSW rebuild fix | 59cb7a6 | cold.len() > warm.len() condition (previous warm.len() > max always false after first rebuild) |
| NREM backward fix | 7d91f80 | refresh_autodiff_state() per cycle; burn global AutodiffServer accumulates ParamId chains → backward O(N) → O(1); 70× speedup |
| k-WTA + cold tail | 00db223 | k-WTA HashSet → two-pass duty-cycle; cold iter_from() O(1) seek for rebuild_from_recent |
| Perf D (LM cache) | bc24928 | Cross-cycle LM hidden-state cache in ActionDecoder.lm_hidden_cache; reduces per-call ORT cost and eliminates outliers; full benefit requires clips to loop (long clips → gradual improvement over hundreds of cycles) |
| Perf E (wake interval) | 19f789d | wake_reconstruct_interval default 1→5; explicit in campaign script; ~80% fewer wake backward passes |

---

## Flamegraph findings (profiling run: 20 cycles, /tmp/somnus_perf2.data)

| Bucket | % total samples | Notes |
|---|---|---|
| ONNX runtime (libonnxruntime) | ~37% | Encoder inference during wake + LM hidden states during sleep. Unavoidable without model changes. |
| burn autodiff graph overhead | ~9% | cleanup_orphaned_entries 3.1%, GraphLocator::select 1.3%, hashbrown 4.4%. Inherent to burn's autodiff; no handle from application code. |
| Audio/mel preprocessing | ~0.5% | CLAP + Whisper mel in tokio workers. |

Flamegraph SVG: `/tmp/somnus_perf2.data` (8 GB) + `/tmp/somnus_flamegraph.svg`

---

## 40-cycle smoke test results (memory/smoke_0013)

- `synthetics_produced=8` every cycle ✓
- No BadMagic / panic / ERROR ✓
- Symbolic episodes correctly tagged Unknown (not Visual) ✓
- `cross_modal_gist_entropy` present from cycle 1 ✓
- `mean_loss` declining over run: 0.055 → 0.034 ✓
- `warm_recall_l2` declining: 0.216 → 0.172 ✓
- `distill_precompute_ms`: 175–900 ms range (noisy; no clip-loop inflection yet — clips too long to loop in 40 cycles)

---

## Next

**Longer campaign** — multi-session run to gather a clean learning trajectory with all fixes in place:
```bash
SESSIONS=10 CYCLES=5 CAPACITY=4096 \
MEMORY_DIR=memory/campaign_$(date +%Y%m%d_%H%M) \
EXTRA_ARGS="--lang-oversample 9 --asr-min-chars 5 --max-warm-records 200000" \
scripts/train_campaign.sh
```

---

## Key facts

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--working-memory-size 8` hardcoded in `train_campaign.sh` — do NOT add to EXTRA_ARGS
- `--asr-min-chars 5` for noisy-audio clips
- `--max-warm-records 200000` recommended for long runs
- `--max-cold-records 500000` recommended for multi-day runs (≈330 MB)
- Wake schedule trigger at 602s/cycle
- EMA_ALPHA = 0.99 for output head inference copy
- CROSS_MODAL_DIM = 2176 (speech+env+visual+language → symbolic)
- GRU_HIDDEN = 256 in binding area; N_HEADS=4, D_HEAD=32
- **All existing memory snapshots incompatible** — use fresh `--memory-dir`
- WAKE_RECONSTRUCT_SALIENCE_THRESHOLD = 0.8
- LR_SALIENCE_CAP = 2.0
- RECALL_BLEND_THRESHOLD = 0.5
- REM_SYNTH_MIN_L2 = 1.0
- LR_SALIENCE_MAX_SCALE = 2.0
- RECORD_BYTES (cold INT8) = 661 (8+8+4+640+1)
- PQ record = 24 B; ingest_source not stored in PQ format (defaults to Unknown on read)
- Warm magic: `SMNW` (4 B) + u16 version = 6 B total
- Hot snapshot magic: `SMNHOT02`; Cold archive magic: `SMNCOLD2`

## IngestSource mapping (current)

| Active regions | IngestSource |
|---|---|
| [0] Speech only | Speech |
| [1] Env only | Env |
| [2] Visual only | Visual |
| [3] Symbolic only | Unknown (derived layer, not a raw encoder input) |
| [4] Language only | Language |
| 2+ regions | Mixed |
| none | Unknown |
