# Session state — somnus

> **Last updated:** 2026-06-01
> **Last commit on main:** 9ed5d18 — Refresh autodiff graph every N wake episodes
> **Branch:** `main` (pushed to Codeberg)

## Status: All perf fixes landed. 1.5h campaign (9 cycles) completed cleanly. 26K warm records. loss 0.47→0.13.

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
| perception_ms timing | 3923686 | 4-phase ingest breakdown (kwta/symbolic/bind/prior) in NDJSON; video_stride 2 in campaign script |
| GRU cache (Perf F) | f9b7e60 | BindingArea carries gru_hidden across calls; 1 GRU step/ingest vs 8; bind 121→5.6ms/ep; **8.8× perception speedup** |
| EMA flat matmul (Perf G) | 536b3f1 | EMA predict via flat Vec<f32> matmul (no burn tensor alloc); bottleneck is memory bandwidth, not dispatch — within noise |
| Autodiff refresh (Perf H) | 9ed5d18 | refresh_autodiff_state() every N wake episodes (at wake_reconstruct tick); bind was growing 16→47ms/ep per cycle, now flat at 2ms/ep |

---

## Perception perf summary (memory/smoke_prior_*)

| Phase | Before (ms/ep) | After (ms/ep) |
|---|---|---|
| kwta | 4.0 | 2.1 |
| symbolic | 6.0 | 3.1 |
| bind | 121.0 | 7.7 |
| prior | 26.0 | 7.9 |
| **total perception** | **172** | **~35** |
| encode (concurrent GPU) | — | speech 61 + env 10 + video 96 |

Perception is **no longer the bottleneck** — encode dominates wall-clock but runs concurrently.
Remaining ~35ms/ep is memory-bandwidth-bound (EMA weight reads ~6.5MB per predict round).
Further wins require smaller models or batching, not code changes.

wgpu attempted (commit 5dbf9c9) and reverted: AMD RX 9070 XT already claimed by MIGraphX ORT → conflict on first video frame (exit 101). Not retryable without running ONNX on CPU.

---

## Campaign results (memory/campaign_20260601_2223)

3 sessions × 3 cycles, CAPACITY=4096, 26K warm records.
- loss: 0.47 → 0.13 across 9 cycles ✓
- bind: 2.1–2.5ms/ep flat (autodiff refresh fix confirmed) ✓
- synthetics_produced=8 every cycle (except s2c1 — NREM salience interrupt skipped REM) ✓
- s2c1 anomaly: NREM exited via salience gate early → buffer not cleared → s2c2 only 3 wake eps before re-trigger. Pre-existing behavior, not new.

## Known issue: NREM salience interrupt skips REM + spill

When NREM exits via salience gate, the episodic buffer is not spilled and REM is skipped. Next wake collects only a few episodes before re-triggering sleep. Benign for learning quality but wasteful. Not urgent.

---

## Next

**Longer campaign** — run more sessions to keep accumulating warm records:
```bash
SESSIONS=10 CYCLES=3 CAPACITY=4096 \
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
