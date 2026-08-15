# Roadmap

> **Contents:** Phase-by-phase record of what was built and when, with exit criteria. Phases 0-14 complete. Phase 9b (live mic/webcam end-to-end validation) remains open.
> **When to read:** Orienting in the project's history; finding which ADR covers a phase; identifying what's next.
> **Status:** coarse phasing. Use as orientation, not a commitment. Update when reality diverges.

Phases below are approximate and overlap — the listing imposes priority, not strict order. Each phase has one concrete *exit criterion*.

## Phase 0 — Scaffolding ✓

Project structure, conventions, behavioral instructions, tech-stack survey, architectural sketch, research notes, ADRs for Phase 1. No production code yet.

**Exit criterion:** ✓ a new collaborator can answer "what is this project, and how do we work on it?" from `ai-docs/`.

## Phase 1 — Sensory capture ✓ (ADRs 0001, 0002)

Mic → `AudioFrame` (cpal, crossbeam bridge). Webcam → `VideoFrame` (nokhwa, pull model). Media file → both streams (ffmpeg-next `MediaFile`, primary test path). Bounded crossbeam channels, drop-oldest on overflow. Binary: `--file`/`--mic`/`--webcam`, tokio async main, periodic metrics log.

**Exit criterion:** ✓ 10-minute file run without OOM, frame-rate drift, or process leak.

## Phase 2 — Encoding (frozen pretrained models) ✓ (ADRs 0003, 0004)

Frames → embeddings via Whisper-tiny (384-dim speech), CLAP-HTSAT-unfused (512-dim env), DINOv2-S/14 (384-dim visual). `ort` 2.0 against system ORT 1.24.4, parity-tested against HF Python reference (≤1e-4).

**Exit criterion:** ✓ embeddings flowing for all three streams in real time within frame-period latency.

## Phase 3 — Perception (sparse coding + binding) ✓ (ADR 0005)

5-region k-WTA (n=1024, k=20, 2% sparsity) + SAE decoder per region (`somnus-perception`). Binding area: single-head cross-attention over region tokens → 640-dim episode vector (ADR 0008). Output reconstruction heads + bidirectional top-down priors (ADRs 0009, 0010). Predictive-coding settling loop with prior_cosine gate (ADR 0011).

**Exit criterion:** ✓ sparse activations per frame; binding outputs a 640-dim episode; reconstruction error observable.

## Phase 4 — Memory hierarchy ✓ (ADR 0006)

Hot: `EpisodicBuffer` (bounded ring). Warm: `SemanticStore` (hnsw_rs HNSW, L2). Cold: `ColdArchive` (INT8, memmap2). HNSW rebuilt from cold on snapshot restore; cross-session recall confirmed stable (L2 1.0-3.5). `somnus-memory` crate.

**Exit criterion:** ✓ top-k retrieval from semantic memory on a cued episode.

## Phase 5 — Sleep cycle ✓ (ADR 0007)

Sleep trigger: episodic buffer high-water mark (80%). NREM: k-WTA+SAE replay with AdamW + cosine-annealed LR; homeostatic decay; hot→cold spill. REM: SAE MSE self-probe per region; cascade-correlation growth when MSE>threshold and warm recall poor. Snapshot (weights + episodic binary + region_sizes + next_id) with restore on startup.

**Exit criterion:** ✓ replay measurably reduces reconstruction error; growth confirmed by unit-count increase.

## Phase 6 — Language region ✓

EPISODE_DIM 512→640, 5-region arrays throughout, Qwen2.5 embed table (896-dim mean-pooled). Stage A: stdin text → language k-WTA → episode (ADR 0013). Stage B: Whisper ASR → language region. Auto-grounding: ASR text stored as `text_target` for projection training (ADR 0014). Action loop: region-token projection + frozen Qwen2.5-0.5B, per-episode gain modulation + autoregressive decode (ADR 0012). Cosine alignment loss for projection (replaces MSE, prevents cross-session gradient explosion, ADR 0015). Multi-clip training: `--file` cycles multiple paths in loop mode (99f3389).

**Exit criterion:** ✓ ASR ingested; language prior_cosine trending upward across sessions (0.043 after session 1, baseline 0.011).

## Phase 9 — Continuous operation (high priority)

### 9a — Schedule fallback sleep trigger ✓

`--max-wake-secs u64` (default 3600) caps wake time regardless of buffer fill rate; `sleep_trigger` field logs `high_water`/`schedule`/`perf_degradation`. Validated: fired at wake_secs=30, buffer 803/4096 (19.6%).

**Exit criterion:** ✓ sleeps on schedule during a quiet mic run where the buffer never fills.

### 9b — Real mic/webcam end-to-end validation (open)

Mic/webcam code exists from Phase 1 but has never been exercised with the full cognitive stack — all calibration is file-based. Needs: `--mic --webcam 0` for one full sleep cycle, confirm embeddings flow/episodes record/sleep fires/snapshot round-trips, fix any live-input-only bugs (timing, buffer sizing, resampler state).

**Exit criterion:** one complete wake→sleep→restore cycle from live input with no crashes or hung threads.

### 9c — Cold→warm promotion on cache miss ✗ (not needed)

Investigated 2026-05-18: warm is already kept in sync with cold at all times (`load_snapshot` rebuilds warm from cold on restart; every sleep-cycle spill also inserts into warm, `sleep.rs:300`). The stale-subset scenario this item targeted doesn't arise. Real scaling concern is HNSW rebuild time (164k records ≈ 3 min at restart) — that's Phase 11 territory. `promote()` stub in `RecollectionLayer` kept for future use, not currently needed.

## Phase 10 — Sleep fidelity (medium priority)

### 10a — Wake-performance degradation trigger ✓

Second sleep-entry condition: if prior_cosine mean (rolling EWMA, α=0.1, baseline = first 20 episodes' arithmetic mean) drops more than `--perf-sleep-threshold` below baseline, sleep early. Default 0.0 (disabled); requires `--prior-beta>0`. Validated: 5× talking_head baseline=0.566, novel noise clip tanks EWMA to 0.505, trigger fires at wake_secs=22 with buffer 8% full.

**Exit criterion:** ✓ perf-degradation trigger fires under injected novel content.

### 10b — Adaptive sleep duration ✓

`--sleep-loss-target f32` (default 0.0=run all traces): NREM exits early when loss EWMA < target (REM/spill/snapshot still run). `--sleep-max-secs u64` (default 0=no cap): exits on wall-time budget, treated as interrupted. `nrem_exit_reason`: `traces_exhausted`/`loss_target_met`/`time_budget`/`salience`. Validated: target=2.0 on 29-trace buffer exits after 2 traces; max-secs=5 on 55-trace buffer exits after 4 (interrupted).

**Exit criterion:** ✓ early exit on fast convergence; time-budget exit correctly marked interrupted.

### 10c — Gist-level self-probe in REM ✓

k-means (k=32, Lloyd's, up to 2048-record sample) on warm episode vectors; probe episodes and their top-1 warm neighbor assigned to nearest centroid; `gist_recall_rate` = fraction sharing a cluster. Active once warm ≥32 records, falls back to SAE MSE otherwise. Growth trigger prefers gist when available (grow if `gist_recall_rate<0.5`). Validated: cycle 2, `gist_recall_rate=0.932` on 41 warm × 59 probe (familiar content, growth correctly suppressed).

**Exit criterion:** ✓ gist-recall logged during REM; growth trigger uses it with SAE-MSE fallback.

### 10d — Working memory ✓

Fixed-capacity ring of recent episode vectors (`--working-memory-size`, default 0=disabled). During binding, the ring feeds a GRU (input=EPISODE_DIM=640, hidden=256); final hidden projects to `[5,D_PROJ=128]`, appended to K/V (Q stays `[5,128]`, output shape unchanged, ~852K new params) — supersedes the originally-described WM-mean-pool approach (ADR 0008 amendment). Cleared on sleep entry (wake-local). Validated: WM=8 vs. WM=0 on the same snapshot — output_mse 40-50% lower with WM (e.g. ep 6: 0.29 vs 0.58).

**Exit criterion:** ✓ output_mse measurably lower with WM enabled on temporally coherent content.

## Phase 11 — Memory scaling (low priority)

### 11a — Compression beyond INT8 ✓

PQ compression (`somnus-memory/src/pq.rs`): m=8 subspaces, k=256 centroids (u8), sub_dim=80. INT8 `ColdArchive` stays the live write path; `convert_cold_to_pq` is offline. 24 bytes/record (id+ts+codes) vs. 660 INT8 → **27.5× reduction** (target was ≥4×). Codebook 655,360 bytes (8×256×80×f32), trained on ≤8192 samples, stored at file head. Magic `SMNCOLDQ`; 6 unit tests pass (round-trip, size, quality).

**Exit criterion:** ✓ 27.5× reduction; L2 error <50% on random vectors (real episode vectors, being correlated, do substantially better).

### 11b — Creative recombination in REM ✓

`synthesize_episode(regional_a, regional_b, alpha)` blends encoder inputs (α=0.5), runs k-WTA (no DC update) + binding (no WM context) → synthetic episode. `measure_output_mse_inference` computes inference-mode output-head MSE without a backward step. REM block (after growth, before `episodic.clear`): `REM_SYNTH_ALPHA=0.5`, `REM_SYNTH_MIN_L2=2.0`, `REM_SYNTH_MAX_N=8` — samples ≤8 real episodes for baseline MSE, pairs with L2>2.0 synthesized, MSE averaged over both source regionals. NDJSON: `synthetics_produced`, `mean_real_inference_mse`, `mean_synthetic_output_mse`. NREM replay of synthetics deferred.

**Exit criterion:** ✓ synthetics produced and logged; `mean_synthetic_output_mse` > `mean_real_inference_mse` confirms genuine novelty to the trained heads.

## Phase 12 — Observability and scaling ✓

Three independent improvements. 12a (eval harness) lands first since 12c (distillation) needs its output for validation; 12b (HNSW persistence) has no dependencies.

### 12a — Metrics eval harness ✓ (ADR 0016)

`scripts/eval_metrics.py`: JSONL paths in, four PNGs out — language trajectory (`prior_cosine_mean` all 5 regions), growth timeline (scatter of `grew=true` cycles + `region_sizes`), per-region `output_mse`, sleep-trigger distribution (+ `nrem_exit_reason` secondary axis). matplotlib+numpy only, tolerates missing fields, sorts chronologically.

**Exit criterion:** ✓ all four charts from one invocation on any `metrics_*.jsonl`.

### 12b — HNSW warm-index persistence ✓ (ADR 0017)

`SemanticStore::dump(dir)`: `hnsw.file_dump` + bincode sidecar `warm/records.bin`. `load_warm(dir)`: `Box::leak`s the `HnswIo` allocation (returned `Hnsw` needs `'static`, one-time leak per process, no double-leak risk — `load_hnsw`'s return type is the trap), falls back to cold-rebuild on missing/corrupt/version-mismatched files. Version tag `warm/version` (magic `SMNWARM`+u16 LE), bumped on EPISODE_DIM/HNSW_M/HNSW_MAX_LAYERS changes. Dumped every sleep, after spill, before snapshot.

**Exit criterion:** restart time <5s for ≥100k records (was ~3min).

### 12c — Regional unit pruning ✓ (ADR 0018)

Units above `N_UNITS=1024` are clamped by `to_dense_n` and don't enter the episode vector, so they're safe to prune. Trigger: REM, `region_sizes[i]>1024+prune_min_grown` AND `gist_recall_rate≥0.8`. Selection: rank grown units (index≥1024) by encoder-weight-row L1 norm, remove bottom `--prune-fraction` (default 0.5). Rebuilds encoder/decoder/head (He-init), resets that region's optimizer moments. `--prune-min-grown` default 64.

**Exit criterion:** after pruning 50%, `per_region_mse` stays within 5% of pre-prune baseline over the next cycle.

## Phase 13 — Continuous training campaign ✓

`--session N` written to every NDJSON line (eval harness labels "S{n}C{m}", session-boundary lines). `scripts/train_campaign.sh`: outer loop over N sessions against the same `--memory-dir`, appends to one `campaign_<ts>/metrics.jsonl`, runs `eval_metrics.py` at the end. Env-configurable: `SESSIONS`, `CYCLES`, `CLIPS`, `MEMORY_DIR`, `EXTRA_ARGS`.

**Exit criterion:** ✓ runs end-to-end with session-aware charts.

**Next step (not coded then):** run a long campaign, check whether language prior_cosine reaches 0.1-0.2 after ~30 sessions.

## Phase 14 — GPU unification ✓

Move perception inference (k-WTA, cross-modal projection, BindingArea GRU) off CPU-only `burn` onto the same ORT+MIGraphX path already used for frozen encoders, so perception scales with the growable core instead of becoming a CPU bottleneck.

- **ADR 0023:** two-graph ONNX export (`perception_encoder.onnx`+`perception_binding.onnx`) via ORT+MIGraphX; `burn` stays CPU-side for training only. Duty-cycle top-k/masking stays on CPU between graphs (avoids a MIGraphX `ScatterElements`/INT64 bug on gfx1201).
- **ADR 0037 (2026-07-12):** superseded opt-in `--onnx-models-dir` with auto-enable (`should_use_ort_inference`, ~1,230 units/region crossover) — rebuilds ONNX from current burn weights at each sleep→wake boundary, no manual flag needed.
- **gaps-resolved.md §24 (2026-07-12/13):** found and fixed two dimension-drift bugs that had silently defeated the GPU path since its first auto-activation (sparse→dense hardcoded baseline N_UNITS instead of grown size; encoder `language_enc` hardcoded a stale pre-ADR-0036 LANG_HIDDEN). Confirmed live once fixed: zero ORT fallback, `perception_total_ms` ~3-4× faster than the pre-crossover CPU baseline.

**Exit criterion:** ✓ Phase 0+1 implemented and live (2026-06-07); auto-enable live and confirmed 2026-07-13 with a genuine ~3-4× speedup — perception now runs on GPU automatically past the crossover, no flag required.
