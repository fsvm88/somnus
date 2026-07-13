# Roadmap

> **Contents:** Phase-by-phase record of what was built and when, with exit criteria. Phases 0–14 complete. Phase 9b (live mic/webcam end-to-end validation) remains open.  
> **When to read:** Orienting in the project's history; finding which ADR covers a phase; identifying what's next.
>
> **Status:** coarse phasing. Use as orientation, not a commitment. Update when reality diverges.

The phases below are approximate and overlap. We do not need to finish a phase before touching the next; the listing imposes priority, not strict order. Each phase has a single concrete *exit criterion* — when we can answer that question, we have made meaningful progress.

## Phase 0 — Scaffolding ✓

- Project structure, conventions, and behavioral instructions in place.
- Tech-stack survey, architectural sketch, research notes, ADRs for Phase 1.
- No production code yet.

**Exit criterion:** ✓ a new collaborator can answer "what is this project, and how do we work on it?" by reading `ai-docs/`.

## Phase 1 — Sensory capture ✓

Get raw audio and video flowing through Rust, end-to-end. Defined in ADRs 0001 and 0002.

- Microphone → `AudioFrame` (cpal, crossbeam callback bridge).
- Webcam → `VideoFrame` (nokhwa, pull model).
- Media file → both streams (ffmpeg-next `MediaFile`; primary test path).
- Backpressure: bounded crossbeam channels, drop-oldest on overflow.
- Binary: `--file` / `--mic` / `--webcam` flags; tokio async main; periodic metrics log.

**Exit criterion:** ✓ 10-minute file run without OOM, frame-rate drift, or process leak.

## Phase 2 — Encoding (frozen pretrained models) ✓

Frames → continuous embeddings via frozen pretrained encoders.

- Whisper-tiny encoder (384-dim speech), CLAP-HTSAT-unfused (512-dim env), DINOv2-S/14 (384-dim visual).
- `ort` 2.0 against system ORT 1.24.4; parity-tested against HuggingFace Python reference (≤1e-4).
- ADRs 0003, 0004.

**Exit criterion:** ✓ embeddings flowing for all three streams in real time within frame-period latency.

## Phase 3 — Perception (sparse coding + binding) ✓

Embeddings → sparse activations per region; multimodal binding.

- 5-region k-WTA (n=1024, k=20, 2% sparsity) + SAE decoder per region; `somnus-perception` crate.
- Binding area: single-head cross-attention over region tokens → 640-dim episode vector (ADR 0008).
- Output reconstruction heads + bidirectional top-down priors (ADRs 0009, 0010).
- Predictive-coding settling loop with prior_cosine gate (ADR 0011).
- ADR 0005.

**Exit criterion:** ✓ sparse activations per frame; binding outputs 640-dim episode vector; reconstruction error observable.

## Phase 4 — Memory hierarchy ✓

Episodic buffer, semantic store, cold archive, and recollection index.

- Hot: `EpisodicBuffer` (bounded ring); warm: `SemanticStore` (hnsw_rs HNSW, L2); cold: `ColdArchive` (INT8, memmap2).
- HNSW rebuilt from cold archive on snapshot restore; cross-session recall confirmed stable (L2 1.0–3.5).
- `somnus-memory` crate. ADR 0006.

**Exit criterion:** ✓ top-k retrieval from semantic memory on a cued episode.

## Phase 5 — Sleep cycle ✓

Wake/sleep transitions; replay-driven consolidation; growth.

- Sleep trigger: episodic buffer high-water mark (80% capacity).
- NREM: k-WTA + SAE replay with AdamW + cosine-annealed LR (neuromodulation); homeostatic decay; hot→cold spill.
- REM: SAE MSE self-probe per region; cascade-correlation growth when MSE > threshold and warm recall poor.
- Snapshot (weights + episodic binary + region_sizes + next_id) with restore on startup.
- ADR 0007.

**Exit criterion:** ✓ replay measurably reduces reconstruction error; growth confirmed by unit count increase.

## Phase 6 — Language region ✓

Language as the 5th perception region, grounding ASR transcriptions and text into the binding area.

- EPISODE_DIM 512→640; 5-region arrays throughout; Qwen2.5 embed table (896-dim mean-pooled).
- Stage A: stdin text → language k-WTA → episode (ADR 0013).
- Stage B: Whisper ASR → language region (Whisper decoder ONNX, last-token LM head wrapper).
- Auto-grounding: ASR text stored as `text_target` for projection training (ADR 0014).
- Action loop: region-token projection + frozen Qwen2.5-0.5B; per-episode gain modulation + autoregressive decode (ADR 0012).
- Cosine alignment loss for projection (replaces MSE; prevents gradient explosion across sessions, ADR 0015).
- Multi-clip training: `--file` accepts multiple paths, cycles in loop mode (commit 99f3389).

**Exit criterion:** ✓ ASR transcriptions ingested; language prior_cosine trending upward across sessions (currently 0.043 after session 1, baseline 0.011).

## Phase 9 — Continuous operation (high priority)

Make the system viable for real, indefinite, unattended operation. Three parallel gaps must close.

### 9a — Schedule fallback sleep trigger ✓

Currently sleep fires only when the episodic buffer hits its high-water mark. In real continuous operation (mic running indefinitely), the buffer may fill slowly depending on input rate. A max-wake-time cap ensures sleep fires on schedule even when the buffer does not fill, and provides a safety net for long quiet periods.

- `--max-wake-secs u64` (default 3600); `sleep_trigger` field in log + NDJSON (`high_water`, `schedule`, `perf_degradation`).
- Validated: fired at wake_secs=30 with buffer_len=803/4096 (19.6%), well below high-water mark.

**Exit criterion:** ✓ system sleeping on schedule during a quiet mic run where the buffer never fills.

### 9b — Real mic/webcam end-to-end validation

The mic and webcam code paths exist from Phase 1 but have never been exercised with the full cognitive stack (encoding → perception → memory → sleep). All calibration has been file-based.

- Run `--mic --webcam 0` for at least one full sleep cycle.
- Confirm: embeddings flow, episodes are recorded, sleep fires, snapshot is written and restored.
- Identify and fix any bugs that only appear under live input (timing, buffer sizing, resampler state).

**Exit criterion:** one complete wake → sleep → restore cycle from live mic + webcam input with no crashes or hung threads.

### 9c — Cold→warm promotion on cache miss ✗ (not needed)

Investigated 2026-05-18. Warm is already kept in sync with cold at all times:

- `load_snapshot` rebuilds warm from the full cold archive on every restart.
- During each sleep cycle, hot→cold spill also inserts every spilled record into warm
  (sleep.rs line 300: `recall.warm.insert(record.clone())`).

So warm == cold union (not-yet-spilled hot) at all points in normal operation. The scenario
this item was designed to fix (warm being a stale subset of cold) does not arise.

The real scaling concern is HNSW rebuild time: 164k records took ~3 minutes at restart.
That is Phase 11 territory (HNSW persistence or lazy cold index), not runtime promotion.
The `promote()` stub in `RecollectionLayer` remains for future use but is not needed now.

## Phase 10 — Sleep fidelity (medium priority)

Improve the sleep cycle to match the architectural intent more closely.

### 10a — Wake-performance degradation trigger ✓

Second sleep entry condition from the architecture. If prediction quality (prior_cosine mean across regions) drops below a calibrated floor during wake, enter sleep early without waiting for the buffer to fill or the schedule to fire.

- Track a rolling EWMA (α=0.1) of per-episode prior_cosine means during wake.
- Capture a baseline from the first 20 episodes of each wake cycle (arithmetic mean).
- When EWMA drops > `--perf-sleep-threshold` below the baseline, set flag → `perf_degradation` trigger.
- Default threshold 0.0 (disabled). Requires `--prior-beta > 0` for cosines to be non-None.
- Validated: 5× talking_head clips establish baseline=0.566; novel noise clip tanks EWMA to 0.505; trigger fires at wake_secs=22 with buffer only 8% full.

**Exit criterion:** ✓ sleep fires on a perf-degradation trigger in a controlled test (inject novel content to tank prior_cosine, observe early sleep entry).

### 10b — Adaptive sleep duration ✓

Sleep currently runs for a fixed episode count (all traces in the buffer). The architecture specifies: run until consolidation goals are met, or a max-sleep-time cap. "Consolidation goals" = mean NREM loss below a threshold AND warm recall L2 below threshold after REM probe.

- `--sleep-loss-target f32` (default 0.0 = run all traces): NREM exits early when loss EWMA < target; REM + spill + snapshot still run (`interrupted=false`).
- `--sleep-max-secs u64` (default 0 = no cap): sleep exits when wall time exceeds cap; treated as interrupted (no REM/spill/snapshot).
- `nrem_exit_reason` field in SleepStats and NDJSON: `"traces_exhausted"`, `"loss_target_met"`, `"time_budget"`, `"salience"`.
- Validated: target=2.0 on 29-trace buffer → exits after 2 traces (`nrem_exit="loss_target_met"`, `interrupted=false`). max-secs=5 on 55-trace buffer → exits after 4 traces (`nrem_exit="time_budget"`, `interrupted=true`).

**Exit criterion:** ✓ sleep exits early on a session where loss converges quickly; time-budget exits as interrupted.

### 10c — Gist-level self-probe in REM ✓

Current REM probe uses SAE reconstruction MSE — a low-level signal. The architecture describes gist-level comparison: right scene, right objects, right events even when details differ. A gist-level probe compares episode *class* rather than pixel-level reconstruction.

- k-means (k=32, Lloyd's, up to 2048-record sample) on warm episode vectors during REM.
- Each probe episode and its top-1 warm neighbor are assigned to nearest centroid.
- `gist_recall_rate` = fraction of probe episodes whose neighbor is in the same cluster.
- Activates when warm store ≥ GIST_K=32 records; falls back to SAE MSE when unavailable.
- Growth trigger uses gist when available: grow when `gist_recall_rate < 0.5`. SAE MSE is still computed and logged but only used as fallback.
- `gist_recall_rate: Option<f32>` added to SleepStats and NDJSON metrics.
- Validated: cycle 2 shows `gist_recall_rate=0.932` on 41 warm records × 59 probe episodes (familiar 4-clip content; growth correctly suppressed).

**Exit criterion:** ✓ gist-recall rate is logged during REM; growth trigger uses it instead of SAE MSE (with SAE MSE fallback when warm is too small).

### 10d — Working memory ✓

The architecture mentions working memory as a transient context buffer during wake — holding the current conversational or environmental context across episodes without writing to semantic memory. Currently each episode is processed independently.

- Add a fixed-capacity ring of recent episode vectors (`--working-memory-size`, default 0 = disabled).
- During binding, the episode ring is fed through a GRU (input=EPISODE_DIM=640, hidden=GRU_HIDDEN=256); final hidden state is projected to [5, D_PROJ=128] and appended to K and V. Q stays [5, 128], output shape is unchanged. GRU adds ~852K learned parameters. (This supersedes the originally-described WM-mean-pool approach, which was replaced in the architectural roundup — see ADR 0008 amendment.)
- Working memory is cleared on sleep entry (wake-local by design).
- Validated: model trained with WM=8 on talking_head clip; re-running WITH WM=8 vs WITHOUT (WM=0) on same snapshot — output_mse consistently 40–50% lower with WM (e.g., episode 6: 0.29 vs 0.58). Output heads learned to predict from WM-augmented episodes; disabling WM reverts to a higher-MSE distribution.

**Exit criterion:** ✓ output_mse measurably lower when working memory is enabled on a WM-trained session with temporally coherent content.

## Phase 11 — Memory scaling (low priority)

Address long-term scalability of the cold archive and semantic store.

### 11a — Compression beyond INT8 ✓

Cold archive currently stores INT8-quantized 640-dim vectors. At scale, this is still 640 bytes/record. Options: product quantization (PQ, ~64 bytes/record), pruning least-accessed records, or knowledge distillation into smaller regional models.

- PQ compression implemented in `crates/somnus-memory/src/pq.rs`: m=8 subspaces, k=256 centroids (u8), sub_dim=80.
- INT8 ColdArchive remains the live write path; `convert_cold_to_pq` is offline conversion.
- Per-record bytes: 24 (id+ts+codes) vs 660 INT8 → 27.5× reduction (well above 4× target).
- Codebook: 655 360 bytes (8×256×80×f32), trained on up to 8192 even-spaced samples, stored at PQ file head.
- File magic `SMNCOLDQ`; all 6 unit tests pass including round-trip, size check, and quality check.

**Exit criterion:** ✓ cold archive size per record reduced by 27.5× (≥ 4×); L2 reconstruction error < 50% on random vectors (real episode vectors, being correlated, will be substantially lower).

### 11b — Creative recombination in REM ✓

Generative episode stitching: combine fragments from different episodes into synthetic novel episodes during REM.

- `PerceptionProcessor::synthesize_episode(&self, regional_a, regional_b, alpha)` — blends encoder inputs (α=0.5), runs through k-WTA (`layer_forward_settling`, no DC update) + binding (no WM context) → synthetic episode vector.
- `PerceptionProcessor::measure_output_mse_inference(&self, episode, regional)` — inference-mode output-head MSE without backward step.
- REM creative recombination block in `sleep.rs` (after growth, before `episodic.clear`):
  - Constants: `REM_SYNTH_ALPHA=0.5`, `REM_SYNTH_MIN_L2=2.0`, `REM_SYNTH_MAX_N=8`.
  - Samples up to 8 real episodes for inference-mode baseline MSE (`mean_real_inference_mse`).
  - Pairs with L2 > 2.0 synthesized; output MSE averaged over both source regionals → `mean_synthetic_output_mse`.
  - Both values logged via `tracing::info!` and NDJSON metrics (`synthetics_produced`, `mean_real_inference_mse`, `mean_synthetic_output_mse`).
- NREM replay of synthetics deferred (exit criterion met without it).

**Exit criterion:** ✓ synthetic episodes produced and logged; `mean_synthetic_output_mse` higher than `mean_real_inference_mse` confirms synthetics are genuinely novel to the trained output heads.

## Phase 12 — Observability and scaling ✓

Three independent improvements that pay dividends across all future work. Ordered by
dependency: 12a (eval harness) should land first because 12c (distillation) requires its
output to validate correctness. 12b (HNSW persistence) has no dependencies.

### 12a — Metrics eval harness ✓

Every sleep cycle writes a JSON line to `memory/metrics_*.jsonl`. There is currently no
tooling to answer the four recurring analysis questions without ad-hoc scripting.

- `scripts/eval_metrics.py`: positional args = one or more JSONL paths; writes four PNGs
  to `--out-dir` (default: same dir as first input file).
  1. **Language trajectory** — `prior_cosine_mean[4]` per cycle, all five regions overlaid.
  2. **Growth timeline** — scatter of cycles where `grew=true`, annotated with
     `regions_grown` and post-sleep `region_sizes`.
  3. **Per-region output_mse** — `per_region_mse[0..4]` across all cycles.
  4. **Sleep trigger distribution** — bar chart of `sleep_trigger` counts; secondary axis
     for `nrem_exit_reason`.
- Requirements: `matplotlib` + `numpy` only; skip missing fields; sort chronologically.
- ADR 0016.

**Exit criterion:** all four charts generated from a single invocation on any `metrics_*.jsonl`.

### 12b — HNSW warm-index persistence ✓

On every restart `load_snapshot` rebuilds warm HNSW from the full cold archive. At 164 k
records this takes ~3 minutes and grows linearly with archive size.

- `SemanticStore::dump(dir)`: calls `hnsw.file_dump` + bincode-serialises the
  `records: HashMap<u64, EpisodeRecord>` sidecar to `warm/records.bin`.
- `SemanticStore::load_warm(dir)`: `Box::leak`s the `HnswIo` allocation so the returned
  `Hnsw` is effectively `'static`; deserialises sidecar; falls back to cold-rebuild on
  missing/corrupt/version-mismatched files.
- Version tag: `warm/version` (6 bytes: magic `SMNWARM` + u16 LE); bump when
  `EPISODE_DIM`, `HNSW_M`, or `HNSW_MAX_LAYERS` change.
- Dump written on every sleep (after hot→cold spill, before snapshot).
- ADR 0017. Lifetime trap: `load_hnsw` returns non-`'static` Hnsw; `Box::leak` is the
  chosen resolution (one-time per process, no double-leak risk).

**Exit criterion:** restart time < 5 s for a ≥ 100 k-record store (vs. current ~3 min).

### 12c — Regional unit pruning ✓

`grow_region` accumulates units without bound. Units above `N_UNITS=1024` are clamped by
`to_dense_n` and do not enter the episode vector, making them safe to prune without
breaking snapshot compatibility.

- Trigger: during REM, after growth check, when `region_sizes[i] > 1024 + prune_min_grown`
  AND `gist_recall_rate ≥ 0.8` (not currently struggling).
- Selection: rank grown units (index ≥ 1024) by L1 norm of encoder weight row; remove
  bottom `--prune-fraction` (default 0.5).
- Rebuild: new encoder Var, SAE decoder, output head (He-init for shape change), reset
  optimizer moments for that region; update `region_sizes[i]`.
- `--prune-fraction f32` (default 0.5); `--prune-min-grown usize` (default 64).
- Requires Phase 12a to validate that `per_region_mse` does not regress post-prune.
- ADR 0018.

**Exit criterion:** after pruning 50% of grown units, `per_region_mse` stays within 5%
of the pre-prune baseline across the next sleep cycle (verified via eval harness).

## Phase 13 — Continuous training campaign ✓

Run enough sessions for the language region to meaningfully converge, and make
multi-session runs operationally easy.

- `--session N` CLI arg: written to every NDJSON metrics line so the eval harness can
  label cycles as "S{n}C{m}" and draw vertical session-boundary lines.
- `scripts/train_campaign.sh`: outer loop over N sessions, each running
  `--max-sleep-cycles CYCLES` against the same `--memory-dir`. Appends to a single
  `campaign_<ts>/metrics.jsonl`; runs `eval_metrics.py` automatically at the end.
  Configurable via env: `SESSIONS`, `CYCLES`, `CLIPS`, `MEMORY_DIR`, `EXTRA_ARGS`.
- `eval_metrics.py` updated: session-aware x-axis labels; vertical dashed lines at
  session boundaries on all four charts.

**Exit criterion:** ✓ `train_campaign.sh` runs end-to-end; charts show `S{n}C{m}` labels
and session-boundary lines; language prior_cosine trajectory interpretable across
multiple sessions from a single command.

**Next step (not coded):** run `SESSIONS=20 CYCLES=3 scripts/train_campaign.sh` and
observe whether language prior_cosine reaches 0.1–0.2 after 60 cycles (~30 sessions
needed by Phase 8 estimate). Use eval harness output to confirm or revise the estimate.

## Phase 14 — GPU unification ✓

Move perception inference (k-WTA, cross-modal projection, BindingArea GRU) off CPU-only
`burn` and onto the same ORT+MIGraphX GPU path already used for the frozen encoders, so
perception scales with the growable core instead of becoming a CPU bottleneck as
`N_UNITS` grows toward brain-inspired sizes.

- ADR 0023: two-graph ONNX export (`perception_encoder.onnx` + `perception_binding.onnx`)
  run via ORT+MIGraphX; `burn` retained CPU-side for training only. Duty-cycle top-k/masking
  stays on CPU between the two graphs to avoid a MIGraphX `ScatterElements`/INT64 bug on
  gfx1201.
- ADR 0037 (2026-07-12) superseded the opt-in `--onnx-models-dir`/`--onnx-script` flags with
  auto-enable: `should_use_ort_inference` detects the CPU/GPU crossover (~1,230 units/region,
  ADR 0023) from live region sizes and switches over automatically, rebuilding the ONNX graphs
  from current `burn` weights at each sleep→wake boundary (`rebuild_ort_session`) — no manual
  flag needed once a region grows past baseline.
- gaps.md §24 (2026-07-12/13) found and fixed two dimension-drift bugs that had silently
  defeated the GPU path since it first auto-activated (sparse→dense conversion hardcoding
  baseline `N_UNITS` instead of each region's grown size; encoder `language_enc` input
  hardcoding a stale pre-ADR-0036 `LANG_HIDDEN`). Confirmed live once both were fixed: zero ORT
  fallback occurrences, `perception_total_ms` ~3-4× faster than the pre-crossover CPU baseline.

**Exit criterion:** ✓ Phase 0 + Phase 1 of ADR 0023 implemented and live (2026-06-07); auto-enable
(ADR 0037) live and confirmed 2026-07-13 with a genuine, measured ~3-4× speedup once gaps.md §24's
two bugs were fixed — perception inference now runs on GPU via ORT/MIGraphX automatically past the
crossover, with no flag required.
