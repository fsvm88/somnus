//! Unified timing/counter instrumentation, in two tiers.
//!
//! - **Core** ([`timed_core!`]/[`counted_core!`]): always compiled in, always recording. This is
//!   the "big item" tier — the same metrics this codebase always reported (ingest phases, sleep
//!   wall-clock phases, encode timers), just recorded through one shared macro instead of a
//!   hand-rolled `Instant::now()`/`.elapsed()` pair duplicated at every call site.
//! - **Fine** ([`timed!`]/[`counted!`]): gated behind the `fine-metrics` Cargo feature (off by
//!   default). With the feature disabled, these expand to a bare passthrough of their body — the
//!   `Instant`/atomic-counter code is textually absent, not just runtime-skipped. Use this tier
//!   for granular sub-breakdowns (e.g. timing a scoring matmul separately from its enclosing
//!   ingest call) that aren't worth paying for on every build.
//!
//! Both tiers write into the same [`METRICS`] registry and are drained together once per sleep
//! cycle via [`Registry::snapshot_and_reset`] — one source of truth, one output format.
//!
//! Guardrail: instrument at *operation* granularity (called O(1) times per ingest call or
//! sleep-cycle phase), never inside a per-element tight loop. If a loop's aggregate cost matters,
//! accumulate a local `(count, Duration)` across it and make one macro call after the loop, not
//! one per iteration — `Instant::now()` and the atomic `fetch_add` are cheap per call but not free
//! at millions of calls. This applies doubly to the Core tier, since it always runs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// One instrumented operation. Doc comments note which existing hand-rolled field/log line each
/// Core-tier variant replaces, where applicable, and which caller(s) it nests inside for
/// operations that sit within an already-timed span (nesting is expected and intentional — a
/// full-coverage pass, ADR 0039 follow-up, deliberately breaks several previously-opaque
/// aggregates like `IngestBind`/`SleepFusedReplayWake`/`SleepRem` into their real sub-steps, so
/// an A/B comparison can attribute a wall-clock delta to the actual changed step instead of just
/// the whole enclosing phase). Genuinely per-element hot loops (audio/video frame decode,
/// per-record cold/PQ archive I/O during spill) are deliberately NOT given their own variants —
/// they're already covered in aggregate by an enclosing phase (`SleepSpill`, `MediaFileDecode`),
/// per the guardrail below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    // ── Core: PerceptionProcessor ingest phases (crates/somnus-perception/src/lib.rs) ──────────
    /// k-WTA forward phase (per-region encode + optional decoder reconstruction error).
    /// Replaces the old `perf_kwta_us` accumulator.
    IngestKwta,
    /// Symbolic cross-modal fusion phase. Replaces `perf_symbolic_us`.
    IngestSymbolic,
    /// Binding-area forward phase. Replaces `perf_bind_us`.
    IngestBind,
    /// Top-down prior blend phase. Replaces `perf_prior_us`.
    IngestPrior,
    /// Predictive-coding settle phase. Replaces `perf_settle_us`.
    IngestSettle,
    /// Post-settle prior update phase. Replaces `perf_prior_update_us`.
    IngestPriorUpdate,

    // ── Core: sleep cycle wall-clock phases (crates/somnus/src/sleep.rs) ───────────────────────
    /// Autodiff-graph refresh before NREM replay begins.
    SleepAutodiffRefresh,
    /// Full NREM replay loop (all epochs/batches). Replaces `nrem_wall_ms`.
    SleepNrem,
    /// Projection-distillation precompute pass before NREM. Replaces `distill_precompute_ms`.
    SleepDistillPrecompute,
    /// One fused SAE-replay + wake-head training step (per NREM batch).
    SleepFusedReplayWake,
    /// One projection-distillation training step (per NREM batch).
    SleepProjDistill,
    /// REM phase (accumulated across its two sub-sections). Replaces `rem_wall_ms`.
    SleepRem,
    /// Hot→cold spill + warm consolidation. Replaces `spill_wall_ms`.
    SleepSpill,
    /// End-of-cycle snapshot write. Replaces `snapshot_wall_ms`.
    SleepSnapshot,
    /// Gist-level k-means clustering over a warm-store sample (ADR 0021 §10c).
    SleepGistKmeans,
    /// Warm HNSW index rebuild from cold (triggered when cold outgrows warm).
    SleepWarmRebuild,
    /// Whole sleep cycle, start to finish. Replaces `total_wall_ms`.
    SleepTotal,

    // ── Core: encoder wall-clock (crates/somnus-encoding/src) ──────────────────────────────────
    /// Whisper speech encode (mel + ORT inference). Same measurement `AudioEmbedding::encode_ms`
    /// used to carry; now the sole source of that field instead of a duplicate timer.
    EncodeSpeech,
    /// Env-sound encode (CLAP/mn10, mel + ORT inference). Same role as `EncodeSpeech`.
    EncodeEnv,
    /// Video-frame encode (DINOv3 + optional change-gate). Same role, `VideoEmbedding::encode_ms`.
    EncodeVideo,

    // ── Core: main event loop (crates/somnus/src/main.rs) ──────────────────────────────────────
    /// One `PerceptionProcessor::ingest_*` call from the event loop (any modality). Replaces
    /// the `perception_total_ms` accumulator.
    MainPerceptionIngest,
    /// One warm-memory HNSW recollection query. Feeds `warm_query_us_mean`.
    MainWarmQuery,

    // ── Fine: k-WTA scoring sub-breakdown (crates/somnus-perception/src/region.rs) ──────────────
    /// `KWtaLayer::forward`'s scoring matmul alone (`self.linear.forward`), excluding
    /// selection/duty-cycle and decoder reconstruction error. The sub-breakdown of
    /// `IngestKwta` that ADR 0038's benchmarking prerequisite actually needs.
    KwtaScoreMatmul,
    /// `KWtaLayer::select_top_k`'s selection body alone (duty-cycle boost + top-k partition +
    /// duty-cycle update), excluding the scoring matmul.
    KwtaSelectTopK,

    // ── Core: remaining somnus-perception steps (lib.rs, binding.rs, onnx_session.rs,
    //    wake_worker.rs) — gaps.md §26 follow-up, full-coverage pass ───────────────────────
    /// `refresh_language_cache`'s `language_layer.forward_no_dc` call, invoked after every
    /// non-language ingest. Previously undocumented-but-claimed as part of `IngestSymbolic`'s
    /// doc comment; actually untimed until this variant.
    RefreshLanguageCache,
    /// `BindingArea::forward` alone — the cross-attention + GRU pass, nested inside whichever
    /// caller is already timed (`IngestBind`, `IngestSettle`, or `SleepRem`'s synth path).
    BindingForward,
    /// `OnnxPerceptionSession::run_encoder` alone — the ORT/GPU dispatch for the 5-region
    /// scoring matmul, nested inside `IngestBind`/`IngestSettle`.
    OrtRunEncoder,
    /// `OnnxPerceptionSession::run_binding` alone — the ORT/GPU dispatch for cross-attention +
    /// GRU, nested inside `IngestBind`/`IngestSettle`.
    OrtRunBinding,
    /// `infer_episode_outputs` — 5 output-head forward passes, called once per trace inside the
    /// NREM DER++ staleness-filter loop (`sleep.rs`), nested inside `SleepNrem`.
    InferEpisodeOutputs,
    /// `fused_replay_wake_step_batch`'s Phase 1 (per-region SAE forward + MSE), nested inside
    /// `SleepFusedReplayWake`.
    FusedReplaySaePhase,
    /// `fused_replay_wake_step_batch`'s Phase 2 (per-head output forward + MSE), nested inside
    /// `SleepFusedReplayWake`.
    FusedReplayHeadPhase,
    /// `fused_replay_wake_step_batch`'s combined backward + optimizer step, nested inside
    /// `SleepFusedReplayWake` — the documented ~150ms/batch-at-128 AdamW cost.
    FusedReplayBackward,
    /// `rebuild_ort_session` — exports weights, shells out to
    /// `scripts/export_perception_graph.py`, reloads ORT sessions. Runs once per sleep→wake
    /// boundary while ORT is active; the Python subprocess makes this the single largest
    /// candidate wall-clock item in the whole crate.
    RebuildOrtSession,
    /// `export_weights_for_onnx` alone — tensor→disk serialization step inside
    /// `rebuild_ort_session`, distinct from the subprocess call itself.
    ExportWeightsForOnnx,
    /// `enable_ort_inference` — loads ORT sessions from disk (session construction/graph
    /// compile), called from both startup and `rebuild_ort_session`.
    EnableOrtInference,
    /// `homeostatic_decay` — scales ~20 weight tensors by a constant. Exactly one call per sleep
    /// cycle, in a gap between `SleepNrem` and `SleepRem` that was previously invisible.
    HomeostaticDecay,
    /// `measure_mse` — per-region k-WTA forward + reconstruction error, inference-only. Called
    /// once per REM-sampled trace inside the REM self-probe, nested inside `SleepRem`.
    MeasureMse,
    /// `collect_high_residual_inputs` — scores every REM trace by reconstruction residual to
    /// seed `grow_region_seeded`. One call per sleep cycle, O(n_traces) internal work.
    CollectHighResidualInputs,
    /// `grow_region` — unseeded region growth (weight-tensor concat). Fallback path or
    /// restart-time pre-grow.
    GrowRegion,
    /// `grow_region_seeded` — cascade-correlation-seeded region growth. Bounded ≤5 calls/cycle.
    GrowRegionSeeded,
    /// `prune_region` — L1-norm-based unit pruning (weight-tensor gather/rebuild). Bounded ≤5
    /// calls/cycle.
    PruneRegion,
    /// `synthesize_episode` — blends two regional-input sets, 5-region forward + binding
    /// forward. REM creative-recombination step, bounded by `REM_SYNTH_MAX_N`.
    SynthesizeEpisode,
    /// `measure_output_mse_inference` — per-head forward + MSE, no backward. REM real/synthetic
    /// quality probe, bounded-multiple per cycle.
    MeasureOutputMseInference,
    /// `wake_reconstruct_step` — synchronous single-episode output-head training (the fallback
    /// path when no `WakeReconWorker` is active). Per-episode, throttled by `wake_interval`.
    WakeReconstructStep,
    /// `wake_worker::train_step` — the async wake-reconstruct path's training step, running on
    /// its own background OS thread. Cross-thread `METRICS.record()` calls are safe (plain
    /// atomics); this is the only variant recorded from outside the main/sleep threads.
    WakeWorkerTrainStep,
    /// `EmaOutputHeads::predict` — the GEMV inside `update_prior_predictions` (`IngestPriorUpdate`)
    /// and every settle iteration (`IngestSettle`). Finest-grained sub-step of both.
    EmaPredict,
    /// `EmaOutputHeads::update_from_weights` — EMA tensor blend, called from the sync
    /// (`wake_reconstruct_step`) and async (`wake_worker::train_step`) training paths.
    EmaUpdateFromWeights,
    /// `EmaOutputHeads::sync_flat` — re-extracts flat host copies from 5 EMA tensors (5
    /// tensor→host transfers). Called from `update_from_weights` and `from_model` (restart).
    EmaSyncFlat,
    /// `refresh_autodiff_state` — remaps every model parameter to a fresh leaf tensor and
    /// reinitializes the optimizer. Its own doc comment calls out a "100-500x NREM slowdown"
    /// this exists to prevent — perf-sensitive by its own admission, previously untimed.
    RefreshAutodiffState,

    // ── Core: somnus-memory (crates/somnus-memory/src) — zero prior instrumentation ───────────────
    /// `ColdArchive::open` — mmap-open (or create) the cold archive file. Per-restart.
    ColdArchiveOpen,
    /// `ColdArchive::truncate_to_recent` — rewrite the cold archive keeping only the most
    /// recent N records.
    ColdArchiveTruncate,
    /// `ColdArchive::evict_lowest_salience` — read every record's salience via mmap, select-nth,
    /// rewrite kept records. One call per sleep cycle when the cold cap is exceeded.
    ColdArchiveEvict,
    /// `SemanticStore::load_warm` — mmap-load the persisted HNSW dump + bitcode sidecar. The
    /// fast branch of startup restore (vs. `SnapshotLoad`'s slow full-cold-rebuild branch).
    SemanticStoreLoadWarm,
    /// `RecollectionLayer::promote` — O(cold_len) linear scan of the entire cold archive to find
    /// one record by id. Flagged in its own doc comment as a future bottleneck at scale.
    RecollectionPromote,
    /// `snapshot::save_episodic` — binary-encode the whole hot episodic buffer to disk. One
    /// call per `SnapshotSave`, itself nested inside `SleepSnapshot`.
    SnapshotSaveEpisodic,
    /// `snapshot::load_episodic` — binary-decode the episodic buffer from disk. Per-restart.
    SnapshotLoadEpisodic,
    /// `snapshot::save_snapshot` — the whole per-cycle save orchestration (episodic + next_id +
    /// atomic rename), nested inside `SleepSnapshot` alongside the HNSW dump/weights save.
    SnapshotSave,
    /// `snapshot::load_snapshot` — the whole startup restore orchestration. Has two very
    /// different-cost branches (fast `SemanticStoreLoadWarm` mmap-load vs. slow full
    /// cold-archive rebuild, documented as up to ~3 min at 164k records) — this variant times
    /// the call as a whole; `SemanticStoreLoadWarm` distinguishes the fast branch.
    SnapshotLoad,
    /// `pq::convert_cold_to_pq` — offline cold→PQ archive conversion (train codebook + re-encode
    /// every record). Not on any current production hot/cold path; instrumented for when it is
    /// invoked (analysis tooling).
    PqConvertColdToPq,

    // ── Core: somnus-sensory (crates/somnus-sensory/src/media_file.rs) ────────────────────────────
    /// Aggregate per-file decode+resample time for one `MediaFile` playback — accumulated
    /// locally across the whole per-packet decode loop (audio + video, both branches) and
    /// recorded once at end-of-stream, per this crate's own guardrail against per-packet
    /// instrumentation at decode-thread frequency.
    MediaFileDecode,

    // ── Core: somnus binary — action decoder, ASR, remaining main-loop steps ───────────────────
    /// `ActionDecoder::new` — loads Qwen3 ONNX sessions, tokenizer, embed table. Per-restart,
    /// only with `--action`.
    ActionDecoderInit,
    /// `ActionDecoder::generate` — the full autoregressive greedy-generation loop (tokenize →
    /// project episode(s) → per-token ORT forward). One call per user/startup query.
    ActionGenerate,
    /// `GainWorker`'s background-thread gain computation (ORT forward → region-token
    /// projection → LM forward → gain norms). Async, throttled by `--gain-interval`; the
    /// worker thread's own atomics-only recording is safe cross-thread, same as
    /// `WakeWorkerTrainStep`.
    GainCompute,
    /// Language-line embedding step in `ingest_language_line` — either `LangEmbedder::embed`
    /// (table mean-pool, default) or `ActionDecoder::text_hidden_states` (`--lang-use-lm`).
    /// Previously invisible: `MainPerceptionIngest` only starts timing after this returns.
    LangEmbed,
    /// `WhisperDecoder::decode_greedy` — autoregressive ASR decode (up to `max_tokens`
    /// sequential ORT calls + repetition penalty + tokenizer decode). One call per audio chunk
    /// that reaches the ASR thread.
    AsrDecode,
    /// `AudioProcessor::ingest`'s resampling step (`SincResampler::process` for both the 16kHz
    /// speech path and the env-encoder-rate path). Per audio chunk, before `EncodeSpeech`/
    /// `EncodeEnv`.
    AudioResample,
    /// `record_episode`'s work outside the already-timed `MainWarmQuery` sub-call — L2-normalize,
    /// recall-blend arithmetic, episodic-buffer push, sink JSON write. Wraps the whole function;
    /// `MainWarmQuery` nests inside it as before.
    MainRecordEpisode,

    // ── Core: value/event counters (not durations) — first real use of the "counted" side of
    //    the two-tier design; everything above this point is exclusively duration-based ───────
    /// A region's `n_units` immediately after a `grow_region`/`grow_region_seeded`/`prune_region`
    /// call. Recorded as a raw value (not an event count) via `counted_core!`, once per growth/
    /// prune event (bounded ≤5/cycle) — answers "how big are the regions actually getting"
    /// without a one-off validation script.
    RegionNUnits,
    /// The paired `k_active` (from `region::scaled_k`) at the same growth/prune event as
    /// `RegionNUnits` — together they give the live sparsity ratio (`k_active / n_units`) that
    /// ADR 0037/0038's scoring-cost analysis depends on.
    RegionKActive,
    /// Number of records actually removed by one `ColdArchive::evict_lowest_salience` call.
    /// Distinct from `ColdArchiveEvict`'s timing, which only counts how many times the function
    /// ran, not how many records it touched.
    ColdArchiveEvictedRecords,
    /// Frames the video-encoder thread actually encoded, accumulated locally and flushed once
    /// per second alongside the existing throttled `tracing::info!` log (same cadence, no new
    /// per-frame overhead).
    VideoFramesEncoded,
    /// Frames the video-encoder thread's change-gate skipped, same per-second flush as
    /// `VideoFramesEncoded` — together they give the change-gate's actual skip rate, previously
    /// visible only in a log line, never in the metrics file.
    VideoFramesSkipped,
    /// `refresh_language_cache` calls where the ORT-active branch skipped the `forward_no_dc`
    /// call entirely. `RefreshLanguageCache`'s own timing wraps both branches uniformly and
    /// can't distinguish them; this plus `RefreshLanguageCacheRun` closes that gap.
    RefreshLanguageCacheSkipped,
    /// `refresh_language_cache` calls where the `forward_no_dc` call actually ran (burn path,
    /// ORT encoder inactive or has no cached language input).
    RefreshLanguageCacheRun,

    // ── Core: operational health for unattended long-running campaigns (2026-07-15 follow-up)
    //    — GPU/ORT degradation, resource sampling, growth batch size, and grouped failure
    //    counters for every recurring warn!/error! site. Grouped by failure *class*, not one
    //    variant per call site — see doc comments for which sites feed each one. ─────────────
    /// `PerceptionProcessor::try_bind`'s ORT encoder call failed; fell back to burn CPU
    /// activations for that ingest. Distinct from `OrtRunEncoder`'s timing, which only covers
    /// the successful path.
    OrtEncoderFallback,
    /// `try_bind`'s ORT binding call failed; fell back to burn `BindingArea::forward`. Distinct
    /// from `OrtRunBinding`'s timing, same reasoning as `OrtEncoderFallback`.
    OrtBindingFallback,
    /// `rebuild_ort_session` failed on first attempt (startup or first post-growth crossover) —
    /// stays on burn inference for the rest of the process (sticky, per `onnx_failed`).
    OrtSessionInitFailed,
    /// `rebuild_ort_session` failed on a rebuild while an ORT session was *already* active
    /// (post-sleep-cycle weight refresh) — continues with the stale-but-working existing
    /// session rather than falling back to burn. Distinct from `OrtSessionInitFailed` since the
    /// failure mode and recovery path differ (keep old session vs. no session at all).
    OrtSessionRebuildFailed,
    /// Current process RSS (bytes), sampled once per sleep cycle via `sysinfo`. Was previously
    /// only computed once at startup (for the auto warm-cap heuristic), never sampled over time
    /// — a slow leak during a long unattended campaign would have been invisible until OOM.
    ProcessRssBytes,
    /// Free disk space (bytes) on the filesystem backing `--memory-dir`, sampled once per sleep
    /// cycle alongside `ProcessRssBytes`. Same `sysinfo` disk lookup the P1a auto-cold-cap
    /// heuristic already used at startup, now factored out and re-sampled periodically.
    MemoryDirDiskFreeBytes,
    /// The actual novelty-scaled batch size (`growth_batch_size`, gaps.md §19) chosen at each
    /// growth event, alongside the existing `RegionNUnits`/`RegionKActive` growth-result
    /// counters — this is the growth *input* they were missing.
    GrowthBatchSize,
    /// One end-of-cycle persistence write failed (episodic snapshot, warm HNSW dump, perception
    /// weights, region sizes, or growth-cooldown counter — all in the same `SleepSnapshot`
    /// block, all the same failure class: disk write failed, next restart falls back to a
    /// staler or fresh copy of that one piece of state). A nonzero count is the health signal;
    /// the log line pinpoints which file.
    SleepPersistenceWriteFailed,
    /// `SemanticStore::dump`'s own wall-clock — previously untimed despite being the same class
    /// of per-cycle I/O as everything else `SleepSnapshot` covers (gaps.md §21's leak lived in
    /// this exact function). Found while adding `SemanticStoreDumpBytes` below.
    SemanticStoreDump,
    /// Bytes actually written by one `SemanticStore::dump` call (`index.hnsw.{data,graph}` +
    /// `records.bin`, summed from disk since `hnsw_rs::file_dump` picks its own file names).
    SemanticStoreDumpBytes,
    /// One NREM training-step operation degraded but the cycle continued: staleness-inference
    /// failure (keeps the sample instead of filtering it), `fused_replay_wake_step_batch`
    /// failure (skips the batch), `projection_step_batch_distill` failure (skips the batch), or
    /// a post-NREM `proj_weight_norm` read failure. All are "this one step didn't happen as
    /// intended, NREM kept going" — the same operational meaning for a health signal.
    NremStepError,
    /// A cold-archive operation failed during the hot→cold spill phase: `ColdArchive::append`
    /// (aborts the rest of that spill) or `evict_lowest_salience` (cap not enforced this cycle).
    ColdArchiveOpError,
    /// A REM self-probe function failed for one sampled trace: `measure_mse`,
    /// `measure_output_mse_inference`, or `synthesize_episode`. Probe-only — doesn't affect
    /// training, but a rising count signals a broken region/decoder worth investigating.
    RemProbeError,
    /// One `PerceptionProcessor::ingest_*` call failed outright (audio/video/language) — the
    /// ingest event itself was dropped, not just degraded. The single highest-value counter in
    /// this batch for an unattended run: a rising rate means real perception data is being lost
    /// every cycle, not just probed imperfectly.
    PerceptionIngestError,
    /// The synchronous `wake_reconstruct_step` fallback path failed for one episode (used when
    /// no async `WakeReconWorker` is active, or per-episode within it).
    WakeReconstructError,
    /// Whisper ASR decode failed or panicked for one audio chunk (the `catch_unwind`-guarded
    /// path, gaps.md's mitigated-not-root-caused ORT panic) — the chunk is dropped either way.
    AsrError,
    /// `ingest_language_line`'s embedding step failed (table mean-pool or `--lang-use-lm`
    /// hidden-state path) — that language line never reached any region.
    LangEmbedError,
    /// `ActionDecoder::generate` failed for one query — covers the interactive path, the
    /// distillation-only path, and the startup query path uniformly (same failure meaning:
    /// one generation request produced nothing).
    ActionGenerationError,
    /// The async `WakeReconWorker` background thread died and its output heads were never
    /// merged back in. Rare but high-severity: if this fires, wake-reconstruct training
    /// silently stopped for the rest of the process with no other visible symptom.
    WakeReconWorkerDied,
    /// A startup restore step fell back to a fresh/default value instead of the persisted one:
    /// perception weight restore, snapshot load, or decoder weight restore. One-time (at
    /// process start) but explains an otherwise-mysterious "why did region sizes reset" report.
    StartupRestoreDegraded,
    /// A media/input-source error: file-loop/file-sequence open failure, microphone
    /// unavailable, live video capture error, audio/video encoder thread init failure, or a
    /// per-frame audio/video encode error. Grouped since they're all "this input source
    /// degraded or dropped a unit of data," the same class `MediaFileDecode`'s own aggregate
    /// timing already treats per-frame errors within.
    MediaSourceError,
    /// Catch-all for the remaining low-frequency `warn!` sites not covered by a more specific
    /// counter above: embedding-sink write/flush failures, EWMA/decoder-weights per-cycle save
    /// failures, metrics-file write failures, the top-level sleep-cycle error handler, ORT
    /// profile flush failures, and auto-ground tokenize failures. A nonzero count here is a
    /// prompt to grep the log rather than a diagnosis in itself.
    MiscWarning,
    /// Bytes written by one `snapshot::save_episodic` call (`fs::metadata` on the written
    /// file), pairing with the pre-existing `SnapshotSaveEpisodic` timing.
    SnapshotEpisodicBytesWritten,
    /// Bytes read by one `snapshot::load_episodic` call, same pairing with
    /// `SnapshotLoadEpisodic`.
    SnapshotEpisodicBytesRead,
    /// Bytes written to the cold archive during one hot→cold spill (`spilled` records ×
    /// `somnus_memory::cold::RECORD_BYTES`) — derived from the record count already being
    /// tallied, not a second measurement.
    ColdArchiveSpillBytesWritten,
    /// A NaN/Inf value was detected in `mean_train_loss`/`mean_output_mse`/`mean_proj_loss` at
    /// the end of one NREM replay. Without this, a training blowup on an unattended run would
    /// just render as a weird number (or `null`, since `serde_json` serializes non-finite
    /// floats that way) with no alarm — this is the alarm.
    TrainingNanDetected,
}

impl Metric {
    pub const COUNT: usize = 100;

    const fn as_index(self) -> usize {
        match self {
            Metric::IngestKwta => 0,
            Metric::IngestSymbolic => 1,
            Metric::IngestBind => 2,
            Metric::IngestPrior => 3,
            Metric::IngestSettle => 4,
            Metric::IngestPriorUpdate => 5,
            Metric::SleepAutodiffRefresh => 6,
            Metric::SleepNrem => 7,
            Metric::SleepDistillPrecompute => 8,
            Metric::SleepFusedReplayWake => 9,
            Metric::SleepProjDistill => 10,
            Metric::SleepRem => 11,
            Metric::SleepSpill => 12,
            Metric::SleepSnapshot => 13,
            Metric::SleepGistKmeans => 14,
            Metric::SleepWarmRebuild => 15,
            Metric::SleepTotal => 16,
            Metric::EncodeSpeech => 17,
            Metric::EncodeEnv => 18,
            Metric::EncodeVideo => 19,
            Metric::MainPerceptionIngest => 20,
            Metric::MainWarmQuery => 21,
            Metric::KwtaScoreMatmul => 22,
            Metric::KwtaSelectTopK => 23,
            Metric::RefreshLanguageCache => 24,
            Metric::BindingForward => 25,
            Metric::OrtRunEncoder => 26,
            Metric::OrtRunBinding => 27,
            Metric::InferEpisodeOutputs => 28,
            Metric::FusedReplaySaePhase => 29,
            Metric::FusedReplayHeadPhase => 30,
            Metric::FusedReplayBackward => 31,
            Metric::RebuildOrtSession => 32,
            Metric::ExportWeightsForOnnx => 33,
            Metric::EnableOrtInference => 34,
            Metric::HomeostaticDecay => 35,
            Metric::MeasureMse => 36,
            Metric::CollectHighResidualInputs => 37,
            Metric::GrowRegion => 38,
            Metric::GrowRegionSeeded => 39,
            Metric::PruneRegion => 40,
            Metric::SynthesizeEpisode => 41,
            Metric::MeasureOutputMseInference => 42,
            Metric::WakeReconstructStep => 43,
            Metric::WakeWorkerTrainStep => 44,
            Metric::EmaPredict => 45,
            Metric::EmaUpdateFromWeights => 46,
            Metric::EmaSyncFlat => 47,
            Metric::RefreshAutodiffState => 48,
            Metric::ColdArchiveOpen => 49,
            Metric::ColdArchiveTruncate => 50,
            Metric::ColdArchiveEvict => 51,
            Metric::SemanticStoreLoadWarm => 52,
            Metric::RecollectionPromote => 53,
            Metric::SnapshotSaveEpisodic => 54,
            Metric::SnapshotLoadEpisodic => 55,
            Metric::SnapshotSave => 56,
            Metric::SnapshotLoad => 57,
            Metric::PqConvertColdToPq => 58,
            Metric::MediaFileDecode => 59,
            Metric::ActionDecoderInit => 60,
            Metric::ActionGenerate => 61,
            Metric::GainCompute => 62,
            Metric::LangEmbed => 63,
            Metric::AsrDecode => 64,
            Metric::AudioResample => 65,
            Metric::MainRecordEpisode => 66,
            Metric::RegionNUnits => 67,
            Metric::RegionKActive => 68,
            Metric::ColdArchiveEvictedRecords => 69,
            Metric::VideoFramesEncoded => 70,
            Metric::VideoFramesSkipped => 71,
            Metric::RefreshLanguageCacheSkipped => 72,
            Metric::RefreshLanguageCacheRun => 73,
            Metric::OrtEncoderFallback => 74,
            Metric::OrtBindingFallback => 75,
            Metric::OrtSessionInitFailed => 76,
            Metric::OrtSessionRebuildFailed => 77,
            Metric::ProcessRssBytes => 78,
            Metric::MemoryDirDiskFreeBytes => 79,
            Metric::GrowthBatchSize => 80,
            Metric::SleepPersistenceWriteFailed => 81,
            Metric::SemanticStoreDump => 82,
            Metric::SemanticStoreDumpBytes => 83,
            Metric::NremStepError => 84,
            Metric::ColdArchiveOpError => 85,
            Metric::RemProbeError => 86,
            Metric::PerceptionIngestError => 87,
            Metric::WakeReconstructError => 88,
            Metric::AsrError => 89,
            Metric::LangEmbedError => 90,
            Metric::ActionGenerationError => 91,
            Metric::WakeReconWorkerDied => 92,
            Metric::StartupRestoreDegraded => 93,
            Metric::MediaSourceError => 94,
            Metric::MiscWarning => 95,
            Metric::SnapshotEpisodicBytesWritten => 96,
            Metric::SnapshotEpisodicBytesRead => 97,
            Metric::ColdArchiveSpillBytesWritten => 98,
            Metric::TrainingNanDetected => 99,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Metric::IngestKwta => "ingest_kwta",
            Metric::IngestSymbolic => "ingest_symbolic",
            Metric::IngestBind => "ingest_bind",
            Metric::IngestPrior => "ingest_prior",
            Metric::IngestSettle => "ingest_settle",
            Metric::IngestPriorUpdate => "ingest_prior_update",
            Metric::SleepAutodiffRefresh => "sleep_autodiff_refresh",
            Metric::SleepNrem => "sleep_nrem",
            Metric::SleepDistillPrecompute => "sleep_distill_precompute",
            Metric::SleepFusedReplayWake => "sleep_fused_replay_wake",
            Metric::SleepProjDistill => "sleep_proj_distill",
            Metric::SleepRem => "sleep_rem",
            Metric::SleepSpill => "sleep_spill",
            Metric::SleepSnapshot => "sleep_snapshot",
            Metric::SleepGistKmeans => "sleep_gist_kmeans",
            Metric::SleepWarmRebuild => "sleep_warm_rebuild",
            Metric::SleepTotal => "sleep_total",
            Metric::EncodeSpeech => "encode_speech",
            Metric::EncodeEnv => "encode_env",
            Metric::EncodeVideo => "encode_video",
            Metric::MainPerceptionIngest => "main_perception_ingest",
            Metric::MainWarmQuery => "main_warm_query",
            Metric::KwtaScoreMatmul => "kwta_score_matmul",
            Metric::KwtaSelectTopK => "kwta_select_topk",
            Metric::RefreshLanguageCache => "refresh_language_cache",
            Metric::BindingForward => "binding_forward",
            Metric::OrtRunEncoder => "ort_run_encoder",
            Metric::OrtRunBinding => "ort_run_binding",
            Metric::InferEpisodeOutputs => "infer_episode_outputs",
            Metric::FusedReplaySaePhase => "fused_replay_sae_phase",
            Metric::FusedReplayHeadPhase => "fused_replay_head_phase",
            Metric::FusedReplayBackward => "fused_replay_backward",
            Metric::RebuildOrtSession => "rebuild_ort_session",
            Metric::ExportWeightsForOnnx => "export_weights_for_onnx",
            Metric::EnableOrtInference => "enable_ort_inference",
            Metric::HomeostaticDecay => "homeostatic_decay",
            Metric::MeasureMse => "measure_mse",
            Metric::CollectHighResidualInputs => "collect_high_residual_inputs",
            Metric::GrowRegion => "grow_region",
            Metric::GrowRegionSeeded => "grow_region_seeded",
            Metric::PruneRegion => "prune_region",
            Metric::SynthesizeEpisode => "synthesize_episode",
            Metric::MeasureOutputMseInference => "measure_output_mse_inference",
            Metric::WakeReconstructStep => "wake_reconstruct_step",
            Metric::WakeWorkerTrainStep => "wake_worker_train_step",
            Metric::EmaPredict => "ema_predict",
            Metric::EmaUpdateFromWeights => "ema_update_from_weights",
            Metric::EmaSyncFlat => "ema_sync_flat",
            Metric::RefreshAutodiffState => "refresh_autodiff_state",
            Metric::ColdArchiveOpen => "cold_archive_open",
            Metric::ColdArchiveTruncate => "cold_archive_truncate",
            Metric::ColdArchiveEvict => "cold_archive_evict",
            Metric::SemanticStoreLoadWarm => "semantic_store_load_warm",
            Metric::RecollectionPromote => "recollection_promote",
            Metric::SnapshotSaveEpisodic => "snapshot_save_episodic",
            Metric::SnapshotLoadEpisodic => "snapshot_load_episodic",
            Metric::SnapshotSave => "snapshot_save",
            Metric::SnapshotLoad => "snapshot_load",
            Metric::PqConvertColdToPq => "pq_convert_cold_to_pq",
            Metric::MediaFileDecode => "media_file_decode",
            Metric::ActionDecoderInit => "action_decoder_init",
            Metric::ActionGenerate => "action_generate",
            Metric::GainCompute => "gain_compute",
            Metric::LangEmbed => "lang_embed",
            Metric::AsrDecode => "asr_decode",
            Metric::AudioResample => "audio_resample",
            Metric::MainRecordEpisode => "main_record_episode",
            Metric::RegionNUnits => "region_n_units",
            Metric::RegionKActive => "region_k_active",
            Metric::ColdArchiveEvictedRecords => "cold_archive_evicted_records",
            Metric::VideoFramesEncoded => "video_frames_encoded",
            Metric::VideoFramesSkipped => "video_frames_skipped",
            Metric::RefreshLanguageCacheSkipped => "refresh_language_cache_skipped",
            Metric::RefreshLanguageCacheRun => "refresh_language_cache_run",
            Metric::OrtEncoderFallback => "ort_encoder_fallback",
            Metric::OrtBindingFallback => "ort_binding_fallback",
            Metric::OrtSessionInitFailed => "ort_session_init_failed",
            Metric::OrtSessionRebuildFailed => "ort_session_rebuild_failed",
            Metric::ProcessRssBytes => "process_rss_bytes",
            Metric::MemoryDirDiskFreeBytes => "memory_dir_disk_free_bytes",
            Metric::GrowthBatchSize => "growth_batch_size",
            Metric::SleepPersistenceWriteFailed => "sleep_persistence_write_failed",
            Metric::SemanticStoreDump => "semantic_store_dump",
            Metric::SemanticStoreDumpBytes => "semantic_store_dump_bytes",
            Metric::NremStepError => "nrem_step_error",
            Metric::ColdArchiveOpError => "cold_archive_op_error",
            Metric::RemProbeError => "rem_probe_error",
            Metric::PerceptionIngestError => "perception_ingest_error",
            Metric::WakeReconstructError => "wake_reconstruct_error",
            Metric::AsrError => "asr_error",
            Metric::LangEmbedError => "lang_embed_error",
            Metric::ActionGenerationError => "action_generation_error",
            Metric::WakeReconWorkerDied => "wake_recon_worker_died",
            Metric::StartupRestoreDegraded => "startup_restore_degraded",
            Metric::MediaSourceError => "media_source_error",
            Metric::MiscWarning => "misc_warning",
            Metric::SnapshotEpisodicBytesWritten => "snapshot_episodic_bytes_written",
            Metric::SnapshotEpisodicBytesRead => "snapshot_episodic_bytes_read",
            Metric::ColdArchiveSpillBytesWritten => "cold_archive_spill_bytes_written",
            Metric::TrainingNanDetected => "training_nan_detected",
        }
    }

    const ALL: [Metric; Metric::COUNT] = [
        Metric::IngestKwta,
        Metric::IngestSymbolic,
        Metric::IngestBind,
        Metric::IngestPrior,
        Metric::IngestSettle,
        Metric::IngestPriorUpdate,
        Metric::SleepAutodiffRefresh,
        Metric::SleepNrem,
        Metric::SleepDistillPrecompute,
        Metric::SleepFusedReplayWake,
        Metric::SleepProjDistill,
        Metric::SleepRem,
        Metric::SleepSpill,
        Metric::SleepSnapshot,
        Metric::SleepGistKmeans,
        Metric::SleepWarmRebuild,
        Metric::SleepTotal,
        Metric::EncodeSpeech,
        Metric::EncodeEnv,
        Metric::EncodeVideo,
        Metric::MainPerceptionIngest,
        Metric::MainWarmQuery,
        Metric::KwtaScoreMatmul,
        Metric::KwtaSelectTopK,
        Metric::RefreshLanguageCache,
        Metric::BindingForward,
        Metric::OrtRunEncoder,
        Metric::OrtRunBinding,
        Metric::InferEpisodeOutputs,
        Metric::FusedReplaySaePhase,
        Metric::FusedReplayHeadPhase,
        Metric::FusedReplayBackward,
        Metric::RebuildOrtSession,
        Metric::ExportWeightsForOnnx,
        Metric::EnableOrtInference,
        Metric::HomeostaticDecay,
        Metric::MeasureMse,
        Metric::CollectHighResidualInputs,
        Metric::GrowRegion,
        Metric::GrowRegionSeeded,
        Metric::PruneRegion,
        Metric::SynthesizeEpisode,
        Metric::MeasureOutputMseInference,
        Metric::WakeReconstructStep,
        Metric::WakeWorkerTrainStep,
        Metric::EmaPredict,
        Metric::EmaUpdateFromWeights,
        Metric::EmaSyncFlat,
        Metric::RefreshAutodiffState,
        Metric::ColdArchiveOpen,
        Metric::ColdArchiveTruncate,
        Metric::ColdArchiveEvict,
        Metric::SemanticStoreLoadWarm,
        Metric::RecollectionPromote,
        Metric::SnapshotSaveEpisodic,
        Metric::SnapshotLoadEpisodic,
        Metric::SnapshotSave,
        Metric::SnapshotLoad,
        Metric::PqConvertColdToPq,
        Metric::MediaFileDecode,
        Metric::ActionDecoderInit,
        Metric::ActionGenerate,
        Metric::GainCompute,
        Metric::LangEmbed,
        Metric::AsrDecode,
        Metric::AudioResample,
        Metric::MainRecordEpisode,
        Metric::RegionNUnits,
        Metric::RegionKActive,
        Metric::ColdArchiveEvictedRecords,
        Metric::VideoFramesEncoded,
        Metric::VideoFramesSkipped,
        Metric::RefreshLanguageCacheSkipped,
        Metric::RefreshLanguageCacheRun,
        Metric::OrtEncoderFallback,
        Metric::OrtBindingFallback,
        Metric::OrtSessionInitFailed,
        Metric::OrtSessionRebuildFailed,
        Metric::ProcessRssBytes,
        Metric::MemoryDirDiskFreeBytes,
        Metric::GrowthBatchSize,
        Metric::SleepPersistenceWriteFailed,
        Metric::SemanticStoreDump,
        Metric::SemanticStoreDumpBytes,
        Metric::NremStepError,
        Metric::ColdArchiveOpError,
        Metric::RemProbeError,
        Metric::PerceptionIngestError,
        Metric::WakeReconstructError,
        Metric::AsrError,
        Metric::LangEmbedError,
        Metric::ActionGenerationError,
        Metric::WakeReconWorkerDied,
        Metric::StartupRestoreDegraded,
        Metric::MediaSourceError,
        Metric::MiscWarning,
        Metric::SnapshotEpisodicBytesWritten,
        Metric::SnapshotEpisodicBytesRead,
        Metric::ColdArchiveSpillBytesWritten,
        Metric::TrainingNanDetected,
    ];
}

/// Fixed-size atomic counter/timer registry, one slot pair per [`Metric`]. Relaxed ordering
/// throughout — these are independent counters, not synchronization primitives.
pub struct Registry {
    counts: [AtomicU64; Metric::COUNT],
    total_ns: [AtomicU64; Metric::COUNT],
}

impl Registry {
    const fn new() -> Self {
        Registry {
            counts: [const { AtomicU64::new(0) }; Metric::COUNT],
            total_ns: [const { AtomicU64::new(0) }; Metric::COUNT],
        }
    }

    /// Record one occurrence of `metric` taking `dur`. Called from both `timed!` (Fine, only
    /// when `fine-metrics` is enabled) and `timed_core!` (Core, always).
    pub fn record(&self, metric: Metric, dur: Duration) {
        let i = metric.as_index();
        self.counts[i].fetch_add(1, Ordering::Relaxed);
        self.total_ns[i].fetch_add(dur.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record `delta` occurrences of `metric` with no timing (a plain counter).
    pub fn add(&self, metric: Metric, delta: u64) {
        self.counts[metric.as_index()].fetch_add(delta, Ordering::Relaxed);
    }

    /// Drain every non-zero metric as `{name: {"count": n, "total_ns": t}}` and reset all
    /// counters to zero. Called once per sleep cycle. With `fine-metrics` off, only Core-tier
    /// entries are ever non-zero; with it on, Fine-tier entries appear alongside them.
    pub fn snapshot_and_reset(&self) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        for m in Metric::ALL {
            let i = m.as_index();
            let count = self.counts[i].swap(0, Ordering::Relaxed);
            let total_ns = self.total_ns[i].swap(0, Ordering::Relaxed);
            if count > 0 || total_ns > 0 {
                out.insert(
                    m.name().to_string(),
                    serde_json::json!({ "count": count, "total_ns": total_ns }),
                );
            }
        }
        serde_json::Value::Object(out)
    }
}

/// The single global registry, shared by both tiers.
pub static METRICS: Registry = Registry::new();

/// **Fine tier.** Time `$body` and record it under `$metric` in [`METRICS`]. With `fine-metrics`
/// off, expands to a bare `$body` — no `Instant::now()` call, no atomic write, textually
/// zero-cost. Use for granular sub-breakdowns not worth paying for on every build.
#[cfg(feature = "fine-metrics")]
#[macro_export]
macro_rules! timed {
    ($metric:expr, $body:expr) => {{
        let __somnus_metrics_t0 = ::std::time::Instant::now();
        let __somnus_metrics_r = $body;
        $crate::METRICS.record($metric, __somnus_metrics_t0.elapsed());
        __somnus_metrics_r
    }};
}

#[cfg(not(feature = "fine-metrics"))]
#[macro_export]
macro_rules! timed {
    ($metric:expr, $body:expr) => {
        $body
    };
}

/// **Fine tier.** Increment `$metric`'s count by `$delta` (no timing) in [`METRICS`]. With
/// `fine-metrics` off, expands to nothing.
#[cfg(feature = "fine-metrics")]
#[macro_export]
macro_rules! counted {
    ($metric:expr, $delta:expr) => {{
        $crate::METRICS.add($metric, $delta);
    }};
}

#[cfg(not(feature = "fine-metrics"))]
#[macro_export]
macro_rules! counted {
    ($metric:expr, $delta:expr) => {};
}

/// **Core tier.** Time `$body`, record it under `$metric` in [`METRICS`] unconditionally (both
/// feature states), and evaluate to `(result, elapsed)` so the caller can reuse the duration for
/// whatever it previously computed by hand (an `_ms`/`_us` field, an accumulator, a log field) —
/// this is the direct replacement for a hand-rolled `let t = Instant::now(); ...;
/// t.elapsed()` pair. Always recording is intentional: this tier *is* the "big item" metrics this
/// codebase has always reported, just funneled through one macro instead of copy-pasted
/// boilerplate at every site.
#[macro_export]
macro_rules! timed_core {
    ($metric:expr, $body:expr) => {{
        let __somnus_metrics_t0 = ::std::time::Instant::now();
        let __somnus_metrics_r = $body;
        let __somnus_metrics_d = __somnus_metrics_t0.elapsed();
        $crate::METRICS.record($metric, __somnus_metrics_d);
        (__somnus_metrics_r, __somnus_metrics_d)
    }};
}

/// **Core tier.** Increment `$metric`'s count by `$delta` (no timing) in [`METRICS`]
/// unconditionally.
#[macro_export]
macro_rules! counted_core {
    ($metric:expr, $delta:expr) => {{
        $crate::METRICS.add($metric, $delta);
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // METRICS is a single global registry; serialize tests so they don't observe each other's
    // writes (cargo test runs tests in a crate concurrently by default).
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn timed_core_records_and_returns_result_and_duration() {
        let _guard = TEST_LOCK.lock().unwrap();
        METRICS.snapshot_and_reset();

        let (r, dur) = timed_core!(Metric::SleepNrem, {
            std::thread::sleep(Duration::from_micros(50));
            42
        });
        assert_eq!(r, 42);
        assert!(dur.as_nanos() > 0);

        let snap = METRICS.snapshot_and_reset();
        let entry = &snap["sleep_nrem"];
        assert_eq!(entry["count"], 1);
        assert!(entry["total_ns"].as_u64().unwrap() > 0);
    }

    #[test]
    fn counted_core_increments_without_timing() {
        let _guard = TEST_LOCK.lock().unwrap();
        METRICS.snapshot_and_reset();
        counted_core!(Metric::MainWarmQuery, 3);
        counted_core!(Metric::MainWarmQuery, 4);
        let snap = METRICS.snapshot_and_reset();
        assert_eq!(snap["main_warm_query"]["count"], 7);
        assert_eq!(snap["main_warm_query"]["total_ns"], 0);
    }

    #[cfg(feature = "fine-metrics")]
    #[test]
    fn timed_records_count_and_nonzero_duration() {
        let _guard = TEST_LOCK.lock().unwrap();
        METRICS.snapshot_and_reset();

        let r = timed!(Metric::KwtaScoreMatmul, {
            std::thread::sleep(Duration::from_micros(50));
            42
        });
        assert_eq!(r, 42);

        let snap = METRICS.snapshot_and_reset();
        let entry = &snap["kwta_score_matmul"];
        assert_eq!(entry["count"], 1);
        assert!(entry["total_ns"].as_u64().unwrap() > 0);
    }

    #[cfg(feature = "fine-metrics")]
    #[test]
    fn counted_increments_without_timing() {
        let _guard = TEST_LOCK.lock().unwrap();
        METRICS.snapshot_and_reset();
        counted!(Metric::IngestKwta, 3);
        counted!(Metric::IngestKwta, 4);
        let snap = METRICS.snapshot_and_reset();
        assert_eq!(snap["ingest_kwta"]["count"], 7);
        assert_eq!(snap["ingest_kwta"]["total_ns"], 0);
    }

    #[test]
    fn snapshot_and_reset_zeroes_registry() {
        let _guard = TEST_LOCK.lock().unwrap();
        let _ = timed_core!(Metric::IngestBind, 1);
        let first = METRICS.snapshot_and_reset();
        assert!(first.get("ingest_bind").is_some());
        let second = METRICS.snapshot_and_reset();
        assert_eq!(second, serde_json::json!({}));
    }
}
