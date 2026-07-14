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
/// Core-tier variant replaces, where applicable. Add variants as specific investigations need
/// them — this is not an exhaustive upfront inventory of every data-touching call site.
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
}

impl Metric {
    pub const COUNT: usize = 24;

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
