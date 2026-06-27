/// Sleep cycle: NREM-like consolidation + REM-like self-probe + growth.
///
/// Blocks the main loop. Checks `interrupt` flag after every replay step;
/// exits early if set. Language-tagged traces are replayed first each cycle
/// so they are never skipped even when the salience gate fires early.
use std::{
    collections::HashMap,
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use anyhow::Result;
use somnus_memory::{
    save_growth_cooldown_remaining, save_region_sizes, save_snapshot, weights_path, EpisodicBuffer,
    IngestSource, RecollectionLayer,
};
use somnus_perception::{PerceptionProcessor, RegionId, N_UNITS};

use crate::action::ActionDecoder;

const DECAY_RATE: f32 = 1e-4;
const GROWTH_THRESHOLD: f32 = 0.1;
const GROWTH_MIN_SAMPLES: usize = 16;
const GROWTH_BATCH: usize = 64;

/// Mean warm-recall L2 above this threshold supplements the MSE growth signal.
/// Only applied when warm had pre-spill entries (i.e., cross-cycle/cross-session data).
/// Kept for the warm_recall_poor log field but no longer gates growth directly —
/// gist_recall_rate is the primary growth signal (see GIST_GROWTH_THRESHOLD).
const WARM_RECALL_GROWTH_THRESHOLD: f32 = 5.0;

/// Neuromodulated learning rate schedule for NREM replay.
///
/// Combines cosine annealing over the cycle length (high plasticity early,
/// low late) with a loss-trend gate: if the loss EWMA is not improving, the
/// multiplier shrinks toward LR_STAGNANT_SCALE, reducing wasted updates.
///
/// `step`        — 0-indexed step within this NREM cycle
/// `total`       — total steps planned for this cycle
/// `loss_ewma`   — exponential moving average of loss up to this step
/// `prev_ewma`   — ewma from the previous step (or 0.0 on the first step)
const LR_MAX: f64 = 1e-3;
const LR_MIN: f64 = 1e-5;
const LR_STAGNANT_SCALE: f64 = 0.25; // multiplier when loss has plateaued
/// Upper bound on warm-recall L2 used for salience normalisation. Traces above
/// this distance are treated as maximally novel; below proportionally so.
///
/// Recalibrated for L2-normalised episode vectors: max possible L2 between unit
/// vectors is 2.0 (orthogonal/opposite). Previous value of 10.0 was calibrated
/// on un-normalised vectors and capped effective salience boost to ~20% of intent.
const LR_SALIENCE_CAP: f32 = 2.0;
/// Maximum LR multiplier for maximally novel (high warm-recall L2) traces.
/// Multiplier is linearly interpolated from 1.0 (familiar) to this value (novel).
const LR_SALIENCE_MAX_SCALE: f64 = 2.0;

/// Neuromodulated LR: cosine annealing × loss-trend gate × salience boost.
///
/// `batch_salience` — mean warm-recall L2 of the batch (high = novel).
/// Sentinel value f32::MAX (no warm neighbour yet, first run) maps to 0.5
/// novelty so the schedule stays well-behaved on fresh memory directories.
fn nrem_lr(step: usize, total: usize, loss_ewma: f32, prev_ewma: f32, batch_salience: f32) -> f64 {
    // Cosine annealing: full cycle from LR_MAX → LR_MIN
    let t = if total > 1 {
        step as f64 / (total - 1) as f64
    } else {
        1.0
    };
    let cosine = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + (std::f64::consts::PI * t).cos());

    // Loss-trend gate: scale down when improvement has stalled
    let trend_scale = if prev_ewma < 1e-9 || step == 0 {
        1.0 // first step — no history yet
    } else {
        let improvement = (prev_ewma - loss_ewma) / prev_ewma;
        if improvement > 0.01 {
            1.0 // improving >1% per step: full lr
        } else if improvement > -0.01 {
            0.5 // plateau (−1%..+1%): half lr
        } else {
            LR_STAGNANT_SCALE // worsening: quarter lr
        }
    };

    // Salience boost: novel traces replay at higher LR so the model spends more
    // gradient budget on content it has seen least. f32::MAX sentinel (no warm
    // neighbour) maps to 0.5 — neutral, not max — to avoid inflated LR on the
    // very first cycle when warm is empty and all saliences are sentinel.
    let salience_norm = if batch_salience >= f32::MAX / 2.0 {
        0.5_f64
    } else {
        (batch_salience / LR_SALIENCE_CAP).clamp(0.0, 1.0) as f64
    };
    let salience_scale = 1.0 + (LR_SALIENCE_MAX_SCALE - 1.0) * salience_norm;

    cosine * trend_scale * salience_scale
}

#[allow(dead_code)]
pub struct SleepStats {
    pub traces_replayed: usize,
    pub mean_train_loss: f32,
    /// Mean MSE of output-head reconstruction during NREM replay (ADR 0010).
    pub mean_output_mse: f32,
    pub per_region_mse: [f32; 5],
    pub regions_grown: Vec<RegionId>,
    pub was_interrupted: bool,
    /// Why the NREM loop exited: "traces_exhausted", "loss_target_met", "time_budget", "salience".
    pub nrem_exit_reason: &'static str,
    /// Mean L2 distance to nearest warm-memory neighbor during REM probe.
    /// Some only when warm had pre-spill entries (cross-cycle/cross-session signal).
    pub warm_recall_l2: Option<f32>,
    /// Fraction of probe episodes whose top-1 warm neighbor is in the same k-means
    /// gist cluster (10c). None when warm is too small to cluster (< GIST_K points).
    pub gist_recall_rate: Option<f32>,
    /// Mean surrogate loss from projection training steps (ADR 0012).
    /// None when no text-target traces were present.
    pub mean_proj_loss: Option<f32>,
    /// Max per-step projection loss seen during NREM (useful for diagnosing spikes).
    pub proj_loss_max: Option<f32>,
    /// Number of traces that had a text_target and were used to train the projection.
    pub decoder_trainable_records: usize,
    /// Number of synthetic episodes produced during REM creative recombination (11b).
    pub synthetics_produced: usize,
    /// Mean output-head MSE on synthetic episodes measured post-NREM (inference mode).
    /// None when no synthetic pairs could be formed.
    pub mean_synthetic_output_mse: Option<f32>,
    /// Mean output-head MSE on real episodes measured post-NREM (inference mode).
    /// None when traces_for_rem is empty.
    pub mean_real_inference_mse: Option<f32>,
    /// Mean Shannon entropy of ingest_source distribution per gist cluster (Gap 8 / ADR 0021).
    /// None when warm is too small to cluster or all episodes have Unknown source.
    /// High = diverse modalities in same cluster (cross-modal grounding).
    pub cross_modal_gist_entropy: Option<f32>,
    /// Gist recall rate computed only on within-session warm records (ADR 0024).
    /// Excludes prior-session embeddings that may have drifted due to NREM weight updates.
    /// None when the within-session warm subset is below GIST_MIN_WARM (typical for cycle 1).
    pub within_session_gist_rate: Option<f32>,
    /// Regions that had grown units pruned during REM (12c).
    pub regions_pruned: Vec<RegionId>,
    /// Total grown units removed across all pruned regions.
    pub total_units_pruned: usize,
    /// Cooldown cycles remaining after this cycle (persisted across sessions).
    pub growth_cooldown_remaining: usize,
    /// Wall time for the NREM batch-replay loop (ms).
    pub nrem_wall_ms: u64,
    /// Wall time to precompute LM hidden-state targets for Gap-6 distillation (ms).
    pub distill_precompute_ms: u64,
    /// Wall time for all REM work: warm probe + MSE measurement + growth + pruning + synthesis (ms).
    pub rem_wall_ms: u64,
    /// Wall time for hot→cold/warm spill and optional warm-HNSW rebuild (ms).
    pub spill_wall_ms: u64,
    /// Wall time for snapshot writes (perception weights, warm HNSW, region sizes) (ms).
    pub snapshot_wall_ms: u64,
    /// Number of language-tagged traces for which a valid LM hidden-state target was computed.
    pub lm_hidden_computed: usize,
    /// Unique LM inference calls made (deduplicated by token-id content).
    /// With lang_oversample=9, this ≈ lm_hidden_computed / 10.
    pub lm_hidden_unique: usize,
    /// Total replay list length fed to the NREM loop (unique + oversampled copies).
    /// Coverage = traces_replayed / nrem_budget_traces.
    pub nrem_budget_traces: usize,
    /// Unique episodes in the episodic buffer at sleep time (before language oversampling).
    pub nrem_unique_traces: usize,
}

/// Number of regions; episode vector is N_EPISODE_REGIONS × (EPISODE_DIM / N_EPISODE_REGIONS).
const N_EPISODE_REGIONS: usize = 5;

/// Number of gist clusters for REM self-probe (10c).
const GIST_K: usize = 32;
/// Max warm-record sample for k-means (larger warm stores are subsampled).
const GIST_SAMPLE: usize = 2048;
/// k-means iteration cap.
const GIST_MAX_ITER: usize = 20;
/// Minimum warm-store size to attempt gist clustering.
const GIST_MIN_WARM: usize = GIST_K;
/// Growth fires when gist-recall rate is below this fraction.
/// Raised from 0.5: with 4-clip loops gist trivially hits 1.0 post-NREM,
/// so 0.5 never fired. 0.85 requires tight gist before suppressing growth.
const GIST_GROWTH_THRESHOLD: f32 = 0.85;
/// Minimum gist recall rate for pruning to be considered safe (12c).
const PRUNE_GIST_THRESHOLD: f32 = 0.8;

/// Interpolation weight for REM creative blending (11b). α=0.5 → equal mix.
const REM_SYNTH_ALPHA: f32 = 0.5;
/// Minimum L2 distance between an episode pair to count as genuinely different content (11b).
///
/// Recalibrated for L2-normalised episode vectors (max distance = 2.0):
///   ~0.0–0.3: same moment / same clip
///   ~0.3–0.8: similar content
///   ~1.0–1.5: different content (target range for synthesis)
///   ~1.5–2.0: nearly opposite (rare)
/// Previous value of 2.0 was calibrated on un-normalised vectors and was at the
/// maximum possible distance for unit vectors, so no pairs ever qualified.
const REM_SYNTH_MIN_L2: f32 = 1.0;
/// Maximum number of synthetic episodes produced per sleep cycle (11b).
const REM_SYNTH_MAX_N: usize = 8;

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn nearest_centroid(centroids: &[Vec<f32>], point: &[f32]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            l2_sq(point, a)
                .partial_cmp(&l2_sq(point, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Minimal Lloyd's k-means. Initialises centroids by even-index sampling (deterministic).
fn kmeans(points: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    if points.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(points.len());
    let dim = points[0].len();

    // Evenly-spaced initialisation — deterministic, no RNG.
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|i| points[i * points.len() / k].clone())
        .collect();

    let mut assignments = vec![0usize; points.len()];

    for _ in 0..max_iter {
        let mut changed = false;
        for (j, p) in points.iter().enumerate() {
            let best = nearest_centroid(&centroids, p);
            if assignments[j] != best {
                assignments[j] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Recompute centroids.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (j, p) in points.iter().enumerate() {
            let c = assignments[j];
            counts[c] += 1;
            for (s, v) in sums[c].iter_mut().zip(p.iter()) {
                *s += v;
            }
        }
        for (c_vec, (sum, cnt)) in centroids.iter_mut().zip(sums.iter().zip(counts.iter())) {
            if *cnt > 0 {
                for (cv, sv) in c_vec.iter_mut().zip(sum.iter()) {
                    *cv = sv / *cnt as f32;
                }
            }
        }
    }
    centroids
}

/// Decompose a 640-dim episode vector into per-region salience weights for the
/// cross-modal gist entropy histogram.
///
/// The episode vector is laid out as N_EPISODE_REGIONS equal-width slices (one per
/// region). We compute the L2 norm of each slice and normalise to sum=1. This lets
/// Mixed episodes distribute weight proportionally across all contributing regions
/// rather than collapsing to a single "Mixed" slot, which would hide cross-modal
/// structure in the gist clusters.
///
/// Degenerate case (all norms zero): returns uniform 1/5 weights.
fn regional_weights(ep_vec: &[f32]) -> [f32; N_EPISODE_REGIONS] {
    let d = ep_vec.len() / N_EPISODE_REGIONS;
    let norms: [f32; N_EPISODE_REGIONS] = std::array::from_fn(|i| {
        ep_vec[i * d..(i + 1) * d]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
    });
    let total: f32 = norms.iter().sum();
    if total == 0.0 {
        return [1.0 / N_EPISODE_REGIONS as f32; N_EPISODE_REGIONS];
    }
    norms.map(|n| n / total)
}

/// `pending_embs` counts embeddings produced by encoder tasks but not yet consumed
/// by the main loop. A high count means the world is active while the system sleeps;
/// once it exceeds `SALIENCE_THRESHOLD` the current replay step is the last one.
#[allow(clippy::too_many_arguments)]
pub fn run_sleep_cycle(
    perception: &mut PerceptionProcessor,
    episodic: &mut EpisodicBuffer,
    recall: &mut RecollectionLayer,
    memory_dir: &Path,
    interrupt: Arc<AtomicBool>,
    pending_embs: Arc<AtomicUsize>,
    decoder: &mut Option<ActionDecoder>,
    sleep_loss_target: f32,
    sleep_max_secs: u64,
    prune_fraction: f32,
    prune_min_grown: usize,
    replay_batch_size: usize,
    nrem_min_traces: usize,
    nrem_salience_threshold: usize,
    lang_oversample: usize,
    replay_epochs: usize,
    max_warm_records: usize,
    max_cold_records: usize,
    growth_cooldown: usize,
    cooldown_remaining: usize,
    session_start_secs: f64,
    training_data_path: Option<&Path>,
) -> Result<SleepStats> {
    let sleep_start = Instant::now();
    tracing::info!(episodic_len = episodic.len(), "sleep cycle started");
    // rem_wall_ms is accumulated across two REM sub-sections; the others are one-shot.
    let mut rem_wall_ms = 0u64;

    let mut traces_replayed = 0usize;
    let mut total_train_loss = 0.0f32;
    let mut total_output_mse = 0.0f32;
    let mut output_mse_n = 0usize;
    let mut was_interrupted = false;
    let mut nrem_exit_reason: &'static str = "traces_exhausted";
    let mut total_proj_loss = 0.0f32;
    let mut proj_loss_n = 0usize;
    let mut proj_loss_max = 0.0f32;

    // Snapshot warm size before spill so REM can detect cross-cycle entries.
    let pre_spill_warm_len = recall.warm.len();

    // ── NREM-like: replay + train ─────────────────────────────────────────────
    // Collect snapshots; avoids borrowing episodic during training.
    // Always start from 0 each cycle: language-priority ordering makes the
    // cross-cycle checkpoint unnecessary and would misplace prioritized traces.
    episodic.replay_checkpoint = 0;
    // Inner tuple for building raw traces before LM hidden states are attached.
    type RawTrace = (Vec<f32>, [Vec<f32>; 5], Option<Vec<u32>>, f32);
    // Fifth field: precomputed LM hidden state (Gap 6 distillation target).
    type TraceSnapshot = (
        Vec<f32>,
        [Vec<f32>; 5],
        Option<Vec<u32>>,
        f32,
        Option<Vec<f32>>,
    );

    // Precompute LM hidden states before the NREM loop so the hot path doesn't
    // pay the lm_session cost per batch. text_hidden_states is &self (inference
    // only), so we borrow decoder immutably here, before the mutable NREM step.
    let raw_traces: Vec<RawTrace> = episodic
        .iter_traces()
        .map(|t| {
            (
                t.record.values.clone(),
                t.regional.clone(),
                t.text_target.clone(),
                t.record.salience,
            )
        })
        .collect();
    let t_distill = Instant::now();
    // Deduplicate LM inference by token-id content. With lang_oversample=N, each
    // language trace appears N+1 times in raw_traces; without dedup that means N+1
    // identical ORT sessions for the same text. Per-cycle cache collapses within-cycle
    // repeats; text_hidden_states() also maintains a cross-cycle cache so clip-loop
    // repetitions across sleep cycles pay the ORT cost only once per session.
    let (lm_hidden_targets, lm_hidden_unique): (Vec<Option<Vec<f32>>>, usize) =
        if let Some(ref mut dec) = decoder {
            let mut cache: HashMap<Vec<u32>, Option<Vec<f32>>> = HashMap::new();
            let targets = raw_traces
                .iter()
                .map(|(_, _, ids, _)| match ids {
                    Some(ids) => cache
                        .entry(ids.clone())
                        .or_insert_with(|| dec.text_hidden_states(ids).ok())
                        .clone(),
                    None => None,
                })
                .collect();
            let unique = cache.len();
            (targets, unique)
        } else {
            (vec![None; raw_traces.len()], 0)
        };
    let lm_hidden_computed = lm_hidden_targets.iter().filter(|h| h.is_some()).count();
    let distill_precompute_ms = t_distill.elapsed().as_millis() as u64;
    let nrem_unique_traces = raw_traces.len();
    tracing::info!(
        lm_hidden_computed,
        lm_hidden_unique,
        distill_precompute_ms,
        "distillation targets precomputed"
    );

    // Export (episode, text) pairs for offline LoRA training (ADR 0033).
    // Appends one JSON line per text-bearing trace to the specified file.
    if let (Some(export_path), Some(dec)) = (training_data_path, decoder.as_ref()) {
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(export_path)
        {
            Ok(mut f) => {
                use std::io::Write;
                let mut exported = 0usize;
                for (ep, _, ids, _) in raw_traces.iter() {
                    let Some(ids) = ids else { continue };
                    let text = match dec.decode_ids(ids) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                    let text_escaped = text
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\r', "\\r");
                    let ep_json: String = ep
                        .iter()
                        .map(|v| format!("{v:.6}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    let _ = writeln!(f, "{{\"episode\":[{ep_json}],\"text\":\"{text_escaped}\"}}");
                    exported += 1;
                }
                tracing::info!(exported, ?export_path, "training data exported");
            }
            Err(e) => tracing::warn!(?e, "failed to open training data export file"),
        }
    }

    let mut traces: Vec<TraceSnapshot> = raw_traces
        .into_iter()
        .zip(lm_hidden_targets)
        .map(|((ep, reg, ids, sal), lm_h)| (ep, reg, ids, sal, lm_h))
        .collect();
    // Language-priority first; then within each group sort by salience descending
    // so the most novel (high L2 from warm) traces are replayed first and are
    // never skipped on early NREM exit.
    traces.sort_by(|(_, _, a_txt, a_sal, _), (_, _, b_txt, b_sal, _)| {
        let a_lang = a_txt.is_some() as u8;
        let b_lang = b_txt.is_some() as u8;
        b_lang.cmp(&a_lang).then(
            b_sal
                .partial_cmp(a_sal)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });

    // Language oversampling: prepend lang_oversample extra copies of language-tagged
    // traces before the full sorted list. With lang_oversample=9 the projection sees
    // each language trace 10× per NREM cycle without changing non-language replay.
    if lang_oversample > 0 {
        let lang_traces: Vec<_> = traces
            .iter()
            .filter(|(_, _, txt, _, _)| txt.is_some())
            .cloned()
            .collect();
        if !lang_traces.is_empty() {
            let extra: Vec<_> = lang_traces
                .iter()
                .cloned()
                .cycle()
                .take(lang_traces.len() * lang_oversample)
                .collect();
            let mut extended = extra;
            extended.extend(traces);
            traces = extended;
        }
    }

    let n_traces = traces.len();
    let mut loss_ewma = 0.0f32;
    const EWMA_ALPHA: f32 = 0.3;

    // Detach all parameters from burn's global AutodiffServer before the NREM loop.
    // Without this, backward() cost grows O(total_prior_optimizer_steps) across cycles
    // because the server accumulates stale graph entries keyed by ParamId. Refreshing
    // replaces every ParamId, freeing accumulated entries on the next backward() call.
    let t_refresh = Instant::now();
    perception.refresh_autodiff_state();
    tracing::info!(
        refresh_ms = t_refresh.elapsed().as_millis(),
        "NREM: autodiff state refreshed"
    );

    let t_nrem = Instant::now();

    let decoder_trainable_records = traces.iter().filter(|(_, _, t, _, _)| t.is_some()).count();
    let replay_epochs = replay_epochs.max(1);
    tracing::info!(decoder_trainable_records, replay_epochs, "NREM: decoder-trainable traces");

    // Effective batch size: fall back to 1 (sequential) when 0 is passed.
    let batch_size = replay_batch_size.max(1);

    // Per-phase timing accumulators (microseconds).
    let mut replay_us_total = 0u64;
    let mut wake_us_total = 0u64;
    let mut proj_us_total = 0u64;
    let mut n_batches = 0u64;

    // Replay epochs: each epoch is a full pass through the trace buffer.
    // Loss EWMA carries forward across epochs for continuous LR scheduling.
    let mut first_batch_ever = true;

    for epoch_idx in 0..replay_epochs {
        let mut batch_start = 0usize; // index into `traces` of the current batch start
        while batch_start < n_traces {
        let salient = nrem_salience_threshold > 0
            && pending_embs.load(Ordering::Relaxed) > nrem_salience_threshold
            && batch_start >= nrem_min_traces;
        if interrupt.load(Ordering::Relaxed) || salient {
            if salient {
                tracing::info!(
                    pending = pending_embs.load(Ordering::Relaxed),
                    threshold = nrem_salience_threshold,
                    "sleep interrupted: salience threshold exceeded"
                );
            }
            was_interrupted = true;
            nrem_exit_reason = "salience";
            break;
        }

        let batch_end = (batch_start + batch_size).min(n_traces);
        let batch = &traces[batch_start..batch_end];

        // Neuromodulated lr: use step index at batch start (cosine schedule is slow).
        // Salience = mean warm-recall L2 of the batch (sentinel f32::MAX excluded from mean).
        let prev_ewma = loss_ewma;
        let non_sentinel: Vec<f32> = batch
            .iter()
            .map(|(_, _, _, s, _)| *s)
            .filter(|&s| s < f32::MAX / 2.0)
            .collect();
        let batch_salience = if non_sentinel.is_empty() {
            f32::MAX
        } else {
            non_sentinel.iter().sum::<f32>() / non_sentinel.len() as f32
        };
        let lr = nrem_lr(batch_start, n_traces, loss_ewma, prev_ewma, batch_salience);
        perception.set_learning_rate(lr);

        // ── Fused SAE replay + output-head training (single backward) ───────
        let regionals: Vec<&[Vec<f32>; 5]> = batch.iter().map(|(_, r, _, _, _)| r).collect();
        let ep_slices: Vec<&[f32]> = batch.iter().map(|(ep, _, _, _, _)| ep.as_slice()).collect();
        let t_replay = Instant::now();
        match perception.fused_replay_wake_step_batch(&regionals, &ep_slices) {
            Ok((loss, mse)) => {
                loss_ewma = if first_batch_ever {
                    first_batch_ever = false;
                    loss
                } else {
                    EWMA_ALPHA * loss + (1.0 - EWMA_ALPHA) * loss_ewma
                };
                total_train_loss += loss * batch.len() as f32;
                traces_replayed += batch.len();
                total_output_mse += mse;
                output_mse_n += 1;
            }
            Err(e) => tracing::warn!(?e, "fused_replay_wake_step_batch error — skipping batch"),
        }
        let fused_us = t_replay.elapsed().as_micros() as u64;
        replay_us_total += fused_us / 2;
        wake_us_total += fused_us / 2;

        // ── Adaptive exit checks (10b) ────────────────────────────────────────
        if sleep_loss_target > 0.0 && batch_start > 0 && loss_ewma < sleep_loss_target {
            tracing::info!(
                loss_ewma,
                sleep_loss_target,
                "NREM: loss target met — early exit"
            );
            nrem_exit_reason = "loss_target_met";
            break;
        }
        if sleep_max_secs > 0 && sleep_start.elapsed().as_secs() >= sleep_max_secs {
            tracing::info!(
                elapsed_secs = sleep_start.elapsed().as_secs(),
                sleep_max_secs,
                "sleep: max time exceeded — interrupting"
            );
            was_interrupted = true;
            nrem_exit_reason = "time_budget";
            break;
        }

        // ── Projection distillation: LM hidden-state targets (Gap 6) ────────────
        let t_proj = Instant::now();
        if let Some(ref mut dec) = decoder {
            let distill_pairs: Vec<(&[f32], &[f32])> = batch
                .iter()
                .filter_map(|(ep, _, _, _, lm_h)| {
                    lm_h.as_ref().map(|h| (ep.as_slice(), h.as_slice()))
                })
                .collect();
            if !distill_pairs.is_empty() {
                match dec.projection_step_batch_distill(&distill_pairs) {
                    Ok(loss) => {
                        let cnt = distill_pairs.len();
                        total_proj_loss += loss * cnt as f32;
                        proj_loss_n += cnt;
                        if loss > proj_loss_max {
                            proj_loss_max = loss;
                        }
                    }
                    Err(e) => tracing::warn!(?e, "projection_step_batch_distill error"),
                }
            }
        }
        proj_us_total += t_proj.elapsed().as_micros() as u64;
        n_batches += 1;

        batch_start = batch_end;
        }
        if epoch_idx + 1 < replay_epochs {
            tracing::info!(epoch_idx = epoch_idx + 1, "NREM: starting epoch");
        }
    }

    let replay_ms_avg = replay_us_total.checked_div(n_batches).unwrap_or(0) / 1000;
    let wake_ms_avg = wake_us_total.checked_div(n_batches).unwrap_or(0) / 1000;
    let proj_ms_avg = proj_us_total.checked_div(n_batches).unwrap_or(0) / 1000;
    let total_ms = (replay_us_total + wake_us_total + proj_us_total) / 1000;
    if n_batches > 0 {
        tracing::info!(
            n_batches,
            replay_ms_avg,
            wake_ms_avg,
            proj_ms_avg,
            total_ms,
            "NREM phase timings"
        );
    }

    // Log projection weight norm after NREM for cross-session spike diagnostics.
    if let Some(ref dec) = decoder {
        match dec.proj_weight_norm() {
            Ok(n) => tracing::info!(
                proj_weight_norm = n,
                "NREM complete: projection weight norm"
            ),
            Err(e) => tracing::warn!(?e, "proj_weight_norm failed"),
        }
    }

    // Restore base lr so grow_region's optimizer reset starts from a known point.
    perception.set_learning_rate(LR_MAX);

    if !was_interrupted {
        // Reset checkpoint after a full cycle.
        episodic.replay_checkpoint = 0;
    }

    let mean_train_loss = if traces_replayed > 0 {
        total_train_loss / traces_replayed as f32
    } else {
        0.0
    };
    let mean_output_mse = if output_mse_n > 0 {
        total_output_mse / output_mse_n as f32
    } else {
        0.0
    };
    let mean_proj_loss = if proj_loss_n > 0 {
        Some(total_proj_loss / proj_loss_n as f32)
    } else {
        None
    };
    let proj_loss_max_opt = if proj_loss_n > 0 {
        Some(proj_loss_max)
    } else {
        None
    };

    let nrem_wall_ms = t_nrem.elapsed().as_millis() as u64;
    tracing::info!(
        traces_replayed,
        mean_train_loss,
        mean_output_mse,
        nrem_wall_ms,
        "NREM replay complete"
    );

    // ── Homeostatic downscaling ───────────────────────────────────────────────
    if !was_interrupted {
        perception.homeostatic_decay(DECAY_RATE);
    }

    // ── REM pre-spill: warm recall probe ─────────────────────────────────────
    let t_rem = Instant::now();
    // Run BEFORE spill so warm only contains prior-cycle entries. Querying after
    // spill would find exact matches (L2=0) from the just-inserted current records.
    let mut per_region_mse = [0.0f32; 5];
    let mut per_region_count = [0usize; 5];
    let mut regions_grown: Vec<RegionId> = Vec::new();
    let mut regions_pruned: Vec<RegionId> = Vec::new();
    let mut total_units_pruned: usize = 0;
    let mut warm_recall_l2: Option<f32> = None;
    let mut gist_recall_rate: Option<f32> = None;
    let mut cross_modal_gist_entropy: Option<f32> = None;
    let mut within_session_gist_rate: Option<f32> = None;

    type RemTrace = (Vec<f32>, [Vec<f32>; 5], IngestSource, f32);
    // Collect traces once for reuse in both warm probe and MSE measurement.
    let traces_for_rem: Vec<RemTrace> = if !was_interrupted {
        episodic
            .iter_traces()
            .map(|t| {
                (
                    t.record.values.clone(),
                    t.regional.clone(),
                    t.record.ingest_source,
                    t.record.salience,
                )
            })
            .collect()
    } else {
        vec![]
    };

    if !was_interrupted && pre_spill_warm_len > 0 {
        // ── Gist clustering (10c) ─────────────────────────────────────────────
        // Build k-means centroids from a sample of the warm store, then use them
        // during the warm probe to compute per-episode gist-recall.
        let gist_centroids: Vec<Vec<f32>> = if pre_spill_warm_len >= GIST_MIN_WARM {
            let sample = recall.warm.sample_values(GIST_SAMPLE);
            tracing::debug!(sample_n = sample.len(), k = GIST_K, "gist: running k-means");
            let t0 = Instant::now();
            let centroids = kmeans(&sample, GIST_K, GIST_MAX_ITER);
            tracing::debug!(
                elapsed_ms = t0.elapsed().as_millis(),
                "gist: k-means complete"
            );
            centroids
        } else {
            Vec::new()
        };
        let gist_available = !gist_centroids.is_empty();

        let mut total_l2 = 0.0f32;
        let mut n_probed = 0usize;
        let mut gist_matches = 0usize;
        // Per-cluster regional weight histogram for cross-modal entropy (Gap 8, ADR 0021).
        // Slots 0–4 map to the 5 perception regions; each episode distributes its
        // salience weight proportionally via regional_weights() (L2-norm attribution).
        let mut cluster_sources: Vec<[f32; 6]> = vec![[0.0f32; 6]; GIST_K];

        for (ep_vec, _, _src, salience) in &traces_for_rem {
            if let Some((dist, neighbor)) =
                recall.warm.query_with_dist(ep_vec, 1).into_iter().next()
            {
                total_l2 += dist;
                n_probed += 1;
                if gist_available {
                    let probe_cluster = nearest_centroid(&gist_centroids, ep_vec);
                    let neighbor_cluster = nearest_centroid(&gist_centroids, &neighbor.values);
                    if probe_cluster == neighbor_cluster {
                        gist_matches += 1;
                    }
                    // Salience weight: unknown (f32::MAX) → 1.0; otherwise clamp [0.01, 2.0].
                    let weight = if *salience >= f32::MAX / 2.0 {
                        1.0f32
                    } else {
                        salience.clamp(0.01, 2.0)
                    };
                    // Distribute weight proportionally across the 5 regional slots
                    // using L2 norms of the episode vector's per-region slices.
                    // This gives meaningful entropy even for Mixed-tagged episodes.
                    let rw = regional_weights(ep_vec);
                    for (slot, &rw_i) in rw.iter().enumerate() {
                        cluster_sources[probe_cluster][slot] += weight * rw_i;
                    }
                }
            }
        }

        // Extend the cluster histogram with a bounded warm sample so that
        // prior-session episodes (cold→warm) contribute to cross_modal_gist_entropy.
        if gist_available {
            for rec in recall.warm.sample_records(GIST_SAMPLE) {
                let probe_cluster = nearest_centroid(&gist_centroids, &rec.values);
                let weight = if rec.salience >= f32::MAX / 2.0 {
                    1.0f32
                } else {
                    rec.salience.clamp(0.01, 2.0)
                };
                let rw = regional_weights(&rec.values);
                for (slot, &rw_i) in rw.iter().enumerate() {
                    cluster_sources[probe_cluster][slot] += weight * rw_i;
                }
            }
        }

        if n_probed > 0 {
            let mean_l2 = total_l2 / n_probed as f32;
            warm_recall_l2 = Some(mean_l2);
            if gist_available {
                let rate = gist_matches as f32 / n_probed as f32;
                gist_recall_rate = Some(rate);

                // Shannon entropy per cluster (bits), averaged over occupied clusters.
                // Slots 0–4 map to the 5 perception regions; slot 5 is unused now that
                // the histogram uses proportional regional weights instead of IngestSource.
                // Counts are salience-weighted so novel episodes contribute more.
                let entropy_sum: f32 = cluster_sources
                    .iter()
                    .filter_map(|counts| {
                        let total: f32 = counts[0..5].iter().sum();
                        if total < 2.0 {
                            return None; // need ≥2 weight to compute entropy
                        }
                        let h: f32 = counts[0..5]
                            .iter()
                            .filter(|&&c| c > 0.0)
                            .map(|&c| {
                                let p = c / total;
                                // Guard p=1.0: log2(1.0)=0.0 exactly, so -p*log2(p)=-0.0
                                // in IEEE 754. Return 0.0 explicitly for the pure-cluster case.
                                if p >= 1.0 {
                                    0.0
                                } else {
                                    -p * p.log2()
                                }
                            })
                            .sum();
                        Some(h)
                    })
                    .sum();
                let n_occupied = cluster_sources
                    .iter()
                    .filter(|counts| counts[0..5].iter().sum::<f32>() >= 2.0)
                    .count();
                if n_occupied > 0 {
                    cross_modal_gist_entropy = Some((entropy_sum / n_occupied as f32).max(0.0));
                }

                tracing::info!(
                    mean_l2,
                    pre_spill_warm_len,
                    gist_recall_rate = rate,
                    gist_matches,
                    n_probed,
                    cross_modal_gist_entropy = ?cross_modal_gist_entropy,
                    "warm recall + gist probe"
                );
            } else {
                tracing::info!(mean_l2, pre_spill_warm_len, "warm recall probe");
            }
        }
    }

    // ── Within-session gist probe ─────────────────────────────────────────────
    // Re-runs gist with a within-session warm subset (timestamp_secs >= session_start_secs).
    // Prior-session embeddings are excluded because NREM weight updates cause embedding
    // drift, making cross-session nearest-neighbor matches semantically unreliable.
    // This is the clean signal for the growth controller (ADR 0024).
    if !was_interrupted {
        let ws_records = recall.warm.records_since(session_start_secs);
        if ws_records.len() >= GIST_MIN_WARM {
            let ws_vecs: Vec<Vec<f32>> = ws_records.iter().map(|r| r.values.clone()).collect();
            tracing::debug!(
                ws_warm_len = ws_records.len(),
                k = GIST_K,
                "within-session gist: running k-means"
            );
            let ws_centroids = kmeans(&ws_vecs, GIST_K, GIST_MAX_ITER);
            let mut ws_matches = 0usize;
            let mut ws_probed = 0usize;
            for (ep_vec, _, _, _) in &traces_for_rem {
                // Brute-force nearest within-session neighbor (no HNSW for this subset).
                let nearest = ws_records.iter().min_by(|a, b| {
                    let da: f32 = a
                        .values
                        .iter()
                        .zip(ep_vec.iter())
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum();
                    let db: f32 = b
                        .values
                        .iter()
                        .zip(ep_vec.iter())
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
                if let Some(neighbor) = nearest {
                    let probe_cluster = nearest_centroid(&ws_centroids, ep_vec);
                    let neighbor_cluster = nearest_centroid(&ws_centroids, &neighbor.values);
                    if probe_cluster == neighbor_cluster {
                        ws_matches += 1;
                    }
                    ws_probed += 1;
                }
            }
            if ws_probed > 0 {
                let rate = ws_matches as f32 / ws_probed as f32;
                within_session_gist_rate = Some(rate);
                tracing::info!(
                    ws_warm_len = ws_records.len(),
                    rate,
                    ws_matches,
                    ws_probed,
                    "within-session gist probe"
                );
            }
        }
    }

    // ── Compression: spill hot → cold; consolidate into warm ─────────────────
    rem_wall_ms += t_rem.elapsed().as_millis() as u64;
    let t_spill = Instant::now();
    if !was_interrupted {
        let mut spilled = 0usize;
        for record in episodic.iter_records() {
            if let Err(e) = recall.cold.append(record) {
                tracing::warn!(?e, "cold archive append error");
                break;
            }
            recall.warm.insert(record.clone());
            spilled += 1;
        }
        tracing::info!(spilled, "hot→cold spill + warm consolidation complete");

        // Warm HNSW size management: rebuild whenever cold has grown past warm.
        // The previous condition (warm.len() > max_warm_records) silently broke after the
        // first rebuild: SemanticStore.insert() drops records once max_records is reached,
        // so warm.len() never exceeds max_warm_records again and the index permanently
        // diverges from cold. cold.len() > warm.len() fires every cycle once the cap is
        // hit, keeping warm synchronised with the latest cold records.
        if max_warm_records > 0 && recall.cold.len() > recall.warm.len() {
            let t0 = Instant::now();
            let loaded = recall
                .warm
                .rebuild_from_recent(&recall.cold, max_warm_records);
            tracing::info!(
                loaded,
                max_warm_records,
                elapsed_ms = t0.elapsed().as_millis(),
                "warm HNSW rebuilt from recent cold records"
            );
        }

        // Cold archive size management: evict lowest-salience records (ADR 0020).
        if max_cold_records > 0 && recall.cold.len() > max_cold_records {
            let before = recall.cold.len();
            match recall.cold.evict_lowest_salience(max_cold_records) {
                Ok(()) => tracing::info!(
                    before,
                    after = max_cold_records,
                    "cold archive: evicted lowest-salience records"
                ),
                Err(e) => tracing::warn!(?e, "cold archive eviction failed"),
            }
        }
    }

    let spill_wall_ms = t_spill.elapsed().as_millis() as u64;
    tracing::info!(spill_wall_ms, "spill complete");

    // ── REM-like: measure MSE + trigger growth ────────────────────────────────
    let t_rem2 = Instant::now();
    if !was_interrupted {
        // Measure SAE reconstruction quality without updating weights.
        for (_, regional, _, _) in &traces_for_rem {
            match perception.measure_mse(regional) {
                Ok(mse) => {
                    for i in 0..5 {
                        if !regional[i].is_empty() {
                            per_region_mse[i] += mse[i];
                            per_region_count[i] += 1;
                        }
                    }
                }
                Err(e) => tracing::warn!(?e, "measure_mse error"),
            }
        }

        for i in 0..5 {
            if per_region_count[i] > 0 {
                per_region_mse[i] /= per_region_count[i] as f32;
            }
        }

        // Growth trigger: within-session gist is the primary signal (no cross-session drift).
        // Falls back to cross-session gist at c1 of each session (within_session always None
        // before the first spill). Last resort: SAE MSE when no warm existed at all (S1C1).
        // warm_recall_poor is logged but no longer gates growth (too noisy).
        //
        // Cooldown: suppresses growth for `growth_cooldown` cycles after any event.
        // c1 cross-session gist is still noisy; cooldown backstops spurious c1 growth.
        let _warm_recall_poor = warm_recall_l2.is_none_or(|l2| l2 > WARM_RECALL_GROWTH_THRESHOLD);
        let gist_poor = gist_recall_rate.is_none_or(|r| r < GIST_GROWTH_THRESHOLD);
        let growth_blocked = cooldown_remaining > 0;
        if growth_blocked {
            tracing::info!(
                cooldown_remaining,
                gist_recall_rate = ?gist_recall_rate,
                "growth suppressed by cooldown"
            );
        }
        let region_ids = [
            RegionId::AudioSpeech,
            RegionId::AudioEnv,
            RegionId::Visual,
            RegionId::Symbolic,
            RegionId::Language,
        ];
        if !growth_blocked {
            // Cascade-correlation init (ADR 0007, P2b): collect high-residual inputs once
            // before the growth loop so that any regions that trigger growth can use them.
            // Cap at 2 × GROWTH_BATCH seeds per region to bound memory.
            let cc_seeds =
                perception.collect_high_residual_inputs(&traces_for_rem, 2 * GROWTH_BATCH);

            for (i, &region) in region_ids.iter().enumerate() {
                // Quality signal: within-session gist primary (no drift); cross-session
                // gist at c1 of any session (ws always None before first spill); MSE
                // last resort when warm was empty (S1C1).
                let quality_poor = match (within_session_gist_rate, gist_recall_rate) {
                    (Some(ws), _) => ws < GIST_GROWTH_THRESHOLD,
                    (None, Some(_)) => gist_poor,
                    (None, None) => per_region_mse[i] > GROWTH_THRESHOLD,
                };
                if per_region_count[i] >= GROWTH_MIN_SAMPLES && quality_poor {
                    tracing::info!(
                        region = ?region,
                        mse = per_region_mse[i],
                        gist_recall_rate = ?gist_recall_rate,
                        seeds = cc_seeds[i].len(),
                        "growing region (cascade-correlation)"
                    );
                    perception.grow_region_seeded(region, GROWTH_BATCH, &cc_seeds[i]);
                    regions_grown.push(region);
                }
            }
        }

        tracing::info!(
            speech_mse = per_region_mse[0],
            env_mse = per_region_mse[1],
            visual_mse = per_region_mse[2],
            symbolic_mse = per_region_mse[3],
            language_mse = per_region_mse[4],
            warm_recall_l2 = ?warm_recall_l2,
            gist_recall_rate = ?gist_recall_rate,
            "REM self-probe complete"
        );
    }

    // ── REM pruning of grown units (12c) ─────────────────────────────────────
    // Prune only when gist recall is high (system not struggling) and prune_fraction > 0.
    if !was_interrupted && prune_fraction > 0.0 {
        // Prefer within-session gist for prune safety check; fall back to cross-session.
        // None at c1 → no pruning (conservative when quality can't be assessed).
        let prune_ok = within_session_gist_rate
            .or(gist_recall_rate)
            .is_some_and(|r| r >= PRUNE_GIST_THRESHOLD);
        if prune_ok {
            let region_ids = [
                RegionId::AudioSpeech,
                RegionId::AudioEnv,
                RegionId::Visual,
                RegionId::Symbolic,
                RegionId::Language,
            ];
            for (i, &region) in region_ids.iter().enumerate() {
                let n_grown = perception.region_sizes()[i].saturating_sub(N_UNITS);
                if n_grown >= prune_min_grown {
                    let n = perception.prune_region(region, prune_fraction);
                    if n > 0 {
                        regions_pruned.push(region);
                        total_units_pruned += n;
                    }
                }
            }
        }
        if !regions_pruned.is_empty() {
            tracing::info!(
                regions_pruned = ?regions_pruned,
                total_units_pruned,
                "REM pruning complete"
            );
        }
    }

    // ── REM creative recombination (11b) ─────────────────────────────────────
    // After NREM training completes, measure inference-mode output MSE on both
    // real and synthetic (blended) episodes to confirm synthetics are novel to
    // the trained output heads.
    let mut synthetics_produced = 0usize;
    // Collected during REM synthesis; pushed to episodic after clear so the
    // next NREM cycle trains on creative recombinations (item 1).
    let mut synth_traces: Vec<(Vec<f32>, [Vec<f32>; 5], f32)> = Vec::new();
    let mut mean_synthetic_output_mse: Option<f32> = None;
    let mut mean_real_inference_mse: Option<f32> = None;

    if !was_interrupted && !traces_for_rem.is_empty() {
        // Measure real inference MSE on a sub-sample of traces_for_rem.
        let real_sample_n = traces_for_rem.len().min(REM_SYNTH_MAX_N);
        let step = traces_for_rem.len().max(1);
        let mut real_mse_total = 0.0f32;
        let mut real_mse_n = 0usize;
        for j in 0..real_sample_n {
            let idx = j * step / real_sample_n;
            let (ep_vec, regional, _, _) = &traces_for_rem[idx];
            match perception.measure_output_mse_inference(ep_vec, regional) {
                Ok(mse) => {
                    real_mse_total += mse;
                    real_mse_n += 1;
                }
                Err(e) => tracing::warn!(?e, "measure_output_mse_inference error (real)"),
            }
        }
        if real_mse_n > 0 {
            mean_real_inference_mse = Some(real_mse_total / real_mse_n as f32);
        }

        // Synthesize pairs with L2 > REM_SYNTH_MIN_L2 (different content).
        let mut synth_mse_total = 0.0f32;
        let mut synth_mse_n = 0usize;
        'outer: for i in 0..traces_for_rem.len() {
            if synthetics_produced >= REM_SYNTH_MAX_N {
                break;
            }
            for j in (i + 1)..traces_for_rem.len() {
                if synthetics_produced >= REM_SYNTH_MAX_N {
                    break 'outer;
                }
                let (ep_a, reg_a, _, _) = &traces_for_rem[i];
                let (_, reg_b, _, _) = &traces_for_rem[j];
                let l2 = l2_sq(ep_a, &traces_for_rem[j].0).sqrt();
                if l2 < REM_SYNTH_MIN_L2 {
                    continue;
                }
                match perception.synthesize_episode(reg_a, reg_b, REM_SYNTH_ALPHA) {
                    Ok(synth_ep) => {
                        synthetics_produced += 1;
                        // Blend regional inputs with the same alpha used for synthesis so
                        // NREM can train on the synthetic's own encoder representation.
                        let synth_regional: [Vec<f32>; 5] = std::array::from_fn(|k| {
                            let a = &reg_a[k];
                            let b = &reg_b[k];
                            if a.is_empty() || b.is_empty() {
                                vec![]
                            } else {
                                a.iter()
                                    .zip(b.iter())
                                    .map(|(&x, &y)| {
                                        REM_SYNTH_ALPHA * x + (1.0 - REM_SYNTH_ALPHA) * y
                                    })
                                    .collect()
                            }
                        });
                        // Salience = source-pair L2 distance (how different the parents were).
                        synth_traces.push((synth_ep.clone(), synth_regional, l2));
                        // Measure novelty: MSE of both source regionals against the synthetic.
                        let mse_a = perception.measure_output_mse_inference(&synth_ep, reg_a);
                        let mse_b = perception.measure_output_mse_inference(&synth_ep, reg_b);
                        match (mse_a, mse_b) {
                            (Ok(a), Ok(b)) => {
                                synth_mse_total += (a + b) / 2.0;
                                synth_mse_n += 1;
                            }
                            (Ok(a), Err(_)) | (Err(_), Ok(a)) => {
                                synth_mse_total += a;
                                synth_mse_n += 1;
                            }
                            _ => {}
                        }
                    }
                    Err(e) => tracing::warn!(?e, "synthesize_episode error"),
                }
            }
        }
        if synth_mse_n > 0 {
            mean_synthetic_output_mse = Some(synth_mse_total / synth_mse_n as f32);
        }

        tracing::info!(
            synthetics_produced,
            mean_real_inference_mse = ?mean_real_inference_mse,
            mean_synthetic_output_mse = ?mean_synthetic_output_mse,
            "REM creative recombination complete"
        );
    }

    // ── Clear hot buffer after spill + REM ───────────────────────────────────
    if !was_interrupted {
        episodic.clear();
        tracing::info!("hot buffer cleared after spill");

        // Seed REM synthetic traces into the hot buffer so the next NREM cycle
        // trains on creative recombinations. L2-normalise for cosine-equivalent
        // recall (matching how real episodes are stored by record_episode).
        if !synth_traces.is_empty() {
            let synth_ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            let n_synth = synth_traces.len();
            for (ep, regional, salience) in synth_traces {
                let mag = ep.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                let ep_norm: Vec<f32> = ep.iter().map(|x| x / mag).collect();
                episodic.push(synth_ts, ep_norm, regional, salience, IngestSource::Mixed);
            }
            tracing::info!(n_synth, "REM synthetics seeded into episodic buffer");
        }
    }

    rem_wall_ms += t_rem2.elapsed().as_millis() as u64;
    tracing::info!(rem_wall_ms, "REM complete");

    // ── Snapshot ──────────────────────────────────────────────────────────────
    let t_snapshot = Instant::now();
    if !was_interrupted {
        if let Err(e) = save_snapshot(episodic, memory_dir) {
            tracing::warn!(?e, "episodic snapshot write failed");
        }
        // Persist warm HNSW index so restart avoids cold-rebuild (~3 min at 100k records).
        let warm_dir = memory_dir.join("warm");
        if let Err(e) = recall.warm.dump(&warm_dir) {
            tracing::warn!(?e, "warm HNSW dump failed — next restart will cold-rebuild");
        }
        // Perception weights (BinFileRecorder → weights.bin) written to snapshot dir.
        let w_path = weights_path(memory_dir);
        if let Err(e) = perception.save_weights(&w_path) {
            tracing::warn!(?e, "weight snapshot write failed");
        }
        // Region sizes written so restore can pre-grow before loading weights.
        if let Err(e) = save_region_sizes(perception.region_sizes(), memory_dir) {
            tracing::warn!(?e, "region sizes write failed");
        }
        // Growth cooldown counter: reset to growth_cooldown when something grew,
        // otherwise decrement toward 0.
        let new_cooldown = if !regions_grown.is_empty() {
            growth_cooldown
        } else {
            cooldown_remaining.saturating_sub(1)
        };
        if let Err(e) = save_growth_cooldown_remaining(new_cooldown, memory_dir) {
            tracing::warn!(?e, "growth cooldown write failed");
        }
        tracing::info!("snapshot written");
    }
    let snapshot_wall_ms = t_snapshot.elapsed().as_millis() as u64;
    let growth_cooldown_remaining = if !regions_grown.is_empty() {
        growth_cooldown
    } else {
        cooldown_remaining.saturating_sub(1)
    };
    tracing::info!(
        nrem_wall_ms,
        distill_precompute_ms,
        rem_wall_ms,
        spill_wall_ms,
        snapshot_wall_ms,
        total_wall_ms = sleep_start.elapsed().as_millis() as u64,
        "sleep cycle phase timings"
    );

    Ok(SleepStats {
        traces_replayed,
        mean_train_loss,
        mean_output_mse,
        per_region_mse,
        regions_grown,
        was_interrupted,
        nrem_exit_reason,
        warm_recall_l2,
        gist_recall_rate,
        cross_modal_gist_entropy,
        within_session_gist_rate,
        mean_proj_loss,
        proj_loss_max: proj_loss_max_opt,
        decoder_trainable_records,
        synthetics_produced,
        mean_synthetic_output_mse,
        mean_real_inference_mse,
        regions_pruned,
        total_units_pruned,
        nrem_wall_ms,
        distill_precompute_ms,
        rem_wall_ms,
        spill_wall_ms,
        snapshot_wall_ms,
        lm_hidden_computed,
        lm_hidden_unique,
        nrem_budget_traces: n_traces,
        nrem_unique_traces,
        growth_cooldown_remaining,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── nrem_lr schedule tests ─────────────────────────────────────────────────

    #[test]
    fn nrem_lr_cosine_decreases_step0_to_last() {
        let lr_first = nrem_lr(0, 20, 0.0, 0.0, f32::MAX);
        let lr_last = nrem_lr(19, 20, 0.0, 0.0, f32::MAX);
        assert!(
            lr_first > lr_last,
            "cosine LR should be highest at step 0 (got {lr_first:.2e}) and lowest at last step (got {lr_last:.2e})"
        );
    }

    #[test]
    fn nrem_lr_stagnant_gate_reduces_lr() {
        // worsening: loss_ewma > prev_ewma by >1% → trend_scale = LR_STAGNANT_SCALE
        // plateau:   |improvement| ≤ 1% → trend_scale = 0.5
        // Both use step=5 (not 0) so the trend gate is active.
        let plateau = nrem_lr(5, 20, 1.0, 1.0, 0.0); // improvement = 0
        let worsening = nrem_lr(5, 20, 1.05, 1.0, 0.0); // improvement = -0.05
        assert!(
            worsening < plateau,
            "worsening loss should reduce LR below plateau (worsening={worsening:.2e}, plateau={plateau:.2e})"
        );
        // Worsening should apply the LR_STAGNANT_SCALE=0.25 multiplier vs plateau's 0.5
        let ratio = worsening / plateau;
        assert!(
            (ratio - LR_STAGNANT_SCALE / 0.5).abs() < 0.05,
            "worsening/plateau ratio should be ~{:.2} (got {ratio:.3})",
            LR_STAGNANT_SCALE / 0.5
        );
    }

    #[test]
    fn nrem_lr_salience_boost_scales_lr() {
        // step=0: prev_ewma check is bypassed → trend_scale=1.0 for both
        let no_salience = nrem_lr(0, 20, 0.0, 0.0, 0.0); // norm=0 → scale=1.0
        let max_salience = nrem_lr(0, 20, 0.0, 0.0, LR_SALIENCE_CAP); // norm=1 → scale=LR_SALIENCE_MAX_SCALE
        let expected_ratio = LR_SALIENCE_MAX_SCALE;
        let actual_ratio = max_salience / no_salience;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "max-salience LR should be {expected_ratio}× no-salience (got {actual_ratio:.3})"
        );
    }

    #[test]
    fn nrem_lr_sentinel_matches_half_salience() {
        // f32::MAX sentinel maps to 0.5 novelty; LR_SALIENCE_CAP/2 also maps to 0.5 novelty.
        let sentinel = nrem_lr(0, 20, 0.0, 0.0, f32::MAX);
        let half_cap = nrem_lr(0, 20, 0.0, 0.0, LR_SALIENCE_CAP * 0.5);
        let rel_diff = (sentinel - half_cap).abs() / sentinel.max(1e-12);
        assert!(
            rel_diff < 1e-6,
            "sentinel and 0.5×cap should give identical LR (diff={rel_diff:.2e})"
        );
    }

    // ── kmeans tests ───────────────────────────────────────────────────────────

    #[test]
    fn kmeans_separates_two_tight_clusters() {
        // Two clusters far apart in 1D.
        let mut points: Vec<Vec<f32>> = (0..20).map(|_| vec![0.1f32]).collect();
        points.extend((0..20).map(|_| vec![9.9f32]));
        let centroids = kmeans(&points, 2, 20);
        assert_eq!(centroids.len(), 2);
        let lo = centroids[0][0].min(centroids[1][0]);
        let hi = centroids[0][0].max(centroids[1][0]);
        assert!(lo < 1.0, "lower centroid should be near 0, got {lo}");
        assert!(hi > 9.0, "upper centroid should be near 10, got {hi}");
    }

    #[test]
    fn kmeans_handles_fewer_points_than_k() {
        let points = vec![vec![1.0f32], vec![2.0f32]];
        let centroids = kmeans(&points, 5, 10); // k > n
        assert_eq!(centroids.len(), 2, "k is clamped to point count");
    }

    #[test]
    fn kmeans_empty_returns_empty() {
        let centroids = kmeans(&[], 4, 10);
        assert!(centroids.is_empty());
    }
}
