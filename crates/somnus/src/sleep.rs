/// Sleep cycle: NREM-like consolidation + REM-like self-probe + growth.
///
/// Blocks the main loop. Checks `interrupt` flag after every replay step;
/// exits early if set. Language-tagged traces are replayed first each cycle
/// so they are never skipped even when the salience gate fires early.
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use anyhow::Result;
use somnus_memory::{
    save_region_sizes, save_snapshot, weights_path, EpisodicBuffer, RecollectionLayer,
};
use somnus_perception::{PerceptionProcessor, RegionId, N_UNITS};

use crate::action::ActionDecoder;

const DECAY_RATE: f32 = 1e-4;
const GROWTH_THRESHOLD: f32 = 0.1;
const GROWTH_MIN_SAMPLES: usize = 16;
const GROWTH_BATCH: usize = 64;
/// Pending-embedding count above which sleep is considered interruptible by external salience.
const SALIENCE_THRESHOLD: usize = 16;
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
const LR_SALIENCE_CAP: f32 = 10.0;
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
    /// Regions that had grown units pruned during REM (12c).
    pub regions_pruned: Vec<RegionId>,
    /// Total grown units removed across all pruned regions.
    pub total_units_pruned: usize,
}

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
const REM_SYNTH_MIN_L2: f32 = 2.0;
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
    lang_oversample: usize,
    max_warm_records: usize,
) -> Result<SleepStats> {
    let sleep_start = Instant::now();
    tracing::info!(episodic_len = episodic.len(), "sleep cycle started");

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
    type TraceSnapshot = (Vec<f32>, [Vec<f32>; 5], Option<Vec<u32>>, f32);
    let mut traces: Vec<TraceSnapshot> = episodic
        .iter_traces()
        .map(|t| {
            (
                t.record.values.clone(),
                t.regional.clone(),
                t.text_target.clone(),
                t.salience,
            )
        })
        .collect();
    // Language-priority first; then within each group sort by salience descending
    // so the most novel (high L2 from warm) traces are replayed first and are
    // never skipped on early NREM exit.
    traces.sort_by(|(_, _, a_txt, a_sal), (_, _, b_txt, b_sal)| {
        let a_lang = a_txt.is_some() as u8;
        let b_lang = b_txt.is_some() as u8;
        b_lang.cmp(&a_lang).then(b_sal.partial_cmp(a_sal).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Language oversampling: prepend lang_oversample extra copies of language-tagged
    // traces before the full sorted list. With lang_oversample=9 the projection sees
    // each language trace 10× per NREM cycle without changing non-language replay.
    if lang_oversample > 0 {
        let lang_traces: Vec<_> = traces
            .iter()
            .filter(|(_, _, txt, _)| txt.is_some())
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

    let decoder_trainable_records = traces.iter().filter(|(_, _, t, _)| t.is_some()).count();
    tracing::info!(decoder_trainable_records, "NREM: decoder-trainable traces");

    // Effective batch size: fall back to 1 (sequential) when 0 is passed.
    let batch_size = replay_batch_size.max(1);

    // Per-phase timing accumulators (microseconds).
    let mut replay_us_total = 0u64;
    let mut wake_us_total = 0u64;
    let mut proj_us_total = 0u64;
    let mut n_batches = 0u64;

    let mut batch_start = 0usize; // index into `traces` of the current batch start
    while batch_start < n_traces {
        let salient = pending_embs.load(Ordering::Relaxed) > SALIENCE_THRESHOLD
            && batch_start >= nrem_min_traces;
        if interrupt.load(Ordering::Relaxed) || salient {
            if salient {
                tracing::info!(
                    pending = pending_embs.load(Ordering::Relaxed),
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
            .map(|(_, _, _, s)| *s)
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
        let regionals: Vec<&[Vec<f32>; 5]> = batch.iter().map(|(_, r, _, _)| r).collect();
        let ep_slices: Vec<&[f32]> = batch.iter().map(|(ep, _, _, _)| ep.as_slice()).collect();
        let t_replay = Instant::now();
        match perception.fused_replay_wake_step_batch(&regionals, &ep_slices) {
            Ok((loss, mse)) => {
                loss_ewma = if batch_start == 0 {
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

        // ── Projection training: batched (separate optimizer, ADR 0012) ────────
        let t_proj = Instant::now();
        if let Some(ref mut dec) = decoder {
            let proj_pairs: Vec<(&[f32], &[u32])> = batch
                .iter()
                .filter_map(|(ep, _, ids, _)| ids.as_ref().map(|ids| (ep.as_slice(), ids.as_slice())))
                .collect();
            if !proj_pairs.is_empty() {
                match dec.projection_step_batch(&proj_pairs) {
                    Ok(loss) => {
                        let cnt = proj_pairs.len();
                        total_proj_loss += loss * cnt as f32;
                        proj_loss_n += cnt;
                        if loss > proj_loss_max {
                            proj_loss_max = loss;
                        }
                    }
                    Err(e) => tracing::warn!(?e, "projection_step_batch error"),
                }
            }
        }
        proj_us_total += t_proj.elapsed().as_micros() as u64;
        n_batches += 1;

        batch_start = batch_end;
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

    tracing::info!(
        traces_replayed,
        mean_train_loss,
        mean_output_mse,
        "NREM replay complete"
    );

    // ── Homeostatic downscaling ───────────────────────────────────────────────
    if !was_interrupted {
        perception.homeostatic_decay(DECAY_RATE);
    }

    // ── REM pre-spill: warm recall probe ─────────────────────────────────────
    // Run BEFORE spill so warm only contains prior-cycle entries. Querying after
    // spill would find exact matches (L2=0) from the just-inserted current records.
    let mut per_region_mse = [0.0f32; 5];
    let mut per_region_count = [0usize; 5];
    let mut regions_grown: Vec<RegionId> = Vec::new();
    let mut regions_pruned: Vec<RegionId> = Vec::new();
    let mut total_units_pruned: usize = 0;
    let mut warm_recall_l2: Option<f32> = None;
    let mut gist_recall_rate: Option<f32> = None;

    // Collect traces once for reuse in both warm probe and MSE measurement.
    let traces_for_rem: Vec<(Vec<f32>, [Vec<f32>; 5])> = if !was_interrupted {
        episodic
            .iter_traces()
            .map(|t| (t.record.values.clone(), t.regional.clone()))
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

        for (ep_vec, _) in &traces_for_rem {
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
                }
            }
        }

        if n_probed > 0 {
            let mean_l2 = total_l2 / n_probed as f32;
            warm_recall_l2 = Some(mean_l2);
            if gist_available {
                let rate = gist_matches as f32 / n_probed as f32;
                gist_recall_rate = Some(rate);
                tracing::info!(
                    mean_l2,
                    pre_spill_warm_len,
                    gist_recall_rate = rate,
                    gist_matches,
                    n_probed,
                    "warm recall + gist probe"
                );
            } else {
                tracing::info!(mean_l2, pre_spill_warm_len, "warm recall probe");
            }
        }
    }

    // ── Compression: spill hot → cold; consolidate into warm ─────────────────
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

        // Warm HNSW size management: if warm has exceeded the cap, rebuild from the
        // most recent max_warm_records entries in cold so the index stays bounded.
        if max_warm_records > 0 && recall.warm.len() > max_warm_records {
            let t0 = Instant::now();
            let loaded = recall.warm.rebuild_from_recent(&recall.cold, max_warm_records);
            tracing::info!(
                loaded,
                max_warm_records,
                elapsed_ms = t0.elapsed().as_millis(),
                "warm HNSW rebuilt from recent cold records"
            );
        }
    }

    // ── REM-like: measure MSE + trigger growth ────────────────────────────────
    if !was_interrupted {
        // Measure SAE reconstruction quality without updating weights.
        for (_, regional) in &traces_for_rem {
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

        // Growth trigger: gist-recall is the primary signal; SAE MSE fallback when warm
        // is too small to cluster. warm_recall_poor is logged but no longer gates growth —
        // the WARM_RECALL_GROWTH_THRESHOLD (15.0) was calibrated too conservatively and
        // prevented growth even when gist was poor.
        let _warm_recall_poor = warm_recall_l2.is_none_or(|l2| l2 > WARM_RECALL_GROWTH_THRESHOLD);
        let gist_poor = gist_recall_rate.is_none_or(|r| r < GIST_GROWTH_THRESHOLD);
        let region_ids = [
            RegionId::AudioSpeech,
            RegionId::AudioEnv,
            RegionId::Visual,
            RegionId::Symbolic,
            RegionId::Language,
        ];
        for (i, &region) in region_ids.iter().enumerate() {
            // Quality signal: gist-recall when available, SAE MSE otherwise.
            let quality_poor = if gist_recall_rate.is_some() {
                gist_poor
            } else {
                per_region_mse[i] > GROWTH_THRESHOLD
            };
            if per_region_count[i] >= GROWTH_MIN_SAMPLES && quality_poor {
                tracing::info!(
                    region = ?region,
                    mse = per_region_mse[i],
                    gist_recall_rate = ?gist_recall_rate,
                    "growing region"
                );
                perception.grow_region(region, GROWTH_BATCH);
                regions_grown.push(region);
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
        let prune_ok = gist_recall_rate.is_some_and(|r| r >= PRUNE_GIST_THRESHOLD);
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
            let (ep_vec, regional) = &traces_for_rem[idx];
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
                let (ep_a, reg_a) = &traces_for_rem[i];
                let (_, reg_b) = &traces_for_rem[j];
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
                                    .map(|(&x, &y)| REM_SYNTH_ALPHA * x + (1.0 - REM_SYNTH_ALPHA) * y)
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
                episodic.push(synth_ts, ep_norm, regional, salience);
            }
            tracing::info!(n_synth, "REM synthetics seeded into episodic buffer");
        }
    }

    // ── Snapshot ──────────────────────────────────────────────────────────────
    if !was_interrupted {
        if let Err(e) = save_snapshot(episodic, memory_dir) {
            tracing::warn!(?e, "episodic snapshot write failed");
        }
        // Persist warm HNSW index so restart avoids cold-rebuild (~3 min at 100k records).
        let warm_dir = memory_dir.join("warm");
        if let Err(e) = recall.warm.dump(&warm_dir) {
            tracing::warn!(?e, "warm HNSW dump failed — next restart will cold-rebuild");
        }
        // Perception weights (safetensors) written to snapshot dir.
        let w_path = weights_path(memory_dir);
        if let Err(e) = perception.save_weights(&w_path) {
            tracing::warn!(?e, "weight snapshot write failed");
        }
        // Region sizes written so restore can pre-grow before loading weights.
        if let Err(e) = save_region_sizes(perception.region_sizes(), memory_dir) {
            tracing::warn!(?e, "region sizes write failed");
        }
        tracing::info!("snapshot written");
    }

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
        mean_proj_loss,
        proj_loss_max: proj_loss_max_opt,
        decoder_trainable_records,
        synthetics_produced,
        mean_synthetic_output_mse,
        mean_real_inference_mse,
        regions_pruned,
        total_units_pruned,
    })
}
