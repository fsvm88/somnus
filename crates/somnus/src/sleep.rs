/// Sleep cycle: NREM-like consolidation + REM-like self-probe + growth.
///
/// Blocks the main loop. Checks `interrupt` flag after every replay step;
/// exits early if set (progress is preserved via `episodic.replay_checkpoint`).
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
use somnus_perception::{PerceptionProcessor, RegionId};

use crate::action::ActionDecoder;

const DECAY_RATE: f32 = 1e-4;
const GROWTH_THRESHOLD: f32 = 0.1;
const GROWTH_MIN_SAMPLES: usize = 16;
const GROWTH_BATCH: usize = 64;
/// Pending-embedding count above which sleep is considered interruptible by external salience.
const SALIENCE_THRESHOLD: usize = 16;
/// Mean warm-recall L2 above this threshold supplements the MSE growth signal.
/// Only applied when warm had pre-spill entries (i.e., cross-cycle/cross-session data).
/// Calibrated 2026-05-14: same-content (single looped video) warm-recall L2 ranges 1.6–3.5
/// (median 2.3, max 3.5). Threshold of 15.0 gives ~4× margin above that maximum, so same-content
/// cycling never triggers growth while genuinely novel input (expected L2 >> 5) will.
const WARM_RECALL_GROWTH_THRESHOLD: f32 = 15.0;

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

fn nrem_lr(step: usize, total: usize, loss_ewma: f32, prev_ewma: f32) -> f64 {
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

    cosine * trend_scale
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
    /// Mean surrogate loss from projection training steps (ADR 0012).
    /// None when no text-target traces were present.
    pub mean_proj_loss: Option<f32>,
    /// Max per-step projection loss seen during NREM (useful for diagnosing spikes).
    pub proj_loss_max: Option<f32>,
    /// Number of traces that had a text_target and were used to train the projection.
    pub decoder_trainable_records: usize,
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
    type TraceSnapshot = (Vec<f32>, [Vec<f32>; 5], Option<Vec<u32>>);
    let traces: Vec<TraceSnapshot> = episodic
        .iter_traces()
        .skip(episodic.replay_checkpoint)
        .map(|t| {
            (
                t.record.values.clone(),
                t.regional.clone(),
                t.text_target.clone(),
            )
        })
        .collect();

    let n_traces = traces.len();
    let mut loss_ewma = 0.0f32;
    const EWMA_ALPHA: f32 = 0.3;

    let decoder_trainable_records = traces.iter().filter(|(_, _, t)| t.is_some()).count();
    tracing::info!(decoder_trainable_records, "NREM: decoder-trainable traces");

    for (step, (episode_vec, regional, text_target)) in traces.iter().enumerate() {
        let salient = pending_embs.load(Ordering::Relaxed) > SALIENCE_THRESHOLD;
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

        // Neuromodulated lr: cosine schedule × loss-trend gate
        let prev_ewma = loss_ewma;
        let lr = nrem_lr(step, n_traces, loss_ewma, prev_ewma);
        perception.set_learning_rate(lr);

        match perception.replay_step(regional) {
            Ok(loss) => {
                loss_ewma = if step == 0 {
                    loss
                } else {
                    EWMA_ALPHA * loss + (1.0 - EWMA_ALPHA) * loss_ewma
                };
                total_train_loss += loss;
                traces_replayed += 1;
            }
            Err(e) => tracing::warn!(?e, "replay_step error — skipping trace"),
        }
        // Always advance checkpoint, even on error, to avoid replaying a broken trace forever.
        episodic.replay_checkpoint += 1;

        // Also train output heads during NREM using the stored episode vector (ADR 0010).
        // The stored episode_vec is a constant (untracked) so gradients flow only through
        // the output head weights, not the binding area or k-WTA layers.
        match perception.wake_reconstruct_step(episode_vec, regional) {
            Ok(mse) => {
                total_output_mse += mse;
                output_mse_n += 1;
            }
            Err(e) => tracing::warn!(?e, "output head replay error"),
        }

        // ── Adaptive exit checks (10b) ────────────────────────────────────────
        // Loss-target: NREM exits cleanly; REM + spill + snapshot still run.
        if sleep_loss_target > 0.0 && step > 0 && loss_ewma < sleep_loss_target {
            tracing::info!(
                loss_ewma,
                sleep_loss_target,
                "NREM: loss target met — early exit"
            );
            nrem_exit_reason = "loss_target_met";
            // Advance checkpoint so the remaining traces are skipped on restore.
            episodic.replay_checkpoint = episodic.len();
            break;
        }
        // Time budget: treat as interrupted (no REM/spill/snapshot).
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

        // Projection training: surrogate MSE loss for traces that carry a text target (ADR 0012).
        if let Some(ref mut dec) = decoder {
            if let Some(ids) = text_target {
                match dec.projection_step(episode_vec, ids) {
                    Ok(loss) => {
                        total_proj_loss += loss;
                        proj_loss_n += 1;
                        if loss > proj_loss_max {
                            proj_loss_max = loss;
                        }
                    }
                    Err(e) => tracing::warn!(?e, "projection_step error"),
                }
            }
        }
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
        if let Err(e) = perception.homeostatic_decay(DECAY_RATE) {
            tracing::warn!(?e, "homeostatic decay error");
        }
    }

    // ── REM pre-spill: warm recall probe ─────────────────────────────────────
    // Run BEFORE spill so warm only contains prior-cycle entries. Querying after
    // spill would find exact matches (L2=0) from the just-inserted current records.
    let mut per_region_mse = [0.0f32; 5];
    let mut per_region_count = [0usize; 5];
    let mut regions_grown: Vec<RegionId> = Vec::new();
    let mut warm_recall_l2: Option<f32> = None;

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
        let mut total_l2 = 0.0f32;
        let mut n_probed = 0usize;
        for (ep_vec, _) in &traces_for_rem {
            if let Some(&(dist, _)) = recall.warm.query_with_dist(ep_vec, 1).first() {
                total_l2 += dist;
                n_probed += 1;
            }
        }
        if n_probed > 0 {
            let mean_l2 = total_l2 / n_probed as f32;
            warm_recall_l2 = Some(mean_l2);
            tracing::info!(mean_l2, pre_spill_warm_len, "warm recall probe");
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

        // Growth: SAE MSE above threshold with sufficient samples.
        // When cross-cycle warm data is available, also require poor warm recall
        // to avoid growing when the model already recognises these patterns.
        let warm_recall_poor = warm_recall_l2.is_none_or(|l2| l2 > WARM_RECALL_GROWTH_THRESHOLD);
        let region_ids = [
            RegionId::AudioSpeech,
            RegionId::AudioEnv,
            RegionId::Visual,
            RegionId::Symbolic,
            RegionId::Language,
        ];
        for (i, &region) in region_ids.iter().enumerate() {
            if per_region_count[i] >= GROWTH_MIN_SAMPLES
                && per_region_mse[i] > GROWTH_THRESHOLD
                && warm_recall_poor
            {
                tracing::info!(
                    region = ?region,
                    mse = per_region_mse[i],
                    "growing region"
                );
                match perception.grow_region(region, GROWTH_BATCH) {
                    Ok(()) => regions_grown.push(region),
                    Err(e) => tracing::warn!(?e, region = ?region, "grow_region error"),
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
            "REM self-probe complete"
        );
    }

    // ── Clear hot buffer after spill + REM ───────────────────────────────────
    if !was_interrupted {
        episodic.clear();
        tracing::info!("hot buffer cleared after spill");
    }

    // ── Snapshot ──────────────────────────────────────────────────────────────
    if !was_interrupted {
        if let Err(e) = save_snapshot(episodic, memory_dir) {
            tracing::warn!(?e, "episodic snapshot write failed");
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
        mean_proj_loss,
        proj_loss_max: proj_loss_max_opt,
        decoder_trainable_records,
    })
}
