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
};

use anyhow::Result;
use somnus_memory::{save_region_sizes, save_snapshot, weights_path, EpisodicBuffer, RecollectionLayer};
use somnus_perception::{PerceptionProcessor, RegionId};

const DECAY_RATE: f32 = 1e-4;
const GROWTH_THRESHOLD: f32 = 0.1;
const GROWTH_MIN_SAMPLES: usize = 16;
const GROWTH_BATCH: usize = 64;
/// Pending-embedding count above which sleep is considered interruptible by external salience.
const SALIENCE_THRESHOLD: usize = 16;

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
    let t = if total > 1 { step as f64 / (total - 1) as f64 } else { 1.0 };
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
    pub per_region_mse: [f32; 4],
    pub regions_grown: Vec<RegionId>,
    pub was_interrupted: bool,
}

/// `pending_embs` counts embeddings produced by encoder tasks but not yet consumed
/// by the main loop. A high count means the world is active while the system sleeps;
/// once it exceeds `SALIENCE_THRESHOLD` the current replay step is the last one.
pub fn run_sleep_cycle(
    perception: &mut PerceptionProcessor,
    episodic: &mut EpisodicBuffer,
    recall: &mut RecollectionLayer,
    memory_dir: &Path,
    interrupt: Arc<AtomicBool>,
    pending_embs: Arc<AtomicUsize>,
) -> Result<SleepStats> {
    tracing::info!(episodic_len = episodic.len(), "sleep cycle started");

    let mut traces_replayed = 0usize;
    let mut total_train_loss = 0.0f32;
    let mut was_interrupted = false;

    // ── NREM-like: replay + train ─────────────────────────────────────────────
    // Collect traces into a snapshot so we don't borrow episodic mutably during training.
    let traces: Vec<_> = episodic
        .iter_traces()
        .skip(episodic.replay_checkpoint)
        .map(|t| t.regional.clone())
        .collect();

    let n_traces = traces.len();
    let mut loss_ewma = 0.0f32;
    const EWMA_ALPHA: f32 = 0.3;

    for (step, regional) in traces.iter().enumerate() {
        let salient = pending_embs.load(Ordering::Relaxed) > SALIENCE_THRESHOLD;
        if interrupt.load(Ordering::Relaxed) || salient {
            if salient {
                tracing::info!(
                    pending = pending_embs.load(Ordering::Relaxed),
                    "sleep interrupted: salience threshold exceeded"
                );
            }
            was_interrupted = true;
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
                episodic.replay_checkpoint += 1;
            }
            Err(e) => tracing::warn!(?e, "replay_step error — skipping trace"),
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

    tracing::info!(traces_replayed, mean_train_loss, "NREM replay complete");

    // ── Homeostatic downscaling ───────────────────────────────────────────────
    if !was_interrupted {
        if let Err(e) = perception.homeostatic_decay(DECAY_RATE) {
            tracing::warn!(?e, "homeostatic decay error");
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

    // ── REM-like: measure MSE, trigger growth ─────────────────────────────────
    let mut per_region_mse = [0.0f32; 4];
    let mut per_region_count = [0usize; 4];
    let mut regions_grown: Vec<RegionId> = Vec::new();

    if !was_interrupted {
        // Measure reconstruction quality without updating weights.
        let traces_for_rem: Vec<_> = episodic.iter_traces().map(|t| t.regional.clone()).collect();
        for regional in &traces_for_rem {
            match perception.measure_mse(regional) {
                Ok(mse) => {
                    for i in 0..4 {
                        if !regional[i].is_empty() {
                            per_region_mse[i] += mse[i];
                            per_region_count[i] += 1;
                        }
                    }
                }
                Err(e) => tracing::warn!(?e, "measure_mse error"),
            }
        }

        for i in 0..4 {
            if per_region_count[i] > 0 {
                per_region_mse[i] /= per_region_count[i] as f32;
            }
        }

        // Growth: flagged regions exceed threshold with sufficient samples.
        let region_ids = [
            RegionId::AudioSpeech,
            RegionId::AudioEnv,
            RegionId::Visual,
            RegionId::Symbolic,
        ];
        for (i, &region) in region_ids.iter().enumerate() {
            if per_region_count[i] >= GROWTH_MIN_SAMPLES && per_region_mse[i] > GROWTH_THRESHOLD {
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
        per_region_mse,
        regions_grown,
        was_interrupted,
    })
}
