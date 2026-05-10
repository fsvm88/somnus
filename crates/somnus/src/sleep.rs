/// Sleep cycle: NREM-like consolidation + REM-like self-probe + growth.
///
/// Blocks the main loop. Checks `interrupt` flag after every replay step;
/// exits early if set (progress is preserved via `episodic.replay_checkpoint`).
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
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

#[allow(dead_code)]
pub struct SleepStats {
    pub traces_replayed: usize,
    pub mean_train_loss: f32,
    pub per_region_mse: [f32; 4],
    pub regions_grown: Vec<RegionId>,
    pub was_interrupted: bool,
}

pub fn run_sleep_cycle(
    perception: &mut PerceptionProcessor,
    episodic: &mut EpisodicBuffer,
    recall: &mut RecollectionLayer,
    memory_dir: &Path,
    interrupt: Arc<AtomicBool>,
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

    for regional in &traces {
        if interrupt.load(Ordering::Relaxed) {
            was_interrupted = true;
            break;
        }
        match perception.replay_step(regional) {
            Ok(loss) => {
                total_train_loss += loss;
                traces_replayed += 1;
                episodic.replay_checkpoint += 1;
            }
            Err(e) => tracing::warn!(?e, "replay_step error — skipping trace"),
        }
    }

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

    // ── Compression: spill hot → cold ────────────────────────────────────────
    if !was_interrupted {
        let mut spilled = 0usize;
        for record in episodic.iter_records() {
            if let Err(e) = recall.cold.append(record) {
                tracing::warn!(?e, "cold archive append error");
                break;
            }
            spilled += 1;
        }
        tracing::info!(spilled, "hot→cold spill complete");
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
