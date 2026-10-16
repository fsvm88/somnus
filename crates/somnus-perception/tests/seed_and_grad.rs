//! Process-isolated checks (burn's RNG seed is global, so these must not share a process with
//! tests that build models concurrently): seeded init reproducibility, and that gradients reach
//! the encoder/decoder/heads (a silently detached autodiff graph would still pass type checks).
use burn::tensor::Device;
use somnus_perception::{PerceptionProcessor, REGION_DIMS};

fn lcg(seed: u64, len: usize) -> Vec<f32> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// burn's RNG state is process-global: serialize the tests in this file.
static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn init_heads(seed: u64) -> Vec<Vec<f32>> {
    // Seed one device instance, construct from a *fresh* one: that is how `run()` does it.
    Device::flex().seed(seed);
    PerceptionProcessor::new(Device::flex())
        .head_weights_flat()
        .to_vec()
}

#[test]
fn seeded_init_is_reproducible_and_seed_dependent() {
    let _g = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let a = init_heads(7);
    let b = init_heads(7);
    let c = init_heads(8);
    assert_eq!(a, b, "same seed must give identical initial weights");
    assert_ne!(a, c, "different seeds must give different initial weights");
}

#[test]
fn replay_step_moves_weights_and_reduces_loss() {
    let _g = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::flex();
    device.seed(1);
    let mut proc = PerceptionProcessor::new(device);
    let regional: [Vec<f32>; 5] = std::array::from_fn(|i| lcg(10 + i as u64, REGION_DIMS[i]));
    let mse_before = proc.measure_mse(&regional).unwrap();
    let losses: Vec<f32> = (0..4)
        .map(|_| proc.replay_step(&regional).unwrap())
        .collect();
    let mse_after = proc.measure_mse(&regional).unwrap();
    assert!(losses.iter().all(|l| l.is_finite()), "losses: {losses:?}");
    assert!(
        losses[3] < losses[0],
        "repeated steps on one batch must reduce the loss: {losses:?}"
    );
    assert_ne!(mse_before, mse_after, "weights must move");
}

/// ADR 0048: a saved `wake_binding` is the *lagged* copy and must survive a restart untouched;
/// only snapshots without `wake_binding.*` (pre-ADR-0048) get it derived from `binding`.
#[test]
fn restore_keeps_saved_lagged_wake_binding() {
    use somnus_perception::RegionId;
    let _g = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::flex();
    device.seed(3);
    let mut a = PerceptionProcessor::new(device.clone());
    let regional: [Vec<f32>; 5] = std::array::from_fn(|i| lcg(50 + i as u64, REGION_DIMS[i]));
    // Train `binding` live without syncing: binding drifts away from its wake copy.
    for _ in 0..3 {
        a.binding_replay_step_batch(&[&regional], 1.0, false)
            .unwrap();
    }
    let r = RegionId::AudioSpeech;
    let (live_a, wake_a) = (
        a.binding_proj_row_norms(r),
        a.wake_binding_proj_row_norms(r),
    );
    assert_ne!(
        live_a, wake_a,
        "binding must have drifted from wake_binding"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights");
    a.save_weights(&path).unwrap();

    let mut b = PerceptionProcessor::new(device);
    b.load_weights(&path).unwrap();
    assert_eq!(b.binding_proj_row_norms(r), live_a, "binding restored");
    assert_eq!(
        b.wake_binding_proj_row_norms(r),
        wake_a,
        "restore must keep the saved lagged wake_binding, not re-sync it from binding"
    );
}
