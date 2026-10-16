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
    let device = Device::flex();
    device.seed(seed);
    PerceptionProcessor::new(device)
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
