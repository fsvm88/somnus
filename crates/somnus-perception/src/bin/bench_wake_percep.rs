/// Micro-benchmark: per-frame wake perception cost.
///
/// Times settle_episode (via ingest_visual with settle_steps=1) and
/// wake_reconstruct_step individually to characterise the real bottleneck.
///
/// Run: cargo run --release -p somnus-perception --bin bench_wake_percep
use burn::tensor::Device;
use somnus_perception::{PerceptionProcessor, EPISODE_DIM, REGION_DIMS};
use std::time::Instant;

const N_REPS: usize = 200;

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

fn rand_vec(len: usize, seed: &mut u64) -> Vec<f32> {
    (0..len).map(|_| lcg(seed)).collect()
}

fn main() {
    let device = Device::flex();
    let mut proc = PerceptionProcessor::new(device.clone());
    proc.set_settle_steps(1);
    proc.set_settle_beta(0.5);

    // Warm up: feed some frames so prior_predictions and latest_inputs are populated.
    let mut seed = 0xdeadbeef_u64;
    for i in 0..20 {
        let vis = rand_vec(REGION_DIMS[2], &mut seed);
        proc.ingest_visual(&vis).ok();
        if i % 4 == 0 {
            let sp = rand_vec(REGION_DIMS[0], &mut seed);
            proc.ingest_speech(&sp).ok();
        }
    }
    println!("Warmed up. Running {} reps each…\n", N_REPS);

    // ── ingest_visual (settle + prior update) ─────────────────────────────────
    let frames: Vec<Vec<f32>> = (0..N_REPS)
        .map(|i| rand_vec(REGION_DIMS[2], &mut seed.wrapping_add(i as u64)))
        .collect();
    let t0 = Instant::now();
    let mut eps = Vec::new();
    for f in &frames {
        if let Ok(out) = proc.ingest_visual(f) {
            eps.push(out);
        }
    }
    let ingest_ms = t0.elapsed().as_millis() as f64 / N_REPS as f64;
    println!("ingest_visual (settle_steps=1 + prior update)");
    println!(
        "  {N_REPS} reps → {ingest_ms:.1} ms/frame  ({:.0} frames/s)\n",
        1000.0 / ingest_ms
    );

    // ── wake_reconstruct_step ─────────────────────────────────────────────────
    // Use the last few episodes from ingest as inputs.
    let regional: [Vec<f32>; 5] = std::array::from_fn(|i| rand_vec(REGION_DIMS[i], &mut seed));
    let ep_data: Vec<f32> = rand_vec(EPISODE_DIM, &mut seed);

    // Warmup backward pass (JIT compilation etc.)
    proc.wake_reconstruct_step(&ep_data, &regional).ok();

    let t1 = Instant::now();
    for _ in 0..N_REPS {
        proc.wake_reconstruct_step(&ep_data, &regional).ok();
    }
    let recon_ms = t1.elapsed().as_millis() as f64 / N_REPS as f64;
    println!("wake_reconstruct_step (forward + backward + optimizer)");
    println!(
        "  {N_REPS} reps → {recon_ms:.1} ms/call  ({:.0} calls/s)\n",
        1000.0 / recon_ms
    );

    println!(
        "Ratio wake_reconstruct / ingest: {:.1}×",
        recon_ms / ingest_ms
    );
}
