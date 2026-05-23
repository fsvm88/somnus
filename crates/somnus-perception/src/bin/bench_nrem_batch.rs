/// Benchmark: sequential replay_step vs batched replay_step_batch.
///
/// Generates N synthetic traces (random f32 regional inputs matching real
/// dimensions), then times one full NREM pass at B=1 (current), B=16, B=32,
/// B=64. Prints throughput in traces/s and relative speedup.
///
/// Run: cargo run --release -p somnus-perception --bin bench_nrem_batch
use candle_core::Device;
use somnus_perception::PerceptionProcessor;
use std::time::Instant;

const N_TRACES: usize = 3276;
const REGION_DIMS: [usize; 5] = [384, 512, 384, 384, 896];
const BATCH_SIZES: &[usize] = &[1, 16, 32, 64, 128];

fn random_regional(seed_base: usize) -> [Vec<f32>; 5] {
    // Deterministic LCG so results are reproducible.
    let mut state = (seed_base as u64).wrapping_add(0xdeadbeef);
    let next = |s: &mut u64| -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    std::array::from_fn(|i| (0..REGION_DIMS[i]).map(|_| next(&mut state)).collect())
}

fn run_sequential(proc: &mut PerceptionProcessor, traces: &[[Vec<f32>; 5]]) -> f32 {
    let mut total = 0.0f32;
    for t in traces {
        total += proc.replay_step(t).unwrap_or(0.0);
    }
    total / traces.len() as f32
}

fn run_batched(proc: &mut PerceptionProcessor, traces: &[[Vec<f32>; 5]], batch_size: usize) -> f32 {
    let mut total = 0.0f32;
    let mut n_batches = 0usize;
    for chunk in traces.chunks(batch_size) {
        let refs: Vec<&[Vec<f32>; 5]> = chunk.iter().collect();
        total += proc.replay_step_batch(&refs).unwrap_or(0.0);
        n_batches += 1;
    }
    total / n_batches.max(1) as f32
}

fn main() {
    let device = Device::Cpu;
    println!("Generating {N_TRACES} synthetic traces…");
    let traces: Vec<[Vec<f32>; 5]> = (0..N_TRACES).map(random_regional).collect();

    println!("Warming up JIT / allocators (1 sequential pass, not timed)…");
    {
        let mut proc = PerceptionProcessor::new(device.clone()).unwrap();
        let _ = run_sequential(&mut proc, &traces[..32]);
    }

    println!("\n{:<10} {:>14} {:>14} {:>10}", "batch_size", "wall_ms", "traces/s", "speedup_vs_1");
    println!("{}", "-".repeat(52));

    let mut baseline_tps = 0.0f64;

    for &bs in BATCH_SIZES {
        // Fresh processor each run so optimizer state + duty cycles start equal.
        let mut proc = PerceptionProcessor::new(device.clone()).unwrap();

        let t0 = Instant::now();
        if bs == 1 {
            run_sequential(&mut proc, &traces);
        } else {
            run_batched(&mut proc, &traces, bs);
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let tps = N_TRACES as f64 / (elapsed_ms / 1000.0);

        if bs == 1 {
            baseline_tps = tps;
        }
        let speedup = tps / baseline_tps.max(1.0);

        println!("{:<10} {:>14.1} {:>14.0} {:>10.2}×", bs, elapsed_ms, tps, speedup);
    }
}
