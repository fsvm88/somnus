/// Benchmark: NREM throughput — replay-only vs fused replay+wake.
///
/// Generates N synthetic traces (random f32 regional inputs + episode vectors),
/// then times one full NREM pass at various batch sizes.  Two modes:
///   replay_only  — replay_step_batch alone (old baseline)
///   fused        — fused_replay_wake_step_batch (current production code)
///
/// Run: cargo run --release -p somnus-perception --bin bench_nrem_batch
use burn::tensor::Device;
use somnus_perception::PerceptionProcessor;
use std::time::Instant;

const N_TRACES: usize = 3276;
const REGION_DIMS: [usize; 5] = [384, 512, 384, 384, 896];
const EPISODE_DIM: usize = 640;
const BATCH_SIZES: &[usize] = &[1, 16, 32, 64, 128];

fn random_regional(seed_base: usize) -> [Vec<f32>; 5] {
    let mut state = (seed_base as u64).wrapping_add(0xdeadbeef);
    let next = |s: &mut u64| -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    std::array::from_fn(|i| (0..REGION_DIMS[i]).map(|_| next(&mut state)).collect())
}

fn random_episode(seed_base: usize) -> Vec<f32> {
    let mut state = (seed_base as u64).wrapping_add(0xcafebabe);
    let next = |s: &mut u64| -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    (0..EPISODE_DIM).map(|_| next(&mut state)).collect()
}

fn run_sequential(proc: &mut PerceptionProcessor, traces: &[[Vec<f32>; 5]]) -> f32 {
    let mut total = 0.0f32;
    for t in traces {
        total += proc.replay_step(t).unwrap_or(0.0);
    }
    total / traces.len() as f32
}

fn run_replay_only(
    proc: &mut PerceptionProcessor,
    traces: &[[Vec<f32>; 5]],
    batch_size: usize,
) -> f32 {
    let mut total = 0.0f32;
    let mut n_batches = 0usize;
    for chunk in traces.chunks(batch_size) {
        let refs: Vec<&[Vec<f32>; 5]> = chunk.iter().collect();
        total += proc.replay_step_batch(&refs).unwrap_or(0.0);
        n_batches += 1;
    }
    total / n_batches.max(1) as f32
}

fn run_fused(
    proc: &mut PerceptionProcessor,
    traces: &[[Vec<f32>; 5]],
    episodes: &[Vec<f32>],
    batch_size: usize,
) -> f32 {
    let mut total = 0.0f32;
    let mut n_batches = 0usize;
    for (chunk_t, chunk_e) in traces.chunks(batch_size).zip(episodes.chunks(batch_size)) {
        let reg_refs: Vec<&[Vec<f32>; 5]> = chunk_t.iter().collect();
        let ep_refs: Vec<&[f32]> = chunk_e.iter().map(|v| v.as_slice()).collect();
        let (loss, _mse) = proc
            .fused_replay_wake_step_batch(&reg_refs, &ep_refs)
            .unwrap_or((0.0, 0.0));
        total += loss;
        n_batches += 1;
    }
    total / n_batches.max(1) as f32
}

fn print_table(header: &str, results: &[(usize, f64, f64)]) {
    println!("\n{header}");
    println!(
        "{:<10} {:>14} {:>14} {:>10}",
        "batch_size", "wall_ms", "traces/s", "speedup_vs_1"
    );
    println!("{}", "-".repeat(52));
    let baseline = results.first().map(|r| r.2).unwrap_or(1.0);
    for &(bs, ms, tps) in results {
        println!(
            "{:<10} {:>14.1} {:>14.0} {:>10.2}×",
            bs,
            ms,
            tps,
            tps / baseline.max(1.0)
        );
    }
}

fn main() {
    let device = Device::flex();
    println!("Generating {N_TRACES} synthetic traces…");
    let traces: Vec<[Vec<f32>; 5]> = (0..N_TRACES).map(random_regional).collect();
    let episodes: Vec<Vec<f32>> = (0..N_TRACES).map(random_episode).collect();

    println!("Warming up (1 sequential pass, not timed)…");
    {
        let mut proc = PerceptionProcessor::new(device.clone());
        let _ = run_sequential(&mut proc, &traces[..32]);
    }

    // ── replay-only (backward reference) ──────────────────────────────────────
    let mut replay_results = Vec::new();
    for &bs in BATCH_SIZES {
        let mut proc = PerceptionProcessor::new(device.clone());
        let t0 = Instant::now();
        if bs == 1 {
            run_sequential(&mut proc, &traces);
        } else {
            run_replay_only(&mut proc, &traces, bs);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        replay_results.push((bs, ms, N_TRACES as f64 / (ms / 1000.0)));
    }
    print_table(
        "replay_step_batch (SAE only, backward reference)",
        &replay_results,
    );

    // ── fused replay+wake ──────────────────────────────────────────────────────
    let mut fused_results = Vec::new();
    for &bs in BATCH_SIZES {
        let mut proc = PerceptionProcessor::new(device.clone());
        let t0 = Instant::now();
        if bs == 1 {
            // sequential fused as baseline
            for (t, e) in traces.iter().zip(episodes.iter()) {
                let _ = proc.fused_replay_wake_step_batch(&[t], &[e.as_slice()]);
            }
        } else {
            run_fused(&mut proc, &traces, &episodes, bs);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        fused_results.push((bs, ms, N_TRACES as f64 / (ms / 1000.0)));
    }
    print_table(
        "fused_replay_wake_step_batch (SAE + output heads, 1 backward)",
        &fused_results,
    );
}
