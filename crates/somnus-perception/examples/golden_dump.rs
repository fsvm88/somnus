//! Numerical reference for the burn 0.21 -> 0.22 migration: load a safetensors weights file
//! (+ `weights.sizes` sidecar next to it), drive a fixed deterministic sequence of forward and
//! training operations, and print a JSON summary. Run on both burn versions and compare with
//! `scripts/compare_golden.py`.
//!
//! Usage: cargo run -p somnus-perception --release --example golden_dump -- <weights.safetensors> > golden.json
use burn::backend::ndarray::NdArrayDevice;
use somnus_perception::{PerceptionProcessor, RegionId, EPISODE_DIM, REGION_DIMS};

/// Deterministic pseudo-random vector in [-1, 1] (LCG; independent of any burn RNG).
fn vec_for(seed: u64, len: usize) -> Vec<f32> {
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

fn stats(v: &[f32]) -> serde_json::Value {
    let n = v.len() as f64;
    let sum: f64 = v.iter().map(|&x| x as f64).sum();
    let sq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let head: Vec<f32> = v.iter().take(8).copied().collect();
    serde_json::json!({ "len": v.len(), "sum": sum, "l2": sq.sqrt(), "mean": sum / n, "head": head })
}

fn main() {
    let weights = std::env::args()
        .nth(1)
        .unwrap_or_else(|| panic!("usage: golden_dump <weights.safetensors>"));
    let path = std::path::Path::new(&weights);
    let sizes: [usize; 5] = std::fs::read_to_string(path.with_extension("sizes"))
        .expect("read .sizes sidecar")
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .expect("5 region sizes");

    let mut proc = PerceptionProcessor::new(NdArrayDevice::Cpu);
    proc.load_weights_safetensors_with_sizes(path, sizes)
        .expect("load weights");
    proc.set_learning_rate(1e-4);

    let mut out = serde_json::Map::new();
    out.insert("sizes".into(), serde_json::json!(proc.region_sizes()));

    // Forward through encoder + binding (stateful: duty cycles, GRU hidden) on a fixed sequence.
    // Two rounds over all four external modalities so an episode (binding output) is produced.
    let mut ingest = Vec::new();
    for round in 0..2u64 {
        for modality in 0..4usize {
            let seed = 100 + round * 10 + modality as u64;
            let o = match modality {
                0 => proc.ingest_speech(&vec_for(seed, REGION_DIMS[0])),
                1 => proc.ingest_env(&vec_for(seed, REGION_DIMS[1])),
                2 => proc.ingest_visual(&vec_for(seed, REGION_DIMS[2])),
                _ => proc.ingest_language(&vec_for(seed, REGION_DIMS[4])),
            }
            .expect("ingest");
            let recon: Vec<(usize, f32)> = o
                .recon_errors
                .iter()
                .map(|(r, e)| (r.as_index(), *e))
                .collect();
            ingest.push(serde_json::json!({
                "round": round,
                "modality": modality,
                "recon_errors": recon,
                "episode": o.episode.as_ref().map(|e| stats(&e.0)),
                "prior_cosines": o.prior_cosines,
                "settle_iters": o.settle_iters,
                "settle_delta": o.settle_delta,
            }));
        }
    }
    out.insert("ingest_speech".into(), ingest.into());

    let regional: [Vec<f32>; 5] = std::array::from_fn(|i| vec_for(200 + i as u64, REGION_DIMS[i]));
    out.insert(
        "measure_mse_before".into(),
        serde_json::json!(proc.measure_mse(&regional).expect("measure_mse")),
    );
    let episode = vec_for(300, EPISODE_DIM);
    let outs = proc.infer_episode_outputs(&episode).expect("infer");
    out.insert(
        "infer_outputs".into(),
        outs.iter().map(|v| stats(v)).collect::<Vec<_>>().into(),
    );

    // Two training steps: loss must be finite and weights must move.
    let heads_before = proc.head_weights_flat();
    let l1 = proc.replay_step(&regional).expect("replay_step 1");
    let l2 = proc.replay_step(&regional).expect("replay_step 2");
    out.insert("replay_loss".into(), serde_json::json!([l1, l2]));
    let heads_after = proc.head_weights_flat();
    out.insert(
        "heads_after".into(),
        heads_after
            .iter()
            .map(|v| stats(v))
            .collect::<Vec<_>>()
            .into(),
    );
    let moved: Vec<f64> = heads_before
        .iter()
        .zip(heads_after.iter())
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(x, y)| ((x - y) as f64).abs())
                .sum::<f64>()
        })
        .collect();
    out.insert("heads_abs_delta".into(), serde_json::json!(moved));
    out.insert(
        "measure_mse_after".into(),
        serde_json::json!(proc.measure_mse(&regional).expect("measure_mse")),
    );
    out.insert(
        "binding_row_norms".into(),
        serde_json::json!([RegionId::AudioSpeech, RegionId::Language]
            .iter()
            .map(|&r| (
                stats(&proc.binding_proj_row_norms(r)),
                stats(&proc.wake_binding_proj_row_norms(r))
            ))
            .collect::<Vec<_>>()),
    );
    println!("{}", serde_json::Value::Object(out));
}
