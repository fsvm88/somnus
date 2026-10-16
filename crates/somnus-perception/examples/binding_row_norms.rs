//! ADR 0041 A/B-pilot inspection: binding-projection row-norm stats for a
//! snapshot's weights, split into base rows (0..N_UNITS) and grown rows
//! (N_UNITS..). Grown rows are zero-initialized by growth and only
//! `binding_replay_step_batch` ever trains them, so a treatment arm whose
//! grown units won k-WTA should show grown rows off zero here while the
//! control arm stays exactly zero (ADR 0041 "Validation plan" / "What would
//! invalidate this").
//!
//! Usage: cargo run -p somnus-perception --release --example binding_row_norms -- <weights.safetensors>
use burn::tensor::Device;
use somnus_perception::{PerceptionProcessor, RegionId, N_UNITS};

fn main() {
    let weights = std::env::args()
        .nth(1)
        .unwrap_or_else(|| panic!("usage: binding_row_norms <weights.safetensors>"));
    let weights_path = std::path::Path::new(&weights);
    let sizes_str =
        std::fs::read_to_string(weights_path.with_extension("sizes")).expect("read .sizes sidecar");
    let sizes: [usize; 5] = sizes_str
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .expect("5 region sizes");

    let mut proc = PerceptionProcessor::new(Device::flex());
    proc.load_weights_with_sizes(weights_path, sizes)
        .expect("load weights");

    let regions = [
        RegionId::AudioSpeech,
        RegionId::AudioEnv,
        RegionId::Visual,
        RegionId::Symbolic,
        RegionId::Language,
    ];
    println!(
        "{:12} {:>6} {:>6} {:>9} {:>12} {:>12} {:>12}",
        "region", "n", "grown", "grown_nz", "grown_mean", "grown_max", "base_mean"
    );
    for region in regions {
        let norms = proc.binding_proj_row_norms(region);
        let n = norms.len();
        let grown = &norms[N_UNITS..];
        let base = &norms[..N_UNITS];
        let grown_nz = grown.iter().filter(|&&v| v > 0.0).count();
        let grown_mean = grown.iter().sum::<f32>() / grown.len().max(1) as f32;
        let grown_max = grown.iter().copied().fold(0.0f32, f32::max);
        let base_mean = base.iter().sum::<f32>() / base.len().max(1) as f32;
        println!(
            "{:12} {:6} {:6} {:9} {:12.6} {:12.6} {:12.6}",
            format!("{:?}", region),
            n,
            grown.len(),
            grown_nz,
            grown_mean,
            grown_max,
            base_mean
        );
    }
}
