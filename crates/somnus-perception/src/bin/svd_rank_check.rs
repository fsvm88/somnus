/// Export somnus's own growable region/binding weight matrices from a trained checkpoint,
/// as raw f32 files + a manifest, for an offline SVD effective-rank check (numpy side).
///
/// Loads `<memory_dir>/snapshot/{region_sizes.bin,weights.bin}` (pre-growing regions to
/// match the saved sizes, same as the live system does on restore), then dumps via
/// `export_weights_for_onnx`. Does not touch decoder_weights.bin (ActionDecoder) — this
/// checks somnus's own region/binding weights, not the LM-side projection.
///
/// Run: cargo run --release -p somnus-perception --bin svd_rank_check -- <memory_dir> <out_dir>
/// Then: scripts/.venv/bin/python3 scripts/svd_rank_check.py <out_dir>
use burn::backend::ndarray::NdArrayDevice;
use somnus_perception::PerceptionProcessor;
use std::path::PathBuf;

fn load_region_sizes(memory_dir: &std::path::Path) -> [usize; 5] {
    let path = memory_dir.join("snapshot").join("region_sizes.bin");
    let data =
        std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    assert!(data.len() >= 40, "{} too short for 5×u64", path.display());
    let mut sizes = [0usize; 5];
    for (i, s) in sizes.iter_mut().enumerate() {
        let bytes: [u8; 8] = data[i * 8..i * 8 + 8].try_into().unwrap();
        *s = u64::from_le_bytes(bytes) as usize;
    }
    sizes
}

fn main() {
    let mut args = std::env::args().skip(1);
    let memory_dir: PathBuf = args
        .next()
        .expect("usage: svd_rank_check <memory_dir> <out_dir>")
        .into();
    let out_dir: PathBuf = args
        .next()
        .expect("usage: svd_rank_check <memory_dir> <out_dir>")
        .into();

    let sizes = load_region_sizes(&memory_dir);
    println!("Region sizes from checkpoint: {sizes:?}");

    let device = NdArrayDevice::Cpu;
    let mut proc = PerceptionProcessor::new(device);

    let weights_path = memory_dir.join("snapshot").join("weights");
    proc.load_weights_with_sizes(&weights_path, sizes)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", weights_path.display()));
    println!("Loaded trained weights from {}", weights_path.display());

    std::fs::create_dir_all(&out_dir).unwrap();
    proc.export_weights_for_onnx(&out_dir).unwrap();
    println!(
        "Exported raw f32 weights + manifest to {}",
        out_dir.display()
    );
}
