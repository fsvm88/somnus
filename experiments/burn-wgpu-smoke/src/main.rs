/// Smoke test: verify burn-wgpu works on AMD RX 9070 XT (gfx1201 / RDNA 4)
/// and benchmark realistic NREM matmul sizes vs CPU (burn/ndarray).
///
/// Two benchmarks:
///   1. 1024×1024 — confirms GPU is working and gives a headline speedup number.
///   2. Realistic NREM sizes — 5 regions, B=64, each [B, n_in] × [n_units, n_in]^T.
///      This is the exact matmul shape from the batched k-WTA forward pass.
use burn::backend::{NdArray, Wgpu, ndarray::NdArrayDevice, wgpu::WgpuDevice};
use burn::tensor::{Tensor, TensorData};
use std::time::Instant;

// Region encoder dims: [AudioSpeech, AudioEnv, Visual, Symbolic, Language]
const REGION_DIMS: [usize; 5] = [384, 512, 384, 384, 896];
const N_UNITS: usize = 512;
const BATCH: usize = 64;
const ITERS: usize = 40;

fn bench_matmul_gpu(b: usize, n_in: usize, n_units: usize, device: &WgpuDevice) -> f64 {
    let a_data: Vec<f32> = (0..b * n_in).map(|i| i as f32 * 0.001).collect();
    let w_data: Vec<f32> = (0..n_units * n_in).map(|i| i as f32 * 0.001).collect();
    let a: Tensor<Wgpu, 2> = Tensor::from_data(TensorData::new(a_data, [b, n_in]), device);
    let w: Tensor<Wgpu, 2> = Tensor::from_data(TensorData::new(w_data, [n_units, n_in]), device);

    // Warmup
    for _ in 0..5 {
        let _ = a.clone().matmul(w.clone().transpose());
    }
    let _: f32 = a.clone().matmul(w.clone().transpose()).sum().into_scalar();

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = a.clone().matmul(w.clone().transpose());
    }
    let sum: f32 = a.clone().matmul(w.clone().transpose()).sum().into_scalar();
    assert!(sum.is_finite(), "GPU result is not finite");
    t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64
}

fn bench_matmul_cpu(b: usize, n_in: usize, n_units: usize, device: &NdArrayDevice) -> f64 {
    let a_data: Vec<f32> = (0..b * n_in).map(|i| i as f32 * 0.001).collect();
    let w_data: Vec<f32> = (0..n_units * n_in).map(|i| i as f32 * 0.001).collect();
    let a: Tensor<NdArray, 2> = Tensor::from_data(TensorData::new(a_data, [b, n_in]), device);
    let w: Tensor<NdArray, 2> = Tensor::from_data(TensorData::new(w_data, [n_units, n_in]), device);

    // Warmup
    for _ in 0..3 {
        let _ = a.clone().matmul(w.clone().transpose());
    }
    let _: f32 = a.clone().matmul(w.clone().transpose()).sum().into_scalar();

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = a.clone().matmul(w.clone().transpose());
    }
    let _: f32 = a.clone().matmul(w.clone().transpose()).sum().into_scalar();
    t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64
}

fn main() {
    println!("burn-wgpu smoke test — AMD RX 9070 XT (gfx1201)");

    let gpu = WgpuDevice::DefaultDevice;
    let cpu = NdArrayDevice::Cpu;

    // ── 1. Large matmul sanity check ──────────────────────────────────────────
    println!("\n[1] 1024×1024 matmul (sanity check)");
    const N: usize = 1024;
    let gpu_ms_large = bench_matmul_gpu(N, N, N, &gpu);
    let cpu_ms_large = bench_matmul_cpu(N, N, N, &cpu);
    println!(
        "  GPU: {gpu_ms_large:.2} ms  CPU: {cpu_ms_large:.2} ms  speedup: {:.1}×",
        cpu_ms_large / gpu_ms_large
    );

    // ── 2. Realistic NREM sizes ───────────────────────────────────────────────
    println!("\n[2] Realistic NREM matmul: B={BATCH}, n_units={N_UNITS}");
    println!("  {:<16} {:>10} {:>10} {:>10}", "region", "gpu ms", "cpu ms", "speedup");

    let region_names = ["AudioSpeech", "AudioEnv", "Visual", "Symbolic", "Language"];
    let mut total_gpu = 0.0f64;
    let mut total_cpu = 0.0f64;

    for (i, &n_in) in REGION_DIMS.iter().enumerate() {
        let gms = bench_matmul_gpu(BATCH, n_in, N_UNITS, &gpu);
        let cms = bench_matmul_cpu(BATCH, n_in, N_UNITS, &cpu);
        let sx = cms / gms;
        println!("  {:<16} {:>10.3} {:>10.3} {:>10.1}×", region_names[i], gms, cms, sx);
        total_gpu += gms;
        total_cpu += cms;
    }

    let total_speedup = total_cpu / total_gpu;
    println!("  {:<16} {:>10.3} {:>10.3} {:>10.1}×", "TOTAL", total_gpu, total_cpu, total_speedup);

    println!("\nResult: GPU gives {total_speedup:.1}× on realistic NREM forward matmuls.");
    if total_speedup >= 2.0 {
        println!("burn+wgpu migration is worth it (>= 2×)");
    } else {
        println!("WARN: GPU advantage is small for these sizes — reconsider migration");
    }
}
