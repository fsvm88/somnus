/// End-to-end NREM step microbenchmark: burn+wgpu (GPU) vs candle CPU reference.
///
/// Measures the full critical path for one region, one batch:
///   linear forward → top-k (CPU readback) → mask → SAE decode → MSE → backward → AdamW
use std::time::Instant;

use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{Tensor, TensorData},
};

type B = Autodiff<Wgpu>;

const N_IN: usize = 896; // Language region (worst case)
const N_UNITS: usize = 512;
const K: usize = 25; // ≈ 5% of N_UNITS
const BATCH: usize = 64;
const ITERS: usize = 40;

/// Combined module: k-WTA encoder + SAE decoder. One backward pass covers both.
#[derive(Module, Debug)]
struct NremModule<B: burn::tensor::backend::Backend> {
    encoder: Linear<B>,
    decoder: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> NremModule<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            encoder: LinearConfig::new(N_IN, N_UNITS).with_bias(false).init(device),
            decoder: LinearConfig::new(N_UNITS, N_IN).with_bias(false).init(device),
        }
    }
}

fn nrem_step(
    model: NremModule<B>,
    optim: &mut impl Optimizer<NremModule<B>, B>,
    input: Tensor<B, 2>,
    device: &WgpuDevice,
) -> NremModule<B> {
    // 1. Encoder forward
    let z = model.encoder.forward(input.clone());

    // 2. Top-k CPU readback — the critical sync point
    let z_vals: Vec<f32> = z
        .clone()
        .detach()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    let mut flat_masks = vec![0.0f32; BATCH * N_UNITS];
    for si in 0..BATCH {
        let row = &z_vals[si * N_UNITS..(si + 1) * N_UNITS];
        let mut idx_val: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        idx_val.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, _) in idx_val.iter().take(K) {
            flat_masks[si * N_UNITS + i] = 1.0;
        }
    }

    // 3. Upload mask back to GPU (detached — no gradient through mask)
    let mask: Tensor<Wgpu, 2> =
        Tensor::from_data(TensorData::new(flat_masks, [BATCH, N_UNITS]), device);
    let mask: Tensor<B, 2> = Tensor::from_inner(mask);

    // 4. Sparse activation + SAE decode
    let sparse_z = z * mask;
    let recon = model.decoder.forward(sparse_z);

    // 5. MSE loss + backward + AdamW
    let loss = (input - recon).powi_scalar(2).mean();
    let grads = loss.backward();
    let grads_p = GradientsParams::from_grads(grads, &model);
    optim.step(1e-3, model, grads_p)
}

pub fn run() {
    let device = WgpuDevice::DefaultDevice;
    let mut model = NremModule::<B>::new(&device);
    let mut optim = AdamWConfig::new().init::<B, NremModule<B>>();

    let inp_data: Vec<f32> = (0..BATCH * N_IN).map(|i| (i as f32 * 0.001).sin()).collect();
    let make_input = || -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(inp_data.clone(), [BATCH, N_IN]), &device)
    };

    // Warmup
    println!("  warming up GPU...");
    for _ in 0..5 {
        model = nrem_step(model, &mut optim, make_input(), &device);
    }
    println!("  warmup done");

    // Timed run
    let t0 = Instant::now();
    for _ in 0..ITERS {
        model = nrem_step(model, &mut optim, make_input(), &device);
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
    let gpu_tps = BATCH as f64 / (gpu_ms / 1000.0);
    println!("  burn+wgpu: {gpu_ms:.2} ms/step  ({gpu_tps:.0} traces/s, one region)");

    // candle CPU reference from bench_nrem_batch: B=64 → 361 t/s over 5 regions
    // Per-region estimate: 5 regions in parallel within one step → 5 × (64/361 s) = 0.88s?
    // No — they're sequential: 1 step = all 5 regions + 1 optimizer call ≈ 177ms.
    // Single-region fraction: 177ms / 5 ≈ 35ms (rough, since optimizer is shared).
    let candle_5region_ms = 177.0f64; // from bench_nrem_batch B=64
    let candle_1region_ms = candle_5region_ms / 5.0;
    let candle_tps = BATCH as f64 / (candle_1region_ms / 1000.0);
    println!("  candle CPU ref: ~{candle_1region_ms:.0} ms/step  ({candle_tps:.0} traces/s) [5-region total / 5]");

    let speedup = candle_1region_ms / gpu_ms;
    println!("  end-to-end speedup (GPU vs CPU): {speedup:.1}×");

    if speedup >= 2.0 {
        println!("  → GPU advantage confirmed (>= 2×): burn migration is justified");
    } else if speedup >= 1.2 {
        println!("  → modest GPU win ({speedup:.1}×): migration ROI is questionable");
    } else {
        println!("  → negligible GPU advantage: readback/small-tensor overhead dominates");
    }

    let _ = model;
}
