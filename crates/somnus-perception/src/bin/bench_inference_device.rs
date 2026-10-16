/// Benchmark (gaps.md R7): does running no-grad inference on an autodiff-enabled burn device cost
/// anything compared with the plain device?
///
/// Since the burn 0.22 port, `PerceptionProcessor` keeps `self.device = device.autodiff()` and builds
/// the inference tensors (k-WTA scoring, SAE reconstruction, `measure_mse`) on it, although those
/// paths never need gradients. This measures the production shapes on both devices:
///   * `Linear` 384 -> 9821 (a grown k-WTA layer), row vector in, readback out;
///   * the same with a 9821 -> 384 SAE decoder step;
///   * a batch-32 matmul (the batched REM scoring shape queued as R4).
///
/// Run: cargo run --release -p somnus-perception --bin bench_inference_device
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, TensorData};
use somnus_perception::TensorF32Ext;
use std::time::Instant;

const N_IN: usize = 384;
const N_UNITS: usize = 9821;
const REPS: usize = 300;

fn input(batch: usize, device: &Device) -> Tensor<2> {
    let data: Vec<f32> = (0..batch * N_IN)
        .map(|i| ((i % 97) as f32) * 0.01 - 0.4)
        .collect();
    Tensor::from_data(TensorData::new(data, [batch, N_IN]), device)
}

fn time(label: &str, mut f: impl FnMut()) -> f64 {
    for _ in 0..10 {
        f();
    }
    let t = Instant::now();
    for _ in 0..REPS {
        f();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    println!("  {label:<34} {ms:8.3} ms/call");
    ms
}

fn run(name: &str, device: &Device) -> [f64; 3] {
    println!("{name} (autodiff = {})", device.is_autodiff());
    let enc: Linear = LinearConfig::new(N_IN, N_UNITS)
        .with_bias(false)
        .init(device);
    let dec: Linear = LinearConfig::new(N_UNITS, N_IN)
        .with_bias(false)
        .init(device);
    let x1 = input(1, device);
    let x32 = input(32, device);
    let a = time("k-WTA score + readback (b=1)", || {
        std::hint::black_box(enc.forward(x1.clone()).to_f32_vec());
    });
    let b = time("score -> decode -> readback (b=1)", || {
        std::hint::black_box(dec.forward(enc.forward(x1.clone())).to_f32_vec());
    });
    let c = time("score + readback (b=32)", || {
        std::hint::black_box(enc.forward(x32.clone()).to_f32_vec());
    });
    [a, b, c]
}

fn main() {
    let plain = Device::flex();
    let auto = Device::flex().autodiff();
    // Two rounds in alternating order; report the second (the first absorbs thread-pool warm-up
    // and allocator effects that otherwise favour whichever device runs second).
    let _ = run("plain device (warm-up round)", &plain);
    let _ = run("autodiff device (warm-up round)", &auto);
    let a = run("autodiff device", &auto);
    let p = run("plain device", &plain);
    println!("\nautodiff / plain:");
    for (label, (x, y)) in ["score b=1", "score+decode b=1", "score b=32"]
        .iter()
        .zip(a.iter().zip(p.iter()))
    {
        println!("  {label:<18} {:.2}x", x / y);
    }
}
