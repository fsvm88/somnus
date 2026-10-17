//! End-to-end encode tests against the real frozen ONNX encoders in `models/`.
//!
//! Skips (loudly, not silently) when the model artifacts aren't on disk — they
//! live in the gitignored `models/` tree (symlinked to /data/somnus/models on the
//! production box), same skip-don't-fail pattern as somnus-perception's
//! ort_grown_region.rs. When present, these pin the exact contracts production
//! depends on: input lengths, output dims, and determinism of every encode call
//! in this crate — the paths that sit on every request's critical path.

use std::path::{Path, PathBuf};

use somnus_encoding::{AudioEncoder, AudioEncoderSource, VideoEncoder};
use somnus_sensory::{PixelFormat, Timestamp, VideoFrame};

#[cfg(not(feature = "mn10_env"))]
const ENV_ONNX_FILE: &str = "clap_audio_encoder_fp16.onnx";
#[cfg(feature = "mn10_env")]
const ENV_ONNX_FILE: &str = "efficientat_mn10_encoder_fp16.onnx";
#[cfg(not(feature = "mn10_env"))]
const ENV_DIM: usize = 512;
#[cfg(feature = "mn10_env")]
const ENV_DIM: usize = 960;
#[cfg(not(feature = "mn10_env"))]
const ENV_SAMPLES: usize = 480_000;
#[cfg(feature = "mn10_env")]
const ENV_SAMPLES: usize = 320_000;

const WHISPER_SAMPLES: usize = 480_000;

fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

/// Skip-with-message when any of the needed artifacts is missing.
fn require_models(files: &[&str]) -> Option<PathBuf> {
    let dir = models_dir();
    for f in files {
        if !dir.join(f).exists() {
            eprintln!("skipping: models/{f} not present (gitignored artifact tree)");
            return None;
        }
    }
    Some(dir)
}

#[test]
fn speech_encode_contract_and_determinism() {
    let Some(dir) = require_models(&["whisper_tiny_encoder.onnx"]) else {
        return;
    };
    let mut enc = AudioEncoder::new(&dir).expect("AudioEncoder::new");

    let pcm = vec![0.0f32; WHISPER_SAMPLES];
    let ts = Timestamp::now();
    let (emb, hidden) = enc.encode_speech(&pcm, ts).expect("encode_speech");

    assert_eq!(emb.source, AudioEncoderSource::Speech);
    assert_eq!(emb.values.len(), 384, "speech embedding must be 384-dim");
    assert_eq!(hidden.len(), 1500 * 384, "hidden states for ASR decoder");
    // Mean-pool invariant: embedding equals the mean of the hidden rows.
    for d in 0..384 {
        let col_mean: f32 = (0..1500).map(|t| hidden[t * 384 + d]).sum::<f32>() / 1500.0;
        assert!(
            (emb.values[d] - col_mean).abs() < 1e-4,
            "dim {d}: {} vs mean {col_mean}",
            emb.values[d]
        );
    }

    // Determinism: same input must give bit-identical output (frozen encoder).
    let (emb2, hidden2) = enc
        .encode_speech(&pcm, Timestamp::now())
        .expect("encode_speech rerun");
    assert_eq!(
        emb.values, emb2.values,
        "speech embedding not deterministic"
    );
    assert_eq!(hidden, hidden2, "hidden states not deterministic");
}

#[test]
fn env_encode_contract_and_determinism() {
    // AudioEncoder::new loads both the whisper and the env encoder.
    let Some(dir) = require_models(&["whisper_tiny_encoder.onnx", ENV_ONNX_FILE]) else {
        return;
    };
    let mut enc = AudioEncoder::new(&dir).expect("AudioEncoder::new");

    let pcm = vec![0.0f32; ENV_SAMPLES];
    let emb = enc.encode_env(&pcm, Timestamp::now()).expect("encode_env");
    assert_eq!(emb.source, AudioEncoderSource::Env);
    assert_eq!(emb.values.len(), ENV_DIM, "env embedding dim mismatch");

    let emb2 = enc
        .encode_env(&pcm, Timestamp::now())
        .expect("encode_env rerun");
    assert_eq!(emb.values, emb2.values, "env embedding not deterministic");
}

#[test]
fn video_encode_contract_and_determinism() {
    // VideoEncoder prefers dinov3_small.onnx and falls back to dinov2_small.onnx;
    // need at least one.
    let dir = models_dir();
    let have = dir.join("dinov3_small.onnx").exists() || dir.join("dinov2_small.onnx").exists();
    if !have {
        eprintln!("skipping: neither dinov3_small.onnx nor dinov2_small.onnx present");
        return;
    }
    let mut enc = VideoEncoder::new(&dir).expect("VideoEncoder::new");

    let frame = VideoFrame {
        data: vec![0u8; 320 * 240 * 3],
        width: 320,
        height: 240,
        format: PixelFormat::Rgb8,
        timestamp: Timestamp::now(),
    };
    let emb = enc.encode(&frame).expect("video encode");
    assert_eq!(emb.values.len(), 384, "video embedding must be 384-dim");
    // A black frame must not encode to an all-zero vector (DINOv2/v3 have bias
    // terms; an all-zero output would indicate a broken pooling slice).
    assert!(emb.values.iter().any(|&v| v != 0.0), "all-zero embedding");

    let emb2 = enc.encode(&frame).expect("video encode rerun");
    assert_eq!(emb.values, emb2.values, "video embedding not deterministic");
}

/// CPU-EP vs MIGraphX-EP video encoder: embeddings must agree (cosine) and timings are printed.
/// `cargo test -p somnus-encoding --release --test encode_models video_cpu_vs_gpu -- --ignored --nocapture`
#[test]
#[ignore]
fn video_cpu_vs_gpu_equivalence_and_timing() {
    let dir = models_dir();
    if !dir.join("dinov3_small.onnx").exists() && !dir.join("dinov2_small.onnx").exists() {
        eprintln!("skipping: no video model");
        return;
    }
    let mut gpu = VideoEncoder::with_provider(&dir, false).expect("gpu enc");
    let mut cpu = VideoEncoder::with_provider(&dir, true).expect("cpu enc");
    let (w, h) = (320usize, 240usize);
    let frames: Vec<VideoFrame> = (0..16)
        .map(|s| VideoFrame {
            data: (0..w * h * 3)
                .map(|i| ((i * 7 + s * 31) % 251) as u8)
                .collect(),
            width: w as u32,
            height: h as u32,
            format: PixelFormat::Rgb8,
            timestamp: Timestamp::now(),
        })
        .collect();
    let time = |enc: &mut VideoEncoder| {
        let _ = enc.encode(&frames[0]).unwrap(); // warm-up (MIGraphX compiles here)
        let t = std::time::Instant::now();
        let out: Vec<_> = frames
            .iter()
            .map(|f| enc.encode(f).unwrap().values)
            .collect();
        (out, t.elapsed().as_secs_f64() * 1e3 / frames.len() as f64)
    };
    let (g, g_ms) = time(&mut gpu);
    let (c, c_ms) = time(&mut cpu);
    let cos = |a: &[f32], b: &[f32]| {
        let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        d / (a.iter().map(|x| x * x).sum::<f32>().sqrt()
            * b.iter().map(|x| x * x).sum::<f32>().sqrt())
    };
    let min_cos = g
        .iter()
        .zip(&c)
        .map(|(a, b)| cos(a, b))
        .fold(1.0f32, f32::min);
    println!("VIDEO_EP gpu {g_ms:.1} ms/frame, cpu {c_ms:.1} ms/frame, min cosine {min_cos:.6}");
    assert!(
        min_cos > 0.999,
        "CPU and GPU embeddings diverge: min cosine {min_cos}"
    );
}
