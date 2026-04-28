//! Diagnostic tests to find WHERE parity errors are maximal.

use somnus_encoding::preprocess::mel::{WhisperMel, ClapMel};

fn load_f32(name: &str) -> (Vec<usize>, Vec<f32>) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let shape: Vec<usize> = std::fs::read_to_string(dir.join(format!("{name}.shape")))
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let bytes = std::fs::read(dir.join(format!("{name}.f32"))).unwrap();
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (shape, data)
}

#[test]
fn whisper_sine440_diagnose() {
    let mel = WhisperMel::new();
    let pcm: Vec<f32> = (0..480_000)
        .map(|i| (2.0_f64 * std::f64::consts::PI * 440.0 * i as f64 / 16_000.0_f64).sin() as f32)
        .collect();
    let got = mel.compute(&pcm);

    let (shape, expected) = load_f32("whisper_sine440");
    assert_eq!(shape, [80, 3000]);

    // Find max error location
    let mut max_err = 0.0f32;
    let mut max_i = 0usize;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
            max_i = i;
        }
    }
    let mel_bin = max_i / 3000;
    let frame   = max_i % 3000;
    println!("\nMax error {max_err:.6} at mel_bin={mel_bin} frame={frame}");
    println!("  got={:.6}  expected={:.6}", got[max_i], expected[max_i]);

    // Print a few values around the max error frame
    println!("Context (mel_bin {mel_bin}, frames {}-{}):",
             frame.saturating_sub(2), (frame+2).min(2999));
    for f in frame.saturating_sub(2)..=(frame+2).min(2999) {
        let g = got[mel_bin * 3000 + f];
        let e = expected[mel_bin * 3000 + f];
        println!("  frame {f}: got={g:.6}  exp={e:.6}  diff={:.6}", (g-e).abs());
    }

    // Interior sample (frame 1500)
    let g11_1500 = got[11 * 3000 + 1500];
    let e11_1500 = expected[11 * 3000 + 1500];
    println!("\nInterior check (mel_bin=11, frame=1500): got={g11_1500:.6}  exp={e11_1500:.6}");

    let g10_1500 = got[10 * 3000 + 1500];
    let e10_1500 = expected[10 * 3000 + 1500];
    println!("Interior check (mel_bin=10, frame=1500): got={g10_1500:.6}  exp={e10_1500:.6}");
}

#[test]
fn clap_sine440_diagnose() {
    let mel = ClapMel::new();
    let pcm: Vec<f32> = (0..480_000)
        .map(|i| (2.0_f64 * std::f64::consts::PI * 440.0 * i as f64 / 48_000.0_f64).sin() as f32)
        .collect();
    let got = mel.compute(&pcm);

    let (shape, expected) = load_f32("clap_sine440");
    assert_eq!(shape, [1001, 64]);

    // Find max error location (layout: time-major [1001, 64])
    let mut max_err = 0.0f32;
    let mut max_i = 0usize;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        if err > max_err { max_err = err; max_i = i; }
    }
    let frame   = max_i / 64;
    let mel_bin = max_i % 64;
    println!("\nCLAP max error {max_err:.6} at frame={frame} mel_bin={mel_bin}");
    println!("  got={:.6}  expected={:.6}", got[max_i], expected[max_i]);

    // Print surrounding mel bins at the same frame
    println!("Mel bins around max at frame {frame}:");
    let bin_start = mel_bin.saturating_sub(3);
    let bin_end   = (mel_bin + 3).min(63);
    for b in bin_start..=bin_end {
        let g = got[frame * 64 + b];
        let e = expected[frame * 64 + b];
        println!("  bin {b}: got={g:.6}  exp={e:.6}  diff={:.6}", (g-e).abs());
    }

    // Check a mid-signal frame at bins 38-44
    let mid = 500;
    println!("\nMid-frame ({mid}) mel_bins 38-44:");
    for b in 38..=44 {
        let g = got[mid * 64 + b];
        let e = expected[mid * 64 + b];
        println!("  bin {b}: got={g:.6}  exp={e:.6}  diff={:.6}", (g-e).abs());
    }
    // Print expected range
    let exp_min = expected.iter().cloned().fold(f32::INFINITY, f32::min);
    let exp_max = expected.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let got_min = got.iter().cloned().fold(f32::INFINITY, f32::min);
    let got_max = got.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("\nExpected range: [{exp_min:.4}, {exp_max:.4}]");
    println!("Got range:      [{got_min:.4}, {got_max:.4}]");

    // Count how many cells exceed 1e-3 error
    let bad_count = got.iter().zip(expected.iter())
        .filter(|(&g, &e)| (g - e).abs() >= 1e-3).count();
    println!("Cells with error >= 1e-3: {bad_count} / {}", got.len());
}
