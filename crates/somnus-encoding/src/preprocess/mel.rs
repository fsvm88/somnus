//! Mel-spectrogram computation matching HuggingFace WhisperFeatureExtractor
//! and ClapFeatureExtractor (verified against HF source in ADR 0004).
//!
//! Uses reflect-padded, center-mode STFT (matching librosa/HF defaults) with a
//! periodic Hann window (divisor N, not N-1).
//!
//! All intermediate computations use f64 (matching numpy's float64 pipeline);
//! conversion to f32 happens only at the final output.

use rustfft::{num_complex::Complex, FftPlanner};

// ── Hann window (periodic, f64) ───────────────────────────────────────────────

fn hann_window_f64(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()))
        .collect()
}

// ── Reflect padding ───────────────────────────────────────────────────────────

fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    // Mirror the first `pad` samples before and last `pad` after.
    // librosa center=True pads with pad = n_fft // 2.
    let n = signal.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(signal[i % n]);
    }
    out.extend_from_slice(signal);
    for i in 1..=pad {
        out.push(signal[n - 1 - i % (n - 1)]);
    }
    out
}

// ── Power STFT (returns f64 power) ───────────────────────────────────────────

/// Returns the power spectrogram `|STFT|²` with shape `(n_frames, n_fft/2+1)` in f64.
/// `power=2.0`, unnormalized FFT (matches librosa/HF).
fn power_stft_f64(signal: &[f32], n_fft: usize, hop_length: usize, window: &[f64]) -> Vec<Vec<f64>> {
    let padded = reflect_pad(signal, n_fft / 2);
    let n_bins = n_fft / 2 + 1;
    let n_frames = 1 + signal.len() / hop_length;

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut frames: Vec<Vec<f64>> = Vec::with_capacity(n_frames);
    let mut buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft];

    for frame in 0..n_frames {
        let start = frame * hop_length;
        for k in 0..n_fft {
            let s = padded.get(start + k).copied().unwrap_or(0.0) as f64;
            buf[k] = Complex::new(s * window[k], 0.0);
        }
        fft.process(&mut buf);
        let power: Vec<f64> = buf[..n_bins]
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();
        frames.push(power);
    }
    frames
}

// ── Mel filterbank ────────────────────────────────────────────────────────────

/// Hz → mel (HTK scale, no area normalisation).
fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Hz → mel (Slaney linear/log composite; used by Whisper/CLAP).
fn hz_to_mel_slaney(hz: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const LOGSTEP: f64 = 0.068_751_777_489_821_22; // ln(6.4) / 27
    if hz < MIN_LOG_HZ {
        (hz - F_MIN) / F_SP
    } else {
        15.0 + (hz / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    const LOGSTEP: f64 = 0.068_751_777_489_821_22; // ln(6.4) / 27
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOGSTEP).exp()
    }
}

/// Build a mel filterbank matrix in f64, shape `(n_mels, n_fft/2+1)`.
fn mel_filterbank_f64(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
    slaney: bool,
) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;
    type MelFn = fn(f64) -> f64;
    let (hz_to_mel, mel_to_hz): (MelFn, MelFn) = if slaney {
        (hz_to_mel_slaney, mel_to_hz_slaney)
    } else {
        (hz_to_mel_htk, mel_to_hz_htk)
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mel_pts: Vec<f64> = (0..=(n_mels + 1))
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    let freqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let mut bank = vec![vec![0.0f64; n_bins]; n_mels];

    for m in 0..n_mels {
        let f_low  = mel_pts[m];
        let f_peak = mel_pts[m + 1];
        let f_high = mel_pts[m + 2];

        let norm_low  = if slaney { 2.0 / (f_high - f_low) } else { 1.0 };
        let norm_high = if slaney { 2.0 / (f_high - f_low) } else { 1.0 };

        for (k, &f) in freqs.iter().enumerate() {
            if f >= f_low && f <= f_peak {
                bank[m][k] = (f - f_low) / (f_peak - f_low) * norm_low;
            } else if f > f_peak && f <= f_high {
                bank[m][k] = (f_high - f) / (f_high - f_peak) * norm_high;
            }
        }
    }
    bank
}

// ── Public mel structs ────────────────────────────────────────────────────────

pub struct WhisperMel {
    window:     Vec<f64>,
    filterbank: Vec<Vec<f64>>,   // (80, 201)
}

impl Default for WhisperMel {
    fn default() -> Self { Self::new() }
}

impl WhisperMel {
    pub fn new() -> Self {
        let window     = hann_window_f64(400);
        let filterbank = mel_filterbank_f64(16_000, 400, 80, 0.0, 8_000.0, true);
        Self { window, filterbank }
    }

    /// Compute Whisper log-mel spectrogram from 480 000 mono f32 samples @ 16 kHz.
    /// Returns shape `(80, 3000)` flattened row-major.
    pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
        let frames = power_stft_f64(pcm, 400, 160, &self.window);
        let n_frames = frames.len();
        let n_mels   = 80usize;

        // Mel filterbank application in f64 → (80, n_frames)
        let mut mel = vec![0.0f64; n_mels * n_frames];
        for (t, frame) in frames.iter().enumerate() {
            for m in 0..n_mels {
                let v: f64 = self.filterbank[m]
                    .iter()
                    .zip(frame.iter())
                    .map(|(w, p)| w * p)
                    .sum();
                mel[m * n_frames + t] = v;
            }
        }

        // Floor, log10 in f64
        mel.iter_mut().for_each(|v| *v = v.max(1e-10).log10());

        // Trim to exactly 3000 frames (drop last frame, matching HF stft[..., :-1])
        let keep = n_frames - 1;
        let mut out_f64 = vec![0.0f64; n_mels * keep];
        for m in 0..n_mels {
            let src = &mel[m * n_frames..m * n_frames + keep];
            out_f64[m * keep..(m + 1) * keep].copy_from_slice(src);
        }

        // Dynamic range: max(log_spec, global_max - 8)
        let global_max = out_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        out_f64.iter_mut().for_each(|v| *v = v.max(global_max - 8.0));

        // Normalise: (x + 4) / 4, convert to f32
        out_f64.iter().map(|&v| ((v + 4.0) / 4.0) as f32).collect()
    }
}

pub struct ClapMel {
    window:     Vec<f64>,
    filterbank: Vec<Vec<f64>>,   // (64, 513)
}

impl Default for ClapMel {
    fn default() -> Self { Self::new() }
}

impl ClapMel {
    pub fn new() -> Self {
        let window     = hann_window_f64(1024);
        // rand_trunc mode: librosa/Slaney mel filters with fmin=50 Hz.
        let filterbank = mel_filterbank_f64(48_000, 1024, 64, 50.0, 14_000.0, true);
        Self { window, filterbank }
    }

    /// Compute CLAP log-mel from 480 000 mono f32 samples @ 48 kHz.
    /// Returns shape `(1001, 64)` flattened row-major.
    pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
        let frames   = power_stft_f64(pcm, 1024, 480, &self.window);
        let n_frames = frames.len();
        let n_mels   = 64usize;

        // Mel filterbank in f64 → (64, n_frames)
        let mut mel = vec![0.0f64; n_mels * n_frames];
        for (t, frame) in frames.iter().enumerate() {
            for m in 0..n_mels {
                let v: f64 = self.filterbank[m]
                    .iter()
                    .zip(frame.iter())
                    .map(|(w, p)| w * p)
                    .sum();
                mel[m * n_frames + t] = v;
            }
        }

        // Floor, power_to_db in f64: 10 * log10(mel)
        mel.iter_mut().for_each(|v| *v = 10.0 * v.max(1e-10).log10());

        // Trim to exactly 1001 frames
        let keep = 1001usize.min(n_frames);

        // Transpose (64, n_frames) → (1001, 64) and convert to f32
        let mut out = vec![0.0f32; keep * n_mels];
        for t in 0..keep {
            for m in 0..n_mels {
                out[t * n_mels + m] = mel[m * n_frames + t] as f32;
            }
        }
        out
    }
}
