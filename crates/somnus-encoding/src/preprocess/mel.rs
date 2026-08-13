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

/// Symmetric (`periodic=False`) Hann window, divisor `n-1` — matches
/// `torch.hann_window(n, periodic=False)`, used by EfficientAT's `AugmentMelSTFT`.
fn hann_window_symmetric_f64(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

/// Zero-pad a window symmetrically to `target_len` — matches PyTorch's `torch.stft`
/// behavior when `win_length < n_fft` (window centered within the FFT frame).
fn center_pad_window(window: &[f64], target_len: usize) -> Vec<f64> {
    let total_pad = target_len - window.len();
    let pad_left = total_pad / 2;
    let pad_right = total_pad - pad_left;
    let mut out = vec![0.0f64; pad_left];
    out.extend_from_slice(window);
    out.extend(vec![0.0f64; pad_right]);
    out
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
fn power_stft_f64(
    signal: &[f32],
    n_fft: usize,
    hop_length: usize,
    window: &[f64],
) -> Vec<Vec<f64>> {
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
        let f_low = mel_pts[m];
        let f_peak = mel_pts[m + 1];
        let f_high = mel_pts[m + 2];

        let norm_low = if slaney { 2.0 / (f_high - f_low) } else { 1.0 };
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

/// Hz → mel (Kaldi scale, `1127*ln(1+f/700)`) — used by `torchaudio.compliance.kaldi`
/// (and thus EfficientAT's `AugmentMelSTFT`). Uses the literal constant `1127.0`
/// rather than the exact `2595/ln(10)` (≈1126.97) used by `hz_to_mel_htk`.
fn hz_to_mel_kaldi(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Kaldi-style mel filterbank (`torchaudio.compliance.kaldi.get_mel_banks`,
/// `vtln_warp_factor=1.0`), shape `(n_mels, n_fft/2+1)`.
///
/// Structurally different from [`mel_filterbank_f64`]'s HTK/Slaney branches: those
/// warp the triangle *vertices* to Hz and interpolate linearly in Hz-space (the
/// librosa `htk=True` convention). Kaldi instead keeps left/center/right edges in
/// **mel space** and compares each FFT bin's own mel value against them directly —
/// a different filter shape despite sharing a mel scale, so it cannot reuse that
/// function's Hz-domain interpolation.
fn kaldi_mel_filterbank_f64(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;
    let mel_low = hz_to_mel_kaldi(fmin);
    let mel_high = hz_to_mel_kaldi(fmax);
    let mel_delta = (mel_high - mel_low) / (n_mels + 1) as f64;
    let fft_bin_width = sample_rate as f64 / n_fft as f64;

    let mut bank = vec![vec![0.0f64; n_bins]; n_mels];
    for (m, row) in bank.iter_mut().enumerate() {
        let left_mel = mel_low + m as f64 * mel_delta;
        let center_mel = mel_low + (m as f64 + 1.0) * mel_delta;
        let right_mel = mel_low + (m as f64 + 2.0) * mel_delta;

        for (k, cell) in row.iter_mut().enumerate() {
            let mel = hz_to_mel_kaldi(fft_bin_width * k as f64);
            let up_slope = (mel - left_mel) / (center_mel - left_mel);
            let down_slope = (right_mel - mel) / (right_mel - center_mel);
            *cell = up_slope.min(down_slope).max(0.0);
        }
    }
    bank
}

// ── Public mel structs ────────────────────────────────────────────────────────

pub struct WhisperMel {
    window: Vec<f64>,
    filterbank: Vec<Vec<f64>>, // (80, 201)
}

impl Default for WhisperMel {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperMel {
    pub fn new() -> Self {
        let window = hann_window_f64(400);
        let filterbank = mel_filterbank_f64(16_000, 400, 80, 0.0, 8_000.0, true);
        Self { window, filterbank }
    }

    /// Compute Whisper log-mel spectrogram from 480 000 mono f32 samples @ 16 kHz.
    /// Returns shape `(80, 3000)` flattened row-major.
    pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
        let frames = power_stft_f64(pcm, 400, 160, &self.window);
        let n_frames = frames.len();
        let n_mels = 80usize;

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
        out_f64
            .iter_mut()
            .for_each(|v| *v = v.max(global_max - 8.0));

        // Normalise: (x + 4) / 4, convert to f32
        out_f64.iter().map(|&v| ((v + 4.0) / 4.0) as f32).collect()
    }
}

pub struct ClapMel {
    window: Vec<f64>,
    filterbank: Vec<Vec<f64>>, // (64, 513)
}

impl Default for ClapMel {
    fn default() -> Self {
        Self::new()
    }
}

impl ClapMel {
    pub fn new() -> Self {
        let window = hann_window_f64(1024);
        // rand_trunc mode: librosa/Slaney mel filters with fmin=50 Hz.
        let filterbank = mel_filterbank_f64(48_000, 1024, 64, 50.0, 14_000.0, true);
        Self { window, filterbank }
    }

    /// Compute CLAP log-mel from 480 000 mono f32 samples @ 48 kHz.
    /// Returns shape `(1001, 64)` flattened row-major.
    pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
        let frames = power_stft_f64(pcm, 1024, 480, &self.window);
        let n_frames = frames.len();
        let n_mels = 64usize;

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
        mel.iter_mut()
            .for_each(|v| *v = 10.0 * v.max(1e-10).log10());

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

/// Difference pre-emphasis matching `AugmentMelSTFT`'s `conv1d(x, [-0.97, 1])`:
/// `out[t] = x[t+1] - 0.97 * x[t]`, length `n-1` (PyTorch conv1d, no padding).
fn preemphasis(pcm: &[f32]) -> Vec<f32> {
    pcm.windows(2).map(|w| w[1] - 0.97 * w[0]).collect()
}

/// EfficientAT (`fschmid56/EfficientAT`) mn10 mel frontend — `AugmentMelSTFT` defaults
/// (n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, fmin=0, fmax=15000),
/// with pre-emphasis and Kaldi-style mel banks (see [`kaldi_mel_filterbank_f64`]).
pub struct Mn10Mel {
    window: Vec<f64>,
    filterbank: Vec<Vec<f64>>, // (128, 513)
}

impl Default for Mn10Mel {
    fn default() -> Self {
        Self::new()
    }
}

impl Mn10Mel {
    pub fn new() -> Self {
        let window = center_pad_window(&hann_window_symmetric_f64(800), 1024);
        let filterbank = kaldi_mel_filterbank_f64(32_000, 1024, 128, 0.0, 15_000.0);
        Self { window, filterbank }
    }

    /// Compute EfficientAT log-mel from 320 000 mono f32 samples @ 32 kHz.
    /// Returns shape `(128, 1000)` flattened row-major (mel-bin major, matching
    /// the backbone's expected `(1, 1, 128, 1000)` input layout — no transpose).
    pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
        let pre = preemphasis(pcm);
        let frames = power_stft_f64(&pre, 1024, 320, &self.window);
        let n_frames = frames.len();
        let n_mels = 128usize;

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

        // Natural log with epsilon (not power_to_db/log10), then fixed normalization.
        mel.iter_mut().for_each(|v| *v = (*v + 0.00001).ln());
        mel.iter().map(|&v| ((v + 4.5) / 5.0) as f32).collect()
    }
}

// ── tests ────────────────────────────────────────────────────────────────────
//
// End-to-end parity against HuggingFace fixtures lives in tests/mel_parity.rs;
// these unit tests pin the individual pieces so a parity failure localises.

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // ── windows ──────────────────────────────────────────────────────────────

    #[test]
    fn hann_periodic_endpoints_and_symmetry() {
        let n = 400; // WhisperMel's window
        let w = hann_window_f64(n);
        assert_eq!(w.len(), n);
        assert!(approx_eq(w[0], 0.0, 1e-12), "w[0]={}", w[0]);
        // Periodic Hann: w[i] = w[n-i], and w[n/2] = 1 exactly.
        assert!(approx_eq(w[n / 2], 1.0, 1e-12), "w[n/2]={}", w[n / 2]);
        for i in 1..n / 2 {
            assert!(
                approx_eq(w[i], w[n - i], 1e-12),
                "asymmetry at {i}: {} vs {}",
                w[i],
                w[n - i]
            );
        }
        // Periodic, not symmetric: w[0] == 0 but the window never returns to 0 at i=n-1+1.
        assert!(w[n - 1] > 0.0);
    }

    #[test]
    fn hann_symmetric_endpoints_and_peak() {
        let n = 800; // Mn10Mel's win_length
        let w = hann_window_symmetric_f64(n);
        assert_eq!(w.len(), n);
        assert!(approx_eq(w[0], 0.0, 1e-12));
        assert!(approx_eq(w[n - 1], 0.0, 1e-12));
        // Even n: the peak sits between the two middle samples, so they are equal
        // and just under 1.0 (odd n would hit 1.0 exactly).
        assert!(approx_eq(w[n / 2 - 1], w[n / 2], 1e-12));
        assert!(
            w[n / 2] < 1.0 && approx_eq(w[n / 2], 1.0, 1e-4),
            "mid={}",
            w[n / 2]
        );
        for i in 1..n / 2 {
            assert!(approx_eq(w[i], w[n - 1 - i], 1e-12), "asymmetry at {i}");
        }
    }

    #[test]
    fn center_pad_window_even_and_odd_padding() {
        let even = center_pad_window(&[1.0, 2.0], 4);
        assert_eq!(even, [0.0, 1.0, 2.0, 0.0]);
        // Odd total padding: floor on the left, extra sample on the right
        // (matches torch.stft's win_length < n_fft padding).
        let odd = center_pad_window(&[1.0, 2.0, 3.0], 6);
        assert_eq!(odd, [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
        // The real Mn10Mel case: 800 → 1024, symmetric pad 112/112.
        let w = center_pad_window(&hann_window_symmetric_f64(800), 1024);
        assert_eq!(w.len(), 1024);
        assert!(w[..112].iter().all(|&v| v == 0.0));
        assert!(w[912..].iter().all(|&v| v == 0.0));
        assert!(approx_eq(w[112], 0.0, 1e-12)); // window start
        assert!(approx_eq(w[112 + 400], 1.0, 1e-4)); // window mid (even length: just under 1)
    }

    // ── reflect padding ──────────────────────────────────────────────────────

    #[test]
    fn reflect_pad_matches_numpy_reflect() {
        // numpy.pad([1,2,3,4,5], 2, mode='reflect') == [3,2,1,2,3,4,5,4,3]
        let out = reflect_pad(&[1.0, 2.0, 3.0, 4.0, 5.0], 2);
        assert_eq!(out, [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn reflect_pad_zero_is_identity() {
        let sig = [0.5f32, -1.0, 0.25];
        assert_eq!(reflect_pad(&sig, 0), sig.to_vec());
    }

    // ── mel scales ───────────────────────────────────────────────────────────

    #[test]
    fn htk_scale_roundtrip_and_anchor() {
        for hz in [0.0, 100.0, 700.0, 1000.0, 4000.0, 8000.0, 16000.0] {
            let m = hz_to_mel_htk(hz);
            let back = mel_to_hz_htk(m);
            assert!(
                approx_eq(back, hz, 1e-9 * hz.max(1.0)),
                "htk roundtrip at {hz}: {back}"
            );
        }
        // HTK anchor: 1000 Hz ≈ 1000 mel.
        assert!(approx_eq(hz_to_mel_htk(1000.0), 1000.0, 0.1));
    }

    #[test]
    fn slaney_scale_roundtrip_and_continuity() {
        // Roundtrip on both the linear branch (< 1000 Hz) and the log branch.
        for hz in [0.0, 200.0, 666.666, 999.0, 1000.0, 2500.0, 8000.0, 24000.0] {
            let m = hz_to_mel_slaney(hz);
            let back = mel_to_hz_slaney(m);
            assert!(
                approx_eq(back, hz, 1e-9 * hz.max(1.0)),
                "slaney roundtrip at {hz}: {back}"
            );
        }
        // The two branches meet at exactly 15 mel / 1000 Hz.
        assert!(approx_eq(hz_to_mel_slaney(1000.0), 15.0, 1e-12));
        assert!(approx_eq(hz_to_mel_slaney(999.999), 15.0, 1e-3));
        assert!(approx_eq(mel_to_hz_slaney(15.0), 1000.0, 1e-9));
    }

    #[test]
    fn kaldi_scale_anchors() {
        assert!(approx_eq(hz_to_mel_kaldi(0.0), 0.0, 1e-12));
        // 1127·ln(2) at 700 Hz; the literal 1127 constant (not 2595/ln(10) ≈ 1126.97)
        // is the point — compare against the exact constant to pin it.
        assert!(approx_eq(
            hz_to_mel_kaldi(700.0),
            1127.0 * 2.0_f64.ln(),
            1e-12
        ));
        assert!(hz_to_mel_kaldi(8000.0) > hz_to_mel_kaldi(1000.0));
    }

    // ── pre-emphasis ─────────────────────────────────────────────────────────

    #[test]
    fn preemphasis_exact_values() {
        // out[t] = x[t+1] - 0.97·x[t], PyTorch conv1d([-0.97, 1]) with no padding.
        let out = preemphasis(&[1.0, 2.0, 3.0]);
        assert_eq!(out.len(), 2);
        assert!((out[0] - (2.0 - 0.97)).abs() < 1e-6);
        assert!((out[1] - (3.0 - 0.97 * 2.0)).abs() < 1e-6);
        assert!(preemphasis(&[1.0]).is_empty());
        assert!(preemphasis(&[]).is_empty());
    }

    // ── filterbanks ──────────────────────────────────────────────────────────

    /// Structural properties shared by both filterbank builders: triangular
    /// (unimodal) rows, non-negative, ordered centre frequencies, support
    /// bounded by [fmin, fmax].
    fn assert_bank_shape(bank: &[Vec<f64>], n_mels: usize, n_bins: usize, sr: u32, n_fft: usize) {
        assert_eq!(bank.len(), n_mels);
        let mut prev_argmax = None;
        for (m, row) in bank.iter().enumerate() {
            assert_eq!(row.len(), n_bins, "row {m} width");
            assert!(row.iter().all(|&v| v >= 0.0), "row {m} negative");
            let peak = row.iter().cloned().fold(f64::MIN, f64::max);
            if peak <= 0.0 {
                continue; // degenerate row (can happen for tiny test configs)
            }
            // Unimodal: non-decreasing up to the argmax, non-increasing after.
            let argmax = row.iter().position(|&v| v == peak).unwrap();
            for k in 1..=argmax {
                assert!(row[k] >= row[k - 1] - 1e-12, "row {m} not rising at {k}");
            }
            for k in argmax + 1..n_bins {
                assert!(row[k] <= row[k - 1] + 1e-12, "row {m} not falling at {k}");
            }
            if let Some(prev) = prev_argmax {
                assert!(argmax >= prev, "centre frequencies not ordered at row {m}");
            }
            prev_argmax = Some(argmax);
            // Support stays inside [fmin, fmax]: bin freqs outside must be zero.
            for (k, &v) in row.iter().enumerate() {
                let hz = k as f64 * sr as f64 / n_fft as f64;
                if v > 0.0 {
                    assert!(hz > 0.0 || m == 0 || v == 0.0, "row {m} nonzero at DC");
                }
            }
        }
    }

    #[test]
    fn mel_filterbank_htk_exact_small_case() {
        // Hand-checkable config: sr=8, n_fft=8 → bins at 0,1,2,3,4 Hz; one mel
        // filter spanning 0–4 Hz. Triangle vertices from the (roundtrip-tested)
        // HTK scale; this pins the interpolation loop itself.
        let bank = mel_filterbank_f64(8, 8, 1, 0.0, 4.0, false);
        assert_eq!(bank.len(), 1);
        let row = &bank[0];
        assert_eq!(row.len(), 5);

        let mel_max = hz_to_mel_htk(4.0);
        let f_peak = mel_to_hz_htk(mel_max / 2.0);
        let expected = [
            0.0,
            1.0 / f_peak,
            (4.0 - 2.0) / (4.0 - f_peak),
            (4.0 - 3.0) / (4.0 - f_peak),
            0.0,
        ];
        for (k, &e) in expected.iter().enumerate() {
            assert!(
                approx_eq(row[k], e, 1e-12),
                "bin {k}: {} vs expected {e}",
                row[k]
            );
        }
        // HTK branch: unit peak height (no area normalisation).
        assert!(row.iter().cloned().fold(0.0f64, f64::max) <= 1.0 + 1e-12);
    }

    #[test]
    fn mel_filterbank_slaney_structure() {
        // Whisper config: 80 mels, 16 kHz, n_fft 400.
        let bank = mel_filterbank_f64(16_000, 400, 80, 0.0, 8_000.0, true);
        assert_bank_shape(&bank, 80, 201, 16_000, 400);
        // Slaney area normalisation makes peak heights shrink with frequency
        // (filters get wider in Hz); check first-row peak > last-row peak.
        let peak = |row: &Vec<f64>| row.iter().cloned().fold(0.0f64, f64::max);
        assert!(peak(&bank[0]) > peak(&bank[79]));
    }

    #[test]
    fn kaldi_mel_filterbank_structure_and_difference() {
        // Mn10 config: 128 mels, 32 kHz, n_fft 1024.
        let kaldi = kaldi_mel_filterbank_f64(32_000, 1024, 128, 0.0, 15_000.0);
        assert_bank_shape(&kaldi, 128, 513, 32_000, 1024);
        // Kaldi triangles peak at ≤ 1 (min of the two slopes).
        for row in &kaldi {
            assert!(row.iter().cloned().fold(0.0f64, f64::max) <= 1.0 + 1e-12);
        }
        // Pin the doc claim: Kaldi banks are NOT the same filter shapes as the
        // Hz-interpolated HTK branch at identical parameters.
        let htk = mel_filterbank_f64(32_000, 1024, 128, 0.0, 15_000.0, false);
        let differ = kaldi
            .iter()
            .zip(htk.iter())
            .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
        assert!(differ, "kaldi and htk banks unexpectedly identical");
    }

    // ── power STFT ───────────────────────────────────────────────────────────

    #[test]
    fn power_stft_frame_count() {
        let sig = vec![0.0f32; 1000];
        let win = vec![1.0f64; 64];
        let frames = power_stft_f64(&sig, 64, 100, &win);
        assert_eq!(frames.len(), 1 + 1000 / 100);
        assert!(frames.iter().all(|f| f.len() == 64 / 2 + 1));
    }

    #[test]
    fn power_stft_dc_concentrates_in_bin_zero() {
        // Rectangular window + constant signal: every interior frame is a pure
        // DC block, whose unnormalised FFT is exactly n_fft at bin 0, 0 elsewhere.
        let sig = vec![0.75f32; 1024];
        let win = vec![1.0f64; 64];
        let frames = power_stft_f64(&sig, 64, 64, &win);
        let mid = &frames[frames.len() / 2];
        let expect = (0.75 * 64.0) * (0.75 * 64.0);
        assert!(
            approx_eq(mid[0], expect, 1e-9),
            "bin0={} expect {expect}",
            mid[0]
        );
        for (k, &p) in mid.iter().enumerate().skip(1) {
            assert!(p < 1e-18 * expect, "bin {k} power {p} not ~0");
        }
    }

    #[test]
    fn power_stft_tone_lands_on_its_bin() {
        // Cosine at exactly bin 5 (5 Hz with sr=64, n_fft=64): power (A·N/2)² at bin 5.
        let n_fft = 64;
        let sr = 64.0;
        let sig: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / sr).cos() as f32)
            .collect();
        let win = vec![1.0f64; n_fft];
        let frames = power_stft_f64(&sig, n_fft, 64, &win);
        let mid = &frames[frames.len() / 2];
        // f32 input rounding limits absolute precision (~1e-5 of the peak power).
        let expect = (n_fft as f64 / 2.0) * (n_fft as f64 / 2.0);
        assert!(
            approx_eq(mid[5], expect, 1e-2),
            "bin5={} expect {expect}",
            mid[5]
        );
        for (k, &p) in mid.iter().enumerate() {
            if k != 5 {
                assert!(p < 1e-12 * expect, "bin {k} power {p} not ~0");
            }
        }
    }

    // ── public mel structs ───────────────────────────────────────────────────

    #[test]
    fn whisper_mel_silence_is_constant() {
        let mel = WhisperMel::new();
        let out = mel.compute(&vec![0.0f32; 480_000]);
        assert_eq!(out.len(), 80 * 3000);
        // Silence: power floored to 1e-10 → log10 = -10 → clamped (max-8) stays -10
        // → normalised (-10 + 4) / 4 = -1.5 everywhere.
        assert!(
            out.iter().all(|&v| v == -1.5),
            "non-constant silence output"
        );
    }

    #[test]
    fn whisper_mel_dynamic_range_clamped() {
        let mel = WhisperMel::new();
        let pcm: Vec<f32> = (0..480_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let out = mel.compute(&pcm);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &v in &out {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        // The max(log, global_max - 8) clamp bounds the range to 8 log10 units
        // = 2.0 after the (x + 4) / 4 normalisation.
        assert!(hi - lo <= 2.0 + 1e-5, "range {} exceeds clamp", hi - lo);
    }

    #[test]
    fn clap_mel_silence_is_constant() {
        let mel = ClapMel::new();
        let out = mel.compute(&vec![0.0f32; 480_000]);
        assert_eq!(out.len(), 1001 * 64);
        // Silence: 10·log10(1e-10) = -100 dB everywhere.
        assert!(
            out.iter().all(|&v| v == -100.0),
            "non-constant silence output"
        );
    }

    #[test]
    fn mn10_mel_silence_is_constant() {
        let mel = Mn10Mel::new();
        let out = mel.compute(&vec![0.0f32; 320_000]);
        assert_eq!(out.len(), 128 * 1000);
        // Silence: ln(0 + 1e-5) normalised by (x + 4.5) / 5.
        let expect = ((1e-5_f64.ln()) + 4.5) / 5.0;
        assert!(
            out.iter().all(|&v| (v as f64 - expect).abs() < 1e-6),
            "non-constant silence output"
        );
    }
}
