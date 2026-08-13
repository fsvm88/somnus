#![allow(dead_code)] // SincResampler used by binary, not the lib itself

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

pub struct SincResampler {
    inner: Async<f32>,
    chunk_frames: usize,
}

impl SincResampler {
    pub fn new(in_rate: u32, out_rate: u32, chunk_frames: usize) -> anyhow::Result<Self> {
        if in_rate == 0 || out_rate == 0 {
            anyhow::bail!("invalid resample rate: in_rate={in_rate}, out_rate={out_rate}");
        }
        let ratio = out_rate as f64 / in_rate as f64;
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: Some(0.95),
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let inner = Async::<f32>::new_sinc(ratio, 2.0, &params, chunk_frames, 1, FixedAsync::Input)
            .map_err(|e| anyhow::anyhow!("rubato init: {e}"))?;
        Ok(Self {
            inner,
            chunk_frames,
        })
    }

    /// Resample a mono f32 slice. Input length must equal the chunk_frames used
    /// at construction time (enforced here — rubato 5 itself silently accepts
    /// longer input, which would mask caller chunking bugs).
    pub fn process(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        if input.len() != self.chunk_frames {
            anyhow::bail!(
                "resample input length {} != chunk_frames {}",
                input.len(),
                self.chunk_frames
            );
        }
        let in_buf = InterleavedSlice::new(input, 1, input.len())
            .map_err(|e| anyhow::anyhow!("rubato in_buf: {e}"))?;
        let out = self.inner.process(&in_buf, None)?;
        Ok(out.take_data())
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: usize = 4096;

    fn sine(freq: f32, rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    /// Best normalized cross-correlation of `sig` against `reference` over a
    /// lag range — absorbs the resampler's fixed pipeline delay.
    fn best_corr(sig: &[f32], reference: &[f32], max_lag: usize) -> f32 {
        let n = reference.len().min(sig.len().saturating_sub(max_lag));
        let mut best = f32::MIN;
        for lag in 0..=max_lag {
            let (mut s_ab, mut s_aa, mut s_bb) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..n {
                let a = sig[i + lag];
                let b = reference[i];
                s_ab += a * b;
                s_aa += a * a;
                s_bb += b * b;
            }
            let denom = (s_aa * s_bb).sqrt();
            if denom > 0.0 {
                best = best.max(s_ab / denom);
            }
        }
        best
    }

    #[test]
    fn new_valid_ratios() {
        assert!(SincResampler::new(48_000, 16_000, CHUNK).is_ok());
        assert!(SincResampler::new(44_100, 48_000, CHUNK).is_ok());
        assert!(SincResampler::new(16_000, 16_000, CHUNK).is_ok());
    }

    #[test]
    fn new_rejects_zero_rates() {
        assert!(SincResampler::new(0, 16_000, CHUNK).is_err());
        assert!(SincResampler::new(48_000, 0, CHUNK).is_err());
    }

    /// Total output count over many chunks must track the rate ratio closely —
    /// the somnus binary's accumulators (crates/somnus/src/lib.rs AudioProcessor) rely
    /// on this to hit exactly WHISPER_SAMPLES / ENV_SAMPLES.
    #[test]
    fn output_count_tracks_ratio() {
        let mut r = SincResampler::new(48_000, 16_000, CHUNK).unwrap();
        let chunk = sine(440.0, 48_000, CHUNK);
        let n_chunks = 50;
        let mut total = 0usize;
        for _ in 0..n_chunks {
            total += r.process(&chunk).unwrap().len();
        }
        let expected = n_chunks * CHUNK / 3; // 48k → 16k
        let drift = (total as f64 - expected as f64).abs();
        assert!(
            drift <= 4.0,
            "total {total} vs expected {expected} (drift {drift})"
        );
    }

    /// DC must pass through a sinc resampler unchanged (once the pipeline has
    /// filled), on both downsampling and upsampling paths.
    #[test]
    fn dc_passthrough() {
        for (in_rate, out_rate) in [(48_000u32, 16_000u32), (16_000, 48_000)] {
            let mut r = SincResampler::new(in_rate, out_rate, CHUNK).unwrap();
            let chunk = vec![0.5f32; CHUNK];
            let mut last = Vec::new();
            for _ in 0..20 {
                last = r.process(&chunk).unwrap();
            }
            assert!(!last.is_empty(), "{in_rate}→{out_rate}: no output");
            let err = last.iter().map(|v| (v - 0.5).abs()).fold(0.0f32, f32::max);
            assert!(
                err < 1e-3,
                "{in_rate}→{out_rate}: DC error {err} (last len {})",
                last.len()
            );
        }
    }

    /// A 440 Hz tone at 48 kHz must come out as a 440 Hz tone at 16 kHz —
    /// verifies the rubato 3→5 upgrade still rescales the time base correctly
    /// (the upgrade was previously verified only by a manual smoke run).
    /// Input must be one continuous signal sliced into chunks (regenerating each
    /// chunk from phase 0 would put discontinuities at every boundary).
    #[test]
    fn sine_frequency_preserved_48k_to_16k() {
        let n_chunks = 30;
        let signal = sine(440.0, 48_000, CHUNK * n_chunks);
        let mut r = SincResampler::new(48_000, 16_000, CHUNK).unwrap();
        let mut out = Vec::new();
        for c in 0..n_chunks {
            out.extend_from_slice(&r.process(&signal[c * CHUNK..(c + 1) * CHUNK]).unwrap());
        }
        // Skip the fill latency, take a steady-state window.
        let skip = 2048;
        assert!(out.len() > skip + 16_000);
        let seg: Vec<f32> = out[skip..skip + 16_000].to_vec(); // 1 s steady-state
        let reference = sine(440.0, 16_000, seg.len());
        let corr = best_corr(&seg, &reference, 128);
        assert!(corr > 0.99, "440 Hz correlation {corr} <= 0.99");
        // And it must NOT match a neighbouring frequency.
        let off = sine(480.0, 16_000, seg.len());
        let corr_off = best_corr(&seg, &off, 128);
        assert!(corr_off < 0.5, "480 Hz spuriously correlates at {corr_off}");
    }

    /// Ratio 1.0 is a real production path: media already at 16 kHz still gets a
    /// `SincResampler::new(rate, 16_000, ...)` (AudioProcessor::ingest builds one
    /// unconditionally). Output must be the input, delayed.
    #[test]
    fn identity_ratio_passthrough() {
        let n_chunks = 20;
        let signal = sine(440.0, 16_000, CHUNK * n_chunks);
        let mut r = SincResampler::new(16_000, 16_000, CHUNK).unwrap();
        let mut out = Vec::new();
        for c in 0..n_chunks {
            out.extend_from_slice(&r.process(&signal[c * CHUNK..(c + 1) * CHUNK]).unwrap());
        }
        let total_in = n_chunks * CHUNK;
        assert!(
            (out.len() as i64 - total_in as i64).abs() <= 4,
            "identity drift: {} vs {total_in}",
            out.len()
        );
        let skip = 2048;
        let seg: Vec<f32> = out[skip..skip + 16_000].to_vec();
        let reference = sine(440.0, 16_000, seg.len());
        let corr = best_corr(&seg, &reference, 128);
        assert!(corr > 0.999, "identity correlation {corr} <= 0.999");
    }

    /// The documented contract: input length must equal chunk_frames. rubato's
    /// FixedAsync::Input rejects anything else — pin that so a future rubato
    /// upgrade can't silently change the chunking AudioProcessor depends on.
    #[test]
    fn wrong_length_input_errors() {
        let mut r = SincResampler::new(48_000, 16_000, CHUNK).unwrap();
        let short = vec![0.0f32; CHUNK - 1];
        let long = vec![0.0f32; CHUNK + 1];
        assert!(r.process(&short).is_err(), "short input accepted");
        assert!(r.process(&long).is_err(), "long input accepted");
        // Resampler must still work normally after the rejected calls.
        let ok = vec![0.0f32; CHUNK];
        assert!(r.process(&ok).is_ok());
    }
}
