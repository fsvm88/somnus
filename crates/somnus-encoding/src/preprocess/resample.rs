#![allow(dead_code)] // SincResampler used by binary, not the lib itself

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

pub struct SincResampler {
    inner: Async<f32>,
}

impl SincResampler {
    pub fn new(in_rate: u32, out_rate: u32, chunk_frames: usize) -> anyhow::Result<Self> {
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
        Ok(Self { inner })
    }

    /// Resample a mono f32 slice. Input length must equal the chunk_frames used
    /// at construction time.
    pub fn process(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        let in_buf = InterleavedSlice::new(input, 1, input.len())
            .map_err(|e| anyhow::anyhow!("rubato in_buf: {e}"))?;
        let out = self.inner.process(&in_buf, None)?;
        Ok(out.take_data())
    }
}
