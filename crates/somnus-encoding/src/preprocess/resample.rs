#![allow(dead_code)] // SincResampler used by binary, not the lib itself

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

pub struct SincResampler {
    inner: SincFixedIn<f32>,
    in_buf: Vec<Vec<f32>>,
    out_buf: Vec<Vec<f32>>,
}

impl SincResampler {
    pub fn new(in_rate: u32, out_rate: u32, chunk_frames: usize) -> anyhow::Result<Self> {
        let ratio = out_rate as f64 / in_rate as f64;
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let inner = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_frames, 1)?;
        let in_buf  = inner.input_buffer_allocate(true);
        let out_buf = inner.output_buffer_allocate(true);
        Ok(Self { inner, in_buf, out_buf })
    }

    /// Resample a mono f32 slice. Input length must equal the chunk_frames used
    /// at construction time.
    pub fn process(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        self.in_buf[0].clear();
        self.in_buf[0].extend_from_slice(input);
        self.inner.process_into_buffer(&self.in_buf, &mut self.out_buf, None)?;
        Ok(self.out_buf[0].clone())
    }
}
