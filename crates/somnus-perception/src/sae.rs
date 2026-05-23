use candle_core::{Device, Module, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::{
    error::PerceptionResult,
    region::{RegionId, SparseActivation, N_UNITS},
};

/// Thin linear decoder: n_units → n_in.
///
/// In Phase 3 the weights are randomly initialised; reconstruction error is
/// observable as a metric but not meaningful until Phase 5 trains the decoder
/// via experience replay.
pub struct SaeDecoder {
    pub region: RegionId,
    pub n_in: usize,
    pub n_units: usize,
    decoder: Linear,
    device: Device,
}

impl SaeDecoder {
    pub fn new(
        region: RegionId,
        n_in: usize,
        vb: VarBuilder,
        device: Device,
    ) -> PerceptionResult<Self> {
        let decoder = linear_no_bias(N_UNITS, n_in, vb)?;
        Ok(Self {
            region,
            n_in,
            n_units: N_UNITS,
            decoder,
            device,
        })
    }

    /// MSE between the original encoder input and the decoder's reconstruction
    /// from the sparse activation (inference path — no gradient tracking).
    pub fn reconstruction_error(
        &self,
        input: &[f32],
        act: &SparseActivation,
    ) -> PerceptionResult<f32> {
        debug_assert_eq!(input.len(), self.n_in);
        let dense = act.to_dense();
        let dense_t = Tensor::from_vec(dense, (1, act.n_units), &self.device)?;
        let recon = self.decoder.forward(&dense_t)?.squeeze(0)?;
        let recon_vec: Vec<f32> = recon.to_vec1()?;
        let mse = input
            .iter()
            .zip(recon_vec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / self.n_in as f32;
        Ok(mse)
    }

    /// MSE loss as a tracked Tensor — used during sleep training.
    ///
    /// `input_vec` is the original encoder embedding.
    /// `sparse_z` is the masked linear output from the k-WTA layer (tracked tensor, shape [n_units]).
    /// Gradient flows: loss → decoder weights → sparse_z → encoder Linear weights.
    pub fn mse_loss_tensor(
        &self,
        input_vec: &[f32],
        sparse_z: &Tensor,
    ) -> PerceptionResult<Tensor> {
        debug_assert_eq!(input_vec.len(), self.n_in);
        let input_t = Tensor::from_vec(input_vec.to_vec(), (self.n_in,), &self.device)?;
        let recon = self.decoder.forward(&sparse_z.unsqueeze(0)?)?.squeeze(0)?;
        let loss = (input_t - recon)?.sqr()?.mean_all()?;
        Ok(loss)
    }

    /// Batched MSE loss for NREM replay.
    ///
    /// `input_batch`: `[B, n_in]` (target, detached).
    /// `sparse_z_batch`: `[B, n_units]` (tracked — gradients flow through this).
    /// Returns a scalar mean MSE over the whole batch × all output dims.
    pub fn mse_loss_batch_tensor(
        &self,
        input_batch: &Tensor,
        sparse_z_batch: &Tensor,
    ) -> PerceptionResult<Tensor> {
        let recon = self.decoder.forward(sparse_z_batch)?; // [B, n_in]
        let loss = (input_batch - recon)?.sqr()?.mean_all()?;
        Ok(loss)
    }

    /// Replace the internal decoder with a new Linear (used during growth).
    pub fn replace_linear(&mut self, new_decoder: Linear, new_n_units: usize) {
        self.decoder = new_decoder;
        self.n_units = new_n_units;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn make_decoder(region: RegionId, n_in: usize) -> SaeDecoder {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        SaeDecoder::new(region, n_in, vb, device).unwrap()
    }

    #[test]
    fn reconstruction_error_is_finite_and_nonneg() {
        let dec = make_decoder(RegionId::Visual, 384);
        let input = vec![0.1f32; 384];
        let act = SparseActivation {
            region: RegionId::Visual,
            indices: (0u16..20).collect(),
            values: vec![0.5f32; 20],
            n_units: N_UNITS,
        };
        let err = dec.reconstruction_error(&input, &act).unwrap();
        assert!(err.is_finite());
        assert!(err >= 0.0);
    }
}
