use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Device, Tensor, TensorData},
};

use crate::region::{RegionId, SparseActivation, N_UNITS};

/// Thin linear decoder: n_units → n_in.
///
/// Weights are randomly initialised; reconstruction error is observable as a metric
/// but not meaningful until Phase 5 trains the decoder via experience replay.
#[derive(Module, Debug)]
pub struct SaeDecoder {
    pub(crate) decoder: Linear,
    #[module(skip)]
    pub region: RegionId,
    #[module(skip)]
    pub n_in: usize,
    #[module(skip)]
    pub n_units: usize,
}

impl SaeDecoder {
    pub fn new(region: RegionId, n_in: usize, device: &Device) -> Self {
        let decoder = LinearConfig::new(N_UNITS, n_in)
            .with_bias(false)
            .init(device);
        Self {
            decoder,
            region,
            n_in,
            n_units: N_UNITS,
        }
    }

    /// MSE between the original encoder input and the decoder's reconstruction
    /// from the sparse activation (inference path — no gradient tracking).
    pub fn reconstruction_error(
        &self,
        input: &[f32],
        act: &SparseActivation,
        device: &Device,
    ) -> f32 {
        debug_assert_eq!(input.len(), self.n_in);
        let dense = act.to_dense_n(self.n_units);
        let dense_t = Tensor::<2>::from_data(TensorData::new(dense, [1, self.n_units]), device);
        let recon_t: Tensor<2> = self.decoder.forward(dense_t); // [1, n_in]
        let recon_vec: Vec<f32> = recon_t
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        input
            .iter()
            .zip(recon_vec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / self.n_in as f32
    }

    /// Decoder reconstruction of `act` as a dense Vec (same length as n_in).
    /// Inference only — no gradient tracking.
    pub fn reconstruct(&self, act: &SparseActivation, device: &Device) -> Vec<f32> {
        let dense = act.to_dense_n(self.n_units);
        let dense_t = Tensor::<2>::from_data(TensorData::new(dense, [1, self.n_units]), device);
        let recon_t: Tensor<2> = self.decoder.forward(dense_t);
        recon_t
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap()
    }

    /// Batched MSE loss for NREM replay (B=1 for sequential, B>1 for batched).
    ///
    /// `input_batch`: `[B, n_in]` (target, detached).
    /// `sparse_z_batch`: `[B, n_units]` (tracked — gradients flow through this).
    /// Returns a scalar mean MSE over the whole batch × all output dims.
    pub fn mse_loss_batch(&self, input_batch: Tensor<2>, sparse_z_batch: Tensor<2>) -> Tensor<1> {
        let recon: Tensor<2> = self.decoder.forward(sparse_z_batch); // [B, n_in]
        (input_batch - recon).powi_scalar(2).mean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn make_decoder(region: RegionId, n_in: usize) -> SaeDecoder {
        let device = Device::flex();
        SaeDecoder::new(region, n_in, &device)
    }

    #[test]
    fn reconstruction_error_is_finite_and_nonneg() {
        let device = Device::flex();
        let dec = make_decoder(RegionId::Visual, 384);
        let input = vec![0.1f32; 384];
        let act = SparseActivation {
            region: RegionId::Visual,
            indices: (0u16..20).collect(),
            values: vec![0.5f32; 20],
            n_units: N_UNITS,
        };
        let err = dec.reconstruction_error(&input, &act, &device);
        assert!(err.is_finite());
        assert!(err >= 0.0);
    }
}
