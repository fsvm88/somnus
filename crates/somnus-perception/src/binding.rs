use candle_core::{Device, Module, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::{
    error::PerceptionResult,
    region::{SparseActivation, N_UNITS},
};

pub const EPISODE_DIM: usize = 512;

/// Fixed-size episode vector produced by the binding area.
/// Always EPISODE_DIM elements (f32).
#[derive(Debug, Clone)]
pub struct EpisodeVector(pub Vec<f32>);

/// Binding area: concat four 1024-dim dense activations → linear → 512-dim.
pub struct BindingArea {
    proj: Linear,
    device: Device,
}

impl BindingArea {
    pub fn new(vb: VarBuilder, device: Device) -> PerceptionResult<Self> {
        let proj = linear_no_bias(N_UNITS * 4, EPISODE_DIM, vb)?;
        Ok(Self { proj, device })
    }

    /// Produce an episode vector from four sparse activations (one per region).
    /// The order must be [AudioSpeech, AudioEnv, Visual, Symbolic].
    pub fn forward(&self, acts: &[&SparseActivation; 4]) -> PerceptionResult<EpisodeVector> {
        let mut concat = Vec::with_capacity(N_UNITS * 4);
        for act in acts {
            concat.extend_from_slice(&act.to_dense());
        }
        let t = Tensor::from_vec(concat, (1, N_UNITS * 4), &self.device)?;
        let out = self.proj.forward(&t)?.squeeze(0)?;
        Ok(EpisodeVector(out.to_vec1()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::RegionId;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn dummy_act(region: RegionId) -> SparseActivation {
        SparseActivation {
            region,
            indices: (0u16..20).collect(),
            values: vec![1.0f32; 20],
        }
    }

    #[test]
    fn binding_output_is_episode_dim() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let area = BindingArea::new(vb, device).unwrap();
        let acts = [
            &dummy_act(RegionId::AudioSpeech),
            &dummy_act(RegionId::AudioEnv),
            &dummy_act(RegionId::Visual),
            &dummy_act(RegionId::Symbolic),
        ];
        let ep = area.forward(&acts).unwrap();
        assert_eq!(ep.0.len(), EPISODE_DIM);
        assert!(ep.0.iter().all(|v| v.is_finite()));
    }
}
