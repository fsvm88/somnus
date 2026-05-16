use candle_core::{Device, Module, Tensor};
use candle_nn::{linear_no_bias, ops::softmax, Linear, VarBuilder};

use crate::{
    error::PerceptionResult,
    region::{SparseActivation, N_UNITS},
};

pub const EPISODE_DIM: usize = 640;

/// Per-region projection dim. 5 × D_PROJ == EPISODE_DIM, so the attention
/// output can be flattened directly to the episode vector — no final linear.
const D_PROJ: usize = 128;
const _: () = assert!(5 * D_PROJ == EPISODE_DIM);

/// Fixed-size episode vector produced by the binding area.
/// Always EPISODE_DIM elements (f32).
#[derive(Debug, Clone)]
pub struct EpisodeVector(pub Vec<f32>);

/// Cross-attention binding area.
///
/// Each region's dense activation is projected from N_UNITS → D_PROJ,
/// forming a sequence of 5 tokens. Single-head scaled dot-product self-attention
/// lets every region attend to every other region. The attention output
/// [5, D_PROJ] is flattened to [640] = EPISODE_DIM without a final linear.
///
/// Input is always clamped to N_UNITS via `to_dense_n` so the binding area
/// remains fixed-size regardless of how much the regional k-WTA layers have grown.
/// Order: [AudioSpeech, AudioEnv, Visual, Symbolic, Language].
/// The Language slot is zero-padded when no text embedding is available.
pub struct BindingArea {
    /// Per-region input projection: N_UNITS → D_PROJ.
    pub(crate) region_projs: [Linear; 5],
    pub(crate) w_q: Linear,
    pub(crate) w_k: Linear,
    pub(crate) w_v: Linear,
    device: Device,
}

impl BindingArea {
    pub fn new(vb: VarBuilder, device: Device) -> PerceptionResult<Self> {
        let region_projs = [
            linear_no_bias(N_UNITS, D_PROJ, vb.pp("region_0"))?,
            linear_no_bias(N_UNITS, D_PROJ, vb.pp("region_1"))?,
            linear_no_bias(N_UNITS, D_PROJ, vb.pp("region_2"))?,
            linear_no_bias(N_UNITS, D_PROJ, vb.pp("region_3"))?,
            linear_no_bias(N_UNITS, D_PROJ, vb.pp("region_4"))?,
        ];
        let w_q = linear_no_bias(D_PROJ, D_PROJ, vb.pp("attn_q"))?;
        let w_k = linear_no_bias(D_PROJ, D_PROJ, vb.pp("attn_k"))?;
        let w_v = linear_no_bias(D_PROJ, D_PROJ, vb.pp("attn_v"))?;
        Ok(Self {
            region_projs,
            w_q,
            w_k,
            w_v,
            device,
        })
    }

    /// Produce an episode vector from five sparse activations (one per region).
    /// Order: [AudioSpeech, AudioEnv, Visual, Symbolic, Language].
    /// Language may be a zero SparseActivation when no text is available.
    pub fn forward(&self, acts: &[&SparseActivation; 5]) -> PerceptionResult<EpisodeVector> {
        // Project each region: N_UNITS → D_PROJ
        let projected: Vec<Tensor> = acts
            .iter()
            .zip(self.region_projs.iter())
            .map(|(act, proj)| -> PerceptionResult<Tensor> {
                let dense = act.to_dense_n(N_UNITS);
                let t = Tensor::from_vec(dense, (1, N_UNITS), &self.device)?;
                Ok(proj.forward(&t)?.squeeze(0)?) // [D_PROJ]
            })
            .collect::<PerceptionResult<_>>()?;

        // Stack to [5, D_PROJ]
        let x = Tensor::stack(&projected, 0)?;

        // Q, K, V projections — all [5, D_PROJ]
        let q = self.w_q.forward(&x)?;
        let k = self.w_k.forward(&x)?;
        let v = self.w_v.forward(&x)?;

        // Scaled dot-product attention
        let scale = 1.0 / (D_PROJ as f64).sqrt();
        let scores = q.matmul(&k.t()?)?.affine(scale, 0.0)?; // [5, 5]
        let attn = softmax(&scores, 1)?; // normalize over keys → [5, 5]
        let out = attn.matmul(&v)?; // [5, D_PROJ]

        // Flatten [5, D_PROJ] → [EPISODE_DIM]
        Ok(EpisodeVector(out.flatten_all()?.to_vec1()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::RegionId;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn make_area() -> BindingArea {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        BindingArea::new(vb, device).unwrap()
    }

    fn dummy_act(region: RegionId) -> SparseActivation {
        SparseActivation {
            region,
            indices: (0u16..20).collect(),
            values: vec![1.0f32; 20],
            n_units: N_UNITS,
        }
    }

    fn zero_act(region: RegionId) -> SparseActivation {
        SparseActivation {
            region,
            indices: vec![],
            values: vec![],
            n_units: N_UNITS,
        }
    }

    #[test]
    fn output_is_episode_dim() {
        let area = make_area();
        let acts = [
            &dummy_act(RegionId::AudioSpeech),
            &dummy_act(RegionId::AudioEnv),
            &dummy_act(RegionId::Visual),
            &dummy_act(RegionId::Symbolic),
            &dummy_act(RegionId::Language),
        ];
        let ep = area.forward(&acts).unwrap();
        assert_eq!(ep.0.len(), EPISODE_DIM);
        assert!(ep.0.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_language_slot_is_finite() {
        let area = make_area();
        let ep = area
            .forward(&[
                &dummy_act(RegionId::AudioSpeech),
                &dummy_act(RegionId::AudioEnv),
                &dummy_act(RegionId::Visual),
                &dummy_act(RegionId::Symbolic),
                &zero_act(RegionId::Language),
            ])
            .unwrap();
        assert_eq!(ep.0.len(), EPISODE_DIM);
        assert!(ep.0.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn output_differs_with_different_inputs() {
        let area = make_area();
        let zero = zero_act(RegionId::Visual);
        let active = dummy_act(RegionId::Visual);

        let ep_zero = area
            .forward(&[
                &dummy_act(RegionId::AudioSpeech),
                &dummy_act(RegionId::AudioEnv),
                &zero,
                &dummy_act(RegionId::Symbolic),
                &zero_act(RegionId::Language),
            ])
            .unwrap();
        let ep_active = area
            .forward(&[
                &dummy_act(RegionId::AudioSpeech),
                &dummy_act(RegionId::AudioEnv),
                &active,
                &dummy_act(RegionId::Symbolic),
                &zero_act(RegionId::Language),
            ])
            .unwrap();
        assert_ne!(ep_zero.0, ep_active.0, "attention should produce different outputs");
    }
}
