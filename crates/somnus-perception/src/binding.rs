use burn::{
    module::Module,
    nn::{Gru, GruConfig, Linear, LinearConfig},
    tensor::{activation, backend::Backend, Tensor, TensorData},
};

use crate::region::{SparseActivation, N_UNITS};

pub const EPISODE_DIM: usize = 640;

/// Per-region projection dim. 5 × D_PROJ == EPISODE_DIM, so the attention
/// output can be flattened directly to the episode vector — no final linear.
const D_PROJ: usize = 128;
const _: () = assert!(5 * D_PROJ == EPISODE_DIM);

/// Number of attention heads. Each head attends over D_HEAD = D_PROJ / N_HEADS features.
const N_HEADS: usize = 4;
const D_HEAD: usize = D_PROJ / N_HEADS;
const _CHECK_HEADS: () = assert!(N_HEADS * D_HEAD == D_PROJ);

/// GRU hidden size for temporal context. Produces a GRU_HIDDEN-dim summary of
/// the recent episode sequence, projected to EPISODE_DIM and reshaped to
/// [5, D_PROJ] for K/V augmentation — same slot layout as the old WM mean.
const GRU_HIDDEN: usize = 256;

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
#[derive(Module, Debug)]
pub struct BindingArea<B: Backend> {
    /// Per-region input projections: N_UNITS → D_PROJ (5 separate fields for derive compat).
    pub(crate) region_0: Linear<B>,
    pub(crate) region_1: Linear<B>,
    pub(crate) region_2: Linear<B>,
    pub(crate) region_3: Linear<B>,
    pub(crate) region_4: Linear<B>,
    pub(crate) w_q: Linear<B>,
    pub(crate) w_k: Linear<B>,
    pub(crate) w_v: Linear<B>,
    /// GRU over recent episode sequence → temporal context token for K/V.
    /// Replaces the old WM mean-vector approach with a learned recurrence.
    pub(crate) temporal_gru: Gru<B>,
    /// Projects GRU final hidden state [GRU_HIDDEN] → [EPISODE_DIM] so it can
    /// be reshaped to [5, D_PROJ] and appended to K and V.
    pub(crate) temporal_proj: Linear<B>,
}

impl<B: Backend> BindingArea<B> {
    pub fn new(device: &B::Device) -> Self {
        let proj = || {
            LinearConfig::new(N_UNITS, D_PROJ)
                .with_bias(false)
                .init(device)
        };
        let attn = || {
            LinearConfig::new(D_PROJ, D_PROJ)
                .with_bias(false)
                .init(device)
        };
        Self {
            region_0: proj(),
            region_1: proj(),
            region_2: proj(),
            region_3: proj(),
            region_4: proj(),
            w_q: attn(),
            w_k: attn(),
            w_v: attn(),
            temporal_gru: GruConfig::new(EPISODE_DIM, GRU_HIDDEN, false).init(device),
            temporal_proj: LinearConfig::new(GRU_HIDDEN, EPISODE_DIM)
                .with_bias(false)
                .init(device),
        }
    }

    /// Produce an episode vector from five sparse activations (one per region).
    /// Order: [AudioSpeech, AudioEnv, Visual, Symbolic, Language].
    /// Language may be a zero SparseActivation when no text is available.
    ///
    /// `episode_sequence`: optional slice of recent episode vectors (newest last),
    /// each EPISODE_DIM-dim. Run through the temporal GRU; the final hidden state
    /// is projected to [5, D_PROJ] and appended to K and V. This replaces the
    /// previous WM-mean approach with a learned recurrent temporal context.
    /// Pass `None` (or empty) for the first episode or during settle.
    pub fn forward(
        &self,
        acts: &[&SparseActivation; 5],
        episode_sequence: Option<&[Vec<f32>]>,
        device: &B::Device,
    ) -> EpisodeVector {
        let region_projs = [
            &self.region_0,
            &self.region_1,
            &self.region_2,
            &self.region_3,
            &self.region_4,
        ];

        // Project each region: N_UNITS → D_PROJ
        let projected: Vec<Tensor<B, 2>> = acts
            .iter()
            .zip(region_projs.iter())
            .map(|(act, proj)| {
                let dense = act.to_dense_n(N_UNITS);
                let t = Tensor::<B, 2>::from_data(TensorData::new(dense, [1, N_UNITS]), device);
                let out: Tensor<B, 2> = proj.forward(t); // [1, D_PROJ]
                out
            })
            .collect();

        // Stack current region tokens: [5, D_PROJ]
        let x: Tensor<B, 2> = Tensor::cat(projected, 0);

        // Q from current regions only: [5, D_PROJ]
        let q: Tensor<B, 2> = self.w_q.forward(x.clone());

        // K and V: current regions + optional GRU temporal context token.
        let (k, v) = if let Some(seq) = episode_sequence.filter(|s| !s.is_empty()) {
            let seq_len = seq.len();
            let flat: Vec<f32> = seq.iter().flat_map(|ep| ep.iter().copied()).collect();
            let seq_t: Tensor<B, 3> =
                Tensor::from_data(TensorData::new(flat, [1, seq_len, EPISODE_DIM]), device);
            // Run GRU; take final hidden state [1, GRU_HIDDEN]
            let gru_out: Tensor<B, 3> = self.temporal_gru.forward(seq_t, None);
            let hidden: Tensor<B, 2> = gru_out.narrow(1, seq_len - 1, 1).squeeze_dim::<2>(1);
            // Project → [1, EPISODE_DIM] → reshape → [5, D_PROJ]
            let ctx: Tensor<B, 2> = self.temporal_proj.forward(hidden);
            let ctx_tokens: Tensor<B, 2> = ctx.reshape([5, D_PROJ]);
            let x_ext: Tensor<B, 2> = Tensor::cat(vec![x, ctx_tokens], 0); // [10, D_PROJ]
            (self.w_k.forward(x_ext.clone()), self.w_v.forward(x_ext))
        } else {
            (self.w_k.forward(x.clone()), self.w_v.forward(x))
        };

        // Multi-head scaled dot-product.
        // Reshape: [seq, D_PROJ] → [seq, N_HEADS, D_HEAD] → [N_HEADS, seq, D_HEAD].
        // The feature axis is split contiguously: head h sees features [h*D_HEAD..(h+1)*D_HEAD]
        // of every token. Reshape order must be [seq, N_HEADS, D_HEAD] first (not
        // [N_HEADS, seq, D_HEAD]) to avoid interleaving tokens across heads.
        let seq_k = k.dims()[0]; // 5 without temporal context, 10 with
        let q3: Tensor<B, 3> = q.reshape([5, N_HEADS, D_HEAD]).swap_dims(0, 1);
        let k3: Tensor<B, 3> = k.reshape([seq_k, N_HEADS, D_HEAD]).swap_dims(0, 1);
        let v3: Tensor<B, 3> = v.reshape([seq_k, N_HEADS, D_HEAD]).swap_dims(0, 1);
        let scale = (D_HEAD as f32).sqrt().recip();
        // scores: [N_HEADS, 5, seq_k]
        let scores3: Tensor<B, 3> = q3.matmul(k3.swap_dims(1, 2)).mul_scalar(scale);
        let attn3: Tensor<B, 3> = activation::softmax(scores3, 2);
        let out3: Tensor<B, 3> = attn3.matmul(v3); // [N_HEADS, 5, D_HEAD]
                                                   // Merge: [N_HEADS, 5, D_HEAD] → [5, N_HEADS, D_HEAD] → [5, D_PROJ]
        let out: Tensor<B, 2> = out3.swap_dims(0, 1).reshape([5, D_PROJ]);

        // Flatten [5, D_PROJ] → [EPISODE_DIM]
        let flat: Vec<f32> = out
            .flatten::<1>(0, 1)
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        EpisodeVector(flat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::RegionId;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;

    fn make_area() -> BindingArea<NdArray> {
        let device = NdArrayDevice::Cpu;
        BindingArea::new(&device)
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
        let device = NdArrayDevice::Cpu;
        let area = make_area();
        let acts = [
            &dummy_act(RegionId::AudioSpeech),
            &dummy_act(RegionId::AudioEnv),
            &dummy_act(RegionId::Visual),
            &dummy_act(RegionId::Symbolic),
            &dummy_act(RegionId::Language),
        ];
        let ep = area.forward(&acts, None, &device);
        assert_eq!(ep.0.len(), EPISODE_DIM);
        assert!(ep.0.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_language_slot_is_finite() {
        let device = NdArrayDevice::Cpu;
        let area = make_area();
        let ep = area.forward(
            &[
                &dummy_act(RegionId::AudioSpeech),
                &dummy_act(RegionId::AudioEnv),
                &dummy_act(RegionId::Visual),
                &dummy_act(RegionId::Symbolic),
                &zero_act(RegionId::Language),
            ],
            None,
            &device,
        );
        assert_eq!(ep.0.len(), EPISODE_DIM);
        assert!(ep.0.iter().all(|v| v.is_finite()));
    }

    /// Test the temporal-context path (seq_k=10): episode_sequence is populated,
    /// GRU produces a context token, K and V become [10, D_PROJ].
    /// This is the common case in real runs (--working-memory-size 8) and exercises
    /// the multi-head reshape at a variable seq_k that differs from the 5-token case.
    #[test]
    fn output_with_temporal_context_is_finite_and_differs() {
        let device = NdArrayDevice::Cpu;
        let area = make_area();
        let acts = [
            &dummy_act(RegionId::AudioSpeech),
            &dummy_act(RegionId::AudioEnv),
            &dummy_act(RegionId::Visual),
            &dummy_act(RegionId::Symbolic),
            &zero_act(RegionId::Language),
        ];
        // 8 past episodes matching --working-memory-size 8.
        // GRU reduces them to a single context token; K/V become [5+5=10, D_PROJ].
        let past: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                (0..EPISODE_DIM)
                    .map(|j| ((i + j) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();
        let ep_ctx = area.forward(&acts, Some(&past), &device);
        let ep_no_ctx = area.forward(&acts, None, &device);
        assert_eq!(ep_ctx.0.len(), EPISODE_DIM);
        assert!(
            ep_ctx.0.iter().all(|v| v.is_finite()),
            "temporal-context output has non-finite values"
        );
        assert_ne!(
            ep_ctx.0, ep_no_ctx.0,
            "temporal context should produce a different episode vector"
        );
    }

    /// Verify head feature partitioning: activating only the first D_HEAD features
    /// of a region produces a different output than activating only the second
    /// D_HEAD block. This catches the wrong-reshape bug where features are
    /// interleaved across heads instead of contiguously partitioned.
    #[test]
    fn different_head_slices_produce_different_outputs() {
        let device = NdArrayDevice::Cpu;
        let area = make_area();

        // Build two activations that differ only in which D_HEAD feature block is active.
        let mut low_block = SparseActivation {
            region: RegionId::AudioSpeech,
            indices: (0u16..D_HEAD as u16).collect(), // features 0..D_HEAD (head-0 slice)
            values: vec![1.0f32; D_HEAD],
            n_units: N_UNITS,
        };
        let high_block = SparseActivation {
            region: RegionId::AudioSpeech,
            indices: (D_HEAD as u16..(2 * D_HEAD) as u16).collect(), // features D_HEAD..2*D_HEAD
            values: vec![1.0f32; D_HEAD],
            n_units: N_UNITS,
        };
        let dummy = dummy_act(RegionId::AudioEnv);
        let zero = zero_act(RegionId::Language);

        let ep_low = area.forward(&[&low_block, &dummy, &dummy, &dummy, &zero], None, &device);
        let ep_high = area.forward(&[&high_block, &dummy, &dummy, &dummy, &zero], None, &device);

        // With correct per-head feature partitioning the two inputs land in
        // different heads, producing different attention patterns and outputs.
        assert_ne!(
            ep_low.0, ep_high.0,
            "head-0 vs head-1 feature block must differ"
        );
        // Suppress unused-mut warning after the initialisation above.
        let _ = &mut low_block;
    }

    #[test]
    fn output_differs_with_different_inputs() {
        let device = NdArrayDevice::Cpu;
        let area = make_area();
        let zero = zero_act(RegionId::Visual);
        let active = dummy_act(RegionId::Visual);

        let ep_zero = area.forward(
            &[
                &dummy_act(RegionId::AudioSpeech),
                &dummy_act(RegionId::AudioEnv),
                &zero,
                &dummy_act(RegionId::Symbolic),
                &zero_act(RegionId::Language),
            ],
            None,
            &device,
        );
        let ep_active = area.forward(
            &[
                &dummy_act(RegionId::AudioSpeech),
                &dummy_act(RegionId::AudioEnv),
                &active,
                &dummy_act(RegionId::Symbolic),
                &zero_act(RegionId::Language),
            ],
            None,
            &device,
        );
        assert_ne!(
            ep_zero.0, ep_active.0,
            "attention should produce different outputs"
        );
    }
}
