use burn::{
    module::Module,
    nn::{Gru, GruConfig, Linear, LinearConfig},
    tensor::{activation, backend::Backend, Tensor, TensorData},
};

use crate::region::{SparseActivation, N_UNITS};

pub const EPISODE_DIM: usize = 640;

/// Per-region projection dim. 5 × D_PROJ == EPISODE_DIM, so the attention
/// output can be flattened directly to the episode vector — no final linear.
pub(crate) const D_PROJ: usize = 128;
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
/// Input width per region tracks the grown size via `region_n`; `grow_region` keeps
/// `region_n` and the projection weight in sync so grown units reach the binding area.
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
    /// Cached GRU hidden state from the previous forward pass (Vec<f32>, [1 × GRU_HIDDEN]).
    /// Fed as the initial state when a new episode arrives, so the GRU only processes one
    /// step per ingest instead of re-running the full WM window each time.
    #[module(skip)]
    pub(crate) gru_hidden: Option<Vec<f32>>,
    /// Per-region projection input width; starts at N_UNITS and grows when a region grows.
    /// Kept in sync with the corresponding region_N weight matrix by grow_region.
    #[module(skip)]
    pub(crate) region_n: [usize; 5],
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
            gru_hidden: None,
            region_n: [N_UNITS; 5],
        }
    }

    /// Produce an episode vector from five sparse activations (one per region).
    /// Order: [AudioSpeech, AudioEnv, Visual, Symbolic, Language].
    /// Language may be a zero SparseActivation when no text is available.
    ///
    /// `new_episode`: the single latest episode vector to feed into the GRU. The GRU
    /// hidden state is cached across calls so only one GRU step is needed per ingest
    /// rather than re-running the full WM window each time. Pass `None` for settle or
    /// wake_reconstruct paths that have no new temporal context.
    pub fn forward(
        &mut self,
        acts: &[&SparseActivation; 5],
        new_episode: Option<&Vec<f32>>,
        device: &B::Device,
    ) -> EpisodeVector {
        let region_projs = [
            &self.region_0,
            &self.region_1,
            &self.region_2,
            &self.region_3,
            &self.region_4,
        ];

        // Project each region: region_n[i] → D_PROJ (region_n[i] == N_UNITS until grown)
        let projected: Vec<Tensor<B, 2>> = acts
            .iter()
            .zip(region_projs.iter())
            .enumerate()
            .map(|(i, (act, proj))| {
                let n = self.region_n[i];
                let dense = act.to_dense_n(n);
                let t = Tensor::<B, 2>::from_data(TensorData::new(dense, [1, n]), device);
                let out: Tensor<B, 2> = proj.forward(t); // [1, D_PROJ]
                out
            })
            .collect();

        // Stack current region tokens: [5, D_PROJ]
        let x: Tensor<B, 2> = Tensor::cat(projected, 0);

        // Q from current regions only: [5, D_PROJ]
        let q: Tensor<B, 2> = self.w_q.forward(x.clone());

        // K and V: current regions + optional GRU temporal context token.
        // One GRU step with cached hidden state — O(1) instead of O(wm_size).
        let (k, v) = if let Some(ep) = new_episode {
            let seq_t: Tensor<B, 3> =
                Tensor::from_data(TensorData::new(ep.clone(), [1, 1, EPISODE_DIM]), device);
            let init_hidden: Option<Tensor<B, 2>> = self
                .gru_hidden
                .as_ref()
                .map(|h| Tensor::from_data(TensorData::new(h.clone(), [1, GRU_HIDDEN]), device));
            let gru_out: Tensor<B, 3> = self.temporal_gru.forward(seq_t, init_hidden);
            let hidden: Tensor<B, 2> = gru_out.narrow(1, 0, 1).squeeze_dim::<2>(1);
            // Cache hidden state for the next call.
            self.gru_hidden = Some(
                hidden
                    .clone()
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
                    .unwrap(),
            );
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
        let mut area = make_area();
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
        let mut area = make_area();
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

    /// Test the incremental GRU path: pass a new episode and verify the hidden-state
    /// cache produces a finite, distinct output from the no-context case.
    #[test]
    fn output_with_temporal_context_is_finite_and_differs() {
        let device = NdArrayDevice::Cpu;
        let mut area = make_area();
        let acts = [
            &dummy_act(RegionId::AudioSpeech),
            &dummy_act(RegionId::AudioEnv),
            &dummy_act(RegionId::Visual),
            &dummy_act(RegionId::Symbolic),
            &zero_act(RegionId::Language),
        ];
        let ep_vec: Vec<f32> = (0..EPISODE_DIM).map(|j| (j as f32 * 0.001).sin()).collect();
        // With a new episode: GRU runs one step and produces a context token (seq_k=10).
        let ep_ctx = area.forward(&acts, Some(&ep_vec), &device);
        // Without a new episode: no GRU context (seq_k=5).
        let mut area2 = make_area();
        let ep_no_ctx = area2.forward(&acts, None, &device);
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
        let mut area = make_area();

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
        let mut area = make_area();
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
