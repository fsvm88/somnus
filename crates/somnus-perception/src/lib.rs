pub mod binding;
pub mod error;
pub mod onnx_session;
pub mod region;
pub mod sae;
pub mod wake_worker;

pub use binding::{BindingArea, EpisodeVector, EPISODE_DIM};
pub use error::{PerceptionError, PerceptionResult};
pub use onnx_session::OnnxPerceptionSession;
pub use region::{KWtaLayer, RegionId, SparseActivation, K_ACTIVE, N_UNITS};
pub use sae::SaeDecoder;
pub use wake_worker::WakeReconWorker;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        Autodiff,
    },
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    optim::adaptor::OptimizerAdaptor,
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Distribution, Tensor, TensorData},
};
use std::{collections::VecDeque, time::Instant};

pub type TrainB = Autodiff<NdArray>;
type PercOptim = OptimizerAdaptor<AdamW, PerceptionModel<TrainB>, TrainB>;

/// EMA (exponential moving average) copies of the 5 output heads, used for all
/// inference paths (top-down prior generation, settling).  Updated after every
/// wake_reconstruct_step; decoupled from the autodiff training heads so that
/// noisy per-frame gradients don't immediately destabilise prior prediction.
const EMA_ALPHA: f32 = 0.99;

#[derive(Clone)]
pub(crate) struct EmaOutputHeads {
    /// EMA tensors for the weight update step (NdArray, no autodiff).
    heads: [burn::tensor::Tensor<NdArray, 2>; 5],
    /// Flat row-major copies of the EMA weights for allocation-free inference.
    /// Layout: [d_in × d_out] (burn stores weight as [d_in, d_out]).
    /// Updated by `sync_flat` after every EMA weight update.
    flat: [Vec<f32>; 5],
    /// d_out for each head: [384, 512, 384, 384, 896].
    d_out: [usize; 5],
}

impl EmaOutputHeads {
    fn from_model(model: &PerceptionModel<TrainB>) -> Self {
        let heads = [
            model.speech_head.weight.val().inner(),
            model.env_head.weight.val().inner(),
            model.visual_head.weight.val().inner(),
            model.symbolic_head.weight.val().inner(),
            model.language_head.weight.val().inner(),
        ];
        let mut s = Self {
            flat: Default::default(),
            d_out: [0; 5],
            heads,
        };
        s.sync_flat();
        s
    }

    fn update(&mut self, model: &PerceptionModel<TrainB>) {
        self.update_from_weights([
            model.speech_head.weight.val().inner(),
            model.env_head.weight.val().inner(),
            model.visual_head.weight.val().inner(),
            model.symbolic_head.weight.val().inner(),
            model.language_head.weight.val().inner(),
        ]);
    }

    /// EMA-blend a fresh set of live head weights (NdArray inner tensors).
    /// Shared by the in-thread `update` path and the async WakeReconWorker.
    pub(crate) fn update_from_weights(&mut self, live: [burn::tensor::Tensor<NdArray, 2>; 5]) {
        for (ema, w) in self.heads.iter_mut().zip(live) {
            *ema = ema.clone() * EMA_ALPHA + w * (1.0 - EMA_ALPHA);
        }
        self.sync_flat();
    }

    /// Flat EMA weights for streaming back from the worker thread.
    pub(crate) fn flat_clone(&self) -> [Vec<f32>; 5] {
        self.flat.clone()
    }

    /// Overwrite the flat inference weights with worker-streamed EMA values.
    /// The `heads` tensors are NOT refreshed: while a WakeReconWorker is active
    /// it owns the authoritative EMA state, and `update()` is never called on
    /// this copy. Restore paths rebuild from the model via `from_model`.
    pub(crate) fn set_flat(&mut self, flat: [Vec<f32>; 5]) {
        self.flat = flat;
    }

    /// Re-extract flat weight arrays from the EMA tensors.
    fn sync_flat(&mut self) {
        for (i, h) in self.heads.iter().enumerate() {
            let dims = h.dims();
            self.d_out[i] = dims[1];
            self.flat[i] = h.clone().into_data().convert::<f32>().to_vec().unwrap();
        }
    }

    /// Inference-only head forward: ep_data @ weight → Vec<f32>.
    /// Uses sgemm for AVX2/FMA-accelerated GEMV.
    fn predict(&self, ep_data: &[f32], idx: usize) -> Vec<f32> {
        let w = &self.flat[idx];
        let d_in = ep_data.len();
        let d_out = self.d_out[idx];
        let mut out = vec![0.0f32; d_out];
        // A: [1, d_in] row-major   B: [d_in, d_out] row-major   C: [1, d_out]
        unsafe {
            matrixmultiply::sgemm(
                1,
                d_in,
                d_out,
                1.0,
                ep_data.as_ptr(),
                d_in as isize,
                1,
                w.as_ptr(),
                d_out as isize,
                1,
                0.0,
                out.as_mut_ptr(),
                d_out as isize,
                1,
            );
        }
        out
    }
}

/// Per-region input dimensions. Order: [AudioSpeech, AudioEnv, Visual, Symbolic, Language].
pub const REGION_DIMS: [usize; 5] = [384, 512, 384, 384, 896];

/// Input dimension for the symbolic cross-modal projection:
/// speech(384) + env(512) + visual(384) + language(896). Missing modalities are zero-padded.
pub const CROSS_MODAL_DIM: usize = 384 + 512 + 384 + 896;

/// Default prior_cosine gate for settling (ADR 0011). Exposed as --settle-gate at runtime.
pub const SETTLE_COSINE_GATE_DEFAULT: f32 = 0.85;

/// L2 delta below which settling is considered converged (ADR 0011).
/// Calibration note (2026-05-31): smoke tests show settle_delta_mean ≈ 0.46 on unit-normalised
/// 640-dim episode vectors, so SETTLE_EPS=4.0 is trivially satisfied every first iteration.
/// train_campaign.sh uses --settle-steps 1 to match this reality. If settle_steps is ever
/// raised above 1, recalibrate SETTLE_EPS to ~0.1–0.2 to allow genuine multi-step convergence.
const SETTLE_EPS: f32 = 4.0;

pub struct TrainConfig {
    pub lr: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self { lr: 1e-3 }
    }
}

/// Outputs from a single `ingest_*` call.
pub struct PerceptionOutput {
    pub recon_errors: Vec<(RegionId, f32)>,
    pub episode: Option<EpisodeVector>,
    /// Raw encoder inputs per region (Some when episode is Some).
    pub regional_inputs: Option<[Vec<f32>; 5]>,
    /// per-region prior cosine similarities (indexed by region id 0–4).
    pub prior_cosines: [Option<f32>; 5],
    /// Settling iterations actually run for this episode (0 = gate did not fire or settle_steps=0).
    pub settle_iters: usize,
    /// L2 distance between the pre-settle and post-settle episode vectors (0.0 when settle_iters=0).
    /// Reveals whether SETTLE_EPS is a tight or trivial convergence criterion.
    pub settle_delta: f32,
}

/// All trainable parameters for the cognitive layer.
#[derive(Module, Debug)]
pub struct PerceptionModel<B: burn::tensor::backend::Backend> {
    pub speech_layer: KWtaLayer<B>,
    pub env_layer: KWtaLayer<B>,
    pub visual_layer: KWtaLayer<B>,
    pub symbolic_layer: KWtaLayer<B>,
    pub language_layer: KWtaLayer<B>,
    pub speech_decoder: SaeDecoder<B>,
    pub env_decoder: SaeDecoder<B>,
    pub visual_decoder: SaeDecoder<B>,
    pub symbolic_decoder: SaeDecoder<B>,
    pub language_decoder: SaeDecoder<B>,
    pub binding: BindingArea<B>,
    pub speech_head: Linear<B>,
    pub env_head: Linear<B>,
    pub visual_head: Linear<B>,
    pub symbolic_head: Linear<B>,
    pub language_head: Linear<B>,
    /// Projects concatenated cross-modal encoder inputs → 384-dim symbolic input.
    /// Input: [speech(384) | env(512) | visual(384) | language(896)] = 2176-dim,
    /// zero-padded for any missing modality.
    pub cross_modal_proj: Linear<B>,
}

impl PerceptionModel<TrainB> {
    pub fn new(device: &NdArrayDevice) -> Self {
        Self {
            speech_layer: KWtaLayer::new(RegionId::AudioSpeech, 384, device),
            env_layer: KWtaLayer::new(RegionId::AudioEnv, 512, device),
            visual_layer: KWtaLayer::new(RegionId::Visual, 384, device),
            symbolic_layer: KWtaLayer::new(RegionId::Symbolic, 384, device),
            language_layer: KWtaLayer::new(RegionId::Language, 896, device),
            speech_decoder: SaeDecoder::new(RegionId::AudioSpeech, 384, device),
            env_decoder: SaeDecoder::new(RegionId::AudioEnv, 512, device),
            visual_decoder: SaeDecoder::new(RegionId::Visual, 384, device),
            symbolic_decoder: SaeDecoder::new(RegionId::Symbolic, 384, device),
            language_decoder: SaeDecoder::new(RegionId::Language, 896, device),
            binding: BindingArea::new(device),
            speech_head: LinearConfig::new(EPISODE_DIM, 384)
                .with_bias(false)
                .init(device),
            env_head: LinearConfig::new(EPISODE_DIM, 512)
                .with_bias(false)
                .init(device),
            visual_head: LinearConfig::new(EPISODE_DIM, 384)
                .with_bias(false)
                .init(device),
            symbolic_head: LinearConfig::new(EPISODE_DIM, 384)
                .with_bias(false)
                .init(device),
            language_head: LinearConfig::new(EPISODE_DIM, 896)
                .with_bias(false)
                .init(device),
            cross_modal_proj: LinearConfig::new(CROSS_MODAL_DIM, 384)
                .with_bias(true)
                .init(device),
        }
    }
}

/// Owns the full cognitive-layer forward pass and sleep-time training.
/// Per-cycle timing breakdown for the wake ingest path (microseconds).
/// Reset and returned by `take_perf_breakdown`.
#[derive(Debug, Default, Clone, Copy)]
pub struct IngestPerfBreakdown {
    /// k-WTA forward + SAE decoder for the triggering region.
    pub kwta_us: u64,
    /// `compute_symbolic_cross_modal` + `refresh_language_cache`.
    pub symbolic_us: u64,
    /// `try_bind` (GRU over WM window + cross-attention).
    pub bind_us: u64,
    /// `maybe_settle_and_update` (settle loop + prior update).
    pub prior_us: u64,
    /// Time inside `settle_episode` only (subset of prior_us; zero when settle didn't fire).
    pub settle_us: u64,
    /// Time inside `update_prior_predictions` only (subset of prior_us; always fires with episode).
    pub prior_update_us: u64,
}

pub struct PerceptionProcessor {
    model: Option<PerceptionModel<TrainB>>,
    optimizer: PercOptim,
    current_lr: f64,
    device: NdArrayDevice,

    /// EMA copies of the output heads used for all inference paths (prior, settle).
    /// Decoupled from the autodiff training heads so gradient noise doesn't
    /// immediately destabilise prior prediction.
    ema_heads: EmaOutputHeads,

    prior_predictions: [Option<Vec<f32>>; 5],
    prior_beta: f32,
    latest: [Option<SparseActivation>; 5],
    latest_inputs: [Option<Vec<f32>>; 5],
    settle_steps: usize,
    settle_beta: f32,
    settle_gate: f32,
    settle_every: usize,
    settle_episode_count: usize,
    region_gain: [f32; 5],
    last_language_input: Option<Vec<f32>>,
    working_memory: VecDeque<Vec<f32>>,
    wm_size: usize,

    /// Cumulative ingest timing accumulators (μs). Reset by `take_perf_breakdown`.
    perf_kwta_us: u64,
    perf_symbolic_us: u64,
    perf_bind_us: u64,
    perf_prior_us: u64,
    perf_settle_us: u64,
    perf_prior_update_us: u64,

    /// ORT binding session (Phase 1). When Some, binding area uses ORT instead of burn.
    ort_session: Option<crate::onnx_session::OnnxPerceptionSession>,

    /// Flat [n_in × d_out] copy of cross_modal_proj weights for allocation-free inference.
    /// Row-major: flat[i * d_out + j] = weight[i][j].  Synced after each NREM weight update.
    cross_modal_proj_flat: Option<Vec<f32>>,
}

impl PerceptionProcessor {
    pub fn new(device: NdArrayDevice) -> Self {
        let model = PerceptionModel::new(&device);
        let ema_heads = EmaOutputHeads::from_model(&model);
        let optimizer = AdamWConfig::new().init::<TrainB, PerceptionModel<TrainB>>();
        Self {
            model: Some(model),
            optimizer,
            ema_heads,
            current_lr: TrainConfig::default().lr,
            device,
            prior_predictions: [None, None, None, None, None],
            prior_beta: 0.0,
            latest: [None, None, None, None, None],
            latest_inputs: [None, None, None, None, None],
            settle_steps: 0,
            settle_beta: 0.5,
            settle_gate: SETTLE_COSINE_GATE_DEFAULT,
            settle_every: 1,
            settle_episode_count: 0,
            region_gain: [1.0; 5],
            last_language_input: None,
            working_memory: VecDeque::new(),
            wm_size: 0,
            perf_kwta_us: 0,
            perf_symbolic_us: 0,
            perf_bind_us: 0,
            perf_prior_us: 0,
            perf_settle_us: 0,
            perf_prior_update_us: 0,
            ort_session: None,
            cross_modal_proj_flat: None,
        }
    }

    /// Sync the flat cross_modal_proj weight copy after model weights update (NREM).
    /// Called whenever ORT sessions are rebuilt so the inference copy stays in sync.
    pub fn sync_cross_modal_flat(&mut self) {
        let model = self.model.as_ref().unwrap();
        let w: Vec<f32> = model
            .cross_modal_proj
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        // burn stores Linear weight as [n_in, n_out] — already row-major [i][j].
        self.cross_modal_proj_flat = Some(w);
    }

    /// Return and reset the cumulative ingest timing accumulators.
    pub fn take_perf_breakdown(&mut self) -> IngestPerfBreakdown {
        let b = IngestPerfBreakdown {
            kwta_us: self.perf_kwta_us,
            symbolic_us: self.perf_symbolic_us,
            bind_us: self.perf_bind_us,
            prior_us: self.perf_prior_us,
            settle_us: self.perf_settle_us,
            prior_update_us: self.perf_prior_update_us,
        };
        self.perf_kwta_us = 0;
        self.perf_symbolic_us = 0;
        self.perf_bind_us = 0;
        self.perf_prior_us = 0;
        self.perf_settle_us = 0;
        self.perf_prior_update_us = 0;
        b
    }

    // ── Ingest ──────────────────────────────────────────────────────────────

    /// Ingest a 384-dim Whisper speech embedding.
    /// Updates AudioSpeech region, then recomputes Symbolic from all available cross-modal inputs.
    pub fn ingest_speech(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let device = self.device;
        let gained_speech = apply_gain(values, self.region_gain[0]);
        let (speech_input, speech_cos) = self.blend_with_prior(&gained_speech, 0);
        if let Some(c) = speech_cos {
            tracing::info!(region = "speech", prior_cosine = c, "top-down prior");
        }

        let t0 = Instant::now();
        let speech_err = if self.ort_session.as_ref().is_some_and(|s| s.has_encoder()) {
            // ORT encoder recomputes all 5 regions in try_bind; skip burn forward.
            // Bootstrap latest[0] if needed so try_bind gate can fire.
            if self.latest[RegionId::AudioSpeech.as_index()].is_none() {
                self.latest[RegionId::AudioSpeech.as_index()] = Some(SparseActivation {
                    region: RegionId::AudioSpeech,
                    indices: (0..K_ACTIVE as u16).collect(),
                    values: vec![1.0f32; K_ACTIVE],
                    n_units: N_UNITS,
                });
            }
            0.0
        } else {
            let act = self
                .model
                .as_mut()
                .unwrap()
                .speech_layer
                .forward(&speech_input, &device);
            let err = self
                .model
                .as_ref()
                .unwrap()
                .speech_decoder
                .reconstruction_error(values, &act, &device);
            self.latest[RegionId::AudioSpeech.as_index()] = Some(act);
            err
        };
        self.perf_kwta_us += t0.elapsed().as_micros() as u64;
        self.latest_inputs[RegionId::AudioSpeech.as_index()] = Some(values.to_vec());

        let t1 = Instant::now();
        let (symbolic_cos, symbolic_err) = self.compute_symbolic_cross_modal();
        self.refresh_language_cache();
        self.perf_symbolic_us += t1.elapsed().as_micros() as u64;

        let t2 = Instant::now();
        let latest_ep = self.working_memory.back().cloned();
        let (episode, regional_inputs) = self.try_bind(latest_ep.as_ref());
        self.perf_bind_us += t2.elapsed().as_micros() as u64;

        let t3 = Instant::now();
        let (episode, settle_iters, settle_delta) =
            self.maybe_settle_and_update(episode, speech_cos);
        self.perf_prior_us += t3.elapsed().as_micros() as u64;

        self.push_to_wm(&episode);

        let mut prior_cosines = [None; 5];
        prior_cosines[0] = speech_cos;
        prior_cosines[3] = symbolic_cos;
        Ok(PerceptionOutput {
            recon_errors: vec![
                (RegionId::AudioSpeech, speech_err),
                (RegionId::Symbolic, symbolic_err),
            ],
            episode,
            regional_inputs,
            settle_iters,
            settle_delta,
            prior_cosines,
        })
    }

    /// Ingest a 512-dim CLAP env embedding.
    pub fn ingest_env(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let device = self.device;
        let gained = apply_gain(values, self.region_gain[1]);
        let (input, cosine) = self.blend_with_prior(&gained, 1);
        if let Some(c) = cosine {
            tracing::info!(region = "env", prior_cosine = c, "top-down prior");
        }

        let t0 = Instant::now();
        let err = if self.ort_session.as_ref().is_some_and(|s| s.has_encoder()) {
            if self.latest[RegionId::AudioEnv.as_index()].is_none() {
                self.latest[RegionId::AudioEnv.as_index()] = Some(SparseActivation {
                    region: RegionId::AudioEnv,
                    indices: (0..K_ACTIVE as u16).collect(),
                    values: vec![1.0f32; K_ACTIVE],
                    n_units: N_UNITS,
                });
            }
            0.0
        } else {
            let act = self
                .model
                .as_mut()
                .unwrap()
                .env_layer
                .forward(&input, &device);
            let e = self
                .model
                .as_ref()
                .unwrap()
                .env_decoder
                .reconstruction_error(values, &act, &device);
            self.latest[RegionId::AudioEnv.as_index()] = Some(act);
            e
        };
        self.perf_kwta_us += t0.elapsed().as_micros() as u64;
        self.latest_inputs[RegionId::AudioEnv.as_index()] = Some(values.to_vec());

        let t1 = Instant::now();
        let (symbolic_cos, symbolic_err) = self.compute_symbolic_cross_modal();
        self.refresh_language_cache();
        self.perf_symbolic_us += t1.elapsed().as_micros() as u64;

        let t2 = Instant::now();
        let latest_ep = self.working_memory.back().cloned();
        let (episode, regional_inputs) = self.try_bind(latest_ep.as_ref());
        self.perf_bind_us += t2.elapsed().as_micros() as u64;

        let t3 = Instant::now();
        let (episode, settle_iters, settle_delta) = self.maybe_settle_and_update(episode, cosine);
        self.perf_prior_us += t3.elapsed().as_micros() as u64;

        self.push_to_wm(&episode);

        let mut prior_cosines = [None; 5];
        prior_cosines[1] = cosine;
        prior_cosines[3] = symbolic_cos;
        Ok(PerceptionOutput {
            recon_errors: vec![
                (RegionId::AudioEnv, err),
                (RegionId::Symbolic, symbolic_err),
            ],
            episode,
            regional_inputs,
            settle_iters,
            settle_delta,
            prior_cosines,
        })
    }

    /// Ingest a 384-dim DINOv2 visual embedding.
    pub fn ingest_visual(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let device = self.device;
        let gained = apply_gain(values, self.region_gain[2]);
        let (input, cosine) = self.blend_with_prior(&gained, 2);
        if let Some(c) = cosine {
            tracing::info!(region = "visual", prior_cosine = c, "top-down prior");
        }

        let t0 = Instant::now();
        let err = if self.ort_session.as_ref().is_some_and(|s| s.has_encoder()) {
            if self.latest[RegionId::Visual.as_index()].is_none() {
                self.latest[RegionId::Visual.as_index()] = Some(SparseActivation {
                    region: RegionId::Visual,
                    indices: (0..K_ACTIVE as u16).collect(),
                    values: vec![1.0f32; K_ACTIVE],
                    n_units: N_UNITS,
                });
            }
            0.0
        } else {
            let act = self
                .model
                .as_mut()
                .unwrap()
                .visual_layer
                .forward(&input, &device);
            let e = self
                .model
                .as_ref()
                .unwrap()
                .visual_decoder
                .reconstruction_error(values, &act, &device);
            self.latest[RegionId::Visual.as_index()] = Some(act);
            e
        };
        self.perf_kwta_us += t0.elapsed().as_micros() as u64;
        self.latest_inputs[RegionId::Visual.as_index()] = Some(values.to_vec());

        let t1 = Instant::now();
        let (symbolic_cos, symbolic_err) = self.compute_symbolic_cross_modal();
        self.refresh_language_cache();
        self.perf_symbolic_us += t1.elapsed().as_micros() as u64;

        let t2 = Instant::now();
        let latest_ep = self.working_memory.back().cloned();
        let (episode, regional_inputs) = self.try_bind(latest_ep.as_ref());
        self.perf_bind_us += t2.elapsed().as_micros() as u64;

        let t3 = Instant::now();
        let (episode, settle_iters, settle_delta) = self.maybe_settle_and_update(episode, cosine);
        self.perf_prior_us += t3.elapsed().as_micros() as u64;

        self.push_to_wm(&episode);

        let mut prior_cosines = [None; 5];
        prior_cosines[2] = cosine;
        prior_cosines[3] = symbolic_cos;
        Ok(PerceptionOutput {
            recon_errors: vec![(RegionId::Visual, err), (RegionId::Symbolic, symbolic_err)],
            episode,
            regional_inputs,
            prior_cosines,
            settle_iters,
            settle_delta,
        })
    }

    /// Ingest a 896-dim language embedding (Qwen mean-pooled token embeddings).
    pub fn ingest_language(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let device = self.device;
        self.last_language_input = Some(values.to_vec());
        let gained = apply_gain(values, self.region_gain[4]);
        let (input, cosine) = self.blend_with_prior(&gained, 4);
        if let Some(c) = cosine {
            tracing::info!(region = "language", prior_cosine = c, "top-down prior");
        }

        let t0 = Instant::now();
        let err = if self.ort_session.as_ref().is_some_and(|s| s.has_encoder()) {
            // Language is not checked in the try_bind gate (indices 0-3 only),
            // so no bootstrap needed; try_bind updates latest[4] via topk_from_z.
            0.0
        } else {
            let act = self
                .model
                .as_mut()
                .unwrap()
                .language_layer
                .forward(&input, &device);
            let e = self
                .model
                .as_ref()
                .unwrap()
                .language_decoder
                .reconstruction_error(values, &act, &device);
            self.latest[RegionId::Language.as_index()] = Some(act);
            e
        };
        self.perf_kwta_us += t0.elapsed().as_micros() as u64;
        self.latest_inputs[RegionId::Language.as_index()] = Some(values.to_vec());

        let t1 = Instant::now();
        let (symbolic_cos, symbolic_err) = self.compute_symbolic_cross_modal();
        self.perf_symbolic_us += t1.elapsed().as_micros() as u64;

        let t2 = Instant::now();
        let latest_ep = self.working_memory.back().cloned();
        let (episode, regional_inputs) = self.try_bind(latest_ep.as_ref());
        self.perf_bind_us += t2.elapsed().as_micros() as u64;

        let t3 = Instant::now();
        let (episode, settle_iters, settle_delta) = self.maybe_settle_and_update(episode, cosine);
        self.perf_prior_us += t3.elapsed().as_micros() as u64;

        self.push_to_wm(&episode);

        let mut prior_cosines = [None; 5];
        prior_cosines[3] = symbolic_cos;
        prior_cosines[4] = cosine;
        Ok(PerceptionOutput {
            recon_errors: vec![
                (RegionId::Language, err),
                (RegionId::Symbolic, symbolic_err),
            ],
            episode,
            regional_inputs,
            prior_cosines,
            settle_iters,
            settle_delta,
        })
    }

    /// Returns `(episode, settle_iters, settle_delta)`.
    /// `settle_iters` is 0 and `settle_delta` is 0.0 when the gate did not fire or
    /// `settle_steps == 0`.
    fn maybe_settle_and_update(
        &mut self,
        episode: Option<EpisodeVector>,
        cosine: Option<f32>,
    ) -> (Option<EpisodeVector>, usize, f32) {
        if let Some(ep) = episode {
            self.settle_episode_count += 1;
            let should_settle = self.settle_steps > 0
                && cosine.is_none_or(|c| c < self.settle_gate)
                && self.settle_episode_count.is_multiple_of(self.settle_every);
            let t_settle = Instant::now();
            let (refined, iters, delta) = if should_settle {
                self.settle_episode(&ep)
            } else {
                (ep, 0, 0.0)
            };
            self.perf_settle_us += t_settle.elapsed().as_micros() as u64;
            let t_prior_update = Instant::now();
            self.update_prior_predictions(&refined.0);
            self.perf_prior_update_us += t_prior_update.elapsed().as_micros() as u64;
            (Some(refined), iters, delta)
        } else {
            (None, 0, 0.0)
        }
    }

    /// Recompute the Symbolic region as a cross-modal convergence layer.
    ///
    /// Concatenates the current encoder embeddings for speech(384), env(512), visual(384),
    /// and language(896), zero-padding any missing modality, then projects through
    /// `cross_modal_proj` (2176→384) before running the symbolic k-WTA.  Called before
    /// every `try_bind` so symbolic always reflects the most current cross-modal picture.
    ///
    /// Returns (symbolic_prior_cosine, symbolic_reconstruction_err).
    fn compute_symbolic_cross_modal(&mut self) -> (Option<f32>, f32) {
        let device = self.device;

        // Build 2176-dim concat: [speech(384) | env(512) | visual(384) | language(896)]
        let mut concat = vec![0.0f32; CROSS_MODAL_DIM];
        let offsets = [(0, 384), (384, 512), (896, 384), (1280, 896)]; // (offset, dim) per region idx [0,1,2,4]
        let region_indices = [0usize, 1, 2, 4];
        for (&region_idx, &(offset, dim)) in region_indices.iter().zip(offsets.iter()) {
            if let Some(inp) = &self.latest_inputs[region_idx] {
                let len = inp.len().min(dim);
                concat[offset..offset + len].copy_from_slice(&inp[..len]);
            }
        }

        // Project to 384-dim.
        // When ORT encoder is active, use the flat weight copy for allocation-free inference
        // instead of going through the Autodiff<NdArray> tensor machinery.
        let projected: Vec<f32> = if self.ort_session.as_ref().is_some_and(|s| s.has_encoder()) {
            if let Some(flat) = &self.cross_modal_proj_flat {
                // Raw matmul: concat[2176] × flat[2176 × 384] → out[384]
                let d_out = 384usize;
                let mut out = vec![0.0f32; d_out];
                for (i, &c) in concat.iter().enumerate() {
                    if c == 0.0 {
                        continue;
                    }
                    let base = i * d_out;
                    for j in 0..d_out {
                        out[j] += c * flat[base + j];
                    }
                }
                out
            } else {
                // Flat weights not yet synced (first call before first NREM); use burn path.
                let concat_t: Tensor<TrainB, 2> =
                    Tensor::from_data(TensorData::new(concat, [1, CROSS_MODAL_DIM]), &device);
                self.model
                    .as_ref()
                    .unwrap()
                    .cross_modal_proj
                    .forward(concat_t)
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
                    .unwrap()
            }
        } else {
            let concat_t: Tensor<TrainB, 2> =
                Tensor::from_data(TensorData::new(concat, [1, CROSS_MODAL_DIM]), &device);
            self.model
                .as_ref()
                .unwrap()
                .cross_modal_proj
                .forward(concat_t)
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap()
        };

        // Store projected embedding: NREM synthesize_episode uses latest_inputs[3] as
        // input to the symbolic layer for dream synthesis, so this must always be set.
        self.latest_inputs[RegionId::Symbolic.as_index()] = Some(projected.clone());

        // When ORT encoder is active: symbolic z (and dc update) come from run_encoder +
        // topk_from_z inside try_bind.  Skip the burn kwta and reconstruction_error.
        // Bootstrap latest[3] with a placeholder on first call so the try_bind gate
        // can fire; try_bind overwrites it immediately with the real ORT z3 activation.
        if self.ort_session.as_ref().is_some_and(|s| s.has_encoder()) {
            if self.latest[RegionId::Symbolic.as_index()].is_none() {
                self.latest[RegionId::Symbolic.as_index()] = Some(SparseActivation {
                    region: RegionId::Symbolic,
                    indices: (0..K_ACTIVE as u16).collect(),
                    values: vec![1.0f32; K_ACTIVE],
                    n_units: N_UNITS,
                });
            }
            return (None, 0.0);
        }

        // Blend with prior and encode through symbolic k-WTA.
        let gained = apply_gain(&projected, self.region_gain[3]);
        let (symbolic_input, symbolic_cos) = self.blend_with_prior(&gained, 3);
        if let Some(c) = symbolic_cos {
            tracing::info!(region = "symbolic", prior_cosine = c, "top-down prior");
        }
        let symbolic_act = self
            .model
            .as_mut()
            .unwrap()
            .symbolic_layer
            .forward(&symbolic_input, &device);
        let symbolic_err = self
            .model
            .as_ref()
            .unwrap()
            .symbolic_decoder
            .reconstruction_error(&projected, &symbolic_act, &device);

        self.latest[RegionId::Symbolic.as_index()] = Some(symbolic_act);

        (symbolic_cos, symbolic_err)
    }

    fn refresh_language_cache(&mut self) {
        // Phase 1.5: when encoder ORT is active, skip — the language z comes from
        // run_encoder in try_bind, which uses latest_inputs[Language] directly.
        if self.ort_session.as_ref().is_some_and(|s| s.has_encoder()) {
            return;
        }
        let device = self.device;
        if let Some(emb) = self.last_language_input.clone() {
            let act = self
                .model
                .as_ref()
                .unwrap()
                .language_layer
                .forward_no_dc(&emb, &device);
            self.latest[RegionId::Language.as_index()] = Some(act);
            self.latest_inputs[RegionId::Language.as_index()] = Some(emb);
        }
    }

    #[allow(clippy::type_complexity)]
    fn try_bind(
        &mut self,
        latest_episode: Option<&Vec<f32>>,
    ) -> (Option<EpisodeVector>, Option<[Vec<f32>; 5]>) {
        // Require all 4 primary regions to have activations before binding fires.
        if self.latest[0].is_none()
            || self.latest[1].is_none()
            || self.latest[2].is_none()
            || self.latest[3].is_none()
        {
            return (None, None);
        }

        // Phase 1.5: encoder ORT path — replace all 5 burn activations with GPU encoder z+topk.
        // Cadence note: duty-cycle updates happen here (once per binding) for all 5 regions,
        // vs. the burn path which updates per-region per-ingest. Validate against burn-only runs.
        if self.ort_session.as_ref().is_some_and(|s| s.has_encoder()) {
            // Collect inputs as owned Vecs (ends borrows of self.latest_inputs).
            let enc_in = [
                self.latest_inputs[0]
                    .clone()
                    .unwrap_or_else(|| vec![0.0f32; 384]),
                self.latest_inputs[1]
                    .clone()
                    .unwrap_or_else(|| vec![0.0f32; 512]),
                self.latest_inputs[2]
                    .clone()
                    .unwrap_or_else(|| vec![0.0f32; 384]),
                self.latest_inputs[4]
                    .clone()
                    .unwrap_or_else(|| vec![0.0f32; 896]),
            ];
            let z_result = self
                .ort_session
                .as_mut()
                .unwrap()
                .run_encoder(&enc_in[0], &enc_in[1], &enc_in[2], &enc_in[3]);
            match z_result {
                Ok([z0, z1, z2, z3, z4]) => {
                    let model = self.model.as_mut().unwrap();
                    self.latest[0] = Some(model.speech_layer.topk_from_z(z0));
                    self.latest[1] = Some(model.env_layer.topk_from_z(z1));
                    self.latest[2] = Some(model.visual_layer.topk_from_z(z2));
                    self.latest[3] = Some(model.symbolic_layer.topk_from_z(z3));
                    // Only update language if we have a real language input; zeros → garbage z.
                    if self.latest_inputs[4].is_some() {
                        self.latest[4] = Some(model.language_layer.topk_from_z(z4));
                    }
                }
                Err(e) => tracing::warn!(?e, "ORT encoder failed; using burn activations"),
            }
        }

        // At this point self.latest[0..3] are Some (checked above; may be ORT or burn).
        let lang_zero = SparseActivation {
            region: RegionId::Language,
            indices: vec![],
            values: vec![],
            n_units: N_UNITS,
        };
        let sp = self.latest[0].as_ref().unwrap();
        let en = self.latest[1].as_ref().unwrap();
        let vi = self.latest[2].as_ref().unwrap();
        let sy = self.latest[3].as_ref().unwrap();
        let lang = self.latest[4].as_ref().unwrap_or(&lang_zero);

        let ep = if let Some(ref mut ort) = self.ort_session {
            // ORT binding path
            let sparse_z: [Vec<f32>; 5] = [
                sp.to_dense_n(N_UNITS),
                en.to_dense_n(N_UNITS),
                vi.to_dense_n(N_UNITS),
                sy.to_dense_n(N_UNITS),
                lang.to_dense_n(N_UNITS),
            ];
            let prev_zeros = vec![0.0f32; EPISODE_DIM];
            let prev = latest_episode.map(|v| v.as_slice()).unwrap_or(&prev_zeros);
            match ort.run_binding(&sparse_z, prev) {
                Ok(ep_vec) => EpisodeVector(ep_vec),
                Err(e) => {
                    tracing::warn!(?e, "ORT binding failed; falling back to burn");
                    let device = self.device;
                    self.model.as_mut().unwrap().binding.forward(
                        &[sp, en, vi, sy, lang],
                        latest_episode,
                        &device,
                    )
                }
            }
        } else {
            // Burn path (original)
            let device = self.device;
            self.model.as_mut().unwrap().binding.forward(
                &[sp, en, vi, sy, lang],
                latest_episode,
                &device,
            )
        };

        let regional = [
            self.latest_inputs[0].clone().unwrap_or_default(),
            self.latest_inputs[1].clone().unwrap_or_default(),
            self.latest_inputs[2].clone().unwrap_or_default(),
            self.latest_inputs[3].clone().unwrap_or_default(),
            self.latest_inputs[4].clone().unwrap_or_default(),
        ];
        (Some(ep), Some(regional))
    }

    // ── Sleep-time training ──────────────────────────────────────────────────

    /// Run one training step on a stored episode trace (SAE reconstruction loss).
    pub fn replay_step(&mut self, regional: &[Vec<f32>; 5]) -> PerceptionResult<f32> {
        let device = self.device;
        let mut total_loss: Option<Tensor<TrainB, 1>> = None;
        let mut n_terms = 0usize;

        // Phase 1: forward passes (duty-cycle updates happen here).
        {
            let model = self.model.as_mut().unwrap();
            for (region_idx, input) in regional.iter().enumerate() {
                if input.is_empty() {
                    continue;
                }
                let (act, z) = match region_idx {
                    0 => model.speech_layer.forward_tracked(input, &device),
                    1 => model.env_layer.forward_tracked(input, &device),
                    2 => model.visual_layer.forward_tracked(input, &device),
                    3 => model.symbolic_layer.forward_tracked(input, &device),
                    4 => model.language_layer.forward_tracked(input, &device),
                    _ => unreachable!(),
                };

                let n = act.n_units;
                let mut mask_vec = vec![0.0f32; n];
                for &i in &act.indices {
                    mask_vec[i as usize] = 1.0;
                }
                let mask_inner: Tensor<NdArray, 2> =
                    Tensor::from_data(TensorData::new(mask_vec, [1, n]), &device);
                let mask: Tensor<TrainB, 2> = Tensor::from_inner(mask_inner);
                let sparse_z = z * mask;

                let target: Tensor<TrainB, 2> =
                    Tensor::from_data(TensorData::new(input.clone(), [1, input.len()]), &device);

                let loss = match region_idx {
                    0 => model.speech_decoder.mse_loss_batch(target, sparse_z),
                    1 => model.env_decoder.mse_loss_batch(target, sparse_z),
                    2 => model.visual_decoder.mse_loss_batch(target, sparse_z),
                    3 => model.symbolic_decoder.mse_loss_batch(target, sparse_z),
                    4 => model.language_decoder.mse_loss_batch(target, sparse_z),
                    _ => unreachable!(),
                };

                total_loss = Some(match total_loss {
                    None => loss,
                    Some(acc) => acc + loss,
                });
                n_terms += 1;
            }
        } // &mut model borrow ends

        // Phase 2: backward + optimizer step.
        if let Some(loss) = total_loss {
            let loss_val: f32 = loss.clone().into_scalar();
            let model = self.model.take().unwrap();
            let grads = loss.backward();
            let grads_p = GradientsParams::from_grads(grads, &model);
            let model = self.optimizer.step(self.current_lr, model, grads_p);
            self.model = Some(model);
            Ok(loss_val / n_terms.max(1) as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Batched NREM replay step: one forward+backward pass per region, all samples.
    pub fn replay_step_batch(&mut self, batch: &[&[Vec<f32>; 5]]) -> PerceptionResult<f32> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        let device = self.device;
        let mut total_loss: Option<Tensor<TrainB, 1>> = None;
        let mut n_terms = 0usize;

        {
            let model = self.model.as_mut().unwrap();

            for region_idx in 0..5usize {
                let valid: Vec<&Vec<f32>> = batch
                    .iter()
                    .filter_map(|r| {
                        let v = &r[region_idx];
                        if v.is_empty() {
                            None
                        } else {
                            Some(v)
                        }
                    })
                    .collect();
                if valid.is_empty() {
                    continue;
                }

                let b = valid.len();
                let n_in = valid[0].len();
                let flat: Vec<f32> = valid.iter().flat_map(|v| v.iter().copied()).collect();

                let input_batch: Tensor<TrainB, 2> =
                    Tensor::from_data(TensorData::new(flat.clone(), [b, n_in]), &device);

                let (z_batch, flat_masks) = match region_idx {
                    0 => model
                        .speech_layer
                        .forward_tracked_batch(input_batch.clone(), b),
                    1 => model
                        .env_layer
                        .forward_tracked_batch(input_batch.clone(), b),
                    2 => model
                        .visual_layer
                        .forward_tracked_batch(input_batch.clone(), b),
                    3 => model
                        .symbolic_layer
                        .forward_tracked_batch(input_batch.clone(), b),
                    4 => model
                        .language_layer
                        .forward_tracked_batch(input_batch.clone(), b),
                    _ => unreachable!(),
                };

                let n_units = flat_masks.len() / b;
                let mask_inner: Tensor<NdArray, 2> =
                    Tensor::from_data(TensorData::new(flat_masks, [b, n_units]), &device);
                let mask: Tensor<TrainB, 2> = Tensor::from_inner(mask_inner);
                let sparse_z = z_batch * mask;

                let target: Tensor<TrainB, 2> =
                    Tensor::from_data(TensorData::new(flat, [b, n_in]), &device);

                let region_loss = match region_idx {
                    0 => model.speech_decoder.mse_loss_batch(target, sparse_z),
                    1 => model.env_decoder.mse_loss_batch(target, sparse_z),
                    2 => model.visual_decoder.mse_loss_batch(target, sparse_z),
                    3 => model.symbolic_decoder.mse_loss_batch(target, sparse_z),
                    4 => model.language_decoder.mse_loss_batch(target, sparse_z),
                    _ => unreachable!(),
                };

                total_loss = Some(match total_loss {
                    None => region_loss,
                    Some(acc) => acc + region_loss,
                });
                n_terms += 1;
            }
        }

        if let Some(loss) = total_loss {
            let loss_val: f32 = loss.clone().into_scalar();
            let model = self.model.take().unwrap();
            let grads = loss.backward();
            let grads_p = GradientsParams::from_grads(grads, &model);
            let model = self.optimizer.step(self.current_lr, model, grads_p);
            self.model = Some(model);
            Ok(loss_val / n_terms.max(1) as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Batched output-head training for NREM replay (ADR 0010).
    pub fn wake_reconstruct_step_batch(
        &mut self,
        episodes: &[&[f32]],
        regionals: &[&[Vec<f32>; 5]],
    ) -> PerceptionResult<f32> {
        if episodes.is_empty() {
            return Ok(0.0);
        }
        let device = self.device;
        let ep_dim = EPISODE_DIM;
        let mut total_loss: Option<Tensor<TrainB, 1>> = None;
        let mut n = 0usize;

        for head_idx in 0..5usize {
            let valid_pairs: Vec<(&[f32], &Vec<f32>)> = episodes
                .iter()
                .zip(regionals.iter())
                .filter_map(|(&ep, &r)| {
                    let inp = &r[head_idx];
                    if inp.is_empty() {
                        None
                    } else {
                        Some((ep, inp))
                    }
                })
                .collect();
            if valid_pairs.is_empty() {
                continue;
            }

            let b = valid_pairs.len();
            let n_out = valid_pairs[0].1.len();
            let flat_eps: Vec<f32> = valid_pairs
                .iter()
                .flat_map(|(ep, _)| ep.iter().copied())
                .collect();
            let flat_targets: Vec<f32> = valid_pairs
                .iter()
                .flat_map(|(_, inp)| inp.iter().copied())
                .collect();

            let ep_batch: Tensor<TrainB, 2> =
                Tensor::from_data(TensorData::new(flat_eps, [b, ep_dim]), &device);
            let target_batch: Tensor<TrainB, 2> =
                Tensor::from_data(TensorData::new(flat_targets, [b, n_out]), &device);

            let pred = {
                let model = self.model.as_ref().unwrap();
                match head_idx {
                    0 => model.speech_head.forward(ep_batch),
                    1 => model.env_head.forward(ep_batch),
                    2 => model.visual_head.forward(ep_batch),
                    3 => model.symbolic_head.forward(ep_batch),
                    4 => model.language_head.forward(ep_batch),
                    _ => unreachable!(),
                }
            };

            let loss: Tensor<TrainB, 1> = (pred - target_batch).powi_scalar(2).mean();
            total_loss = Some(match total_loss {
                None => loss,
                Some(acc) => acc + loss,
            });
            n += 1;
        }

        if let Some(loss) = total_loss {
            let loss_val: f32 = loss.clone().into_scalar();
            let model = self.model.take().unwrap();
            let grads = loss.backward();
            let grads_p = GradientsParams::from_grads(grads, &model);
            let model = self.optimizer.step(self.current_lr, model, grads_p);
            self.model = Some(model);
            Ok(loss_val / n.max(1) as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Fused NREM replay + output-head training in one backward pass.
    ///
    /// Semantically identical to calling `replay_step_batch` then
    /// `wake_reconstruct_step_batch`, but accumulates all losses into a single
    /// combined tensor and calls `backward()` + `optimizer.step()` once instead
    /// of twice. Saves one full AdamW pass per batch (~150ms at batch=128).
    ///
    /// Safe to fuse because the two computation graphs are completely independent:
    /// SAE path uses stored regional vectors; head path uses stored episode vectors.
    /// No shared intermediate activations → combined gradients are identical.
    pub fn fused_replay_wake_step_batch(
        &mut self,
        batch: &[&[Vec<f32>; 5]],
        episodes: &[&[f32]],
    ) -> PerceptionResult<(f32, f32)> {
        if batch.is_empty() {
            return Ok((0.0, 0.0));
        }
        let device = self.device;

        // ── Phase 1: SAE replay losses (needs &mut model for duty-cycle updates) ──
        let mut sae_loss: Option<Tensor<TrainB, 1>> = None;
        let mut sae_n_terms = 0usize;
        {
            let model = self.model.as_mut().unwrap();
            for region_idx in 0..5usize {
                let valid: Vec<&Vec<f32>> = batch
                    .iter()
                    .filter_map(|r| {
                        let v = &r[region_idx];
                        if v.is_empty() {
                            None
                        } else {
                            Some(v)
                        }
                    })
                    .collect();
                if valid.is_empty() {
                    continue;
                }
                let b = valid.len();
                let n_in = valid[0].len();
                let flat: Vec<f32> = valid.iter().flat_map(|v| v.iter().copied()).collect();

                let input_batch: Tensor<TrainB, 2> =
                    Tensor::from_data(TensorData::new(flat.clone(), [b, n_in]), &device);

                let (z_batch, flat_masks) = match region_idx {
                    0 => model
                        .speech_layer
                        .forward_tracked_batch(input_batch.clone(), b),
                    1 => model
                        .env_layer
                        .forward_tracked_batch(input_batch.clone(), b),
                    2 => model
                        .visual_layer
                        .forward_tracked_batch(input_batch.clone(), b),
                    3 => model
                        .symbolic_layer
                        .forward_tracked_batch(input_batch.clone(), b),
                    4 => model
                        .language_layer
                        .forward_tracked_batch(input_batch.clone(), b),
                    _ => unreachable!(),
                };

                let n_units = flat_masks.len() / b;
                let mask_inner: Tensor<NdArray, 2> =
                    Tensor::from_data(TensorData::new(flat_masks, [b, n_units]), &device);
                let mask: Tensor<TrainB, 2> = Tensor::from_inner(mask_inner);
                let sparse_z = z_batch * mask;
                let target: Tensor<TrainB, 2> =
                    Tensor::from_data(TensorData::new(flat, [b, n_in]), &device);

                let region_loss = match region_idx {
                    0 => model.speech_decoder.mse_loss_batch(target, sparse_z),
                    1 => model.env_decoder.mse_loss_batch(target, sparse_z),
                    2 => model.visual_decoder.mse_loss_batch(target, sparse_z),
                    3 => model.symbolic_decoder.mse_loss_batch(target, sparse_z),
                    4 => model.language_decoder.mse_loss_batch(target, sparse_z),
                    _ => unreachable!(),
                };
                sae_loss = Some(match sae_loss {
                    None => region_loss,
                    Some(a) => a + region_loss,
                });
                sae_n_terms += 1;
            }
        } // &mut borrow of model ends here

        // ── Phase 2: output-head losses (needs only &model, not &mut) ───────────
        let ep_dim = EPISODE_DIM;
        let mut wake_loss: Option<Tensor<TrainB, 1>> = None;
        let mut wake_n_terms = 0usize;
        if !episodes.is_empty() {
            for head_idx in 0..5usize {
                let valid_pairs: Vec<(&[f32], &Vec<f32>)> = episodes
                    .iter()
                    .zip(batch.iter())
                    .filter_map(|(&ep, &r)| {
                        let inp = &r[head_idx];
                        if inp.is_empty() {
                            None
                        } else {
                            Some((ep, inp))
                        }
                    })
                    .collect();
                if valid_pairs.is_empty() {
                    continue;
                }
                let b = valid_pairs.len();
                let n_out = valid_pairs[0].1.len();
                let flat_eps: Vec<f32> = valid_pairs
                    .iter()
                    .flat_map(|(ep, _)| ep.iter().copied())
                    .collect();
                let flat_tgt: Vec<f32> = valid_pairs
                    .iter()
                    .flat_map(|(_, inp)| inp.iter().copied())
                    .collect();

                let ep_batch: Tensor<TrainB, 2> =
                    Tensor::from_data(TensorData::new(flat_eps, [b, ep_dim]), &device);
                let target_batch: Tensor<TrainB, 2> =
                    Tensor::from_data(TensorData::new(flat_tgt, [b, n_out]), &device);

                let pred = {
                    let model = self.model.as_ref().unwrap();
                    match head_idx {
                        0 => model.speech_head.forward(ep_batch),
                        1 => model.env_head.forward(ep_batch),
                        2 => model.visual_head.forward(ep_batch),
                        3 => model.symbolic_head.forward(ep_batch),
                        4 => model.language_head.forward(ep_batch),
                        _ => unreachable!(),
                    }
                };
                let h_loss: Tensor<TrainB, 1> = (pred - target_batch).powi_scalar(2).mean();
                wake_loss = Some(match wake_loss {
                    None => h_loss,
                    Some(a) => a + h_loss,
                });
                wake_n_terms += 1;
            }
        }

        // ── Single combined backward + optimizer step ────────────────────────────
        let sae_val = sae_loss
            .as_ref()
            .map(|l| l.clone().into_scalar() / sae_n_terms.max(1) as f32)
            .unwrap_or(0.0);
        let wake_val = wake_loss
            .as_ref()
            .map(|l| l.clone().into_scalar() / wake_n_terms.max(1) as f32)
            .unwrap_or(0.0);

        let combined = match (sae_loss, wake_loss) {
            (Some(sl), Some(wl)) => Some(sl + wl),
            (Some(sl), None) => Some(sl),
            (None, Some(wl)) => Some(wl),
            (None, None) => None,
        };
        if let Some(loss) = combined {
            let model = self.model.take().unwrap();
            let grads = loss.backward();
            let grads_p = GradientsParams::from_grads(grads, &model);
            let model = self.optimizer.step(self.current_lr, model, grads_p);
            self.model = Some(model);
        }

        Ok((sae_val, wake_val))
    }

    /// Run one wake-time reconstruction step on a just-produced episode.
    pub fn wake_reconstruct_step(
        &mut self,
        episode: &[f32],
        regional: &[Vec<f32>; 5],
    ) -> PerceptionResult<f32> {
        let device = self.device;
        let ep_t: Tensor<TrainB, 2> = Tensor::from_data(
            TensorData::new(episode.to_vec(), [1, episode.len()]),
            &device,
        );
        let mut total_loss: Option<Tensor<TrainB, 1>> = None;
        let mut n = 0usize;

        for (head_idx, input) in regional.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let target: Tensor<TrainB, 2> =
                Tensor::from_data(TensorData::new(input.clone(), [1, input.len()]), &device);

            let pred = {
                let model = self.model.as_ref().unwrap();
                match head_idx {
                    0 => model.speech_head.forward(ep_t.clone()),
                    1 => model.env_head.forward(ep_t.clone()),
                    2 => model.visual_head.forward(ep_t.clone()),
                    3 => model.symbolic_head.forward(ep_t.clone()),
                    4 => model.language_head.forward(ep_t.clone()),
                    _ => unreachable!(),
                }
            };

            let loss: Tensor<TrainB, 1> = (pred - target).powi_scalar(2).mean();
            total_loss = Some(match total_loss {
                None => loss,
                Some(acc) => acc + loss,
            });
            n += 1;
        }

        if let Some(loss) = total_loss {
            let loss_val: f32 = loss.clone().into_scalar();
            let model = self.model.take().unwrap();
            let grads = loss.backward();
            let grads_p = GradientsParams::from_grads(grads, &model);
            let model = self.optimizer.step(self.current_lr, model, grads_p);
            self.ema_heads.update(&model);
            self.model = Some(model);
            Ok(loss_val / n.max(1) as f32)
        } else {
            Ok(0.0)
        }
    }

    // ── Async wake reconstruction (WakeReconWorker) ──────────────────────────

    /// Spawn a background worker that runs wake_reconstruct_step off the ingest
    /// loop. Seeds it with the current head weights and EMA state. See
    /// `wake_worker` module docs for the weight-sync protocol.
    pub fn create_wake_recon_worker(&self) -> WakeReconWorker {
        WakeReconWorker::spawn(self.head_weights_flat(), self.ema_heads.clone())
    }

    /// Flat [EPISODE_DIM × d_out] copies of the 5 output-head weights.
    pub fn head_weights_flat(&self) -> [Vec<f32>; 5] {
        let m = self.model.as_ref().unwrap();
        [
            &m.speech_head,
            &m.env_head,
            &m.visual_head,
            &m.symbolic_head,
            &m.language_head,
        ]
        .map(|h| {
            h.weight
                .val()
                .inner()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap()
        })
    }

    /// Write worker-trained head weights back into the model (sleep-entry merge).
    /// ParamIds are preserved so the main optimizer's AdamW moments stay attached.
    pub fn set_head_weights_flat(&mut self, weights: [Vec<f32>; 5]) {
        let device = self.device;
        let m = self.model.as_mut().unwrap();
        let heads = [
            &mut m.speech_head,
            &mut m.env_head,
            &mut m.visual_head,
            &mut m.symbolic_head,
            &mut m.language_head,
        ];
        for (h, (flat, d_out)) in heads.into_iter().zip(weights.into_iter().zip(REGION_DIMS)) {
            let id = h.weight.id;
            // require_grad is mandatory: Param::initialized inherits the tensor's
            // flag, and from_data tensors don't track gradients — without it the
            // merged heads would be silently frozen for NREM training.
            let t: Tensor<TrainB, 2> =
                Tensor::from_data(TensorData::new(flat, [EPISODE_DIM, d_out]), &device)
                    .require_grad();
            h.weight = Param::initialized(id, t);
        }
    }

    /// Apply worker-streamed flat EMA weights to the inference path.
    pub fn apply_ema_flat(&mut self, flat: [Vec<f32>; 5]) {
        self.ema_heads.set_flat(flat);
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// Measure SAE reconstruction MSE per region (inference, no weight update).
    pub fn measure_mse(&mut self, regional: &[Vec<f32>; 5]) -> PerceptionResult<[f32; 5]> {
        let device = self.device;
        let mut result = [0.0f32; 5];
        for (idx, input) in regional.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let act = {
                let model = self.model.as_mut().unwrap();
                match idx {
                    0 => model.speech_layer.forward(input, &device),
                    1 => model.env_layer.forward(input, &device),
                    2 => model.visual_layer.forward(input, &device),
                    3 => model.symbolic_layer.forward(input, &device),
                    4 => model.language_layer.forward(input, &device),
                    _ => unreachable!(),
                }
            };
            let err = {
                let model = self.model.as_ref().unwrap();
                match idx {
                    0 => model
                        .speech_decoder
                        .reconstruction_error(input, &act, &device),
                    1 => model.env_decoder.reconstruction_error(input, &act, &device),
                    2 => model
                        .visual_decoder
                        .reconstruction_error(input, &act, &device),
                    3 => model
                        .symbolic_decoder
                        .reconstruction_error(input, &act, &device),
                    4 => model
                        .language_decoder
                        .reconstruction_error(input, &act, &device),
                    _ => unreachable!(),
                }
            };
            result[idx] = err;
        }
        Ok(result)
    }

    /// Blend two sets of regional encoder inputs and produce a synthetic episode.
    /// No DC update — inference only.
    pub fn synthesize_episode(
        &mut self,
        regional_a: &[Vec<f32>; 5],
        regional_b: &[Vec<f32>; 5],
        alpha: f32,
    ) -> PerceptionResult<Vec<f32>> {
        let device = self.device;
        let lang_zero = SparseActivation {
            region: RegionId::Language,
            indices: vec![],
            values: vec![],
            n_units: N_UNITS,
        };
        let mut acts: [Option<SparseActivation>; 5] = [None, None, None, None, None];
        for i in 0..5 {
            let a = &regional_a[i];
            let b = &regional_b[i];
            if a.is_empty() || b.is_empty() {
                continue;
            }
            let blended: Vec<f32> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| alpha * x + (1.0 - alpha) * y)
                .collect();
            let model = self.model.as_ref().unwrap();
            acts[i] = Some(match i {
                0 => model.speech_layer.forward_no_dc(&blended, &device),
                1 => model.env_layer.forward_no_dc(&blended, &device),
                2 => model.visual_layer.forward_no_dc(&blended, &device),
                3 => model.symbolic_layer.forward_no_dc(&blended, &device),
                4 => model.language_layer.forward_no_dc(&blended, &device),
                _ => unreachable!(),
            });
        }
        let (a0, a1, a2, a3) = match (&acts[0], &acts[1], &acts[2], &acts[3]) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return Err(PerceptionError::BindingIncomplete),
        };
        let a4 = acts[4].as_ref().unwrap_or(&lang_zero);
        let episode =
            self.model
                .as_mut()
                .unwrap()
                .binding
                .forward(&[a0, a1, a2, a3, a4], None, &device);
        Ok(episode.0)
    }

    /// Mean output-head MSE for one episode — no backward step.
    pub fn measure_output_mse_inference(
        &self,
        episode: &[f32],
        regional: &[Vec<f32>; 5],
    ) -> PerceptionResult<f32> {
        let device = self.device;
        let ep_t: Tensor<TrainB, 2> =
            Tensor::from_data(TensorData::new(episode.to_vec(), [1, EPISODE_DIM]), &device);
        let model = self.model.as_ref().unwrap();
        let mut total = 0.0f32;
        let mut n = 0usize;
        for (head_idx, input) in regional.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let pred = match head_idx {
                0 => model.speech_head.forward(ep_t.clone()),
                1 => model.env_head.forward(ep_t.clone()),
                2 => model.visual_head.forward(ep_t.clone()),
                3 => model.symbolic_head.forward(ep_t.clone()),
                4 => model.language_head.forward(ep_t.clone()),
                _ => unreachable!(),
            };
            let target: Tensor<TrainB, 2> =
                Tensor::from_data(TensorData::new(input.clone(), [1, input.len()]), &device);
            let mse: f32 = (pred - target).powi_scalar(2).mean().into_scalar();
            total += mse;
            n += 1;
        }
        Ok(if n > 0 { total / n as f32 } else { 0.0 })
    }

    // ── Homeostasis ──────────────────────────────────────────────────────────

    /// Apply global weight decay to all trainable Linear weights.
    pub fn homeostatic_decay(&mut self, rate: f32) {
        let scale = 1.0 - rate;
        let model = self.model.as_mut().unwrap();
        scale_weight(&mut model.speech_layer.linear, scale);
        scale_weight(&mut model.env_layer.linear, scale);
        scale_weight(&mut model.visual_layer.linear, scale);
        scale_weight(&mut model.symbolic_layer.linear, scale);
        scale_weight(&mut model.language_layer.linear, scale);
        scale_weight(&mut model.speech_decoder.decoder, scale);
        scale_weight(&mut model.env_decoder.decoder, scale);
        scale_weight(&mut model.visual_decoder.decoder, scale);
        scale_weight(&mut model.symbolic_decoder.decoder, scale);
        scale_weight(&mut model.language_decoder.decoder, scale);
        scale_weight(&mut model.binding.region_0, scale);
        scale_weight(&mut model.binding.region_1, scale);
        scale_weight(&mut model.binding.region_2, scale);
        scale_weight(&mut model.binding.region_3, scale);
        scale_weight(&mut model.binding.region_4, scale);
        scale_weight(&mut model.binding.w_q, scale);
        scale_weight(&mut model.binding.w_k, scale);
        scale_weight(&mut model.binding.w_v, scale);
        scale_weight(&mut model.speech_head, scale);
        scale_weight(&mut model.env_head, scale);
        scale_weight(&mut model.visual_head, scale);
        scale_weight(&mut model.symbolic_head, scale);
        scale_weight(&mut model.language_head, scale);
    }

    /// Grow a region by appending `n_new` units with small random encoder weights.
    pub fn grow_region(&mut self, region: RegionId, n_new: usize) {
        let device = self.device;
        let n_in = REGION_DIMS[region.as_index()];
        let model = self.model.as_mut().unwrap();

        let new_n = {
            let layer = Self::layer_mut(model, region);
            let old_n = layer.n_units;
            let new_n = old_n + n_new;

            // Encoder weight shape: [n_in, n_units] — grow by appending new columns (dim 1).
            let old_w: Tensor<NdArray, 2> = layer.linear.weight.val().inner();
            let new_cols: Tensor<NdArray, 2> =
                Tensor::random([n_in, n_new], Distribution::Normal(0.0, 0.01), &device);
            let new_w: Tensor<TrainB, 2> =
                Tensor::from_inner(Tensor::cat(vec![old_w, new_cols], 1));
            layer.linear.weight = Param::initialized(ParamId::new(), new_w);
            layer.n_units = new_n;
            layer.duty_cycle.extend(vec![0.0f32; n_new]);
            new_n
        };

        {
            // Decoder weight shape: [n_units, n_in] — grow by appending new rows (dim 0).
            let decoder = Self::decoder_mut(model, region);
            let old_w: Tensor<NdArray, 2> = decoder.decoder.weight.val().inner();
            let new_rows: Tensor<NdArray, 2> = Tensor::zeros([n_new, n_in], &device);
            let new_w: Tensor<TrainB, 2> =
                Tensor::from_inner(Tensor::cat(vec![old_w, new_rows], 0));
            decoder.decoder.weight = Param::initialized(ParamId::new(), new_w);
            decoder.n_units = new_n;
        }

        self.optimizer = AdamWConfig::new().init::<TrainB, PerceptionModel<TrainB>>();
        tracing::info!(region = ?region, n_new, "region grown");
    }

    /// Cascade-correlation init: grow a region using high-residual inputs as encoder
    /// weight seeds (ADR 0007, P2b). `sorted_inputs` should be ordered by reconstruction
    /// residual magnitude (descending). Falls back to random init when the list is empty.
    ///
    /// Each new unit's encoder column is set to the corresponding seed (normalized to
    /// unit L2 × 0.01 scale), cycling through seeds when `n_new > sorted_inputs.len()`.
    /// The decoder row is initialised to zeros (same as `grow_region`).
    pub fn grow_region_seeded(
        &mut self,
        region: RegionId,
        n_new: usize,
        sorted_inputs: &[Vec<f32>],
    ) {
        if sorted_inputs.is_empty() {
            return self.grow_region(region, n_new);
        }
        let device = self.device;
        let n_in = REGION_DIMS[region.as_index()];
        let model = self.model.as_mut().unwrap();

        let new_n = {
            let layer = Self::layer_mut(model, region);
            let old_n = layer.n_units;
            let new_n = old_n + n_new;

            // Build encoder columns: normalised seed × 0.01 (matches grow_region scale).
            let mut cols: Vec<f32> = Vec::with_capacity(n_in * n_new);
            let n_seeds = sorted_inputs.len();
            for k in 0..n_new {
                let seed = &sorted_inputs[k % n_seeds];
                let norm: f32 =
                    seed.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                for &x in seed {
                    cols.push(x / norm * 0.01);
                }
            }
            let cols_t: Tensor<TrainB, 2> = Tensor::from_inner(
                Tensor::<NdArray, 2>::from_data(
                    burn::tensor::TensorData::new(cols, [n_in, n_new]),
                    &device,
                ),
            );

            let old_w: Tensor<NdArray, 2> = layer.linear.weight.val().inner();
            let new_w: Tensor<TrainB, 2> =
                Tensor::from_inner(Tensor::cat(vec![old_w, cols_t.inner()], 1));
            layer.linear.weight = Param::initialized(ParamId::new(), new_w);
            layer.n_units = new_n;
            layer.duty_cycle.extend(vec![0.0f32; n_new]);
            new_n
        };

        {
            let decoder = Self::decoder_mut(model, region);
            let old_w: Tensor<NdArray, 2> = decoder.decoder.weight.val().inner();
            let new_rows: Tensor<NdArray, 2> = Tensor::zeros([n_new, n_in], &device);
            let new_w: Tensor<TrainB, 2> =
                Tensor::from_inner(Tensor::cat(vec![old_w, new_rows], 0));
            decoder.decoder.weight = Param::initialized(ParamId::new(), new_w);
            decoder.n_units = new_n;
        }

        self.optimizer = AdamWConfig::new().init::<TrainB, PerceptionModel<TrainB>>();
        tracing::info!(region = ?region, n_new, "region grown (seeded)");
    }

    /// For each region, return inputs from `traces` sorted by reconstruction residual
    /// magnitude (highest first). Used by `grow_region_seeded` to orient new units
    /// toward inputs the current network reconstructs poorly (cascade-correlation, P2b).
    ///
    /// Traces with empty regional input are skipped. Returns at most `cap` inputs per
    /// region to bound memory use.
    #[allow(clippy::type_complexity)]
    pub fn collect_high_residual_inputs<T>(
        &mut self,
        traces: &[(Vec<f32>, [Vec<f32>; 5], T)],
        cap: usize,
    ) -> [Vec<Vec<f32>>; 5] {
        let device = self.device;
        // Per-region: (residual_mse, input_vec)
        let mut scored: [Vec<(f32, Vec<f32>)>; 5] = Default::default();

        for (_, regional, _) in traces {
            for (idx, input) in regional.iter().enumerate() {
                if input.is_empty() {
                    continue;
                }
                let (act, recon) = {
                    let model = self.model.as_mut().unwrap();
                    let act = match idx {
                        0 => model.speech_layer.forward(input, &device),
                        1 => model.env_layer.forward(input, &device),
                        2 => model.visual_layer.forward(input, &device),
                        3 => model.symbolic_layer.forward(input, &device),
                        4 => model.language_layer.forward(input, &device),
                        _ => unreachable!(),
                    };
                    let recon = match idx {
                        0 => model.speech_decoder.reconstruct(&act, &device),
                        1 => model.env_decoder.reconstruct(&act, &device),
                        2 => model.visual_decoder.reconstruct(&act, &device),
                        3 => model.symbolic_decoder.reconstruct(&act, &device),
                        4 => model.language_decoder.reconstruct(&act, &device),
                        _ => unreachable!(),
                    };
                    (act, recon)
                };
                let _ = act;
                let mse: f32 = input
                    .iter()
                    .zip(&recon)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    / input.len() as f32;
                scored[idx].push((mse, input.clone()));
            }
        }

        let mut result: [Vec<Vec<f32>>; 5] = Default::default();
        for i in 0..5 {
            scored[i].sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            result[i] = scored[i]
                .drain(..)
                .take(cap)
                .map(|(_, v)| v)
                .collect();
        }
        result
    }

    /// Prune low-magnitude grown units from a region (Phase 12c).
    /// Only removes units with index >= N_UNITS. Returns number pruned.
    pub fn prune_region(&mut self, region: RegionId, prune_fraction: f32) -> usize {
        let device = self.device;
        let n_in = REGION_DIMS[region.as_index()];
        let model = self.model.as_mut().unwrap();

        let current_n = Self::layer_mut(model, region).n_units;
        let n_grown = current_n.saturating_sub(N_UNITS);
        if n_grown == 0 {
            return 0;
        }
        let n_prune = ((n_grown as f32 * prune_fraction).round() as usize).min(n_grown);
        if n_prune == 0 {
            return 0;
        }

        // Encoder weight shape: [n_in, current_n] — units are columns.
        let enc_flat: Vec<f32> = Self::layer_mut(model, region)
            .linear
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        // L1 norm of unit i = sum over rows j of abs(enc_flat[j * current_n + i]).
        let mut grown_norms: Vec<(usize, f32)> = (N_UNITS..current_n)
            .map(|i| {
                let norm: f32 = (0..n_in).map(|j| enc_flat[j * current_n + i].abs()).sum();
                (i, norm)
            })
            .collect();
        grown_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let remove: std::collections::HashSet<usize> =
            grown_norms[..n_prune].iter().map(|&(i, _)| i).collect();
        let keep: Vec<usize> = (0..current_n).filter(|i| !remove.contains(i)).collect();
        let new_n = keep.len();

        // Keep columns `keep` from [n_in, current_n] → [n_in, new_n].
        let new_enc_flat: Vec<f32> = (0..n_in)
            .flat_map(|j| {
                let enc = &enc_flat;
                keep.iter()
                    .map(move |&i| enc[j * current_n + i])
                    .collect::<Vec<_>>()
            })
            .collect();

        // Decoder weight shape: [current_n, n_in] — units are rows.
        let dec_flat: Vec<f32> = Self::decoder_mut(model, region)
            .decoder
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        // Keep rows `keep` from [current_n, n_in] → [new_n, n_in].
        let new_dec_flat: Vec<f32> = keep
            .iter()
            .flat_map(|&i| dec_flat[i * n_in..(i + 1) * n_in].iter().copied())
            .collect();

        {
            let layer = Self::layer_mut(model, region);
            let new_enc: Tensor<TrainB, 2> =
                Tensor::from_data(TensorData::new(new_enc_flat, [n_in, new_n]), &device);
            layer.linear.weight = Param::initialized(ParamId::new(), new_enc);
            let old_dc = layer.duty_cycle.clone();
            layer.duty_cycle = keep.iter().map(|&i| old_dc[i]).collect();
            layer.n_units = new_n;
        }

        {
            let decoder = Self::decoder_mut(model, region);
            let new_dec: Tensor<TrainB, 2> =
                Tensor::from_data(TensorData::new(new_dec_flat, [new_n, n_in]), &device);
            decoder.decoder.weight = Param::initialized(ParamId::new(), new_dec);
            decoder.n_units = new_n;
        }

        self.optimizer = AdamWConfig::new().init::<TrainB, PerceptionModel<TrainB>>();
        tracing::info!(region = ?region, old_n = current_n, new_n, n_pruned = n_prune, "region pruned");
        n_prune
    }

    // ── ORT inference session management ────────────────────────────────────

    /// Load ORT inference sessions (binding + encoder if present).
    /// Call once at startup (and after each sleep cycle via `rebuild_ort_session`).
    /// `use_migraphx`: use AMD MIGraphX GPU EP; false → CPU EP (for testing).
    /// Phase 1.5: also loads `perception_encoder.onnx` if it exists alongside `bind_path`.
    pub fn enable_ort_inference(
        &mut self,
        bind_path: &std::path::Path,
        use_migraphx: bool,
    ) -> anyhow::Result<()> {
        let mut sess = crate::onnx_session::OnnxPerceptionSession::load(bind_path, use_migraphx)?;
        // Phase 1.5: attach encoder session if its ONNX file exists in the same directory.
        let enc_path = bind_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("perception_encoder.onnx");
        if enc_path.exists() {
            sess = sess.with_encoder(&enc_path, use_migraphx)?;
            tracing::info!(
                migraphx = use_migraphx,
                "ORT encoder+binding sessions loaded"
            );
        } else {
            tracing::info!(
                migraphx = use_migraphx,
                "ORT binding session loaded (no encoder)"
            );
        }
        self.ort_session = Some(sess);
        self.sync_cross_modal_flat();
        Ok(())
    }

    /// Export current burn weights to `weights_dir`, call the Python graph-builder script,
    /// then reload the ORT sessions. Call at each sleep→wake boundary when the model
    /// weights have changed (NREM training, growth, prune).
    ///
    /// `script_path`: path to `scripts/export_perception_graph.py`
    /// `onnx_dir`:    where to write (and read from) the ONNX graphs
    /// `use_migraphx`: true for production GPU; false for CPU-only testing
    pub fn rebuild_ort_session(
        &mut self,
        script_path: &std::path::Path,
        onnx_dir: &std::path::Path,
        use_migraphx: bool,
    ) -> anyhow::Result<()> {
        // Step 1: export burn weights
        let weights_dir = onnx_dir.join("weights");
        self.export_weights_for_onnx(&weights_dir)?;

        // Step 2: run Python to build ONNX graphs
        let status = std::process::Command::new("python3")
            .arg(script_path)
            .arg("--weights-dir")
            .arg(&weights_dir)
            .arg("--out-dir")
            .arg(onnx_dir)
            .status()
            .map_err(|e| anyhow::anyhow!("failed to spawn Python: {e}"))?;
        if !status.success() {
            anyhow::bail!("export_perception_graph.py exited with {status}");
        }

        // Step 3: reload ORT sessions with new weights
        let bind_path = onnx_dir.join("perception_binding.onnx");
        self.enable_ort_inference(&bind_path, use_migraphx)?;
        self.sync_cross_modal_flat();
        tracing::info!("ORT sessions rebuilt after sleep cycle");
        Ok(())
    }

    /// Disable ORT inference (fall back to burn for all inference).
    pub fn disable_ort_inference(&mut self) {
        self.ort_session = None;
    }

    /// True if ORT binding inference is active.
    pub fn ort_active(&self) -> bool {
        self.ort_session.is_some()
    }

    // ── State accessors ──────────────────────────────────────────────────────

    pub fn set_region_gain(&mut self, gain: [f32; 5]) {
        self.region_gain = gain;
    }

    pub fn reset_region_gain(&mut self) {
        self.region_gain = [1.0; 5];
    }

    pub fn set_wm_size(&mut self, size: usize) {
        self.wm_size = size;
        self.working_memory.truncate(size);
    }

    pub fn clear_working_memory(&mut self) {
        self.working_memory.clear();
    }

    pub fn set_prior_beta(&mut self, beta: f32) {
        self.prior_beta = beta;
    }

    pub fn set_settle_steps(&mut self, steps: usize) {
        self.settle_steps = steps;
    }

    pub fn set_settle_beta(&mut self, beta: f32) {
        self.settle_beta = beta;
    }

    pub fn set_settle_gate(&mut self, gate: f32) {
        self.settle_gate = gate;
    }

    pub fn set_settle_every(&mut self, every: usize) {
        self.settle_every = every.max(1);
    }

    /// Detach all model parameters from the accumulated autodiff graph and reset the optimizer.
    ///
    /// burn's global AutodiffServer accumulates stale backward-step entries each time the
    /// optimizer writes the same ParamId. After N optimizer steps the backward() traversal
    /// must cross an N-step chain even though the forward graph is constant-depth — causing
    /// the 100-500x NREM slowdown observed within each session.
    ///
    /// Calling this before the NREM loop re-wraps every parameter as a fresh leaf tensor
    /// with a new ParamId, letting the server free accumulated entries on the next backward.
    /// Equivalent to what grow_region / prune_region do incidentally (hence why post-growth
    /// NREM cycles are always fast).
    pub fn refresh_autodiff_state(&mut self) {
        use burn::module::ModuleMapper;
        struct GraphDetacher;
        impl ModuleMapper<TrainB> for GraphDetacher {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<TrainB, D>>,
            ) -> Param<Tensor<TrainB, D>> {
                let (_, tensor, mapper) = param.consume();
                let fresh = Tensor::from_inner(tensor.inner()).require_grad();
                Param::from_mapped_value(ParamId::new(), fresh, mapper)
            }
        }
        let model = self.model.take().unwrap();
        self.model = Some(model.map(&mut GraphDetacher));
        self.optimizer = AdamWConfig::new().init::<TrainB, PerceptionModel<TrainB>>();
    }

    pub fn region_sizes(&self) -> [usize; 5] {
        let m = self.model.as_ref().unwrap();
        [
            m.speech_layer.n_units,
            m.env_layer.n_units,
            m.visual_layer.n_units,
            m.symbolic_layer.n_units,
            m.language_layer.n_units,
        ]
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.current_lr = lr;
    }

    pub fn learning_rate(&self) -> f64 {
        self.current_lr
    }

    // ── Save / Load ──────────────────────────────────────────────────────────

    /// Save weights to a binary file (burn BinFileRecorder).
    /// Also writes a `<path>.sizes` sidecar with grown region sizes.
    pub fn save_weights(&mut self, path: &std::path::Path) -> PerceptionResult<()> {
        let sizes = self.region_sizes();
        let sizes_str = sizes
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(path.with_extension("sizes"), &sizes_str)
            .map_err(|e| PerceptionError::Burn(e.to_string()))?;

        // Clone is cheap (Arc-backed tensors); saves clone, keeps original in self.model.
        let model_clone = self.model.as_ref().unwrap().clone();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let path_str = path
            .to_str()
            .ok_or_else(|| PerceptionError::Burn("invalid path".into()))?;
        model_clone
            .save_file(path_str, &recorder)
            .map_err(|e| PerceptionError::Burn(e.to_string()))
    }

    /// Load weights from a binary file written by `save_weights`.
    pub fn load_weights(&mut self, path: &std::path::Path) -> PerceptionResult<()> {
        let device = self.device;
        // Extract path string before taking the model so an invalid-path error
        // doesn't leave self.model = None.
        let path_str = path
            .to_str()
            .ok_or_else(|| PerceptionError::Burn("invalid path".into()))?;
        let saved_sizes = self.region_sizes();
        let model = self.model.take().unwrap();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        match model.load_file(path_str, &recorder, &device) {
            Ok(loaded) => {
                self.ema_heads = EmaOutputHeads::from_model(&loaded);
                self.model = Some(loaded);
                Ok(())
            }
            Err(e) => {
                // load_file consumes the model whether it succeeds or not.
                // Restore a fresh model grown to the same sizes so all callers
                // remain functional after a failed weight restore.
                self.model = Some(PerceptionModel::new(&device));
                for (i, &region) in [
                    RegionId::AudioSpeech,
                    RegionId::AudioEnv,
                    RegionId::Visual,
                    RegionId::Symbolic,
                    RegionId::Language,
                ]
                .iter()
                .enumerate()
                {
                    let n_new = saved_sizes[i].saturating_sub(N_UNITS);
                    if n_new > 0 {
                        self.grow_region(region, n_new);
                    }
                }
                Err(PerceptionError::Burn(e.to_string()))
            }
        }
    }

    /// Export all inference-path weights to a directory for ONNX graph construction.
    ///
    /// Writes one raw float32 `.f32` file per weight tensor plus a JSON manifest
    /// (`weights_manifest.json`) so `scripts/export_perception_graph.py` can load them
    /// and build the two ONNX inference graphs.
    ///
    /// Weight naming convention (must match Python script):
    ///   W_enc_{i}              – KWtaLayer linear, shape [n_in, n_units]  (i = 0..4)
    ///   W_cross                – cross_modal_proj, shape [CROSS_MODAL_DIM, 384]
    ///   W_region_{i}           – BindingArea region projection, shape [N_UNITS, D_PROJ]
    ///   W_q / W_k / W_v        – Attention QKV, shape [D_PROJ, D_PROJ]
    ///   gru_{gate}_input_W     – GRU input weight, shape [GRU_INPUT, GRU_HIDDEN]  (gate=update/reset/new)
    ///   gru_{gate}_hidden_W    – GRU hidden weight, shape [GRU_HIDDEN, GRU_HIDDEN]
    ///   W_temporal_proj        – GRU hidden → EPISODE_DIM, shape [GRU_HIDDEN, EPISODE_DIM]
    pub fn export_weights_for_onnx(&self, dir: &std::path::Path) -> PerceptionResult<()> {
        std::fs::create_dir_all(dir).map_err(|e| PerceptionError::Burn(e.to_string()))?;
        let m = self.model.as_ref().unwrap();

        let mut manifest: Vec<serde_json::Value> = Vec::new();

        let mut write = |name: &str, data: Vec<f32>, shape: Vec<usize>| -> PerceptionResult<()> {
            let fname = format!("{}.f32", name);
            let path = dir.join(&fname);
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(&path, &bytes).map_err(|e| PerceptionError::Burn(e.to_string()))?;
            manifest.push(serde_json::json!({
                "name":  name,
                "shape": shape,
                "file":  fname,
            }));
            Ok(())
        };

        // ── KWtaLayer encoder weights [n_in, n_units] ────────────────────────
        let enc_layers = [
            (&m.speech_layer.linear, 384usize),
            (&m.env_layer.linear, 512usize),
            (&m.visual_layer.linear, 384usize),
            (&m.symbolic_layer.linear, 384usize),
            (&m.language_layer.linear, 896usize),
        ];
        for (i, (layer, n_in)) in enc_layers.iter().enumerate() {
            let n_units = layer.weight.val().dims()[1];
            let w: Vec<f32> = layer
                .weight
                .val()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            write(&format!("W_enc_{i}"), w, vec![*n_in, n_units])?;
        }

        // ── Cross-modal projection [CROSS_MODAL_DIM, 384] + bias [384] ──────
        {
            let dims = m.cross_modal_proj.weight.val().dims();
            let w: Vec<f32> = m
                .cross_modal_proj
                .weight
                .val()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            write("W_cross", w, vec![dims[0], dims[1]])?;
            // cross_modal_proj has bias=true; bias shape [384]
            if let Some(bias) = m.cross_modal_proj.bias.as_ref() {
                let b: Vec<f32> = bias.val().into_data().convert::<f32>().to_vec().unwrap();
                write("W_cross_bias", b, vec![dims[1]])?;
            }
        }

        // ── BindingArea region projections [N_UNITS, D_PROJ] ─────────────────
        let region_projs = [
            &m.binding.region_0,
            &m.binding.region_1,
            &m.binding.region_2,
            &m.binding.region_3,
            &m.binding.region_4,
        ];
        for (i, proj) in region_projs.iter().enumerate() {
            let dims = proj.weight.val().dims();
            let w: Vec<f32> = proj
                .weight
                .val()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            write(&format!("W_region_{i}"), w, vec![dims[0], dims[1]])?;
        }

        // ── Attention QKV [D_PROJ, D_PROJ] ───────────────────────────────────
        for (name, param) in [
            ("W_q", &m.binding.w_q),
            ("W_k", &m.binding.w_k),
            ("W_v", &m.binding.w_v),
        ] {
            let dims = param.weight.val().dims();
            let w: Vec<f32> = param
                .weight
                .val()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            write(name, w, vec![dims[0], dims[1]])?;
        }

        // ── GRU weights [in, out] (burn convention; Python script transposes) ─
        let gru = &m.binding.temporal_gru;
        for (gate_name, gate) in [
            ("update", &gru.update_gate),
            ("reset", &gru.reset_gate),
            ("new", &gru.new_gate),
        ] {
            let wi_dims = gate.input_transform.weight.val().dims();
            let wi: Vec<f32> = gate
                .input_transform
                .weight
                .val()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            write(
                &format!("gru_{gate_name}_input_W"),
                wi,
                vec![wi_dims[0], wi_dims[1]],
            )?;

            let wh_dims = gate.hidden_transform.weight.val().dims();
            let wh: Vec<f32> = gate
                .hidden_transform
                .weight
                .val()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            write(
                &format!("gru_{gate_name}_hidden_W"),
                wh,
                vec![wh_dims[0], wh_dims[1]],
            )?;
        }

        // ── temporal_proj [GRU_HIDDEN, EPISODE_DIM] ──────────────────────────
        {
            let dims = m.binding.temporal_proj.weight.val().dims();
            let w: Vec<f32> = m
                .binding
                .temporal_proj
                .weight
                .val()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            write("W_temporal_proj", w, vec![dims[0], dims[1]])?;
        }

        // ── Write manifest ────────────────────────────────────────────────────
        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| PerceptionError::Burn(e.to_string()))?;
        std::fs::write(dir.join("weights_manifest.json"), manifest_json)
            .map_err(|e| PerceptionError::Burn(e.to_string()))?;

        tracing::info!(
            n_tensors = manifest.len(),
            "perception weights exported for ONNX"
        );
        Ok(())
    }

    // ── ONNX validation helpers ──────────────────────────────────────────────

    /// Return per-region duty cycles (for ONNX validation: must pass same dc to Python topk).
    pub fn duty_cycles(&self) -> [Vec<f32>; 5] {
        let m = self.model.as_ref().unwrap();
        [
            m.speech_layer.duty_cycle.clone(),
            m.env_layer.duty_cycle.clone(),
            m.visual_layer.duty_cycle.clone(),
            m.symbolic_layer.duty_cycle.clone(),
            m.language_layer.duty_cycle.clone(),
        ]
    }

    /// Run just the 5 encoder linear forwards + cross-modal projection, no topk, no side effects.
    /// Returns raw pre-topk activations [z_speech, z_env, z_visual, z_symbolic, z_language].
    /// Language input is zero-padded when not provided.
    pub fn encoder_z_only(
        &self,
        speech: &[f32],
        env: &[f32],
        visual: &[f32],
        language: Option<&[f32]>,
    ) -> [Vec<f32>; 5] {
        let device = self.device;
        let m = self.model.as_ref().unwrap();

        let lang_zeros: Vec<f32>;
        let lang = match language {
            Some(l) => l,
            None => {
                lang_zeros = vec![0.0f32; REGION_DIMS[4]];
                &lang_zeros
            }
        };

        // cross_modal_proj → symbolic encoder input
        let mut concat = vec![0.0f32; CROSS_MODAL_DIM];
        let offsets = [(0usize, 384usize), (384, 512), (896, 384), (1280, 896)];
        let srcs: [&[f32]; 4] = [speech, env, visual, lang];
        for (src, (off, dim)) in srcs.iter().zip(offsets.iter()) {
            let len = src.len().min(*dim);
            concat[*off..*off + len].copy_from_slice(&src[..len]);
        }
        let cross_t: Tensor<NdArray, 2> =
            Tensor::from_data(TensorData::new(concat, [1, CROSS_MODAL_DIM]), &device);
        let symbolic_enc: Vec<f32> = m
            .cross_modal_proj
            .forward(Tensor::<TrainB, 2>::from_inner(cross_t))
            .inner()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let enc_inputs: [&[f32]; 5] = [speech, env, visual, &symbolic_enc, lang];
        let layers = [
            &m.speech_layer.linear,
            &m.env_layer.linear,
            &m.visual_layer.linear,
            &m.symbolic_layer.linear,
            &m.language_layer.linear,
        ];

        std::array::from_fn(|i| {
            let t: Tensor<NdArray, 2> = Tensor::from_data(
                TensorData::new(enc_inputs[i].to_vec(), [1, enc_inputs[i].len()]),
                &device,
            );
            layers[i]
                .forward(Tensor::<TrainB, 2>::from_inner(t))
                .inner()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap()
        })
    }

    /// Run just the binding area on pre-built sparse activations. No side effects.
    /// sparse_z: 5 dense [N_UNITS] vectors (non-zero entries = active units)
    /// Returns (episode [EPISODE_DIM], gru_hidden_out [GRU_HIDDEN]).
    pub fn binding_only(
        &mut self,
        sparse_z: &[Vec<f32>; 5],
        prev_ep: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>) {
        let device = self.device;
        let acts: [SparseActivation; 5] = std::array::from_fn(|i| {
            let dense = &sparse_z[i];
            let mut indices = Vec::with_capacity(K_ACTIVE);
            let mut values = Vec::with_capacity(K_ACTIVE);
            for (j, &v) in dense.iter().enumerate() {
                if v != 0.0 {
                    indices.push(j as u16);
                    values.push(v);
                }
            }
            SparseActivation {
                region: [
                    RegionId::AudioSpeech,
                    RegionId::AudioEnv,
                    RegionId::Visual,
                    RegionId::Symbolic,
                    RegionId::Language,
                ][i],
                indices,
                values,
                n_units: N_UNITS,
            }
        });
        let act_refs: [&SparseActivation; 5] = [&acts[0], &acts[1], &acts[2], &acts[3], &acts[4]];
        let ep = self.model.as_mut().unwrap().binding.forward(
            &act_refs,
            prev_ep.map(|p| p.to_vec()).as_ref(),
            &device,
        );
        let gru_h = self
            .model
            .as_ref()
            .unwrap()
            .binding
            .gru_hidden
            .clone()
            .unwrap_or_else(|| vec![0.0; 256]);
        (ep.0, gru_h)
    }

    /// Pre-grow regions to `sizes`, then load weights.
    pub fn load_weights_with_sizes(
        &mut self,
        path: &std::path::Path,
        sizes: [usize; 5],
    ) -> PerceptionResult<()> {
        let regions = [
            RegionId::AudioSpeech,
            RegionId::AudioEnv,
            RegionId::Visual,
            RegionId::Symbolic,
            RegionId::Language,
        ];
        let current = self.region_sizes();
        for (i, &region) in regions.iter().enumerate() {
            if sizes[i] > current[i] {
                self.grow_region(region, sizes[i] - current[i]);
            }
        }
        self.load_weights(path)
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Returns `(refined_episode, iters_run, final_delta)`.
    ///
    /// Three execution paths, selected at runtime:
    ///   A. ORT encoder + topk_no_dc + ORT binding  (encoder + binding sessions active)
    ///   B. burn kwta   + ORT binding               (binding session only)
    ///   C. burn kwta   + burn binding               (no ORT)
    ///
    /// In all ORT paths the live GRU hidden state is saved before the loop and restored
    /// after each binding call so settle iterations don't advance cross-episode state.
    fn settle_episode(&mut self, ep: &EpisodeVector) -> (EpisodeVector, usize, f32) {
        let device = self.device;
        let use_ort = self.ort_session.is_some();
        let use_encoder = use_ort && self.ort_session.as_ref().is_some_and(|s| s.has_encoder());
        let saved_hidden = if use_ort {
            self.ort_session.as_ref().map(|s| s.gru_hidden.clone())
        } else {
            None
        };

        let mut current = ep.clone();
        let mut last_delta = 0.0f32;

        for iter in 0..self.settle_steps {
            // EMA predictions — pure Rust matmul, no borrow of model or ort.
            let mut preds: [Vec<f32>; 5] = Default::default();
            for (i, pred) in preds.iter_mut().enumerate() {
                *pred = self.ema_heads.predict(&current.0, i);
            }

            let refined = if use_encoder {
                // ── Path A: ORT encoder → topk_no_dc → ORT binding ──────────────
                // Encoder takes 4 raw modality inputs (speech/env/visual/lang);
                // symbolic is produced by the encoder's internal cross-modal projection.
                const ENC_REGIONS: [(usize, usize); 4] = [(0, 384), (1, 512), (2, 384), (4, 896)];
                let mut blended_enc: [Vec<f32>; 4] = Default::default();
                for (slot, &(ri, default_dim)) in ENC_REGIONS.iter().enumerate() {
                    blended_enc[slot] = match self.latest_inputs[ri].as_deref() {
                        Some(raw) => {
                            let rn = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                            let pn = preds[ri]
                                .iter()
                                .map(|x| x * x)
                                .sum::<f32>()
                                .sqrt()
                                .max(1e-6);
                            let amp = rn / pn;
                            raw.iter()
                                .zip(&preds[ri])
                                .map(|(&r, &p)| {
                                    (1.0 - self.settle_beta) * r + self.settle_beta * p * amp
                                })
                                .collect()
                        }
                        None => vec![0.0f32; default_dim],
                    };
                }

                // ORT encoder — borrow ends when enc_result is returned.
                let enc_result = self.ort_session.as_mut().unwrap().run_encoder(
                    &blended_enc[0],
                    &blended_enc[1],
                    &blended_enc[2],
                    &blended_enc[3],
                );

                match enc_result {
                    Ok([z0, z1, z2, z3, z4]) => {
                        // topk_no_dc: immutable model borrow, dropped before binding.
                        let sparse_z: [Vec<f32>; 5] = {
                            let m = self.model.as_ref().unwrap();
                            [
                                m.speech_layer.topk_from_z_no_dc(z0).to_dense_n(N_UNITS),
                                m.env_layer.topk_from_z_no_dc(z1).to_dense_n(N_UNITS),
                                m.visual_layer.topk_from_z_no_dc(z2).to_dense_n(N_UNITS),
                                m.symbolic_layer.topk_from_z_no_dc(z3).to_dense_n(N_UNITS),
                                m.language_layer.topk_from_z_no_dc(z4).to_dense_n(N_UNITS),
                            ]
                        };
                        // ORT binding — restore hidden so settle doesn't advance live GRU.
                        let ort = self.ort_session.as_mut().unwrap();
                        match ort.run_binding(&sparse_z, &current.0) {
                            Ok(ep_vec) => {
                                if let Some(ref h) = saved_hidden {
                                    ort.gru_hidden = h.clone();
                                }
                                EpisodeVector(ep_vec)
                            }
                            Err(e) => {
                                tracing::warn!(?e, "ORT binding in settle (encoder path) failed");
                                if let Some(ref h) = saved_hidden {
                                    ort.gru_hidden = h.clone();
                                }
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(?e, "ORT encoder in settle failed; aborting settle");
                        break;
                    }
                }
            } else {
                // ── Paths B & C: burn kwta (both), then ORT or burn binding ─────
                let mut acts: [Option<SparseActivation>; 5] = Default::default();
                {
                    let model = self.model.as_mut().unwrap();
                    for i in 0..5 {
                        if let Some(raw) = &self.latest_inputs[i] {
                            let rn = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                            let pn = preds[i].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                            let amp = rn / pn;
                            let blended: Vec<f32> = raw
                                .iter()
                                .zip(&preds[i])
                                .map(|(&r, &p)| {
                                    (1.0 - self.settle_beta) * r + self.settle_beta * p * amp
                                })
                                .collect();
                            acts[i] = Some(match i {
                                0 => model.speech_layer.forward_no_dc(&blended, &device),
                                1 => model.env_layer.forward_no_dc(&blended, &device),
                                2 => model.visual_layer.forward_no_dc(&blended, &device),
                                3 => model.symbolic_layer.forward_no_dc(&blended, &device),
                                4 => model.language_layer.forward_no_dc(&blended, &device),
                                _ => unreachable!(),
                            });
                        }
                    }
                } // model borrow dropped

                if use_ort {
                    // ── Path B: ORT binding ──────────────────────────────────────
                    let sparse_z: [Vec<f32>; 5] = std::array::from_fn(|i| {
                        acts[i]
                            .as_ref()
                            .map(|a| a.to_dense_n(N_UNITS))
                            .unwrap_or_else(|| vec![0.0f32; N_UNITS])
                    });
                    let ort = self.ort_session.as_mut().unwrap();
                    match ort.run_binding(&sparse_z, &current.0) {
                        Ok(ep_vec) => {
                            if let Some(ref h) = saved_hidden {
                                ort.gru_hidden = h.clone();
                            }
                            EpisodeVector(ep_vec)
                        }
                        Err(e) => {
                            tracing::warn!(
                                ?e,
                                "ORT binding in settle failed; falling back to burn"
                            );
                            if let Some(ref h) = saved_hidden {
                                ort.gru_hidden = h.clone();
                            }
                            let (a0, a1, a2, a3) = match (&acts[0], &acts[1], &acts[2], &acts[3]) {
                                (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                                _ => break,
                            };
                            let lang_zero = SparseActivation {
                                region: RegionId::Language,
                                indices: vec![],
                                values: vec![],
                                n_units: N_UNITS,
                            };
                            let a4 = acts[4].as_ref().unwrap_or(&lang_zero);
                            self.model.as_mut().unwrap().binding.forward(
                                &[a0, a1, a2, a3, a4],
                                None,
                                &device,
                            )
                        }
                    }
                } else {
                    // ── Path C: burn binding ─────────────────────────────────────
                    let (a0, a1, a2, a3) = match (&acts[0], &acts[1], &acts[2], &acts[3]) {
                        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                        _ => break,
                    };
                    let lang_zero = SparseActivation {
                        region: RegionId::Language,
                        indices: vec![],
                        values: vec![],
                        n_units: N_UNITS,
                    };
                    let a4 = acts[4].as_ref().unwrap_or(&lang_zero);
                    self.model.as_mut().unwrap().binding.forward(
                        &[a0, a1, a2, a3, a4],
                        None,
                        &device,
                    )
                }
            };

            let delta: f32 = current
                .0
                .iter()
                .zip(&refined.0)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            last_delta = delta;
            tracing::debug!(iter, delta, "settle delta");
            current = refined;
            if delta < SETTLE_EPS {
                return (current, iter + 1, last_delta);
            }
        }

        (current, self.settle_steps, last_delta)
    }

    fn update_prior_predictions(&mut self, episode: &[f32]) {
        for i in 0..5 {
            self.prior_predictions[i] = Some(self.ema_heads.predict(episode, i));
        }
    }

    fn blend_with_prior(&self, values: &[f32], idx: usize) -> (Vec<f32>, Option<f32>) {
        if self.prior_beta <= 0.0 {
            return (values.to_vec(), None);
        }
        match &self.prior_predictions[idx] {
            None => (values.to_vec(), None),
            Some(prior) => {
                let cosine = cosine_sim(values, prior);
                let beta = self.prior_beta;
                let raw_norm = values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                let prior_norm = prior.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                let amp_scale = raw_norm / prior_norm;
                let blended = values
                    .iter()
                    .zip(prior)
                    .map(|(&r, &p)| (1.0 - beta) * r + beta * p * amp_scale)
                    .collect();
                (blended, Some(cosine))
            }
        }
    }

    fn push_to_wm(&mut self, episode: &Option<EpisodeVector>) {
        if self.wm_size == 0 {
            return;
        }
        if let Some(ep) = episode {
            self.working_memory.push_back(ep.0.clone());
            while self.working_memory.len() > self.wm_size {
                self.working_memory.pop_front();
            }
        }
    }

    fn layer_mut(model: &mut PerceptionModel<TrainB>, region: RegionId) -> &mut KWtaLayer<TrainB> {
        match region {
            RegionId::AudioSpeech => &mut model.speech_layer,
            RegionId::AudioEnv => &mut model.env_layer,
            RegionId::Visual => &mut model.visual_layer,
            RegionId::Symbolic => &mut model.symbolic_layer,
            RegionId::Language => &mut model.language_layer,
        }
    }

    fn decoder_mut(
        model: &mut PerceptionModel<TrainB>,
        region: RegionId,
    ) -> &mut SaeDecoder<TrainB> {
        match region {
            RegionId::AudioSpeech => &mut model.speech_decoder,
            RegionId::AudioEnv => &mut model.env_decoder,
            RegionId::Visual => &mut model.visual_decoder,
            RegionId::Symbolic => &mut model.symbolic_decoder,
            RegionId::Language => &mut model.language_decoder,
        }
    }
}

/// Scale a Linear weight tensor in-place, preserving its ParamId for optimizer compat.
fn scale_weight(linear: &mut Linear<TrainB>, scale: f32) {
    let id = linear.weight.id;
    let new_w = linear.weight.val().mul_scalar(scale);
    linear.weight = Param::initialized(id, new_w);
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn apply_gain(values: &[f32], gain: f32) -> Vec<f32> {
    if (gain - 1.0).abs() < 1e-6 {
        return values.to_vec();
    }
    values.iter().map(|&x| x * gain).collect()
}
