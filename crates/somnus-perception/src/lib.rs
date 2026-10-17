pub mod binding;
pub mod error;
pub mod onnx_session;
pub mod region;
pub mod sae;
mod tensor_ext;
pub mod wake_worker;
pub use tensor_ext::TensorF32Ext;

pub use binding::{BindingArea, EpisodeVector, EPISODE_DIM};
pub use error::{PerceptionError, PerceptionResult};
pub use onnx_session::OnnxPerceptionSession;
pub use region::{scaled_k, KWtaLayer, RegionId, SparseActivation, K_ACTIVE, N_UNITS};
pub use sae::SaeDecoder;
pub use wake_worker::WakeReconWorker;

use burn::{
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    optim::{AdamWConfig, GradientsParams, ModuleOptimizer},
    tensor::{Device, Distribution, Int, Tensor, TensorData},
};
use std::collections::VecDeque;

type PercOptim = ModuleOptimizer;

/// EMA (exponential moving average) copies of the 5 output heads, used for all
/// inference paths (top-down prior generation, settling).  Updated after every
/// wake_reconstruct_step; decoupled from the autodiff training heads so that
/// noisy per-frame gradients don't immediately destabilise prior prediction.
const EMA_ALPHA: f32 = 0.99;

#[derive(Clone)]
pub(crate) struct EmaOutputHeads {
    /// EMA tensors for the weight update step (NdArray, no autodiff).
    heads: [burn::tensor::Tensor<2>; 5],
    /// Flat row-major copies of the EMA weights for allocation-free inference.
    /// Layout: [d_in × d_out] (burn stores weight as [d_in, d_out]).
    /// Updated by `sync_flat` after every EMA weight update.
    flat: [Vec<f32>; 5],
    /// d_out for each head: REGION_DIMS = [384, ENV_DIM, 384, 384, LANG_HIDDEN].
    d_out: [usize; 5],
}

impl EmaOutputHeads {
    fn from_model(model: &PerceptionModel) -> Self {
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

    fn update(&mut self, model: &PerceptionModel) {
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
    pub(crate) fn update_from_weights(&mut self, live: [burn::tensor::Tensor<2>; 5]) {
        somnus_metrics::timed_core!(somnus_metrics::Metric::EmaUpdateFromWeights, {
            for (ema, w) in self.heads.iter_mut().zip(live) {
                *ema = ema.clone() * EMA_ALPHA + w * (1.0 - EMA_ALPHA);
            }
            self.sync_flat();
        });
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
        somnus_metrics::timed_core!(somnus_metrics::Metric::EmaSyncFlat, {
            for (i, h) in self.heads.iter().enumerate() {
                let dims = h.dims();
                self.d_out[i] = dims[1];
                self.flat[i] = h.clone().to_f32_vec();
            }
        });
    }

    /// Inference-only head forward: ep_data @ weight → Vec<f32>.
    /// Uses sgemm for AVX2/FMA-accelerated GEMV.
    fn predict(&self, ep_data: &[f32], idx: usize) -> Vec<f32> {
        let (out, _) = somnus_metrics::timed_core!(somnus_metrics::Metric::EmaPredict, {
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
        });
        out
    }
}

/// LM hidden dimension for the language region — must match the action-decoder LM_HIDDEN
/// in crates/somnus/src/action.rs (Qwen3-0.6B, ADR 0035). Cold-started projection/output-head
/// weights are shape-locked to it.
pub const LANG_HIDDEN: usize = 1024;

/// AudioEnv region input dimension. 512 = CLAP-HTSAT (default). 960 = EfficientAT mn10
/// (gaps.md item 9 A/B candidate, `--features mn10_env`) — a cold-start-required swap since
/// the two encoders produce differently-shaped embeddings; not a runtime-selectable flag.
#[cfg(not(feature = "mn10_env"))]
pub const ENV_DIM: usize = 512;
#[cfg(feature = "mn10_env")]
pub const ENV_DIM: usize = 960;

/// Per-region input dimensions. Order: [AudioSpeech, AudioEnv, Visual, Symbolic, Language].
pub const REGION_DIMS: [usize; 5] = [384, ENV_DIM, 384, 384, LANG_HIDDEN];

/// Input dimension for the symbolic cross-modal projection:
/// speech(384) + env(ENV_DIM) + visual(384) + language(LANG_HIDDEN). Missing modalities are zero-padded.
pub const CROSS_MODAL_DIM: usize = 384 + ENV_DIM + 384 + LANG_HIDDEN;

/// Byte offsets into the CROSS_MODAL_DIM concat buffer — see `compute_symbolic_cross_modal`.
const CM_OFF_SPEECH: usize = 0;
const CM_OFF_ENV: usize = CM_OFF_SPEECH + 384;
const CM_OFF_VISUAL: usize = CM_OFF_ENV + ENV_DIM;
const CM_OFF_LANG: usize = CM_OFF_VISUAL + 384;

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
pub struct PerceptionModel {
    pub speech_layer: KWtaLayer,
    pub env_layer: KWtaLayer,
    pub visual_layer: KWtaLayer,
    pub symbolic_layer: KWtaLayer,
    pub language_layer: KWtaLayer,
    pub speech_decoder: SaeDecoder,
    pub env_decoder: SaeDecoder,
    pub visual_decoder: SaeDecoder,
    pub symbolic_decoder: SaeDecoder,
    pub language_decoder: SaeDecoder,
    pub binding: BindingArea,
    /// Lagged snapshot of `binding`'s trained tensors (region_0..4, w_q/w_k/w_v),
    /// hard-copied from `binding` only every `--binding-wake-lag-cycles` sleep
    /// cycles (ADR 0048). All wake-time inference (`try_bind`/`settle_episode`/
    /// `binding_only`) and ORT export read from this copy, never from `binding`
    /// directly — `binding` is the training-only copy `binding_replay_step_batch`
    /// writes to. Decouples "how often the binder trains" from "how often wake
    /// sees a new binder generation", the confirmed root cause of gaps.md §33's
    /// mean_output_mse regression (engineering-log.md §47).
    pub wake_binding: BindingArea,
    pub speech_head: Linear,
    pub env_head: Linear,
    pub visual_head: Linear,
    pub symbolic_head: Linear,
    pub language_head: Linear,
    /// Projects concatenated cross-modal encoder inputs → 384-dim symbolic input.
    /// Input: [speech(384) | env(ENV_DIM) | visual(384) | language(LANG_HIDDEN)] = CROSS_MODAL_DIM,
    /// zero-padded for any missing modality.
    pub cross_modal_proj: Linear,
}

impl PerceptionModel {
    /// k-WTA layer for region index `idx` (0 speech, 1 env, 2 visual, 3 symbolic, 4 language).
    pub fn layer(&self, idx: usize) -> &KWtaLayer {
        match idx {
            0 => &self.speech_layer,
            1 => &self.env_layer,
            2 => &self.visual_layer,
            3 => &self.symbolic_layer,
            4 => &self.language_layer,
            _ => unreachable!("region index {idx}"),
        }
    }

    pub fn layer_mut(&mut self, idx: usize) -> &mut KWtaLayer {
        match idx {
            0 => &mut self.speech_layer,
            1 => &mut self.env_layer,
            2 => &mut self.visual_layer,
            3 => &mut self.symbolic_layer,
            4 => &mut self.language_layer,
            _ => unreachable!("region index {idx}"),
        }
    }

    /// SAE decoder for region index `idx`.
    pub fn decoder(&self, idx: usize) -> &SaeDecoder {
        match idx {
            0 => &self.speech_decoder,
            1 => &self.env_decoder,
            2 => &self.visual_decoder,
            3 => &self.symbolic_decoder,
            4 => &self.language_decoder,
            _ => unreachable!("region index {idx}"),
        }
    }

    pub fn new(device: &Device) -> Self {
        // wake_binding must start byte-identical to binding (ADR 0048). A plain
        // `binding.clone()` does NOT achieve this: burn's derived `Module` Clone
        // re-runs each `Param`'s lazy random initializer independently per clone
        // (confirmed empirically — `Linear::clone()` does not preserve a
        // materialized weight value, only `Tensor::clone()` does), so two
        // BindingArea::new() calls or one clone() would each draw fresh random
        // weights. Copy each trained tensor's already-materialized value
        // explicitly instead — the same pattern `sync_wake_binding` uses later.
        let binding = BindingArea::new(device);
        let mut wake_binding = BindingArea::new(device);
        macro_rules! seed_wake_binding_field {
            ($field:ident) => {
                wake_binding.$field.weight =
                    Param::initialized(ParamId::new(), binding.$field.weight.val());
            };
        }
        seed_wake_binding_field!(region_0);
        seed_wake_binding_field!(region_1);
        seed_wake_binding_field!(region_2);
        seed_wake_binding_field!(region_3);
        seed_wake_binding_field!(region_4);
        seed_wake_binding_field!(w_q);
        seed_wake_binding_field!(w_k);
        seed_wake_binding_field!(w_v);
        wake_binding.copy_temporal_from(&binding);
        Self {
            speech_layer: KWtaLayer::new(RegionId::AudioSpeech, 384, device),
            env_layer: KWtaLayer::new(RegionId::AudioEnv, ENV_DIM, device),
            visual_layer: KWtaLayer::new(RegionId::Visual, 384, device),
            symbolic_layer: KWtaLayer::new(RegionId::Symbolic, 384, device),
            language_layer: KWtaLayer::new(RegionId::Language, LANG_HIDDEN, device),
            speech_decoder: SaeDecoder::new(RegionId::AudioSpeech, 384, device),
            env_decoder: SaeDecoder::new(RegionId::AudioEnv, ENV_DIM, device),
            visual_decoder: SaeDecoder::new(RegionId::Visual, 384, device),
            symbolic_decoder: SaeDecoder::new(RegionId::Symbolic, 384, device),
            language_decoder: SaeDecoder::new(RegionId::Language, LANG_HIDDEN, device),
            binding,
            wake_binding,
            speech_head: LinearConfig::new(EPISODE_DIM, 384)
                .with_bias(false)
                .init(device),
            env_head: LinearConfig::new(EPISODE_DIM, ENV_DIM)
                .with_bias(false)
                .init(device),
            visual_head: LinearConfig::new(EPISODE_DIM, 384)
                .with_bias(false)
                .init(device),
            symbolic_head: LinearConfig::new(EPISODE_DIM, 384)
                .with_bias(false)
                .init(device),
            language_head: LinearConfig::new(EPISODE_DIM, LANG_HIDDEN)
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
    model: Option<PerceptionModel>,
    optimizer: PercOptim,
    current_lr: f64,
    device: Device,

    /// EMA copies of the output heads used for all inference paths (prior, settle).
    /// Decoupled from the autodiff training heads so gradient noise doesn't
    /// immediately destabilise prior prediction.
    ema_heads: EmaOutputHeads,

    prior_predictions: [Option<Vec<f32>>; 5],
    prior_betas: [f32; 5],
    latest: [Option<SparseActivation>; 5],
    latest_inputs: [Option<Vec<f32>>; 5],
    settle_steps: usize,
    settle_beta: f32,
    settle_gate: f32,
    /// L2 norm of the encoder columns given to new units by `grow_region_seeded` (default 0.01,
    /// the historical value; gaps.md §50 experiment `--grown-unit-init-scale`).
    grown_init_scale: f32,
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

    /// L1 sparsity penalty coefficient added to the SAE reconstruction loss during NREM.
    /// 0.0 (default) = off; positive values encourage sparser codes.
    sae_l1_coeff: f32,

    /// Use batch-quantile k-WTA threshold instead of per-sample top-k during NREM training.
    /// false (default) = per-sample top-k; true = batch-quantile threshold.
    batch_topk: bool,

    /// DER++ Phase 3: auto-adapt the weight on `wake_loss` (MSE against stale stored
    /// outputs) instead of a fixed implicit λ=1. false (default) = off, unweighted sum.
    der_auto_lambda: bool,
    /// Running mean/mean-square of `wake_loss` (Welford-lite EWMA) used to derive λ =
    /// 1/variance when `der_auto_lambda` is on. `None` until the first sample is seen.
    der_lambda_stats: Option<(f32, f32)>,
}

/// One region's duty-cycle diagnostics (see [`PerceptionProcessor::duty_cycle_report`]).
/// `sum_dc_*` is the total expected active slots per ingest contributed by the original / grown
/// units (their sum equals `k_active` in steady state). `grown_units_ever_fired` counts grown
/// units with duty cycle above 1e-4 (EMA alpha 1/1000, so "fired recently").
#[derive(Debug, Clone)]
pub struct RegionDutyReport {
    pub region: usize,
    pub n_units: usize,
    pub k_active: usize,
    pub sum_dc_base: f32,
    pub sum_dc_grown: f32,
    pub base_units_over_half: usize,
    pub grown_units_ever_fired: usize,
    pub base_units_silent: usize,
    pub top5: Vec<(usize, f32)>,
    /// Mean |z| of active slots held by original / grown units (cumulative since process start;
    /// NaN when that group never held a slot), and the share of active slots held by grown units.
    pub mean_abs_active_base: f64,
    pub mean_abs_active_grown: f64,
    pub grown_slot_share: f64,
}

impl PerceptionProcessor {
    /// The model is `None` only while moved into an optimizer step (`take()` / put-back);
    /// any access in that window is a logic bug, so fail with a message that says so.
    #[inline]
    fn model_ref(&self) -> &PerceptionModel {
        self.model
            .as_ref()
            .expect("PerceptionProcessor.model accessed while taken for an optimizer step")
    }

    #[inline]
    fn model_mut(&mut self) -> &mut PerceptionModel {
        self.model
            .as_mut()
            .expect("PerceptionProcessor.model accessed while taken for an optimizer step")
    }

    /// `device` is the plain execution device (e.g. `Device::flex()`); training state lives on
    /// its autodiff-enabled counterpart (autodiff is a device property in burn 0.22).
    pub fn new(device: Device) -> Self {
        let device = device.autodiff();
        let model = PerceptionModel::new(&device);
        let ema_heads = EmaOutputHeads::from_model(&model);
        let optimizer = AdamWConfig::new().init();
        Self {
            model: Some(model),
            optimizer,
            ema_heads,
            current_lr: TrainConfig::default().lr,
            device,
            prior_predictions: [None, None, None, None, None],
            prior_betas: [0.0; 5],
            latest: [None, None, None, None, None],
            latest_inputs: [None, None, None, None, None],
            settle_steps: 0,
            settle_beta: 0.5,
            settle_gate: SETTLE_COSINE_GATE_DEFAULT,
            grown_init_scale: 0.01,
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
            sae_l1_coeff: 0.0,
            batch_topk: true,
            der_auto_lambda: false,
            der_lambda_stats: None,
        }
    }

    /// Sync the flat cross_modal_proj weight copy after model weights update (NREM).
    /// Called whenever ORT sessions are rebuilt so the inference copy stays in sync.
    pub fn sync_cross_modal_flat(&mut self) {
        let model = self.model_ref();
        let w: Vec<f32> = model.cross_modal_proj.weight.val().to_f32_vec();
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
        self.ingest_region(RegionId::AudioSpeech, values)
    }

    /// Ingest an ENV_DIM-dim env embedding (CLAP or mn10 depending on feature).
    pub fn ingest_env(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        self.ingest_region(RegionId::AudioEnv, values)
    }

    /// Ingest a 384-dim DINOv2 visual embedding.
    pub fn ingest_visual(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        self.ingest_region(RegionId::Visual, values)
    }

    /// Ingest a LANG_HIDDEN-dim language embedding (mean-pooled LM hidden states / token embeddings).
    pub fn ingest_language(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        self.ingest_region(RegionId::Language, values)
    }

    /// Shared ingest pipeline for the four external modalities: gain -> top-down prior blend ->
    /// k-WTA + reconstruction error (burn path, or a bootstrap placeholder when the ORT encoder
    /// owns the forward) -> cross-modal symbolic -> bind -> settle/prior update -> working memory.
    fn ingest_region(
        &mut self,
        region: RegionId,
        values: &[f32],
    ) -> PerceptionResult<PerceptionOutput> {
        debug_assert_ne!(
            region,
            RegionId::Symbolic,
            "symbolic is derived, never ingested"
        );
        let device = self.device.clone();
        let idx = region.as_index();
        let is_language = region == RegionId::Language;
        if is_language {
            self.last_language_input = Some(values.to_vec());
        }
        let gained = apply_gain(values, self.region_gain[idx]);
        let (input, cosine) = self.blend_with_prior(&gained, idx);
        if let Some(c) = cosine {
            let label = match region {
                RegionId::AudioSpeech => "speech",
                RegionId::AudioEnv => "env",
                RegionId::Visual => "visual",
                RegionId::Symbolic => "symbolic",
                RegionId::Language => "language",
            };
            tracing::info!(region = label, prior_cosine = c, "top-down prior");
        }

        let (err, d0) = somnus_metrics::timed_core!(somnus_metrics::Metric::IngestKwta, {
            if self.ort_session.as_ref().is_some_and(|s| s.has_encoder()) {
                // ORT encoder recomputes all 5 regions in try_bind; skip burn forward.
                // Bootstrap latest[idx] if needed so the try_bind gate can fire. Language is
                // not checked in that gate (indices 0-3 only) and try_bind updates latest[4]
                // via topk_from_z, so it needs no bootstrap.
                if !is_language && self.latest[idx].is_none() {
                    let n = self.model_ref().layer(idx).n_units;
                    self.latest[idx] = Some(bootstrap_placeholder(region, n));
                }
                0.0
            } else {
                let act = self.model_mut().layer_mut(idx).forward(&input, &device);
                let e = self
                    .model_ref()
                    .decoder(idx)
                    .reconstruction_error(values, &act, &device);
                self.latest[idx] = Some(act);
                e
            }
        });
        self.perf_kwta_us += d0.as_micros() as u64;
        self.latest_inputs[idx] = Some(values.to_vec());

        let ((symbolic_cos, symbolic_err), d1) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::IngestSymbolic,
            self.compute_symbolic_cross_modal()
        );
        if !is_language {
            self.refresh_language_cache();
        }
        self.perf_symbolic_us += d1.as_micros() as u64;

        let latest_ep = self.working_memory.back().cloned();
        let ((episode, regional_inputs), d2) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::IngestBind,
            self.try_bind(latest_ep.as_ref())
        );
        self.perf_bind_us += d2.as_micros() as u64;

        let ((episode, settle_iters, settle_delta), d3) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::IngestPrior,
            self.maybe_settle_and_update(episode, cosine)
        );
        self.perf_prior_us += d3.as_micros() as u64;

        self.push_to_wm(&episode);

        let mut prior_cosines = [None; 5];
        prior_cosines[idx] = cosine;
        prior_cosines[RegionId::Symbolic.as_index()] = symbolic_cos;
        Ok(PerceptionOutput {
            recon_errors: vec![(region, err), (RegionId::Symbolic, symbolic_err)],
            episode,
            regional_inputs,
            settle_iters,
            settle_delta,
            prior_cosines,
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
            let ((refined, iters, delta), d_settle) =
                somnus_metrics::timed_core!(somnus_metrics::Metric::IngestSettle, {
                    if should_settle {
                        self.settle_episode(&ep)
                    } else {
                        (ep, 0, 0.0)
                    }
                });
            self.perf_settle_us += d_settle.as_micros() as u64;
            let (_, d_prior_update) = somnus_metrics::timed_core!(
                somnus_metrics::Metric::IngestPriorUpdate,
                self.update_prior_predictions(&refined.0)
            );
            self.perf_prior_update_us += d_prior_update.as_micros() as u64;
            (Some(refined), iters, delta)
        } else {
            (None, 0, 0.0)
        }
    }

    /// Recompute the Symbolic region as a cross-modal convergence layer.
    ///
    /// Concatenates the current encoder embeddings for speech(384), env(ENV_DIM), visual(384),
    /// and language(LANG_HIDDEN), zero-padding any missing modality, then projects through
    /// `cross_modal_proj` (CROSS_MODAL_DIM→384) before running the symbolic k-WTA.  Called before
    /// every `try_bind` so symbolic always reflects the most current cross-modal picture.
    ///
    /// Returns (symbolic_prior_cosine, symbolic_reconstruction_err).
    fn compute_symbolic_cross_modal(&mut self) -> (Option<f32>, f32) {
        let device = self.device.clone();

        // Build CROSS_MODAL_DIM-dim concat: [speech(384) | env(ENV_DIM) | visual(384) | language(LANG_HIDDEN)]
        let mut concat = vec![0.0f32; CROSS_MODAL_DIM];
        let offsets = [
            (CM_OFF_SPEECH, 384),
            (CM_OFF_ENV, ENV_DIM),
            (CM_OFF_VISUAL, 384),
            (CM_OFF_LANG, LANG_HIDDEN),
        ]; // (offset, dim) per region idx [0,1,2,4]
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
                // Raw matmul: concat[CROSS_MODAL_DIM] × flat[CROSS_MODAL_DIM × 384] → out[384]
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
                let concat_t: Tensor<2> =
                    Tensor::from_data(TensorData::new(concat, [1, CROSS_MODAL_DIM]), &device);
                self.model
                    .as_ref()
                    .unwrap()
                    .cross_modal_proj
                    .forward(concat_t)
                    .to_f32_vec()
            }
        } else {
            let concat_t: Tensor<2> =
                Tensor::from_data(TensorData::new(concat, [1, CROSS_MODAL_DIM]), &device);
            self.model
                .as_ref()
                .unwrap()
                .cross_modal_proj
                .forward(concat_t)
                .to_f32_vec()
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
                let n = self.model_ref().symbolic_layer.n_units;
                self.latest[RegionId::Symbolic.as_index()] =
                    Some(bootstrap_placeholder(RegionId::Symbolic, n));
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
        somnus_metrics::timed_core!(somnus_metrics::Metric::RefreshLanguageCache, {
            // Phase 1.5: when encoder ORT is active, skip — the language z comes from
            // run_encoder in try_bind, which uses latest_inputs[Language] directly.
            if !self.ort_session.as_ref().is_some_and(|s| s.has_encoder()) {
                somnus_metrics::counted_core!(somnus_metrics::Metric::RefreshLanguageCacheRun, 1);
                let device = self.device.clone();
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
            } else {
                somnus_metrics::counted_core!(somnus_metrics::Metric::RefreshLanguageCacheSkipped, 1);
            }
        });
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
                    .unwrap_or_else(|| vec![0.0f32; ENV_DIM]),
                self.latest_inputs[2]
                    .clone()
                    .unwrap_or_else(|| vec![0.0f32; 384]),
                self.latest_inputs[4]
                    .clone()
                    .unwrap_or_else(|| vec![0.0f32; LANG_HIDDEN]),
            ];
            let z_result = self
                .ort_session
                .as_mut()
                .unwrap()
                .run_encoder(&enc_in[0], &enc_in[1], &enc_in[2], &enc_in[3]);
            match z_result {
                Ok([z0, z1, z2, z3, z4]) => {
                    let model = self.model.as_mut().expect(
                        "PerceptionProcessor.model accessed while taken for an optimizer step",
                    );
                    self.latest[0] = Some(model.speech_layer.topk_from_z(z0));
                    self.latest[1] = Some(model.env_layer.topk_from_z(z1));
                    self.latest[2] = Some(model.visual_layer.topk_from_z(z2));
                    self.latest[3] = Some(model.symbolic_layer.topk_from_z(z3));
                    // Only update language if we have a real language input; zeros → garbage z.
                    if self.latest_inputs[4].is_some() {
                        self.latest[4] = Some(model.language_layer.topk_from_z(z4));
                    }
                }
                Err(e) => {
                    tracing::warn!(?e, "ORT encoder failed; using burn activations");
                    somnus_metrics::counted_core!(somnus_metrics::Metric::OrtEncoderFallback, 1);
                }
            }
        }

        // At this point self.latest[0..3] are Some (checked above; may be ORT or burn).
        let lang_zero = SparseActivation {
            region: RegionId::Language,
            indices: vec![],
            values: vec![],
            n_units: self.model_ref().language_layer.n_units,
        };
        let sp = self.latest[0].as_ref().unwrap();
        let en = self.latest[1].as_ref().unwrap();
        let vi = self.latest[2].as_ref().unwrap();
        let sy = self.latest[3].as_ref().unwrap();
        let lang = self.latest[4].as_ref().unwrap_or(&lang_zero);

        let ep = if let Some(ref mut ort) = self.ort_session {
            // ORT binding path — each SparseActivation is padded/truncated to the
            // session's per-region k_active by run_binding itself (gaps.md §24's
            // n_units-mismatch fix generalized: no dense conversion or size lookup
            // needed here at all now).
            let sparse: [SparseActivation; 5] =
                [sp.clone(), en.clone(), vi.clone(), sy.clone(), lang.clone()];
            let prev_zeros = vec![0.0f32; EPISODE_DIM];
            let prev = latest_episode.map(|v| v.as_slice()).unwrap_or(&prev_zeros);
            match ort.run_binding(&sparse, prev) {
                Ok(ep_vec) => EpisodeVector(ep_vec),
                Err(e) => {
                    tracing::warn!(?e, "ORT binding failed; falling back to burn");
                    somnus_metrics::counted_core!(somnus_metrics::Metric::OrtBindingFallback, 1);
                    let device = self.device.clone();
                    self.model
                        .as_mut()
                        .expect(
                            "PerceptionProcessor.model accessed while taken for an optimizer step",
                        )
                        .wake_binding
                        .forward(&[sp, en, vi, sy, lang], latest_episode, &device)
                }
            }
        } else {
            // Burn path (original)
            let device = self.device.clone();
            self.model
                .as_mut()
                .expect("PerceptionProcessor.model accessed while taken for an optimizer step")
                .wake_binding
                .forward(&[sp, en, vi, sy, lang], latest_episode, &device)
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
        let device = self.device.clone();
        let mut total_loss: Option<Tensor<1>> = None;
        let mut n_terms = 0usize;

        // Phase 1: forward passes (duty-cycle updates happen here).
        {
            let model = self.model_mut();
            for (region_idx, input) in regional.iter().enumerate() {
                if input.is_empty() {
                    continue;
                }
                let (act, z) = model.layer_mut(region_idx).forward_tracked(input, &device);

                let n = act.n_units;
                let mut mask_vec = vec![0.0f32; n];
                for &i in &act.indices {
                    mask_vec[i as usize] = 1.0;
                }
                let mask_inner: Tensor<2> =
                    Tensor::from_data(TensorData::new(mask_vec, [1, n]), &device);
                let mask: Tensor<2> = Tensor::from_inner(mask_inner);
                let sparse_z = z * mask;

                let target: Tensor<2> =
                    Tensor::from_data(TensorData::new(input.clone(), [1, input.len()]), &device);

                let loss = model.decoder(region_idx).mse_loss_batch(target, sparse_z);

                total_loss = Some(match total_loss {
                    None => loss,
                    Some(acc) => acc + loss,
                });
                n_terms += 1;
            }
        } // &mut model borrow ends

        // Phase 2: backward + optimizer step.
        if let Some(loss) = total_loss {
            let loss_val: f32 = loss.clone().into_scalar::<f32>();
            let model = self
                .model
                .take()
                .expect("model already taken (re-entrant optimizer step?)");
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
        let device = self.device.clone();
        let batch_topk = self.batch_topk;
        let mut total_loss: Option<Tensor<1>> = None;
        let mut n_terms = 0usize;

        {
            let model = self.model_mut();

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

                let input_batch: Tensor<2> =
                    Tensor::from_data(TensorData::new(flat.clone(), [b, n_in]), &device);

                let (z_batch, flat_masks) = match region_idx {
                    0 => {
                        model
                            .speech_layer
                            .forward_tracked_batch(input_batch.clone(), b, batch_topk)
                    }
                    1 => model
                        .env_layer
                        .forward_tracked_batch(input_batch.clone(), b, batch_topk),
                    2 => {
                        model
                            .visual_layer
                            .forward_tracked_batch(input_batch.clone(), b, batch_topk)
                    }
                    3 => model.symbolic_layer.forward_tracked_batch(
                        input_batch.clone(),
                        b,
                        batch_topk,
                    ),
                    4 => model.language_layer.forward_tracked_batch(
                        input_batch.clone(),
                        b,
                        batch_topk,
                    ),
                    _ => unreachable!(),
                };

                let n_units = flat_masks.len() / b;
                let mask_inner: Tensor<2> =
                    Tensor::from_data(TensorData::new(flat_masks, [b, n_units]), &device);
                let mask: Tensor<2> = Tensor::from_inner(mask_inner);
                let sparse_z = z_batch * mask;

                let target: Tensor<2> =
                    Tensor::from_data(TensorData::new(flat, [b, n_in]), &device);

                let region_loss = model.decoder(region_idx).mse_loss_batch(target, sparse_z);

                total_loss = Some(match total_loss {
                    None => region_loss,
                    Some(acc) => acc + region_loss,
                });
                n_terms += 1;
            }
        }

        if let Some(loss) = total_loss {
            let loss_val: f32 = loss.clone().into_scalar::<f32>();
            let model = self
                .model
                .take()
                .expect("model already taken (re-entrant optimizer step?)");
            let grads = loss.backward();
            let grads_p = GradientsParams::from_grads(grads, &model);
            let model = self.optimizer.step(self.current_lr, model, grads_p);
            self.model = Some(model);
            Ok(loss_val / n_terms.max(1) as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Live-episode binding training step for NREM replay (ADR 0041, gaps.md §33).
    ///
    /// Recomputes episodes live through the binding area from the stored regional
    /// inputs and trains the five output heads against those same regional targets on
    /// the live episodes — the identical targets `wake_reconstruct_step_batch` uses;
    /// only the episode's provenance changes (live-through-binding instead of stored).
    /// Gradients reach the binding weights (the active rows of each region projection
    /// — including grown units' zero-initialized rows, which no other path ever trains
    /// — plus W_q/W_k/W_v) and the heads. Activation values enter the graph as
    /// materialized data (same as every other episode-consuming path), so encoder
    /// columns are trained by the SAE term in the same NREM batch, not by this step.
    ///
    /// Deliberately a separate method with its own optimizer step rather than a fusion
    /// into `fused_replay_wake_step_batch`: keeps the existing fused path byte-identical
    /// for the A/B control arm (ADR 0041). Gated behind `--binding-replay` at the call
    /// site; the `NremBindingReplay` Core metric stays at zero when off.
    /// Hard-copy `binding`'s currently-trained tensors (region_0..4, w_q, w_k, w_v)
    /// into `wake_binding` (ADR 0048). Plain value copy — the target gets a fresh
    /// `ParamId` and no autodiff graph link back to `binding`; GRU/temporal_proj are
    /// copied too (`copy_temporal_from`) — neither copy trains them, but they are random per
    /// instance, so they must be aligned with `binding` explicitly.
    /// Called by the sleep-cycle driver loop on a `--binding-wake-lag-cycles` cadence,
    /// not every cycle — that's the whole point of this ADR.
    pub fn sync_wake_binding(&mut self) {
        somnus_metrics::timed_core!(somnus_metrics::Metric::BindingWakeSync, {
            let model = self.model_mut();

            // Fine-tier divergence gauge (ADR 0048): L2 distance between binding and
            // wake_binding, summed across the 8 trained tensors, measured *before* this
            // sync overwrites wake_binding — quantifies how much the live copy drifted
            // during the lag window just closed. Textually absent (not just skipped) when
            // `fine-metrics` is off, per this tier's cost discipline.
            #[cfg(feature = "fine-metrics")]
            {
                let mut divergence_l2_sq = 0.0f64;
                macro_rules! accumulate_divergence {
                    ($field:ident) => {{
                        let live: Vec<f32> = model.binding.$field.weight.val().to_f32_vec();
                        let wake: Vec<f32> = model.wake_binding.$field.weight.val().to_f32_vec();
                        for (a, b) in live.iter().zip(wake.iter()) {
                            let d = (*a - *b) as f64;
                            divergence_l2_sq += d * d;
                        }
                    }};
                }
                accumulate_divergence!(region_0);
                accumulate_divergence!(region_1);
                accumulate_divergence!(region_2);
                accumulate_divergence!(region_3);
                accumulate_divergence!(region_4);
                accumulate_divergence!(w_q);
                accumulate_divergence!(w_k);
                accumulate_divergence!(w_v);
                let divergence_l2 = divergence_l2_sq.sqrt() as f32;
                somnus_metrics::counted!(
                    somnus_metrics::Metric::BindingWakeDivergenceL2,
                    (divergence_l2 * 1e6) as u64
                );
            }

            macro_rules! copy_linear {
                ($field:ident) => {
                    model.wake_binding.$field.weight =
                        Param::initialized(ParamId::new(), model.binding.$field.weight.val());
                };
            }
            copy_linear!(region_0);
            copy_linear!(region_1);
            copy_linear!(region_2);
            copy_linear!(region_3);
            copy_linear!(region_4);
            copy_linear!(w_q);
            copy_linear!(w_k);
            copy_linear!(w_v);
            // GRU gates + temporal_proj are not trained but are per-instance random: keep the
            // wake copy identical to `binding` (and to what a snapshot restored into it).
            model.wake_binding.copy_temporal_from(&model.binding);
        });
    }

    pub fn binding_replay_step_batch(
        &mut self,
        batch: &[&[Vec<f32>; 5]],
        lr_scale: f32,
        freeze_heads: bool,
    ) -> PerceptionResult<f32> {
        let (result, _) = somnus_metrics::timed_core!(somnus_metrics::Metric::NremBindingReplay, {
            self.binding_replay_step_batch_inner(batch, lr_scale, freeze_heads)
        });
        result
    }

    fn binding_replay_step_batch_inner(
        &mut self,
        batch: &[&[Vec<f32>; 5]],
        lr_scale: f32,
        freeze_heads: bool,
    ) -> PerceptionResult<f32> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        let device = self.device.clone();
        let batch_topk = self.batch_topk;
        let region_ids = [
            RegionId::AudioSpeech,
            RegionId::AudioEnv,
            RegionId::Visual,
            RegionId::Symbolic,
            RegionId::Language,
        ];
        let mut total_loss: Option<Tensor<1>> = None;
        let mut n_terms = 0usize;

        {
            let model = self.model_mut();

            // Phase 1: per-region batched encoder forwards (same pattern as
            // replay_step_batch) → per-sample SparseActivations.
            let mut acts: Vec<[Option<SparseActivation>; 5]> = batch
                .iter()
                .map(|_| std::array::from_fn(|_| None))
                .collect();
            for region_idx in 0..5usize {
                let valid: Vec<(usize, &Vec<f32>)> = batch
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| {
                        let v = &r[region_idx];
                        if v.is_empty() {
                            None
                        } else {
                            Some((i, v))
                        }
                    })
                    .collect();
                if valid.is_empty() {
                    continue;
                }
                let b = valid.len();
                let n_in = valid[0].1.len();
                let flat: Vec<f32> = valid.iter().flat_map(|(_, v)| v.iter().copied()).collect();
                let input_batch: Tensor<2> =
                    Tensor::from_data(TensorData::new(flat, [b, n_in]), &device);
                let (z_batch, flat_masks) =
                    model
                        .layer_mut(region_idx)
                        .forward_tracked_batch(input_batch, b, batch_topk);
                let n_units = flat_masks.len() / b;
                let z_flat: Vec<f32> = z_batch.to_f32_vec();
                for (j, &(sample_i, _)) in valid.iter().enumerate() {
                    let mask_row = &flat_masks[j * n_units..(j + 1) * n_units];
                    let z_row = &z_flat[j * n_units..(j + 1) * n_units];
                    let mut indices = Vec::new();
                    let mut values = Vec::new();
                    for (u, &m) in mask_row.iter().enumerate() {
                        if m != 0.0 {
                            indices.push(u as u16);
                            values.push(z_row[u]);
                        }
                    }
                    acts[sample_i][region_idx] = Some(SparseActivation {
                        region: region_ids[region_idx],
                        indices,
                        values,
                        n_units,
                    });
                }
            }

            // Phase 2: live episodes through the binding area (graph-preserving), one
            // per sample with at least one non-empty region. Empty regions become
            // zero-token placeholders, same as the wake path's lang_zero sites.
            let mut episodes: Vec<Tensor<2>> = Vec::new();
            let mut episode_sample: Vec<usize> = Vec::new();
            for (sample_i, sample_acts) in acts.iter().enumerate() {
                if sample_acts.iter().all(|a| a.is_none()) {
                    continue;
                }
                let placeholders: [SparseActivation; 5] =
                    std::array::from_fn(|i| SparseActivation {
                        region: region_ids[i],
                        indices: vec![],
                        values: vec![],
                        n_units: model.binding.region_n[i],
                    });
                let refs: [&SparseActivation; 5] =
                    std::array::from_fn(|i| sample_acts[i].as_ref().unwrap_or(&placeholders[i]));
                let ep: Tensor<2> = model.binding.forward_tracked(&refs, &device);
                episodes.push(ep);
                episode_sample.push(sample_i);
            }
            if episodes.is_empty() {
                return Ok(0.0);
            }
            let ep_batch: Tensor<2> = Tensor::cat(episodes, 0); // [B', EPISODE_DIM]

            // Phase 3: output-head loss against the regional targets — same semantics
            // as wake_reconstruct_step_batch, with live episodes instead of stored.
            for head_idx in 0..5usize {
                let valid: Vec<(usize, &Vec<f32>)> = episode_sample
                    .iter()
                    .map(|&sample_i| &batch[sample_i])
                    .enumerate()
                    .filter_map(|(row, r)| {
                        let inp = &r[head_idx];
                        if inp.is_empty() {
                            None
                        } else {
                            Some((row, inp))
                        }
                    })
                    .collect();
                if valid.is_empty() {
                    continue;
                }
                let b = valid.len();
                let n_out = valid[0].1.len();
                let rows: Vec<i64> = valid.iter().map(|&(row, _)| row as i64).collect();
                let flat_targets: Vec<f32> = valid
                    .iter()
                    .flat_map(|(_, inp)| inp.iter().copied())
                    .collect();
                let idx_t: Tensor<1, Int> = Tensor::from_data(TensorData::new(rows, [b]), &device);
                let ep_sub: Tensor<2> = ep_batch.clone().select(0, idx_t);
                let target_batch: Tensor<2> =
                    Tensor::from_data(TensorData::new(flat_targets, [b, n_out]), &device);
                let pred = match head_idx {
                    0 => model.speech_head.forward(ep_sub),
                    1 => model.env_head.forward(ep_sub),
                    2 => model.visual_head.forward(ep_sub),
                    3 => model.symbolic_head.forward(ep_sub),
                    4 => model.language_head.forward(ep_sub),
                    _ => unreachable!(),
                };
                let loss: Tensor<1> = (pred - target_batch).powi_scalar(2).mean();
                total_loss = Some(match total_loss {
                    None => loss,
                    Some(acc) => acc + loss,
                });
                n_terms += 1;
            }
        }

        if let Some(loss) = total_loss {
            // lr_scale multiplies the *learning rate*, not the loss (gaps.md §33 / ADR 0041
            // Validation-plan item 5 correction): the optimizer here is AdamW, which normalizes
            // each parameter update by its running second-moment estimate, so scaling the loss
            // tensor before backward() cancels out almost entirely — candidate (a)'s original
            // "loss-scale" pilot was a no-op by construction, not a real dosage test. Scaling
            // the step size actually changes the update magnitude.
            let loss_val: f32 = loss.clone().into_scalar::<f32>();
            let model = self
                .model
                .take()
                .expect("model already taken (re-entrant optimizer step?)");
            let grads = loss.backward();
            let mut grads_p = GradientsParams::from_grads(grads, &model);
            // freeze_heads (gaps.md §33 candidate (d), 2026-08-16): the five output heads are
            // shared with fused_replay_wake_step_batch, which trains them on *stored* episodes
            // in the same NREM batch while this pass trains them on *live* (still-moving)
            // episodes — a plausible direct cause of the dosage-invariant mean_output_mse
            // regression across all four prior pilots (ADR 0041). Removing the heads' grads
            // here leaves them untouched by this step's optimizer.step() while gradient still
            // flows *through* them into the live episode tensor and on into the binding
            // weights (Burn only skips the update for params with no registered gradient —
            // it doesn't prevent backward() from computing one).
            if freeze_heads {
                let model_ref = &model;
                for head in [
                    &model_ref.speech_head,
                    &model_ref.env_head,
                    &model_ref.visual_head,
                    &model_ref.symbolic_head,
                    &model_ref.language_head,
                ] {
                    grads_p.remove::<2>(head.weight.id);
                    if let Some(b) = &head.bias {
                        grads_p.remove::<1>(b.id);
                    }
                }
            }
            let model = self
                .optimizer
                .step(self.current_lr * lr_scale as f64, model, grads_p);
            self.model = Some(model);
            // Fine tier: the A/B-comparison-specific signal for ADR 0041 (mean loss =
            // (summed scaled value / count) / 1e6).
            somnus_metrics::counted!(
                somnus_metrics::Metric::BindingReplayLoss,
                (loss_val * 1e6).max(0.0) as u64
            );
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
        let device = self.device.clone();
        let ep_dim = EPISODE_DIM;
        let mut total_loss: Option<Tensor<1>> = None;
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

            let ep_batch: Tensor<2> =
                Tensor::from_data(TensorData::new(flat_eps, [b, ep_dim]), &device);
            let target_batch: Tensor<2> =
                Tensor::from_data(TensorData::new(flat_targets, [b, n_out]), &device);

            let pred = {
                let model = self.model_ref();
                match head_idx {
                    0 => model.speech_head.forward(ep_batch),
                    1 => model.env_head.forward(ep_batch),
                    2 => model.visual_head.forward(ep_batch),
                    3 => model.symbolic_head.forward(ep_batch),
                    4 => model.language_head.forward(ep_batch),
                    _ => unreachable!(),
                }
            };

            let loss: Tensor<1> = (pred - target_batch).powi_scalar(2).mean();
            total_loss = Some(match total_loss {
                None => loss,
                Some(acc) => acc + loss,
            });
            n += 1;
        }

        if let Some(loss) = total_loss {
            let loss_val: f32 = loss.clone().into_scalar::<f32>();
            let model = self
                .model
                .take()
                .expect("model already taken (re-entrant optimizer step?)");
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
        let device = self.device.clone();

        // ── Phase 1: SAE replay losses (needs &mut model for duty-cycle updates) ──
        let sae_l1_coeff = self.sae_l1_coeff;
        let batch_topk = self.batch_topk;
        let ((sae_loss, sae_n_terms), _) =
            somnus_metrics::timed_core!(somnus_metrics::Metric::FusedReplaySaePhase, {
                let mut sae_loss: Option<Tensor<1>> = None;
                let mut sae_n_terms = 0usize;
                {
                    let model = self.model_mut();
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

                        let input_batch: Tensor<2> =
                            Tensor::from_data(TensorData::new(flat.clone(), [b, n_in]), &device);

                        let (z_batch, flat_masks) = model
                            .layer_mut(region_idx)
                            .forward_tracked_batch(input_batch.clone(), b, batch_topk);

                        let n_units = flat_masks.len() / b;
                        let mask_inner: Tensor<2> =
                            Tensor::from_data(TensorData::new(flat_masks, [b, n_units]), &device);
                        let mask: Tensor<2> = Tensor::from_inner(mask_inner);
                        let sparse_z = z_batch * mask;
                        let l1_term: Option<Tensor<1>> = if sae_l1_coeff > 0.0 {
                            Some(sparse_z.clone().abs().mean())
                        } else {
                            None
                        };
                        let target: Tensor<2> =
                            Tensor::from_data(TensorData::new(flat, [b, n_in]), &device);

                        let mse_loss = model.decoder(region_idx).mse_loss_batch(target, sparse_z);
                        let region_loss = match l1_term {
                            Some(l1) => mse_loss + l1.mul_scalar(sae_l1_coeff),
                            None => mse_loss,
                        };
                        sae_loss = Some(match sae_loss {
                            None => region_loss,
                            Some(a) => a + region_loss,
                        });
                        sae_n_terms += 1;
                    }
                } // &mut borrow of model ends here
                (sae_loss, sae_n_terms)
            });

        // ── Phase 2: output-head losses (needs only &model, not &mut) ───────────
        let ep_dim = EPISODE_DIM;
        let ((wake_loss, wake_n_terms), _) =
            somnus_metrics::timed_core!(somnus_metrics::Metric::FusedReplayHeadPhase, {
                let mut wake_loss: Option<Tensor<1>> = None;
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

                        let ep_batch: Tensor<2> =
                            Tensor::from_data(TensorData::new(flat_eps, [b, ep_dim]), &device);
                        let target_batch: Tensor<2> =
                            Tensor::from_data(TensorData::new(flat_tgt, [b, n_out]), &device);

                        let pred = {
                            let model = self.model_ref();
                            match head_idx {
                                0 => model.speech_head.forward(ep_batch),
                                1 => model.env_head.forward(ep_batch),
                                2 => model.visual_head.forward(ep_batch),
                                3 => model.symbolic_head.forward(ep_batch),
                                4 => model.language_head.forward(ep_batch),
                                _ => unreachable!(),
                            }
                        };
                        let h_loss: Tensor<1> = (pred - target_batch).powi_scalar(2).mean();
                        wake_loss = Some(match wake_loss {
                            None => h_loss,
                            Some(a) => a + h_loss,
                        });
                        wake_n_terms += 1;
                    }
                }
                (wake_loss, wake_n_terms)
            });

        // ── Single combined backward + optimizer step ────────────────────────────
        let sae_val = sae_loss
            .as_ref()
            .map(|l| l.clone().into_scalar::<f32>() / sae_n_terms.max(1) as f32)
            .unwrap_or(0.0);
        let wake_val = wake_loss
            .as_ref()
            .map(|l| l.clone().into_scalar::<f32>() / wake_n_terms.max(1) as f32)
            .unwrap_or(0.0);

        // DER++ Phase 3: auto-λ. Weight wake_loss (MSE against stale stored outputs) by the
        // inverse of its own running variance instead of a hand-tuned constant, so a noisy
        // wake_loss term contributes less to the combined gradient (homoscedastic uncertainty
        // weighting; ai-docs/gaps.md item 7).
        let der_lambda: f32 = if self.der_auto_lambda && wake_n_terms > 0 {
            const EWMA_ALPHA: f32 = 0.05;
            let (mean, msq) = self
                .der_lambda_stats
                .unwrap_or((wake_val, wake_val * wake_val));
            let mean = EWMA_ALPHA * wake_val + (1.0 - EWMA_ALPHA) * mean;
            let msq = EWMA_ALPHA * wake_val * wake_val + (1.0 - EWMA_ALPHA) * msq;
            self.der_lambda_stats = Some((mean, msq));
            (1.0 / (msq - mean * mean).max(1e-6)).clamp(0.1, 10.0)
        } else {
            1.0
        };

        let combined = match (sae_loss, wake_loss) {
            (Some(sl), Some(wl)) => Some(sl + wl.mul_scalar(der_lambda)),
            (Some(sl), None) => Some(sl),
            (None, Some(wl)) => Some(wl.mul_scalar(der_lambda)),
            (None, None) => None,
        };
        somnus_metrics::timed_core!(somnus_metrics::Metric::FusedReplayBackward, {
            if let Some(loss) = combined {
                let model = self
                    .model
                    .take()
                    .expect("model already taken (re-entrant optimizer step?)");
                let grads = loss.backward();
                let grads_p = GradientsParams::from_grads(grads, &model);
                let model = self.optimizer.step(self.current_lr, model, grads_p);
                self.model = Some(model);
            }
        });

        Ok((sae_val, wake_val))
    }

    /// Run one wake-time reconstruction step on a just-produced episode.
    pub fn wake_reconstruct_step(
        &mut self,
        episode: &[f32],
        regional: &[Vec<f32>; 5],
    ) -> PerceptionResult<f32> {
        let (result, _) = somnus_metrics::timed_core!(somnus_metrics::Metric::WakeReconstructStep, {
            let device = self.device.clone();
            let ep_t: Tensor<2> = Tensor::from_data(
                TensorData::new(episode.to_vec(), [1, episode.len()]),
                &device,
            );
            let mut total_loss: Option<Tensor<1>> = None;
            let mut n = 0usize;

            for (head_idx, input) in regional.iter().enumerate() {
                if input.is_empty() {
                    continue;
                }
                let target: Tensor<2> =
                    Tensor::from_data(TensorData::new(input.clone(), [1, input.len()]), &device);

                let pred = {
                    let model = self.model_ref();
                    match head_idx {
                        0 => model.speech_head.forward(ep_t.clone()),
                        1 => model.env_head.forward(ep_t.clone()),
                        2 => model.visual_head.forward(ep_t.clone()),
                        3 => model.symbolic_head.forward(ep_t.clone()),
                        4 => model.language_head.forward(ep_t.clone()),
                        _ => unreachable!(),
                    }
                };

                let loss: Tensor<1> = (pred - target).powi_scalar(2).mean();
                total_loss = Some(match total_loss {
                    None => loss,
                    Some(acc) => acc + loss,
                });
                n += 1;
            }

            if let Some(loss) = total_loss {
                let loss_val: f32 = loss.clone().into_scalar::<f32>();
                let model = self
                    .model
                    .take()
                    .expect("model already taken (re-entrant optimizer step?)");
                let grads = loss.backward();
                let grads_p = GradientsParams::from_grads(grads, &model);
                let model = self.optimizer.step(self.current_lr, model, grads_p);
                self.ema_heads.update(&model);
                self.model = Some(model);
                Ok(loss_val / n.max(1) as f32)
            } else {
                Ok(0.0)
            }
        });
        result
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
        let m = self.model_ref();
        [
            &m.speech_head,
            &m.env_head,
            &m.visual_head,
            &m.symbolic_head,
            &m.language_head,
        ]
        .map(|h| h.weight.val().inner().to_f32_vec())
    }

    /// Write worker-trained head weights back into the model (sleep-entry merge).
    /// ParamIds are preserved so the main optimizer's AdamW moments stay attached.
    pub fn set_head_weights_flat(&mut self, weights: [Vec<f32>; 5]) {
        let device = self.device.clone();
        let m = self.model_mut();
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
            let t: Tensor<2> =
                Tensor::from_data(TensorData::new(flat, [EPISODE_DIM, d_out]), &device)
                    .require_grad();
            h.weight = Param::initialized(id, t);
        }
    }

    /// Per-row L2 norms of one region's binding-projection weight, row order
    /// matching unit indices (ADR 0041 inspection surface). Grown rows are
    /// zero-initialized by `grow_region`/`grow_region_seeded` and only
    /// `binding_replay_step_batch` ever trains them — this is how the A/B pilot
    /// verifies they actually learn.
    pub fn binding_proj_row_norms(&self, region: RegionId) -> Vec<f32> {
        let m = self.model_ref();
        let proj = match region {
            RegionId::AudioSpeech => &m.binding.region_0,
            RegionId::AudioEnv => &m.binding.region_1,
            RegionId::Visual => &m.binding.region_2,
            RegionId::Symbolic => &m.binding.region_3,
            RegionId::Language => &m.binding.region_4,
        };
        let flat: Vec<f32> = proj.weight.val().inner().to_f32_vec();
        let dims = proj.weight.val().inner().dims();
        let (n, d) = (dims[0], dims[1]);
        (0..n)
            .map(|r| {
                let row = &flat[r * d..(r + 1) * d];
                row.iter().map(|v| v * v).sum::<f32>().sqrt()
            })
            .collect()
    }

    /// Same as `binding_proj_row_norms` but for `wake_binding` (ADR 0048 inspection
    /// surface): lets a test or pilot confirm the wake copy stays frozen between
    /// `sync_wake_binding` calls while `binding` (via `binding_proj_row_norms`) keeps
    /// moving under `binding_replay_step_batch`.
    pub fn wake_binding_proj_row_norms(&self, region: RegionId) -> Vec<f32> {
        let m = self.model_ref();
        let proj = match region {
            RegionId::AudioSpeech => &m.wake_binding.region_0,
            RegionId::AudioEnv => &m.wake_binding.region_1,
            RegionId::Visual => &m.wake_binding.region_2,
            RegionId::Symbolic => &m.wake_binding.region_3,
            RegionId::Language => &m.wake_binding.region_4,
        };
        let flat: Vec<f32> = proj.weight.val().inner().to_f32_vec();
        let dims = proj.weight.val().inner().dims();
        let (n, d) = (dims[0], dims[1]);
        (0..n)
            .map(|r| {
                let row = &flat[r * d..(r + 1) * d];
                row.iter().map(|v| v * v).sum::<f32>().sqrt()
            })
            .collect()
    }

    /// Apply worker-streamed flat EMA weights to the inference path.
    pub fn apply_ema_flat(&mut self, flat: [Vec<f32>; 5]) {
        self.ema_heads.set_flat(flat);
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// Per-input SAE reconstruction MSE for region `idx`, batched in chunks of `SCORE_CHUNK`.
    /// Inference only: no weight update, no duty-cycle update.
    fn score_region(&self, idx: usize, inputs: &[&[f32]]) -> Vec<f32> {
        const SCORE_CHUNK: usize = 256;
        let model = self.model_ref();
        let mut out = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(SCORE_CHUNK) {
            let acts = model.layer(idx).forward_no_dc_batch(chunk, &self.device);
            out.extend(
                model
                    .decoder(idx)
                    .reconstruction_errors_batch(chunk, &acts, &self.device),
            );
        }
        out
    }

    /// Measure SAE reconstruction MSE per region (inference, no weight update).
    pub fn measure_mse(&mut self, regional: &[Vec<f32>; 5]) -> PerceptionResult<[f32; 5]> {
        Ok(self.measure_mse_many(&[regional])[0])
    }

    /// `measure_mse` for many traces at once (one batched matmul per region instead of one
    /// per trace). Empty region slots stay 0.0 in their trace's result.
    pub fn measure_mse_many(&self, traces: &[&[Vec<f32>; 5]]) -> Vec<[f32; 5]> {
        let (result, _) = somnus_metrics::timed_core!(somnus_metrics::Metric::MeasureMse, {
            let mut result = vec![[0.0f32; 5]; traces.len()];
            for idx in 0..5 {
                let rows: Vec<usize> = (0..traces.len())
                    .filter(|&t| !traces[t][idx].is_empty())
                    .collect();
                let inputs: Vec<&[f32]> = rows.iter().map(|&t| traces[t][idx].as_slice()).collect();
                for (&t, err) in rows.iter().zip(self.score_region(idx, &inputs)) {
                    result[t][idx] = err;
                }
            }
            result
        });
        result
    }

    /// Blend two sets of regional encoder inputs and produce a synthetic episode.
    /// No DC update — inference only.
    pub fn synthesize_episode(
        &mut self,
        regional_a: &[Vec<f32>; 5],
        regional_b: &[Vec<f32>; 5],
        alpha: f32,
    ) -> PerceptionResult<Vec<f32>> {
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::SynthesizeEpisode,
            self.synthesize_episode_inner(regional_a, regional_b, alpha)
        );
        result
    }

    fn synthesize_episode_inner(
        &mut self,
        regional_a: &[Vec<f32>; 5],
        regional_b: &[Vec<f32>; 5],
        alpha: f32,
    ) -> PerceptionResult<Vec<f32>> {
        let device = self.device.clone();
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
            let model = self.model_ref();
            acts[i] = Some(model.layer(i).forward_no_dc(&blended, &device));
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
        let (result, _) =
            somnus_metrics::timed_core!(somnus_metrics::Metric::MeasureOutputMseInference, {
                let device = self.device.clone();
                let ep_t: Tensor<2> =
                    Tensor::from_data(TensorData::new(episode.to_vec(), [1, EPISODE_DIM]), &device);
                let model = self.model_ref();
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
                    let target: Tensor<2> = Tensor::from_data(
                        TensorData::new(input.clone(), [1, input.len()]),
                        &device,
                    );
                    let mse: f32 = (pred - target).powi_scalar(2).mean().into_scalar::<f32>();
                    total += mse;
                    n += 1;
                }
                if n > 0 {
                    total / n as f32
                } else {
                    0.0
                }
            });
        Ok(result)
    }

    /// Infer predicted outputs (regional embeddings) from an episode vector (DER++ stale detection).
    /// Returns [Vec<f32>; 5] with one prediction per region. Empty regions get empty vectors.
    /// Used to detect staleness: compare predicted vs stored outputs via cosine similarity.
    pub fn infer_episode_outputs(&self, episode: &[f32]) -> PerceptionResult<[Vec<f32>; 5]> {
        let (outputs, _) = somnus_metrics::timed_core!(somnus_metrics::Metric::InferEpisodeOutputs, {
            let device = self.device.clone();
            let ep_t: Tensor<2> =
                Tensor::from_data(TensorData::new(episode.to_vec(), [1, EPISODE_DIM]), &device);
            let model = self.model_ref();

            let mut outputs: [Vec<f32>; 5] = Default::default();
            let heads: [&Linear; 5] = [
                &model.speech_head,
                &model.env_head,
                &model.visual_head,
                &model.symbolic_head,
                &model.language_head,
            ];

            for (i, head) in heads.iter().enumerate() {
                let pred: Tensor<2> = head.forward(ep_t.clone());
                let pred_vec: Vec<f32> = pred
                    .into_data()
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .map_err(|e| PerceptionError::Burn(format!("tensor extraction: {}", e)))?;
                outputs[i] = pred_vec;
            }
            outputs
        });
        Ok(outputs)
    }

    // ── Homeostasis ──────────────────────────────────────────────────────────

    /// Apply global weight decay to all trainable Linear weights.
    pub fn homeostatic_decay(&mut self, rate: f32) {
        somnus_metrics::timed_core!(somnus_metrics::Metric::HomeostaticDecay, {
            let scale = 1.0 - rate;
            let model = self.model_mut();
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
        });
    }

    /// Grow a region by appending `n_new` units with small random encoder weights.
    pub fn grow_region(&mut self, region: RegionId, n_new: usize) {
        somnus_metrics::timed_core!(somnus_metrics::Metric::GrowRegion, {
            self.grow_region_inner(region, n_new)
        });
    }

    fn grow_region_inner(&mut self, region: RegionId, n_new: usize) {
        let device = self.device.clone();
        let n_in = REGION_DIMS[region.as_index()];
        let model = self.model_mut();

        let new_n = {
            let layer = Self::layer_mut(model, region);
            let old_n = layer.n_units;
            let new_n = old_n + n_new;

            // Encoder weight shape: [n_in, n_units] — grow by appending new columns (dim 1).
            let old_w: Tensor<2> = layer.linear.weight.val().inner();
            let new_cols: Tensor<2> =
                Tensor::random([n_in, n_new], Distribution::Normal(0.0, 0.01), &device);
            let new_w: Tensor<2> = Tensor::from_inner(Tensor::cat(vec![old_w, new_cols], 1));
            layer.linear.weight = Param::initialized(ParamId::new(), new_w);
            layer.n_units = new_n;
            // Rescale k so sparsity ratio (k/n) stays roughly constant as the region
            // grows, instead of decaying toward 0 (see gaps.md §17).
            layer.k_active = region::scaled_k(new_n);
            layer.duty_cycle.extend(vec![0.0f32; n_new]);
            new_n
        };

        {
            // Decoder weight shape: [n_units, n_in] — grow by appending new rows (dim 0).
            let decoder = Self::decoder_mut(model, region);
            let old_w: Tensor<2> = decoder.decoder.weight.val().inner();
            let new_rows: Tensor<2> = Tensor::zeros([n_new, n_in], &device);
            let new_w: Tensor<2> = Tensor::from_inner(Tensor::cat(vec![old_w, new_rows], 0));
            decoder.decoder.weight = Param::initialized(ParamId::new(), new_w);
            decoder.n_units = new_n;
        }

        {
            // Binding projection weight: [n_units, D_PROJ] — grow by appending zero rows.
            // New units have zero projection weights; gradients will shape them over time.
            let binding_proj = match region {
                RegionId::AudioSpeech => &mut model.binding.region_0,
                RegionId::AudioEnv => &mut model.binding.region_1,
                RegionId::Visual => &mut model.binding.region_2,
                RegionId::Symbolic => &mut model.binding.region_3,
                RegionId::Language => &mut model.binding.region_4,
            };
            let old_w: Tensor<2> = binding_proj.weight.val().inner();
            let d_proj = old_w.dims()[1];
            let new_rows: Tensor<2> = Tensor::zeros([n_new, d_proj], &device);
            let new_w: Tensor<2> = Tensor::from_inner(Tensor::cat(vec![old_w, new_rows], 0));
            binding_proj.weight = Param::initialized(ParamId::new(), new_w);
            model.binding.region_n[region.as_index()] = new_n;
        }

        {
            // ADR 0048: wake_binding's shape must never lag behind binding's — only
            // trained *values* lag (via sync_wake_binding), never shape. Mirror the
            // identical zero-row append applied to binding above.
            let wake_binding_proj = match region {
                RegionId::AudioSpeech => &mut model.wake_binding.region_0,
                RegionId::AudioEnv => &mut model.wake_binding.region_1,
                RegionId::Visual => &mut model.wake_binding.region_2,
                RegionId::Symbolic => &mut model.wake_binding.region_3,
                RegionId::Language => &mut model.wake_binding.region_4,
            };
            let old_w: Tensor<2> = wake_binding_proj.weight.val().inner();
            let d_proj = old_w.dims()[1];
            let new_rows: Tensor<2> = Tensor::zeros([n_new, d_proj], &device);
            let new_w: Tensor<2> = Tensor::from_inner(Tensor::cat(vec![old_w, new_rows], 0));
            wake_binding_proj.weight = Param::initialized(ParamId::new(), new_w);
            model.wake_binding.region_n[region.as_index()] = new_n;
        }

        // No optimizer reset: burn's OptimizerAdaptor keys momentum state by ParamId
        // (see burn_optim::simple::adaptor). The three tensors touched above got fresh
        // ParamIds, so they naturally start with fresh (zero) Adam state on the next
        // step; every other param's id is untouched and keeps its accumulated momentum
        // instead of being wiped for no reason (gaps.md §18).
        somnus_metrics::counted_core!(somnus_metrics::Metric::RegionNUnits, new_n as u64);
        somnus_metrics::counted_core!(
            somnus_metrics::Metric::RegionKActive,
            Self::layer_mut(model, region).k_active as u64
        );
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
        // Empty-seed calls degrade to a plain grow_region (own GrowRegion metric already
        // covers it) — not timed as GrowRegionSeeded, since no seeding actually happens.
        if sorted_inputs.is_empty() {
            self.grow_region(region, n_new);
            return;
        }
        somnus_metrics::timed_core!(somnus_metrics::Metric::GrowRegionSeeded, {
            self.grow_region_seeded_inner(region, n_new, sorted_inputs)
        });
    }

    fn grow_region_seeded_inner(
        &mut self,
        region: RegionId,
        n_new: usize,
        sorted_inputs: &[Vec<f32>],
    ) {
        let device = self.device.clone();
        let n_in = REGION_DIMS[region.as_index()];
        let init_scale = self.grown_init_scale;
        let model = self.model_mut();

        let new_n = {
            let layer = Self::layer_mut(model, region);
            let old_n = layer.n_units;
            let new_n = old_n + n_new;

            // Build encoder columns: normalised seed × `init_scale` (default 0.01, matches grow_region).
            let mut cols: Vec<f32> = Vec::with_capacity(n_in * n_new);
            let n_seeds = sorted_inputs.len();
            for k in 0..n_new {
                let seed = &sorted_inputs[k % n_seeds];
                let norm: f32 = seed.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                for &x in seed {
                    cols.push(x / norm * init_scale);
                }
            }
            let cols_t: Tensor<2> = Tensor::from_inner(Tensor::<2>::from_data(
                burn::tensor::TensorData::new(cols, [n_in, n_new]),
                &device,
            ));

            let old_w: Tensor<2> = layer.linear.weight.val().inner();
            let new_w: Tensor<2> = Tensor::from_inner(Tensor::cat(vec![old_w, cols_t.inner()], 1));
            layer.linear.weight = Param::initialized(ParamId::new(), new_w);
            layer.n_units = new_n;
            layer.k_active = region::scaled_k(new_n);
            layer.duty_cycle.extend(vec![0.0f32; n_new]);
            new_n
        };

        {
            let decoder = Self::decoder_mut(model, region);
            let old_w: Tensor<2> = decoder.decoder.weight.val().inner();
            let new_rows: Tensor<2> = Tensor::zeros([n_new, n_in], &device);
            let new_w: Tensor<2> = Tensor::from_inner(Tensor::cat(vec![old_w, new_rows], 0));
            decoder.decoder.weight = Param::initialized(ParamId::new(), new_w);
            decoder.n_units = new_n;
        }

        {
            let binding_proj = match region {
                RegionId::AudioSpeech => &mut model.binding.region_0,
                RegionId::AudioEnv => &mut model.binding.region_1,
                RegionId::Visual => &mut model.binding.region_2,
                RegionId::Symbolic => &mut model.binding.region_3,
                RegionId::Language => &mut model.binding.region_4,
            };
            let old_w: Tensor<2> = binding_proj.weight.val().inner();
            let d_proj = old_w.dims()[1];
            let new_rows: Tensor<2> = Tensor::zeros([n_new, d_proj], &device);
            let new_w: Tensor<2> = Tensor::from_inner(Tensor::cat(vec![old_w, new_rows], 0));
            binding_proj.weight = Param::initialized(ParamId::new(), new_w);
            model.binding.region_n[region.as_index()] = new_n;
        }

        {
            // ADR 0048: mirror the shape change onto wake_binding — same reasoning
            // as grow_region_inner (shape lockstep, values lag separately).
            let wake_binding_proj = match region {
                RegionId::AudioSpeech => &mut model.wake_binding.region_0,
                RegionId::AudioEnv => &mut model.wake_binding.region_1,
                RegionId::Visual => &mut model.wake_binding.region_2,
                RegionId::Symbolic => &mut model.wake_binding.region_3,
                RegionId::Language => &mut model.wake_binding.region_4,
            };
            let old_w: Tensor<2> = wake_binding_proj.weight.val().inner();
            let d_proj = old_w.dims()[1];
            let new_rows: Tensor<2> = Tensor::zeros([n_new, d_proj], &device);
            let new_w: Tensor<2> = Tensor::from_inner(Tensor::cat(vec![old_w, new_rows], 0));
            wake_binding_proj.weight = Param::initialized(ParamId::new(), new_w);
            model.wake_binding.region_n[region.as_index()] = new_n;
        }

        // See grow_region: no optimizer reset needed, ParamId-keyed state handles it (gaps.md §18).
        somnus_metrics::counted_core!(somnus_metrics::Metric::RegionNUnits, new_n as u64);
        somnus_metrics::counted_core!(
            somnus_metrics::Metric::RegionKActive,
            Self::layer_mut(model, region).k_active as u64
        );
        tracing::info!(region = ?region, n_new, "region grown (seeded)");
    }

    /// For each region, return inputs from `traces` sorted by reconstruction residual
    /// magnitude (highest first). Used by `grow_region_seeded` to orient new units
    /// toward inputs the current network reconstructs poorly (cascade-correlation, P2b).
    ///
    /// Traces with empty regional input are skipped. Returns at most `cap` inputs per
    /// region to bound memory use.
    #[allow(clippy::type_complexity)]
    pub fn collect_high_residual_inputs<T, U>(
        &mut self,
        traces: &[(Vec<f32>, [Vec<f32>; 5], T, U)],
        cap: usize,
    ) -> [Vec<Vec<f32>>; 5] {
        let (result, _) =
            somnus_metrics::timed_core!(somnus_metrics::Metric::CollectHighResidualInputs, {
                let mut result: [Vec<Vec<f32>>; 5] = Default::default();
                for (idx, out) in result.iter_mut().enumerate() {
                    let inputs: Vec<&[f32]> = traces
                        .iter()
                        .map(|(_, regional, _, _)| regional[idx].as_slice())
                        .filter(|v| !v.is_empty())
                        .collect();
                    let mse = self.score_region(idx, &inputs);
                    // Highest residual first; only the winners are cloned.
                    let mut order: Vec<usize> = (0..inputs.len()).collect();
                    let by_mse_desc = |a: &usize, b: &usize| {
                        mse[*b]
                            .partial_cmp(&mse[*a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    };
                    if cap < order.len() {
                        order.select_nth_unstable_by(cap, by_mse_desc);
                        order.truncate(cap);
                    }
                    order.sort_by(by_mse_desc);
                    *out = order.into_iter().map(|i| inputs[i].to_vec()).collect();
                }
                result
            });
        result
    }

    /// Prune low-magnitude grown units from a region (Phase 12c).
    /// Only removes units with index >= N_UNITS. Returns number pruned.
    pub fn prune_region(&mut self, region: RegionId, prune_fraction: f32) -> usize {
        let (n_pruned, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::PruneRegion,
            self.prune_region_inner(region, prune_fraction)
        );
        n_pruned
    }

    fn prune_region_inner(&mut self, region: RegionId, prune_fraction: f32) -> usize {
        let device = self.device.clone();
        let n_in = REGION_DIMS[region.as_index()];
        let model = self.model_mut();

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
            .to_f32_vec();

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
            .to_f32_vec();
        // Keep rows `keep` from [current_n, n_in] → [new_n, n_in].
        let new_dec_flat: Vec<f32> = keep
            .iter()
            .flat_map(|&i| dec_flat[i * n_in..(i + 1) * n_in].iter().copied())
            .collect();

        {
            let layer = Self::layer_mut(model, region);
            let new_enc: Tensor<2> =
                Tensor::from_data(TensorData::new(new_enc_flat, [n_in, new_n]), &device);
            layer.linear.weight = Param::initialized(ParamId::new(), new_enc);
            let old_dc = layer.duty_cycle.clone();
            layer.duty_cycle = keep.iter().map(|&i| old_dc[i]).collect();
            layer.n_units = new_n;
            layer.k_active = region::scaled_k(new_n);
        }

        {
            let decoder = Self::decoder_mut(model, region);
            let new_dec: Tensor<2> =
                Tensor::from_data(TensorData::new(new_dec_flat, [new_n, n_in]), &device);
            decoder.decoder.weight = Param::initialized(ParamId::new(), new_dec);
            decoder.n_units = new_n;
        }

        {
            // Binding projection: [current_n, D_PROJ] — keep rows matching `keep`.
            let binding_proj = match region {
                RegionId::AudioSpeech => &mut model.binding.region_0,
                RegionId::AudioEnv => &mut model.binding.region_1,
                RegionId::Visual => &mut model.binding.region_2,
                RegionId::Symbolic => &mut model.binding.region_3,
                RegionId::Language => &mut model.binding.region_4,
            };
            let bind_flat: Vec<f32> = binding_proj.weight.val().to_f32_vec();
            let d_proj = bind_flat.len() / current_n;
            let new_bind_flat: Vec<f32> = keep
                .iter()
                .flat_map(|&i| bind_flat[i * d_proj..(i + 1) * d_proj].iter().copied())
                .collect();
            let new_bind: Tensor<2> =
                Tensor::from_data(TensorData::new(new_bind_flat, [new_n, d_proj]), &device);
            binding_proj.weight = Param::initialized(ParamId::new(), new_bind);
            model.binding.region_n[region.as_index()] = new_n;
        }

        {
            // ADR 0048: mirror the identical row-removal onto wake_binding using the
            // same `keep` index set — shape must never lag, only trained values do.
            let wake_binding_proj = match region {
                RegionId::AudioSpeech => &mut model.wake_binding.region_0,
                RegionId::AudioEnv => &mut model.wake_binding.region_1,
                RegionId::Visual => &mut model.wake_binding.region_2,
                RegionId::Symbolic => &mut model.wake_binding.region_3,
                RegionId::Language => &mut model.wake_binding.region_4,
            };
            let bind_flat: Vec<f32> = wake_binding_proj.weight.val().to_f32_vec();
            let d_proj = bind_flat.len() / current_n;
            let new_bind_flat: Vec<f32> = keep
                .iter()
                .flat_map(|&i| bind_flat[i * d_proj..(i + 1) * d_proj].iter().copied())
                .collect();
            let new_bind: Tensor<2> =
                Tensor::from_data(TensorData::new(new_bind_flat, [new_n, d_proj]), &device);
            wake_binding_proj.weight = Param::initialized(ParamId::new(), new_bind);
            model.wake_binding.region_n[region.as_index()] = new_n;
        }

        // See grow_region: no optimizer reset needed, ParamId-keyed state handles it (gaps.md §18).
        somnus_metrics::counted_core!(somnus_metrics::Metric::RegionNUnits, new_n as u64);
        somnus_metrics::counted_core!(
            somnus_metrics::Metric::RegionKActive,
            Self::layer_mut(model, region).k_active as u64
        );
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
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::EnableOrtInference,
            self.enable_ort_inference_inner(bind_path, use_migraphx)
        );
        result
    }

    fn enable_ort_inference_inner(
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
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::RebuildOrtSession,
            self.rebuild_ort_session_inner(script_path, onnx_dir, use_migraphx)
        );
        result
    }

    fn rebuild_ort_session_inner(
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
        self.prior_betas = [beta; 5];
    }

    pub fn set_prior_beta_region(&mut self, idx: usize, beta: f32) {
        if idx < 5 {
            self.prior_betas[idx] = beta;
        }
    }

    pub fn prior_betas(&self) -> [f32; 5] {
        self.prior_betas
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

    /// Encoder-column L2 norm for units added by `grow_region_seeded` (default 0.01).
    pub fn set_grown_init_scale(&mut self, scale: f32) {
        self.grown_init_scale = scale;
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
        somnus_metrics::timed_core!(somnus_metrics::Metric::RefreshAutodiffState, {
            use burn::module::ModuleMapper;
            struct GraphDetacher;
            impl ModuleMapper for GraphDetacher {
                fn map_float<const D: usize>(
                    &mut self,
                    param: Param<Tensor<D>>,
                ) -> Param<Tensor<D>> {
                    let (_, tensor, mapper) = param.consume();
                    let fresh = tensor.inner().autodiff().require_grad();
                    Param::from_mapped_value(ParamId::new(), fresh, mapper)
                }
            }
            let model = self
                .model
                .take()
                .expect("model already taken (re-entrant optimizer step?)");
            self.model = Some(model.map(&mut GraphDetacher));
            self.optimizer = AdamWConfig::new().init();
        });
    }

    pub fn region_sizes(&self) -> [usize; 5] {
        let m = self.model_ref();
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

    pub fn set_sae_l1_coeff(&mut self, coeff: f32) {
        self.sae_l1_coeff = coeff;
    }

    pub fn set_batch_topk(&mut self, enabled: bool) {
        self.batch_topk = enabled;
    }

    pub fn set_der_auto_lambda(&mut self, enabled: bool) {
        self.der_auto_lambda = enabled;
    }

    // ── Save / Load ──────────────────────────────────────────────────────────

    /// Save weights as safetensors (`<path>.safetensors`; any extension on `path` is replaced).
    /// Also writes a `<path>.sizes` sidecar with grown region sizes.
    pub fn save_weights(&mut self, path: &std::path::Path) -> PerceptionResult<()> {
        use burn_store::{ModuleSnapshot, SafetensorsStore};
        let sizes = self.region_sizes();
        let sizes_str = sizes
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let io_err = |e: std::io::Error| PerceptionError::Burn(e.to_string());
        somnus_memory::fsutil::write_atomic(&path.with_extension("sizes"), sizes_str)
            .map_err(io_err)?;

        // Write beside the target and rename: a kill mid-write must not truncate the only copy.
        let dst = path.with_extension("safetensors");
        let tmp = somnus_memory::fsutil::tmp_path(&dst);
        let mut store = SafetensorsStore::from_file(&tmp);
        self.model_ref()
            .save_into(&mut store)
            .map_err(|e| PerceptionError::Burn(format!("{e:?}")))?;
        somnus_memory::fsutil::commit_tmp(&tmp, &dst).map_err(io_err)
    }

    /// Load weights written by `save_weights` (`<path>.safetensors`). Regions must already be
    /// grown to the saved sizes (see `load_weights_with_sizes`). Tensors missing from the file
    /// (e.g. `wake_binding.*` in snapshots that predate ADR 0048) keep their current values;
    /// `wake_binding` is then synced from the loaded `binding`.
    ///
    /// On failure the model is replaced by a fresh one grown to the same sizes, so callers stay
    /// functional after a failed restore.
    pub fn load_weights(&mut self, path: &std::path::Path) -> PerceptionResult<()> {
        use burn_store::{ModuleSnapshot, SafetensorsStore};
        let saved_sizes = self.region_sizes();
        let mut store =
            SafetensorsStore::from_file(path.with_extension("safetensors")).allow_partial(true);
        let applied = self
            .model_mut()
            .load_from(&mut store)
            .map_err(|e| PerceptionError::Burn(format!("{e:?}")))
            .and_then(|r| {
                if r.errors.is_empty() {
                    // Snapshots from before ADR 0048 (or converted from them) carry no
                    // `wake_binding.*`: those must be derived from `binding`. A snapshot that
                    // does carry them must keep its (possibly lagged) wake copy untouched.
                    Ok(r.missing
                        .iter()
                        .any(|(p, _)| p.starts_with("wake_binding.")))
                } else {
                    Err(PerceptionError::Burn(format!("{:?}", r.errors)))
                }
            });
        match applied {
            Ok(wake_missing) => {
                // `region_n` is #[module(skip)], so it is not serialised; rebuild it from the
                // loaded projection shapes (binding and wake_binding independently, ADR 0048).
                let m = self.model_mut();
                m.binding.region_n = [
                    m.binding.region_0.weight.val().inner().dims()[0],
                    m.binding.region_1.weight.val().inner().dims()[0],
                    m.binding.region_2.weight.val().inner().dims()[0],
                    m.binding.region_3.weight.val().inner().dims()[0],
                    m.binding.region_4.weight.val().inner().dims()[0],
                ];
                m.wake_binding.region_n = [
                    m.wake_binding.region_0.weight.val().inner().dims()[0],
                    m.wake_binding.region_1.weight.val().inner().dims()[0],
                    m.wake_binding.region_2.weight.val().inner().dims()[0],
                    m.wake_binding.region_3.weight.val().inner().dims()[0],
                    m.wake_binding.region_4.weight.val().inner().dims()[0],
                ];
                // Loaded tensors carry no autodiff association: re-wrap every parameter as a
                // fresh tracked leaf (and reset the optimizer) before any training step.
                self.refresh_autodiff_state();
                if wake_missing {
                    self.sync_wake_binding();
                }
                self.ema_heads = EmaOutputHeads::from_model(self.model_ref());
                Ok(())
            }
            Err(e) => {
                let device = self.device.clone();
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
                Err(e)
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
    ///   W_region_{i}           – BindingArea region projection, shape [region_n[i], D_PROJ]
    ///   W_q / W_k / W_v        – Attention QKV, shape [D_PROJ, D_PROJ]
    ///   gru_{gate}_input_W     – GRU input weight, shape [GRU_INPUT, GRU_HIDDEN]  (gate=update/reset/new)
    ///   gru_{gate}_hidden_W    – GRU hidden weight, shape [GRU_HIDDEN, GRU_HIDDEN]
    ///   W_temporal_proj        – GRU hidden → EPISODE_DIM, shape [GRU_HIDDEN, EPISODE_DIM]
    pub fn export_weights_for_onnx(&self, dir: &std::path::Path) -> PerceptionResult<()> {
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::ExportWeightsForOnnx,
            self.export_weights_for_onnx_inner(dir)
        );
        result
    }

    fn export_weights_for_onnx_inner(&self, dir: &std::path::Path) -> PerceptionResult<()> {
        std::fs::create_dir_all(dir).map_err(|e| PerceptionError::Burn(e.to_string()))?;
        let m = self.model_ref();

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
            (&m.env_layer.linear, ENV_DIM),
            (&m.visual_layer.linear, 384usize),
            (&m.symbolic_layer.linear, 384usize),
            (&m.language_layer.linear, LANG_HIDDEN),
        ];
        for (i, (layer, n_in)) in enc_layers.iter().enumerate() {
            let n_units = layer.weight.val().dims()[1];
            let w: Vec<f32> = layer.weight.val().to_f32_vec();
            write(&format!("W_enc_{i}"), w, vec![*n_in, n_units])?;
        }

        // ── Cross-modal projection [CROSS_MODAL_DIM, 384] + bias [384] ──────
        {
            let dims = m.cross_modal_proj.weight.val().dims();
            let w: Vec<f32> = m.cross_modal_proj.weight.val().to_f32_vec();
            write("W_cross", w, vec![dims[0], dims[1]])?;
            // cross_modal_proj has bias=true; bias shape [384]
            if let Some(bias) = m.cross_modal_proj.bias.as_ref() {
                let b: Vec<f32> = bias.val().to_f32_vec();
                write("W_cross_bias", b, vec![dims[1]])?;
            }
        }

        // ── BindingArea region projections [N_UNITS, D_PROJ] ─────────────────
        let region_projs = [
            &m.wake_binding.region_0,
            &m.wake_binding.region_1,
            &m.wake_binding.region_2,
            &m.wake_binding.region_3,
            &m.wake_binding.region_4,
        ];
        for (i, proj) in region_projs.iter().enumerate() {
            let dims = proj.weight.val().dims();
            let w: Vec<f32> = proj.weight.val().to_f32_vec();
            write(&format!("W_region_{i}"), w, vec![dims[0], dims[1]])?;
        }

        // ── Attention QKV [D_PROJ, D_PROJ] ───────────────────────────────────
        for (name, param) in [
            ("W_q", &m.wake_binding.w_q),
            ("W_k", &m.wake_binding.w_k),
            ("W_v", &m.wake_binding.w_v),
        ] {
            let dims = param.weight.val().dims();
            let w: Vec<f32> = param.weight.val().to_f32_vec();
            write(name, w, vec![dims[0], dims[1]])?;
        }

        // ── GRU weights [in, out] (burn convention; Python script transposes) ─
        let gru = &m.wake_binding.temporal_gru;
        for (gate_name, gate) in [
            ("update", &gru.update_gate),
            ("reset", &gru.reset_gate),
            ("new", &gru.new_gate),
        ] {
            let wi_dims = gate.input_transform.weight.val().dims();
            let wi: Vec<f32> = gate.input_transform.weight.val().to_f32_vec();
            write(
                &format!("gru_{gate_name}_input_W"),
                wi,
                vec![wi_dims[0], wi_dims[1]],
            )?;

            let wh_dims = gate.hidden_transform.weight.val().dims();
            let wh: Vec<f32> = gate.hidden_transform.weight.val().to_f32_vec();
            write(
                &format!("gru_{gate_name}_hidden_W"),
                wh,
                vec![wh_dims[0], wh_dims[1]],
            )?;
        }

        // ── temporal_proj [GRU_HIDDEN, EPISODE_DIM] ──────────────────────────
        {
            let dims = m.wake_binding.temporal_proj.weight.val().dims();
            let w: Vec<f32> = m.wake_binding.temporal_proj.weight.val().to_f32_vec();
            write("W_temporal_proj", w, vec![dims[0], dims[1]])?;
        }

        // ── Write manifest ────────────────────────────────────────────────────
        // k_active per region (speech/env/visual/symbolic/language order, matching
        // W_region_{i}/sp_{i}): the ONNX binding graph's sparse-gather inputs need a
        // static per-region k baked in at export time. Read directly from each
        // KWtaLayer's live field rather than recomputed in Python from n_units — this
        // codebase has hit the "value re-derived in Python drifts from Rust ground
        // truth" bug three times already (§17, §22, §24).
        let k_active = [
            m.speech_layer.k_active,
            m.env_layer.k_active,
            m.visual_layer.k_active,
            m.symbolic_layer.k_active,
            m.language_layer.k_active,
        ];
        let manifest_json = serde_json::to_string_pretty(&serde_json::json!({
            "tensors": manifest,
            "k_active": k_active,
        }))
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

    /// `(mean |duty_cycle - k_active/n_units|, max duty_cycle)` over every k-WTA unit of all 5
    /// regions. Health summary of the homeostatic boosting state (Core metrics
    /// `KwtaDutyDevMean` / `KwtaDutyMax`).
    pub fn duty_cycle_summary(&self) -> (f32, f32) {
        let m = self.model_ref();
        let (mut dev_sum, mut n, mut max) = (0.0f64, 0usize, 0.0f32);
        for idx in 0..5 {
            let layer = m.layer(idx);
            let target = layer.k_active as f32 / layer.n_units as f32;
            for &dc in &layer.duty_cycle {
                dev_sum += (dc - target).abs() as f64;
                max = max.max(dc);
            }
            n += layer.duty_cycle.len();
        }
        ((dev_sum / n.max(1) as f64) as f32, max)
    }

    /// Per-region duty-cycle diagnostics (gaps.md §50): how the firing load splits between the
    /// original `N_UNITS` and grown units, and which units saturate. Logged once per sleep cycle.
    pub fn duty_cycle_report(&self) -> Vec<RegionDutyReport> {
        let m = self.model_ref();
        (0..5)
            .map(|idx| {
                let layer = m.layer(idx);
                let dc = &layer.duty_cycle;
                let base_end = N_UNITS.min(dc.len());
                let mut order: Vec<usize> = (0..dc.len()).collect();
                order.sort_by(|&a, &b| {
                    dc[b]
                        .partial_cmp(&dc[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let top: Vec<(usize, f32)> = order.iter().take(5).map(|&i| (i, dc[i])).collect();
                let sum_base: f32 = dc[..base_end].iter().sum();
                let sum_grown: f32 = dc[base_end..].iter().sum();
                RegionDutyReport {
                    region: idx,
                    n_units: dc.len(),
                    k_active: layer.k_active,
                    sum_dc_base: sum_base,
                    sum_dc_grown: sum_grown,
                    base_units_over_half: dc[..base_end].iter().filter(|&&d| d > 0.5).count(),
                    grown_units_ever_fired: dc[base_end..].iter().filter(|&&d| d > 1e-4).count(),
                    base_units_silent: dc[..base_end].iter().filter(|&&d| d <= 1e-4).count(),
                    top5: top,
                    mean_abs_active_base: layer.slot_stats.base_abs_sum
                        / layer.slot_stats.base_n as f64,
                    mean_abs_active_grown: layer.slot_stats.grown_abs_sum
                        / layer.slot_stats.grown_n as f64,
                    grown_slot_share: layer.slot_stats.grown_n as f64
                        / (layer.slot_stats.base_n + layer.slot_stats.grown_n).max(1) as f64,
                }
            })
            .collect()
    }

    /// Return per-region duty cycles (for ONNX validation: must pass same dc to Python topk).
    pub fn duty_cycles(&self) -> [Vec<f32>; 5] {
        let m = self.model_ref();
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
        let device = self.device.clone();
        let m = self.model_ref();

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
        let offsets = [
            (CM_OFF_SPEECH, 384usize),
            (CM_OFF_ENV, ENV_DIM),
            (CM_OFF_VISUAL, 384),
            (CM_OFF_LANG, LANG_HIDDEN),
        ];
        let srcs: [&[f32]; 4] = [speech, env, visual, lang];
        for (src, (off, dim)) in srcs.iter().zip(offsets.iter()) {
            let len = src.len().min(*dim);
            concat[*off..*off + len].copy_from_slice(&src[..len]);
        }
        let cross_t: Tensor<2> =
            Tensor::from_data(TensorData::new(concat, [1, CROSS_MODAL_DIM]), &device);
        let symbolic_enc: Vec<f32> = m
            .cross_modal_proj
            .forward(Tensor::<2>::from_inner(cross_t))
            .inner()
            .to_f32_vec();

        let enc_inputs: [&[f32]; 5] = [speech, env, visual, &symbolic_enc, lang];
        let layers = [
            &m.speech_layer.linear,
            &m.env_layer.linear,
            &m.visual_layer.linear,
            &m.symbolic_layer.linear,
            &m.language_layer.linear,
        ];

        std::array::from_fn(|i| {
            let t: Tensor<2> = Tensor::from_data(
                TensorData::new(enc_inputs[i].to_vec(), [1, enc_inputs[i].len()]),
                &device,
            );
            layers[i]
                .forward(Tensor::<2>::from_inner(t))
                .inner()
                .to_f32_vec()
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
        let device = self.device.clone();
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
                n_units: dense.len(),
            }
        });
        let act_refs: [&SparseActivation; 5] = [&acts[0], &acts[1], &acts[2], &acts[3], &acts[4]];
        let ep = self.model_mut().wake_binding.forward(
            &act_refs,
            prev_ep.map(|p| p.to_vec()).as_ref(),
            &device,
        );
        let gru_h = self
            .model
            .as_ref()
            .unwrap()
            .wake_binding
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
        let device = self.device.clone();
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
                const ENC_REGIONS: [(usize, usize); 4] =
                    [(0, 384), (1, ENV_DIM), (2, 384), (4, LANG_HIDDEN)];
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
                        let sparse: [SparseActivation; 5] = {
                            let m = self.model_ref();
                            [
                                m.speech_layer.topk_from_z_no_dc(z0),
                                m.env_layer.topk_from_z_no_dc(z1),
                                m.visual_layer.topk_from_z_no_dc(z2),
                                m.symbolic_layer.topk_from_z_no_dc(z3),
                                m.language_layer.topk_from_z_no_dc(z4),
                            ]
                        };
                        // ORT binding — restore hidden so settle doesn't advance live GRU.
                        let ort = self.ort_session.as_mut().unwrap();
                        match ort.run_binding(&sparse, &current.0) {
                            Ok(ep_vec) => {
                                if let Some(ref h) = saved_hidden {
                                    ort.gru_hidden = h.clone();
                                }
                                EpisodeVector(ep_vec)
                            }
                            Err(e) => {
                                tracing::warn!(?e, "ORT binding in settle (encoder path) failed");
                                somnus_metrics::counted_core!(
                                    somnus_metrics::Metric::OrtBindingFallback,
                                    1
                                );
                                if let Some(ref h) = saved_hidden {
                                    ort.gru_hidden = h.clone();
                                }
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(?e, "ORT encoder in settle failed; aborting settle");
                        somnus_metrics::counted_core!(somnus_metrics::Metric::OrtEncoderFallback, 1);
                        break;
                    }
                }
            } else {
                // ── Paths B & C: burn kwta (both), then ORT or burn binding ─────
                let mut acts: [Option<SparseActivation>; 5] = Default::default();
                {
                    let model = self.model.as_mut().expect(
                        "PerceptionProcessor.model accessed while taken for an optimizer step",
                    );
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
                            acts[i] = Some(model.layer(i).forward_no_dc(&blended, &device));
                        }
                    }
                } // model borrow dropped

                if use_ort {
                    // ── Path B: ORT binding ── missing regions become an empty
                    // placeholder SparseActivation; run_binding pads it to exactly
                    // k_active with zero-contribution entries itself.
                    let region_sizes = self.region_sizes();
                    let region_ids = [
                        RegionId::AudioSpeech,
                        RegionId::AudioEnv,
                        RegionId::Visual,
                        RegionId::Symbolic,
                        RegionId::Language,
                    ];
                    let sparse: [SparseActivation; 5] = std::array::from_fn(|i| {
                        acts[i].clone().unwrap_or_else(|| SparseActivation {
                            region: region_ids[i],
                            indices: vec![],
                            values: vec![],
                            n_units: region_sizes[i],
                        })
                    });
                    let ort = self.ort_session.as_mut().unwrap();
                    match ort.run_binding(&sparse, &current.0) {
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
                            somnus_metrics::counted_core!(somnus_metrics::Metric::OrtBindingFallback, 1);
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
                            self.model_mut().wake_binding.forward(
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
                    self.model_mut()
                        .wake_binding
                        .forward(&[a0, a1, a2, a3, a4], None, &device)
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
        let beta = self.prior_betas[idx];
        if beta <= 0.0 {
            return (values.to_vec(), None);
        }
        match &self.prior_predictions[idx] {
            None => (values.to_vec(), None),
            Some(prior) => {
                let cosine = cosine_sim(values, prior);
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

    fn layer_mut(model: &mut PerceptionModel, region: RegionId) -> &mut KWtaLayer {
        match region {
            RegionId::AudioSpeech => &mut model.speech_layer,
            RegionId::AudioEnv => &mut model.env_layer,
            RegionId::Visual => &mut model.visual_layer,
            RegionId::Symbolic => &mut model.symbolic_layer,
            RegionId::Language => &mut model.language_layer,
        }
    }

    fn decoder_mut(model: &mut PerceptionModel, region: RegionId) -> &mut SaeDecoder {
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
fn scale_weight(linear: &mut Linear, scale: f32) {
    let id = linear.weight.id;
    let new_w = linear.weight.val().mul_scalar(scale);
    linear.weight = Param::initialized(id, new_w);
}

/// Cosine similarity between two vectors (handles zero-norm cases).
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

/// Cold-start placeholder activation for the ORT-encoder path (bypasses the burn k-WTA
/// layer, so it never gets a real `SparseActivation` until `try_bind` runs). Sized from
/// the layer's *current* `n_units` (which may already reflect prior growth after a
/// snapshot reload) rather than the compile-time baseline, and `k` scaled to match —
/// otherwise a grown region would get a placeholder with a stale, too-small k/n ratio.
fn bootstrap_placeholder(region: RegionId, n_units: usize) -> SparseActivation {
    let k = region::scaled_k(n_units);
    SparseActivation {
        region,
        indices: (0..k as u16).collect(),
        values: vec![1.0f32; k],
        n_units,
    }
}

fn apply_gain(values: &[f32], gain: f32) -> Vec<f32> {
    if (gain - 1.0).abs() < 1e-6 {
        return values.to_vec();
    }
    values.iter().map(|&x| x * gain).collect()
}

#[cfg(test)]
mod binding_replay_tests {
    //! ADR 0041 (gaps.md §33) positive-case pin: grown binding-projection rows
    //! learn once grown units are active. Lives in-crate (not in `tests/`)
    //! because it must make grown units win k-WTA deterministically, which
    //! requires encoder-weight surgery the public API doesn't expose.

    use super::*;

    /// `grow_region` inits new encoder columns at Normal(0, 0.01) — far below
    /// the Kaiming-scale base columns — so on a single repeated input grown
    /// units never beat the base winner set on a test-sized step budget (the
    /// duty-cycle boost, ≤ k/n ≈ 0.02 growing ~0.001/step, cannot bridge the
    /// z-score gap; grown units break in only on production timescales, via
    /// homeostatic rotation over many cycles of a diverse corpus — pinned as
    /// the control arm by `binding_replay_trains_only_active_binding_rows` in
    /// tests/burn_path_contracts.rs). To pin the §33 fix itself — gradients
    /// reaching grown binding rows through the live binding forward — set the
    /// grown columns to the input direction at unit scale so they win k-WTA
    /// from the first step.
    #[test]
    fn binding_replay_trains_grown_binding_rows_when_active() {
        let device = Device::flex();
        let mut proc = PerceptionProcessor::new(device.clone());
        const N_GROWN: usize = 8;
        proc.grow_region(RegionId::AudioSpeech, N_GROWN);

        let regional: [Vec<f32>; 5] = std::array::from_fn(|i| {
            (0..REGION_DIMS[i])
                .map(|j| ((j as f32 + 1.0) * 0.01 * (i as f32 + 1.0)).sin())
                .collect()
        });

        {
            let model = proc.model.as_mut().unwrap();
            let layer = &mut model.speech_layer;
            let n = layer.n_units;
            assert_eq!(n, N_UNITS + N_GROWN);
            let n_in = REGION_DIMS[RegionId::AudioSpeech.as_index()];
            let mut flat: Vec<f32> = layer.linear.weight.val().inner().to_f32_vec();
            let norm: f32 = regional[0].iter().map(|x| x * x).sum::<f32>().sqrt();
            for c in N_UNITS..n {
                for (r, &x) in regional[0].iter().enumerate() {
                    flat[r * n + c] = x / norm;
                }
            }
            let new_w: Tensor<2> = Tensor::from_data(TensorData::new(flat, [n_in, n]), &device);
            layer.linear.weight = Param::initialized(ParamId::new(), new_w);
        }
        // Mirror the production sequence: growth in REM, then the next NREM
        // opens with refresh_autodiff_state, remapping every param to a fresh
        // require_grad leaf.
        proc.refresh_autodiff_state();

        // The None-context binding pass must keep the GRU temporal components
        // out of the graph entirely (ADR 0041): no gradient entry, so the
        // optimizer never touches them.
        let temporal_norm = |p: &PerceptionProcessor| {
            let m = p.model.as_ref().unwrap();
            let flat: Vec<f32> = m.binding.temporal_proj.weight.val().inner().to_f32_vec();
            flat.iter().map(|v| v * v).sum::<f32>()
        };
        let temporal_before = temporal_norm(&proc);

        let norms_before = proc.binding_proj_row_norms(RegionId::AudioSpeech);
        assert_eq!(norms_before.len(), N_UNITS + N_GROWN);
        assert!(
            norms_before[N_UNITS..].iter().all(|&n| n == 0.0),
            "grown binding rows must start exactly zero"
        );

        let first_loss = proc
            .binding_replay_step_batch(&[&regional], 1.0, false)
            .unwrap();
        assert!(first_loss.is_finite() && first_loss >= 0.0);
        let mut last_loss = first_loss;
        for _ in 0..29 {
            last_loss = proc
                .binding_replay_step_batch(&[&regional], 1.0, false)
                .unwrap();
            assert!(last_loss.is_finite(), "binding replay loss went non-finite");
        }
        assert!(
            last_loss < first_loss,
            "binding replay loss did not decrease on repeated input: {first_loss} -> {last_loss}"
        );

        let norms_after = proc.binding_proj_row_norms(RegionId::AudioSpeech);
        assert!(
            norms_after[N_UNITS..].iter().any(|&n| n > 0.0),
            "grown binding rows must move off zero once active grown units' gradients reach them"
        );
        assert!(
            norms_after[..N_UNITS].iter().any(|&n| n > 0.0),
            "base binding rows must remain live"
        );
        assert_eq!(
            temporal_norm(&proc),
            temporal_before,
            "temporal components must not train in the None-context binding pass"
        );
    }

    /// `freeze_heads` (gaps.md §33 candidate (d), 2026-08-16): the loss must still fall (proof
    /// gradient reaches the binding weights through the heads) while the heads' own weights
    /// stay byte-identical (proof they're excluded from this step's optimizer.step()). Pins
    /// the mechanism before piloting it — same discipline as the other `binding_replay_tests`.
    #[test]
    fn binding_replay_freeze_heads_trains_binding_not_heads() {
        let device = Device::flex();
        let mut proc = PerceptionProcessor::new(device.clone());

        let regional: [Vec<f32>; 5] = std::array::from_fn(|i| {
            (0..REGION_DIMS[i])
                .map(|j| ((j as f32 + 1.0) * 0.01 * (i as f32 + 1.0)).sin())
                .collect()
        });

        let head_data = |p: &PerceptionProcessor| -> Vec<f32> {
            p.model
                .as_ref()
                .unwrap()
                .speech_head
                .weight
                .val()
                .to_f32_vec()
        };
        let binding_norm = |p: &PerceptionProcessor| -> f32 {
            p.binding_proj_row_norms(RegionId::AudioSpeech).iter().sum()
        };

        let head_before = head_data(&proc);
        let binding_before = binding_norm(&proc);

        let first_loss = proc
            .binding_replay_step_batch(&[&regional], 1.0, true)
            .unwrap();
        assert!(first_loss.is_finite() && first_loss >= 0.0);
        let mut last_loss = first_loss;
        for _ in 0..29 {
            last_loss = proc
                .binding_replay_step_batch(&[&regional], 1.0, true)
                .unwrap();
            assert!(last_loss.is_finite(), "binding replay loss went non-finite");
        }
        assert!(
            last_loss < first_loss,
            "binding replay loss did not decrease under freeze_heads: {first_loss} -> {last_loss}"
        );

        assert_eq!(
            head_data(&proc),
            head_before,
            "head weights must stay byte-identical under freeze_heads=true"
        );
        assert_ne!(
            binding_norm(&proc),
            binding_before,
            "binding weights must still move under freeze_heads=true \
             (gradient flows through the frozen heads, not into them)"
        );
    }
}
