pub mod binding;
pub mod error;
pub mod region;
pub mod sae;

pub use binding::{BindingArea, EpisodeVector, EPISODE_DIM};
pub use error::{PerceptionError, PerceptionResult};
pub use region::{KWtaLayer, RegionId, SparseActivation, K_ACTIVE, N_UNITS};
pub use sae::SaeDecoder;

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
use std::collections::VecDeque;

pub type TrainB = Autodiff<NdArray>;
type PercOptim = OptimizerAdaptor<AdamW, PerceptionModel<TrainB>, TrainB>;

/// Per-region input dimensions. Order: [AudioSpeech, AudioEnv, Visual, Symbolic, Language].
pub const REGION_DIMS: [usize; 5] = [384, 512, 384, 384, 896];

/// prior_cosine below this gate activates settling (ADR 0011).
const SETTLE_COSINE_GATE: f32 = 0.85;

/// L2 delta below which settling is considered converged (ADR 0011).
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
        }
    }
}

/// Owns the full cognitive-layer forward pass and sleep-time training.
pub struct PerceptionProcessor {
    model: Option<PerceptionModel<TrainB>>,
    optimizer: PercOptim,
    current_lr: f64,
    device: NdArrayDevice,

    prior_predictions: [Option<Vec<f32>>; 5],
    prior_beta: f32,
    latest: [Option<SparseActivation>; 5],
    latest_inputs: [Option<Vec<f32>>; 5],
    settle_steps: usize,
    settle_beta: f32,
    region_gain: [f32; 5],
    last_language_input: Option<Vec<f32>>,
    working_memory: VecDeque<Vec<f32>>,
    wm_size: usize,
}

impl PerceptionProcessor {
    pub fn new(device: NdArrayDevice) -> Self {
        let model = PerceptionModel::new(&device);
        let optimizer = AdamWConfig::new().init::<TrainB, PerceptionModel<TrainB>>();
        Self {
            model: Some(model),
            optimizer,
            current_lr: TrainConfig::default().lr,
            device,
            prior_predictions: [None, None, None, None, None],
            prior_beta: 0.0,
            latest: [None, None, None, None, None],
            latest_inputs: [None, None, None, None, None],
            settle_steps: 0,
            settle_beta: 0.5,
            region_gain: [1.0; 5],
            last_language_input: None,
            working_memory: VecDeque::new(),
            wm_size: 0,
        }
    }

    // ── Ingest ──────────────────────────────────────────────────────────────

    /// Ingest a 384-dim Whisper speech embedding.
    /// Updates AudioSpeech and Symbolic regions (same input, independent weights).
    pub fn ingest_speech(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let device = self.device.clone();
        let gained_speech = apply_gain(values, self.region_gain[0]);
        let gained_symbolic = apply_gain(values, self.region_gain[3]);
        let (speech_input, speech_cos) = self.blend_with_prior(&gained_speech, 0);
        let (symbolic_input, symbolic_cos) = self.blend_with_prior(&gained_symbolic, 3);
        if let Some(c) = speech_cos {
            tracing::info!(region = "speech", prior_cosine = c, "top-down prior");
        }
        if let Some(c) = symbolic_cos {
            tracing::info!(region = "symbolic", prior_cosine = c, "top-down prior");
        }

        let (speech_act, symbolic_act) = {
            let model = self.model.as_mut().unwrap();
            let sa = model.speech_layer.forward(&speech_input, &device);
            let ya = model.symbolic_layer.forward(&symbolic_input, &device);
            (sa, ya)
        };

        let (speech_err, symbolic_err) = {
            let model = self.model.as_ref().unwrap();
            let se = model
                .speech_decoder
                .reconstruction_error(values, &speech_act, &device);
            let ye = model
                .symbolic_decoder
                .reconstruction_error(values, &symbolic_act, &device);
            (se, ye)
        };

        self.latest[RegionId::AudioSpeech.as_index()] = Some(speech_act);
        self.latest[RegionId::Symbolic.as_index()] = Some(symbolic_act);
        self.latest_inputs[RegionId::AudioSpeech.as_index()] = Some(values.to_vec());
        self.latest_inputs[RegionId::Symbolic.as_index()] = Some(values.to_vec());

        self.refresh_language_cache();
        let wm = self.wm_mean_vec();
        let (episode, regional_inputs) = self.try_bind(wm.as_deref());
        let min_cos = min_cosine_2(speech_cos, symbolic_cos);
        let episode = self.maybe_settle_and_update(episode, min_cos);
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
            prior_cosines,
        })
    }

    /// Ingest a 512-dim CLAP env embedding.
    pub fn ingest_env(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let device = self.device.clone();
        let gained = apply_gain(values, self.region_gain[1]);
        let (input, cosine) = self.blend_with_prior(&gained, 1);
        if let Some(c) = cosine {
            tracing::info!(region = "env", prior_cosine = c, "top-down prior");
        }

        let act = self
            .model
            .as_mut()
            .unwrap()
            .env_layer
            .forward(&input, &device);
        let err = self
            .model
            .as_ref()
            .unwrap()
            .env_decoder
            .reconstruction_error(values, &act, &device);
        self.latest[RegionId::AudioEnv.as_index()] = Some(act);
        self.latest_inputs[RegionId::AudioEnv.as_index()] = Some(values.to_vec());

        self.refresh_language_cache();
        let wm = self.wm_mean_vec();
        let (episode, regional_inputs) = self.try_bind(wm.as_deref());
        let episode = self.maybe_settle_and_update(episode, cosine);
        self.push_to_wm(&episode);

        let mut prior_cosines = [None; 5];
        prior_cosines[1] = cosine;
        Ok(PerceptionOutput {
            recon_errors: vec![(RegionId::AudioEnv, err)],
            episode,
            regional_inputs,
            prior_cosines,
        })
    }

    /// Ingest a 384-dim DINOv2 visual embedding.
    pub fn ingest_visual(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let device = self.device.clone();
        let gained = apply_gain(values, self.region_gain[2]);
        let (input, cosine) = self.blend_with_prior(&gained, 2);
        if let Some(c) = cosine {
            tracing::info!(region = "visual", prior_cosine = c, "top-down prior");
        }

        let act = self
            .model
            .as_mut()
            .unwrap()
            .visual_layer
            .forward(&input, &device);
        let err = self
            .model
            .as_ref()
            .unwrap()
            .visual_decoder
            .reconstruction_error(values, &act, &device);
        self.latest[RegionId::Visual.as_index()] = Some(act);
        self.latest_inputs[RegionId::Visual.as_index()] = Some(values.to_vec());

        self.refresh_language_cache();
        let wm = self.wm_mean_vec();
        let (episode, regional_inputs) = self.try_bind(wm.as_deref());
        let episode = self.maybe_settle_and_update(episode, cosine);
        self.push_to_wm(&episode);

        let mut prior_cosines = [None; 5];
        prior_cosines[2] = cosine;
        Ok(PerceptionOutput {
            recon_errors: vec![(RegionId::Visual, err)],
            episode,
            regional_inputs,
            prior_cosines,
        })
    }

    /// Ingest a 896-dim language embedding (Qwen mean-pooled token embeddings).
    pub fn ingest_language(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let device = self.device.clone();
        self.last_language_input = Some(values.to_vec());
        let gained = apply_gain(values, self.region_gain[4]);
        let (input, cosine) = self.blend_with_prior(&gained, 4);
        if let Some(c) = cosine {
            tracing::info!(region = "language", prior_cosine = c, "top-down prior");
        }

        let act = self
            .model
            .as_mut()
            .unwrap()
            .language_layer
            .forward(&input, &device);
        let err = self
            .model
            .as_ref()
            .unwrap()
            .language_decoder
            .reconstruction_error(values, &act, &device);
        self.latest[RegionId::Language.as_index()] = Some(act);
        self.latest_inputs[RegionId::Language.as_index()] = Some(values.to_vec());

        let wm = self.wm_mean_vec();
        let (episode, regional_inputs) = self.try_bind(wm.as_deref());
        let episode = self.maybe_settle_and_update(episode, cosine);
        self.push_to_wm(&episode);

        let mut prior_cosines = [None; 5];
        prior_cosines[4] = cosine;
        Ok(PerceptionOutput {
            recon_errors: vec![(RegionId::Language, err)],
            episode,
            regional_inputs,
            prior_cosines,
        })
    }

    fn maybe_settle_and_update(
        &mut self,
        episode: Option<EpisodeVector>,
        cosine: Option<f32>,
    ) -> Option<EpisodeVector> {
        if let Some(ep) = episode {
            let should_settle =
                self.settle_steps > 0 && cosine.is_none_or(|c| c < SETTLE_COSINE_GATE);
            let refined = if should_settle {
                let (r, _) = self.settle_episode(&ep);
                r
            } else {
                ep
            };
            self.update_prior_predictions(&refined.0);
            Some(refined)
        } else {
            None
        }
    }

    fn refresh_language_cache(&mut self) {
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
    }

    #[allow(clippy::type_complexity)]
    fn try_bind(
        &self,
        wm_context: Option<&[f32]>,
    ) -> (Option<EpisodeVector>, Option<[Vec<f32>; 5]>) {
        match (
            self.latest[0].as_ref(),
            self.latest[1].as_ref(),
            self.latest[2].as_ref(),
            self.latest[3].as_ref(),
        ) {
            (Some(sp), Some(en), Some(vi), Some(sy)) => {
                let device = self.device.clone();
                let lang_zero = SparseActivation {
                    region: RegionId::Language,
                    indices: vec![],
                    values: vec![],
                    n_units: N_UNITS,
                };
                let lang = self.latest[4].as_ref().unwrap_or(&lang_zero);
                let ep = self.model.as_ref().unwrap().binding.forward(
                    &[sp, en, vi, sy, lang],
                    wm_context,
                    &device,
                );
                let regional = [
                    self.latest_inputs[0].clone().unwrap_or_default(),
                    self.latest_inputs[1].clone().unwrap_or_default(),
                    self.latest_inputs[2].clone().unwrap_or_default(),
                    self.latest_inputs[3].clone().unwrap_or_default(),
                    self.latest_inputs[4].clone().unwrap_or_default(),
                ];
                (Some(ep), Some(regional))
            }
            _ => (None, None),
        }
    }

    // ── Sleep-time training ──────────────────────────────────────────────────

    /// Run one training step on a stored episode trace (SAE reconstruction loss).
    pub fn replay_step(&mut self, regional: &[Vec<f32>; 5]) -> PerceptionResult<f32> {
        let device = self.device.clone();
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
        let device = self.device.clone();
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
        let device = self.device.clone();
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

    /// Run one wake-time reconstruction step on a just-produced episode.
    pub fn wake_reconstruct_step(
        &mut self,
        episode: &[f32],
        regional: &[Vec<f32>; 5],
    ) -> PerceptionResult<f32> {
        let device = self.device.clone();
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
            self.model = Some(model);
            Ok(loss_val / n.max(1) as f32)
        } else {
            Ok(0.0)
        }
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// Measure SAE reconstruction MSE per region (inference, no weight update).
    pub fn measure_mse(&mut self, regional: &[Vec<f32>; 5]) -> PerceptionResult<[f32; 5]> {
        let device = self.device.clone();
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
        &self,
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
        let model = self.model.as_ref().unwrap();
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
        let episode = model.binding.forward(&[a0, a1, a2, a3, a4], None, &device);
        Ok(episode.0)
    }

    /// Mean output-head MSE for one episode — no backward step.
    pub fn measure_output_mse_inference(
        &self,
        episode: &[f32],
        regional: &[Vec<f32>; 5],
    ) -> PerceptionResult<f32> {
        let device = self.device.clone();
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
        let device = self.device.clone();
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

    /// Prune low-magnitude grown units from a region (Phase 12c).
    /// Only removes units with index >= N_UNITS. Returns number pruned.
    pub fn prune_region(&mut self, region: RegionId, prune_fraction: f32) -> usize {
        let device = self.device.clone();
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
                let norm: f32 = (0..n_in)
                    .map(|j| enc_flat[j * current_n + i].abs())
                    .sum();
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
                keep.iter().map(move |&i| enc[j * current_n + i]).collect::<Vec<_>>()
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
        let device = self.device.clone();
        let model = self.model.take().unwrap();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let path_str = path
            .to_str()
            .ok_or_else(|| PerceptionError::Burn("invalid path".into()))?;
        let model = model
            .load_file(path_str, &recorder, &device)
            .map_err(|e| PerceptionError::Burn(e.to_string()))?;
        self.model = Some(model);
        Ok(())
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

    fn settle_episode(&self, ep: &EpisodeVector) -> (EpisodeVector, usize) {
        let device = self.device.clone();
        let model = self.model.as_ref().unwrap();
        let mut current = ep.clone();

        for iter in 0..self.settle_steps {
            let cur_t: Tensor<TrainB, 2> = Tensor::from_data(
                TensorData::new(current.0.clone(), [1, EPISODE_DIM]),
                &device,
            );

            let mut preds: [Vec<f32>; 5] = Default::default();
            let heads = [
                &model.speech_head,
                &model.env_head,
                &model.visual_head,
                &model.symbolic_head,
                &model.language_head,
            ];
            for (i, head) in heads.iter().enumerate() {
                preds[i] = head
                    .forward(cur_t.clone())
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
                    .unwrap();
            }

            let mut acts: [Option<SparseActivation>; 5] = Default::default();
            for i in 0..5 {
                if let Some(raw) = &self.latest_inputs[i] {
                    let raw_norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                    let pred_norm = preds[i].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                    let amp_scale = raw_norm / pred_norm;
                    let blended: Vec<f32> = raw
                        .iter()
                        .zip(&preds[i])
                        .map(|(&r, &p)| {
                            (1.0 - self.settle_beta) * r + self.settle_beta * p * amp_scale
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
            let refined = model.binding.forward(&[a0, a1, a2, a3, a4], None, &device);

            let delta: f32 = current
                .0
                .iter()
                .zip(&refined.0)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            tracing::info!(iter, delta, "settle delta");
            let converged = delta < SETTLE_EPS;
            current = refined;
            if converged {
                return (current, iter + 1);
            }
        }

        (current, self.settle_steps)
    }

    fn update_prior_predictions(&mut self, episode: &[f32]) {
        let device = self.device.clone();
        let ep_t: Tensor<TrainB, 2> =
            Tensor::from_data(TensorData::new(episode.to_vec(), [1, EPISODE_DIM]), &device);
        let preds = {
            let model = self.model.as_ref().unwrap();
            let heads = [
                &model.speech_head,
                &model.env_head,
                &model.visual_head,
                &model.symbolic_head,
                &model.language_head,
            ];
            let mut out = [vec![], vec![], vec![], vec![], vec![]];
            for (i, head) in heads.iter().enumerate() {
                out[i] = head
                    .forward(ep_t.clone())
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
                    .unwrap();
            }
            out
        };
        for (i, p) in preds.into_iter().enumerate() {
            self.prior_predictions[i] = Some(p);
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

    fn wm_mean_vec(&self) -> Option<Vec<f32>> {
        if self.wm_size == 0 || self.working_memory.is_empty() {
            return None;
        }
        let n = self.working_memory.len() as f32;
        let mut mean = vec![0.0f32; EPISODE_DIM];
        for ep in &self.working_memory {
            for (m, v) in mean.iter_mut().zip(ep.iter()) {
                *m += v / n;
            }
        }
        Some(mean)
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

    fn layer_mut(
        model: &mut PerceptionModel<TrainB>,
        region: RegionId,
    ) -> &mut KWtaLayer<TrainB> {
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

fn min_cosine_2(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some(x), Some(y)) => Some(x.min(y)),
    }
}

fn apply_gain(values: &[f32], gain: f32) -> Vec<f32> {
    if (gain - 1.0).abs() < 1e-6 {
        return values.to_vec();
    }
    values.iter().map(|&x| x * gain).collect()
}
