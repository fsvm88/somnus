pub mod binding;
pub mod error;
pub mod region;
pub mod sae;

pub use binding::{BindingArea, EpisodeVector, EPISODE_DIM};
pub use error::{PerceptionError, PerceptionResult};
pub use region::{KWtaLayer, RegionId, SparseActivation, K_ACTIVE, N_UNITS};
pub use sae::SaeDecoder;

use candle_core::{DType, Device, Module, Tensor, Var};
use candle_nn::{linear_no_bias, AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::collections::VecDeque;

/// Hyperparameters for sleep-time training.
pub struct TrainConfig {
    pub lr: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self { lr: 1e-3 }
    }
}

/// Per-region input dimensions (encoder output sizes).
/// Order: [AudioSpeech, AudioEnv, Visual, Symbolic, Language].
pub const REGION_DIMS: [usize; 5] = [384, 512, 384, 384, 896];

/// prior_cosine below this threshold activates the settling loop (ADR 0011).
/// Calibrated 2026-05-17: familiar visual prior_cosine min=0.977, novel min=0.06–0.16.
/// 0.85 sits well below the familiar floor so settling never fires on known content,
/// while catching all genuinely novel episodes where settling reduces delta by ~42%.
const SETTLE_COSINE_GATE: f32 = 0.85;

/// L2 delta below which the settling loop is considered converged (ADR 0011).
// Calibrated 2026-05-17: familiar content has mean settle delta ~3.8 at iter=0 and
// diverges slightly with iterations; novel content (talking_head) has mean 6.1→3.5
// over 3 steps. 4.0 allows early exit on the lower tail of novel content without
// triggering on familiar content where settling is counterproductive.
const SETTLE_EPS: f32 = 4.0;

/// Reconstruction errors for each region updated in a single `ingest_*` call.
pub struct PerceptionOutput {
    /// Reconstruction errors for regions updated this call (region, MSE).
    pub recon_errors: Vec<(RegionId, f32)>,
    /// Episode vector from the binding area, produced once all four non-language
    /// regions have at least one activation.
    pub episode: Option<EpisodeVector>,
    /// Raw encoder inputs per region that produced this episode (Some when episode is Some).
    /// Order: [AudioSpeech(384), AudioEnv(512), Visual(384), Symbolic(384), Language(896)].
    /// Language slot is empty vec when no language input has been received.
    pub regional_inputs: Option<[Vec<f32>; 5]>,
    /// Cosine similarity of the top-down prior prediction vs the incoming encoder
    /// embedding, per region updated this call. Only Some when `prior_beta > 0`
    /// and a prior prediction exists for that region. Indexed by region id (0–4).
    pub prior_cosines: [Option<f32>; 5],
}

/// Owns the full cognitive-layer forward pass: five k-WTA layers, five SAE
/// decoders, and the binding area.
pub struct PerceptionProcessor {
    speech_layer: KWtaLayer,
    env_layer: KWtaLayer,
    visual_layer: KWtaLayer,
    symbolic_layer: KWtaLayer,
    language_layer: KWtaLayer,

    speech_decoder: SaeDecoder,
    env_decoder: SaeDecoder,
    visual_decoder: SaeDecoder,
    symbolic_decoder: SaeDecoder,
    language_decoder: SaeDecoder,

    binding: BindingArea,

    /// Per-region output heads: EPISODE_DIM → encoder_dim. Predict encoder
    /// embeddings from the episode vector (ADR 0009). Also serve as the
    /// top-down feedback path (ADR 0010).
    output_heads: [Linear; 5],

    /// Predicted encoder embeddings from the previous episode (ADR 0010).
    /// None until the first episode is produced. Updated after every episode.
    prior_predictions: [Option<Vec<f32>>; 5],
    /// Blend weight β for top-down prior injection. 0.0 = disabled (default).
    prior_beta: f32,

    /// Latest sparse activation per region; None until first embedding arrives.
    latest: [Option<SparseActivation>; 5],
    /// Latest raw encoder input per region; None until first embedding arrives.
    latest_inputs: [Option<Vec<f32>>; 5],

    /// Max settling iterations per episode (ADR 0011). 0 = disabled.
    settle_steps: usize,
    /// Fraction of output-head prediction blended into settling inputs (ADR 0011).
    settle_beta: f32,

    /// Per-region gain multipliers applied to raw encoder inputs before k-WTA (ADR 0012).
    /// Derived from the ActionDecoder's LM attention over region tokens each episode.
    /// Reset to 1.0 at sleep entry. Default: [1.0; 5] (no-op).
    region_gain: [f32; 5],

    /// Fixed-capacity ring of recent episode vectors for cross-episode context.
    /// Cleared at sleep entry. Disabled when wm_size == 0.
    working_memory: VecDeque<Vec<f32>>,
    wm_size: usize,

    /// Holds all trainable parameters; needed for optimizer, save, and load.
    varmap: VarMap,
    /// AdamW optimizer over all k-WTA + SAE + binding weights.
    optimizer: AdamW,

    device: Device,
}

impl PerceptionProcessor {
    pub fn new(device: Device) -> PerceptionResult<Self> {
        let varmap = VarMap::new();
        let vb = |name: &str| -> VarBuilder<'_> {
            VarBuilder::from_varmap(&varmap, DType::F32, &device).pp(name)
        };

        let speech_layer = KWtaLayer::new(
            RegionId::AudioSpeech,
            384,
            vb("speech_layer"),
            device.clone(),
        )?;
        let env_layer = KWtaLayer::new(RegionId::AudioEnv, 512, vb("env_layer"), device.clone())?;
        let visual_layer =
            KWtaLayer::new(RegionId::Visual, 384, vb("visual_layer"), device.clone())?;
        let symbolic_layer = KWtaLayer::new(
            RegionId::Symbolic,
            384,
            vb("symbolic_layer"),
            device.clone(),
        )?;
        let language_layer = KWtaLayer::new(
            RegionId::Language,
            896,
            vb("language_layer"),
            device.clone(),
        )?;

        let speech_decoder = SaeDecoder::new(
            RegionId::AudioSpeech,
            384,
            vb("speech_decoder"),
            device.clone(),
        )?;
        let env_decoder =
            SaeDecoder::new(RegionId::AudioEnv, 512, vb("env_decoder"), device.clone())?;
        let visual_decoder =
            SaeDecoder::new(RegionId::Visual, 384, vb("visual_decoder"), device.clone())?;
        let symbolic_decoder = SaeDecoder::new(
            RegionId::Symbolic,
            384,
            vb("symbolic_decoder"),
            device.clone(),
        )?;
        let language_decoder = SaeDecoder::new(
            RegionId::Language,
            896,
            vb("language_decoder"),
            device.clone(),
        )?;

        let binding = BindingArea::new(vb("binding"), device.clone())?;

        let vb_out = vb("output_heads");
        let output_heads = [
            linear_no_bias(EPISODE_DIM, REGION_DIMS[0], vb_out.pp("0"))?, // speech: 640→384
            linear_no_bias(EPISODE_DIM, REGION_DIMS[1], vb_out.pp("1"))?, // env: 640→512
            linear_no_bias(EPISODE_DIM, REGION_DIMS[2], vb_out.pp("2"))?, // visual: 640→384
            linear_no_bias(EPISODE_DIM, REGION_DIMS[3], vb_out.pp("3"))?, // symbolic: 640→384
            linear_no_bias(EPISODE_DIM, REGION_DIMS[4], vb_out.pp("4"))?, // language: 640→896
        ];

        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: TrainConfig::default().lr,
                ..Default::default()
            },
        )?;

        Ok(Self {
            speech_layer,
            env_layer,
            visual_layer,
            symbolic_layer,
            language_layer,
            speech_decoder,
            env_decoder,
            visual_decoder,
            symbolic_decoder,
            language_decoder,
            binding,
            output_heads,
            prior_predictions: [None, None, None, None, None],
            prior_beta: 0.0,
            latest: [None, None, None, None, None],
            latest_inputs: [None, None, None, None, None],
            settle_steps: 0,
            settle_beta: 0.5,
            region_gain: [1.0; 5],
            working_memory: VecDeque::new(),
            wm_size: 0,
            varmap,
            optimizer,
            device,
        })
    }

    /// Ingest a 384-dim Whisper speech embedding.
    /// Updates AudioSpeech and Symbolic regions (same input, independent weights).
    pub fn ingest_speech(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
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

        let speech_act = self.speech_layer.forward(&speech_input)?;
        let symbolic_act = self.symbolic_layer.forward(&symbolic_input)?;

        let speech_err = self
            .speech_decoder
            .reconstruction_error(values, &speech_act)?;
        let symbolic_err = self
            .symbolic_decoder
            .reconstruction_error(values, &symbolic_act)?;

        self.latest[RegionId::AudioSpeech.as_index()] = Some(speech_act);
        self.latest[RegionId::Symbolic.as_index()] = Some(symbolic_act);
        self.latest_inputs[RegionId::AudioSpeech.as_index()] = Some(values.to_vec());
        self.latest_inputs[RegionId::Symbolic.as_index()] = Some(values.to_vec());

        let wm = self.wm_mean_vec();
        let (episode, regional_inputs) = self.try_bind(wm.as_deref())?;
        let min_cos = min_cosine_2(speech_cos, symbolic_cos);
        let episode = self.maybe_settle_and_update(episode, min_cos)?;
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
        let gained = apply_gain(values, self.region_gain[1]);
        let (input, cosine) = self.blend_with_prior(&gained, 1);
        if let Some(c) = cosine {
            tracing::info!(region = "env", prior_cosine = c, "top-down prior");
        }
        let act = self.env_layer.forward(&input)?;
        let err = self.env_decoder.reconstruction_error(values, &act)?;
        self.latest[RegionId::AudioEnv.as_index()] = Some(act);
        self.latest_inputs[RegionId::AudioEnv.as_index()] = Some(values.to_vec());

        let wm = self.wm_mean_vec();
        let (episode, regional_inputs) = self.try_bind(wm.as_deref())?;
        let episode = self.maybe_settle_and_update(episode, cosine)?;
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
        let gained = apply_gain(values, self.region_gain[2]);
        let (input, cosine) = self.blend_with_prior(&gained, 2);
        if let Some(c) = cosine {
            tracing::info!(region = "visual", prior_cosine = c, "top-down prior");
        }
        let act = self.visual_layer.forward(&input)?;
        let err = self.visual_decoder.reconstruction_error(values, &act)?;
        self.latest[RegionId::Visual.as_index()] = Some(act);
        self.latest_inputs[RegionId::Visual.as_index()] = Some(values.to_vec());

        let wm = self.wm_mean_vec();
        let (episode, regional_inputs) = self.try_bind(wm.as_deref())?;
        let episode = self.maybe_settle_and_update(episode, cosine)?;
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

    /// Ingest a 896-dim Qwen language embedding (mean-pooled token embeddings).
    ///
    /// Language is optional: episodes are produced from regions 0–3; this region
    /// zero-pads the binding area's language slot when absent (ADR 0013).
    pub fn ingest_language(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let gained = apply_gain(values, self.region_gain[4]);
        let (input, cosine) = self.blend_with_prior(&gained, 4);
        if let Some(c) = cosine {
            tracing::info!(region = "language", prior_cosine = c, "top-down prior");
        }
        let act = self.language_layer.forward(&input)?;
        let err = self.language_decoder.reconstruction_error(values, &act)?;
        self.latest[RegionId::Language.as_index()] = Some(act);
        self.latest_inputs[RegionId::Language.as_index()] = Some(values.to_vec());

        let wm = self.wm_mean_vec();
        let (episode, regional_inputs) = self.try_bind(wm.as_deref())?;
        let episode = self.maybe_settle_and_update(episode, cosine)?;
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

    /// `cosine` is the minimum prior_cosine computed in this ingest call (across
    /// all regions updated). None means no prior exists yet → settle permissively.
    fn maybe_settle_and_update(
        &mut self,
        episode: Option<EpisodeVector>,
        cosine: Option<f32>,
    ) -> PerceptionResult<Option<EpisodeVector>> {
        if let Some(ep) = episode {
            let should_settle =
                self.settle_steps > 0 && cosine.is_none_or(|c| c < SETTLE_COSINE_GATE);
            let refined = if should_settle {
                let (r, _) = self.settle_episode(&ep)?;
                r
            } else {
                ep
            };
            self.update_prior_predictions(&refined.0)?;
            Ok(Some(refined))
        } else {
            Ok(None)
        }
    }

    /// Attempt to produce an episode vector.
    ///
    /// Regions 0–3 (AudioSpeech, AudioEnv, Visual, Symbolic) must all be present.
    /// Region 4 (Language) is optional: a zero SparseActivation is used when absent.
    #[allow(clippy::type_complexity)]
    fn try_bind(
        &self,
        wm_context: Option<&[f32]>,
    ) -> PerceptionResult<(Option<EpisodeVector>, Option<[Vec<f32>; 5]>)> {
        match (
            self.latest[0].as_ref(),
            self.latest[1].as_ref(),
            self.latest[2].as_ref(),
            self.latest[3].as_ref(),
        ) {
            (Some(sp), Some(en), Some(vi), Some(sy)) => {
                let lang_zero = SparseActivation {
                    region: RegionId::Language,
                    indices: vec![],
                    values: vec![],
                    n_units: N_UNITS,
                };
                let lang = self.latest[4].as_ref().unwrap_or(&lang_zero);
                let ep = self.binding.forward(&[sp, en, vi, sy, lang], wm_context)?;
                let regional = [
                    self.latest_inputs[0].clone().unwrap_or_default(),
                    self.latest_inputs[1].clone().unwrap_or_default(),
                    self.latest_inputs[2].clone().unwrap_or_default(),
                    self.latest_inputs[3].clone().unwrap_or_default(),
                    self.latest_inputs[4].clone().unwrap_or_default(),
                ];
                Ok((Some(ep), Some(regional)))
            }
            _ => Ok((None, None)),
        }
    }

    // ── Sleep-time training ─────────────────────────────────────────────────

    /// Run one training step on a stored episode trace.
    ///
    /// Computes SAE reconstruction MSE for each region using the stored encoder
    /// inputs, accumulates into a combined loss, and calls `optimizer.backward_step`.
    /// Returns mean MSE across regions (for monitoring).
    pub fn replay_step(&mut self, regional: &[Vec<f32>; 5]) -> PerceptionResult<f32> {
        let dev = self.device.clone();
        let mut total_loss: Option<Tensor> = None;
        let mut n_regions = 0usize;

        for (idx, input) in regional.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let (act, z) = self.layer_forward_tracked(idx, input)?;
            let mask_vec: Vec<f32> = {
                let mut m = vec![0.0f32; act.n_units];
                for &i in &act.indices {
                    m[i as usize] = 1.0;
                }
                m
            };
            let mask = Tensor::from_vec(mask_vec, (act.n_units,), &dev)?;
            let sparse_z = z.mul(&mask)?;
            let loss = self.decoder_mse_loss(idx, input, &sparse_z)?;
            total_loss = Some(match total_loss {
                None => loss,
                Some(acc) => acc.add(&loss)?,
            });
            n_regions += 1;
        }

        if let Some(loss) = total_loss {
            let loss_val = loss.to_scalar::<f32>()?;
            self.optimizer.backward_step(&loss)?;
            Ok(loss_val / n_regions.max(1) as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Batched NREM replay step: process `batch.len()` traces in a single
    /// forward+backward pass per region. One `optimizer.backward_step` for the
    /// whole batch — amortises optimizer overhead and enables larger BLAS tiles.
    ///
    /// Loss is mean MSE across all (sample, region) pairs; semantically
    /// equivalent to the mean of `batch.len()` sequential `replay_step` calls.
    pub fn replay_step_batch(&mut self, batch: &[&[Vec<f32>; 5]]) -> PerceptionResult<f32> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        let dev = self.device.clone();
        let mut total_loss: Option<Tensor> = None;
        let mut n_terms = 0usize;

        for region_idx in 0..5usize {
            // Collect non-empty inputs for this region; remember which batch slots contributed.
            let valid: Vec<&Vec<f32>> = batch
                .iter()
                .filter_map(|r| {
                    let v = &r[region_idx];
                    if v.is_empty() { None } else { Some(v) }
                })
                .collect();
            if valid.is_empty() {
                continue;
            }

            let b = valid.len();
            let n_in = valid[0].len();

            let flat: Vec<f32> = valid.iter().flat_map(|v| v.iter().copied()).collect();
            let input_batch = Tensor::from_vec(flat, (b, n_in), &dev)?;

            let (z_batch, mask_t) = match region_idx {
                0 => self.speech_layer.forward_tracked_batch(&input_batch, b)?,
                1 => self.env_layer.forward_tracked_batch(&input_batch, b)?,
                2 => self.visual_layer.forward_tracked_batch(&input_batch, b)?,
                3 => self.symbolic_layer.forward_tracked_batch(&input_batch, b)?,
                4 => self.language_layer.forward_tracked_batch(&input_batch, b)?,
                _ => unreachable!(),
            };

            let sparse_z = z_batch.mul(&mask_t)?;

            let region_loss = match region_idx {
                0 => self.speech_decoder.mse_loss_batch_tensor(&input_batch, &sparse_z)?,
                1 => self.env_decoder.mse_loss_batch_tensor(&input_batch, &sparse_z)?,
                2 => self.visual_decoder.mse_loss_batch_tensor(&input_batch, &sparse_z)?,
                3 => self.symbolic_decoder.mse_loss_batch_tensor(&input_batch, &sparse_z)?,
                4 => self.language_decoder.mse_loss_batch_tensor(&input_batch, &sparse_z)?,
                _ => unreachable!(),
            };

            total_loss = Some(match total_loss {
                None => region_loss,
                Some(acc) => acc.add(&region_loss)?,
            });
            n_terms += 1;
        }

        if let Some(loss) = total_loss {
            let loss_val = loss.to_scalar::<f32>()?;
            self.optimizer.backward_step(&loss)?;
            Ok(loss_val / n_terms.max(1) as f32)
        } else {
            Ok(0.0)
        }
    }

    fn layer_forward_tracked(
        &mut self,
        idx: usize,
        input: &[f32],
    ) -> PerceptionResult<(SparseActivation, Tensor)> {
        match idx {
            0 => self.speech_layer.forward_tracked(input),
            1 => self.env_layer.forward_tracked(input),
            2 => self.visual_layer.forward_tracked(input),
            3 => self.symbolic_layer.forward_tracked(input),
            4 => self.language_layer.forward_tracked(input),
            _ => unreachable!(),
        }
    }

    fn decoder_mse_loss(
        &self,
        idx: usize,
        input: &[f32],
        sparse_z: &Tensor,
    ) -> PerceptionResult<Tensor> {
        match idx {
            0 => self.speech_decoder.mse_loss_tensor(input, sparse_z),
            1 => self.env_decoder.mse_loss_tensor(input, sparse_z),
            2 => self.visual_decoder.mse_loss_tensor(input, sparse_z),
            3 => self.symbolic_decoder.mse_loss_tensor(input, sparse_z),
            4 => self.language_decoder.mse_loss_tensor(input, sparse_z),
            _ => unreachable!(),
        }
    }

    /// Measure SAE reconstruction MSE per region without updating weights.
    /// Returns per-region MSE: [speech, env, visual, symbolic, language].
    pub fn measure_mse(&mut self, regional: &[Vec<f32>; 5]) -> PerceptionResult<[f32; 5]> {
        let mut result = [0.0f32; 5];
        for (idx, input) in regional.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let act = self.layer_forward_inference(idx, input)?;
            result[idx] = self.decoder_recon_error(idx, input, &act)?;
        }
        Ok(result)
    }

    fn layer_forward_inference(
        &mut self,
        idx: usize,
        input: &[f32],
    ) -> PerceptionResult<SparseActivation> {
        match idx {
            0 => self.speech_layer.forward(input),
            1 => self.env_layer.forward(input),
            2 => self.visual_layer.forward(input),
            3 => self.symbolic_layer.forward(input),
            4 => self.language_layer.forward(input),
            _ => unreachable!(),
        }
    }

    fn decoder_recon_error(
        &self,
        idx: usize,
        input: &[f32],
        act: &SparseActivation,
    ) -> PerceptionResult<f32> {
        match idx {
            0 => self.speech_decoder.reconstruction_error(input, act),
            1 => self.env_decoder.reconstruction_error(input, act),
            2 => self.visual_decoder.reconstruction_error(input, act),
            3 => self.symbolic_decoder.reconstruction_error(input, act),
            4 => self.language_decoder.reconstruction_error(input, act),
            _ => unreachable!(),
        }
    }

    /// Apply global weight decay to all k-WTA and SAE Linear weights.
    pub fn homeostatic_decay(&mut self, rate: f32) -> PerceptionResult<()> {
        let scale = 1.0 - rate;
        let vars = self.varmap.all_vars();
        for var in &vars {
            let scaled = (var.as_tensor() * scale as f64)?;
            var.set(&scaled)?;
        }
        Ok(())
    }

    /// Grow a region by appending `n_new` units with small random weights.
    pub fn grow_region(&mut self, region: RegionId, n_new: usize) -> PerceptionResult<()> {
        let idx = region.as_index();
        let (enc_key, dec_key) = Self::layer_keys(region);
        let n_in = REGION_DIMS[idx];

        let (layer, _decoder) = self.layer_mut(region);
        let old_n = layer.n_units;
        let new_n = old_n + n_new;

        {
            let old_w = {
                let data = self.varmap.data().lock().unwrap();
                data[enc_key].as_tensor().clone()
            };
            let new_rows = Tensor::randn(0.0f32, 0.01, (n_new, n_in), &self.device)?;
            let expanded = Tensor::cat(&[&old_w, &new_rows], 0)?;
            let new_var = Var::from_tensor(&expanded)?;
            let new_tensor = new_var.as_tensor().clone();
            self.varmap
                .data()
                .lock()
                .unwrap()
                .insert(enc_key.to_string(), new_var);
            let layer = self.layer_mut(region).0;
            layer.linear = Linear::new(new_tensor, None);
            layer.n_units = new_n;
            layer.duty_cycle.extend(vec![0.0f32; n_new]);
        }

        {
            let old_w = {
                let data = self.varmap.data().lock().unwrap();
                data[dec_key].as_tensor().clone()
            };
            let new_cols = Tensor::zeros((n_in, n_new), DType::F32, &self.device)?;
            let expanded = Tensor::cat(&[&old_w, &new_cols], 1)?;
            let new_var = Var::from_tensor(&expanded)?;
            let new_tensor = new_var.as_tensor().clone();
            self.varmap
                .data()
                .lock()
                .unwrap()
                .insert(dec_key.to_string(), new_var);
            let decoder = self.layer_mut(region).1;
            decoder.replace_linear(Linear::new(new_tensor, None), new_n);
        }

        self.optimizer = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr: TrainConfig::default().lr,
                ..Default::default()
            },
        )?;

        tracing::info!(region = ?region, old_n, new_n, "region grown");
        Ok(())
    }

    /// Prune low-magnitude grown units from a region (Phase 12c).
    ///
    /// Only removes units with index ≥ N_UNITS (added by `grow_region`). Baseline
    /// units 0..N_UNITS are never touched — preserves snapshot compatibility and the
    /// fixed episode-vector contribution from `to_dense_n(N_UNITS)`.
    ///
    /// Selection: rank grown units by L1 norm of their encoder weight row ascending;
    /// remove the bottom `prune_fraction`. Rebuilds optimizer after reshape.
    ///
    /// Returns `Ok(0)` when there are no grown units to prune.
    pub fn prune_region(
        &mut self,
        region: RegionId,
        prune_fraction: f32,
    ) -> PerceptionResult<usize> {
        let (enc_key, dec_key) = Self::layer_keys(region);

        let current_n = self.layer_mut(region).0.n_units;
        let n_grown = current_n.saturating_sub(N_UNITS);
        if n_grown == 0 {
            return Ok(0);
        }

        let n_prune = ((n_grown as f32 * prune_fraction).round() as usize).min(n_grown);
        if n_prune == 0 {
            return Ok(0);
        }

        // Compute L1 norms for grown rows (indices N_UNITS..current_n)
        let enc_w_vec: Vec<Vec<f32>> = {
            let data = self.varmap.data().lock().unwrap();
            data[enc_key].as_tensor().to_vec2::<f32>()?
        };
        let mut grown_norms: Vec<(usize, f32)> = (N_UNITS..current_n)
            .map(|i| {
                let norm: f32 = enc_w_vec[i].iter().map(|&x| x.abs()).sum();
                (i, norm)
            })
            .collect();
        grown_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Indices to remove: lowest-norm grown units
        let remove: std::collections::HashSet<usize> =
            grown_norms[..n_prune].iter().map(|&(i, _)| i).collect();
        let keep: Vec<usize> = (0..current_n).filter(|i| !remove.contains(i)).collect();
        let new_n = keep.len();

        let keep_t = Tensor::from_vec(
            keep.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            (keep.len(),),
            &self.device,
        )?;

        // Slice encoder rows → [new_n, n_in]
        let enc_w = {
            let data = self.varmap.data().lock().unwrap();
            data[enc_key].as_tensor().clone()
        };
        let new_enc_w = enc_w.index_select(&keep_t, 0)?;
        let new_enc_var = Var::from_tensor(&new_enc_w)?;
        let new_enc_t = new_enc_var.as_tensor().clone();
        self.varmap
            .data()
            .lock()
            .unwrap()
            .insert(enc_key.to_string(), new_enc_var);

        // Slice decoder columns → [n_in, new_n]
        let dec_w = {
            let data = self.varmap.data().lock().unwrap();
            data[dec_key].as_tensor().clone()
        };
        let new_dec_w = dec_w.index_select(&keep_t, 1)?;
        let new_dec_var = Var::from_tensor(&new_dec_w)?;
        let new_dec_t = new_dec_var.as_tensor().clone();
        self.varmap
            .data()
            .lock()
            .unwrap()
            .insert(dec_key.to_string(), new_dec_var);

        // Update layer
        {
            let (layer, decoder) = self.layer_mut(region);
            layer.linear = Linear::new(new_enc_t, None);
            let old_dc = layer.duty_cycle.clone();
            layer.duty_cycle = keep.iter().map(|&i| old_dc[i]).collect();
            layer.n_units = new_n;
            decoder.replace_linear(Linear::new(new_dec_t, None), new_n);
        }

        // Rebuild optimizer (moment buffers are stale after shape change)
        self.optimizer = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr: TrainConfig::default().lr,
                ..Default::default()
            },
        )?;

        tracing::info!(
            region = ?region,
            old_n = current_n,
            new_n,
            n_pruned = n_prune,
            "region pruned"
        );
        Ok(n_prune)
    }

    /// Batched output-head training for NREM replay (ADR 0010).
    ///
    /// `episodes` and `regionals` are parallel slices of length B.
    /// For each output head, only samples with a non-empty regional input
    /// contribute to the loss. Single backward_step for the whole batch.
    pub fn wake_reconstruct_step_batch(
        &mut self,
        episodes: &[&[f32]],
        regionals: &[&[Vec<f32>; 5]],
    ) -> PerceptionResult<f32> {
        if episodes.is_empty() {
            return Ok(0.0);
        }
        let dev = self.device.clone();
        let ep_dim = episodes[0].len();
        let mut total_loss: Option<Tensor> = None;
        let mut n = 0usize;

        for head_idx in 0..5usize {
            let valid_pairs: Vec<(&[f32], &Vec<f32>)> = episodes
                .iter()
                .zip(regionals.iter())
                .filter_map(|(&ep, &r)| {
                    let inp = &r[head_idx];
                    if inp.is_empty() { None } else { Some((ep, inp)) }
                })
                .collect();

            if valid_pairs.is_empty() {
                continue;
            }

            let b = valid_pairs.len();
            let n_in = valid_pairs[0].1.len();

            let flat_eps: Vec<f32> = valid_pairs
                .iter()
                .flat_map(|(ep, _)| ep.iter().copied())
                .collect();
            let flat_targets: Vec<f32> = valid_pairs
                .iter()
                .flat_map(|(_, inp)| inp.iter().copied())
                .collect();

            let ep_batch = Tensor::from_vec(flat_eps, (b, ep_dim), &dev)?;
            let target_batch = Tensor::from_vec(flat_targets, (b, n_in), &dev)?;

            let pred = self.output_heads[head_idx].forward(&ep_batch)?; // [B, n_in]
            let loss = (pred - target_batch)?.sqr()?.mean_all()?;

            total_loss = Some(match total_loss {
                None => loss,
                Some(acc) => acc.add(&loss)?,
            });
            n += 1;
        }

        if let Some(loss) = total_loss {
            let loss_val = loss.to_scalar::<f32>()?;
            self.optimizer.backward_step(&loss)?;
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
        let dev = self.device.clone();
        let ep_t = Tensor::from_slice(episode, (1, episode.len()), &dev)?;

        let mut total_loss: Option<Tensor> = None;
        let mut n = 0usize;

        for (i, input) in regional.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let pred = self.output_heads[i].forward(&ep_t)?.squeeze(0)?;
            let target = Tensor::from_slice(input.as_slice(), (input.len(),), &dev)?;
            let diff = (pred - target)?;
            let loss = diff.sqr()?.mean_all()?;
            total_loss = Some(match total_loss {
                None => loss,
                Some(acc) => acc.add(&loss)?,
            });
            n += 1;
        }

        if let Some(loss) = total_loss {
            let loss_val = loss.to_scalar::<f32>()?;
            self.optimizer.backward_step(&loss)?;
            Ok(loss_val / n as f32)
        } else {
            Ok(0.0)
        }
    }

    // ── Top-down prior helpers (ADR 0010) ───────────────────────────────────────

    /// Set per-region gain multipliers from the ActionDecoder (ADR 0012).
    pub fn set_region_gain(&mut self, gain: [f32; 5]) {
        self.region_gain = gain;
    }

    /// Reset all region gains to 1.0 (no-op). Called at sleep entry.
    pub fn reset_region_gain(&mut self) {
        self.region_gain = [1.0; 5];
    }

    /// Configure working-memory capacity. 0 disables working memory (default).
    pub fn set_wm_size(&mut self, size: usize) {
        self.wm_size = size;
        self.working_memory.truncate(size);
    }

    /// Clear the working-memory ring. Called at sleep entry (wake-local by design).
    pub fn clear_working_memory(&mut self) {
        self.working_memory.clear();
    }

    fn wm_mean_vec(&self) -> Option<Vec<f32>> {
        if self.wm_size == 0 || self.working_memory.is_empty() {
            return None;
        }
        let n = self.working_memory.len() as f32;
        let dim = EPISODE_DIM;
        let mut mean = vec![0.0f32; dim];
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

    /// Set the prior blend weight β. 0.0 disables prior injection (default).
    pub fn set_prior_beta(&mut self, beta: f32) {
        self.prior_beta = beta;
    }

    /// Set the number of settling iterations per episode (ADR 0011). 0 = disabled.
    pub fn set_settle_steps(&mut self, steps: usize) {
        self.settle_steps = steps;
    }

    /// Set the settle blend weight β (fraction of output-head prediction vs. raw input).
    pub fn set_settle_beta(&mut self, beta: f32) {
        self.settle_beta = beta;
    }

    /// Iteratively refine an episode by cycling output-head predictions back through
    /// the k-WTA layers (without updating duty cycles) and re-binding (ADR 0011).
    ///
    /// Returns the refined episode and the number of iterations actually run.
    fn settle_episode(&self, ep: &EpisodeVector) -> PerceptionResult<(EpisodeVector, usize)> {
        let dev = &self.device;
        let mut current = ep.clone();

        for iter in 0..self.settle_steps {
            let cur_t = Tensor::from_slice(&current.0, (1, current.0.len()), dev)?;

            let mut preds: [Vec<f32>; 5] = [vec![], vec![], vec![], vec![], vec![]];
            for (i, pred) in preds.iter_mut().enumerate() {
                *pred = self.output_heads[i]
                    .forward(&cur_t)?
                    .squeeze(0)?
                    .to_vec1()?;
            }

            let mut acts: [Option<SparseActivation>; 5] = [None, None, None, None, None];
            for i in 0..5 {
                if let Some(raw) = &self.latest_inputs[i] {
                    let blended: Vec<f32> = raw
                        .iter()
                        .zip(&preds[i])
                        .map(|(&r, &p)| (1.0 - self.settle_beta) * r + self.settle_beta * p)
                        .collect();
                    acts[i] = Some(self.layer_forward_settling(i, &blended)?);
                }
            }

            // Regions 0–3 required; region 4 (language) optional.
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

            let refined = self.binding.forward(&[a0, a1, a2, a3, a4], None)?;
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
                return Ok((current, iter + 1));
            }
        }

        Ok((current, self.settle_steps))
    }

    fn layer_forward_settling(
        &self,
        idx: usize,
        input: &[f32],
    ) -> PerceptionResult<SparseActivation> {
        match idx {
            0 => self.speech_layer.forward_no_dc(input),
            1 => self.env_layer.forward_no_dc(input),
            2 => self.visual_layer.forward_no_dc(input),
            3 => self.symbolic_layer.forward_no_dc(input),
            4 => self.language_layer.forward_no_dc(input),
            _ => unreachable!(),
        }
    }

    fn blend_with_prior(&self, values: &[f32], idx: usize) -> (Vec<f32>, Option<f32>) {
        if self.prior_beta <= 0.0 {
            return (values.to_vec(), None);
        }
        match &self.prior_predictions[idx] {
            None => (values.to_vec(), None),
            Some(prior) => {
                let cosine = Self::cosine_sim(values, prior);
                let beta = self.prior_beta;
                let blended = values
                    .iter()
                    .zip(prior)
                    .map(|(&r, &p)| (1.0 - beta) * r + beta * p)
                    .collect();
                (blended, Some(cosine))
            }
        }
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

    fn update_prior_predictions(&mut self, episode: &[f32]) -> PerceptionResult<()> {
        let dev = self.device.clone();
        let ep_t = Tensor::from_slice(episode, (1, episode.len()), &dev)?;
        for i in 0..5 {
            let pred_vec: Vec<f32> = self.output_heads[i].forward(&ep_t)?.squeeze(0)?.to_vec1()?;
            self.prior_predictions[i] = Some(pred_vec);
        }
        Ok(())
    }

    /// Current n_units per region: [speech, env, visual, symbolic, language].
    pub fn region_sizes(&self) -> [usize; 5] {
        [
            self.speech_layer.n_units,
            self.env_layer.n_units,
            self.visual_layer.n_units,
            self.symbolic_layer.n_units,
            self.language_layer.n_units,
        ]
    }

    /// Set the optimizer learning rate (used by the sleep neuromodulation schedule).
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Current optimizer learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.optimizer.learning_rate()
    }

    /// Save all weights to a safetensors file.
    pub fn save_weights(&self, path: &std::path::Path) -> PerceptionResult<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    /// Load weights from a safetensors file.
    pub fn load_weights(&mut self, path: &std::path::Path) -> PerceptionResult<()> {
        self.varmap.load(path)?;
        self.rebuild_linears()
    }

    /// Pre-grow regions to `sizes`, load weights, then rebuild all layer Linears.
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
                self.grow_region(region, sizes[i] - current[i])?;
            }
        }
        self.varmap.load(path)?;
        self.rebuild_linears()
    }

    fn rebuild_linears(&mut self) -> PerceptionResult<()> {
        let (
            sw,
            sb,
            dw_s,
            ew,
            eb,
            dw_e,
            vw,
            vb_t,
            dw_v,
            yw,
            yb,
            dw_y,
            lw,
            lb,
            dw_l,
            brp0,
            brp1,
            brp2,
            brp3,
            brp4,
            bwq,
            bwk,
            bwv,
            oh0,
            oh1,
            oh2,
            oh3,
            oh4,
        ) = {
            let data = self.varmap.data().lock().unwrap();
            (
                data["speech_layer.weight"].as_tensor().clone(),
                data.get("speech_layer.bias").map(|v| v.as_tensor().clone()),
                data["speech_decoder.weight"].as_tensor().clone(),
                data["env_layer.weight"].as_tensor().clone(),
                data.get("env_layer.bias").map(|v| v.as_tensor().clone()),
                data["env_decoder.weight"].as_tensor().clone(),
                data["visual_layer.weight"].as_tensor().clone(),
                data.get("visual_layer.bias").map(|v| v.as_tensor().clone()),
                data["visual_decoder.weight"].as_tensor().clone(),
                data["symbolic_layer.weight"].as_tensor().clone(),
                data.get("symbolic_layer.bias")
                    .map(|v| v.as_tensor().clone()),
                data["symbolic_decoder.weight"].as_tensor().clone(),
                data["language_layer.weight"].as_tensor().clone(),
                data.get("language_layer.bias")
                    .map(|v| v.as_tensor().clone()),
                data["language_decoder.weight"].as_tensor().clone(),
                data["binding.region_0.weight"].as_tensor().clone(),
                data["binding.region_1.weight"].as_tensor().clone(),
                data["binding.region_2.weight"].as_tensor().clone(),
                data["binding.region_3.weight"].as_tensor().clone(),
                data["binding.region_4.weight"].as_tensor().clone(),
                data["binding.attn_q.weight"].as_tensor().clone(),
                data["binding.attn_k.weight"].as_tensor().clone(),
                data["binding.attn_v.weight"].as_tensor().clone(),
                data["output_heads.0.weight"].as_tensor().clone(),
                data["output_heads.1.weight"].as_tensor().clone(),
                data["output_heads.2.weight"].as_tensor().clone(),
                data["output_heads.3.weight"].as_tensor().clone(),
                data["output_heads.4.weight"].as_tensor().clone(),
            )
        };

        let sn = self.speech_layer.n_units;
        self.speech_layer.linear = Linear::new(sw, if sn == N_UNITS { sb } else { None });
        self.speech_decoder
            .replace_linear(Linear::new(dw_s, None), sn);

        let en = self.env_layer.n_units;
        self.env_layer.linear = Linear::new(ew, if en == N_UNITS { eb } else { None });
        self.env_decoder.replace_linear(Linear::new(dw_e, None), en);

        let vn = self.visual_layer.n_units;
        self.visual_layer.linear = Linear::new(vw, if vn == N_UNITS { vb_t } else { None });
        self.visual_decoder
            .replace_linear(Linear::new(dw_v, None), vn);

        let yn = self.symbolic_layer.n_units;
        self.symbolic_layer.linear = Linear::new(yw, if yn == N_UNITS { yb } else { None });
        self.symbolic_decoder
            .replace_linear(Linear::new(dw_y, None), yn);

        let ln = self.language_layer.n_units;
        self.language_layer.linear = Linear::new(lw, if ln == N_UNITS { lb } else { None });
        self.language_decoder
            .replace_linear(Linear::new(dw_l, None), ln);

        self.binding.region_projs[0] = Linear::new(brp0, None);
        self.binding.region_projs[1] = Linear::new(brp1, None);
        self.binding.region_projs[2] = Linear::new(brp2, None);
        self.binding.region_projs[3] = Linear::new(brp3, None);
        self.binding.region_projs[4] = Linear::new(brp4, None);
        self.binding.w_q = Linear::new(bwq, None);
        self.binding.w_k = Linear::new(bwk, None);
        self.binding.w_v = Linear::new(bwv, None);

        self.output_heads[0] = Linear::new(oh0, None);
        self.output_heads[1] = Linear::new(oh1, None);
        self.output_heads[2] = Linear::new(oh2, None);
        self.output_heads[3] = Linear::new(oh3, None);
        self.output_heads[4] = Linear::new(oh4, None);
        Ok(())
    }

    fn layer_keys(region: RegionId) -> (&'static str, &'static str) {
        match region {
            RegionId::AudioSpeech => ("speech_layer.weight", "speech_decoder.weight"),
            RegionId::AudioEnv => ("env_layer.weight", "env_decoder.weight"),
            RegionId::Visual => ("visual_layer.weight", "visual_decoder.weight"),
            RegionId::Symbolic => ("symbolic_layer.weight", "symbolic_decoder.weight"),
            RegionId::Language => ("language_layer.weight", "language_decoder.weight"),
        }
    }

    fn layer_mut(&mut self, region: RegionId) -> (&mut KWtaLayer, &mut SaeDecoder) {
        match region {
            RegionId::AudioSpeech => (&mut self.speech_layer, &mut self.speech_decoder),
            RegionId::AudioEnv => (&mut self.env_layer, &mut self.env_decoder),
            RegionId::Visual => (&mut self.visual_layer, &mut self.visual_decoder),
            RegionId::Symbolic => (&mut self.symbolic_layer, &mut self.symbolic_decoder),
            RegionId::Language => (&mut self.language_layer, &mut self.language_decoder),
        }
    }

    /// Blend two sets of regional encoder inputs and run through k-WTA + binding
    /// to produce a synthetic episode vector. No DC update — inference only.
    ///
    /// Skips any region whose input in either set is empty. Returns
    /// `PerceptionError::BindingIncomplete` when any of the four non-language
    /// regions (0–3) is missing from at least one source.
    pub fn synthesize_episode(
        &self,
        regional_a: &[Vec<f32>; 5],
        regional_b: &[Vec<f32>; 5],
        alpha: f32,
    ) -> PerceptionResult<Vec<f32>> {
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
            acts[i] = Some(self.layer_forward_settling(i, &blended)?);
        }
        let (a0, a1, a2, a3) = match (&acts[0], &acts[1], &acts[2], &acts[3]) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return Err(PerceptionError::BindingIncomplete),
        };
        let a4 = acts[4].as_ref().unwrap_or(&lang_zero);
        let episode = self.binding.forward(&[a0, a1, a2, a3, a4], None)?;
        Ok(episode.0)
    }

    /// Measure mean output-head MSE for a single episode without any backward
    /// step. Skips regions whose input is empty.
    pub fn measure_output_mse_inference(
        &self,
        episode: &[f32],
        regional: &[Vec<f32>; 5],
    ) -> PerceptionResult<f32> {
        let dev = self.device.clone();
        let ep_t = Tensor::from_slice(episode, (1, episode.len()), &dev)?;
        let mut total = 0.0f32;
        let mut n = 0usize;
        for (i, input) in regional.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let pred = self.output_heads[i].forward(&ep_t)?.squeeze(0)?;
            let target = Tensor::from_slice(input.as_slice(), (input.len(),), &dev)?;
            let mse = (pred - target)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
            total += mse;
            n += 1;
        }
        Ok(if n > 0 { total / n as f32 } else { 0.0 })
    }
}

/// Return the minimum of two `Option<f32>` cosine values.
/// Both None → None (no prior yet, settle permissively).
/// One Some → that value. Both Some → minimum (most-novel signal gates settling).
fn min_cosine_2(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some(x), Some(y)) => Some(x.min(y)),
    }
}

/// Multiply every element of `values` by `gain`. Returns a new Vec.
fn apply_gain(values: &[f32], gain: f32) -> Vec<f32> {
    if (gain - 1.0).abs() < 1e-6 {
        return values.to_vec();
    }
    values.iter().map(|&x| x * gain).collect()
}
