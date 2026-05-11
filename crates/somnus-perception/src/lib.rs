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
pub const REGION_DIMS: [usize; 4] = [384, 512, 384, 384];

/// Reconstruction errors for each region updated in a single `ingest_*` call.
pub struct PerceptionOutput {
    /// Reconstruction errors for regions updated this call (region, MSE).
    pub recon_errors: Vec<(RegionId, f32)>,
    /// Episode vector from the binding area, produced once all four regions
    /// have at least one activation.
    pub episode: Option<EpisodeVector>,
    /// Raw encoder inputs per region that produced this episode (Some when episode is Some).
    /// Order: [AudioSpeech(384), AudioEnv(512), Visual(384), Symbolic(384)].
    pub regional_inputs: Option<[Vec<f32>; 4]>,
}

/// Owns the full cognitive-layer forward pass: four k-WTA layers, four SAE
/// decoders, and the binding area.
pub struct PerceptionProcessor {
    speech_layer: KWtaLayer,
    env_layer: KWtaLayer,
    visual_layer: KWtaLayer,
    symbolic_layer: KWtaLayer,

    speech_decoder: SaeDecoder,
    env_decoder: SaeDecoder,
    visual_decoder: SaeDecoder,
    symbolic_decoder: SaeDecoder,

    binding: BindingArea,

    /// Per-region output heads: EPISODE_DIM → encoder_dim. Predict encoder
    /// embeddings from the episode vector (ADR 0009). Also serve as the
    /// top-down feedback path (ADR 0010).
    output_heads: [Linear; 4],

    /// Predicted encoder embeddings from the previous episode (ADR 0010).
    /// None until the first episode is produced. Updated after every episode.
    prior_predictions: [Option<Vec<f32>>; 4],
    /// Blend weight β for top-down prior injection. 0.0 = disabled (default).
    prior_beta: f32,

    /// Latest sparse activation per region; None until first embedding arrives.
    latest: [Option<SparseActivation>; 4],
    /// Latest raw encoder input per region; None until first embedding arrives.
    latest_inputs: [Option<Vec<f32>>; 4],

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

        let binding = BindingArea::new(vb("binding"), device.clone())?;

        let vb_out = vb("output_heads");
        let output_heads = [
            linear_no_bias(EPISODE_DIM, REGION_DIMS[0], vb_out.pp("0"))?, // speech: 512→384
            linear_no_bias(EPISODE_DIM, REGION_DIMS[1], vb_out.pp("1"))?, // env: 512→512
            linear_no_bias(EPISODE_DIM, REGION_DIMS[2], vb_out.pp("2"))?, // visual: 512→384
            linear_no_bias(EPISODE_DIM, REGION_DIMS[3], vb_out.pp("3"))?, // symbolic: 512→384
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
            speech_decoder,
            env_decoder,
            visual_decoder,
            symbolic_decoder,
            binding,
            output_heads,
            prior_predictions: [None, None, None, None],
            prior_beta: 0.0,
            latest: [None, None, None, None],
            latest_inputs: [None, None, None, None],
            varmap,
            optimizer,
            device,
        })
    }

    /// Ingest a 384-dim Whisper speech embedding.
    /// Updates AudioSpeech and Symbolic regions (same input, independent weights).
    pub fn ingest_speech(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let (speech_input, speech_cos) = self.blend_with_prior(values, 0);
        let (symbolic_input, symbolic_cos) = self.blend_with_prior(values, 3);
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
        // Store raw encoder inputs; training targets must be unbiased (ADR 0010).
        self.latest_inputs[RegionId::AudioSpeech.as_index()] = Some(values.to_vec());
        self.latest_inputs[RegionId::Symbolic.as_index()] = Some(values.to_vec());

        let (episode, regional_inputs) = self.try_bind()?;
        if let Some(ref ep) = episode {
            self.update_prior_predictions(&ep.0)?;
        }
        Ok(PerceptionOutput {
            recon_errors: vec![
                (RegionId::AudioSpeech, speech_err),
                (RegionId::Symbolic, symbolic_err),
            ],
            episode,
            regional_inputs,
        })
    }

    /// Ingest a 512-dim CLAP env embedding.
    pub fn ingest_env(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let (input, cosine) = self.blend_with_prior(values, 1);
        if let Some(c) = cosine {
            tracing::info!(region = "env", prior_cosine = c, "top-down prior");
        }
        let act = self.env_layer.forward(&input)?;
        let err = self.env_decoder.reconstruction_error(values, &act)?;
        self.latest[RegionId::AudioEnv.as_index()] = Some(act);
        self.latest_inputs[RegionId::AudioEnv.as_index()] = Some(values.to_vec());

        let (episode, regional_inputs) = self.try_bind()?;
        if let Some(ref ep) = episode {
            self.update_prior_predictions(&ep.0)?;
        }
        Ok(PerceptionOutput {
            recon_errors: vec![(RegionId::AudioEnv, err)],
            episode,
            regional_inputs,
        })
    }

    /// Ingest a 384-dim DINOv2 visual embedding.
    pub fn ingest_visual(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let (input, cosine) = self.blend_with_prior(values, 2);
        if let Some(c) = cosine {
            tracing::info!(region = "visual", prior_cosine = c, "top-down prior");
        }
        let act = self.visual_layer.forward(&input)?;
        let err = self.visual_decoder.reconstruction_error(values, &act)?;
        self.latest[RegionId::Visual.as_index()] = Some(act);
        self.latest_inputs[RegionId::Visual.as_index()] = Some(values.to_vec());

        let (episode, regional_inputs) = self.try_bind()?;
        if let Some(ref ep) = episode {
            self.update_prior_predictions(&ep.0)?;
        }
        Ok(PerceptionOutput {
            recon_errors: vec![(RegionId::Visual, err)],
            episode,
            regional_inputs,
        })
    }

    #[allow(clippy::type_complexity)]
    fn try_bind(&self) -> PerceptionResult<(Option<EpisodeVector>, Option<[Vec<f32>; 4]>)> {
        match (
            self.latest[0].as_ref(),
            self.latest[1].as_ref(),
            self.latest[2].as_ref(),
            self.latest[3].as_ref(),
        ) {
            (Some(sp), Some(en), Some(vi), Some(sy)) => {
                let ep = self.binding.forward(&[sp, en, vi, sy])?;
                let regional = [
                    self.latest_inputs[0].clone().unwrap_or_default(),
                    self.latest_inputs[1].clone().unwrap_or_default(),
                    self.latest_inputs[2].clone().unwrap_or_default(),
                    self.latest_inputs[3].clone().unwrap_or_default(),
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
    pub fn replay_step(&mut self, regional: &[Vec<f32>; 4]) -> PerceptionResult<f32> {
        let dev = self.device.clone();
        let mut total_loss: Option<Tensor> = None;
        let mut n_regions = 0usize;

        // Each region processed separately to avoid mixed mutable/immutable borrows.
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
            _ => unreachable!(),
        }
    }

    /// Measure SAE reconstruction MSE per region without updating weights.
    /// Returns per-region MSE: [speech, env, visual, symbolic].
    pub fn measure_mse(&mut self, regional: &[Vec<f32>; 4]) -> PerceptionResult<[f32; 4]> {
        let mut result = [0.0f32; 4];
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
            _ => unreachable!(),
        }
    }

    /// Apply global weight decay to all k-WTA and SAE Linear weights.
    /// Each weight element is multiplied by `(1 - rate)`.
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
    ///
    /// Updates the k-WTA layer and SAE decoder weights in the VarMap,
    /// rebuilds their Linear layers, extends duty_cycle, and resets the
    /// optimizer with the updated var list.
    pub fn grow_region(&mut self, region: RegionId, n_new: usize) -> PerceptionResult<()> {
        let idx = region.as_index();
        let (enc_key, dec_key) = Self::layer_keys(region);
        let n_in = REGION_DIMS[idx];

        let (layer, _decoder) = self.layer_mut(region);
        let old_n = layer.n_units;
        let new_n = old_n + n_new;

        // Expand encoder weight: [old_n, n_in] → [new_n, n_in]
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

        // Expand SAE decoder weight: [n_in, old_n] → [n_in, new_n]
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

        // Reset optimizer with updated var list.
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

    /// Run one wake-time reconstruction step on a just-produced episode.
    ///
    /// For each non-empty region, runs the output head forward on `episode` and
    /// computes MSE against the actual encoder input. Backprops through the heads
    /// and the binding area. Returns mean MSE over active regions.
    pub fn wake_reconstruct_step(
        &mut self,
        episode: &[f32],
        regional: &[Vec<f32>; 4],
    ) -> PerceptionResult<f32> {
        let dev = self.device.clone();
        let ep_t = Tensor::from_slice(episode, (1, episode.len()), &dev)?;

        let mut total_loss: Option<Tensor> = None;
        let mut n = 0usize;

        for (i, input) in regional.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let pred = self.output_heads[i].forward(&ep_t)?.squeeze(0)?; // [dim]
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

    /// Set the prior blend weight β. 0.0 disables prior injection (default).
    pub fn set_prior_beta(&mut self, beta: f32) {
        self.prior_beta = beta;
    }

    /// Blend `values` with the stored prior prediction for `idx` if β > 0 and
    /// a prior exists. Returns the (possibly biased) input and the cosine
    /// similarity between raw and prior (Some only when a prior was applied).
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

    /// Run the four output heads on `episode` in inference mode and store the
    /// predictions as the priors for the next ingest call.
    fn update_prior_predictions(&mut self, episode: &[f32]) -> PerceptionResult<()> {
        let dev = self.device.clone();
        let ep_t = Tensor::from_slice(episode, (1, episode.len()), &dev)?;
        for i in 0..4 {
            let pred_vec: Vec<f32> = self.output_heads[i].forward(&ep_t)?.squeeze(0)?.to_vec1()?;
            self.prior_predictions[i] = Some(pred_vec);
        }
        Ok(())
    }

    /// Current n_units per region: [speech, env, visual, symbolic].
    pub fn region_sizes(&self) -> [usize; 4] {
        [
            self.speech_layer.n_units,
            self.env_layer.n_units,
            self.visual_layer.n_units,
            self.symbolic_layer.n_units,
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
    ///
    /// `Var::set` does not update Tensor clones already held by layers, so
    /// `rebuild_linears` is called after load to propagate the new values.
    pub fn load_weights(&mut self, path: &std::path::Path) -> PerceptionResult<()> {
        self.varmap.load(path)?;
        self.rebuild_linears()
    }

    /// Pre-grow regions to `sizes`, load weights, then rebuild all layer Linears.
    ///
    /// Call this on restore when region sizes from a prior session differ from
    /// the default `N_UNITS`; `sizes` comes from `load_region_sizes`.
    pub fn load_weights_with_sizes(
        &mut self,
        path: &std::path::Path,
        sizes: [usize; 4],
    ) -> PerceptionResult<()> {
        let regions = [
            RegionId::AudioSpeech,
            RegionId::AudioEnv,
            RegionId::Visual,
            RegionId::Symbolic,
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

    /// Rebuild all layer Linear objects from the current VarMap contents.
    ///
    /// Required because `Var::set` (called inside `varmap.load`) writes new data
    /// into the Var but does not update Tensor handles cloned before the set.
    /// Grown layers (n_units > N_UNITS) have no bias; unmodified layers keep theirs.
    fn rebuild_linears(&mut self) -> PerceptionResult<()> {
        // Collect tensors with the lock held, then drop it before mutating layers.
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
            brp0,
            brp1,
            brp2,
            brp3,
            bwq,
            bwk,
            bwv,
            oh0,
            oh1,
            oh2,
            oh3,
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
                data["binding.region_0.weight"].as_tensor().clone(),
                data["binding.region_1.weight"].as_tensor().clone(),
                data["binding.region_2.weight"].as_tensor().clone(),
                data["binding.region_3.weight"].as_tensor().clone(),
                data["binding.attn_q.weight"].as_tensor().clone(),
                data["binding.attn_k.weight"].as_tensor().clone(),
                data["binding.attn_v.weight"].as_tensor().clone(),
                data["output_heads.0.weight"].as_tensor().clone(),
                data["output_heads.1.weight"].as_tensor().clone(),
                data["output_heads.2.weight"].as_tensor().clone(),
                data["output_heads.3.weight"].as_tensor().clone(),
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

        self.binding.region_projs[0] = Linear::new(brp0, None);
        self.binding.region_projs[1] = Linear::new(brp1, None);
        self.binding.region_projs[2] = Linear::new(brp2, None);
        self.binding.region_projs[3] = Linear::new(brp3, None);
        self.binding.w_q = Linear::new(bwq, None);
        self.binding.w_k = Linear::new(bwk, None);
        self.binding.w_v = Linear::new(bwv, None);

        self.output_heads[0] = Linear::new(oh0, None);
        self.output_heads[1] = Linear::new(oh1, None);
        self.output_heads[2] = Linear::new(oh2, None);
        self.output_heads[3] = Linear::new(oh3, None);
        Ok(())
    }

    fn layer_keys(region: RegionId) -> (&'static str, &'static str) {
        match region {
            RegionId::AudioSpeech => ("speech_layer.weight", "speech_decoder.weight"),
            RegionId::AudioEnv => ("env_layer.weight", "env_decoder.weight"),
            RegionId::Visual => ("visual_layer.weight", "visual_decoder.weight"),
            RegionId::Symbolic => ("symbolic_layer.weight", "symbolic_decoder.weight"),
        }
    }

    fn layer_mut(&mut self, region: RegionId) -> (&mut KWtaLayer, &mut SaeDecoder) {
        match region {
            RegionId::AudioSpeech => (&mut self.speech_layer, &mut self.speech_decoder),
            RegionId::AudioEnv => (&mut self.env_layer, &mut self.env_decoder),
            RegionId::Visual => (&mut self.visual_layer, &mut self.visual_decoder),
            RegionId::Symbolic => (&mut self.symbolic_layer, &mut self.symbolic_decoder),
        }
    }
}
