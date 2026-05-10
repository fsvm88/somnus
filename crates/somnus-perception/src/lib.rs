pub mod binding;
pub mod error;
pub mod region;
pub mod sae;

pub use binding::{BindingArea, EpisodeVector, EPISODE_DIM};
pub use error::{PerceptionError, PerceptionResult};
pub use region::{KWtaLayer, RegionId, SparseActivation, K_ACTIVE, N_UNITS};
pub use sae::SaeDecoder;

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};

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
        let speech_act = self.speech_layer.forward(values)?;
        let symbolic_act = self.symbolic_layer.forward(values)?;

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

        let (episode, regional_inputs) = self.try_bind()?;
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
        let act = self.env_layer.forward(values)?;
        let err = self.env_decoder.reconstruction_error(values, &act)?;
        self.latest[RegionId::AudioEnv.as_index()] = Some(act);
        self.latest_inputs[RegionId::AudioEnv.as_index()] = Some(values.to_vec());

        let (episode, regional_inputs) = self.try_bind()?;
        Ok(PerceptionOutput {
            recon_errors: vec![(RegionId::AudioEnv, err)],
            episode,
            regional_inputs,
        })
    }

    /// Ingest a 384-dim DINOv2 visual embedding.
    pub fn ingest_visual(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let act = self.visual_layer.forward(values)?;
        let err = self.visual_decoder.reconstruction_error(values, &act)?;
        self.latest[RegionId::Visual.as_index()] = Some(act);
        self.latest_inputs[RegionId::Visual.as_index()] = Some(values.to_vec());

        let (episode, regional_inputs) = self.try_bind()?;
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

    /// Save all weights to a safetensors file.
    pub fn save_weights(&self, path: &std::path::Path) -> PerceptionResult<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    /// Load weights from a safetensors file (shapes must match).
    pub fn load_weights(&mut self, path: &std::path::Path) -> PerceptionResult<()> {
        self.varmap.load(path)?;
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
