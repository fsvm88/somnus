pub mod binding;
pub mod error;
pub mod region;
pub mod sae;

pub use binding::{BindingArea, EpisodeVector, EPISODE_DIM};
pub use error::{PerceptionError, PerceptionResult};
pub use region::{KWtaLayer, RegionId, SparseActivation, K_ACTIVE, N_UNITS};
pub use sae::SaeDecoder;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

/// Reconstruction errors for each region updated in a single `ingest_*` call.
pub struct PerceptionOutput {
    /// Reconstruction errors for regions updated this call (region, MSE).
    pub recon_errors: Vec<(RegionId, f32)>,
    /// Episode vector from the binding area, produced once all four regions
    /// have at least one activation.
    pub episode: Option<EpisodeVector>,
}

/// Owns the full cognitive-layer forward pass: four k-WTA layers, four SAE
/// decoders, and the binding area.
///
/// Call `ingest_speech`, `ingest_env`, or `ingest_visual` whenever an
/// embedding arrives; the processor maintains the latest sparse activation per
/// region and produces an episode vector as soon as all four are populated.
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

    // Latest sparse activation per region; None until first embedding arrives.
    latest: [Option<SparseActivation>; 4],
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

        let binding = BindingArea::new(vb("binding"), device)?;

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
        })
    }

    /// Ingest a 384-dim Whisper speech embedding.
    /// Updates both AudioSpeech and Symbolic regions (same input, independent weights).
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

        let episode = self.try_bind()?;
        Ok(PerceptionOutput {
            recon_errors: vec![
                (RegionId::AudioSpeech, speech_err),
                (RegionId::Symbolic, symbolic_err),
            ],
            episode,
        })
    }

    /// Ingest a 512-dim CLAP env embedding.
    pub fn ingest_env(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let act = self.env_layer.forward(values)?;
        let err = self.env_decoder.reconstruction_error(values, &act)?;
        self.latest[RegionId::AudioEnv.as_index()] = Some(act);

        let episode = self.try_bind()?;
        Ok(PerceptionOutput {
            recon_errors: vec![(RegionId::AudioEnv, err)],
            episode,
        })
    }

    /// Ingest a 384-dim DINOv2 visual embedding.
    pub fn ingest_visual(&mut self, values: &[f32]) -> PerceptionResult<PerceptionOutput> {
        let act = self.visual_layer.forward(values)?;
        let err = self.visual_decoder.reconstruction_error(values, &act)?;
        self.latest[RegionId::Visual.as_index()] = Some(act);

        let episode = self.try_bind()?;
        Ok(PerceptionOutput {
            recon_errors: vec![(RegionId::Visual, err)],
            episode,
        })
    }

    fn try_bind(&self) -> PerceptionResult<Option<EpisodeVector>> {
        match (
            self.latest[0].as_ref(),
            self.latest[1].as_ref(),
            self.latest[2].as_ref(),
            self.latest[3].as_ref(),
        ) {
            (Some(sp), Some(en), Some(vi), Some(sy)) => {
                Ok(Some(self.binding.forward(&[sp, en, vi, sy])?))
            }
            _ => Ok(None),
        }
    }
}
