use std::path::Path;

use somnus_sensory::Timestamp;
use ndarray::{Array3, Array4};
use ort::{session::Session, value::TensorRef};

use crate::{
    preprocess::mel::{ClapMel, WhisperMel},
    session_with_migraphx, EncodingError, EncodingResult,
};

pub struct AudioEmbedding {
    pub source: AudioEncoderSource,
    pub values: Vec<f32>, // 384 (Speech) or 512 (Env)
    pub timestamp: Timestamp,
    /// Wall time for this encode call (mel computation + ORT inference), in milliseconds.
    pub encode_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioEncoderSource {
    /// Whisper-tiny encoder, 384-dim, speech content.
    Speech,
    /// CLAP-HTSAT-unfused encoder, 512-dim, environmental sound.
    Env,
}

pub struct AudioEncoder {
    speech_session: Session,
    env_session: Session,
    speech_mel: WhisperMel,
    env_mel: ClapMel,
}

const WHISPER_SAMPLES: usize = 480_000; // 30 s @ 16 kHz
const CLAP_SAMPLES: usize = 480_000; // 10 s @ 48 kHz

impl AudioEncoder {
    pub fn new(models_dir: &Path) -> EncodingResult<Self> {
        let speech_session = session_with_migraphx(&models_dir.join("whisper_tiny_encoder.onnx"))?;
        let env_session = session_with_migraphx(&models_dir.join("clap_audio_encoder.onnx"))?;
        Ok(Self {
            speech_session,
            env_session,
            speech_mel: WhisperMel::new(),
            env_mel: ClapMel::new(),
        })
    }

    /// Encode 480 000 f32 samples @ 16 kHz → 384-dim speech embedding.
    ///
    /// Also returns the raw encoder hidden state (1500 × 384, row-major) so the
    /// caller can optionally run Whisper greedy decoding for ASR (ADR 0013).
    pub fn encode_speech(
        &mut self,
        pcm_16k: &[f32],
        ts: Timestamp,
    ) -> EncodingResult<(AudioEmbedding, Vec<f32>)> {
        let t0 = std::time::Instant::now();
        debug_assert_eq!(pcm_16k.len(), WHISPER_SAMPLES);
        if pcm_16k.len() != WHISPER_SAMPLES {
            return Err(EncodingError::WrongLength {
                expected: WHISPER_SAMPLES,
                got: pcm_16k.len(),
            });
        }

        let mel_flat = self.speech_mel.compute(pcm_16k);
        // Shape: (1, 80, 3000)
        let mel_arr = Array3::from_shape_vec((1, 80, 3000), mel_flat)
            .map_err(|e| EncodingError::Image(e.to_string()))?;
        let tensor = TensorRef::from_array_view(&mel_arr)?;

        let outputs = self
            .speech_session
            .run(ort::inputs!["input_features" => tensor])?;
        let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims != [1, 1500, 384] {
            return Err(EncodingError::ShapeMismatch {
                expected: vec![1, 1500, 384],
                got: dims,
            });
        }

        let hidden_flat = data.to_vec(); // [1500 * 384] for optional ASR
        let values = mean_pool_2d(&hidden_flat, 1500, 384);
        Ok((
            AudioEmbedding {
                source: AudioEncoderSource::Speech,
                values,
                timestamp: ts,
                encode_ms: t0.elapsed().as_millis() as u64,
            },
            hidden_flat,
        ))
    }

    /// Encode 480 000 f32 samples @ 48 kHz → 512-dim env embedding.
    pub fn encode_env(&mut self, pcm_48k: &[f32], ts: Timestamp) -> EncodingResult<AudioEmbedding> {
        let t0 = std::time::Instant::now();
        debug_assert_eq!(pcm_48k.len(), CLAP_SAMPLES);
        if pcm_48k.len() != CLAP_SAMPLES {
            return Err(EncodingError::WrongLength {
                expected: CLAP_SAMPLES,
                got: pcm_48k.len(),
            });
        }

        let mel_flat = self.env_mel.compute(pcm_48k);
        // Shape: (1, 1, 1001, 64)
        let mel_arr = Array4::from_shape_vec((1, 1, 1001, 64), mel_flat)
            .map_err(|e| EncodingError::Image(e.to_string()))?;
        let tensor = TensorRef::from_array_view(&mel_arr)?;

        let outputs = self
            .env_session
            .run(ort::inputs!["input_features" => tensor])?;
        let (shape, data) = outputs["audio_embeds"].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims != [1, 512] {
            return Err(EncodingError::ShapeMismatch {
                expected: vec![1, 512],
                got: dims,
            });
        }

        let values = data.to_vec();
        Ok(AudioEmbedding {
            source: AudioEncoderSource::Env,
            values,
            timestamp: ts,
            encode_ms: t0.elapsed().as_millis() as u64,
        })
    }
}

/// Mean-pool a `(seq_len, dim)` flat f32 slice along the sequence axis.
pub(crate) fn mean_pool_2d(tensor: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    for t in 0..seq_len {
        for d in 0..dim {
            out[d] += tensor[t * dim + d];
        }
    }
    let n = seq_len as f32;
    out.iter_mut().for_each(|v| *v /= n);
    out
}
