use std::path::Path;

use ndarray::{Array3, Array4};
use ort::{session::Session, value::TensorRef};
use somnus_sensory::Timestamp;

#[cfg(not(feature = "mn10_env"))]
use crate::preprocess::mel::ClapMel;
#[cfg(feature = "mn10_env")]
use crate::preprocess::mel::Mn10Mel;
use crate::{preprocess::mel::WhisperMel, session_with_migraphx, EncodingError, EncodingResult};

pub struct AudioEmbedding {
    pub source: AudioEncoderSource,
    pub values: Vec<f32>, // 384 (Speech) or ENV_DIM (Env: 512 CLAP / 960 mn10)
    pub timestamp: Timestamp,
    /// Wall time for this encode call (mel computation + ORT inference), in milliseconds.
    pub encode_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioEncoderSource {
    /// Whisper-tiny encoder, 384-dim, speech content.
    Speech,
    /// Environmental sound encoder — CLAP-HTSAT-unfused (512-dim, default) or EfficientAT
    /// mn10 (960-dim, `--features mn10_env`, gaps.md item 9).
    Env,
}

#[cfg(not(feature = "mn10_env"))]
type EnvMel = ClapMel;
#[cfg(feature = "mn10_env")]
type EnvMel = Mn10Mel;

pub struct AudioEncoder {
    speech_session: Session,
    env_session: Session,
    speech_mel: WhisperMel,
    env_mel: EnvMel,
}

const WHISPER_SAMPLES: usize = 480_000; // 30 s @ 16 kHz
                                        // 10 s of env audio: 480 000 @ 48 kHz (CLAP) or 320 000 @ 32 kHz (mn10).
#[cfg(not(feature = "mn10_env"))]
const ENV_SAMPLES: usize = 480_000;
#[cfg(feature = "mn10_env")]
const ENV_SAMPLES: usize = 320_000;
#[cfg(not(feature = "mn10_env"))]
const ENV_ONNX_FILE: &str = "clap_audio_encoder_fp16.onnx";
#[cfg(feature = "mn10_env")]
const ENV_ONNX_FILE: &str = "efficientat_mn10_encoder_fp16.onnx";
#[cfg(not(feature = "mn10_env"))]
const ENV_DIM: i64 = 512;
#[cfg(feature = "mn10_env")]
const ENV_DIM: i64 = 960;
#[cfg(not(feature = "mn10_env"))]
const ENV_OUTPUT_NAME: &str = "audio_embeds";
#[cfg(feature = "mn10_env")]
const ENV_OUTPUT_NAME: &str = "env_embedding";

impl AudioEncoder {
    pub fn new(models_dir: &Path) -> EncodingResult<Self> {
        let speech_session = session_with_migraphx(&models_dir.join("whisper_tiny_encoder.onnx"))?;
        let env_session = session_with_migraphx(&models_dir.join(ENV_ONNX_FILE))?;
        Ok(Self {
            speech_session,
            env_session,
            speech_mel: WhisperMel::new(),
            env_mel: EnvMel::new(),
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
        let ((values, hidden_flat), d0) =
            somnus_metrics::timed_core!(somnus_metrics::Metric::EncodeSpeech, {
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
                (values, hidden_flat)
            });
        Ok((
            AudioEmbedding {
                source: AudioEncoderSource::Speech,
                values,
                timestamp: ts,
                encode_ms: d0.as_millis() as u64,
            },
            hidden_flat,
        ))
    }

    /// Encode `ENV_SAMPLES` f32 samples → `ENV_DIM`-dim env embedding (CLAP: 480 000 @
    /// 48 kHz → 512-dim; mn10: 320 000 @ 32 kHz → 960-dim).
    pub fn encode_env(&mut self, pcm_env: &[f32], ts: Timestamp) -> EncodingResult<AudioEmbedding> {
        let (values, d0) = somnus_metrics::timed_core!(somnus_metrics::Metric::EncodeEnv, {
            debug_assert_eq!(pcm_env.len(), ENV_SAMPLES);
            if pcm_env.len() != ENV_SAMPLES {
                return Err(EncodingError::WrongLength {
                    expected: ENV_SAMPLES,
                    got: pcm_env.len(),
                });
            }

            let mel_flat = self.env_mel.compute(pcm_env);
            #[cfg(not(feature = "mn10_env"))]
            let mel_shape = (1, 1, 1001, 64);
            #[cfg(feature = "mn10_env")]
            let mel_shape = (1, 1, 128, 1000);
            let mel_arr = Array4::from_shape_vec(mel_shape, mel_flat)
                .map_err(|e| EncodingError::Image(e.to_string()))?;
            let tensor = TensorRef::from_array_view(&mel_arr)?;

            #[cfg(not(feature = "mn10_env"))]
            let input_name = "input_features";
            #[cfg(feature = "mn10_env")]
            let input_name = "mel_spectrogram";
            let outputs = self.env_session.run(ort::inputs![input_name => tensor])?;
            let (shape, data) = outputs[ENV_OUTPUT_NAME].try_extract_tensor::<f32>()?;
            let dims: Vec<i64> = shape.iter().copied().collect();
            if dims != [1, ENV_DIM] {
                return Err(EncodingError::ShapeMismatch {
                    expected: vec![1, ENV_DIM],
                    got: dims,
                });
            }

            data.to_vec()
        });
        Ok(AudioEmbedding {
            source: AudioEncoderSource::Env,
            values,
            timestamp: ts,
            encode_ms: d0.as_millis() as u64,
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

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_2d_exact_small_case() {
        // seq_len=2, dim=3: column-wise means of [[1,2,3],[4,5,6]] = [2.5, 3.5, 4.5].
        let flat = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(mean_pool_2d(&flat, 2, 3), [2.5, 3.5, 4.5]);
    }

    #[test]
    fn mean_pool_2d_single_row_is_identity() {
        let flat = [0.5f32, -1.0, 2.0];
        assert_eq!(mean_pool_2d(&flat, 1, 3), flat.to_vec());
    }

    #[test]
    fn mean_pool_2d_constant_rows_yield_constant() {
        // 1500×384 is the real Whisper hidden-state shape; constant rows must
        // pool to exactly that constant.
        let flat = vec![0.75f32; 1500 * 384];
        let out = mean_pool_2d(&flat, 1500, 384);
        assert_eq!(out.len(), 384);
        assert!(out.iter().all(|&v| v == 0.75));
    }
}
