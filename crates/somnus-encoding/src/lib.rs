//! `somnus-encoding` — Phase 2 encoding pipeline.
//!
//! Wraps frozen ONNX encoders (Whisper-tiny, CLAP-HTSAT-unfused, DINOv2-S/14)
//! and mel-spectrogram preprocessing. All APIs are synchronous; async bridging
//! is handled by the `somnus` binary via `spawn_blocking`.

pub mod preprocess;
pub mod audio;
pub mod video;

pub use audio::{AudioEmbedding, AudioEncoder, AudioEncoderSource};
pub use video::{VideoEmbedding, VideoEncoder};

use somnus_sensory::PixelFormat;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EncodingError {
    #[error("OnnxRuntime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("expected {expected} samples, got {got}")]
    WrongLength { expected: usize, got: usize },

    #[error("unexpected output shape {got:?}, expected {expected:?}")]
    ShapeMismatch { expected: Vec<i64>, got: Vec<i64> },

    #[error("unsupported pixel format: {0:?}")]
    UnsupportedPixelFormat(PixelFormat),

    #[error("image processing error: {0}")]
    Image(String),
}

pub type EncodingResult<T> = Result<T, EncodingError>;
