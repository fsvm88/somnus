//! `somnus-encoding` — Phase 2 encoding pipeline.
//!
//! Wraps frozen ONNX encoders (Whisper-tiny, CLAP-HTSAT-unfused, DINOv2-S/14)
//! and mel-spectrogram preprocessing. All APIs are synchronous; async bridging
//! is handled by the `somnus` binary via `spawn_blocking`.

pub mod audio;
pub mod preprocess;
pub mod video;

/// Register MIGraphX EP on a session builder using the generic key-value ORT API.
///
/// The struct-based `SessionOptionsAppendExecutionProvider_MIGraphX` triggers MIGraphX's
/// internal problem-cache auto-save with an empty path, causing GPU kernel failure.
/// The generic API (identical to Python's `providers=["MIGraphXExecutionProvider"]`)
/// bypasses this bug.
pub(crate) fn session_with_migraphx(
    path: &std::path::Path,
) -> EncodingResult<ort::session::Session> {
    use ort::AsPointer;
    use std::ffi::CString;
    let mut builder = ort::session::Session::builder()?;
    let name = CString::new("MIGraphXExecutionProvider").unwrap();
    let status = unsafe {
        (ort::api().SessionOptionsAppendExecutionProvider)(
            builder.ptr_mut(),
            name.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };
    if !status.0.is_null() {
        unsafe { (ort::api().ReleaseStatus)(status.0) };
    }
    Ok(builder.commit_from_file(path)?)
}

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
