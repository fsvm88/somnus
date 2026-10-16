//! `somnus-encoding` — Phase 2 encoding pipeline.
//!
//! Wraps frozen ONNX encoders (Whisper-tiny, CLAP-HTSAT-unfused, DINOv2-S/14)
//! and mel-spectrogram preprocessing. All APIs are synchronous; async bridging
//! is handled by the `somnus` binary via `spawn_blocking`.

pub mod audio;
pub mod preprocess;
pub mod video;

/// Intra-op thread count for the encoder ORT sessions (0 = ORT's default, i.e. every core).
static ORT_INTRA_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Cap the intra-op threads of the encoder sessions created after this call (0 = ORT default).
///
/// The encoders run alongside burn's rayon pool, the perception ORT sessions and the ASR decoder;
/// leaving each session at "all cores" oversubscribes the machine (measured: `encode_speech`
/// 1.08 s/call in the live pipeline vs ~54 ms in isolation).
pub fn set_ort_intra_threads(n: usize) {
    ORT_INTRA_THREADS.store(n, std::sync::atomic::Ordering::Relaxed);
}

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
    let intra = ORT_INTRA_THREADS.load(std::sync::atomic::Ordering::Relaxed);
    if intra > 0 {
        builder = builder
            .with_intra_threads(intra)
            .map_err(|e| EncodingError::Ort(e.into()))?;
        builder = builder
            .with_inter_threads(1)
            .map_err(|e| EncodingError::Ort(e.into()))?;
    }
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
