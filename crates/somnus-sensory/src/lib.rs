//! `somnus-sensory` — Phase 1 sensory pipeline.
//!
//! Provides audio and video capture from live devices and media files,
//! typed frame structures, and per-source metrics. All types are sync;
//! async bridging is handled by the `somnus` binary.

mod audio;
mod error;
mod frame;
mod media_file;
mod metrics;
mod video;

pub use audio::{AudioCapture, AUDIO_CHANNEL_CAPACITY};
pub use error::{SensorError, SensorResult};
pub use frame::{AudioFrame, PixelFormat, Timestamp, VideoFrame};
pub use media_file::{MediaFile, MediaFileStreams};
pub use metrics::{MetricsSnapshot, SensorMetrics};
pub use video::{VideoCapture, VIDEO_CHANNEL_CAPACITY};
