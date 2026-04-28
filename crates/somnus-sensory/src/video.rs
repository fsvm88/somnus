use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use std::sync::Arc;

use crate::{PixelFormat, SensorError, SensorMetrics, SensorResult, Timestamp, VideoFrame};

/// Reserved for Phase 2 async bridge; live capture uses a pull loop.
pub const VIDEO_CHANNEL_CAPACITY: usize = 16;

/// Live webcam capture handle. Pull one frame at a time with `next_frame()`.
/// Dropping this releases the camera device.
pub struct VideoCapture {
    camera: Camera,
    pub metrics: Arc<SensorMetrics>,
}

// SAFETY: `VideoCapture` is always used from a single dedicated pull thread.
// `nokhwa::Camera` holds `Box<dyn CaptureBackendTrait>` which lacks a `Send`
// bound, but the v4l2 backend does not use internal thread-local state; moving
// the handle between threads (once) before exclusive use is safe.
unsafe impl Send for VideoCapture {}

impl VideoCapture {
    /// Open camera at `index` (0 = first attached camera).
    /// Requests the highest available frame rate in RGB format.
    pub fn open(index: u32) -> SensorResult<Self> {
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera = Camera::new(CameraIndex::Index(index), requested)
            .map_err(|_| SensorError::NoVideoDevice { index })?;
        camera
            .open_stream()
            .map_err(|e| SensorError::VideoCapture(e.to_string()))?;

        Ok(Self {
            camera,
            metrics: Arc::new(SensorMetrics::default()),
        })
    }

    /// Block until the next frame is available.
    /// Returns an `Rgb8`-format `VideoFrame` at the camera's native resolution.
    pub fn next_frame(&mut self) -> SensorResult<VideoFrame> {
        let frame = self
            .camera
            .frame()
            .map_err(|e| SensorError::VideoCapture(e.to_string()))?;
        let resolution = frame.resolution();
        self.metrics.increment_captured();
        Ok(VideoFrame {
            data: frame.buffer().to_vec(),
            width: resolution.width(),
            height: resolution.height(),
            format: PixelFormat::Rgb8,
            timestamp: Timestamp::now(),
        })
    }
}
