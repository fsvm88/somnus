use std::time::{Instant, SystemTime};

/// Dual timestamp: monotonic for ordering/duration; wall-clock for log messages.
/// Both are sampled at the same moment at the capture site.
#[derive(Debug, Clone)]
pub struct Timestamp {
    pub monotonic: Instant,
    pub wall: SystemTime,
}

impl Timestamp {
    pub fn now() -> Self {
        Self {
            monotonic: Instant::now(),
            wall: SystemTime::now(),
        }
    }
}

/// Interleaved f32 PCM samples from one cpal input-callback invocation.
///
/// `channels = 1` → mono samples; `channels = 2` → LRLR… interleaved.
/// All samples are normalised to `[-1.0, 1.0]` regardless of the device's
/// native integer format.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u8,
    pub timestamp: Timestamp,
}

impl AudioFrame {
    /// Number of complete sample frames (multi-channel groups).
    ///
    /// Equals `samples.len() / channels`.
    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Duration of the captured audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.frame_count() as f64 / self.sample_rate as f64
    }
}

/// Raw pixel layout as delivered by the capture device.
///
/// Conversion to a uniform format (e.g. `Rgb8`) happens in `somnus-encoding`,
/// not here. Phase 1 passes frames through unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
    Bgr8,
    Yuyv422,
    Mjpeg,
}

/// Packed pixel data from one camera frame, row-major.
///
/// `data.len()` equals `width * height * bytes_per_pixel()` for fixed-size
/// formats, or a variable number of bytes for `Mjpeg`.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub timestamp: Timestamp,
}

impl VideoFrame {
    /// Bytes per pixel for packed formats; returns 0 for variable-length `Mjpeg`.
    pub fn bytes_per_pixel(&self) -> usize {
        match self.format {
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
            PixelFormat::Yuyv422 => 2,
            PixelFormat::Mjpeg => 0,
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_audio(n_samples: usize, channels: u8, sample_rate: u32) -> AudioFrame {
        AudioFrame {
            samples: vec![0.0f32; n_samples],
            sample_rate,
            channels,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn frame_count_mono() {
        let f = make_audio(512, 1, 44100);
        assert_eq!(f.frame_count(), 512);
    }

    #[test]
    fn frame_count_stereo() {
        let f = make_audio(1024, 2, 44100);
        assert_eq!(f.frame_count(), 512);
    }

    #[test]
    fn duration_secs_mono() {
        // 44100 samples at 44100 Hz = 1.0 second
        let f = make_audio(44100, 1, 44100);
        let d = f.duration_secs();
        assert!((d - 1.0).abs() < 1e-9, "expected ~1.0 s, got {d}");
    }

    #[test]
    fn duration_secs_stereo() {
        // 88200 interleaved samples / 2 channels = 44100 frames at 44100 Hz = 1.0 s
        let f = make_audio(88200, 2, 44100);
        let d = f.duration_secs();
        assert!((d - 1.0).abs() < 1e-9, "expected ~1.0 s, got {d}");
    }

    #[test]
    fn bytes_per_pixel_values() {
        assert_eq!(
            VideoFrame {
                data: vec![],
                width: 1,
                height: 1,
                format: PixelFormat::Rgb8,
                timestamp: Timestamp::now()
            }
            .bytes_per_pixel(),
            3
        );
        assert_eq!(
            VideoFrame {
                data: vec![],
                width: 1,
                height: 1,
                format: PixelFormat::Bgr8,
                timestamp: Timestamp::now()
            }
            .bytes_per_pixel(),
            3
        );
        assert_eq!(
            VideoFrame {
                data: vec![],
                width: 1,
                height: 1,
                format: PixelFormat::Yuyv422,
                timestamp: Timestamp::now()
            }
            .bytes_per_pixel(),
            2
        );
        assert_eq!(
            VideoFrame {
                data: vec![],
                width: 1,
                height: 1,
                format: PixelFormat::Mjpeg,
                timestamp: Timestamp::now()
            }
            .bytes_per_pixel(),
            0
        );
    }
}
