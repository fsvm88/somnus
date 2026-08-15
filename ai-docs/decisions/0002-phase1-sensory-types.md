# 0002. Phase 1 sensory pipeline: concrete types and interfaces

- **Status:** accepted
- **Date:** 2026-04-26

> **Decision:** AudioFrame + VideoFrame types via cpal (mic) + nokhwa (webcam) + ffmpeg-next (files); crossbeam bounded channels with drop-oldest backpressure.  
> **Keywords:** AudioFrame, VideoFrame, cpal, nokhwa, ffmpeg-next, crossbeam, backpressure, MediaFile, sensory pipeline

## Context

ADR 0001 established the workspace layout. This ADR fixes the concrete Rust types for Phase 1 (frames, errors, metrics, capture handles) and the channel topology, so implementation is just writing code.

**Constraints:** cpal callbacks run on a dedicated platform thread — must not block/panic/allocate excessively. Video capture (v4l2) is pull-based. Bounded channels with drop-oldest backpressure, not blocking producers or unbounded growth. Binary uses `tokio`; library crates stay sync/async-agnostic. `thiserror` in libraries, `anyhow` in the binary. `tracing` only, no `println!`.

**Additional constraints:** must accept a video file as input (no camera available) — a media file provides both audio+video from one path. Mic audio can run alongside file audio (no sample-level mixing in Phase 1 — sources fan independently into the same channel). Graceful Ctrl+C shutdown, frames in flight discarded not flushed. Phase 1 "done" = capture pipeline + metrics + clean shutdown; encoding is Phase 2, no visualizer yet.

## Decision

### Module layout (`crates/somnus-sensory/src/`)

```
lib.rs          re-exports: AudioFrame, VideoFrame, Timestamp, PixelFormat,
                AudioCapture, VideoCapture, MediaFile, MediaFileStreams,
                SensorError, SensorResult, SensorMetrics, MetricsSnapshot
frame.rs        Timestamp, AudioFrame, VideoFrame, PixelFormat
audio.rs        AudioCapture, AUDIO_CHANNEL_CAPACITY
video.rs        VideoCapture, VIDEO_CHANNEL_CAPACITY
media_file.rs   MediaFile, MediaFileStreams
error.rs        SensorError, SensorResult
metrics.rs      SensorMetrics, MetricsSnapshot
```

---

### `frame.rs`

```rust
use std::time::{Instant, SystemTime};

/// Dual timestamp. Monotonic for ordering and duration math; wall-clock for logs.
/// Both are sampled at the same moment (in the capture callback / pull site).
#[derive(Debug, Clone)]
pub struct Timestamp {
    pub monotonic: Instant,
    pub wall: SystemTime,
}

impl Timestamp {
    pub fn now() -> Self {
        Self { monotonic: Instant::now(), wall: SystemTime::now() }
    }
}

/// Interleaved f32 PCM samples from one cpal input callback invocation.
/// `channels = 1` → mono; `channels = 2` → LRLR…
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u8,
    pub timestamp: Timestamp,
}

impl AudioFrame {
    /// Number of complete sample frames (samples.len() / channels).
    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Duration of the captured audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.frame_count() as f64 / self.sample_rate as f64
    }
}

/// Raw pixel layout coming off the camera.
/// Conversion to `Rgb8` happens in `somnus-encoding`, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
    Bgr8,
    Yuyv422,
    Mjpeg,
}

/// Packed pixels from one camera frame, row-major.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub timestamp: Timestamp,
}

impl VideoFrame {
    /// Bytes per pixel; returns 0 for variable-length formats (Mjpeg).
    pub fn bytes_per_pixel(&self) -> usize {
        match self.format {
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
            PixelFormat::Yuyv422 => 2,
            PixelFormat::Mjpeg => 0,
        }
    }
}
```

---

### `error.rs`

```rust
#[derive(Debug, thiserror::Error)]
pub enum SensorError {
    #[error("audio device unavailable")]
    NoAudioDevice,

    #[error("audio build stream: {0}")]
    AudioBuild(#[source] cpal::BuildStreamError),

    #[error("audio play stream: {0}")]
    AudioPlay(#[source] cpal::PlayStreamError),

    #[error("audio default config: {0}")]
    AudioConfig(#[source] cpal::DefaultStreamConfigError),

    #[error("video device {index} unavailable")]
    NoVideoDevice { index: u32 },

    #[error("video frame capture: {0}")]
    VideoCapture(String),

    #[error("media file open: {path}: {reason}")]
    MediaFileOpen { path: std::path::PathBuf, reason: String },

    #[error("media file has no {0} stream")]
    MediaFileNoStream(&'static str),

    #[error("media file decode: {0}")]
    MediaFileDecode(String),

    #[error("sensor channel closed")]
    ChannelClosed,

    #[error("unsupported audio sample format: {0:?}")]
    UnsupportedSampleFormat(cpal::SampleFormat),
}

pub type SensorResult<T> = Result<T, SensorError>;
```

---

### `metrics.rs`

```rust
use std::sync::atomic::{AtomicU64, Ordering};

/// Shared counters; safe to read from any thread.
/// Embedded in `Arc<SensorMetrics>` so capture handles and the main loop
/// can both observe them without locking.
#[derive(Debug, Default)]
pub struct SensorMetrics {
    pub frames_captured: AtomicU64,
    pub frames_dropped: AtomicU64,
}

impl SensorMetrics {
    pub fn increment_captured(&self) {
        self.frames_captured.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_dropped(&self) {
        self.frames_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Point-in-time snapshot; values are not atomically consistent with
    /// each other but are close enough for logging purposes.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            frames_captured: self.frames_captured.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    pub frames_captured: u64,
    pub frames_dropped: u64,
}

impl MetricsSnapshot {
    /// Fraction of produced frames that were dropped (0.0–1.0).
    pub fn drop_rate(&self) -> f64 {
        let total = self.frames_captured + self.frames_dropped;
        if total == 0 { 0.0 } else { self.frames_dropped as f64 / total as f64 }
    }
}
```

---

### `audio.rs`

```rust
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::FromSample;
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::Arc;
use crate::{AudioFrame, SensorError, SensorMetrics, SensorResult, Timestamp};

/// Frames in the crossbeam ring before the callback starts dropping oldest.
/// At a typical 10 ms chunk period this is ~2.5 s of headroom.
pub const AUDIO_CHANNEL_CAPACITY: usize = 256;

/// Live audio capture handle. Dropping this stops the stream.
pub struct AudioCapture {
    _stream: cpal::Stream,  // kept alive by ownership; drop = stop
    pub receiver: Receiver<AudioFrame>,
    pub metrics: Arc<SensorMetrics>,
}

impl AudioCapture {
    /// Start capturing from `device`, or the system default if `None`.
    /// Returns once the stream is playing. Frames arrive on `self.receiver`.
    pub fn start(device: Option<cpal::Device>) -> SensorResult<Self> {
        let host = cpal::default_host();
        let device = device
            .or_else(|| host.default_input_device())
            .ok_or(SensorError::NoAudioDevice)?;

        let config = device
            .default_input_config()
            .map_err(SensorError::AudioConfig)?;

        let metrics = Arc::new(SensorMetrics::default());
        let (tx, rx) = bounded::<AudioFrame>(AUDIO_CHANNEL_CAPACITY);

        let stream = Self::build_stream(&device, config, tx, rx.clone(), Arc::clone(&metrics))?;
        stream.play().map_err(SensorError::AudioPlay)?;

        Ok(Self { _stream: stream, receiver: rx, metrics })
    }

    fn build_stream(
        device: &cpal::Device,
        config: cpal::SupportedStreamConfig,
        tx: Sender<AudioFrame>,
        rx_drain: Receiver<AudioFrame>,    // used only for drop-oldest drain
        metrics: Arc<SensorMetrics>,
    ) -> SensorResult<cpal::Stream> {
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as u8;
        let fmt = config.sample_format();
        let stream_config: cpal::StreamConfig = config.into();

        // Expands to a typed callback for a given SizedSample type.
        // f32::from_sample(s) (cpal 0.17 re-exports dasp_sample::FromSample)
        // normalises any integer or float sample to [-1.0, 1.0] f32.
        macro_rules! make_cb {
            ($T:ty) => {{
                let tx = tx.clone();
                let rx_drain = rx_drain.clone();
                let metrics = metrics.clone();
                move |data: &[$T], _: &cpal::InputCallbackInfo| {
                    let frame = AudioFrame {
                        samples: data.iter().map(|&s| f32::from_sample(s)).collect(),
                        sample_rate,
                        channels,
                        timestamp: Timestamp::now(),
                    };
                    metrics.increment_captured();
                    if let Err(crossbeam_channel::TrySendError::Full(frame)) = tx.try_send(frame) {
                        let _ = rx_drain.try_recv();
                        metrics.increment_dropped();
                        tracing::trace!("audio channel full; dropped oldest frame");
                        let _ = tx.try_send(frame);
                    }
                }
            }};
        }

        let err_cb = |err| tracing::error!(?err, "audio stream error");
        let stream = match fmt {
            cpal::SampleFormat::I8  => device.build_input_stream(&stream_config, make_cb!(i8),  err_cb, None),
            cpal::SampleFormat::I16 => device.build_input_stream(&stream_config, make_cb!(i16), err_cb, None),
            cpal::SampleFormat::I32 => device.build_input_stream(&stream_config, make_cb!(i32), err_cb, None),
            cpal::SampleFormat::I64 => device.build_input_stream(&stream_config, make_cb!(i64), err_cb, None),
            cpal::SampleFormat::U8  => device.build_input_stream(&stream_config, make_cb!(u8),  err_cb, None),
            cpal::SampleFormat::U16 => device.build_input_stream(&stream_config, make_cb!(u16), err_cb, None),
            cpal::SampleFormat::U32 => device.build_input_stream(&stream_config, make_cb!(u32), err_cb, None),
            cpal::SampleFormat::U64 => device.build_input_stream(&stream_config, make_cb!(u64), err_cb, None),
            cpal::SampleFormat::F32 => device.build_input_stream(&stream_config, make_cb!(f32), err_cb, None),
            cpal::SampleFormat::F64 => device.build_input_stream(&stream_config, make_cb!(f64), err_cb, None),
            other => return Err(SensorError::UnsupportedSampleFormat(other)),
        };
        stream.map_err(SensorError::AudioBuild)
    }
}
```

**Design notes:** `default_input_config()` often returns I16 on Linux ALSA, not F32 — the `make_cb!` macro generates a typed callback per `SizedSample` type, normalizing everything to `[-1.0,1.0]` f32 via `f32::from_sample`; I8/I16/I32/I64/U8/U16/U32/U64/F32/F64 handled, I24/U24/DSD → `UnsupportedSampleFormat` (rare). The callback holds both `tx` and a cloned `rx_drain` for drop-oldest; never blocks (`try_send`/`try_recv` are O(1)). `_stream` is held only to prevent `Drop` from stopping playback. **Multi-source fan-in:** `Sender<T>: Clone` lets a `MediaFile`'s audio and a live `AudioCapture` send into the same downstream channel — no mixer type needed, frames interleave by arrival order; sample-level mixing deferred to Phase 2+.

---

### `video.rs`

`VideoCapture` is webcam-only. Video from files is handled by `MediaFile`
(see below). This keeps the nokhwa and ffmpeg dependencies separate.

Video uses a pull model: the caller drives the loop and requests frames one by one.
Unlike audio, there is no platform callback to bridge.

```rust
use nokhwa::{Camera, CameraIndex, pixel_format::RgbFormat};
use nokhwa::utils::{RequestedFormat, RequestedFormatType};
use std::sync::Arc;
use crate::{PixelFormat, SensorError, SensorMetrics, SensorResult, Timestamp, VideoFrame};

pub const VIDEO_CHANNEL_CAPACITY: usize = 16; // reserved for Phase 2 bridge

/// Live video capture handle. Pull frames with `next_frame()`.
/// Dropping this releases the camera device.
pub struct VideoCapture {
    camera: Camera,
    pub metrics: Arc<SensorMetrics>,
}

impl VideoCapture {
    /// Open camera at `index` (0 = first camera).
    /// Requests 30 fps RGB at the camera's native resolution.
    pub fn open(index: u32) -> SensorResult<Self> {
        let requested = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::AbsoluteHighestFrameRate,
        );
        let mut camera = Camera::new(CameraIndex::Index(index), requested)
            .map_err(|e| SensorError::NoVideoDevice { index })?;
        camera
            .open_stream()
            .map_err(|e| SensorError::VideoCapture(e.to_string()))?;

        Ok(Self { camera, metrics: Arc::new(SensorMetrics::default()) })
    }

    /// Block until the next frame is available from the camera.
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
```

**Crate selection — `nokhwa`:** alternatives were `v4l` (Linux-only, minimal, lower-level) and `rscam` (older, less maintained). `nokhwa` chosen for its consistent cross-platform interface and active maintenance — keeps the door open if hardware ever moves off Linux; `features = ["input-native"]` avoids an OpenCV dependency. API verified against nokhwa 0.10.10: `Camera::frame() -> Result<Buffer, NokhwaError>`, `Buffer::buffer() -> &[u8]`, `RequestedFormatType::AbsoluteHighestFrameRate` exists.

---

### `media_file.rs`

Decodes both audio and video streams from a single media file (MP4, MKV, etc.) — the primary use case is testing without a camera. **Phase 1 limitation:** both streams must be present; audio-only or video-only files fail `start()` with `MediaFileNoStream`. **Thread architecture:** a single background thread demuxes and decodes both streams inline (avoids inter-thread packet queues / sharing a format context); fast enough for typical H.264+AAC files with no perceptible latency from serializing the two.

```rust
use std::path::PathBuf;
use std::sync::Arc;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use ffmpeg_next as ffmpeg;
use ffmpeg::{
    codec,
    channel_layout::ChannelLayout,
    format::{self, sample::Type as SampleType, Pixel},
    frame,
    media::Type as MediaType,
    software::{resampling, scaling},
};
use crate::{AudioFrame, PixelFormat, SensorError, SensorMetrics, SensorResult, Timestamp, VideoFrame};

/// Decoded streams from a media file. Dropping this stops decoding.
pub struct MediaFileStreams {
    /// Decoded audio frames (f32 PCM interleaved, sample rate from the file's audio track).
    pub audio: Receiver<AudioFrame>,
    /// Decoded video frames (Rgb8 packed, resolution from the file's video track).
    pub video: Receiver<VideoFrame>,
    pub audio_metrics: Arc<SensorMetrics>,
    pub video_metrics: Arc<SensorMetrics>,
    _handles: MediaFileHandles,
}

/// Open a media file and start decoding on a background thread.
pub struct MediaFile {
    path: PathBuf,
}

impl MediaFile {
    pub fn open(path: impl Into<PathBuf>) -> SensorResult<Self> {
        let path = path.into();
        if !path.exists() {
            return Err(SensorError::MediaFileOpen {
                path,
                reason: "file not found".into(),
            });
        }
        Ok(Self { path })
    }

    /// Open the format context, create decoders and format converters, then spawn the
    /// decode thread. All initialisation errors surface here before the thread starts.
    pub fn start(self) -> SensorResult<MediaFileStreams> {
        ffmpeg::init().map_err(|e| SensorError::MediaFileDecode(e.to_string()))?;

        // ── format context ─────────────────────────────────────────────────
        let mut ictx = format::input(&self.path)
            .map_err(|e| SensorError::MediaFileOpen {
                path: self.path.clone(),
                reason: e.to_string(),
            })?;

        // ── audio decoder ──────────────────────────────────────────────────
        // Nested scope so the stream borrow on ictx ends before we look up the video stream.
        let (audio_idx, mut audio_decoder) = {
            let stream = ictx.streams().best(MediaType::Audio)
                .ok_or(SensorError::MediaFileNoStream("audio"))?;
            let idx = stream.index();
            let dec = codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| SensorError::MediaFileDecode(e.to_string()))?
                .decoder()
                .audio()
                .map_err(|e| SensorError::MediaFileDecode(e.to_string()))?;
            (idx, dec)
        };

        // Some containers omit channel layout (bits == 0); fall back to channel count.
        let channel_layout = if audio_decoder.channel_layout().bits() == 0 {
            match audio_decoder.channels() {
                1 => ChannelLayout::MONO,
                _ => ChannelLayout::STEREO,
            }
        } else {
            audio_decoder.channel_layout()
        };

        // Resample any source format to interleaved f32 at the file's own sample rate.
        // Handles FLTP (f32 planar), S16 (i16 packed), etc. Same-rate: no buffering delay.
        let mut audio_resampler = resampling::Context::get(
            audio_decoder.format(),
            channel_layout,
            audio_decoder.rate(),
            format::Sample::F32(SampleType::Packed),
            channel_layout,
            audio_decoder.rate(),
        ).map_err(|e| SensorError::MediaFileDecode(e.to_string()))?;

        let sample_rate = audio_decoder.rate();
        let channels    = audio_decoder.channels() as u8;

        // ── video decoder ──────────────────────────────────────────────────
        let (video_idx, mut video_decoder) = {
            let stream = ictx.streams().best(MediaType::Video)
                .ok_or(SensorError::MediaFileNoStream("video"))?;
            let idx = stream.index();
            let dec = codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| SensorError::MediaFileDecode(e.to_string()))?
                .decoder()
                .video()
                .map_err(|e| SensorError::MediaFileDecode(e.to_string()))?;
            (idx, dec)
        };

        // Scale any source pixel format to packed RGB24.
        let mut video_scaler = scaling::Context::get(
            video_decoder.format(),
            video_decoder.width(),
            video_decoder.height(),
            Pixel::RGB24,
            video_decoder.width(),
            video_decoder.height(),
            scaling::flag::Flags::BILINEAR,
        ).map_err(|e| SensorError::MediaFileDecode(e.to_string()))?;

        let width  = video_decoder.width();
        let height = video_decoder.height();

        // ── channels ───────────────────────────────────────────────────────
        let (audio_tx, audio_rx) = bounded::<AudioFrame>(256);
        let (video_tx, video_rx) = bounded::<VideoFrame>(16);
        // Drain clones: used inside the decode thread for drop-oldest backpressure.
        // Single producer → the retry after drain is always safe.
        let audio_drain = audio_rx.clone();
        let video_drain = video_rx.clone();

        let audio_metrics   = Arc::new(SensorMetrics::default());
        let video_metrics   = Arc::new(SensorMetrics::default());
        let audio_metrics_t = Arc::clone(&audio_metrics);
        let video_metrics_t = Arc::clone(&video_metrics);

        // ── decode thread ──────────────────────────────────────────────────
        let handle = std::thread::spawn(move || {
            // Reuse raw decoded frames; allocate converter output fresh each call
            // because frame sizes can vary (VFR, codec flush boundaries).
            let mut audio_raw = frame::Audio::empty();
            let mut video_raw = frame::Video::empty();

            // Main demux + decode loop.
            for (stream, packet) in ictx.packets() {
                if stream.index() == audio_idx {
                    if audio_decoder.send_packet(&packet).is_err() { break; }
                    while audio_decoder.receive_frame(&mut audio_raw).is_ok() {
                        let mut conv = frame::Audio::empty();
                        if audio_resampler.run(&audio_raw, &mut conv).is_err() { continue; }
                        if conv.samples() == 0 { continue; }
                        let af = AudioFrame {
                            samples:     f32_from_audio_frame(&conv),
                            sample_rate,
                            channels,
                            timestamp:   Timestamp::now(),
                        };
                        audio_metrics_t.increment_captured();
                        if !send_drop_oldest(&audio_tx, &audio_drain, af, &audio_metrics_t) {
                            return; // consumer gone; skip flush
                        }
                    }
                } else if stream.index() == video_idx {
                    if video_decoder.send_packet(&packet).is_err() { break; }
                    while video_decoder.receive_frame(&mut video_raw).is_ok() {
                        let mut conv = frame::Video::empty();
                        if video_scaler.run(&video_raw, &mut conv).is_err() { continue; }
                        let vf = VideoFrame {
                            data:      rgb_from_video_frame(&conv),
                            width,
                            height,
                            format:    PixelFormat::Rgb8,
                            timestamp: Timestamp::now(),
                        };
                        video_metrics_t.increment_captured();
                        if !send_drop_oldest(&video_tx, &video_drain, vf, &video_metrics_t) {
                            return;
                        }
                    }
                }
            }

            // ── flush audio decoder ────────────────────────────────────────
            audio_decoder.send_eof().ok();
            while audio_decoder.receive_frame(&mut audio_raw).is_ok() {
                let mut conv = frame::Audio::empty();
                if audio_resampler.run(&audio_raw, &mut conv).is_ok() && conv.samples() > 0 {
                    let af = AudioFrame {
                        samples: f32_from_audio_frame(&conv),
                        sample_rate, channels,
                        timestamp: Timestamp::now(),
                    };
                    audio_metrics_t.increment_captured();
                    send_drop_oldest(&audio_tx, &audio_drain, af, &audio_metrics_t);
                }
            }
            // Drain resampler residue (same-rate resampler holds ≤ 1 frame; loop is short).
            loop {
                let mut conv = frame::Audio::empty();
                match audio_resampler.flush(&mut conv) {
                    Ok(delay) => {
                        if conv.samples() > 0 {
                            let af = AudioFrame {
                                samples: f32_from_audio_frame(&conv),
                                sample_rate, channels,
                                timestamp: Timestamp::now(),
                            };
                            audio_metrics_t.increment_captured();
                            send_drop_oldest(&audio_tx, &audio_drain, af, &audio_metrics_t);
                        }
                        if delay.is_none() { break; }
                    }
                    Err(_) => break,
                }
            }

            // ── flush video decoder ────────────────────────────────────────
            video_decoder.send_eof().ok();
            while video_decoder.receive_frame(&mut video_raw).is_ok() {
                let mut conv = frame::Video::empty();
                if video_scaler.run(&video_raw, &mut conv).is_ok() {
                    let vf = VideoFrame {
                        data: rgb_from_video_frame(&conv),
                        width, height,
                        format: PixelFormat::Rgb8,
                        timestamp: Timestamp::now(),
                    };
                    video_metrics_t.increment_captured();
                    send_drop_oldest(&video_tx, &video_drain, vf, &video_metrics_t);
                }
            }
            // Scaler is stateless: no flush needed.
        });

        Ok(MediaFileStreams {
            audio: audio_rx,
            video: video_rx,
            audio_metrics,
            video_metrics,
            _handles: MediaFileHandles { _decode_thread: handle },
        })
    }
}

// Private: keeps decode thread alive. Dropping detaches the thread;
// it exits when both send channels detect Disconnected.
struct MediaFileHandles {
    _decode_thread: std::thread::JoinHandle<()>,
}

// Extract interleaved f32 samples from an F32(Packed) audio frame (plane 0).
fn f32_from_audio_frame(frame: &frame::Audio) -> Vec<f32> {
    frame.data(0)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}

// Copy RGB24 pixels from a video frame, stripping row padding.
// stride(0) may exceed width * 3 on some decoders/platforms.
fn rgb_from_video_frame(frame: &frame::Video) -> Vec<u8> {
    let w      = frame.width()  as usize;
    let h      = frame.height() as usize;
    let stride = frame.stride(0);
    let src    = frame.data(0);
    if stride == w * 3 {
        src[..w * h * 3].to_vec()
    } else {
        let mut out = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            out.extend_from_slice(&src[row * stride..row * stride + w * 3]);
        }
        out
    }
}

// Send with drop-oldest backpressure. Returns true if the channel is alive.
// Single-producer: after draining one slot the retry is always safe.
fn send_drop_oldest<T>(
    tx:      &Sender<T>,
    drain:   &Receiver<T>,
    frame:   T,
    metrics: &SensorMetrics,
) -> bool {
    match tx.try_send(frame) {
        Ok(()) => true,
        Err(TrySendError::Full(frame)) => {
            let _ = drain.try_recv();
            metrics.increment_dropped();
            tx.try_send(frame).is_ok()
        }
        Err(TrySendError::Disconnected(_)) => false,
    }
}
```

**Design notes:** `ffmpeg::init()` is idempotent (static `AtomicBool`), safe to call from `start()` regardless of what else calls it. Decode thread reuses `audio_raw`/`video_raw` across iterations; converter output (`conv`) allocated fresh per call since frame sizes vary across codec flush boundaries. Channel-layout fallback (`bits()==0`) avoids a resampler init panic on containers that omit it (raw PCM, some MP4 muxings). Resampler flush loop ends on `delay.is_none()` — same-rate conversion holds ≤1 frame, so it runs at most twice. Video scaler is stateless, no flush needed. Audio-only/video-only files unsupported in Phase 1.

**Crate selection — `ffmpeg-next`:** alternatives were `video-rs` (limited audio support) and `gstreamer` (heavyweight pipeline model) and `symphonia`+`video-rs` (pure Rust, but can't share one format context for a single file). `ffmpeg-next` chosen for exposing both streams through one format context plus resampling/pixel-conversion via libswresample/libswscale. Requires `libavcodec`/`libavformat`/`libavutil`/`libswresample`/`libswscale` on the build host (`pacman -S ffmpeg`).

---

### Channel topology

```
Phase 1 — live capture (webcam + mic):

  [cpal callback]──crossbeam::bounded(256)──► [spawn_blocking bridge]──► tokio::mpsc──► [consumer]
  [VideoCapture thread]──────────────────────► [spawn_blocking bridge]──► tokio::mpsc──► [consumer]

Phase 1 — file input (MediaFile only):

  [audio decoder thread]──crossbeam::bounded──► \
                                                 fan-in via Sender::clone ──► tokio::mpsc──► [consumer]
  [video decoder thread]───crossbeam::bounded──► [spawn_blocking bridge]───► tokio::mpsc──► [consumer]

Phase 1 — mixed (MediaFile + live mic):

  [cpal callback]─────────────────────────────► \
                                                 same crossbeam Sender clone ──► tokio::mpsc──► [consumer]
  [audio decoder thread]──crossbeam::bounded──► /

Phase 2 change:
  - Replace per-consumer mpsc with tokio::sync::broadcast (one sender, N subscribers)
  - somnus-encoding subscribes to both audio and video broadcast channels
```

The Phase 1 binary bridges sync capture to async using:

```rust
// One-time bridge task per capture source:
let (async_tx, async_rx) = tokio::sync::mpsc::channel::<AudioFrame>(64);
tokio::task::spawn_blocking(move || {
    while let Ok(frame) = audio.receiver.recv() {
        if async_tx.blocking_send(frame).is_err() {
            break;  // consumer dropped
        }
    }
});
// async_rx consumed in an async task
```

---

### Shutdown cascade

On Ctrl+C the shutdown propagates automatically through the channel graph:

```
1. ctrl_c() resolves → main loop break → main() returns
2. audio_tx / video_tx (tokio mpsc Senders) drop
   → spawn_blocking bridges: blocking_send returns Err → bridges exit
   → audio_cb_rx / video_cb_rx (crossbeam Receivers, held by bridges) drop
3. crossbeam fan-in channels now have no receivers → Disconnected
   → forwarding threads: tx.try_send returns TrySendError::Disconnected → break
   → forward_frames returns; threads exit
4. Forwarding threads drop their src Receivers (file_audio_rx / mic.receiver / file_video_rx)
   → MediaFile decoder threads: their senders get Disconnected on next send → exit
   → mic's AudioCapture drops → _stream drops → cpal stream stops
5. VideoCapture pull loop: TrySendError::Disconnected → break → cam drops → device released
```

Requirements: forwarding threads MUST match `TrySendError::Disconnected` and break — silently ignoring send errors breaks the cascade (thread loops forever, source never released). `MediaFileStreams._handles` being dropped on destructure is intentional — decoder threads detect Disconnected on their own senders and exit themselves; JoinHandles aren't the lifetime mechanism. Frames in flight at shutdown are discarded, no flush attempted.

---

### Cargo.toml sketches

**Workspace root (`Cargo.toml`):**

```toml
[workspace]
members = ["crates/somnus", "crates/somnus-sensory"]
resolver = "2"

[workspace.dependencies]
anyhow            = "1"
clap              = { version = "4", features = ["derive"] }
cpal              = "0.17"
crossbeam-channel = "0.5"
ffmpeg-next       = "8"
nokhwa            = { version = "0.10", features = ["input-native"] }
thiserror         = "2"
tokio             = { version = "1", features = ["full"] }
tracing           = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
```

**`crates/somnus-sensory/Cargo.toml`:**

```toml
[package]
name    = "somnus-sensory"
version = "0.1.0"
edition = "2021"

[dependencies]
cpal              = { workspace = true }
crossbeam-channel = { workspace = true }
ffmpeg-next       = { workspace = true }
nokhwa            = { workspace = true }
thiserror         = { workspace = true }
tracing           = { workspace = true }
```

**`crates/somnus/Cargo.toml`:**

```toml
[package]
name    = "somnus"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "somnus"
path = "src/main.rs"

[dependencies]
somnus-sensory        = { path = "../somnus-sensory" }
anyhow             = { workspace = true }
clap               = { workspace = true }
tokio              = { workspace = true }
tracing            = { workspace = true }
tracing-subscriber = { workspace = true }
```

---

### Binary sketch (`crates/somnus/src/main.rs`)

The binary accepts an optional `--file` argument (via `clap`). If supplied,
`MediaFile` provides both audio and video streams from one format context; the
file is opened **exactly once** and `MediaFileStreams` is destructured immediately
so each receiver can be moved into its own forwarding thread. An optional `--mic`
flag additionally enables live microphone input alongside file playback. If neither
`--file` nor a webcam is used together, the fallback is live capture only.

```rust
use anyhow::Result;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use somnus_sensory::{
    AudioCapture, AudioFrame, MediaFile, MediaFileStreams, SensorMetrics,
    VideoCapture, VideoFrame,
};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::sync::mpsc;
use tracing::info;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    file: Option<PathBuf>,
    #[arg(long)]
    mic: bool,
    #[arg(long, default_value = "0")]
    webcam: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    // ── open file once; destructure both streams immediately ───────────────
    let (file_audio_rx, file_video_rx, file_audio_metrics, file_video_metrics) =
        match args.file.as_ref()
            .map(|p| MediaFile::open(p).and_then(|mf| mf.start()))
            .transpose()?
        {
            Some(MediaFileStreams { audio, video, audio_metrics, video_metrics, .. }) =>
                (Some(audio), Some(video), Some(audio_metrics), Some(video_metrics)),
            None => (None, None, None, None),
        };

    // ── audio setup ────────────────────────────────────────────────────────
    // audio_drain is a receiver clone; forwarding threads use it for drop-oldest
    // backpressure on the fan-in channel. The async bridge holds audio_cb_rx.
    let (audio_cb_tx, audio_cb_rx) = bounded::<AudioFrame>(256);
    let audio_drain = audio_cb_rx.clone();
    let mut audio_metric_handles: Vec<Arc<SensorMetrics>> = Vec::new();

    if let Some(rx) = file_audio_rx {
        audio_metric_handles.push(file_audio_metrics.unwrap());
        let tx = audio_cb_tx.clone();
        let drain = audio_drain.clone();
        std::thread::spawn(move || forward_frames(rx, tx, drain));
    }

    if args.mic || args.file.is_none() {
        let mic = AudioCapture::start(None)?;
        audio_metric_handles.push(Arc::clone(&mic.metrics));
        let tx = audio_cb_tx.clone();
        let drain = audio_drain.clone();
        std::thread::spawn(move || {
            // mic.receiver moved here; mic dropped at closure end → cpal stream stops.
            forward_frames(mic.receiver, tx, drain);
        });
    }

    // ── video setup ────────────────────────────────────────────────────────
    let (video_cb_tx, video_cb_rx) = bounded::<VideoFrame>(16);
    let video_drain = video_cb_rx.clone();
    let video_metrics_handle: Arc<SensorMetrics>;

    if let Some(rx) = file_video_rx {
        video_metrics_handle = file_video_metrics.unwrap();
        let tx = video_cb_tx.clone();
        let drain = video_drain.clone();
        std::thread::spawn(move || forward_frames(rx, tx, drain));
    } else {
        let mut cam = VideoCapture::open(args.webcam)?;
        video_metrics_handle = Arc::clone(&cam.metrics);
        let tx = video_cb_tx.clone();
        let drain = video_drain.clone();
        std::thread::spawn(move || loop {
            match cam.next_frame() {
                Ok(frame) => match tx.try_send(frame) {
                    Ok(()) => {}
                    Err(TrySendError::Full(frame)) => {
                        let _ = drain.try_recv();
                        if tx.try_send(frame).is_err() { break; }
                    }
                    Err(TrySendError::Disconnected(_)) => break,
                },
                Err(e) => { tracing::error!(?e, "video capture error"); break; }
            }
        });
    }

    // ── async bridges ──────────────────────────────────────────────────────
    let (audio_tx, mut audio_rx) = mpsc::channel::<AudioFrame>(64);
    tokio::task::spawn_blocking(move || {
        while let Ok(f) = audio_cb_rx.recv() {
            if audio_tx.blocking_send(f).is_err() { break; }
        }
    });

    let (video_tx, mut video_rx) = mpsc::channel::<VideoFrame>(16);
    tokio::task::spawn_blocking(move || {
        while let Ok(f) = video_cb_rx.recv() {
            if video_tx.blocking_send(f).is_err() { break; }
        }
    });

    // ── main loop ──────────────────────────────────────────────────────────
    let mut metrics_tick = tokio::time::interval(Duration::from_secs(10));

    loop {
        tokio::select! {
            Some(frame) = audio_rx.recv() => {
                tracing::debug!(frames = frame.frame_count(), duration = frame.duration_secs(), "audio");
            }
            Some(frame) = video_rx.recv() => {
                tracing::debug!(width = frame.width, height = frame.height, format = ?frame.format, "video");
            }
            _ = metrics_tick.tick() => {
                let v = video_metrics_handle.snapshot();
                info!(video_captured = v.frames_captured, video_dropped = v.frames_dropped);
                for m in &audio_metric_handles {
                    let a = m.snapshot();
                    info!(audio_captured = a.frames_captured,
                          audio_drop_rate = format!("{:.1}%", a.drop_rate() * 100.0));
                }
            }
            _ = tokio::signal::ctrl_c() => {
                info!("shutting down");
                break;
            }
        }
    }
    Ok(())
}

/// Forward frames from src to tx with drop-oldest backpressure via drain.
/// Exits cleanly when tx is disconnected — the shutdown signal from the consumer.
fn forward_frames<T>(src: Receiver<T>, tx: Sender<T>, drain: Receiver<T>) {
    while let Ok(frame) = src.recv() {
        match tx.try_send(frame) {
            Ok(()) => {}
            Err(TrySendError::Full(frame)) => {
                let _ = drain.try_recv();
                if tx.try_send(frame).is_err() { break; }
            }
            Err(TrySendError::Disconnected(_)) => break,
        }
    }
}
```

**Notes:** `MediaFileStreams` destructured with `..` drops `_handles` immediately — detaches the decoder threads, which exit via Disconnected detection (see Shutdown cascade). The webcam pull loop can't use `forward_frames` (`VideoCapture` has no source `Receiver`, it's a blocking pull model) — drop-oldest is inlined identically. `audio_drain`/`video_drain` clones are held only as drain templates, not consumed by the bridge.

---

### Phase 2+ crate sketches (structural only — types TBD in Phase 2 ADRs)

**`somnus-encoding`** (Phase 2)
- Deps: `somnus-sensory`, `ort`, `ndarray` (or `candle-core` for tensor ops)
- Primary types: `AudioEmbedding`, `VideoEmbedding`, `EncoderHandle`
- Primary ops: `AudioFrame → AudioEmbedding`, `VideoFrame → VideoEmbedding`
- One `ort::Session` per frozen model; sessions live for the process lifetime
- Channel shape: `mpsc::Receiver<AudioFrame>` → `broadcast::Sender<AudioEmbedding>`

**`somnus-perception`** (Phase 3)
- Deps: `somnus-encoding`, `candle-core`, `candle-nn`
- Primary types: `SparseActivation { indices: Vec<u16>, values: Vec<f32>, region: RegionId }`
- Primary ops: `Embedding → SparseActivation` (k-WTA), `SparseActivation → Episode`
- One `candle::Tensor` weight matrix per region; updated during sleep only

**`somnus-memory`** (Phase 4)
- Deps: `somnus-perception`, `hnsw_rs` (HNSW; superseded usearch per ADR 0006), `memmap2` (cold tier)
- Three storage structs: `EpisodicBuffer` (hot/VRAM), `SemanticStore` (warm/RAM),
  `ColdArchive` (disk)
- HNSW index managed inside `SemanticStore`; rebuilt lazily when tombstone
  fraction exceeds threshold
- `Episode` type: `{ id: u64, timestamp: Timestamp, activations: Vec<SparseActivation>,
  binding_embedding: Vec<half::f16> }` (`f16` is unstable; use the `half` crate)

**`somnus-sleep`** (Phase 5)
- Deps: all above
- Owns the sleep/wake state machine; runs as a separate tokio task
- Signals: `SleepTrigger` enum (`EpisodicHighWaterMark`, `PerformanceDegradation`,
  `Scheduled`)
- Phases: `NremLike` (replay, homeostasis, compression) then `RemLike`
  (self-probe, optional growth)

## Alternatives considered

- **`VideoSource`/`AudioSource` enums dispatching webcam-vs-file inside `VideoCapture`/audio capture.** Rejected: would hide the nokhwa/ffmpeg-next backend split inside one type, and a video file's audio track has no business being decoded by `VideoCapture` anyway — `MediaFile` owning both streams is cleaner. `Sender::clone` fan-in achieves the same result at the binary level with no library-level coupling.
- **Sample-level audio mixing in Phase 1.** Requires resampling + timeline sync across sources — deferred to Phase 2+; arrival-order interleaving is enough to verify capture works.
- **`tokio::sync::broadcast` for Phase 1 audio.** Discards on a slow reader instead of dropping oldest — wrong semantics once Phase 2 has multiple consumers at different speeds; revisit then.
- **`tokio::sync::mpsc` directly in the cpal callback.** Not possible — `mpsc::Sender` can't be cloned for the drain path; crossbeam gives both send and drain on one channel.
- **`ndarray` as the tensor type in `somnus-sensory`.** Rejected — no ML computation here; that's `somnus-encoding`'s job.

## Consequences

- Phase 1 implementation order: frame → error → metrics → audio → video → media_file → binary.
- `ffmpeg-next` requires libavcodec/libavformat/libavutil/libswresample/libswscale on the build host (`pacman -S ffmpeg`) — a hard build dependency.
- `AudioCapture`'s crossbeam channel is internal; Phase 2 topology changes stay isolated to `audio.rs` + the binary's bridge code.
- `nokhwa` is unused on the primary (file-input) test path — first exercised once a webcam is available.

## What would invalidate this

- The cpal callback's `iter().map().collect()` allocation measurably increasing latency (unlikely at ~10ms chunks) → pre-allocated ring buffer + `Arc<[f32]>` zero-copy frames.
- `nokhwa` failing to open the camera or producing wrong pixel formats on the target hardware → fall back to `v4l`.
- Phase 2 encoding needing different drop semantics per consumer → redesign the channel topology before Phase 2 ships.
