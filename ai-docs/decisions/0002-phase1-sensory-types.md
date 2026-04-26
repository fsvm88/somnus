# 0002. Phase 1 sensory pipeline: concrete types and interfaces

- **Status:** accepted
- **Date:** 2026-04-26

## Context

ADR 0001 established the workspace layout. This ADR fixes the concrete Rust types
for Phase 1: frames, errors, metrics, capture handles, and the channel topology
that connects them. The goal is that implementation is about writing code, not
resolving architecture.

Constraints carried in from prior research:

- cpal audio callbacks run on a dedicated platform thread; they must not block,
  panic, or allocate excessively.
- Video capture (v4l2 on Linux) is a pull API: the caller requests a frame.
- A bounded channel with drop-oldest backpressure is preferred over blocking the
  producer or unbounded growth.
- The binary uses `tokio`; library crates are `async`-agnostic (sync types only).
- `thiserror` in libraries; `anyhow` in the binary.
- `tracing` for all telemetry; no `println!` or `eprintln!` in production paths.

Additional constraints settled post-design:

- The system must accept a video file as input (user has no camera). A media file
  provides both audio and video streams from one path.
- Mic audio can run alongside file audio simultaneously (e.g., processing a movie
  while a person speaks in the room). No sample-level mixing in Phase 1 — sources
  independently produce `AudioFrame`s that fan into the same downstream channel.
- Graceful shutdown on Ctrl+C is required; frames in flight are discarded, not
  flushed.
- Phase 1 "done" = capture pipeline running, metrics logging, clean shutdown.
  Encoding (frozen models via `ort`) is Phase 2. No visualizer in Phase 1.

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
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::Arc;

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

        device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let frame = AudioFrame {
                        samples: data.to_vec(),
                        sample_rate,
                        channels,
                        timestamp: Timestamp::now(),
                    };
                    metrics.increment_captured();
                    // Drop-oldest: attempt send; if full, drain one then retry.
                    // Using try_send's error path avoids the TOCTOU race of
                    // is_full() followed by try_recv().
                    if let Err(crossbeam_channel::TrySendError::Full(frame)) = tx.try_send(frame) {
                        let _ = rx_drain.try_recv();
                        metrics.increment_dropped();
                        tracing::trace!("audio channel full; dropped oldest frame");
                        let _ = tx.try_send(frame);
                    }
                },
                |err| tracing::error!(?err, "audio stream error"),
                None,
            )
            .map_err(SensorError::AudioBuild)
    }
}
```

**Design notes:**

- The callback captures both `tx` (Sender) and `rx_drain` (a clone of the same
  Receiver). `crossbeam_channel` Receivers are cloneable; the main consumer's
  Receiver sees the same channel minus whichever frames were drained.
- The callback never blocks. `try_send` and `try_recv` are both O(1).
- `_stream` is never read; it is held only to prevent `Drop` from stopping playback.
- **Multi-source fan-in:** because `crossbeam_channel::Sender<T>` is `Clone`, a
  `MediaFile`'s audio output and a live `AudioCapture` can both send into the
  *same* downstream channel by cloning the `Sender`. The binary handles this; no
  special mixer type is needed. Frames from both sources are interleaved by
  arrival order. Sample-level mixing (summing waveforms) is deferred to Phase 2+.

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

**Crate selection — `nokhwa`:**

Alternatives:
- `v4l` (Linux-only v4l2 wrapper, minimal deps, lower-level API)
- `rscam` (older v4l2 wrapper, less maintained)
- `nokhwa` (cross-platform, actively maintained, clean frame API)

`nokhwa` chosen because it wraps platform APIs behind a consistent interface and
is actively maintained. The project runs on Linux; `v4l` would also work, but
`nokhwa` keeps the option open if the hardware ever moves. Add
`features = ["input-native"]` for native platform capture without requiring OpenCV.

---

### `media_file.rs`

Decodes both audio and video streams from a single media file (MP4, MKV, etc.).
The primary use case is testing without a camera: supply a video file and get
the same `AudioFrame` / `VideoFrame` stream types as live capture.

```rust
use std::path::PathBuf;
use crossbeam_channel::Receiver;

/// Decoded streams from a media file. Dropping this stops decoding.
pub struct MediaFileStreams {
    /// Decoded audio frames (f32 PCM, sample rate from the file's audio track).
    pub audio: Receiver<AudioFrame>,
    /// Decoded video frames (Rgb8, resolution from the file's video track).
    pub video: Receiver<VideoFrame>,
    pub audio_metrics: Arc<SensorMetrics>,
    pub video_metrics: Arc<SensorMetrics>,
    _handles: MediaFileHandles, // background thread handles; drop = stop
}

/// Open a media file and start decoding both streams on background threads.
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

    /// Start decoding. Spawns two background threads (audio decoder, video decoder).
    /// Both channels are bounded with drop-oldest backpressure.
    pub fn start(self) -> SensorResult<MediaFileStreams> {
        // implementation uses ffmpeg-next to open the format context,
        // find the best audio and video streams, and decode packets in
        // separate threads → crossbeam senders
        todo!()
    }
}

// Private: keeps decoder thread handles alive.
struct MediaFileHandles {
    _audio_thread: std::thread::JoinHandle<()>,
    _video_thread: std::thread::JoinHandle<()>,
}
```

**Decoding implementation sketch** (inside `MediaFile::start`):

```rust
// Open format context (ffmpeg-next):
let mut ictx = ffmpeg_next::format::input(&self.path)
    .map_err(|e| SensorError::MediaFileOpen { path: self.path.clone(), reason: e.to_string() })?;

// Find best audio / video stream index:
let audio_stream_idx = ictx.streams()
    .best(ffmpeg_next::media::Type::Audio)
    .ok_or(SensorError::MediaFileNoStream("audio"))?
    .index();
let video_stream_idx = ictx.streams()
    .best(ffmpeg_next::media::Type::Video)
    .ok_or(SensorError::MediaFileNoStream("video"))?
    .index();

// Build decoders, spawn one thread per stream, packet-demux loop.
// Audio decoder → resample to f32 if needed → AudioFrame → audio_tx
// Video decoder → convert to Rgb8 via ffmpeg swscale → VideoFrame → video_tx
```

**Crate selection — `ffmpeg-next`:**

Alternatives:
- `video-rs` (higher-level ffmpeg wrapper; audio support is limited)
- `gstreamer` (pipeline model; heavyweight, hard to bundle)
- `symphonia` + `video-rs` (pure Rust; symphonia handles audio; video-rs handles video;
  but they cannot share a single format context for the same file)

`ffmpeg-next` chosen because it exposes both the audio and video streams of a
single file through one format context, handles resampling and pixel-format
conversion via libswresample/libswscale, and supports every common container and
codec. Requires `libavcodec`, `libavformat`, `libavutil`, `libswresample`,
`libswscale` installed on the system (Arch: `pacman -S ffmpeg`).

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

### Cargo.toml sketches

**Workspace root (`Cargo.toml`):**

```toml
[workspace]
members = ["crates/somnus", "crates/somnus-sensory"]
resolver = "2"

[workspace.dependencies]
anyhow            = "1"
cpal              = "0.15"
crossbeam-channel = "0.5"
ffmpeg-next       = "7"
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
somnus-sensory  = { path = "../somnus-sensory" }
anyhow       = { workspace = true }
tokio        = { workspace = true }
tracing      = { workspace = true }
tracing-subscriber = { workspace = true }
```

---

### Binary sketch (`crates/somnus/src/main.rs`)

The binary accepts an optional `--file` argument. If supplied, `MediaFile` provides
both audio and video; an optional `--mic` flag additionally enables live microphone
input. If neither `--file` nor `--mic` is given with a live webcam, it falls back
to live capture.

```rust
use anyhow::Result;
use crossbeam_channel::bounded;
use somnus_sensory::{AudioCapture, AudioFrame, MediaFile, VideoCapture, VideoFrame};
use std::{path::PathBuf, time::Duration};
use tokio::sync::mpsc;
use tracing::info;

#[derive(Debug)]
struct Args {
    file: Option<PathBuf>,
    mic: bool,       // mix in live mic alongside file audio
    webcam: Option<u32>, // None when --file is supplied
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = parse_args(); // simple manual parse or `clap` — TBD at implementation

    // ── audio setup ────────────────────────────────────────────────────────
    // Shared bounded channel: multiple producers can clone the sender.
    let (audio_cb_tx, audio_cb_rx) = bounded::<AudioFrame>(256);
    let mut audio_metric_handles = vec![];

    if let Some(ref path) = args.file {
        let streams = MediaFile::open(path)?.start()?;
        audio_metric_handles.push(streams.audio_metrics.clone());
        let tx = audio_cb_tx.clone();
        std::thread::spawn(move || {
            while let Ok(frame) = streams.audio.recv() {
                if tx.is_full() { let _ = tx.try_recv(); } // drop-oldest reuse
                let _ = tx.try_send(frame);
            }
        });
    }

    if args.mic || args.file.is_none() {
        let mic = AudioCapture::start(None)?;
        audio_metric_handles.push(mic.metrics.clone());
        let tx = audio_cb_tx.clone();
        std::thread::spawn(move || {
            while let Ok(frame) = mic.receiver.recv() {
                if tx.is_full() { let _ = tx.try_recv(); }
                let _ = tx.try_send(frame);
            }
        });
    }

    // ── video setup ────────────────────────────────────────────────────────
    let (video_cb_tx, video_cb_rx) = bounded::<VideoFrame>(16);
    let video_metrics;

    if let Some(ref path) = args.file {
        let streams = MediaFile::open(path)?.start()?;
        video_metrics = streams.video_metrics.clone();
        let tx = video_cb_tx.clone();
        std::thread::spawn(move || {
            while let Ok(frame) = streams.video.recv() {
                if tx.is_full() { let _ = tx.try_recv(); }
                let _ = tx.try_send(frame);
            }
        });
    } else {
        let mut cam = VideoCapture::open(args.webcam.unwrap_or(0))?;
        video_metrics = cam.metrics.clone();
        std::thread::spawn(move || loop {
            match cam.next_frame() {
                Ok(frame) => { if video_cb_tx.is_full() { let _ = video_cb_tx.try_recv(); }
                               let _ = video_cb_tx.try_send(frame); }
                Err(e)    => { tracing::error!(?e, "video capture error"); break; }
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
                tracing::debug!(
                    frames   = frame.frame_count(),
                    duration = frame.duration_secs(),
                    "audio"
                );
            }
            Some(frame) = video_rx.recv() => {
                tracing::debug!(
                    width  = frame.width,
                    height = frame.height,
                    format = ?frame.format,
                    "video"
                );
            }
            _ = metrics_tick.tick() => {
                let v = video_metrics.snapshot();
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
```

**Notes:**

- `MediaFile::open` is called twice when a file is the source (once for audio,
  once for video), because `start()` consumes `self`. Either restructure
  `MediaFile` to yield both streams in a single `start()` call — which is the
  design above via `MediaFileStreams` — or open the file once and split the
  decoders internally. The implementation should take the latter path; the sketch
  above is simplified.
- The drop-oldest reuse in the std::thread spawns is a simplified inline version;
  in implementation use the same `try_send` error-path pattern from `audio.rs`.

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
- Deps: `somnus-perception`, `usearch` (HNSW), `memmap2` (cold tier)
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

**`VideoSource` enum inside `VideoCapture`.** An alternative to `MediaFile` would
be a `VideoSource { Webcam(u32), File(PathBuf) }` enum that `VideoCapture` dispatches
on. Rejected because: (1) it would hide the `nokhwa` vs `ffmpeg-next` backend split
inside one type, making the dependency graph harder to reason about; (2) a video file
also has an audio track, which `VideoCapture` has no business decoding — `MediaFile`
owns both streams cleanly.

**`AudioSource` enum for mic vs file.** Symmetric to the VideoSource alternative.
Rejected for the same reason: mic uses cpal, file uses ffmpeg-next; mixing them in
one enum couples unrelated backends. The fan-in via `Sender::clone` achieves the
same result at the binary level with no coupling in the library.

**Sample-level audio mixing in Phase 1.** True mixing (summing waveforms from mic
and file to produce a single coherent audio stream) requires resampling both sources
to a common rate and synchronizing their timelines. Deferred to Phase 2+: Phase 1
interleaves frames by arrival order, which is sufficient for verifying capture works.

**`tokio::sync::broadcast` for Phase 1 audio.** Broadcast channels discard
messages when a slow reader falls behind rather than dropping the oldest. For Phase 1
with one consumer this doesn't matter. For Phase 2 with multiple consumers at
different speeds, the semantics are wrong. The Phase 2 ADR will revisit.

**`tokio::sync::mpsc` directly in the cpal callback.** Not possible: `mpsc::Sender`
cannot be cloned for the drain path. crossbeam gives both send and drain on the same
channel.

**`ndarray` as the tensor type in `somnus-sensory`.** Rejected: `somnus-sensory` has no
ML computation. Conversion to tensors happens in `somnus-encoding`.

## Consequences

- Phase 1 implementation order: frame → error → metrics → audio → video →
  media_file → binary.
- `ffmpeg-next` requires libavcodec/libavformat/libavutil/libswresample/libswscale
  installed on the build host (`pacman -S ffmpeg` on Arch). This is a hard build
  dependency.
- The crossbeam channel in `AudioCapture` is internal. Phase 2 channel topology
  changes are isolated to `audio.rs` and the bridge code in the binary.
- `nokhwa` is unused in the primary test path (file input). It will first be
  exercised when a webcam is available.

## What would invalidate this

- If the cpal callback's `to_vec()` allocation measurably increases callback
  latency (it shouldn't at typical chunk sizes), replace with a pre-allocated
  ring buffer and zero-copy frames.
- If `nokhwa` fails to open the camera or produces incorrect pixel formats on the
  target hardware, replace with `v4l` (Linux-only but smaller and more direct).
- If Phase 2 encoding proves that multiple simultaneous consumers need different
  drop semantics, redesign the channel topology before Phase 2 ships.
