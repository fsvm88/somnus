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

## Decision

### Module layout (`crates/somnus-sensory/src/`)

```
lib.rs        re-exports: AudioFrame, VideoFrame, Timestamp, PixelFormat,
              AudioCapture, VideoCapture, SensorError, SensorResult,
              SensorMetrics, MetricsSnapshot
frame.rs      Timestamp, AudioFrame, VideoFrame, PixelFormat
audio.rs      AudioCapture, AUDIO_CHANNEL_CAPACITY
video.rs      VideoCapture, VIDEO_CHANNEL_CAPACITY
error.rs      SensorError, SensorResult
metrics.rs    SensorMetrics, MetricsSnapshot
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

---

### `video.rs`

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

### Channel topology

```
Phase 1:
                                               consumer
  [cpal callback]──crossbeam::bounded(256)──► [tokio::task::spawn_blocking bridge]
                                                         │
                                                         ▼
                                               tokio::sync::mpsc::Receiver<AudioFrame>
                                                         │
                                                         ▼
                                                [async audio consumer task]

  [video thread]──────────────────────────────► VideoCapture::next_frame() (blocking pull)
                                                         │
                                                tokio::task::spawn_blocking
                                                         │
                                                         ▼
                                                [async video consumer task]

Phase 2 change:
  - Add a fanout bridge: crossbeam → tokio::sync::broadcast<AudioFrame>
  - somnus-encoding subscribes; somnus-perception subscribes (once it exists)
  - VideoCapture wrapped in a dedicated thread → crossbeam channel → broadcast
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
anyhow       = "1"
cpal         = "0.15"
crossbeam-channel = "0.5"
nokhwa       = { version = "0.10", features = ["input-native"] }
thiserror    = "2"
tokio        = { version = "1", features = ["full"] }
tracing      = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
```

**`crates/somnus-sensory/Cargo.toml`:**

```toml
[package]
name    = "somnus-sensory"
version = "0.1.0"
edition = "2021"

[dependencies]
cpal             = { workspace = true }
crossbeam-channel = { workspace = true }
nokhwa           = { workspace = true }
thiserror        = { workspace = true }
tracing          = { workspace = true }
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

```rust
use anyhow::Result;
use somnus_sensory::{AudioCapture, VideoCapture, MetricsSnapshot};
use tokio::sync::mpsc;
use std::time::Duration;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let audio = AudioCapture::start(None)?;
    let mut video = VideoCapture::open(0)?;

    let audio_metrics = audio.metrics.clone();
    let video_metrics = video.metrics.clone();

    // Bridge audio: sync crossbeam → async tokio mpsc
    let (audio_tx, mut audio_rx) = mpsc::channel(64);
    tokio::task::spawn_blocking(move || {
        while let Ok(frame) = audio.receiver.recv() {
            if audio_tx.blocking_send(frame).is_err() {
                break;
            }
        }
    });

    // Bridge video: blocking pull → async tokio mpsc
    let (video_tx, mut video_rx) = mpsc::channel(16);
    tokio::task::spawn_blocking(move || loop {
        match video.next_frame() {
            Ok(frame) => {
                if video_tx.blocking_send(frame).is_err() {
                    break;
                }
            }
            Err(e) => {
                tracing::error!(?e, "video capture error");
                break;
            }
        }
    });

    // Periodic metrics log
    let mut metrics_tick = tokio::time::interval(Duration::from_secs(10));

    loop {
        tokio::select! {
            Some(frame) = audio_rx.recv() => {
                tracing::debug!(
                    frames = frame.frame_count(),
                    duration_ms = frame.duration_secs() * 1000.0,
                    "audio"
                );
            }
            Some(frame) = video_rx.recv() => {
                tracing::debug!(
                    width = frame.width,
                    height = frame.height,
                    format = ?frame.format,
                    "video"
                );
            }
            _ = metrics_tick.tick() => {
                let a = audio_metrics.snapshot();
                let v = video_metrics.snapshot();
                info!(
                    audio_captured = a.frames_captured,
                    audio_dropped = a.frames_dropped,
                    audio_drop_rate = format!("{:.2}%", a.drop_rate() * 100.0),
                    video_captured = v.frames_captured,
                    "metrics"
                );
            }
        }
    }
}
```

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

**`tokio::sync::broadcast` for Phase 1 audio.** Broadcast channels discard
messages when a slow reader falls behind rather than dropping the oldest. For
Phase 1 with one consumer, this doesn't matter. For Phase 2 with multiple
consumers at different speeds, the semantics are wrong (a slow encoder would
cause frame loss for everyone). The crossbeam-bounded + explicit drain approach
gives controlled drop-oldest semantics independent of consumer speed. The Phase 2
ADR will revisit whether a fanout broadcast is acceptable or whether each consumer
gets its own bounded channel.

**`tokio::sync::mpsc` directly in the cpal callback.** The callback is not in an
async context; `mpsc::Sender::try_send` is available but `mpsc::Sender` cannot be
cloned into both the callback (for sending) and the drain path (for drop-oldest).
crossbeam gives both.

**`ndarray` as the tensor type in `somnus-sensory`.** Rejected: `somnus-sensory` has no
ML computation. It captures raw bytes and f32 samples. Introducing a tensor
dependency here would couple the sensory layer to the ML stack unnecessarily.
Conversion to tensors happens in `somnus-encoding`.

## Consequences

- Phase 1 implementation can begin directly: create the workspace, initialize the
  two crates, implement types in the order defined above (frame → error → metrics
  → audio → video → binary).
- The crossbeam channel in `AudioCapture` is an internal detail. If Phase 2
  requires a different channel topology (broadcast, multiple consumers), the
  change is isolated to `audio.rs` and the bridge code in the binary.
- `nokhwa` is a non-trivial dependency. If it causes problems on the target
  hardware, fall back to the `v4l` crate with minimal API changes.

## What would invalidate this

- If the cpal callback's `to_vec()` allocation measurably increases callback
  latency (it shouldn't at typical chunk sizes), replace with a pre-allocated
  ring buffer and zero-copy frames.
- If `nokhwa` fails to open the camera or produces incorrect pixel formats on the
  target hardware, replace with `v4l` (Linux-only but smaller and more direct).
- If Phase 2 encoding proves that multiple simultaneous consumers need different
  drop semantics, redesign the channel topology before Phase 2 ships.
