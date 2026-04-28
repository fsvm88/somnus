use anyhow::Result;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use somnus_sensory::{
    AudioCapture, AudioFrame, MediaFile, MediaFileStreams, SensorMetrics, VideoCapture, VideoFrame,
};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::sync::mpsc;
use tracing::info;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(about = "somnus sensory pipeline — Phase 1")]
struct Args {
    /// Media file to decode (must contain both audio and video streams).
    #[arg(long)]
    file: Option<PathBuf>,

    /// Enable live microphone input alongside file playback.
    #[arg(long)]
    mic: bool,

    /// Webcam device index (ignored when --file is supplied without --mic).
    #[arg(long, default_value = "0")]
    webcam: u32,
}

// ── entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    // ── open file once; destructure both streams immediately ──────────────────
    // `..` drops `_handles` (the JoinHandles), detaching the decode thread.
    // The thread exits when its senders detect Disconnected on shutdown.
    let (file_audio_rx, file_video_rx, file_audio_metrics, file_video_metrics) = match args
        .file
        .as_ref()
        .map(|p| MediaFile::open(p).and_then(|mf| mf.start()))
        .transpose()?
    {
        Some(MediaFileStreams {
            audio,
            video,
            audio_metrics,
            video_metrics,
            ..
        }) => (
            Some(audio),
            Some(video),
            Some(audio_metrics),
            Some(video_metrics),
        ),
        None => (None, None, None, None),
    };

    // ── audio fan-in setup ────────────────────────────────────────────────────
    // Multiple sources (file + mic) fan into the same crossbeam channel by
    // sharing the Sender. Frames from each source are interleaved by arrival.
    let (audio_cb_tx, audio_cb_rx) = bounded::<AudioFrame>(256);
    let audio_drain = audio_cb_rx.clone();
    let mut audio_metric_handles: Vec<Arc<SensorMetrics>> = Vec::new();

    if let Some(rx) = file_audio_rx {
        audio_metric_handles.push(file_audio_metrics.unwrap());
        let tx = audio_cb_tx.clone();
        let drain = audio_drain.clone();
        std::thread::spawn(move || forward_frames(rx, tx, drain));
    }

    // Enable live mic when --mic is set, or when no file is supplied at all.
    if args.mic || args.file.is_none() {
        let mic = AudioCapture::start(None)?;
        audio_metric_handles.push(Arc::clone(&mic.metrics));
        let tx = audio_cb_tx.clone();
        let drain = audio_drain.clone();
        std::thread::spawn(move || {
            // mic dropped at closure end → cpal stream stops.
            forward_frames(mic.receiver, tx, drain);
        });
    }

    // ── video setup ───────────────────────────────────────────────────────────
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
                        if tx.try_send(frame).is_err() {
                            break;
                        }
                    }
                    Err(TrySendError::Disconnected(_)) => break,
                },
                Err(e) => {
                    tracing::error!(?e, "video capture error");
                    break;
                }
            }
        });
    }

    // ── async bridges ─────────────────────────────────────────────────────────
    // Bridges the sync crossbeam channels into tokio mpsc for the async main loop.
    let (audio_tx, mut audio_rx) = mpsc::channel::<AudioFrame>(64);
    tokio::task::spawn_blocking(move || {
        while let Ok(f) = audio_cb_rx.recv() {
            if audio_tx.blocking_send(f).is_err() {
                break;
            }
        }
    });

    let (video_tx, mut video_rx) = mpsc::channel::<VideoFrame>(16);
    tokio::task::spawn_blocking(move || {
        while let Ok(f) = video_cb_rx.recv() {
            if video_tx.blocking_send(f).is_err() {
                break;
            }
        }
    });

    // ── main loop ─────────────────────────────────────────────────────────────
    let mut metrics_tick = tokio::time::interval(Duration::from_secs(10));

    loop {
        tokio::select! {
            Some(frame) = audio_rx.recv() => {
                tracing::debug!(
                    frame_count = frame.frame_count(),
                    duration_secs = frame.duration_secs(),
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
                let v = video_metrics_handle.snapshot();
                info!(
                    video_captured = v.frames_captured,
                    video_dropped   = v.frames_dropped,
                    "video metrics"
                );
                for m in &audio_metric_handles {
                    let a = m.snapshot();
                    info!(
                        audio_captured  = a.frames_captured,
                        audio_drop_rate = format!("{:.1}%", a.drop_rate() * 100.0),
                        "audio metrics"
                    );
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

// ── helpers ───────────────────────────────────────────────────────────────────

/// Forward frames from `src` to `tx` with drop-oldest backpressure via `drain`.
/// Exits cleanly when `tx` is disconnected — the shutdown signal propagating back
/// from the consumer.
fn forward_frames<T>(src: Receiver<T>, tx: Sender<T>, drain: Receiver<T>) {
    while let Ok(frame) = src.recv() {
        match tx.try_send(frame) {
            Ok(()) => {}
            Err(TrySendError::Full(frame)) => {
                let _ = drain.try_recv();
                if tx.try_send(frame).is_err() {
                    break;
                }
            }
            Err(TrySendError::Disconnected(_)) => break,
        }
    }
}
