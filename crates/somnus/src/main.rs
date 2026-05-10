use anyhow::Result;
use candle_core::Device as CandleDevice;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use somnus_encoding::{
    preprocess::resample::SincResampler, AudioEmbedding, AudioEncoder, AudioEncoderSource,
    VideoEmbedding, VideoEncoder,
};
use somnus_perception::PerceptionProcessor;
use somnus_sensory::{
    AudioCapture, AudioFrame, MediaFile, MediaFileStreams, SensorMetrics, Timestamp, VideoCapture,
    VideoFrame,
};
use std::{
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc;
use tracing::info;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(about = "somnus sensory + encoding + perception pipeline — Phase 3")]
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

    /// Directory containing ONNX model files.
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    /// Write embeddings as newline-delimited JSON to this file.
    #[arg(long)]
    embeddings_out: Option<PathBuf>,
}

// ── audio accumulator + resampler ─────────────────────────────────────────────

const AUDIO_CHUNK: usize = 4096; // rubato fixed input chunk
const WHISPER_SAMPLES: usize = 480_000; // 30 s @ 16 kHz
const CLAP_SAMPLES: usize = 480_000; // 10 s @ 48 kHz

struct AudioProcessor {
    encoder: AudioEncoder,
    r16k: Option<SincResampler>,
    r48k: Option<SincResampler>,
    in_rate: Option<u32>,
    src_buf: Vec<f32>,
    buf_16k: Vec<f32>,
    buf_48k: Vec<f32>,
    ts_16k: Option<Timestamp>,
    ts_48k: Option<Timestamp>,
}

impl AudioProcessor {
    fn new(models_dir: &std::path::Path) -> Result<Self> {
        Ok(Self {
            encoder: AudioEncoder::new(models_dir)?,
            r16k: None,
            r48k: None,
            in_rate: None,
            src_buf: Vec::new(),
            buf_16k: Vec::new(),
            buf_48k: Vec::new(),
            ts_16k: None,
            ts_48k: None,
        })
    }

    fn ingest(&mut self, frame: AudioFrame) -> Result<Vec<AudioEmbedding>> {
        // Reject mid-stream rate change — resamplers are fixed at init.
        if let Some(known) = self.in_rate {
            if frame.sample_rate != known {
                tracing::warn!(
                    got = frame.sample_rate,
                    expected = known,
                    "audio sample rate changed mid-stream — frame dropped"
                );
                return Ok(vec![]);
            }
        }

        // Downmix to mono.
        let ch = frame.channels as usize;
        let mono: Vec<f32> = if ch == 1 {
            frame.samples.clone()
        } else {
            (0..frame.frame_count())
                .map(|i| (0..ch).map(|c| frame.samples[i * ch + c]).sum::<f32>() / ch as f32)
                .collect()
        };

        // Lazy-init resamplers on first frame.
        if self.in_rate.is_none() {
            let rate = frame.sample_rate;
            self.r16k = Some(SincResampler::new(rate, 16_000, AUDIO_CHUNK)?);
            self.r48k = Some(SincResampler::new(rate, 48_000, AUDIO_CHUNK)?);
            self.in_rate = Some(rate);
            info!(in_rate = rate, "audio resamplers initialised");
        }

        // Timestamp of the oldest unencoded sample in each output buffer.
        if self.ts_16k.is_none() {
            self.ts_16k = Some(frame.timestamp.clone());
        }
        if self.ts_48k.is_none() {
            self.ts_48k = Some(frame.timestamp.clone());
        }

        self.src_buf.extend_from_slice(&mono);

        // Resample in fixed-size chunks.
        while self.src_buf.len() >= AUDIO_CHUNK {
            let chunk = self.src_buf[..AUDIO_CHUNK].to_vec();
            self.src_buf.drain(..AUDIO_CHUNK);
            self.buf_16k
                .extend_from_slice(&self.r16k.as_mut().unwrap().process(&chunk)?);
            self.buf_48k
                .extend_from_slice(&self.r48k.as_mut().unwrap().process(&chunk)?);
        }

        // Emit embeddings when output buffers are full.
        let mut embs = Vec::new();

        if self.buf_16k.len() >= WHISPER_SAMPLES {
            let pcm: Vec<f32> = self.buf_16k.drain(..WHISPER_SAMPLES).collect();
            let ts = self.ts_16k.take().unwrap();
            let start = std::time::Instant::now();
            let emb = self.encoder.encode_speech(&pcm, ts)?;
            tracing::info!(
                dim = emb.values.len(),
                encode_ms = start.elapsed().as_millis(),
                "speech embedding"
            );
            embs.push(emb);
        }

        if self.buf_48k.len() >= CLAP_SAMPLES {
            let pcm: Vec<f32> = self.buf_48k.drain(..CLAP_SAMPLES).collect();
            let ts = self.ts_48k.take().unwrap();
            let start = std::time::Instant::now();
            let emb = self.encoder.encode_env(&pcm, ts)?;
            tracing::info!(
                dim = emb.values.len(),
                encode_ms = start.elapsed().as_millis(),
                "env embedding"
            );
            embs.push(emb);
        }

        Ok(embs)
    }
}

// ── NDJSON sink helpers ───────────────────────────────────────────────────────

fn wall_secs(t: &SystemTime) -> f64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn sink_write(sink: &mut Option<BufWriter<std::fs::File>>, line: &str) {
    if let Some(w) = sink {
        if let Err(e) = writeln!(w, "{}", line) {
            tracing::warn!(?e, "embedding sink write failed");
        }
    }
}

fn sink_flush(sink: &mut Option<BufWriter<std::fs::File>>) {
    if let Some(w) = sink {
        if let Err(e) = w.flush() {
            tracing::warn!(?e, "embedding sink flush failed");
        }
    }
}

fn audio_emb_json(emb: &AudioEmbedding) -> String {
    let kind = match emb.source {
        AudioEncoderSource::Speech => "speech",
        AudioEncoderSource::Env => "env",
    };
    serde_json::json!({
        "type":           kind,
        "timestamp_secs": wall_secs(&emb.timestamp.wall),
        "dim":            emb.values.len(),
        "values":         emb.values,
    })
    .to_string()
}

fn video_emb_json(emb: &VideoEmbedding) -> String {
    serde_json::json!({
        "type":           "video",
        "timestamp_secs": wall_secs(&emb.timestamp.wall),
        "dim":            emb.values.len(),
        "values":         emb.values,
    })
    .to_string()
}

// ── entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    ort::init_from("/usr/lib/libonnxruntime.so")?.commit();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    // Verify model files are present before constructing sessions.
    for name in &[
        "whisper_tiny_encoder.onnx",
        "clap_audio_encoder.onnx",
        "dinov2_small.onnx",
    ] {
        let p = args.models_dir.join(name);
        if !p.exists() {
            anyhow::bail!("model file not found: {}", p.display());
        }
    }

    // Open embedding sink if requested.
    let mut sink: Option<BufWriter<std::fs::File>> = if let Some(path) = &args.embeddings_out {
        let f = std::fs::File::create(path)?;
        info!(path = %path.display(), "embedding sink opened");
        Some(BufWriter::new(f))
    } else {
        None
    };

    // ── open file once; destructure both streams immediately ──────────────────
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

    // ── audio fan-in ──────────────────────────────────────────────────────────
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

    // ── async bridges: crossbeam → tokio mpsc ─────────────────────────────────
    let (audio_tx, audio_rx) = mpsc::channel::<AudioFrame>(64);
    tokio::task::spawn_blocking(move || {
        while let Ok(f) = audio_cb_rx.recv() {
            if audio_tx.blocking_send(f).is_err() {
                break;
            }
        }
    });

    let (video_tx, video_rx) = mpsc::channel::<VideoFrame>(16);
    tokio::task::spawn_blocking(move || {
        while let Ok(f) = video_cb_rx.recv() {
            if video_tx.blocking_send(f).is_err() {
                break;
            }
        }
    });

    // ── audio encoder task ────────────────────────────────────────────────────
    let (audio_emb_tx, mut audio_emb_rx) = mpsc::channel::<AudioEmbedding>(32);
    {
        let models_dir = args.models_dir.clone();
        let mut audio_rx = audio_rx;
        tokio::task::spawn_blocking(move || {
            let mut proc = match AudioProcessor::new(&models_dir) {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!(?e, "audio encoder init failed");
                    return;
                }
            };
            while let Some(frame) = audio_rx.blocking_recv() {
                match proc.ingest(frame) {
                    Ok(embs) => {
                        for emb in embs {
                            if audio_emb_tx.blocking_send(emb).is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) => tracing::error!(?e, "audio encode error"),
                }
            }
        });
    }

    // ── video encoder task ────────────────────────────────────────────────────
    let (video_emb_tx, mut video_emb_rx) = mpsc::channel::<VideoEmbedding>(16);
    {
        let models_dir = args.models_dir.clone();
        let mut video_rx = video_rx;
        tokio::task::spawn_blocking(move || {
            let mut enc = match VideoEncoder::new(&models_dir) {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!(?e, "video encoder init failed");
                    return;
                }
            };
            let mut last_log = std::time::Instant::now();
            let mut frames_this_sec: u32 = 0;
            let mut total_ms_this_sec: u128 = 0;
            while let Some(frame) = video_rx.blocking_recv() {
                let start = std::time::Instant::now();
                match enc.encode(&frame) {
                    Ok(emb) => {
                        let encode_ms = start.elapsed().as_millis();
                        frames_this_sec += 1;
                        total_ms_this_sec += encode_ms;
                        tracing::debug!(dim = emb.values.len(), encode_ms, "video embedding");
                        if last_log.elapsed() >= Duration::from_secs(1) {
                            let avg_ms = total_ms_this_sec / frames_this_sec.max(1) as u128;
                            tracing::info!(
                                frames = frames_this_sec,
                                dim = emb.values.len(),
                                avg_encode_ms = avg_ms,
                                "video embeddings/s"
                            );
                            last_log = std::time::Instant::now();
                            frames_this_sec = 0;
                            total_ms_this_sec = 0;
                        }
                        if video_emb_tx.blocking_send(emb).is_err() {
                            return;
                        }
                    }
                    Err(e) => tracing::error!(?e, "video encode error"),
                }
            }
        });
    }

    // ── perception processor ──────────────────────────────────────────────────
    let mut perception = match PerceptionProcessor::new(CandleDevice::Cpu) {
        Ok(p) => p,
        Err(e) => anyhow::bail!("perception processor init failed: {e}"),
    };
    info!("perception processor initialised (n=1024, k=20, cpu)");

    // ── main loop ─────────────────────────────────────────────────────────────
    let mut metrics_tick = tokio::time::interval(Duration::from_secs(10));

    loop {
        tokio::select! {
            Some(emb) = audio_emb_rx.recv() => {
                sink_write(&mut sink, &audio_emb_json(&emb));
                let result = match emb.source {
                    AudioEncoderSource::Speech => perception.ingest_speech(&emb.values),
                    AudioEncoderSource::Env    => perception.ingest_env(&emb.values),
                };
                match result {
                    Ok(out) => log_perception_output(&out, &emb.timestamp, &mut sink),
                    Err(e)  => tracing::error!(?e, "perception error (audio)"),
                }
            }
            Some(emb) = video_emb_rx.recv() => {
                sink_write(&mut sink, &video_emb_json(&emb));
                match perception.ingest_visual(&emb.values) {
                    Ok(out) => log_perception_output(&out, &emb.timestamp, &mut sink),
                    Err(e)  => tracing::error!(?e, "perception error (video)"),
                }
            }
            _ = metrics_tick.tick() => {
                sink_flush(&mut sink);
                let v = video_metrics_handle.snapshot();
                info!(
                    video_captured = v.frames_captured,
                    video_dropped  = v.frames_dropped,
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
                sink_flush(&mut sink);
                info!("shutting down");
                break;
            }
        }
    }

    Ok(())
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn log_perception_output(
    out: &somnus_perception::PerceptionOutput,
    ts: &Timestamp,
    sink: &mut Option<BufWriter<std::fs::File>>,
) {
    for &(region, err) in &out.recon_errors {
        tracing::debug!(region = ?region, recon_mse = err, "reconstruction error");
    }
    if let Some(ep) = &out.episode {
        tracing::info!(dim = ep.0.len(), "episode vector");
        let line = serde_json::json!({
            "type":           "episode",
            "timestamp_secs": wall_secs(&ts.wall),
            "dim":            ep.0.len(),
            "values":         ep.0,
        })
        .to_string();
        sink_write(sink, &line);
    }
}

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
