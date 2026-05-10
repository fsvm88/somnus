mod sleep;

use anyhow::Result;
use candle_core::Device as CandleDevice;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use somnus_encoding::{
    preprocess::resample::SincResampler, AudioEmbedding, AudioEncoder, AudioEncoderSource,
    VideoEmbedding, VideoEncoder,
};
use somnus_memory::{
    load_snapshot, weights_path, ColdArchive, EpisodicBuffer, RecollectionLayer, SemanticStore,
};
use somnus_perception::{PerceptionProcessor, EPISODE_DIM};
use somnus_sensory::{
    AudioCapture, AudioFrame, MediaFile, MediaFileStreams, SensorMetrics, Timestamp, VideoCapture,
    VideoFrame,
};
use std::{
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc;
use tracing::info;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(about = "somnus sensory + encoding + perception + memory + sleep pipeline — Phase 5")]
struct Args {
    #[arg(long)]
    file: Option<PathBuf>,

    #[arg(long)]
    mic: bool,

    #[arg(long, default_value = "0")]
    webcam: u32,

    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    #[arg(long)]
    embeddings_out: Option<PathBuf>,

    /// Maximum number of episodes in the hot episodic buffer.
    #[arg(long, default_value = "4096")]
    episodic_capacity: usize,

    /// Directory for memory files (cold archive, snapshots).
    #[arg(long, default_value = "memory")]
    memory_dir: PathBuf,

    /// Number of nearest episodes to retrieve on each query (logged at debug level).
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Fraction of episodic capacity that triggers a sleep cycle (0.0–1.0).
    #[arg(long, default_value = "0.8")]
    sleep_high_water: f64,

    /// Maximum wake duration in seconds before a sleep cycle is forced.
    #[arg(long, default_value = "600")]
    max_wake_secs: u64,
}

// ── audio accumulator + resampler ─────────────────────────────────────────────

const AUDIO_CHUNK: usize = 4096;
const WHISPER_SAMPLES: usize = 480_000;
const CLAP_SAMPLES: usize = 480_000;

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

        let ch = frame.channels as usize;
        let mono: Vec<f32> = if ch == 1 {
            frame.samples.clone()
        } else {
            (0..frame.frame_count())
                .map(|i| (0..ch).map(|c| frame.samples[i * ch + c]).sum::<f32>() / ch as f32)
                .collect()
        };

        if self.in_rate.is_none() {
            let rate = frame.sample_rate;
            self.r16k = Some(SincResampler::new(rate, 16_000, AUDIO_CHUNK)?);
            self.r48k = Some(SincResampler::new(rate, 48_000, AUDIO_CHUNK)?);
            self.in_rate = Some(rate);
            info!(in_rate = rate, "audio resamplers initialised");
        }

        if self.ts_16k.is_none() {
            self.ts_16k = Some(frame.timestamp.clone());
        }
        if self.ts_48k.is_none() {
            self.ts_48k = Some(frame.timestamp.clone());
        }

        self.src_buf.extend_from_slice(&mono);

        while self.src_buf.len() >= AUDIO_CHUNK {
            let chunk = self.src_buf[..AUDIO_CHUNK].to_vec();
            self.src_buf.drain(..AUDIO_CHUNK);
            self.buf_16k
                .extend_from_slice(&self.r16k.as_mut().unwrap().process(&chunk)?);
            self.buf_48k
                .extend_from_slice(&self.r48k.as_mut().unwrap().process(&chunk)?);
        }

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

    let mut sink: Option<BufWriter<std::fs::File>> = if let Some(path) = &args.embeddings_out {
        let f = std::fs::File::create(path)?;
        info!(path = %path.display(), "embedding sink opened");
        Some(BufWriter::new(f))
    } else {
        None
    };

    // ── perception processor ──────────────────────────────────────────────────
    let mut perception = match PerceptionProcessor::new(CandleDevice::Cpu) {
        Ok(p) => p,
        Err(e) => anyhow::bail!("perception processor init failed: {e}"),
    };
    info!("perception processor initialised (n=1024, k=20, cpu)");

    // ── memory hierarchy ──────────────────────────────────────────────────────
    std::fs::create_dir_all(&args.memory_dir)?;
    let cold = match ColdArchive::open(&args.memory_dir) {
        Ok(c) => c,
        Err(e) => anyhow::bail!("cold archive init failed: {e}"),
    };

    // Restore from snapshot if available; otherwise start fresh.
    let (mut episodic, semantic) =
        match load_snapshot(&args.memory_dir, args.episodic_capacity, EPISODE_DIM) {
            Ok(Some((ep, sem))) => {
                info!(
                    episodic_len = ep.len(),
                    semantic_len = sem.len(),
                    "restored from snapshot"
                );
                // Restore perception weights if present.
                let w_path = weights_path(&args.memory_dir);
                if w_path.exists() {
                    if let Err(e) = perception.load_weights(&w_path) {
                        tracing::warn!(?e, "weight restore failed — starting with fresh weights");
                    } else {
                        info!("perception weights restored");
                    }
                }
                (ep, sem)
            }
            Ok(None) => {
                info!("no snapshot found — starting fresh");
                (
                    EpisodicBuffer::new(args.episodic_capacity),
                    SemanticStore::new(EPISODE_DIM),
                )
            }
            Err(e) => {
                tracing::warn!(?e, "snapshot load failed — starting fresh");
                (
                    EpisodicBuffer::new(args.episodic_capacity),
                    SemanticStore::new(EPISODE_DIM),
                )
            }
        };

    let mut recall = RecollectionLayer::new(semantic, cold);
    info!(
        episodic_capacity = args.episodic_capacity,
        memory_dir = %args.memory_dir.display(),
        "memory hierarchy ready"
    );

    let high_water = (args.episodic_capacity as f64 * args.sleep_high_water) as usize;
    let max_wake = Duration::from_secs(args.max_wake_secs);
    let mut wake_start = std::time::Instant::now();

    let interrupt = Arc::new(AtomicBool::new(false));
    let interrupt_sleep = Arc::clone(&interrupt);

    // ── open file once; destructure both streams ──────────────────────────────
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

    // ── async bridges ─────────────────────────────────────────────────────────
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

    // ── main loop ─────────────────────────────────────────────────────────────
    let mut metrics_tick = tokio::time::interval(Duration::from_secs(10));

    loop {
        // ── sleep trigger check ───────────────────────────────────────────────
        let should_sleep = episodic.len() >= high_water || wake_start.elapsed() >= max_wake;

        if should_sleep && !episodic.is_empty() {
            interrupt.store(false, Ordering::Relaxed);
            match sleep::run_sleep_cycle(
                &mut perception,
                &mut episodic,
                &mut recall,
                &args.memory_dir,
                Arc::clone(&interrupt_sleep),
            ) {
                Ok(stats) => info!(
                    traces_replayed = stats.traces_replayed,
                    mean_loss = stats.mean_train_loss,
                    interrupted = stats.was_interrupted,
                    "sleep cycle complete"
                ),
                Err(e) => tracing::warn!(?e, "sleep cycle error"),
            }
            wake_start = std::time::Instant::now();
        }

        tokio::select! {
            Some(emb) = audio_emb_rx.recv() => {
                sink_write(&mut sink, &audio_emb_json(&emb));
                let result = match emb.source {
                    AudioEncoderSource::Speech => perception.ingest_speech(&emb.values),
                    AudioEncoderSource::Env    => perception.ingest_env(&emb.values),
                };
                match result {
                    Ok(out) => record_episode(
                        &out, &emb.timestamp,
                        &mut sink, &mut episodic,
                        args.top_k, &recall,
                    ),
                    Err(e)  => tracing::error!(?e, "perception error (audio)"),
                }
            }
            Some(emb) = video_emb_rx.recv() => {
                sink_write(&mut sink, &video_emb_json(&emb));
                match perception.ingest_visual(&emb.values) {
                    Ok(out) => record_episode(
                        &out, &emb.timestamp,
                        &mut sink, &mut episodic,
                        args.top_k, &recall,
                    ),
                    Err(e) => tracing::error!(?e, "perception error (video)"),
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
                info!(
                    episodic_len = episodic.len(),
                    semantic_len = recall.warm.len(),
                    cold_len     = recall.cold.len(),
                    "memory metrics"
                );
            }
            _ = tokio::signal::ctrl_c() => {
                interrupt.store(true, Ordering::Relaxed);
                sink_flush(&mut sink);
                info!("shutting down");
                break;
            }
        }
    }

    Ok(())
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Push an episode into the hot buffer (without touching warm tier — sleep does that).
fn record_episode(
    out: &somnus_perception::PerceptionOutput,
    ts: &Timestamp,
    sink: &mut Option<BufWriter<std::fs::File>>,
    episodic: &mut EpisodicBuffer,
    top_k: usize,
    recall: &RecollectionLayer,
) {
    for &(region, err) in &out.recon_errors {
        tracing::debug!(region = ?region, recon_mse = err, "reconstruction error");
    }
    if let Some(ep) = &out.episode {
        let ts_secs = wall_secs(&ts.wall);
        tracing::info!(dim = ep.0.len(), "episode vector");

        let regional = out
            .regional_inputs
            .clone()
            .unwrap_or([vec![], vec![], vec![], vec![]]);

        let id = episodic.push(ts_secs, ep.0.clone(), regional);

        if tracing::enabled!(tracing::Level::DEBUG) {
            let neighbours = recall.query_warm(&ep.0, top_k);
            tracing::debug!(
                episode_id = id,
                retrieved = neighbours.len(),
                top1_id = neighbours.first().map(|r| r.id),
                "memory retrieval"
            );
        }

        let line = serde_json::json!({
            "type":           "episode",
            "id":             id,
            "timestamp_secs": ts_secs,
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
