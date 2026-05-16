mod action;
mod sleep;

use action::ActionDecoder;

use anyhow::Result;
use candle_core::Device as CandleDevice;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use somnus_encoding::{
    preprocess::resample::SincResampler, AudioEmbedding, AudioEncoder, AudioEncoderSource,
    VideoEmbedding, VideoEncoder,
};
use somnus_memory::{
    load_region_sizes, load_snapshot, weights_path, ColdArchive, EpisodicBuffer, RecollectionLayer,
    SemanticStore,
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
        atomic::{AtomicBool, AtomicUsize, Ordering},
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

    /// Top-down prior blend weight β (0.0 = disabled). Enable only after
    /// output_mse has converged; random heads inject noise if enabled too early.
    #[arg(long, default_value = "0.0")]
    prior_beta: f32,

    /// Number of settling iterations per episode (0 = disabled). Enable only
    /// after output_mse has converged; random heads distort settling inputs (ADR 0011).
    #[arg(long, default_value = "0")]
    settle_steps: usize,

    /// Settling blend weight β: fraction of output-head prediction mixed into
    /// settling inputs. 0.0 = raw only (no-op loop); 1.0 = pure prediction.
    #[arg(long, default_value = "0.5")]
    settle_beta: f32,

    /// Enable ActionDecoder: per-episode LM forward for gain modulation + stdin generation (ADR 0012).
    /// Requires qwen25_lm.onnx, qwen25_embed_table.safetensors, qwen25_tokenizer.json in --models-dir.
    #[arg(long, default_value = "false")]
    action: bool,

    /// Enable per-region gain modulation from LM attention (ADR 0012).
    /// Only effective when --action is set.
    #[arg(long, default_value = "true")]
    action_gain: bool,

    /// Maximum tokens to generate per query (ADR 0012).
    #[arg(long, default_value = "128")]
    action_max_tokens: usize,

    /// Loop the input file indefinitely. Restarts from the beginning each time
    /// the file ends, keeping warm memory alive across iterations. Useful for
    /// calibration runs where a single file is too short to trigger multiple
    /// sleep cycles.
    #[arg(long, default_value = "false")]
    file_loop: bool,
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
        .with_writer(std::io::stderr)
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
    perception.set_prior_beta(args.prior_beta);
    perception.set_settle_steps(args.settle_steps);
    perception.set_settle_beta(args.settle_beta);
    info!(
        prior_beta = args.prior_beta,
        settle_steps = args.settle_steps,
        settle_beta = args.settle_beta,
        "perception processor initialised (n=1024, k=20, cpu)"
    );

    // ── memory hierarchy ──────────────────────────────────────────────────────
    std::fs::create_dir_all(&args.memory_dir)?;
    let cold = match ColdArchive::open(&args.memory_dir) {
        Ok(c) => c,
        Err(e) => anyhow::bail!("cold archive init failed: {e}"),
    };

    // Restore from snapshot if available; otherwise start fresh.
    let mut snapshot_found = false;
    let (mut episodic, semantic) =
        match load_snapshot(&args.memory_dir, args.episodic_capacity, EPISODE_DIM) {
            Ok(Some((ep, sem))) => {
                snapshot_found = true;
                info!(
                    episodic_len = ep.len(),
                    semantic_len = sem.len(),
                    "restored from snapshot"
                );
                // Restore perception weights; pre-grow regions if they grew last session.
                let w_path = weights_path(&args.memory_dir);
                if w_path.exists() {
                    let sizes =
                        load_region_sizes(&args.memory_dir).unwrap_or([somnus_perception::N_UNITS; 5]);
                    if let Err(e) = perception.load_weights_with_sizes(&w_path, sizes) {
                        tracing::warn!(?e, "weight restore failed — starting with fresh weights");
                    } else {
                        info!(sizes = ?sizes, "perception weights restored");
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

    if args.settle_steps > 0 && !snapshot_found {
        tracing::warn!(
            settle_steps = args.settle_steps,
            "settling active but no prior snapshot — output heads are random; \
             settle noise expected until output_mse converges"
        );
    }

    // ── action decoder ────────────────────────────────────────────────────────
    let mut decoder: Option<ActionDecoder> = if args.action {
        match ActionDecoder::new(&args.models_dir, CandleDevice::Cpu) {
            Ok(mut d) => {
                let dec_w = args.memory_dir.join("snapshot").join("decoder_weights.safetensors");
                if dec_w.exists() {
                    if let Err(e) = d.load_weights(&dec_w) {
                        tracing::warn!(?e, "decoder weight restore failed — starting fresh");
                    } else {
                        info!("decoder projection weights restored");
                    }
                } else if snapshot_found {
                    tracing::warn!(
                        "snapshot found but no decoder_weights.safetensors — \
                         projection starts fresh; generation incoherent until trained"
                    );
                }
                info!("action decoder ready");
                Some(d)
            }
            Err(e) => {
                tracing::warn!(?e, "action decoder init failed — decoder disabled");
                None
            }
        }
    } else {
        None
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

    // Counts embeddings produced by encoder tasks but not yet consumed by the main loop.
    // Sleep monitors this to detect external salience while the system is consolidating.
    let pending_embs = Arc::new(AtomicUsize::new(0));

    // ── stdin query channel (ADR 0012) ────────────────────────────────────────
    // Each line on stdin is a query: triggers generation + stores text_target on current episode.
    let (query_tx, mut query_rx) = tokio::sync::mpsc::channel::<String>(8);
    if decoder.is_some() {
        let tx = query_tx;
        tokio::task::spawn_blocking(move || {
            use std::io::BufRead;
            for line in std::io::stdin().lock().lines().map_while(Result::ok) {
                let trimmed = line.trim().to_string();
                if !trimmed.is_empty() && tx.blocking_send(trimmed).is_err() {
                    break;
                }
            }
        });
    }

    // ── audio fan-in ──────────────────────────────────────────────────────────
    let (audio_cb_tx, audio_cb_rx) = bounded::<AudioFrame>(256);
    let audio_drain = audio_cb_rx.clone();
    let mut audio_metric_handles: Vec<Arc<SensorMetrics>> = Vec::new();

    // ── video setup ───────────────────────────────────────────────────────────
    let (video_cb_tx, video_cb_rx) = bounded::<VideoFrame>(16);
    let video_drain = video_cb_rx.clone();
    let mut video_metrics_handle: Arc<SensorMetrics>;

    if let Some(ref file_path) = args.file {
        if args.file_loop {
            // Looping file source: restart from the beginning each time the
            // file ends, keeping the same downstream channels (and therefore
            // warm memory) alive across iterations.
            let path = file_path.clone();
            let atx = audio_cb_tx.clone();
            let vtx = video_cb_tx.clone();
            std::thread::spawn(move || loop {
                let streams = match MediaFile::open(&path).and_then(|mf| mf.start()) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!(?e, "file loop: failed to (re)open file");
                        break;
                    }
                };
                let atx2 = atx.clone();
                let vtx2 = vtx.clone();
                let ah = std::thread::spawn(move || forward_frames_blocking(streams.audio, atx2));
                let vh = std::thread::spawn(move || forward_frames_blocking(streams.video, vtx2));
                let a_normal = ah.join().unwrap_or(false);
                let v_normal = vh.join().unwrap_or(false);
                // If downstream closed, stop looping.
                if !a_normal || !v_normal {
                    break;
                }
                tracing::info!("file loop: restarting from beginning");
            });
            // Use a dummy metrics handle for loop mode (per-iteration metrics not tracked).
            video_metrics_handle = Arc::new(SensorMetrics::default());
        } else {
            // One-shot file source.
            let MediaFileStreams {
                audio,
                video,
                audio_metrics,
                video_metrics,
                ..
            } = MediaFile::open(file_path).and_then(|mf| mf.start())?;
            audio_metric_handles.push(audio_metrics);
            video_metrics_handle = video_metrics;
            let atx = audio_cb_tx.clone();
            let vtx = video_cb_tx.clone();
            std::thread::spawn(move || forward_frames_blocking(audio, atx));
            std::thread::spawn(move || forward_frames_blocking(video, vtx));
        }
    } else {
        // No file — fall through to mic/webcam paths below.
        video_metrics_handle = Arc::new(SensorMetrics::default()); // placeholder, overwritten below
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

    if args.file.is_none() {
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

    // All source threads have their own clones of the cb senders. Drop the
    // originals now so that when every source thread finishes (e.g. file EOF),
    // the bridge tasks below see channel closure and exit cleanly.
    drop(audio_cb_tx);
    drop(video_cb_tx);

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
        let pending = Arc::clone(&pending_embs);
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
                            if audio_emb_tx.blocking_send(emb).is_ok() {
                                pending.fetch_add(1, Ordering::Relaxed);
                            } else {
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
        let pending = Arc::clone(&pending_embs);
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
                        if video_emb_tx.blocking_send(emb).is_ok() {
                            pending.fetch_add(1, Ordering::Relaxed);
                        } else {
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
    // In file (non-loop) mode the embedding channels close when the file is
    // exhausted.  Track closure so we can exit cleanly without Ctrl-C.
    let file_mode = args.file.is_some() && !args.file_loop;
    let mut audio_emb_closed = false;
    let mut video_emb_closed = false;

    loop {
        // ── drain any pending queries before potentially sleeping ─────────────
        // This ensures queries are never blocked behind a sleep cycle.
        while let Ok(query) = query_rx.try_recv() {
            tracing::info!(query = %query, "query received (pre-sleep drain)");
            let ep_vec = episodic.iter_traces().last().map(|t| t.record.values.clone());
            match ep_vec {
                None => {
                    println!("{}", serde_json::json!({"type":"action_response","error":"no episode yet"}));
                }
                Some(ep) => {
                    if let Some(ref mut dec) = decoder {
                        if let Ok(ids) = dec.tokenize(&query) {
                            episodic.set_last_text_target(ids);
                        }
                        tracing::info!("starting generation (pre-sleep)");
                        let ep_clone = ep.clone();
                        let query_clone = query.clone();
                        let max_tokens = args.action_max_tokens;
                        let result = tokio::task::block_in_place(|| {
                            dec.generate(&ep_clone, &query_clone, max_tokens)
                        });
                        match result {
                            Ok(response) => {
                                let ts_ms = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .map(|d| d.as_millis())
                                    .unwrap_or(0);
                                tracing::info!(tokens = response.split_whitespace().count(), "generation complete");
                                println!("{}", serde_json::json!({
                                    "type": "action_response",
                                    "ts": ts_ms,
                                    "query": query,
                                    "response": response,
                                }));
                            }
                            Err(e) => tracing::warn!(?e, "generation failed"),
                        }
                    }
                }
            }
        }

        // ── sleep trigger check ───────────────────────────────────────────────
        let should_sleep = episodic.len() >= high_water || wake_start.elapsed() >= max_wake;

        if should_sleep && !episodic.is_empty() {
            interrupt.store(false, Ordering::Relaxed);
            pending_embs.store(0, Ordering::Relaxed);
            perception.reset_region_gain(); // stale gain must not carry across sleep boundary
            match sleep::run_sleep_cycle(
                &mut perception,
                &mut episodic,
                &mut recall,
                &args.memory_dir,
                Arc::clone(&interrupt_sleep),
                Arc::clone(&pending_embs),
                &mut decoder,
            ) {
                Ok(stats) => {
                    info!(
                        traces_replayed = stats.traces_replayed,
                        mean_loss = stats.mean_train_loss,
                        mean_output_mse = stats.mean_output_mse,
                        warm_recall_l2 = ?stats.warm_recall_l2,
                        mean_proj_loss = ?stats.mean_proj_loss,
                        decoder_trainable = stats.decoder_trainable_records,
                        interrupted = stats.was_interrupted,
                        "sleep cycle complete"
                    );
                    // Save decoder projection weights alongside perception weights.
                    if let Some(ref dec) = decoder {
                        let dec_w = args.memory_dir.join("snapshot").join("decoder_weights.safetensors");
                        if let Err(e) = dec.save_weights(&dec_w) {
                            tracing::warn!(?e, "decoder weights snapshot failed");
                        }
                    }
                }
                Err(e) => tracing::warn!(?e, "sleep cycle error"),
            }
            wake_start = std::time::Instant::now();
        }

        tokio::select! {
            maybe = audio_emb_rx.recv(), if !audio_emb_closed => {
                match maybe {
                    Some(emb) => {
                        let _ = pending_embs.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| Some(x.saturating_sub(1)));
                        sink_write(&mut sink, &audio_emb_json(&emb));
                        let result = match emb.source {
                            AudioEncoderSource::Speech => perception.ingest_speech(&emb.values),
                            AudioEncoderSource::Env    => perception.ingest_env(&emb.values),
                        };
                        match result {
                            Ok(out) => {
                                if let (Some(ep), Some(regional)) = (&out.episode, &out.regional_inputs) {
                                    match perception.wake_reconstruct_step(&ep.0, regional) {
                                        Ok(mse) => tracing::info!(output_mse = mse, "wake reconstruction"),
                                        Err(e)  => tracing::warn!(?e, "wake reconstruct error"),
                                    }
                                }
                                record_episode(&out, &emb.timestamp, &mut sink, &mut episodic, args.top_k, &recall);
                                if args.action_gain {
                                    if let Some(ep) = &out.episode {
                                        if let Some(ref mut dec) = decoder {
                                            match dec.update_gain(&ep.0) {
                                                Ok(()) => perception.set_region_gain(dec.gain),
                                                Err(e) => tracing::warn!(?e, "gain update failed"),
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => tracing::error!(?e, "perception error (audio)"),
                        }
                    }
                    None => { audio_emb_closed = true; }
                }
            }
            maybe = video_emb_rx.recv(), if !video_emb_closed => {
                match maybe {
                    Some(emb) => {
                        let _ = pending_embs.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| Some(x.saturating_sub(1)));
                        sink_write(&mut sink, &video_emb_json(&emb));
                        match perception.ingest_visual(&emb.values) {
                            Ok(out) => {
                                if let (Some(ep), Some(regional)) = (&out.episode, &out.regional_inputs) {
                                    match perception.wake_reconstruct_step(&ep.0, regional) {
                                        Ok(mse) => tracing::info!(output_mse = mse, "wake reconstruction"),
                                        Err(e)  => tracing::warn!(?e, "wake reconstruct error"),
                                    }
                                }
                                record_episode(&out, &emb.timestamp, &mut sink, &mut episodic, args.top_k, &recall);
                                if args.action_gain {
                                    if let Some(ep) = &out.episode {
                                        if let Some(ref mut dec) = decoder {
                                            match dec.update_gain(&ep.0) {
                                                Ok(()) => perception.set_region_gain(dec.gain),
                                                Err(e) => tracing::warn!(?e, "gain update failed"),
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => tracing::error!(?e, "perception error (video)"),
                        }
                    }
                    None => { video_emb_closed = true; }
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
            Some(query) = query_rx.recv() => {
                // Query: generate a response using the current episode as context (ADR 0012).
                // The query text is also stored as text_target on the latest episode for NREM training.
                tracing::info!(query = %query, "query received");
                let ep_vec = episodic.iter_traces().last().map(|t| t.record.values.clone());
                match ep_vec {
                    None => {
                        println!("{}", serde_json::json!({"type":"action_response","error":"no episode yet"}));
                    }
                    Some(ep) => {
                        if let Some(ref mut dec) = decoder {
                            // Store query as text_target before generation so it's available for NREM.
                            if let Ok(ids) = dec.tokenize(&query) {
                                episodic.set_last_text_target(ids);
                            }
                            tracing::info!("starting generation");
                            // Run generation in a blocking thread so we don't stall the async runtime.
                            let ep_clone = ep.clone();
                            let query_clone = query.clone();
                            let max_tokens = args.action_max_tokens;
                            // SAFETY: ActionDecoder is not Send. We run generation synchronously
                            // here (blocking the current async task via block_in_place).
                            let result = tokio::task::block_in_place(|| {
                                dec.generate(&ep_clone, &query_clone, max_tokens)
                            });
                            match result {
                                Ok(response) => {
                                    let ts_ms = SystemTime::now()
                                        .duration_since(UNIX_EPOCH)
                                        .map(|d| d.as_millis())
                                        .unwrap_or(0);
                                    tracing::info!(tokens = response.split_whitespace().count(), "generation complete");
                                    println!("{}", serde_json::json!({
                                        "type": "action_response",
                                        "ts": ts_ms,
                                        "query": query,
                                        "response": response,
                                    }));
                                }
                                Err(e) => tracing::warn!(?e, "generation failed"),
                            }
                        }
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                interrupt.store(true, Ordering::Relaxed);
                sink_flush(&mut sink);
                info!("shutting down");
                break;
            }
        }

        if file_mode && audio_emb_closed && video_emb_closed {
            sink_flush(&mut sink);
            info!("file source exhausted — exiting");
            break;
        }
    }

    Ok(())
}

// ── recollection constants ────────────────────────────────────────────────────

/// Blend weight for the retrieved episode: `α*retrieved + (1-α)*current`.
/// 0.1 means recalled memory contributes 10% to the stored episode.
/// Intentional feedback loop: recall shapes future recall (memory-of-memory).
const RECALL_BLEND_ALPHA: f32 = 0.1;

/// L2 distance below which we blend the retrieved episode into the current one.
/// Calibrated 2026-05-14: wake-time recall L2 on same-content (single looped video) has
/// p95=4.1 and max=5.3. Threshold of 5.5 covers the same-content distribution; any episode
/// beyond this distance is dissimilar enough that blending would inject noise.
const RECALL_BLEND_THRESHOLD: f32 = 5.5;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Push an episode into the hot buffer (without touching warm tier — sleep does that).
/// Queries warm memory first; blends the top-1 retrieved vector into the stored
/// episode when it is within RECALL_BLEND_THRESHOLD (L2).
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

        let regional = out
            .regional_inputs
            .clone()
            .unwrap_or([vec![], vec![], vec![], vec![], vec![]]);

        // Recollection: find similar past episodes from warm memory.
        let neighbours = recall.query_warm_with_dist(&ep.0, top_k);
        let (episode_vec, blended) = if let Some(&(dist, rec)) = neighbours.first() {
            tracing::info!(
                top1_id = rec.id,
                top1_timestamp_secs = rec.timestamp_secs,
                l2_dist = dist,
                retrieved = neighbours.len(),
                "recollection"
            );
            if dist < RECALL_BLEND_THRESHOLD {
                let v: Vec<f32> = ep
                    .0
                    .iter()
                    .zip(rec.values.iter())
                    .map(|(&cur, &ret)| (1.0 - RECALL_BLEND_ALPHA) * cur + RECALL_BLEND_ALPHA * ret)
                    .collect();
                (v, true)
            } else {
                (ep.0.clone(), false)
            }
        } else {
            (ep.0.clone(), false)
        };

        let id = episodic.push(ts_secs, episode_vec.clone(), regional);
        tracing::info!(
            dim = episode_vec.len(),
            episode_id = id,
            blended,
            "episode vector"
        );

        let line = serde_json::json!({
            "type":           "episode",
            "id":             id,
            "timestamp_secs": ts_secs,
            "dim":            episode_vec.len(),
            "values":         episode_vec,
        })
        .to_string();
        sink_write(sink, &line);
    }
}

/// Lossy forward: drops oldest frame when channel is full. Used for live sources
/// (microphone, webcam) where falling behind means the data is stale anyway.
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

/// Lossless forward: blocks until the downstream channel has room. Used for file
/// sources where every frame is needed and there is no notion of "stale data".
/// Returns `true` if the source was exhausted normally, `false` if the
/// downstream channel closed first (used by the file-loop thread to decide
/// whether to restart or exit).
fn forward_frames_blocking<T: Send>(src: Receiver<T>, tx: Sender<T>) -> bool {
    while let Ok(frame) = src.recv() {
        if tx.send(frame).is_err() {
            return false; // downstream closed
        }
    }
    true // source exhausted normally
}
