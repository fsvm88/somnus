#![recursion_limit = "256"]

mod action;
mod sleep;
mod whisper_asr;

/// Register MIGraphX EP using the generic key-value ORT API, then build the session.
/// Avoids the struct-based API which triggers MIGraphX's internal problem-cache auto-save
/// with an empty path (causing GPU kernel failure on gfx1201 RDNA4).
pub(crate) fn migraphx_session(path: &std::path::Path) -> anyhow::Result<ort::session::Session> {
    use ort::AsPointer;
    let mut builder = ort::session::Session::builder()?;
    // Leave ~half of cores free for burn-ndarray's matmul thread pool.
    // MIGraphX does the heavy lifting on GPU; the CPU threads here are
    // only used for pre/post-processing and graph fallback ops.
    let (intra, inter) = onnx_thread_budget();
    builder = builder
        .with_intra_threads(intra)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_inter_threads(inter)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let name = std::ffi::CString::new("MIGraphXExecutionProvider").unwrap();
    let status = unsafe {
        (ort::api().SessionOptionsAppendExecutionProvider)(
            builder.ptr_mut(),
            name.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };
    if !status.0.is_null() {
        unsafe { (ort::api().ReleaseStatus)(status.0) };
    }
    Ok(builder.commit_from_file(path)?)
}

/// CPU-only session for models with dynamic shapes (Whisper decoder, Qwen LM).
/// MIGraphX requires static shapes; these models grow their sequence dimension at runtime.
///
/// `profile_prefix`: when `Some`, enables ORT Chrome-trace profiling. ORT writes
/// `<prefix>_<timestamp>.json`. Call `session.end_profiling()` before process exit
/// to flush the file — `process::exit(0)` skips destructors and loses the data.
pub(crate) fn cpu_session(
    path: &std::path::Path,
    profile_prefix: Option<&std::path::Path>,
) -> anyhow::Result<ort::session::Session> {
    let (intra, inter) = onnx_thread_budget();
    let mut builder = ort::session::Session::builder()?
        .with_intra_threads(intra)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_inter_threads(inter)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if let Some(prefix) = profile_prefix {
        builder = builder
            .with_profiling(prefix)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    Ok(builder.commit_from_file(path)?)
}

/// Compute (intra_op, inter_op) thread counts for ONNX sessions.
///
/// Reserves roughly half the logical cores for burn-ndarray's rayon thread pool.
/// Both pools compete for the same hardware threads during NREM replay, so this
/// cap prevents ONNX from starving burn's matmul workers.
fn onnx_thread_budget() -> (usize, usize) {
    let total = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let intra = (total / 2).max(1);
    let inter = (total / 4).max(1);
    (intra, inter)
}

use action::{ActionDecoder, GainWorker};
use whisper_asr::WhisperDecoder;

use anyhow::Result;
use burn::backend::ndarray::NdArrayDevice;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use somnus_encoding::{
    preprocess::resample::SincResampler, AudioEmbedding, AudioEncoder, AudioEncoderSource,
    VideoEmbedding, VideoEncoder,
};
use somnus_memory::{
    load_region_sizes, load_snapshot, weights_path, ColdArchive, EpisodicBuffer, IngestSource,
    RecollectionLayer, SemanticStore,
};
use somnus_perception::{PerceptionProcessor, WakeReconWorker, EPISODE_DIM};
use somnus_sensory::{
    AudioCapture, AudioFrame, MediaFile, SensorMetrics, Timestamp, VideoCapture, VideoFrame,
};
use safetensors::SafeTensors;
use std::{
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::info;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(about = "somnus sensory + encoding + perception + memory + sleep pipeline — Phase 5")]
struct Args {
    /// One or more media files to use as input. May be specified multiple times.
    /// In loop mode (--file-loop) the files cycle in order indefinitely.
    /// Without --file-loop each file is played once and then the process exits.
    #[arg(long, num_args = 1..)]
    file: Vec<PathBuf>,

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

    /// Cosine similarity gate for settling: settle fires when prior_cosine < this
    /// threshold. Raise toward 1.0 to reduce settle frequency. Default 0.85.
    #[arg(long, default_value = "0.85")]
    settle_gate: f32,

    /// Run settling only every N episodes (1 = every episode). Subsamples settle
    /// to reduce GPU cost. E.g. --settle-every 2 halves settle overhead.
    #[arg(long, default_value = "1")]
    settle_every: usize,

    /// Working-memory ring size: number of recent episode vectors to carry as
    /// cross-attention context in the binding area (ADR 0010/10d). 0 = disabled (default).
    /// Cleared at sleep entry (wake-local by design).
    #[arg(long, default_value = "0")]
    working_memory_size: usize,

    /// Cap on the number of records in the warm HNSW index. When warm reaches
    /// this limit, the index is rebuilt during sleep keeping only the most recent
    /// N records from cold. 0 = unlimited (default). Suggested: 200_000 for
    /// long-running sessions to bound RAM usage (~500 MB at 640-dim f32).
    #[arg(long, default_value = "0")]
    max_warm_records: usize,

    /// ef parameter for HNSW warm-memory queries. Higher ef = more accurate recall
    /// but slower search. Default 64. Lower values (e.g. 16) give ~4× faster queries
    /// with some recall loss — validate with eval_metrics.py warm_recall_l2.
    #[arg(long, default_value = "64")]
    recall_ef: usize,

    /// Run wake_reconstruct_step on a background worker thread instead of
    /// blocking the ingest loop (saves ~1–3ms/ep amortized, eliminates 5–15ms
    /// latency spikes on high-salience episodes). Head weights merge back into
    /// the main model at sleep entry; EMA prediction weights stream back
    /// continuously with a few-episode lag.
    #[arg(long, default_value = "false")]
    wake_recon_async: bool,

    /// Cap on the number of records in the cold archive. When the archive exceeds
    /// this limit during sleep, the oldest records are dropped and the file is
    /// rewritten keeping only the most recent N. 0 = unlimited (default).
    /// Suggested: 500_000 for multi-day runs (~330 MB at 640-dim INT8).
    #[arg(long, default_value = "0")]
    max_cold_records: usize,

    /// Enable ActionDecoder: per-episode LM forward for gain modulation + stdin generation (ADR 0012).
    /// Requires qwen25_lm.onnx, qwen25_embed_table.safetensors, qwen25_tokenizer.json in --models-dir.
    #[arg(long, default_value = "false")]
    action: bool,

    /// Enable per-region gain modulation from LM attention (ADR 0012).
    /// Only effective when --action is set.
    #[arg(long, default_value = "true")]
    action_gain: bool,

    /// Run the gain update every N episodes instead of every episode.
    /// Gain changes slowly; N=20 is sufficient while keeping the background worker busy < 50%.
    #[arg(long, default_value = "20")]
    gain_interval: usize,

    /// Run wake_reconstruct_step every N episodes. Higher values reduce CPU overhead at
    /// the cost of less frequent output-head gradient updates. N=1 fires every episode.
    #[arg(long, default_value = "5")]
    wake_reconstruct_interval: usize,

    /// Encode only every Nth video frame through DINOv2. N=1 encodes all frames;
    /// N=2 halves video encoder load with negligible scene-understanding loss.
    #[arg(long, default_value = "1")]
    video_stride: usize,

    /// Maximum tokens to generate per query (ADR 0012).
    #[arg(long, default_value = "128")]
    action_max_tokens: usize,

    /// Loop the input file indefinitely. Restarts from the beginning each time
    /// the file ends, keeping warm memory alive across iterations. Useful for
    /// calibration runs where a single file is too short to trigger multiple
    /// sleep cycles.
    #[arg(long, default_value = "false")]
    file_loop: bool,

    /// Enable language region (ADR 0013, Stage A). Reads text from stdin; each
    /// line is tokenized via Qwen2.5 BPE, mean-pooled through the frozen embed
    /// table, and ingested into the language k-WTA region.
    /// Requires qwen25_embed_table.safetensors and qwen25_tokenizer.json in
    /// --models-dir. When combined with --action, the same stdin line is also
    /// routed to the query channel for generation.
    #[arg(long, default_value = "false")]
    language: bool,

    /// Write tracing logs directly to this file instead of stderr.
    /// Avoids the broken-pipe issue when running under `cargo run` and
    /// redirecting stderr (cargo's internal pipe breaks when cargo exits).
    /// Prefer `scripts/run.sh` which builds and execs the binary directly.
    #[arg(long)]
    log_file: Option<PathBuf>,

    /// Exit cleanly after this many sleep cycles (0 = run indefinitely).
    /// Useful for deterministic calibration runs.
    #[arg(long, default_value = "0")]
    max_sleep_cycles: usize,

    /// Session index written to every NDJSON metrics line.
    /// Used by train_campaign.sh to number consecutive runs in a multi-session
    /// campaign; 0 means "not part of a campaign" and is omitted from the line.
    #[arg(long, default_value = "0")]
    session: usize,

    /// Append one NDJSON metrics line per sleep cycle to this file.
    /// Contains: cycle, ts, traces_replayed, mean_loss, mean_output_mse,
    /// warm_recall_l2, was_interrupted, region_sizes, grew, wake_episodes,
    /// asr_count, lang_ingest_count, mean_output_mse_wake, prior_cosine_mean.
    #[arg(long)]
    metrics_file: Option<PathBuf>,

    /// Write the process PID to this file on startup (for status scripts).
    #[arg(long)]
    pid_file: Option<PathBuf>,

    /// Fire a one-shot generation query after the first episode is recorded.
    /// Requires --action. Useful for non-interactive evaluation runs.
    #[arg(long)]
    query: Option<String>,

    /// If > 0.0, trigger sleep early when the per-episode prior_cosine EWMA drops
    /// more than this amount below the healthy baseline captured at wake start.
    /// Set to 0.0 (default) to disable; calibrate from observed prior_cosine ranges
    /// before enabling. Trigger is logged as "perf_degradation".
    #[arg(long, default_value = "0.0")]
    perf_sleep_threshold: f64,

    /// NREM exits early when loss EWMA drops below this value (0.0 = run all traces).
    /// Useful when consolidation converges quickly on small or familiar buffers.
    /// Exit is logged as "loss_target_met" in the sleep summary.
    #[arg(long, default_value = "0.0")]
    sleep_loss_target: f32,

    /// Hard cap on total sleep-cycle wall time in seconds (0 = no cap).
    /// When exceeded the cycle is treated as interrupted (no spill, no snapshot).
    /// Exit is logged as "time_budget" in the sleep summary.
    #[arg(long, default_value = "0")]
    sleep_max_secs: u64,

    /// Fraction of grown units to prune per region during REM (0.0 = disabled).
    /// Only units above the N_UNITS=1024 baseline are candidates. Pruning fires
    /// when gist_recall_rate ≥ 0.8 (familiar content; system not struggling).
    #[arg(long, default_value = "0.0")]
    prune_fraction: f32,

    /// Minimum number of grown units a region must have before pruning is attempted.
    /// Avoids thrashing on small growth steps.
    #[arg(long, default_value = "64")]
    prune_min_grown: usize,

    /// Number of traces processed per forward+backward pass during NREM replay.
    /// Higher values amortise optimizer overhead and improve BLAS utilisation.
    /// 0 or 1 = sequential (original behaviour). Default: 128.
    #[arg(long, default_value = "128")]
    replay_batch_size: usize,

    /// Minimum number of NREM traces replayed before the salience gate can
    /// interrupt sleep. Prevents early-exit on the very first batch when
    /// diverse/long clips produce many novel embeddings immediately.
    /// 0 = no floor (original behaviour); try 500–1000 with long-clip training.
    #[arg(long, default_value = "0")]
    nrem_min_traces: usize,

    /// Pending-embedding count above which NREM is interrupted to allow live
    /// input to be processed. 0 = disabled (default; correct for file-mode
    /// campaigns where encoders run faster than real-time and would otherwise
    /// abort sleep prematurely). Set to ~16 for live ingest mode.
    #[arg(long, default_value = "0")]
    nrem_salience_threshold: usize,

    /// Minimum character count for an ASR transcription to be forwarded to the
    /// language ingest path. Filters noise/silence artifacts from non-speech
    /// clips (Whisper often outputs 1–7 char strings on urban/nature audio).
    /// 0 = accept all non-empty strings. Default: 20.
    #[arg(long, default_value = "20")]
    asr_min_chars: usize,

    /// How many extra times language-tagged traces are repeated at the front of
    /// the NREM replay queue (in addition to the one baseline pass).
    /// 0 = no oversampling (default). 9 = replay language traces 10× total.
    /// Increases the projection training budget without adding new data.
    #[arg(long, default_value = "0")]
    lang_oversample: usize,

    /// Enable ORT Chrome-trace profiling for CPU sessions (Qwen LM).
    /// Writes `<dir>/qwen25_lm_<timestamp>.json` — open in https://ui.perfetto.dev
    /// or chrome://tracing. Profiling data is flushed before process exit.
    /// GPU sessions (Whisper encoder, CLAP, DINOv2, Qwen gain) are covered by
    /// rocprof instead; see scripts/profile.sh.
    #[arg(long)]
    ort_profile_dir: Option<PathBuf>,

    /// Enable ORT binding inference (Phase 1 GPU unification).
    /// Directory where ONNX files are written and loaded from.
    /// On startup and after each sleep cycle, burn weights are exported and
    /// the ONNX graphs are rebuilt via scripts/export_perception_graph.py.
    /// Requires Python3 with onnx + onnxruntime installed.
    #[arg(long)]
    onnx_models_dir: Option<PathBuf>,

    /// Path to scripts/export_perception_graph.py (default: auto-detected relative
    /// to the binary location). Only needed when --onnx-models-dir is set.
    #[arg(long)]
    onnx_script: Option<PathBuf>,

    /// Use MIGraphX GPU EP for ORT binding session (requires AMD GPU + MIGraphX).
    /// When false and --onnx-models-dir is set, uses CPU EP (for testing / validation).
    #[arg(long, default_value = "true")]
    onnx_migraphx: bool,
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
    /// Sends encoder hidden states to the dedicated ASR background thread (ADR 0013).
    /// Non-blocking (try_send): drops the chunk if the decoder is still busy.
    asr_tx: Option<std::sync::mpsc::SyncSender<Vec<f32>>>,
}

impl AudioProcessor {
    fn new(
        models_dir: &std::path::Path,
        asr_tx: Option<std::sync::mpsc::SyncSender<Vec<f32>>>,
    ) -> Result<Self> {
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
            asr_tx,
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
            let (emb, hidden_flat) = self.encoder.encode_speech(&pcm, ts)?;
            tracing::info!(
                dim = emb.values.len(),
                encode_ms = start.elapsed().as_millis(),
                "speech embedding"
            );
            // ASR: send hidden states to the dedicated ASR thread (non-blocking).
            // If the decoder is still busy the chunk is dropped; audio processing continues.
            if let Some(ref tx) = self.asr_tx {
                if tx.try_send(hidden_flat).is_err() {
                    tracing::debug!("ASR thread busy — hidden state dropped");
                }
            }
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

// ── language embedder ─────────────────────────────────────────────────────────

const LANG_DIM: usize = 896; // Qwen2.5 hidden / embedding dimension

struct LangEmbedder {
    tokenizer: Tokenizer,
    /// Flat row-major [vocab_size * LANG_DIM] f32 embed table.
    embed_table: Vec<f32>,
}

impl LangEmbedder {
    fn new(models_dir: &std::path::Path) -> Result<Self> {
        let embed_path = models_dir.join("qwen25_embed_table.safetensors");
        let tok_path = models_dir.join("qwen25_tokenizer.json");
        let bytes = std::fs::read(&embed_path)?;
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("safetensors load failed: {e}"))?;
        let tv = st
            .tensor("embed_tokens.weight")
            .map_err(|e| anyhow::anyhow!("embed_tokens.weight missing: {e}"))?;
        let embed_table: Vec<f32> = tv
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
        let vocab = embed_table.len() / LANG_DIM;
        info!(vocab, "language embedder ready");
        Ok(Self {
            tokenizer,
            embed_table,
        })
    }

    /// Tokenize `text` → mean-pool token embeddings → 896-dim Vec<f32>.
    /// Empty or all-whitespace input returns a zero vector (filtered upstream).
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        let ids = enc.get_ids();
        if ids.is_empty() {
            return Ok(vec![0.0f32; LANG_DIM]);
        }
        let mut mean = vec![0.0f32; LANG_DIM];
        for &id in ids {
            let start = id as usize * LANG_DIM;
            for (a, b) in mean
                .iter_mut()
                .zip(&self.embed_table[start..start + LANG_DIM])
            {
                *a += b;
            }
        }
        let n = ids.len() as f32;
        for v in &mut mean {
            *v /= n;
        }
        Ok(mean)
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
    let args = Args::parse();

    // Init tracing before ort so startup errors are captured.
    // When --log-file is set, write directly to that file to avoid the
    // broken-pipe problem that occurs when running under `cargo run` and
    // redirecting stderr (cargo's internal pipe breaks when cargo exits).
    {
        use tracing_subscriber::fmt::writer::BoxMakeWriter;
        let writer: BoxMakeWriter = if let Some(ref path) = args.log_file {
            let f = std::fs::File::create(path)?;
            BoxMakeWriter::new(std::sync::Mutex::new(f))
        } else {
            BoxMakeWriter::new(std::io::stderr)
        };
        tracing_subscriber::fmt()
            .with_writer(writer)
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .init();
    }

    ort::init_from("/usr/lib/libonnxruntime.so")?.commit();

    // Write PID file for status scripts.
    if let Some(ref pid_path) = args.pid_file {
        std::fs::write(pid_path, format!("{}\n", std::process::id()))?;
    }

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

    if args.language {
        for name in &["qwen25_embed_table.safetensors", "qwen25_tokenizer.json"] {
            let p = args.models_dir.join(name);
            if !p.exists() {
                anyhow::bail!("language model file not found: {}", p.display());
            }
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
    let mut perception = PerceptionProcessor::new(NdArrayDevice::Cpu);
    perception.set_prior_beta(args.prior_beta);
    perception.set_settle_steps(args.settle_steps);
    perception.set_settle_beta(args.settle_beta);
    perception.set_settle_gate(args.settle_gate);
    perception.set_settle_every(args.settle_every);
    perception.set_wm_size(args.working_memory_size);
    info!(
        prior_beta = args.prior_beta,
        settle_steps = args.settle_steps,
        settle_beta = args.settle_beta,
        working_memory_size = args.working_memory_size,
        "perception processor initialised (n=1024, k=20, cpu)"
    );

    // ── ORT binding session (Phase 1 GPU unification) ─────────────────────────
    let onnx_script_path: Option<PathBuf> = args.onnx_models_dir.as_ref().map(|_| {
        args.onnx_script.clone().unwrap_or_else(|| {
            // Default: find script relative to the binary
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."))
                .join("../../scripts/export_perception_graph.py")
        })
    });
    if let (Some(ref onnx_dir), Some(ref script)) = (&args.onnx_models_dir, &onnx_script_path) {
        std::fs::create_dir_all(onnx_dir)?;
        info!(onnx_dir = %onnx_dir.display(), migraphx = args.onnx_migraphx, "building ORT binding session");
        match perception.rebuild_ort_session(script, onnx_dir, args.onnx_migraphx) {
            Ok(()) => info!("ORT binding session active"),
            Err(e) => tracing::warn!(?e, "ORT session init failed — running with burn inference"),
        }
    }

    // ── memory hierarchy ──────────────────────────────────────────────────────
    std::fs::create_dir_all(&args.memory_dir)?;
    let cold = match ColdArchive::open(&args.memory_dir) {
        Ok(c) => c,
        Err(e) => anyhow::bail!("cold archive init failed: {e}"),
    };

    // Restore from snapshot if available; otherwise start fresh.
    let mut snapshot_found = false;
    let (mut episodic, mut semantic) =
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

    if args.max_warm_records > 0 {
        semantic.set_max_records(args.max_warm_records);
    }

    if args.settle_steps > 0 && !snapshot_found {
        tracing::warn!(
            settle_steps = args.settle_steps,
            "settling active but no prior snapshot — output heads are random; \
             settle noise expected until output_mse converges"
        );
    }

    // ── action decoder ────────────────────────────────────────────────────────
    let mut decoder: Option<ActionDecoder> = if args.action {
        match ActionDecoder::new(
            &args.models_dir,
            NdArrayDevice::Cpu,
            args.ort_profile_dir.as_deref(),
        ) {
            Ok(mut d) => {
                let dec_w = args
                    .memory_dir
                    .join("snapshot")
                    .join("decoder_weights.safetensors");
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

    // Move the gain ORT session to a background thread so ~54ms Qwen calls don't
    // block the tokio event loop. Only created when both --action and --action-gain are set.
    let gain_worker: Option<GainWorker> = if args.action_gain {
        decoder.as_mut().and_then(|d| d.create_gain_worker())
    } else {
        None
    };
    if gain_worker.is_some() {
        info!("gain worker spawned — Qwen gain updates running in background");
    }

    // ── wake-reconstruction worker (perception weights already restored above) ─
    let wake_recon_worker: Option<WakeReconWorker> = if args.wake_recon_async {
        info!("wake-recon worker spawned — wake_reconstruct_step running in background");
        Some(perception.create_wake_recon_worker())
    } else {
        None
    };
    let mut wake_recon_dropped: u64 = 0;

    // ── language embedder ─────────────────────────────────────────────────────
    let lang_embedder: Option<LangEmbedder> = if args.language {
        match LangEmbedder::new(&args.models_dir) {
            Ok(le) => Some(le),
            Err(e) => anyhow::bail!("language embedder init failed: {e}"),
        }
    } else {
        None
    };

    let mut recall = RecollectionLayer::new(semantic, cold);
    recall.warm.set_search_ef(args.recall_ef);
    info!(
        episodic_capacity = args.episodic_capacity,
        memory_dir = %args.memory_dir.display(),
        recall_ef = args.recall_ef,
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

    // ── metrics infrastructure ────────────────────────────────────────────────
    let mut metrics_sink: Option<std::fs::File> = if let Some(ref path) = args.metrics_file {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        info!(path = %path.display(), "metrics file opened");
        Some(f)
    } else {
        None
    };
    let mut sleep_cycle_count: usize = 0;
    // Tracks whether the --query one-shot generation has already fired.
    let mut startup_query_fired = false;
    // Per-wake-cycle counters — reset after each sleep cycle.
    let mut wake_episodes: usize = 0;
    let mut wake_output_mse_sum: f64 = 0.0;
    let mut wake_output_mse_n: usize = 0;
    let mut prior_cosine_sums: [f64; 5] = [0.0; 5];
    let mut prior_cosine_counts: [usize; 5] = [0; 5];
    let mut lang_ingest_count: usize = 0;
    let mut gain_episode_counter: usize = 0;
    let mut wake_reconstruct_counter: usize = 0;
    // ADR 0022 instrumentation additions.
    let mut salience_sum: f64 = 0.0;
    let mut salience_n: usize = 0;
    let mut salience_min: f32 = f32::MAX;
    let mut high_salience_episodes: usize = 0;
    let mut blend_count: usize = 0;
    let mut wake_reconstruct_throttle_passed: usize = 0;
    let mut episodes_speech: usize = 0;
    let mut episodes_env: usize = 0;
    let mut episodes_visual: usize = 0;
    let mut episodes_language: usize = 0;
    let mut episodes_mixed: usize = 0;
    let mut settle_trigger_count: usize = 0;
    let mut settle_iters_sum: usize = 0;
    let mut settle_delta_sum: f64 = 0.0;
    // Per-cycle encode/perception timing (ms) — reset after each sleep cycle.
    let mut speech_encode_total_ms: u64 = 0;
    let mut env_encode_total_ms: u64 = 0;
    let mut video_encode_total_ms: u64 = 0;
    let mut perception_total_ms: u64 = 0;
    // Per-wake-cycle EWMA state for the perf-degradation sleep trigger (10a).
    // cosine_baseline: mean prior_cosine of the first COSINE_BASELINE_N episodes.
    // cosine_ewma: exponentially weighted moving average (α = COSINE_EWMA_ALPHA).
    // perf_sleep_flag: set when EWMA drops > perf_sleep_threshold below baseline.
    let mut cosine_ewma: Option<f64> = None;
    let mut cosine_baseline: Option<f64> = None;
    let mut cosine_baseline_accum: f64 = 0.0;
    let mut cosine_baseline_n: usize = 0;
    let mut perf_sleep_flag = false;
    // ASR count shared with the audio encoder task.
    let asr_count = Arc::new(AtomicUsize::new(0));

    // ── Whisper ASR decoder (Stage B, ADR 0013) ──────────────────────────────
    // Soft failure: if the decoder ONNX is absent, ASR is silently disabled.
    let whisper_dec: Option<WhisperDecoder> = if args.language {
        match WhisperDecoder::new(&args.models_dir) {
            Ok(d) => {
                info!("whisper ASR decoder ready");
                Some(d)
            }
            Err(e) => {
                tracing::warn!(
                    ?e,
                    "whisper ASR decoder not available — text-only language ingestion"
                );
                None
            }
        }
    } else {
        None
    };

    // ── stdin channels ────────────────────────────────────────────────────────
    // query_rx: stdin lines routed to ActionDecoder for generation (ADR 0012).
    // lang_rx:  stdin lines routed to language region ingestion (ADR 0013).
    //   Sources: (a) stdin reader task, (b) audio encoder task via Whisper ASR.
    // When both --action and --language are set, each stdin line feeds both paths.
    let (query_tx, mut query_rx) = tokio::sync::mpsc::channel::<String>(8);
    let (lang_tx, mut lang_rx) = tokio::sync::mpsc::channel::<String>(64);
    {
        let needs_stdin = decoder.is_some() || lang_embedder.is_some();
        if needs_stdin {
            let qtx = if decoder.is_some() {
                Some(query_tx)
            } else {
                None
            };
            // Clone lang_tx for the stdin reader; the original stays in scope for the audio task.
            let ltx_stdin = if lang_embedder.is_some() {
                Some(lang_tx.clone())
            } else {
                None
            };
            tokio::task::spawn_blocking(move || {
                use std::io::BufRead;
                for line in std::io::stdin().lock().lines().map_while(Result::ok) {
                    let trimmed = line.trim().to_string();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if let Some(ref l) = ltx_stdin {
                        let _ = l.blocking_send(trimmed.clone());
                    }
                    if let Some(ref q) = qtx {
                        if q.blocking_send(trimmed).is_err() {
                            break;
                        }
                    }
                }
            });
        }
    }

    // ── audio fan-in ──────────────────────────────────────────────────────────
    let (audio_cb_tx, audio_cb_rx) = bounded::<AudioFrame>(256);
    let audio_drain = audio_cb_rx.clone();
    let mut audio_metric_handles: Vec<Arc<SensorMetrics>> = Vec::new();

    // ── video setup ───────────────────────────────────────────────────────────
    let (video_cb_tx, video_cb_rx) = bounded::<VideoFrame>(16);
    let video_drain = video_cb_rx.clone();
    let mut video_metrics_handle: Arc<SensorMetrics>;

    if !args.file.is_empty() {
        if args.file_loop {
            // Looping multi-file source: rotate through all files in order,
            // cycling back to the first when the last finishes.
            let paths = args.file.clone();
            let atx = audio_cb_tx.clone();
            let vtx = video_cb_tx.clone();
            std::thread::spawn(move || {
                let mut idx = 0usize;
                loop {
                    let path = &paths[idx % paths.len()];
                    tracing::info!(file = ?path, "file loop: opening");
                    let streams = match MediaFile::open(path).and_then(|mf| mf.start()) {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::error!(?e, file = ?path, "file loop: failed to open");
                            break;
                        }
                    };
                    let atx2 = atx.clone();
                    let vtx2 = vtx.clone();
                    let ah =
                        std::thread::spawn(move || forward_frames_blocking(streams.audio, atx2));
                    let vh =
                        std::thread::spawn(move || forward_frames_blocking(streams.video, vtx2));
                    let a_normal = ah.join().unwrap_or(false);
                    let v_normal = vh.join().unwrap_or(false);
                    if !a_normal || !v_normal {
                        break;
                    }
                    idx += 1;
                }
            });
            video_metrics_handle = Arc::new(SensorMetrics::default());
        } else {
            // One-shot: stream all files in order then let channels close.
            let paths = args.file.clone();
            let atx = audio_cb_tx.clone();
            let vtx = video_cb_tx.clone();
            // Open first file for metrics; remaining files are opened inside the thread.
            let first = MediaFile::open(&paths[0]).and_then(|mf| mf.start())?;
            audio_metric_handles.push(first.audio_metrics);
            video_metrics_handle = first.video_metrics;
            let first_audio = first.audio;
            let first_video = first.video;
            let paths_v = paths.clone();
            std::thread::spawn(move || {
                forward_frames_blocking(first_audio, atx.clone());
                for path in &paths_v[1..] {
                    tracing::info!(file = ?path, "file sequence: opening next");
                    let streams = match MediaFile::open(path).and_then(|mf| mf.start()) {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::error!(?e, file = ?path, "file sequence: failed to open");
                            return;
                        }
                    };
                    if !forward_frames_blocking(streams.audio, atx.clone()) {
                        return;
                    }
                }
            });
            std::thread::spawn(move || {
                forward_frames_blocking(first_video, vtx.clone());
                for path in &paths[1..] {
                    let streams = match MediaFile::open(path).and_then(|mf| mf.start()) {
                        Ok(s) => s,
                        Err(_) => return,
                    };
                    if !forward_frames_blocking(streams.video, vtx.clone()) {
                        return;
                    }
                }
            });
        }
    } else {
        // No file — fall through to mic/webcam paths below.
        video_metrics_handle = Arc::new(SensorMetrics::default()); // placeholder, overwritten below
    }

    if args.mic || args.file.is_empty() {
        match AudioCapture::start(None) {
            Ok(mic) => {
                audio_metric_handles.push(Arc::clone(&mic.metrics));
                let tx = audio_cb_tx.clone();
                let drain = audio_drain.clone();
                std::thread::spawn(move || {
                    forward_frames(mic.receiver, tx, drain);
                });
            }
            Err(e) => {
                tracing::warn!(?e, "microphone unavailable — running without mic audio");
            }
        }
    }

    if args.file.is_empty() {
        match VideoCapture::open(args.webcam) {
            Ok(mut cam) => {
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
            Err(e) => {
                tracing::warn!(
                    ?e,
                    webcam = args.webcam,
                    "webcam unavailable — running audio-only"
                );
            }
        }
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

    // ── ASR background thread (Stage B, ADR 0013) ────────────────────────────
    // Decouples Whisper decoding from audio accumulation: the audio thread sends
    // encoder hidden states via a bounded channel and returns immediately; decoding
    // happens concurrently.  Channel capacity 2: one being decoded + one buffered.
    let asr_hidden_tx: Option<std::sync::mpsc::SyncSender<Vec<f32>>> =
        if let Some(dec) = whisper_dec {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(2);
            let lang_tx_asr = lang_tx.clone();
            let asr_cnt = Arc::clone(&asr_count);
            let asr_min_chars = args.asr_min_chars;
            std::thread::spawn(move || {
                let mut dec = dec;
                while let Ok(hidden) = rx.recv() {
                    match dec.decode_greedy(&hidden, 200) {
                        Ok(text) if text.len() >= asr_min_chars.max(1) => {
                            tracing::info!(chars = text.len(), "whisper ASR");
                            asr_cnt.fetch_add(1, Ordering::Relaxed);
                            let _ = lang_tx_asr.blocking_send(text);
                        }
                        Ok(text) if !text.is_empty() => {
                            tracing::debug!(
                                chars = text.len(),
                                "whisper ASR filtered (below min_chars)"
                            );
                        }
                        Ok(_) => {}
                        Err(e) => tracing::warn!(?e, "whisper ASR failed"),
                    }
                }
            });
            Some(tx)
        } else {
            None
        };

    // ── audio encoder task ────────────────────────────────────────────────────
    let (audio_emb_tx, mut audio_emb_rx) = mpsc::channel::<AudioEmbedding>(32);
    {
        let models_dir = args.models_dir.clone();
        let mut audio_rx = audio_rx;
        let pending = Arc::clone(&pending_embs);
        tokio::task::spawn_blocking(move || {
            let mut proc = match AudioProcessor::new(&models_dir, asr_hidden_tx) {
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

    // Original lang_tx is no longer needed; clones are held by stdin reader and ASR thread.
    // Dropping it here ensures lang_rx closes once both senders finish.
    drop(lang_tx);

    // ── video encoder task ────────────────────────────────────────────────────
    let (video_emb_tx, mut video_emb_rx) = mpsc::channel::<VideoEmbedding>(16);
    {
        let models_dir = args.models_dir.clone();
        let mut video_rx = video_rx;
        let pending = Arc::clone(&pending_embs);
        let video_stride = args.video_stride.max(1);
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
            let mut frame_counter: usize = 0;
            while let Some(frame) = video_rx.blocking_recv() {
                frame_counter += 1;
                if !frame_counter.is_multiple_of(video_stride) {
                    continue;
                }
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
    let file_mode = !args.file.is_empty() && !args.file_loop;
    let mut audio_emb_closed = false;
    let mut video_emb_closed = false;
    // lang_rx is from stdin, not from a file, so it does not participate in the
    // file_mode exit check. Pre-close the guard when language is disabled so the
    // select arm is never polled (avoids a perpetually-ready None future).
    let mut lang_closed = !args.language;

    loop {
        // ── drain any pending queries before potentially sleeping ─────────────
        // This ensures queries are never blocked behind a sleep cycle.
        while let Ok(query) = query_rx.try_recv() {
            tracing::info!(query = %query, "query received (pre-sleep drain)");
            let ep_vec = episodic
                .iter_traces()
                .last()
                .map(|t| t.record.values.clone());
            match ep_vec {
                None => {
                    println!(
                        "{}",
                        serde_json::json!({"type":"action_response","error":"no episode yet"})
                    );
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
                                tracing::info!(
                                    tokens = response.split_whitespace().count(),
                                    "generation complete"
                                );
                                println!(
                                    "{}",
                                    serde_json::json!({
                                        "type": "action_response",
                                        "ts": ts_ms,
                                        "query": query,
                                        "response": response,
                                    })
                                );
                            }
                            Err(e) => tracing::warn!(?e, "generation failed"),
                        }
                    }
                }
            }
        }

        // ── poll gain worker result (non-blocking) ────────────────────────────
        if let Some(ref gw) = gain_worker {
            if let Some(g) = gw.try_recv_gain() {
                perception.set_region_gain(g);
            }
        }

        // ── poll wake-recon worker EMA updates (non-blocking) ─────────────────
        if let Some(ref w) = wake_recon_worker {
            while let Some((flat, loss)) = w.try_recv_ema() {
                perception.apply_ema_flat(flat);
                wake_output_mse_sum += loss as f64;
                wake_output_mse_n += 1;
                tracing::info!(output_mse = loss, "wake reconstruction (async)");
            }
        }

        // ── drain pending language lines before potentially sleeping ──────────
        while !lang_closed {
            match lang_rx.try_recv() {
                Ok(text) => {
                    if let Some(r) = ingest_language_line(
                        &text,
                        &lang_embedder,
                        &mut perception,
                        &mut sink,
                        &mut episodic,
                        args.top_k,
                        &recall,
                        &mut decoder,
                        &mut wake_reconstruct_counter,
                        args.wake_reconstruct_interval,
                        wake_recon_worker.as_ref(),
                        &mut wake_recon_dropped,
                    ) {
                        lang_ingest_count += 1;
                        if r.episode_produced {
                            wake_episodes += 1;
                            if let Some(m) = episode_cosine_mean(&r.prior_cosines) {
                                update_perf_ewma(
                                    m,
                                    &mut cosine_ewma,
                                    &mut cosine_baseline,
                                    &mut cosine_baseline_accum,
                                    &mut cosine_baseline_n,
                                    &mut perf_sleep_flag,
                                    args.perf_sleep_threshold,
                                );
                            }
                        }
                        if let Some(m) = r.reconstruct_mse {
                            wake_output_mse_sum += m as f64;
                            wake_output_mse_n += 1;
                        }
                        if r.throttle_passed {
                            wake_reconstruct_throttle_passed += 1;
                        }
                        if r.settle_iters > 0 {
                            settle_trigger_count += 1;
                            settle_iters_sum += r.settle_iters;
                            settle_delta_sum += r.settle_delta as f64;
                        }
                        if let Some(ref es) = r.ep_stats {
                            accumulate_salience(
                                es,
                                &mut salience_sum,
                                &mut salience_n,
                                &mut salience_min,
                                &mut high_salience_episodes,
                                &mut blend_count,
                                &mut episodes_speech,
                                &mut episodes_env,
                                &mut episodes_visual,
                                &mut episodes_language,
                                &mut episodes_mixed,
                            );
                        }
                        accumulate_cosines(
                            &r.prior_cosines,
                            &mut prior_cosine_sums,
                            &mut prior_cosine_counts,
                        );
                    }
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    lang_closed = true;
                }
            }
        }

        // ── sleep trigger check ───────────────────────────────────────────────
        let high_water_hit = episodic.len() >= high_water;
        let schedule_hit = wake_start.elapsed() >= max_wake;
        let perf_degradation_hit = perf_sleep_flag && args.perf_sleep_threshold > 0.0;
        let should_sleep = high_water_hit || schedule_hit || perf_degradation_hit;
        let sleep_trigger = if high_water_hit {
            "high_water"
        } else if perf_degradation_hit {
            "perf_degradation"
        } else {
            "schedule"
        };

        if should_sleep && !episodic.is_empty() {
            info!(
                trigger = sleep_trigger,
                buffer_len = episodic.len(),
                buffer_cap = args.episodic_capacity,
                wake_secs = wake_start.elapsed().as_secs(),
                "entering sleep"
            );
            interrupt.store(false, Ordering::Relaxed);
            pending_embs.store(0, Ordering::Relaxed);
            perception.reset_region_gain(); // stale gain must not carry across sleep boundary
            perception.clear_working_memory(); // working memory is wake-local
                                               // Merge worker-trained head weights into the model so NREM training
                                               // continues from them (drains the worker's queue; applies any
                                               // in-flight EMA updates so they aren't lost).
            if let Some(ref w) = wake_recon_worker {
                let mut pending_ema: Option<[Vec<f32>; 5]> = None;
                match w.fetch_weights(|flat, loss| {
                    wake_output_mse_sum += loss as f64;
                    wake_output_mse_n += 1;
                    pending_ema = Some(flat);
                }) {
                    Some(weights) => {
                        perception.set_head_weights_flat(weights);
                        if let Some(flat) = pending_ema {
                            perception.apply_ema_flat(flat);
                        }
                    }
                    None => tracing::warn!("wake-recon worker died — heads not merged"),
                }
            }
            match sleep::run_sleep_cycle(
                &mut perception,
                &mut episodic,
                &mut recall,
                &args.memory_dir,
                Arc::clone(&interrupt_sleep),
                Arc::clone(&pending_embs),
                &mut decoder,
                args.sleep_loss_target,
                args.sleep_max_secs,
                args.prune_fraction,
                args.prune_min_grown,
                args.replay_batch_size,
                args.nrem_min_traces,
                args.nrem_salience_threshold,
                args.lang_oversample,
                args.max_warm_records,
                args.max_cold_records,
            ) {
                Ok(stats) => {
                    sleep_cycle_count += 1;
                    let cycle_asr = asr_count.swap(0, Ordering::Relaxed);
                    info!(
                        cycle         = sleep_cycle_count,
                        traces_replayed = stats.traces_replayed,
                        mean_loss     = stats.mean_train_loss,
                        mean_output_mse = stats.mean_output_mse,
                        warm_recall_l2 = ?stats.warm_recall_l2,
                        mean_proj_loss = ?stats.mean_proj_loss,
                        cross_modal_gist_entropy = ?stats.cross_modal_gist_entropy,
                        decoder_trainable = stats.decoder_trainable_records,
                        interrupted   = stats.was_interrupted,
                        nrem_exit     = stats.nrem_exit_reason,
                        wake_episodes,
                        asr_count     = cycle_asr,
                        lang_ingest   = lang_ingest_count,
                        "sleep cycle complete"
                    );
                    // Save decoder projection weights alongside perception weights.
                    if let Some(ref dec) = decoder {
                        let dec_w = args
                            .memory_dir
                            .join("snapshot")
                            .join("decoder_weights.safetensors");
                        if let Err(e) = dec.save_weights(&dec_w) {
                            tracing::warn!(?e, "decoder weights snapshot failed");
                        }
                        // Push updated projection weights to gain worker (non-blocking).
                        if let Some(ref gw) = gain_worker {
                            gw.update_weights(dec.proj_weights_flat());
                        }
                    }
                    // Write metrics NDJSON line.
                    if let Some(ref mut mf) = metrics_sink {
                        let region_sizes = perception.region_sizes();
                        let grew = !stats.regions_grown.is_empty();
                        let mean_mse_wake = if wake_output_mse_n > 0 {
                            Some(wake_output_mse_sum / wake_output_mse_n as f64)
                        } else {
                            None
                        };
                        let prior_cosine_mean: Vec<Option<f64>> = (0..5)
                            .map(|i| {
                                if prior_cosine_counts[i] > 0 {
                                    Some(prior_cosine_sums[i] / prior_cosine_counts[i] as f64)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        // ADR 0022: salience stats (None when warm was empty all cycle).
                        let mean_salience_wake = if salience_n > 0 {
                            Some(salience_sum / salience_n as f64)
                        } else {
                            None
                        };
                        let min_salience_wake = if salience_n > 0 {
                            Some(salience_min)
                        } else {
                            None
                        };
                        let perf_breakdown = perception.take_perf_breakdown();
                        let ts = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .map(|d| d.as_secs_f64())
                            .unwrap_or(0.0);
                        let line = serde_json::json!({
                            "cycle":             sleep_cycle_count,
                            "session":           if args.session > 0 { Some(args.session) } else { None },
                            "ts":                ts,
                            "sleep_trigger":     sleep_trigger,
                            "traces_replayed":   stats.traces_replayed,
                            "nrem_budget_traces":        stats.nrem_budget_traces,
                            "nrem_unique_traces":        stats.nrem_unique_traces,
                            "mean_loss":         stats.mean_train_loss,
                            "mean_output_mse":   stats.mean_output_mse,
                            "per_region_mse":    stats.per_region_mse,
                            "warm_recall_l2":    stats.warm_recall_l2,
                            "gist_recall_rate":        stats.gist_recall_rate,
                            "cross_modal_gist_entropy": stats.cross_modal_gist_entropy,
                            "was_interrupted":   stats.was_interrupted,
                            "nrem_exit_reason":  stats.nrem_exit_reason,
                            "region_sizes":      region_sizes,
                            "warm_len":          recall.warm.len(),
                            "cold_len":          recall.cold.len(),
                            "grew":              grew,
                            "regions_grown":     stats.regions_grown.iter().map(|r| r.as_index()).collect::<Vec<_>>(),
                            "wake_episodes":     wake_episodes,
                            "episodes_speech":   episodes_speech,
                            "episodes_env":      episodes_env,
                            "episodes_visual":   episodes_visual,
                            "episodes_language": episodes_language,
                            "episodes_mixed":    episodes_mixed,
                            "asr_count":         cycle_asr,
                            "lang_ingest_count": lang_ingest_count,
                            "mean_salience_wake":       mean_salience_wake,
                            "min_salience_wake":        min_salience_wake,
                            "high_salience_episodes":   high_salience_episodes,
                            "blend_count":              blend_count,
                            "wake_reconstruct_attempts":       wake_reconstruct_counter,
                            "wake_reconstruct_throttle_passed": wake_reconstruct_throttle_passed,
                            "wake_reconstruct_fired":          wake_output_mse_n,
                            "wake_recon_dropped":              wake_recon_dropped,
                            "settle_trigger_count":  settle_trigger_count,
                            "settle_iters_sum":      settle_iters_sum,
                            "settle_delta_mean": if settle_trigger_count > 0 { Some(settle_delta_sum / settle_trigger_count as f64) } else { None },
                            "mean_output_mse_wake": mean_mse_wake,
                            "prior_cosine_mean": prior_cosine_mean,
                            "mean_proj_loss":    stats.mean_proj_loss,
                            "proj_loss_max":     stats.proj_loss_max,
                            "decoder_trainable": stats.decoder_trainable_records,
                            "synthetics_produced":        stats.synthetics_produced,
                            "mean_real_inference_mse":    stats.mean_real_inference_mse,
                            "mean_synthetic_output_mse":  stats.mean_synthetic_output_mse,
                            "regions_pruned":             stats.regions_pruned.len(),
                            "total_units_pruned":         stats.total_units_pruned,
                            "nrem_wall_ms":               stats.nrem_wall_ms,
                            "distill_precompute_ms":      stats.distill_precompute_ms,
                            "rem_wall_ms":                stats.rem_wall_ms,
                            "spill_wall_ms":              stats.spill_wall_ms,
                            "snapshot_wall_ms":           stats.snapshot_wall_ms,
                            "lm_hidden_computed":         stats.lm_hidden_computed,
                            "lm_hidden_unique":           stats.lm_hidden_unique,
                            "speech_encode_total_ms":     speech_encode_total_ms,
                            "env_encode_total_ms":        env_encode_total_ms,
                            "video_encode_total_ms":      video_encode_total_ms,
                            "gain_update_total_ms":       0u64,
                            "perception_total_ms":        perception_total_ms,
                            "perc_kwta_us":         perf_breakdown.kwta_us,
                            "perc_symbolic_us":     perf_breakdown.symbolic_us,
                            "perc_bind_us":         perf_breakdown.bind_us,
                            "perc_prior_us":        perf_breakdown.prior_us,
                            "perc_settle_us":       perf_breakdown.settle_us,
                            "perc_prior_update_us": perf_breakdown.prior_update_us,
                        })
                        .to_string();
                        use std::io::Write as _;
                        if let Err(e) = writeln!(mf, "{line}") {
                            tracing::warn!(?e, "metrics write failed");
                        }
                    }
                    // Reset per-wake-cycle counters.
                    wake_episodes = 0;
                    wake_output_mse_sum = 0.0;
                    wake_output_mse_n = 0;
                    prior_cosine_sums = [0.0; 5];
                    prior_cosine_counts = [0; 5];
                    lang_ingest_count = 0;
                    gain_episode_counter = 0;
                    wake_reconstruct_counter = 0;
                    // ADR 0022 counters.
                    salience_sum = 0.0;
                    salience_n = 0;
                    salience_min = f32::MAX;
                    high_salience_episodes = 0;
                    blend_count = 0;
                    wake_reconstruct_throttle_passed = 0;
                    wake_recon_dropped = 0;
                    episodes_speech = 0;
                    episodes_env = 0;
                    episodes_visual = 0;
                    episodes_language = 0;
                    episodes_mixed = 0;
                    settle_trigger_count = 0;
                    settle_iters_sum = 0;
                    settle_delta_sum = 0.0;
                    speech_encode_total_ms = 0;
                    env_encode_total_ms = 0;
                    video_encode_total_ms = 0;
                    perception_total_ms = 0;
                    cosine_ewma = None;
                    cosine_baseline = None;
                    cosine_baseline_accum = 0.0;
                    cosine_baseline_n = 0;
                    perf_sleep_flag = false;
                }
                Err(e) => tracing::warn!(?e, "sleep cycle error"),
            }
            // Push NREM-trained head weights back to the worker for the next
            // wake phase (also after a failed cycle — partial training counts).
            if let Some(ref w) = wake_recon_worker {
                w.set_weights(perception.head_weights_flat());
            }
            wake_start = std::time::Instant::now();

            // Rebuild ORT binding session with updated weights after sleep training.
            if let (Some(ref onnx_dir), Some(ref script)) =
                (&args.onnx_models_dir, &onnx_script_path)
            {
                match perception.rebuild_ort_session(script, onnx_dir, args.onnx_migraphx) {
                    Ok(()) => tracing::debug!("ORT binding session rebuilt"),
                    Err(e) => tracing::warn!(
                        ?e,
                        "ORT session rebuild failed — continuing with existing session"
                    ),
                }
            }

            // Exit after N sleep cycles if requested.
            if args.max_sleep_cycles > 0 && sleep_cycle_count >= args.max_sleep_cycles {
                info!(
                    cycles = sleep_cycle_count,
                    "max-sleep-cycles reached — exiting"
                );
                sink_flush(&mut sink);
                break;
            }
        }

        tokio::select! {
            maybe = audio_emb_rx.recv(), if !audio_emb_closed => {
                match maybe {
                    Some(emb) => {
                        let _ = pending_embs.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| Some(x.saturating_sub(1)));
                        sink_write(&mut sink, &audio_emb_json(&emb));
                        match emb.source {
                            AudioEncoderSource::Speech => speech_encode_total_ms += emb.encode_ms,
                            AudioEncoderSource::Env    => env_encode_total_ms    += emb.encode_ms,
                        }
                        let t_perc = std::time::Instant::now();
                        let result = match emb.source {
                            AudioEncoderSource::Speech => perception.ingest_speech(&emb.values),
                            AudioEncoderSource::Env    => perception.ingest_env(&emb.values),
                        };
                        perception_total_ms += t_perc.elapsed().as_millis() as u64;
                        match result {
                            Ok(out) => {
                                accumulate_cosines(&out.prior_cosines, &mut prior_cosine_sums, &mut prior_cosine_counts);
                                if out.settle_iters > 0 { settle_trigger_count += 1; settle_iters_sum += out.settle_iters; settle_delta_sum += out.settle_delta as f64; }
                                let rec = record_episode(&out, &emb.timestamp, &mut sink, &mut episodic, args.top_k, &recall);
                                if let Some(ref rec) = rec {
                                    accumulate_salience(rec, &mut salience_sum, &mut salience_n, &mut salience_min, &mut high_salience_episodes, &mut blend_count, &mut episodes_speech, &mut episodes_env, &mut episodes_visual, &mut episodes_language, &mut episodes_mixed);
                                }
                                if let (Some(ep), Some(regional)) = (&out.episode, &out.regional_inputs) {
                                    wake_reconstruct_counter += 1;
                                    if wake_reconstruct_counter.is_multiple_of(args.wake_reconstruct_interval) {
                                        wake_reconstruct_throttle_passed += 1;
                                        if rec.as_ref().is_some_and(|r| r.salience > WAKE_RECONSTRUCT_SALIENCE_THRESHOLD) {
                                            if let Some(ref w) = wake_recon_worker {
                                                if !w.request_train(ep.0.clone(), regional.clone(), perception.learning_rate()) {
                                                    wake_recon_dropped += 1;
                                                }
                                            } else {
                                                match perception.wake_reconstruct_step(&ep.0, regional) {
                                                    Ok(mse) => {
                                                        wake_output_mse_sum += mse as f64;
                                                        wake_output_mse_n += 1;
                                                        tracing::info!(output_mse = mse, "wake reconstruction");
                                                    }
                                                    Err(e) => tracing::warn!(?e, "wake reconstruct error"),
                                                }
                                            }
                                        }
                                        // Refresh autodiff graph every N episodes to prevent
                                        // node accumulation from forward passes slowing into_data().
                                        perception.refresh_autodiff_state();
                                    }
                                    wake_episodes += 1;
                                    if let Some(m) = episode_cosine_mean(&out.prior_cosines) {
                                        update_perf_ewma(m, &mut cosine_ewma, &mut cosine_baseline, &mut cosine_baseline_accum, &mut cosine_baseline_n, &mut perf_sleep_flag, args.perf_sleep_threshold);
                                    }
                                }
                                if !startup_query_fired && out.episode.is_some() {
                                    startup_query_fired = maybe_startup_query(&args.query, &mut decoder, &episodic, args.action_max_tokens);
                                }
                                if let Some(ref gw) = gain_worker {
                                    if let Some(ep) = &out.episode {
                                        gain_episode_counter += 1;
                                        if gain_episode_counter.is_multiple_of(args.gain_interval) {
                                            gw.request_gain(ep.0.clone());
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
                        video_encode_total_ms += emb.encode_ms;
                        let t_perc = std::time::Instant::now();
                        let vis_result = perception.ingest_visual(&emb.values);
                        perception_total_ms += t_perc.elapsed().as_millis() as u64;
                        match vis_result {
                            Ok(out) => {
                                accumulate_cosines(&out.prior_cosines, &mut prior_cosine_sums, &mut prior_cosine_counts);
                                if out.settle_iters > 0 { settle_trigger_count += 1; settle_iters_sum += out.settle_iters; settle_delta_sum += out.settle_delta as f64; }
                                let rec = record_episode(&out, &emb.timestamp, &mut sink, &mut episodic, args.top_k, &recall);
                                if let Some(ref rec) = rec {
                                    accumulate_salience(rec, &mut salience_sum, &mut salience_n, &mut salience_min, &mut high_salience_episodes, &mut blend_count, &mut episodes_speech, &mut episodes_env, &mut episodes_visual, &mut episodes_language, &mut episodes_mixed);
                                }
                                if let (Some(ep), Some(regional)) = (&out.episode, &out.regional_inputs) {
                                    wake_reconstruct_counter += 1;
                                    if wake_reconstruct_counter.is_multiple_of(args.wake_reconstruct_interval) {
                                        wake_reconstruct_throttle_passed += 1;
                                        if rec.as_ref().is_some_and(|r| r.salience > WAKE_RECONSTRUCT_SALIENCE_THRESHOLD) {
                                            if let Some(ref w) = wake_recon_worker {
                                                if !w.request_train(ep.0.clone(), regional.clone(), perception.learning_rate()) {
                                                    wake_recon_dropped += 1;
                                                }
                                            } else {
                                                match perception.wake_reconstruct_step(&ep.0, regional) {
                                                    Ok(mse) => {
                                                        wake_output_mse_sum += mse as f64;
                                                        wake_output_mse_n += 1;
                                                        tracing::info!(output_mse = mse, "wake reconstruction");
                                                    }
                                                    Err(e) => tracing::warn!(?e, "wake reconstruct error"),
                                                }
                                            }
                                        }
                                        // Refresh autodiff graph every N episodes to prevent
                                        // node accumulation from forward passes slowing into_data().
                                        perception.refresh_autodiff_state();
                                    }
                                    wake_episodes += 1;
                                    if let Some(m) = episode_cosine_mean(&out.prior_cosines) {
                                        update_perf_ewma(m, &mut cosine_ewma, &mut cosine_baseline, &mut cosine_baseline_accum, &mut cosine_baseline_n, &mut perf_sleep_flag, args.perf_sleep_threshold);
                                    }
                                }
                                if !startup_query_fired && out.episode.is_some() {
                                    startup_query_fired = maybe_startup_query(&args.query, &mut decoder, &episodic, args.action_max_tokens);
                                }
                                if let Some(ref gw) = gain_worker {
                                    if let Some(ep) = &out.episode {
                                        gain_episode_counter += 1;
                                        if gain_episode_counter.is_multiple_of(args.gain_interval) {
                                            gw.request_gain(ep.0.clone());
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
            maybe = lang_rx.recv(), if !lang_closed => {
                match maybe {
                    Some(text) => {
                        if let Some(r) = ingest_language_line(
                            &text,
                            &lang_embedder,
                            &mut perception,
                            &mut sink,
                            &mut episodic,
                            args.top_k,
                            &recall,
                            &mut decoder,
                            &mut wake_reconstruct_counter,
                            args.wake_reconstruct_interval,
                            wake_recon_worker.as_ref(),
                            &mut wake_recon_dropped,
                        ) {
                            lang_ingest_count += 1;
                            perception_total_ms += r.perception_ms;
                            if r.episode_produced {
                                wake_episodes += 1;
                                if let Some(m) = episode_cosine_mean(&r.prior_cosines) {
                                    update_perf_ewma(m, &mut cosine_ewma, &mut cosine_baseline, &mut cosine_baseline_accum, &mut cosine_baseline_n, &mut perf_sleep_flag, args.perf_sleep_threshold);
                                }
                                if let Some(ref gw) = gain_worker {
                                    if let Some(ref ep) = r.last_episode {
                                        gain_episode_counter += 1;
                                        if gain_episode_counter.is_multiple_of(args.gain_interval) {
                                            gw.request_gain(ep.clone());
                                        }
                                    }
                                }
                            }
                            if let Some(m) = r.reconstruct_mse { wake_output_mse_sum += m as f64; wake_output_mse_n += 1; }
                            if r.throttle_passed { wake_reconstruct_throttle_passed += 1; }
                            if let Some(ref es) = r.ep_stats {
                                accumulate_salience(es, &mut salience_sum, &mut salience_n, &mut salience_min, &mut high_salience_episodes, &mut blend_count, &mut episodes_speech, &mut episodes_env, &mut episodes_visual, &mut episodes_language, &mut episodes_mixed);
                            }
                            accumulate_cosines(&r.prior_cosines, &mut prior_cosine_sums, &mut prior_cosine_counts);
                        }
                    }
                    None => { lang_closed = true; }
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

    // Flush ORT profiling data before skipping destructors via process::exit.
    // process::exit(0) skips Drop, so profile files would be lost without this explicit call.
    // Only lm_session (CPU) is profiled; GPU sessions use rocprof instead.
    if args.ort_profile_dir.is_some() {
        if let Some(ref mut dec) = decoder {
            match dec.end_profiling() {
                Ok(path) => info!(path, "ORT lm_session profile written"),
                Err(e) => tracing::warn!(?e, "ORT profile flush failed"),
            }
        }
    }

    // MIGraphX EP segfaults during ORT session drop on exit (gfx1201/RDNA4 cleanup bug).
    // All data is written to disk at this point; skip destructors.
    std::process::exit(0)
}

// ── recollection constants ────────────────────────────────────────────────────

/// Blend weight for the retrieved episode: `α*retrieved + (1-α)*current`.
/// 0.1 means recalled memory contributes 10% to the stored episode.
/// Intentional feedback loop: recall shapes future recall (memory-of-memory).
const RECALL_BLEND_ALPHA: f32 = 0.1;

/// L2 distance below which we blend the retrieved episode into the current one.
/// Recalibrated for L2-normalised episode vectors (max L2 = 2.0).
/// Original value (5.5) was calibrated on un-normalised vectors (p95=4.1, max=5.3).
/// For unit vectors, same-content repeats land at dist ≈ 0.05–0.3; different content
/// at ≈ 1.0–1.8. Threshold of 0.5 blends within the same-content distribution only —
/// preserving the intent of "recall shapes future recall for familiar episodes" while
/// rejecting cross-content noise injection.
const RECALL_BLEND_THRESHOLD: f32 = 0.5;

/// Minimum salience (warm-recall L2) for wake_reconstruct_step to fire.
/// Recalibrated for L2-normalised episode vectors (max L2 = 2.0).
/// Original value of 2.0 was at the maximum possible distance for unit vectors,
/// so reconstruction effectively only fired when warm was empty (f32::MAX sentinel).
/// 0.8 fires on content that is genuinely novel (dist > 0.8 ≈ cosine_sim < 0.68)
/// while skipping familiar repeats. f32::MAX always passes (fresh memory dir).
const WAKE_RECONSTRUCT_SALIENCE_THRESHOLD: f32 = 0.8;

// ── language ingest helper ────────────────────────────────────────────────────

/// Result returned by `ingest_language_line` (ADR 0022).
struct LangIngestResult {
    episode_produced: bool,
    /// Raw episode vector when an episode was produced; used by gain worker.
    last_episode: Option<Vec<f32>>,
    reconstruct_mse: Option<f32>,
    prior_cosines: [Option<f32>; 5],
    /// Per-episode stats for cycle-level accumulation; None when no episode was produced.
    ep_stats: Option<RecordedEpisode>,
    /// True when the wake_reconstruct throttle interval passed (before the salience gate).
    throttle_passed: bool,
    /// Settling iterations run for this episode (0 = gate did not fire).
    settle_iters: usize,
    /// Final L2 delta from the settle loop (0.0 when settle_iters=0).
    settle_delta: f32,
    /// Wall-clock ms spent in `perception.ingest_language`.
    perception_ms: u64,
}

/// Embed one stdin line and run the full episode lifecycle through `ingest_language`.
/// No-op when `lang_embedder` is None.
#[allow(clippy::too_many_arguments)]
fn ingest_language_line(
    text: &str,
    lang_embedder: &Option<LangEmbedder>,
    perception: &mut PerceptionProcessor,
    sink: &mut Option<BufWriter<std::fs::File>>,
    episodic: &mut EpisodicBuffer,
    top_k: usize,
    recall: &RecollectionLayer,
    decoder: &mut Option<ActionDecoder>,
    wake_reconstruct_counter: &mut usize,
    wake_reconstruct_interval: usize,
    wake_recon_worker: Option<&WakeReconWorker>,
    wake_recon_dropped: &mut u64,
) -> Option<LangIngestResult> {
    let Some(le) = lang_embedder else { return None };
    let emb = match le.embed(text) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(?e, "language embed failed");
            return None;
        }
    };
    tracing::info!(chars = text.len(), "language ingest");
    let t_perc = std::time::Instant::now();
    match perception.ingest_language(&emb) {
        Ok(out) => {
            let perception_ms = t_perc.elapsed().as_millis() as u64;
            let ts = Timestamp::now();
            let episode_produced = out.episode.is_some();
            let prior_cosines = out.prior_cosines;
            let settle_iters = out.settle_iters;
            let settle_delta = out.settle_delta;
            let ep_stats = record_episode(&out, &ts, sink, episodic, top_k, recall);
            let mut throttle_passed = false;
            let reconstruct_mse =
                if let (Some(ep), Some(regional)) = (&out.episode, &out.regional_inputs) {
                    *wake_reconstruct_counter += 1;
                    if wake_reconstruct_counter.is_multiple_of(wake_reconstruct_interval) {
                        throttle_passed = true;
                        let mse = if ep_stats
                            .as_ref()
                            .is_some_and(|r| r.salience > WAKE_RECONSTRUCT_SALIENCE_THRESHOLD)
                        {
                            if let Some(w) = wake_recon_worker {
                                // Async: loss arrives later via try_recv_ema at the poll site.
                                if !w.request_train(
                                    ep.0.clone(),
                                    regional.clone(),
                                    perception.learning_rate(),
                                ) {
                                    *wake_recon_dropped += 1;
                                }
                                None
                            } else {
                                match perception.wake_reconstruct_step(&ep.0, regional) {
                                    Ok(mse) => {
                                        tracing::info!(output_mse = mse, "wake reconstruction");
                                        Some(mse)
                                    }
                                    Err(e) => {
                                        tracing::warn!(?e, "wake reconstruct error");
                                        None
                                    }
                                }
                            }
                        } else {
                            None
                        };
                        perception.refresh_autodiff_state();
                        mse
                    } else {
                        None
                    }
                } else {
                    None
                };
            // Auto-grounding (ADR 0014): every language ingest (stdin or ASR) becomes
            // a text_target on the episode that was just pushed. This trains the
            // ActionDecoder projection during NREM without requiring manual queries.
            if out.episode.is_some() {
                if let Some(dec) = decoder.as_ref() {
                    match dec.tokenize(text) {
                        Ok(ids) => {
                            episodic.set_last_text_target(ids);
                        }
                        Err(e) => tracing::warn!(?e, "auto-ground tokenize failed"),
                    }
                }
            }
            let last_episode = out.episode.as_ref().map(|ep| ep.0.clone());
            Some(LangIngestResult {
                episode_produced,
                last_episode,
                reconstruct_mse,
                prior_cosines,
                ep_stats,
                throttle_passed,
                settle_iters,
                settle_delta,
                perception_ms,
            })
        }
        Err(e) => {
            tracing::error!(?e, "perception error (language)");
            None
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Fire the --query one-shot generation. Returns true if the query was sent (caller should set
/// startup_query_fired). No-ops when query is None or decoder is absent.
fn maybe_startup_query(
    query: &Option<String>,
    decoder: &mut Option<ActionDecoder>,
    episodic: &EpisodicBuffer,
    max_tokens: usize,
) -> bool {
    let Some(ref q) = query else { return false };
    let Some(ref mut dec) = decoder else {
        return false;
    };
    let Some(ep) = episodic
        .iter_traces()
        .last()
        .map(|t| t.record.values.clone())
    else {
        return false;
    };
    let q_clone = q.clone();
    let result = tokio::task::block_in_place(|| dec.generate(&ep, &q_clone, max_tokens));
    match result {
        Ok(response) => {
            let ts_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            println!(
                "{}",
                serde_json::json!({
                    "type":     "startup_query_response",
                    "ts":       ts_ms,
                    "query":    q_clone,
                    "response": response,
                })
            );
        }
        Err(e) => tracing::warn!(?e, "startup query generation failed"),
    }
    true
}

/// Push an episode into the hot buffer (without touching warm tier — sleep does that).
/// Per-episode stats returned by `record_episode` (ADR 0022).
struct RecordedEpisode {
    /// Warm-recall L2 distance. `f32::MAX` when warm memory is empty (no prior context).
    salience: f32,
    /// Whether the current episode was blended with the top-1 retrieved memory.
    blended: bool,
    /// Dominant modality tag derived from which regional inputs were non-empty.
    source: IngestSource,
}

/// Queries warm memory first; blends the top-1 retrieved vector into the stored
/// episode when it is within RECALL_BLEND_THRESHOLD (L2).
/// Returns `RecordedEpisode` with salience, blend flag, and source for the caller
/// to accumulate cycle-level stats (ADR 0022). Returns `None` when no episode was produced.
fn record_episode(
    out: &somnus_perception::PerceptionOutput,
    ts: &Timestamp,
    sink: &mut Option<BufWriter<std::fs::File>>,
    episodic: &mut EpisodicBuffer,
    top_k: usize,
    recall: &RecollectionLayer,
) -> Option<RecordedEpisode> {
    for &(region, err) in &out.recon_errors {
        tracing::debug!(region = ?region, recon_mse = err, "reconstruction error");
    }
    let Some(ep) = &out.episode else {
        return None;
    };
    let ts_secs = wall_secs(&ts.wall);

    let regional = out
        .regional_inputs
        .clone()
        .unwrap_or([vec![], vec![], vec![], vec![], vec![]]);

    // L2-normalise the episode vector so HNSW uses cosine-equivalent distance.
    let ep_norm = {
        let mag = ep.0.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        ep.0.iter().map(|x| x / mag).collect::<Vec<f32>>()
    };

    // Recollection: find similar past episodes from warm memory.
    let neighbours = recall.query_warm_with_dist(&ep_norm, top_k);
    let (episode_vec, blended, salience) = if let Some(&(dist, rec)) = neighbours.first() {
        tracing::info!(
            top1_id = rec.id,
            top1_timestamp_secs = rec.timestamp_secs,
            l2_dist = dist,
            retrieved = neighbours.len(),
            "recollection"
        );
        if dist < RECALL_BLEND_THRESHOLD {
            let v: Vec<f32> = ep_norm
                .iter()
                .zip(rec.values.iter())
                .map(|(&cur, &ret)| (1.0 - RECALL_BLEND_ALPHA) * cur + RECALL_BLEND_ALPHA * ret)
                .collect();
            (v, true, dist)
        } else {
            (ep_norm.clone(), false, dist)
        }
    } else {
        (ep_norm.clone(), false, f32::MAX)
    };

    // Derive source tag from which regional inputs were non-empty.
    // Regions: 0=Speech, 1=Env, 2=Visual, 3=Symbolic, 4=Language.
    let active: Vec<usize> = (0..5).filter(|&i| !regional[i].is_empty()).collect();
    let ingest_source = match active.as_slice() {
        [0] => IngestSource::Speech,
        [1] => IngestSource::Env,
        [2] => IngestSource::Visual,
        [3] => IngestSource::Unknown,
        [4] => IngestSource::Language,
        _ if active.is_empty() => IngestSource::Unknown,
        _ => IngestSource::Mixed,
    };
    let id = episodic.push(
        ts_secs,
        episode_vec.clone(),
        regional,
        salience,
        ingest_source,
    );
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
    Some(RecordedEpisode {
        salience,
        blended,
        source: ingest_source,
    })
}

/// Accumulate per-episode salience and modality stats into cycle-level counters (ADR 0022).
/// Sentinel `f32::MAX` (warm memory empty) is excluded from salience sums and high_salience count.
#[allow(clippy::too_many_arguments)]
fn accumulate_salience(
    rec: &RecordedEpisode,
    salience_sum: &mut f64,
    salience_n: &mut usize,
    salience_min: &mut f32,
    high_salience_episodes: &mut usize,
    blend_count: &mut usize,
    episodes_speech: &mut usize,
    episodes_env: &mut usize,
    episodes_visual: &mut usize,
    episodes_language: &mut usize,
    episodes_mixed: &mut usize,
) {
    if rec.salience < f32::MAX / 2.0 {
        *salience_sum += rec.salience as f64;
        *salience_n += 1;
        *salience_min = salience_min.min(rec.salience);
        if rec.salience > WAKE_RECONSTRUCT_SALIENCE_THRESHOLD {
            *high_salience_episodes += 1;
        }
    }
    if rec.blended {
        *blend_count += 1;
    }
    match rec.source {
        IngestSource::Speech => *episodes_speech += 1,
        IngestSource::Env => *episodes_env += 1,
        IngestSource::Visual => *episodes_visual += 1,
        IngestSource::Language => *episodes_language += 1,
        _ => *episodes_mixed += 1,
    }
}

/// Accumulate per-region prior cosines into running sums/counts for metrics.
fn accumulate_cosines(cosines: &[Option<f32>; 5], sums: &mut [f64; 5], counts: &mut [usize; 5]) {
    for (i, c) in cosines.iter().enumerate() {
        if let Some(v) = c {
            sums[i] += *v as f64;
            counts[i] += 1;
        }
    }
}

/// Compute the mean of active (non-None) prior_cosines for a single episode.
fn episode_cosine_mean(cosines: &[Option<f32>; 5]) -> Option<f64> {
    let (sum, n) = cosines
        .iter()
        .filter_map(|c| c.map(|v| v as f64))
        .fold((0.0f64, 0usize), |(s, n), v| (s + v, n + 1));
    if n > 0 {
        Some(sum / n as f64)
    } else {
        None
    }
}

const COSINE_EWMA_ALPHA: f64 = 0.1;
const COSINE_BASELINE_N: usize = 20;

/// Update per-episode prior_cosine EWMA and perf-degradation flag.
///
/// Captures a baseline from the first `COSINE_BASELINE_N` episodes; afterwards
/// updates the EWMA and sets `perf_sleep_flag` when the EWMA drops more than
/// `threshold` below the baseline. At threshold == 0.0 the flag is never set.
fn update_perf_ewma(
    episode_mean: f64,
    ewma: &mut Option<f64>,
    baseline: &mut Option<f64>,
    baseline_accum: &mut f64,
    baseline_n: &mut usize,
    perf_sleep_flag: &mut bool,
    threshold: f64,
) {
    // Accumulate baseline from first N episodes.
    if baseline.is_none() {
        *baseline_accum += episode_mean;
        *baseline_n += 1;
        if *baseline_n >= COSINE_BASELINE_N {
            *baseline = Some(*baseline_accum / *baseline_n as f64);
            tracing::debug!(
                cosine_baseline = baseline.unwrap(),
                "perf-ewma baseline captured"
            );
        }
        // Also seed the EWMA from the very first episode.
        *ewma = Some(match *ewma {
            None => episode_mean,
            Some(prev) => COSINE_EWMA_ALPHA * episode_mean + (1.0 - COSINE_EWMA_ALPHA) * prev,
        });
        return;
    }

    // Update EWMA.
    let prev = ewma.unwrap_or(episode_mean);
    let next = COSINE_EWMA_ALPHA * episode_mean + (1.0 - COSINE_EWMA_ALPHA) * prev;
    *ewma = Some(next);

    tracing::debug!(
        cosine_ewma = next,
        cosine_baseline = baseline.unwrap(),
        "perf-ewma update"
    );

    // Set flag if EWMA has dropped below baseline by more than threshold.
    if threshold > 0.0 && !*perf_sleep_flag {
        let drop = baseline.unwrap() - next;
        if drop > threshold {
            tracing::info!(
                cosine_ewma = next,
                cosine_baseline = baseline.unwrap(),
                drop,
                threshold,
                "perf-degradation threshold exceeded — flagging early sleep"
            );
            *perf_sleep_flag = true;
        }
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
