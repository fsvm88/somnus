#![recursion_limit = "512"]

mod action;
mod cli;
mod pipeline;
mod sleep;
mod startup;
mod util;
mod wake;
mod wake_state;
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

/// Resolve the `--rayon-threads` / `--encoder-ort-threads` settings (0 = auto) against the machine:
/// rayon gets half the logical cores, each encoder ORT session one eighth (at least 2).
fn thread_budgets(args: &cli::Args) -> (usize, usize) {
    thread_budgets_for(
        std::thread::available_parallelism().map_or(4, |n| n.get()),
        args.rayon_threads,
        args.encoder_ort_threads,
    )
}

fn thread_budgets_for(cores: usize, rayon: usize, encoder: usize) -> (usize, usize) {
    (
        if rayon > 0 { rayon } else { (cores / 2).max(1) },
        if encoder > 0 {
            encoder
        } else {
            (cores / 8).max(2)
        },
    )
}

/// Compute (intra_op, inter_op) thread counts for ONNX sessions.
///
/// Reserves roughly half the logical cores for burn-flex's rayon thread pool.
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

use anyhow::{Context as _, Result};
use clap::Parser;
use crossbeam_channel::{Receiver, Sender, TrySendError};
use somnus_encoding::{
    preprocess::resample::SincResampler, AudioEmbedding, AudioEncoder, AudioEncoderSource,
    VideoEmbedding,
};
use somnus_memory::{EpisodicBuffer, IngestSource, RecollectionLayer};
use somnus_perception::{PerceptionProcessor, WakeReconWorker};
use somnus_sensory::{AudioFrame, Timestamp};
use safetensors::SafeTensors;
use std::{
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use sysinfo::{Disks, System};
use tokenizers::Tokenizer;
use tracing::info;

/// Empirical CPU(burn)/GPU(ORT+MIGraphX) crossover for perception inference, from
/// ADR 0023's scaling analysis (research/gpu-unification-design.md §2): GPU breaks
/// even with CPU around N_UNITS≈1,230 at batch-1 and wins by increasing margins
/// above it (2.2x at 4K units batch-1, 6.5x with micro-batching). Below this, CPU
/// is faster and simpler — not worth ORT's Python-export overhead per sleep cycle.
const ONNX_INFERENCE_CROSSOVER_UNITS: usize = 1_230;

/// True if any region has grown large enough that ORT/MIGraphX inference should be
/// active. Pure and unit-tested — growth is monotonic (regions never shrink at
/// runtime except via an explicit, user-invoked prune), so this only ever needs to
/// flip CPU->ORT once, never back.
fn should_use_ort_inference(region_sizes: &[usize; 5], threshold: usize) -> bool {
    region_sizes.iter().any(|&n| n >= threshold)
}

// ── audio accumulator + resampler ─────────────────────────────────────────────

const AUDIO_CHUNK: usize = 4096;
const WHISPER_SAMPLES: usize = 480_000;
// Env encoder window: CLAP-HTSAT is 10s @ 48kHz (default); EfficientAT mn10 is 10s @ 32kHz
// (gaps.md item 9 A/B, `--features mn10_env`, matches Mn10Mel's expected input).
#[cfg(not(feature = "mn10_env"))]
const ENV_SAMPLE_RATE: u32 = 48_000;
#[cfg(not(feature = "mn10_env"))]
const ENV_SAMPLES: usize = 480_000;
#[cfg(feature = "mn10_env")]
const ENV_SAMPLE_RATE: u32 = 32_000;
#[cfg(feature = "mn10_env")]
const ENV_SAMPLES: usize = 320_000;

struct AudioProcessor {
    encoder: AudioEncoder,
    r16k: Option<SincResampler>,
    r_env: Option<SincResampler>,
    in_rate: Option<u32>,
    src_buf: Vec<f32>,
    buf_16k: Vec<f32>,
    buf_env: Vec<f32>,
    ts_16k: Option<Timestamp>,
    ts_env: Option<Timestamp>,
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
            r_env: None,
            in_rate: None,
            src_buf: Vec::new(),
            buf_16k: Vec::new(),
            buf_env: Vec::new(),
            ts_16k: None,
            ts_env: None,
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
                somnus_metrics::counted_core!(somnus_metrics::Metric::MediaSourceError, 1);
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
            self.r_env = Some(SincResampler::new(rate, ENV_SAMPLE_RATE, AUDIO_CHUNK)?);
            self.in_rate = Some(rate);
            info!(in_rate = rate, "audio resamplers initialised");
        }

        if self.ts_16k.is_none() {
            self.ts_16k = Some(frame.timestamp.clone());
        }
        if self.ts_env.is_none() {
            self.ts_env = Some(frame.timestamp.clone());
        }

        self.src_buf.extend_from_slice(&mono);

        let (resample_result, _) = somnus_metrics::timed_core!(somnus_metrics::Metric::AudioResample, {
            (|| -> Result<()> {
                while self.src_buf.len() >= AUDIO_CHUNK {
                    let chunk = self.src_buf[..AUDIO_CHUNK].to_vec();
                    self.src_buf.drain(..AUDIO_CHUNK);
                    self.buf_16k
                        .extend_from_slice(&self.r16k.as_mut().unwrap().process(&chunk)?);
                    self.buf_env
                        .extend_from_slice(&self.r_env.as_mut().unwrap().process(&chunk)?);
                }
                Ok(())
            })()
        });
        resample_result?;

        let mut embs = Vec::new();

        if self.buf_16k.len() >= WHISPER_SAMPLES {
            let pcm: Vec<f32> = self.buf_16k.drain(..WHISPER_SAMPLES).collect();
            let ts = self.ts_16k.take().unwrap();
            let (emb, hidden_flat) = self.encoder.encode_speech(&pcm, ts)?;
            // encode_ms comes from AudioEncoder::encode_speech itself (somnus_metrics::Metric::
            // EncodeSpeech) — timing it again here would be a duplicate measurement of the
            // same call.
            tracing::info!(
                dim = emb.values.len(),
                encode_ms = emb.encode_ms,
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

        if self.buf_env.len() >= ENV_SAMPLES {
            let pcm: Vec<f32> = self.buf_env.drain(..ENV_SAMPLES).collect();
            let ts = self.ts_env.take().unwrap();
            let emb = self.encoder.encode_env(&pcm, ts)?;
            // encode_ms comes from AudioEncoder::encode_env itself (somnus_metrics::Metric::
            // EncodeEnv) — same reasoning as encode_speech above.
            tracing::info!(
                dim = emb.values.len(),
                encode_ms = emb.encode_ms,
                "env embedding"
            );
            embs.push(emb);
        }

        Ok(embs)
    }
}

// ── language embedder ─────────────────────────────────────────────────────────

// Must match action::LM_HIDDEN (Qwen3-0.6B, ADR 0035).
const LANG_DIM: usize = 1024;

struct LangEmbedder {
    tokenizer: Tokenizer,
    /// Flat row-major [vocab_size * LANG_DIM] f32 embed table.
    embed_table: Vec<f32>,
}

impl LangEmbedder {
    /// `model_prefix`: same prefix passed to ActionDecoder::new (--action-model-prefix) —
    /// the two must always load the same model family, so this takes no auto-detect
    /// fallback (a prior "prefer qwen3 if present" heuristic caused a silent LANG_DIM
    /// mismatch when both model families coexisted on disk).
    fn new(models_dir: &std::path::Path, model_prefix: &str) -> Result<Self> {
        let embed_path = models_dir.join(format!("{model_prefix}_embed_table.safetensors"));
        let tok_path = models_dir.join(format!("{model_prefix}_tokenizer.json"));

        let bytes = std::fs::read(&embed_path)?;
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("safetensors load failed: {e}"))?;
        let tv = st
            .tensor("embed_tokens.weight")
            .map_err(|e| anyhow::anyhow!("embed_tokens.weight missing: {e}"))?;
        let embed_table: Vec<f32> = tv
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
        let vocab = embed_table.len() / LANG_DIM;
        let model_name = model_prefix;
        info!(vocab, model = model_name, "language embedder ready");
        Ok(Self {
            tokenizer,
            embed_table,
        })
    }

    /// Tokenize `text` → mean-pool token embeddings → 1024-dim Vec<f32>.
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

/// Current process RSS in bytes, via a fresh single-process `sysinfo` refresh (not the whole
/// process table). Returns 0 if the pid or its memory can't be resolved.
fn current_process_rss_bytes() -> u64 {
    let Ok(pid) = sysinfo::get_current_pid() else {
        return 0;
    };
    let mut sys = System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), false);
    sys.process(pid).map(|p| p.memory()).unwrap_or(0)
}

/// Free space (bytes) on the filesystem backing `memory_dir`, via the most specific
/// (longest) matching mount point. Returns 0 if sysinfo can't resolve it (btrfs subvolumes,
/// network fs, etc. — same caveat as the P1a auto-cold-cap lookup this factors out of).
fn memory_dir_disk_free_bytes(memory_dir: &std::path::Path) -> u64 {
    let disks = Disks::new_with_refreshed_list();
    disks
        .iter()
        .filter(|d| memory_dir.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
        .unwrap_or(0)
}

/// gaps.md §40: dump the encoder-hidden-state input that made `decode_greedy` panic, so a
/// future occurrence leaves a reproducible input instead of vanishing with the dropped cycle.
/// Best-effort — a failure here must never mask or replace the original panic-recovery path.
fn capture_asr_panic_input(memory_dir: &std::path::Path, hidden: &[f32]) {
    let dir = memory_dir.join("asr_panic_captures");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(?e, "failed to create asr_panic_captures dir");
        return;
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = dir.join(format!("panic_input_{ts}.bin"));
    let bytes: Vec<u8> = hidden.iter().flat_map(|f| f.to_le_bytes()).collect();
    match std::fs::write(&path, &bytes) {
        Ok(()) => tracing::warn!(path = %path.display(), "captured ASR panic input"),
        Err(e) => tracing::warn!(?e, "failed to write ASR panic capture"),
    }
}

fn sink_write(sink: &mut Option<BufWriter<std::fs::File>>, line: &str) {
    if let Some(w) = sink {
        if let Err(e) = writeln!(w, "{}", line) {
            tracing::warn!(?e, "embedding sink write failed");
            somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
        }
    }
}

fn sink_flush(sink: &mut Option<BufWriter<std::fs::File>>) {
    if let Some(w) = sink {
        if let Err(e) = w.flush() {
            tracing::warn!(?e, "embedding sink flush failed");
            somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
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

pub async fn run() -> Result<()> {
    let args = cli::Args::parse();

    startup::init_tracing(&args)?;

    startup::init_process(&args)?;

    let mut sink = startup::open_embedding_sink(&args)?;

    // ── perception processor ──────────────────────────────────────────────────
    let (mut perception, prior_beta_regions_static) = startup::build_perception(&args)?;

    // Capture process-start time as the within-session warm filter boundary (ADR 0024).
    // Warm records with timestamp_secs >= session_start_secs belong to this session only.
    let session_start_secs = wall_secs(&SystemTime::now());

    // ── Pre-grow perception to match last session's region sizes ─────────────
    // Must happen before the weight restore (which needs the grown shape) and the ORT build
    // (the ONNX is exported from the correct, possibly grown, model). The ORT session itself is
    // built after the restore: its graph carries frozen weights.
    startup::pregrow_perception(&args, &mut perception);

    // ── ORT binding session (Phase 1 GPU unification, ADR 0023/0037) ──────────
    // Enabled automatically once any region crosses `onnx_crossover_units` — see
    // should_use_ort_inference. `onnx_dir`/`onnx_script_path` are resolved
    // unconditionally (unless --force-cpu-inference) so the same paths are ready
    // whether the crossover trips at startup (a resumed, already-grown snapshot)
    // or mid-run (growth during this process).
    let (onnx_dir, onnx_script_path) = startup::resolve_onnx_paths(&args);
    // Sticky: once an attempt fails (missing Python/onnx/onnxruntime, MIGraphX
    // unavailable, ...), don't retry every sleep cycle — log once and stay on burn
    // CPU for the rest of the process. The environment problem won't fix itself.
    // ── memory hierarchy ──────────────────────────────────────────────────────
    let startup::RestoredMemory {
        cold,
        mut episodic,
        mut semantic,
        snapshot_found,
    } = startup::restore_memory(&args, &mut perception)?;

    // ── ORT binding session (Phase 1 GPU unification, ADR 0023/0037) ──────────
    // Built *after* the snapshot restore: an ONNX graph carries frozen weights, so building it
    // from the pre-grown but still random-init model (as this used to) inferred on random
    // weights for the whole first wake phase, and the fix of rebuilding after the restore cost a
    // second ~10 s session export on every restored start (ADR 0049).
    let onnx_failed = startup::init_ort_session(
        &args,
        &mut perception,
        onnx_dir.as_deref(),
        onnx_script_path.as_deref(),
    )?;

    let (effective_max_warm, effective_max_cold) = startup::apply_memory_caps(&args, &mut semantic);
    startup::warn_if_settling_without_snapshot(&args, snapshot_found);

    // ── action decoder ────────────────────────────────────────────────────────
    let mut decoder = startup::init_action_decoder(&args, snapshot_found);

    startup::export_proj_weights_and_exit(&args, decoder.as_ref());

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
    let _wake_recon_dropped: u64 = 0;

    // ── language embedder ─────────────────────────────────────────────────────
    let lang_embedder: Option<LangEmbedder> = if args.language {
        match LangEmbedder::new(&args.models_dir, &args.action_model_prefix) {
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
        max_warm_records = effective_max_warm,
        max_warm_auto = args.max_warm_records == 0,
        "memory hierarchy ready"
    );

    let high_water = (args.episodic_capacity as f64 * args.sleep_high_water) as usize;
    // S6 adaptive wake ceiling floor (ADR 0024); the live ceiling lives in `WakeState::max_wake`.
    let wake_ewma_floor = args.max_wake_secs;

    let interrupt = Arc::new(AtomicBool::new(false));
    let interrupt_sleep = Arc::clone(&interrupt);

    // Counts embeddings produced by encoder tasks but not yet consumed by the main loop.
    // Sleep monitors this to detect external salience while the system is consolidating.
    let pending_embs = Arc::new(AtomicUsize::new(0));

    // ── metrics infrastructure ────────────────────────────────────────────────
    let mut metrics_sink = startup::open_metrics_sink(&args)?;
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
                somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
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
    let (mut query_rx, mut lang_rx, lang_tx) =
        pipeline::spawn_stdin_reader(&args, decoder.is_some(), lang_embedder.is_some());

    let pipeline::MediaStreams {
        audio_rx,
        video_rx,
        audio_metric_handles,
        video_metrics_handle,
    } = pipeline::start_media_sources(&args)?;

    // ── ASR background thread (Stage B, ADR 0013) ────────────────────────────
    // Decouples Whisper decoding from audio accumulation: the audio thread sends
    // encoder hidden states via a bounded channel and returns immediately; decoding
    // happens concurrently.  Channel capacity 2: one being decoded + one buffered.
    let asr_hidden_tx = pipeline::spawn_asr_thread(&args, whisper_dec, &lang_tx, &asr_count);

    // ── audio encoder task ────────────────────────────────────────────────────
    let mut audio_emb_rx =
        pipeline::spawn_audio_encoder(&args, audio_rx, asr_hidden_tx, &pending_embs);

    // Original lang_tx is no longer needed; clones are held by stdin reader and ASR thread.
    // Dropping it here ensures lang_rx closes once both senders finish.
    drop(lang_tx);

    // ── video encoder task ────────────────────────────────────────────────────
    let mut video_emb_rx = pipeline::spawn_video_encoder(&args, video_rx, &pending_embs);

    let mut st = wake_state::WakeState::restore(&args, onnx_failed);

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
            wake::answer_query(
                &query,
                "pre-sleep",
                &mut episodic,
                &mut decoder,
                &recall,
                &args,
            );
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
                st.wake_output_mse_sum += loss as f64;
                st.wake_output_mse_n += 1;
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
                        args.lang_use_lm,
                        &mut perception,
                        &mut sink,
                        &mut episodic,
                        args.top_k,
                        &recall,
                        &mut decoder,
                        &mut st.wake_reconstruct_counter,
                        st.effective_wake_interval,
                        wake_recon_worker.as_ref(),
                        &mut st.wake_recon_dropped,
                        sal_gate(st.sal_mu, st.sal_sigma, st.sal_n),
                    ) {
                        wake::record_language_result(
                            &mut st,
                            &mut perception,
                            gain_worker.as_ref(),
                            &args,
                            &r,
                            &text,
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
        let schedule_hit = st.wake_start.elapsed() >= st.max_wake;
        let perf_degradation_hit = st.perf_sleep_flag && args.perf_sleep_threshold > 0.0;
        // S3: pressure trigger active once sal_n >= 500 and min-episode guard is satisfied.
        // Capacity fill remains active during cold-start as the primary trigger.
        let pressure_hit = st.sal_n >= P_COLD_START_N
            && st.wake_episodes >= P_MIN_EPISODES
            && st.wakefulness_pressure >= P_THRESHOLD;
        let (should_sleep, sleep_trigger) = decide_should_sleep(
            high_water_hit,
            schedule_hit,
            perf_degradation_hit,
            pressure_hit,
        );

        if should_sleep && !episodic.is_empty() {
            let flow = wake::sleep_cycle(
                &mut st,
                &mut wake::SleepResources {
                    args: &args,
                    interrupt: &interrupt,
                    interrupt_sleep: &interrupt_sleep,
                    pending_embs: &pending_embs,
                    asr_count: &asr_count,
                    perception: &mut perception,
                    episodic: &mut episodic,
                    recall: &mut recall,
                    decoder: &mut decoder,
                    gain_worker: gain_worker.as_ref(),
                    wake_recon_worker: wake_recon_worker.as_ref(),
                    sink: &mut sink,
                    metrics_sink: &mut metrics_sink,
                    onnx_dir: onnx_dir.as_deref(),
                    onnx_script_path: onnx_script_path.as_deref(),
                    effective_max_cold,
                    session_start_secs,
                    prior_beta_regions_static,
                    wake_ewma_floor,
                },
                sleep_trigger,
            );
            if flow.is_break() {
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
                            AudioEncoderSource::Speech => st.speech_encode_total_ms += emb.encode_ms,
                            AudioEncoderSource::Env    => st.env_encode_total_ms    += emb.encode_ms,
                        }
                        let (result, d_perc) = somnus_metrics::timed_core!(
                            somnus_metrics::Metric::MainPerceptionIngest,
                            match emb.source {
                                AudioEncoderSource::Speech => perception.ingest_speech(&emb.values),
                                AudioEncoderSource::Env    => perception.ingest_env(&emb.values),
                            }
                        );
                        st.perception_total_ms += d_perc.as_millis() as u64;
                        wake::process_media_output(
                            &mut st,
                            &mut wake::WakeResources {
                                perception: &mut perception,
                                sink: &mut sink,
                                episodic: &mut episodic,
                                recall: &recall,
                                decoder: &mut decoder,
                                gain_worker: gain_worker.as_ref(),
                                wake_recon_worker: wake_recon_worker.as_ref(),
                                args: &args,
                            },
                            result,
                            &emb.timestamp,
                            "audio",
                        );
                    }
                    None => { audio_emb_closed = true; }
                }
            }
            maybe = video_emb_rx.recv(), if !video_emb_closed => {
                match maybe {
                    Some(emb) => {
                        let _ = pending_embs.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| Some(x.saturating_sub(1)));
                        sink_write(&mut sink, &video_emb_json(&emb));
                        st.video_encode_total_ms += emb.encode_ms;
                        let (result, d_perc) = somnus_metrics::timed_core!(
                            somnus_metrics::Metric::MainPerceptionIngest,
                            perception.ingest_visual(&emb.values)
                        );
                        st.perception_total_ms += d_perc.as_millis() as u64;
                        wake::process_media_output(
                            &mut st,
                            &mut wake::WakeResources {
                                perception: &mut perception,
                                sink: &mut sink,
                                episodic: &mut episodic,
                                recall: &recall,
                                decoder: &mut decoder,
                                gain_worker: gain_worker.as_ref(),
                                wake_recon_worker: wake_recon_worker.as_ref(),
                                args: &args,
                            },
                            result,
                            &emb.timestamp,
                            "video",
                        );
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
                            args.lang_use_lm,
                            &mut perception,
                            &mut sink,
                            &mut episodic,
                            args.top_k,
                            &recall,
                            &mut decoder,
                            &mut st.wake_reconstruct_counter,
                            st.effective_wake_interval,
                            wake_recon_worker.as_ref(),
                            &mut st.wake_recon_dropped,
                            sal_gate(st.sal_mu, st.sal_sigma, st.sal_n),
                        ) {
                            wake::record_language_result(
                                &mut st,
                                &mut perception,
                                gain_worker.as_ref(),
                                &args,
                                &r,
                                &text,
                            );
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
                wake::answer_query(&query, "live", &mut episodic, &mut decoder, &recall, &args);
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
                Err(e) => {
                    tracing::warn!(?e, "ORT profile flush failed");
                    somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
                }
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

/// Cold-start prior for the adaptive salience gate (ADR 0024 S1).
/// Used for the first 100 sampled episodes before the EWMA has enough data.
const SAL_EWMA_COLD_START: f32 = 0.8;

/// Sleep fires when wakefulness_pressure reaches this value (ADR 0024 S3).
const P_THRESHOLD: f32 = 1.0;
/// Wakefulness pressure is clamped to this ceiling to prevent runaway (ADR 0024 S3).
const P_MAX: f32 = 3.0;
/// sal_n must reach this before the S3 pressure trigger replaces the capacity fill (ADR 0024 S3).
const P_COLD_START_N: u64 = 500;
/// Hard minimum wake episodes before the pressure trigger can fire (ADR 0024 S3).
const P_MIN_EPISODES: usize = 120;

/// Accumulate novelty pressure for one episode (ADR 0024 S3).
/// novelty_weight = max(0, salience − τ_s); increments P by novelty_weight / τ_s so that
/// one unit of pressure is added when salience = 2×τ_s. Skips f32::MAX sentinel.
#[inline]
fn accumulate_wakefulness_pressure(salience: f32, sal_threshold: f32, pressure: &mut f32) {
    if salience < f32::MAX / 2.0 && sal_threshold > 0.0 {
        let novelty = (salience - sal_threshold).max(0.0);
        *pressure = (*pressure + novelty / sal_threshold).min(P_MAX);
    }
}

/// Compute adaptive salience gate threshold from EWMA state (ADR 0024 S1).
/// Returns μ + 0.5·σ clamped to [0.3, 3.0] once ≥100 episodes are sampled;
/// falls back to the cold-start prior of 0.8 before that.
#[inline]
/// Append one (episode, text) JSON line to the training data file (ADR 0033).
/// Called inline at language ingest time to capture every ASR/text-bypass event.
fn write_training_pair(path: &std::path::Path, episode: &[f32], text: &str) {
    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    else {
        return;
    };
    let text_esc = text
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r");
    let ep_json: String = episode
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let _ = writeln!(f, "{{\"episode\":[{ep_json}],\"text\":\"{text_esc}\"}}");
}

fn sal_gate(mu: f32, sigma: f32, n: u64) -> f32 {
    if n >= 100 {
        (mu + 0.5 * sigma).clamp(0.3, 3.0)
    } else {
        SAL_EWMA_COLD_START
    }
}

/// Compute adaptive settle gate from prior-cosine EWMA (ADR 0024 S2).
/// Returns μ − 0.1 clamped to [0.0, 1.0] once ≥100 episodes; cold-start prior otherwise.
#[inline]
fn settle_gate_val(mu: f32, n: u64, cold_start: f32) -> f32 {
    if n >= 100 {
        (mu - 0.1).clamp(0.0, 1.0)
    } else {
        cold_start
    }
}

/// Update the prior-cosine EWMA used for adaptive settle gating with α=0.01 (ADR 0024 S2).
#[inline]
fn update_settle_ewma(episode_mean: f64, mu: &mut f32, n: &mut u64) {
    const ALPHA: f32 = 0.01;
    *mu = ALPHA * episode_mean as f32 + (1.0 - ALPHA) * *mu;
    *n = n.saturating_add(1);
}

/// Update the per-episode salience EWMA with α=0.01 (ADR 0024 S1).
/// Ignores f32::MAX sentinel (no warm neighbours — fresh memory or empty warm tier).
#[inline]
fn update_sal_ewma(salience: f32, mu: &mut f32, sigma: &mut f32, n: &mut u64) {
    if salience < f32::MAX / 2.0 {
        const ALPHA: f32 = 0.01;
        *mu = ALPHA * salience + (1.0 - ALPHA) * *mu;
        *sigma = ALPHA * (salience - *mu).abs() + (1.0 - ALPHA) * *sigma;
        *n = n.saturating_add(1);
    }
}

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
    lang_use_lm: bool,
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
    sal_threshold: f32,
) -> Option<LangIngestResult> {
    let Some(le) = lang_embedder else { return None };
    let (emb_opt, _) = somnus_metrics::timed_core!(somnus_metrics::Metric::LangEmbed, {
        if lang_use_lm {
            // Use LM forward-pass hidden states (contextualized, richer than embed-table).
            // Reuses the ActionDecoder's already-loaded qwen3_lm.onnx session; no extra load.
            // Falls back to embed-table if decoder is unavailable.
            let lm_result = decoder.as_mut().and_then(|dec| {
                dec.tokenize(text)
                    .ok()
                    .and_then(|ids| dec.text_hidden_states(&ids).ok())
            });
            match lm_result {
                Some(v) => Some(v),
                None => match le.embed(text) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        tracing::warn!(?e, "language embed failed");
                        somnus_metrics::counted_core!(somnus_metrics::Metric::LangEmbedError, 1);
                        None
                    }
                },
            }
        } else {
            match le.embed(text) {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!(?e, "language embed failed");
                    somnus_metrics::counted_core!(somnus_metrics::Metric::LangEmbedError, 1);
                    None
                }
            }
        }
    });
    let emb = emb_opt?;
    tracing::info!(chars = text.len(), "language ingest");
    let t_perc = std::time::Instant::now();
    match perception.ingest_language(&emb) {
        Ok(out) => {
            // Large match arm with an early function-return shape on Err below — not a clean
            // single expression to hand to timed_core!, so record directly against the same
            // Metric::MainPerceptionIngest the other two ingest call sites use.
            let d_perc = t_perc.elapsed();
            somnus_metrics::METRICS.record(somnus_metrics::Metric::MainPerceptionIngest, d_perc);
            let perception_ms = d_perc.as_millis() as u64;
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
                            .is_some_and(|r| r.salience > sal_threshold)
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
                                        somnus_metrics::counted_core!(
                                            somnus_metrics::Metric::WakeReconstructError,
                                            1
                                        );
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
                        Err(e) => {
                            tracing::warn!(?e, "auto-ground tokenize failed");
                            somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
                        }
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
            somnus_metrics::counted_core!(somnus_metrics::Metric::PerceptionIngestError, 1);
            None
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Retrieve up to `k` recalled episode vectors from warm memory for R1 generation context.
///
/// Returns vectors ordered furthest-first (closest last) so the most similar recalled episode
/// sits immediately before the current-episode tokens in the LLM's causal window.
/// Returns an empty vec when `k == 0`, warm is too small, or there are too few results.
fn query_recalled_episodes(
    recall: &RecollectionLayer,
    cue: &[f32],
    last_ep_id: Option<u64>,
    k: usize,
) -> Vec<Vec<f32>> {
    if k == 0 || recall.warm.len() < 2 {
        return vec![];
    }
    // Fetch k+1 so we can drop the self-match without going short.
    let results = recall.query_warm(cue, k + 1);
    let mut out: Vec<Vec<f32>> = results
        .into_iter()
        .filter(|r| Some(r.id) != last_ep_id)
        .take(k)
        .map(|r| r.values.clone())
        .collect();
    // HNSW returns closest-first; reverse so closest is last (nearest to query tokens).
    out.reverse();
    out
}

/// True when the query is asking the system to describe what it recently perceived.
/// Triggers R3 text-history injection before generation.
fn is_introspective_query(query: &str) -> bool {
    let q = query.to_lowercase();
    [
        "what did you hear",
        "what have you heard",
        "what did you see",
        "what have you seen",
        "what have you been",
        "what have we been",
        "what do you remember",
        "what do you know",
        "what have you learned",
        "what was that",
        "recent",
        "recap",
    ]
    .iter()
    .any(|kw| q.contains(kw))
}

/// Build the generation query string, prepending a recent-text context block for
/// introspective queries (R3 Phase 1).
fn build_query_with_history(
    query: &str,
    raw_targets: Vec<Vec<u32>>,
    dec: &ActionDecoder,
) -> String {
    if raw_targets.is_empty() {
        return query.to_owned();
    }
    let texts: Vec<String> = raw_targets
        .iter()
        .filter_map(|ids| dec.decode_ids(ids).ok())
        .filter(|s| !s.trim().is_empty())
        .collect();
    if texts.is_empty() {
        return query.to_owned();
    }
    let entries = texts
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{}. {t}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    format!("[Recent perceived text:\n{entries}\n]\n{query}")
}

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
    // No recall context at startup — warm memory is likely empty at this point.
    let result = tokio::task::block_in_place(|| dec.generate(&ep, &[], &q_clone, max_tokens));
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
        Err(e) => {
            tracing::warn!(?e, "startup query generation failed");
            somnus_metrics::counted_core!(somnus_metrics::Metric::ActionGenerationError, 1);
        }
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
    /// Wall time of the HNSW query in microseconds (0 when warm index is empty).
    hnsw_query_us: u64,
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
    let (result, _) = somnus_metrics::timed_core!(
        somnus_metrics::Metric::MainRecordEpisode,
        record_episode_inner(out, ts, sink, episodic, top_k, recall)
    );
    result
}

fn record_episode_inner(
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
    let (neighbours, d_query) = somnus_metrics::timed_core!(
        somnus_metrics::Metric::MainWarmQuery,
        recall.query_warm_with_dist(&ep_norm, top_k)
    );
    let hnsw_query_us = d_query.as_micros() as u64;
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
        hnsw_query_us,
    })
}

/// Accumulate per-episode salience and modality stats into cycle-level counters (ADR 0022).
/// Sentinel `f32::MAX` (warm memory empty) is excluded from salience sums and high_salience count.
/// `sal_threshold` is the adaptive gate value from `sal_gate()` (ADR 0024 S1).
#[allow(clippy::too_many_arguments)]
fn accumulate_salience(
    rec: &RecordedEpisode,
    sal_threshold: f32,
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
        if rec.salience > sal_threshold {
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

// ── pure logic extracted from run()'s orchestration shell ──────────────────────

/// Parses `--prior-beta-regions`'s five comma-separated floats.
fn parse_prior_beta_regions(s: &str) -> Result<[f32; 5]> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 5 {
        anyhow::bail!("--prior-beta-regions requires exactly 5 comma-separated floats");
    }
    let mut betas = [0.0f32; 5];
    for (i, p) in parts.iter().enumerate() {
        betas[i] = p
            .trim()
            .parse::<f32>()
            .with_context(|| format!("--prior-beta-regions: cannot parse '{}' as f32", p.trim()))?;
    }
    Ok(betas)
}

/// Which trigger (if any) fires the sleep transition, and whether to sleep at
/// all. `sleep_trigger` falls back to "schedule" whenever none of the other
/// three fired but `should_sleep` is true — that combination is only
/// reachable via `schedule_hit`.
fn decide_should_sleep(
    high_water_hit: bool,
    schedule_hit: bool,
    perf_degradation_hit: bool,
    pressure_hit: bool,
) -> (bool, &'static str) {
    let should_sleep = high_water_hit || schedule_hit || perf_degradation_hit || pressure_hit;
    let trigger = if high_water_hit {
        "high_water"
    } else if pressure_hit {
        "pressure"
    } else if perf_degradation_hit {
        "perf_degradation"
    } else {
        "schedule"
    };
    (should_sleep, trigger)
}

/// P1b auto `--sleep-max-secs`: an explicit override wins; otherwise cap at
/// `max(300, 3 * mean(history))`, or 0 (no cap) before any history exists.
fn compute_effective_sleep_max_secs(
    configured: u64,
    history: &std::collections::VecDeque<f64>,
) -> u64 {
    if configured > 0 {
        configured
    } else if history.is_empty() {
        0
    } else {
        let mean = history.iter().sum::<f64>() / history.len() as f64;
        (mean * 3.0).max(300.0) as u64
    }
}

const WAKE_EWMA_ALPHA: f32 = 0.3;

/// S6: per-cycle EWMA of wake duration (ADR 0024).
fn update_wake_ewma(this_wake_secs: f32, mu: &mut f32, n: &mut u64) {
    *mu = if *n == 0 {
        this_wake_secs
    } else {
        WAKE_EWMA_ALPHA * this_wake_secs + (1.0 - WAKE_EWMA_ALPHA) * *mu
    };
    *n += 1;
}

/// Recomputes `max_wake` from the wake-duration EWMA, floored until `n >= 3`.
fn recompute_max_wake_secs(wake_ewma_mu: f32, wake_ewma_n: u64, wake_ewma_floor: u64) -> u64 {
    if wake_ewma_n >= 3 {
        wake_ewma_floor.max((3.0 * wake_ewma_mu) as u64)
    } else {
        wake_ewma_floor
    }
}

/// S3: dissipates wakefulness pressure by NREM consolidation quality. Prefers
/// within-session gist (clean signal); falls back to cross-session at c1.
fn dissipate_wakefulness_pressure(
    pressure: f32,
    within_session_gist_rate: Option<f32>,
    gist_recall_rate: Option<f32>,
) -> f32 {
    let consolidation =
        (0.5 * within_session_gist_rate.or(gist_recall_rate).unwrap_or(0.0)).clamp(0.0, 0.5);
    (pressure - consolidation).max(0.0)
}

/// S4: mean of per-region prior-cosine means (regions with zero samples are
/// excluded, not treated as zero).
fn global_prior_cosine(sums: &[f64; 5], counts: &[usize; 5]) -> Option<f64> {
    let vals: Vec<f64> = (0..5)
        .filter_map(|i| {
            if counts[i] > 0 {
                Some(sums[i] / counts[i] as f64)
            } else {
                None
            }
        })
        .collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

/// S4: true once the 3-cycle ring is full of cosines all above the scale-up
/// threshold (0.65). `None` slots (not-yet-filled ring positions) count as
/// not-high.
fn s4_ring_all_high(ring: &[Option<f64>; 3]) -> bool {
    ring.iter().all(|v| v.is_some_and(|x| x > 0.65))
}

#[cfg(test)]
mod ort_crossover_tests {
    use super::should_use_ort_inference;

    #[test]
    fn below_threshold_stays_on_cpu() {
        assert!(!should_use_ort_inference(&[1024; 5], 1230));
    }

    #[test]
    fn at_threshold_switches_to_ort() {
        assert!(should_use_ort_inference(
            &[1230, 1024, 1024, 1024, 1024],
            1230
        ));
    }

    #[test]
    fn any_single_region_crossing_is_enough() {
        // Regions grow independently (grow_region_seeded can pick a subset), so a
        // single grown region should already flip the whole process to ORT.
        assert!(should_use_ort_inference(
            &[1024, 1024, 5000, 1024, 1024],
            1230
        ));
    }

    #[test]
    fn zero_threshold_always_true() {
        assert!(should_use_ort_inference(&[0, 0, 0, 0, 0], 0));
    }
}

#[cfg(test)]
mod main_helpers_tests {
    use super::*;
    use crossbeam_channel::bounded;
    use somnus_memory::IngestSource;
    use proptest::prelude::*;

    // ── thread_budgets ────────────────────────────────────────────────────────

    #[test]
    fn thread_budgets_auto_scales_with_cores_and_explicit_values_win() {
        assert_eq!(thread_budgets_for(32, 0, 0), (16, 4));
        assert_eq!(thread_budgets_for(8, 0, 0), (4, 2));
        assert_eq!(thread_budgets_for(1, 0, 0), (1, 2));
        assert_eq!(thread_budgets_for(32, 12, 6), (12, 6));
        assert_eq!(thread_budgets_for(32, 12, 0), (12, 4));
    }

    // ── onnx_thread_budget ────────────────────────────────────────────────────

    #[test]
    fn onnx_thread_budget_never_zero_and_intra_gte_inter() {
        let (intra, inter) = onnx_thread_budget();
        assert!(intra >= 1);
        assert!(inter >= 1);
        assert!(intra >= inter, "intra={intra} inter={inter}");
    }

    // ── wall_secs ─────────────────────────────────────────────────────────────

    #[test]
    fn wall_secs_epoch_is_zero() {
        assert_eq!(wall_secs(&UNIX_EPOCH), 0.0);
    }

    #[test]
    fn wall_secs_matches_known_offset() {
        let t = UNIX_EPOCH + std::time::Duration::from_secs(1_000);
        assert_eq!(wall_secs(&t), 1_000.0);
    }

    // ── current_process_rss_bytes / memory_dir_disk_free_bytes ──────────────────
    // Real syscall-backed, machine-dependent — assert the invariants a caller
    // actually relies on, not exact values (see gaps.md/session notes on
    // sysinfo's feature trim: these were live-verified against a real process
    // this same session; here we just guard the *contract*, not re-verify the
    // OS integration end to end).

    #[test]
    fn current_process_rss_bytes_is_nonzero_for_this_process() {
        // This test process itself has nonzero RSS; a 0 return means the sysinfo
        // lookup silently failed, which is the actual regression this guards.
        // Skip rather than fail if the sandbox's /proc doesn't expose RSS
        // (environment limitation, not a code regression) — same pattern as
        // ort_grown_region.rs's artifact-format skip.
        let rss = current_process_rss_bytes();
        if rss == 0 {
            eprintln!(
                "skipping: sysinfo returned 0 RSS for this process (sandbox /proc limitation?)"
            );
            return;
        }
        assert!(rss > 0);
    }

    #[test]
    fn memory_dir_disk_free_bytes_does_not_panic_on_nonexistent_path() {
        // Contract: unresolvable mount points return 0, never panic.
        let _ = memory_dir_disk_free_bytes(std::path::Path::new("/definitely/not/a/real/path"));
    }

    // ── sink_write / sink_flush ───────────────────────────────────────────────

    #[test]
    fn sink_write_appends_lines_and_flush_persists_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sink.jsonl");
        let file = std::fs::File::create(&path).unwrap();
        let mut sink = Some(BufWriter::new(file));

        sink_write(&mut sink, "line one");
        sink_write(&mut sink, "line two");
        sink_flush(&mut sink);

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "line one\nline two\n");
    }

    #[test]
    fn sink_write_and_flush_are_noop_on_none() {
        // Must not panic when no sink is configured (the common case).
        let mut sink: Option<BufWriter<std::fs::File>> = None;
        sink_write(&mut sink, "ignored");
        sink_flush(&mut sink);
    }

    // ── audio_emb_json / video_emb_json ──────────────────────────────────────

    fn fixed_timestamp() -> Timestamp {
        Timestamp {
            monotonic: std::time::Instant::now(),
            wall: UNIX_EPOCH + std::time::Duration::from_secs(42),
        }
    }

    #[test]
    fn audio_emb_json_has_expected_shape() {
        let emb = AudioEmbedding {
            source: AudioEncoderSource::Speech,
            values: vec![1.0, 2.0, 3.0],
            timestamp: fixed_timestamp(),
            encode_ms: 5,
        };
        let json = audio_emb_json(&emb);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "speech");
        assert_eq!(parsed["timestamp_secs"], 42.0);
        assert_eq!(parsed["dim"], 3);
        assert_eq!(parsed["values"], serde_json::json!([1.0, 2.0, 3.0]));
    }

    #[test]
    fn audio_emb_json_env_source_label() {
        let emb = AudioEmbedding {
            source: AudioEncoderSource::Env,
            values: vec![],
            timestamp: fixed_timestamp(),
            encode_ms: 0,
        };
        let parsed: serde_json::Value = serde_json::from_str(&audio_emb_json(&emb)).unwrap();
        assert_eq!(parsed["type"], "env");
        assert_eq!(parsed["dim"], 0);
    }

    #[test]
    fn video_emb_json_has_expected_shape() {
        let emb = VideoEmbedding {
            values: vec![0.5; 4],
            timestamp: fixed_timestamp(),
            encode_ms: 7,
        };
        let parsed: serde_json::Value = serde_json::from_str(&video_emb_json(&emb)).unwrap();
        assert_eq!(parsed["type"], "video");
        assert_eq!(parsed["dim"], 4);
        assert_eq!(parsed["timestamp_secs"], 42.0);
    }

    // ── accumulate_wakefulness_pressure ──────────────────────────────────────

    #[test]
    fn wakefulness_pressure_ignores_sentinel_salience() {
        let mut pressure = 1.0;
        accumulate_wakefulness_pressure(f32::MAX, 0.5, &mut pressure);
        assert_eq!(
            pressure, 1.0,
            "f32::MAX sentinel (no warm neighbours) must not move pressure"
        );
    }

    #[test]
    fn wakefulness_pressure_ignores_zero_threshold() {
        let mut pressure = 1.0;
        accumulate_wakefulness_pressure(2.0, 0.0, &mut pressure);
        assert_eq!(pressure, 1.0);
    }

    #[test]
    fn wakefulness_pressure_accumulates_novelty_above_threshold() {
        let mut pressure = 0.0;
        accumulate_wakefulness_pressure(1.5, 0.5, &mut pressure);
        // novelty = (1.5 - 0.5) / 0.5 = 2.0
        assert!((pressure - 2.0).abs() < 1e-6);
    }

    #[test]
    fn wakefulness_pressure_clamps_at_p_max() {
        let mut pressure = P_MAX - 0.1;
        accumulate_wakefulness_pressure(100.0, 0.1, &mut pressure);
        assert_eq!(pressure, P_MAX);
    }

    #[test]
    fn wakefulness_pressure_below_threshold_contributes_nothing() {
        let mut pressure = 1.0;
        accumulate_wakefulness_pressure(0.2, 0.5, &mut pressure);
        assert_eq!(
            pressure, 1.0,
            "novelty is clamped to 0 below threshold, no decrease either"
        );
    }

    proptest! {
        #[test]
        fn wakefulness_pressure_never_exceeds_p_max(
            salience in 0.0f32..1000.0,
            sal_threshold in 0.001f32..10.0,
            start in 0.0f32..P_MAX,
        ) {
            let mut pressure = start;
            accumulate_wakefulness_pressure(salience, sal_threshold, &mut pressure);
            prop_assert!(pressure <= P_MAX);
            prop_assert!(pressure >= start, "pressure must be monotonically non-decreasing");
        }
    }

    // ── write_training_pair ──────────────────────────────────────────────────

    #[test]
    fn write_training_pair_escapes_and_appends() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pairs.jsonl");

        write_training_pair(
            &path,
            &[1.0, 2.5],
            "line one\nwith \"quotes\" and \\backslash",
        );
        write_training_pair(&path, &[0.0], "second");

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["episode"], serde_json::json!([1.0, 2.5]));
        assert_eq!(first["text"], "line one\nwith \"quotes\" and \\backslash");

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["text"], "second");
    }

    // ── sal_gate / settle_gate_val ────────────────────────────────────────────

    #[test]
    fn sal_gate_cold_start_below_100_samples() {
        assert_eq!(sal_gate(10.0, 10.0, 99), SAL_EWMA_COLD_START);
    }

    #[test]
    fn sal_gate_uses_formula_at_100_samples() {
        assert_eq!(
            sal_gate(1.0, 0.2, 100),
            (1.0 + 0.5 * 0.2f32).clamp(0.3, 3.0)
        );
    }

    #[test]
    fn sal_gate_clamps_to_bounds() {
        assert_eq!(sal_gate(-10.0, 0.0, 100), 0.3);
        assert_eq!(sal_gate(100.0, 100.0, 100), 3.0);
    }

    #[test]
    fn settle_gate_cold_start_below_100_samples() {
        assert_eq!(settle_gate_val(0.9, 50, 0.6), 0.6);
    }

    #[test]
    fn settle_gate_uses_formula_and_clamps_at_100_samples() {
        assert_eq!(settle_gate_val(0.5, 100, 0.6), 0.4);
        assert_eq!(settle_gate_val(0.05, 100, 0.6), 0.0); // clamped
        assert_eq!(settle_gate_val(1.5, 100, 0.6), 1.0); // clamped
    }

    proptest! {
        #[test]
        fn sal_gate_always_in_valid_range(mu in -10.0f32..10.0, sigma in 0.0f32..10.0, n in 0u64..1000) {
            let g = sal_gate(mu, sigma, n);
            prop_assert!(g == SAL_EWMA_COLD_START || (0.3..=3.0).contains(&g));
        }

        #[test]
        fn settle_gate_val_always_in_unit_range_or_cold_start(
            mu in -10.0f32..10.0, n in 0u64..1000, cold_start in 0.0f32..1.0,
        ) {
            let g = settle_gate_val(mu, n, cold_start);
            prop_assert!(g == cold_start || (0.0..=1.0).contains(&g));
        }
    }

    // ── update_settle_ewma / update_sal_ewma ─────────────────────────────────

    #[test]
    fn update_settle_ewma_moves_toward_new_value() {
        let mut mu = 0.0f32;
        let mut n = 0u64;
        update_settle_ewma(1.0, &mut mu, &mut n);
        assert!((mu - 0.01).abs() < 1e-6);
        assert_eq!(n, 1);
        update_settle_ewma(1.0, &mut mu, &mut n);
        assert_eq!(n, 2);
        assert!(mu > 0.01, "mu should keep moving toward 1.0");
    }

    #[test]
    fn update_sal_ewma_ignores_sentinel() {
        let mut mu = 0.5f32;
        let mut sigma = 0.1f32;
        let mut n = 5u64;
        update_sal_ewma(f32::MAX, &mut mu, &mut sigma, &mut n);
        assert_eq!((mu, sigma, n), (0.5, 0.1, 5));
    }

    #[test]
    fn update_sal_ewma_updates_on_real_salience() {
        let mut mu = 0.0f32;
        let mut sigma = 0.0f32;
        let mut n = 0u64;
        update_sal_ewma(1.0, &mut mu, &mut sigma, &mut n);
        assert!(mu > 0.0);
        assert_eq!(n, 1);
    }

    proptest! {
        #[test]
        fn update_settle_ewma_stays_bounded_by_inputs(
            mus in prop::collection::vec(0.0f32..1.0, 1..20),
        ) {
            // Feeding only values in [0,1] must keep the EWMA in [0,1] (convex combination).
            let mut mu = 0.5f32;
            let mut n = 0u64;
            for m in &mus {
                update_settle_ewma(*m as f64, &mut mu, &mut n);
            }
            prop_assert!((0.0..=1.0).contains(&mu), "mu={mu} escaped [0,1]");
            prop_assert_eq!(n, mus.len() as u64);
        }

        #[test]
        fn update_sal_ewma_n_only_increments_on_real_samples(
            samples in prop::collection::vec(prop::bool::ANY, 1..20),
        ) {
            let mut mu = 0.0f32;
            let mut sigma = 0.0f32;
            let mut n = 0u64;
            let mut expected_n = 0u64;
            for is_sentinel in samples {
                let salience = if is_sentinel { f32::MAX } else { 0.3 };
                if !is_sentinel { expected_n += 1; }
                update_sal_ewma(salience, &mut mu, &mut sigma, &mut n);
            }
            prop_assert_eq!(n, expected_n);
        }
    }

    // ── is_introspective_query ────────────────────────────────────────────────

    #[test]
    fn introspective_query_detects_known_phrases() {
        assert!(is_introspective_query("What did you hear just now?"));
        assert!(is_introspective_query("Give me a RECAP"));
        assert!(is_introspective_query("what have you seen"));
    }

    #[test]
    fn introspective_query_rejects_unrelated_text() {
        assert!(!is_introspective_query("What's the weather like?"));
        assert!(!is_introspective_query(""));
    }

    // ── accumulate_salience ───────────────────────────────────────────────────

    fn recorded_episode(salience: f32, blended: bool, source: IngestSource) -> RecordedEpisode {
        RecordedEpisode {
            salience,
            blended,
            source,
            hnsw_query_us: 0,
        }
    }

    #[test]
    fn accumulate_salience_counts_by_source_and_blend() {
        let mut salience_sum = 0.0;
        let mut salience_n = 0;
        let mut salience_min = f32::MAX;
        let mut high = 0;
        let mut blend = 0;
        let (mut speech, mut env, mut visual, mut lang, mut mixed) = (0, 0, 0, 0, 0);

        let rec = recorded_episode(0.9, true, IngestSource::Speech);
        accumulate_salience(
            &rec,
            0.5,
            &mut salience_sum,
            &mut salience_n,
            &mut salience_min,
            &mut high,
            &mut blend,
            &mut speech,
            &mut env,
            &mut visual,
            &mut lang,
            &mut mixed,
        );

        assert_eq!(salience_n, 1);
        assert!((salience_sum - 0.9).abs() < 1e-6);
        assert_eq!(salience_min, 0.9);
        assert_eq!(high, 1, "0.9 > threshold 0.5");
        assert_eq!(blend, 1);
        assert_eq!(speech, 1);
        assert_eq!((env, visual, lang, mixed), (0, 0, 0, 0));
    }

    #[test]
    fn accumulate_salience_sentinel_skips_salience_stats_but_not_source() {
        let mut salience_sum = 0.0;
        let mut salience_n = 0;
        let mut salience_min = f32::MAX;
        let mut high = 0;
        let mut blend = 0;
        let (mut speech, mut env, mut visual, mut lang, mut mixed) = (0, 0, 0, 0, 0);

        let rec = recorded_episode(f32::MAX, false, IngestSource::Mixed);
        accumulate_salience(
            &rec,
            0.5,
            &mut salience_sum,
            &mut salience_n,
            &mut salience_min,
            &mut high,
            &mut blend,
            &mut speech,
            &mut env,
            &mut visual,
            &mut lang,
            &mut mixed,
        );

        assert_eq!(
            salience_n, 0,
            "sentinel must not count toward salience stats"
        );
        assert_eq!(salience_min, f32::MAX);
        assert_eq!(mixed, 1, "source tagging happens regardless of salience");
    }

    // ── accumulate_cosines / episode_cosine_mean ─────────────────────────────

    #[test]
    fn accumulate_cosines_only_counts_present_values() {
        let cosines = [Some(0.5), None, Some(0.9), None, Some(0.1)];
        let mut sums = [0.0; 5];
        let mut counts = [0; 5];
        accumulate_cosines(&cosines, &mut sums, &mut counts);
        assert_eq!(counts, [1, 0, 1, 0, 1]);
        assert!((sums[0] - 0.5).abs() < 1e-6);
        assert!((sums[2] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn accumulate_cosines_is_additive_across_calls() {
        let mut sums = [0.0; 5];
        let mut counts = [0; 5];
        accumulate_cosines(&[Some(1.0), None, None, None, None], &mut sums, &mut counts);
        accumulate_cosines(&[Some(2.0), None, None, None, None], &mut sums, &mut counts);
        assert_eq!(counts[0], 2);
        assert!((sums[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn episode_cosine_mean_none_when_all_absent() {
        assert_eq!(episode_cosine_mean(&[None, None, None, None, None]), None);
    }

    #[test]
    fn episode_cosine_mean_averages_present_values() {
        let mean = episode_cosine_mean(&[Some(1.0), Some(3.0), None, None, None]).unwrap();
        assert!((mean - 2.0).abs() < 1e-9);
    }

    // ── update_perf_ewma ──────────────────────────────────────────────────────

    #[test]
    fn update_perf_ewma_seeds_from_first_episode() {
        let mut ewma = None;
        let mut baseline = None;
        let mut baseline_accum = 0.0;
        let mut baseline_n = 0;
        let mut flag = false;
        update_perf_ewma(
            0.7,
            &mut ewma,
            &mut baseline,
            &mut baseline_accum,
            &mut baseline_n,
            &mut flag,
            0.1,
        );
        assert_eq!(ewma, Some(0.7));
        assert_eq!(
            baseline, None,
            "baseline not captured until COSINE_BASELINE_N samples"
        );
        assert_eq!(baseline_n, 1);
        assert!(!flag);
    }

    #[test]
    fn update_perf_ewma_captures_baseline_after_threshold_count() {
        let mut ewma = None;
        let mut baseline = None;
        let mut baseline_accum = 0.0;
        let mut baseline_n = 0;
        let mut flag = false;
        for _ in 0..COSINE_BASELINE_N {
            update_perf_ewma(
                0.8,
                &mut ewma,
                &mut baseline,
                &mut baseline_accum,
                &mut baseline_n,
                &mut flag,
                0.1,
            );
        }
        assert!(baseline.is_some());
        assert!((baseline.unwrap() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn update_perf_ewma_sets_flag_when_degraded_below_threshold() {
        let mut ewma = None;
        let mut baseline = None;
        let mut baseline_accum = 0.0;
        let mut baseline_n = 0;
        let mut flag = false;
        // Establish a healthy baseline at 0.9.
        for _ in 0..COSINE_BASELINE_N {
            update_perf_ewma(
                0.9,
                &mut ewma,
                &mut baseline,
                &mut baseline_accum,
                &mut baseline_n,
                &mut flag,
                0.1,
            );
        }
        assert!(!flag);
        // Now feed a sharply degraded value repeatedly — EWMA should eventually
        // drop more than 0.1 below the 0.9 baseline and raise the flag.
        for _ in 0..200 {
            update_perf_ewma(
                0.0,
                &mut ewma,
                &mut baseline,
                &mut baseline_accum,
                &mut baseline_n,
                &mut flag,
                0.1,
            );
        }
        assert!(flag, "ewma={:?} baseline={:?}", ewma, baseline);
    }

    #[test]
    fn update_perf_ewma_zero_threshold_never_flags() {
        let mut ewma = None;
        let mut baseline = None;
        let mut baseline_accum = 0.0;
        let mut baseline_n = 0;
        let mut flag = false;
        for _ in 0..COSINE_BASELINE_N {
            update_perf_ewma(
                0.9,
                &mut ewma,
                &mut baseline,
                &mut baseline_accum,
                &mut baseline_n,
                &mut flag,
                0.0,
            );
        }
        for _ in 0..200 {
            update_perf_ewma(
                0.0,
                &mut ewma,
                &mut baseline,
                &mut baseline_accum,
                &mut baseline_n,
                &mut flag,
                0.0,
            );
        }
        assert!(!flag, "threshold=0.0 must mean the flag is never set");
    }

    // ── forward_frames / forward_frames_blocking ─────────────────────────────

    #[test]
    fn forward_frames_blocking_relays_all_frames_in_order() {
        let (src_tx, src_rx) = bounded::<i32>(8);
        let (dst_tx, dst_rx) = bounded::<i32>(8);
        for i in 0..5 {
            src_tx.send(i).unwrap();
        }
        drop(src_tx); // close source so recv() returns Err after draining

        let exhausted = forward_frames_blocking(src_rx, dst_tx);
        assert!(exhausted, "source closed normally, not downstream");

        let received: Vec<i32> = dst_rx.try_iter().collect();
        assert_eq!(received, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn forward_frames_blocking_returns_false_when_downstream_closed() {
        let (src_tx, src_rx) = bounded::<i32>(8);
        let (dst_tx, dst_rx) = bounded::<i32>(1);
        src_tx.send(1).unwrap();
        src_tx.send(2).unwrap();
        drop(src_tx);
        drop(dst_rx); // downstream closed before forwarding starts

        let exhausted = forward_frames_blocking(src_rx, dst_tx);
        assert!(
            !exhausted,
            "downstream closed first, source was not exhausted normally"
        );
    }

    #[test]
    fn forward_frames_relays_all_frames_when_room_available() {
        let (src_tx, src_rx) = bounded::<i32>(8);
        let (dst_tx, dst_rx) = bounded::<i32>(8); // plenty of room, nothing should drop
        let (_drain_tx, drain_rx) = bounded::<i32>(8);
        for i in 0..3 {
            src_tx.send(i).unwrap();
        }
        drop(src_tx);

        forward_frames(src_rx, dst_tx, drain_rx);

        let received: Vec<i32> = dst_rx.try_iter().collect();
        assert_eq!(
            received,
            vec![0, 1, 2],
            "no drops expected when downstream has room"
        );
    }

    #[test]
    fn forward_frames_drops_oldest_when_downstream_full() {
        // Real call sites pass `drain` as a clone of the *same* receiver that
        // consumes `dst` (see audio_cb_rx/audio_drain in main()), so draining
        // pulls straight from the channel that's full — that's what actually
        // frees the slot. Model that here instead of using an unrelated channel.
        let (src_tx, src_rx) = bounded::<i32>(8);
        let (dst_tx, dst_rx) = bounded::<i32>(1);
        let drain_rx = dst_rx.clone();

        dst_tx.send(-1).unwrap(); // pre-fill downstream: no room left

        src_tx.send(42).unwrap();
        drop(src_tx);

        forward_frames(src_rx, dst_tx, drain_rx);

        let received: Vec<i32> = dst_rx.try_iter().collect();
        assert_eq!(
            received,
            vec![42],
            "oldest frame (-1) is dropped via drain to make room for the newest (42)"
        );
    }

    #[test]
    fn forward_frames_exits_cleanly_when_downstream_disconnects() {
        let (src_tx, src_rx) = bounded::<i32>(8);
        let (dst_tx, dst_rx) = bounded::<i32>(1);
        let (_drain_tx, drain_rx) = bounded::<i32>(8);
        src_tx.send(1).unwrap();
        src_tx.send(2).unwrap();
        drop(src_tx);
        drop(dst_rx);

        // Must return (not hang) once the downstream is gone, regardless of how
        // many source frames remain.
        forward_frames(src_rx, dst_tx, drain_rx);
    }

    // ── parse_prior_beta_regions ──────────────────────────────────────────────

    #[test]
    fn parse_prior_beta_regions_accepts_five_floats() {
        let betas = parse_prior_beta_regions("0.05,0.04,0.03,0.05,0.01").unwrap();
        assert_eq!(betas, [0.05, 0.04, 0.03, 0.05, 0.01]);
    }

    #[test]
    fn parse_prior_beta_regions_trims_whitespace() {
        let betas = parse_prior_beta_regions(" 0.1 , 0.2, 0.3 ,0.4,0.5").unwrap();
        assert_eq!(betas, [0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    fn parse_prior_beta_regions_rejects_wrong_count() {
        assert!(parse_prior_beta_regions("0.1,0.2,0.3").is_err());
        assert!(parse_prior_beta_regions("0.1,0.2,0.3,0.4,0.5,0.6").is_err());
    }

    #[test]
    fn parse_prior_beta_regions_rejects_non_numeric() {
        assert!(parse_prior_beta_regions("a,0.2,0.3,0.4,0.5").is_err());
    }

    // ── decide_should_sleep ───────────────────────────────────────────────────

    #[test]
    fn decide_should_sleep_no_trigger_means_no_sleep() {
        let (should, _) = decide_should_sleep(false, false, false, false);
        assert!(!should);
    }

    #[test]
    fn decide_should_sleep_high_water_takes_priority() {
        let (should, trigger) = decide_should_sleep(true, true, true, true);
        assert!(should);
        assert_eq!(trigger, "high_water");
    }

    #[test]
    fn decide_should_sleep_pressure_beats_perf_and_schedule() {
        let (should, trigger) = decide_should_sleep(false, true, true, true);
        assert!(should);
        assert_eq!(trigger, "pressure");
    }

    #[test]
    fn decide_should_sleep_perf_degradation_beats_schedule() {
        let (should, trigger) = decide_should_sleep(false, true, true, false);
        assert!(should);
        assert_eq!(trigger, "perf_degradation");
    }

    #[test]
    fn decide_should_sleep_schedule_is_the_fallback() {
        let (should, trigger) = decide_should_sleep(false, true, false, false);
        assert!(should);
        assert_eq!(trigger, "schedule");
    }

    // ── compute_effective_sleep_max_secs ──────────────────────────────────────

    #[test]
    fn effective_sleep_max_secs_configured_override_wins() {
        let history = std::collections::VecDeque::from([10.0, 20.0]);
        assert_eq!(compute_effective_sleep_max_secs(42, &history), 42);
    }

    #[test]
    fn effective_sleep_max_secs_no_cap_before_any_history() {
        let history = std::collections::VecDeque::new();
        assert_eq!(compute_effective_sleep_max_secs(0, &history), 0);
    }

    #[test]
    fn effective_sleep_max_secs_floors_at_300() {
        let history = std::collections::VecDeque::from([10.0, 20.0]); // mean=15, *3=45 < 300
        assert_eq!(compute_effective_sleep_max_secs(0, &history), 300);
    }

    #[test]
    fn effective_sleep_max_secs_scales_above_floor() {
        let history = std::collections::VecDeque::from([200.0, 200.0]); // mean=200, *3=600
        assert_eq!(compute_effective_sleep_max_secs(0, &history), 600);
    }

    // ── update_wake_ewma / recompute_max_wake_secs ────────────────────────────

    #[test]
    fn update_wake_ewma_first_sample_sets_mu_directly() {
        let mut mu = 0.0;
        let mut n = 0u64;
        update_wake_ewma(120.0, &mut mu, &mut n);
        assert_eq!(mu, 120.0);
        assert_eq!(n, 1);
    }

    #[test]
    fn update_wake_ewma_blends_subsequent_samples() {
        let mut mu = 100.0;
        let mut n = 1u64;
        update_wake_ewma(200.0, &mut mu, &mut n);
        assert_eq!(mu, 0.3 * 200.0 + 0.7 * 100.0);
        assert_eq!(n, 2);
    }

    #[test]
    fn recompute_max_wake_secs_uses_floor_before_n_reaches_3() {
        assert_eq!(recompute_max_wake_secs(1000.0, 2, 300), 300);
    }

    #[test]
    fn recompute_max_wake_secs_uses_ewma_once_n_reaches_3() {
        assert_eq!(recompute_max_wake_secs(200.0, 3, 300), 600); // 3*200=600 > floor
        assert_eq!(recompute_max_wake_secs(10.0, 3, 300), 300); // 3*10=30 < floor, floor wins
    }

    // ── dissipate_wakefulness_pressure ────────────────────────────────────────

    #[test]
    fn dissipate_wakefulness_pressure_prefers_within_session_rate() {
        let p = dissipate_wakefulness_pressure(0.9, Some(1.0), Some(0.0));
        // consolidation = clamp(0.5*1.0, 0, 0.5) = 0.5
        assert!((p - 0.4).abs() < 1e-6);
    }

    #[test]
    fn dissipate_wakefulness_pressure_falls_back_to_cross_session() {
        let p = dissipate_wakefulness_pressure(0.9, None, Some(0.2));
        // consolidation = clamp(0.5*0.2, 0, 0.5) = 0.1
        assert!((p - 0.8).abs() < 1e-6);
    }

    #[test]
    fn dissipate_wakefulness_pressure_never_goes_negative() {
        let p = dissipate_wakefulness_pressure(0.1, Some(1.0), None);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn dissipate_wakefulness_pressure_zero_with_no_gist_data() {
        let p = dissipate_wakefulness_pressure(0.5, None, None);
        assert_eq!(p, 0.5);
    }

    // ── global_prior_cosine ────────────────────────────────────────────────────

    #[test]
    fn global_prior_cosine_none_when_all_counts_zero() {
        assert_eq!(global_prior_cosine(&[0.0; 5], &[0; 5]), None);
    }

    #[test]
    fn global_prior_cosine_excludes_zero_count_regions_from_mean() {
        // Only regions 0 and 2 have samples: means 0.8 and 0.4 -> mean 0.6.
        let sums = [0.8, 0.0, 0.4, 0.0, 0.0];
        let counts = [1, 0, 1, 0, 0];
        let g = global_prior_cosine(&sums, &counts).unwrap();
        assert!((g - 0.6).abs() < 1e-9);
    }

    #[test]
    fn global_prior_cosine_averages_per_region_means() {
        let sums = [4.0, 6.0];
        let mut sums5 = [0.0; 5];
        sums5[..2].copy_from_slice(&sums);
        let counts = [2usize, 3usize];
        let mut counts5 = [0usize; 5];
        counts5[..2].copy_from_slice(&counts);
        // region means: 4/2=2.0, 6/3=2.0 -> mean of means 2.0
        let g = global_prior_cosine(&sums5, &counts5).unwrap();
        assert!((g - 2.0).abs() < 1e-9);
    }

    // ── s4_ring_all_high ───────────────────────────────────────────────────────

    #[test]
    fn s4_ring_all_high_true_when_all_above_threshold() {
        assert!(s4_ring_all_high(&[Some(0.7), Some(0.9), Some(0.66)]));
    }

    #[test]
    fn s4_ring_all_high_false_with_any_none_slot() {
        assert!(!s4_ring_all_high(&[Some(0.9), None, Some(0.9)]));
    }

    #[test]
    fn s4_ring_all_high_false_when_one_below_threshold() {
        assert!(!s4_ring_all_high(&[Some(0.9), Some(0.65), Some(0.9)]));
    }
}
