//! The wake loop's I/O plumbing, split out of `run()`: the stdin reader, the media sources
//! (file / microphone / webcam) and their async bridges, the ASR thread and the audio/video
//! encoder tasks. Each `spawn_*`/`start_*` function returns the channel end the loop consumes.
//! The bodies are the former inline blocks, moved verbatim.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::Result;
use crossbeam_channel::{bounded, TrySendError};
use somnus_encoding::{AudioEmbedding, VideoEmbedding, VideoEncoder};
use somnus_sensory::{AudioCapture, AudioFrame, MediaFile, SensorMetrics, VideoCapture, VideoFrame};
use tokio::sync::mpsc;

use crate::{
    capture_asr_panic_input, cli::Args, forward_frames, forward_frames_blocking,
    whisper_asr::WhisperDecoder, AudioProcessor,
};

/// Route stdin lines: `query_rx` carries lines for the action decoder (ADR 0012), `lang_rx` lines
/// for language-region ingestion (ADR 0013). The ASR thread also feeds `lang_rx` through the
/// returned `lang_tx` (the caller drops it once every other sender is cloned off). The reader
/// only runs when a decoder or a language embedder exists.
pub(crate) fn spawn_stdin_reader(
    args: &Args,
    has_decoder: bool,
    has_lang: bool,
) -> (
    mpsc::Receiver<String>,
    mpsc::Receiver<String>,
    mpsc::Sender<String>,
) {
    let (query_tx, query_rx) = tokio::sync::mpsc::channel::<String>(8);
    let (lang_tx, lang_rx) = tokio::sync::mpsc::channel::<String>(64);
    {
        let needs_stdin = has_decoder || has_lang;
        if needs_stdin {
            let qtx = if has_decoder && !args.action_distill_only {
                Some(query_tx)
            } else {
                None
            };
            // Clone lang_tx for the stdin reader; the original stays in scope for the audio task.
            let ltx_stdin = if has_lang {
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
    (query_rx, lang_rx, lang_tx)
}

/// The decoded media streams the encoder tasks consume, plus the sensor-metrics handles the
/// periodic metrics tick reads.
pub(crate) struct MediaStreams {
    pub(crate) audio_rx: mpsc::Receiver<AudioFrame>,
    pub(crate) video_rx: mpsc::Receiver<VideoFrame>,
    pub(crate) audio_metric_handles: Vec<Arc<SensorMetrics>>,
    pub(crate) video_metrics_handle: Arc<SensorMetrics>,
}

/// Start the audio/video sources selected by the CLI (looping or one-shot file(s), microphone,
/// webcam), fan them into bounded channels and bridge those to async receivers.
pub(crate) fn start_media_sources(args: &Args) -> Result<MediaStreams> {
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
                            somnus_metrics::counted_core!(somnus_metrics::Metric::MediaSourceError, 1);
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
                            somnus_metrics::counted_core!(somnus_metrics::Metric::MediaSourceError, 1);
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
                somnus_metrics::counted_core!(somnus_metrics::Metric::MediaSourceError, 1);
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
                            somnus_metrics::counted_core!(somnus_metrics::Metric::MediaSourceError, 1);
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
                somnus_metrics::counted_core!(somnus_metrics::Metric::MediaSourceError, 1);
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
    Ok(MediaStreams {
        audio_rx,
        video_rx,
        audio_metric_handles,
        video_metrics_handle,
    })
}

/// Whisper ASR on its own thread (Stage B, ADR 0013): the audio task sends encoder hidden states
/// through a bounded channel and returns immediately; decoded text goes to `lang_tx`. `None`
/// when no decoder is available.
pub(crate) fn spawn_asr_thread(
    args: &Args,
    whisper_dec: Option<WhisperDecoder>,
    lang_tx: &mpsc::Sender<String>,
    asr_count: &Arc<AtomicUsize>,
) -> Option<std::sync::mpsc::SyncSender<Vec<f32>>> {
    if let Some(dec) = whisper_dec {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(2);
        let lang_tx_asr = lang_tx.clone();
        let asr_cnt = Arc::clone(asr_count);
        let asr_min_chars = args.asr_min_chars;
        let memory_dir = args.memory_dir.clone();
        std::thread::spawn(move || {
            let mut dec = dec;
            // gaps.md §40: does one ORT panic leave the Session's ORT-internal state
            // (not a Rust Mutex — no poisoning primitive is involved) permanently
            // degraded, or does the thread fully recover? Tracked in-thread since this
            // is the only thread that ever touches `dec`/`Session`.
            let mut panicked_this_session = false;
            while let Ok(hidden) = rx.recv() {
                // ORT's Whisper decoder occasionally panics instead of returning Err
                // (e.g. "GetElementType is not implemented") — an intermittent, unfixed
                // ORT-side issue (see ai-docs/a_b_qwen3_findings.md). catch_unwind stops
                // that from silently killing this thread and ending ASR for the rest of
                // the session; we just drop the one bad cycle and keep decoding.
                let hidden_for_capture = hidden.clone();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    dec.decode_greedy(&hidden, 200)
                }));
                let failed = !matches!(result, Ok(Ok(_)));
                if panicked_this_session {
                    if failed {
                        somnus_metrics::counted_core!(somnus_metrics::Metric::AsrPostPanicFailure, 1);
                    } else {
                        somnus_metrics::counted_core!(somnus_metrics::Metric::AsrPostPanicSuccess, 1);
                    }
                }
                match result {
                    Ok(Ok(text)) if text.len() >= asr_min_chars.max(1) => {
                        tracing::info!(chars = text.len(), "whisper ASR");
                        asr_cnt.fetch_add(1, Ordering::Relaxed);
                        let _ = lang_tx_asr.blocking_send(text);
                    }
                    Ok(Ok(text)) if !text.is_empty() => {
                        tracing::debug!(
                            chars = text.len(),
                            "whisper ASR filtered (below min_chars)"
                        );
                    }
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => {
                        tracing::warn!(?e, "whisper ASR failed");
                        somnus_metrics::counted_core!(somnus_metrics::Metric::AsrError, 1);
                    }
                    Err(_) => {
                        tracing::warn!("whisper ASR panicked — dropping this cycle");
                        somnus_metrics::counted_core!(somnus_metrics::Metric::AsrPanic, 1);
                        panicked_this_session = true;
                        capture_asr_panic_input(&memory_dir, &hidden_for_capture);
                    }
                }
            }
        });
        Some(tx)
    } else {
        None
    }
}

/// Audio encoder task: frames in, embeddings out. Each embedding bumps `pending_embs` (sleep
/// watches it to detect external salience).
pub(crate) fn spawn_audio_encoder(
    args: &Args,
    audio_rx: mpsc::Receiver<AudioFrame>,
    asr_hidden_tx: Option<std::sync::mpsc::SyncSender<Vec<f32>>>,
    pending_embs: &Arc<AtomicUsize>,
) -> mpsc::Receiver<AudioEmbedding> {
    let (audio_emb_tx, audio_emb_rx) = mpsc::channel::<AudioEmbedding>(32);
    {
        let models_dir = args.models_dir.clone();
        let mut audio_rx = audio_rx;
        let pending = Arc::clone(pending_embs);
        tokio::task::spawn_blocking(move || {
            let mut proc = match AudioProcessor::new(&models_dir, asr_hidden_tx) {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!(?e, "audio encoder init failed");
                    somnus_metrics::counted_core!(somnus_metrics::Metric::MediaSourceError, 1);
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
                    Err(e) => {
                        tracing::error!(?e, "audio encode error");
                        somnus_metrics::counted_core!(somnus_metrics::Metric::MediaSourceError, 1);
                    }
                }
            }
        });
    }
    audio_emb_rx
}

/// Video encoder task: frames in (stride / change-gating applied), embeddings out.
pub(crate) fn spawn_video_encoder(
    args: &Args,
    video_rx: mpsc::Receiver<VideoFrame>,
    pending_embs: &Arc<AtomicUsize>,
) -> mpsc::Receiver<VideoEmbedding> {
    let (video_emb_tx, video_emb_rx) = mpsc::channel::<VideoEmbedding>(16);
    {
        let models_dir = args.models_dir.clone();
        let mut video_rx = video_rx;
        let pending = Arc::clone(pending_embs);
        let video_stride = args.video_stride.max(1);
        let video_change_threshold = args.video_change_threshold;
        let video_change_max_skip = args.video_change_max_skip.max(1);
        tokio::task::spawn_blocking(move || {
            let mut enc = match VideoEncoder::new(&models_dir) {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!(?e, "video encoder init failed");
                    somnus_metrics::counted_core!(somnus_metrics::Metric::MediaSourceError, 1);
                    return;
                }
            };
            let mut last_log = std::time::Instant::now();
            let mut frames_this_sec: u32 = 0;
            let mut total_ms_this_sec: u128 = 0;
            let mut frame_counter: usize = 0;
            let mut skipped_this_sec: u32 = 0;
            let mut last_encoded_frame: Option<VideoFrame> = None;
            let mut change_skip_streak: usize = 0;
            while let Some(frame) = video_rx.blocking_recv() {
                frame_counter += 1;
                if !frame_counter.is_multiple_of(video_stride) {
                    continue;
                }
                if video_change_threshold > 0.0 {
                    if let Some(prev) = &last_encoded_frame {
                        if frame.mean_abs_diff(prev) < video_change_threshold
                            && change_skip_streak < video_change_max_skip
                        {
                            change_skip_streak += 1;
                            skipped_this_sec += 1;
                            continue;
                        }
                    }
                    change_skip_streak = 0;
                }
                match enc.encode(&frame) {
                    Ok(emb) => {
                        // encode_ms comes from VideoEncoder::encode itself
                        // (somnus_metrics::Metric::EncodeVideo) — avoids a duplicate timer.
                        let encode_ms = emb.encode_ms as u128;
                        frames_this_sec += 1;
                        total_ms_this_sec += encode_ms;
                        tracing::debug!(dim = emb.values.len(), encode_ms, "video embedding");
                        if last_log.elapsed() >= Duration::from_secs(1) {
                            let avg_ms = total_ms_this_sec / frames_this_sec.max(1) as u128;
                            tracing::info!(
                                frames = frames_this_sec,
                                skipped = skipped_this_sec,
                                dim = emb.values.len(),
                                avg_encode_ms = avg_ms,
                                "video embeddings/s"
                            );
                            somnus_metrics::counted_core!(
                                somnus_metrics::Metric::VideoFramesEncoded,
                                frames_this_sec as u64
                            );
                            somnus_metrics::counted_core!(
                                somnus_metrics::Metric::VideoFramesSkipped,
                                skipped_this_sec as u64
                            );
                            last_log = std::time::Instant::now();
                            frames_this_sec = 0;
                            total_ms_this_sec = 0;
                            skipped_this_sec = 0;
                        }
                        if video_change_threshold > 0.0 {
                            last_encoded_frame = Some(frame);
                        }
                        if video_emb_tx.blocking_send(emb).is_ok() {
                            pending.fetch_add(1, Ordering::Relaxed);
                        } else {
                            return;
                        }
                    }
                    Err(e) => {
                        tracing::error!(?e, "video encode error");
                        somnus_metrics::counted_core!(somnus_metrics::Metric::MediaSourceError, 1);
                    }
                }
            }
        });
    }
    video_emb_rx
}
