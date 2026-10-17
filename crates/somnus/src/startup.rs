//! Startup stages of `run()`, pulled out of the monolithic entry point: logging, model-file
//! preflight, perception construction, memory restore, ORT/decoder initialisation. Each takes the
//! parsed CLI (and the pieces it needs) and returns what the wake loop will own, so `run()` reads
//! as a sequence of named steps.

use std::{
    io::BufWriter,
    path::{Path, PathBuf},
};

use anyhow::Result;
use burn::tensor::Device;
use somnus_memory::{
    load_region_sizes, load_snapshot, weights_path, ColdArchive, EpisodicBuffer, RecollectionLayer,
    SemanticStore,
};
use somnus_perception::{PerceptionProcessor, WakeReconWorker, EPISODE_DIM};
use sysinfo::System;
use tracing::info;

use crate::{
    action::{self, ActionDecoder, GainWorker},
    cli::Args,
    memory_dir_disk_free_bytes, parse_prior_beta_regions, should_use_ort_inference, thread_budgets,
    whisper_asr::WhisperDecoder,
    LangEmbedder,
};

/// Initialise tracing before ORT so startup errors are captured. With `--log-file`, write
/// directly to that file to avoid the broken-pipe problem under `cargo run` with redirected
/// stderr (cargo's internal pipe breaks when cargo exits).
pub(crate) fn init_tracing(args: &Args) -> Result<()> {
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
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn")),
        )
        .init();
    Ok(())
}

/// Fail fast, with the missing path, when the configured inputs need model files that are absent.
pub(crate) fn check_model_files(args: &Args) -> Result<()> {
    if !args.file.is_empty() {
        #[cfg(not(feature = "mn10_env"))]
        let env_encoder_name = "clap_audio_encoder.onnx";
        #[cfg(feature = "mn10_env")]
        let env_encoder_name = "efficientat_mn10_encoder_fp16.onnx";
        for name in &[
            "whisper_tiny_encoder.onnx",
            env_encoder_name,
            "dinov2_small.onnx",
        ] {
            let p = args.models_dir.join(name);
            if !p.exists() {
                anyhow::bail!("model file not found: {}", p.display());
            }
        }
    }

    if args.language {
        for suffix in &["_embed_table.safetensors", "_tokenizer.json"] {
            let p = args
                .models_dir
                .join(format!("{}{suffix}", args.action_model_prefix));
            if !p.exists() {
                anyhow::bail!("language model file not found: {}", p.display());
            }
        }
    }
    Ok(())
}

/// Build the perception processor and apply the CLI's prior/settle/working-memory/SAE settings.
/// Also returns the static per-region prior betas when `--prior-beta-regions` was given.
pub(crate) fn build_perception(args: &Args) -> Result<(PerceptionProcessor, Option<[f32; 5]>)> {
    let mut perception = PerceptionProcessor::new(Device::flex());
    let prior_beta_regions_static: Option<[f32; 5]> = if let Some(ref s) = args.prior_beta_regions {
        let betas = parse_prior_beta_regions(s)?;
        perception.set_prior_beta(0.0); // clear base first
        for (i, &b) in betas.iter().enumerate() {
            perception.set_prior_beta_region(i, b);
        }
        Some(betas)
    } else {
        perception.set_prior_beta(args.prior_beta);
        None
    };
    perception.set_settle_steps(args.settle_steps);
    perception.set_settle_beta(args.settle_beta);
    perception.set_settle_gate(args.settle_gate);
    perception.set_grown_init_scale(args.grown_unit_init_scale);
    perception.set_settle_every(args.settle_every);
    perception.set_wm_size(args.working_memory_size);
    perception.set_sae_l1_coeff(args.sae_l1_coeff);
    perception.set_batch_topk(args.batch_topk);
    perception.set_der_auto_lambda(args.der_auto_lambda);
    info!(
        prior_beta = args.prior_beta,
        prior_beta_regions = ?prior_beta_regions_static,
        settle_steps = args.settle_steps,
        settle_beta = args.settle_beta,
        working_memory_size = args.working_memory_size,
        batch_topk = args.batch_topk,
        der_auto_lambda = args.der_auto_lambda,
        "perception processor initialised (n=1024, k=20, cpu)"
    );
    Ok((perception, prior_beta_regions_static))
}

/// Process-wide runtime setup: ONNX Runtime, thread budgets (before the first rayon use or
/// encoder session), the RNG seed, the PID file and the model-file preflight.
pub(crate) fn init_process(args: &Args) -> Result<()> {
    ort::init_from("/usr/lib/libonnxruntime.so")?.commit();

    // Thread budgets must be set before the first rayon use / encoder session is created.
    let (rayon_threads, encoder_ort_threads) = thread_budgets(args);
    rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_threads)
        .build_global()
        .map_err(|e| anyhow::anyhow!("rayon pool: {e}"))?;
    somnus_encoding::set_ort_intra_threads(encoder_ort_threads);
    info!(
        rayon_threads,
        encoder_ort_threads,
        logical_cores = std::thread::available_parallelism().map_or(0, |n| n.get()),
        "thread budgets"
    );

    if let Some(seed) = args.seed {
        Device::flex().seed(seed);
        info!(seed, "burn RNG seeded");
    }

    // Write PID file for status scripts.
    if let Some(ref pid_path) = args.pid_file {
        std::fs::write(pid_path, format!("{}\n", std::process::id()))?;
    }

    check_model_files(args)?;
    Ok(())
}

/// Open the `--embeddings-out` NDJSON sink, if requested.
pub(crate) fn open_embedding_sink(args: &Args) -> Result<Option<BufWriter<std::fs::File>>> {
    let sink: Option<BufWriter<std::fs::File>> = if let Some(path) = &args.embeddings_out {
        let f = std::fs::File::create(path)?;
        info!(path = %path.display(), "embedding sink opened");
        Some(BufWriter::new(f))
    } else {
        None
    };
    Ok(sink)
}

/// Grow the freshly built perception model to last session's region sizes. Must happen before
/// the weight restore (which needs the grown shape) and the ORT build (the ONNX is exported
/// from the correct, possibly grown, model).
pub(crate) fn pregrow_perception(args: &Args, perception: &mut PerceptionProcessor) {
    {
        let sizes = load_region_sizes(&args.memory_dir).unwrap_or([somnus_perception::N_UNITS; 5]);
        let regions = [
            somnus_perception::RegionId::AudioSpeech,
            somnus_perception::RegionId::AudioEnv,
            somnus_perception::RegionId::Visual,
            somnus_perception::RegionId::Symbolic,
            somnus_perception::RegionId::Language,
        ];
        let current = perception.region_sizes();
        let mut grew = false;
        for (i, &region) in regions.iter().enumerate() {
            if sizes[i] > current[i] {
                perception.grow_region(region, sizes[i] - current[i]);
                grew = true;
            }
        }
        if grew {
            info!(sizes = ?sizes, "perception pre-grown to match saved region sizes");
        }
    }
}

/// Where the ORT binding graphs live and which script exports them. Both are `None` with
/// `--force-cpu-inference`; they are resolved unconditionally otherwise so the same paths are
/// ready whether the crossover trips at startup (a resumed, already-grown snapshot) or mid-run.
pub(crate) fn resolve_onnx_paths(args: &Args) -> (Option<PathBuf>, Option<PathBuf>) {
    let onnx_dir: Option<PathBuf> = if args.force_cpu_inference {
        None
    } else {
        Some(
            args.onnx_models_dir
                .clone()
                .unwrap_or_else(|| args.memory_dir.join("onnx_models")),
        )
    };
    let onnx_script_path: Option<PathBuf> = onnx_dir.as_ref().map(|_| {
        args.onnx_script.clone().unwrap_or_else(|| {
            // Default: find script relative to the binary
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."))
                .join("../../scripts/export_perception_graph.py")
        })
    });
    (onnx_dir, onnx_script_path)
}

/// What a snapshot restore (or a fresh start) yields for the memory hierarchy.
pub(crate) struct RestoredMemory {
    pub(crate) cold: ColdArchive,
    pub(crate) episodic: EpisodicBuffer,
    pub(crate) semantic: SemanticStore,
    pub(crate) snapshot_found: bool,
}

/// Open the cold archive and restore the episodic buffer, warm index and perception weights from
/// `memory_dir/snapshot`, or start fresh. A failed restore degrades (counted) instead of aborting.
pub(crate) fn restore_memory(
    args: &Args,
    perception: &mut PerceptionProcessor,
) -> Result<RestoredMemory> {
    std::fs::create_dir_all(&args.memory_dir)?;
    let cold = match ColdArchive::open(&args.memory_dir) {
        Ok(c) => c,
        Err(e) => anyhow::bail!("cold archive init failed: {e}"),
    };

    // Restore from snapshot if available; otherwise start fresh.
    let mut snapshot_found = false;
    let (episodic, semantic) =
        match load_snapshot(&args.memory_dir, args.episodic_capacity, EPISODE_DIM) {
            Ok(Some((ep, sem))) => {
                snapshot_found = true;
                info!(
                    episodic_len = ep.len(),
                    semantic_len = sem.len(),
                    "restored from snapshot"
                );
                // Restore perception weights; pre-grow regions if they grew last session.
                // save_weights writes `<stem>.safetensors`; check that, not the stem path itself.
                let w_path = weights_path(&args.memory_dir);
                if w_path.with_extension("safetensors").exists() {
                    let sizes =
                        load_region_sizes(&args.memory_dir).unwrap_or([somnus_perception::N_UNITS; 5]);
                    if let Err(e) = perception.load_weights_with_sizes(&w_path, sizes) {
                        tracing::warn!(?e, "weight restore failed — starting with fresh weights");
                        somnus_metrics::counted_core!(somnus_metrics::Metric::StartupRestoreDegraded, 1);
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
                somnus_metrics::counted_core!(somnus_metrics::Metric::StartupRestoreDegraded, 1);
                (
                    EpisodicBuffer::new(args.episodic_capacity),
                    SemanticStore::new(EPISODE_DIM),
                )
            }
        };
    Ok(RestoredMemory {
        cold,
        episodic,
        semantic,
        snapshot_found,
    })
}

/// Build the ORT binding session if any region is past the crossover. Runs *after* the snapshot
/// restore: an ONNX graph carries frozen weights (ADR 0049). Returns `true` when the attempt
/// failed; that is sticky for the process (the environment problem won't fix itself).
pub(crate) fn init_ort_session(
    args: &Args,
    perception: &mut PerceptionProcessor,
    onnx_dir: Option<&Path>,
    onnx_script_path: Option<&Path>,
) -> Result<bool> {
    let mut onnx_failed = false;
    if let (Some(dir), Some(script)) = (onnx_dir, onnx_script_path) {
        if should_use_ort_inference(&perception.region_sizes(), args.onnx_crossover_units) {
            std::fs::create_dir_all(dir)?;
            info!(
                onnx_dir = %dir.display(),
                migraphx = args.onnx_migraphx,
                region_sizes = ?perception.region_sizes(),
                threshold = args.onnx_crossover_units,
                "region size past ORT crossover — building ORT binding session"
            );
            match perception.rebuild_ort_session(script, dir, args.onnx_migraphx) {
                Ok(()) => info!("ORT binding session active"),
                Err(e) => {
                    tracing::warn!(?e, "ORT session init failed — running with burn inference");
                    somnus_metrics::counted_core!(somnus_metrics::Metric::OrtSessionInitFailed, 1);
                    onnx_failed = true;
                }
            }
        }
    }
    Ok(onnx_failed)
}

/// Warm/cold capacity caps. `--max-warm-records` 0 = auto from free RAM (S5, ADR 0024); `--max-cold-records`
/// 0 = auto from free disk (P1a). Applies the warm cap to `semantic`; returns `(warm, cold)`.
pub(crate) fn apply_memory_caps(args: &Args, semantic: &mut SemanticStore) -> (usize, usize) {
    // S5: auto-seed max_warm_records from available RAM when not set (ADR 0024).
    // Warm HNSW is ~2500 bytes/record (640-dim f32 + graph edges); target 35% of free RAM.
    let effective_max_warm = if args.max_warm_records > 0 {
        args.max_warm_records
    } else {
        let mut sys = System::new();
        sys.refresh_memory();
        let free_bytes = sys.available_memory(); // bytes
        let auto = ((free_bytes as f64 * 0.35) / 2500.0) as usize;
        auto.clamp(1_000, 1_000_000)
    };
    semantic.set_max_records(effective_max_warm);

    // P1a: auto-derive --max-cold-records from disk free space when unset (ADR 0024 ops table).
    // Formula: free_disk_bytes × 0.20 / RECORD_BYTES.
    let cold_record_bytes = somnus_memory::cold::RECORD_BYTES as u64;
    let effective_max_cold = if args.max_cold_records > 0 {
        args.max_cold_records
    } else {
        let free_bytes = memory_dir_disk_free_bytes(&args.memory_dir);
        if free_bytes == 0 {
            // Disk not found via sysinfo (btrfs subvolumes, network fs, etc.).
            // Fall back to unlimited so we don't cap too aggressively.
            tracing::warn!(
                memory_dir = %args.memory_dir.display(),
                "auto cold cap: disk not found via sysinfo — cap disabled"
            );
            somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
            0
        } else {
            let auto = ((free_bytes as f64 * 0.20) / cold_record_bytes as f64) as usize;
            auto.clamp(10_000, 10_000_000)
        }
    };
    tracing::info!(effective_max_cold, "cold archive cap (auto)");
    (effective_max_warm, effective_max_cold)
}

/// Settling without a prior snapshot runs against random output heads: say so.
pub(crate) fn warn_if_settling_without_snapshot(args: &Args, snapshot_found: bool) {
    if args.settle_steps > 0 && !snapshot_found {
        tracing::warn!(
            settle_steps = args.settle_steps,
            "settling active but no prior snapshot — output heads are random; \
             settle noise expected until output_mse converges"
        );
        somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
    }
}

/// Build the action decoder (when `--action`) and restore its projection weights. A failed init
/// disables the decoder instead of aborting.
pub(crate) fn init_action_decoder(args: &Args, snapshot_found: bool) -> Option<ActionDecoder> {
    if args.action {
        match ActionDecoder::new(
            &args.models_dir,
            Device::flex(),
            args.ort_profile_dir.as_deref(),
            args.gpu_generation,
            args.linear_proj,
            &args.action_model_prefix,
        ) {
            Ok(mut d) => {
                // save_weights/load_weights take the stem path and add ".safetensors".
                let dec_w = args.memory_dir.join("snapshot").join("decoder_weights");
                if dec_w.with_extension("safetensors").exists() {
                    if let Err(e) = d.load_weights(&dec_w) {
                        tracing::warn!(?e, "decoder weight restore failed — starting fresh");
                        somnus_metrics::counted_core!(somnus_metrics::Metric::StartupRestoreDegraded, 1);
                    } else {
                        info!("decoder projection weights restored");
                    }
                } else if snapshot_found {
                    tracing::warn!(
                        "snapshot found but no decoder_weights.safetensors — \
                         projection starts fresh; generation incoherent until trained"
                    );
                    somnus_metrics::counted_core!(somnus_metrics::Metric::StartupRestoreDegraded, 1);
                }
                info!("action decoder ready");
                Some(d)
            }
            Err(e) => {
                tracing::warn!(?e, "action decoder init failed — decoder disabled");
                somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
                None
            }
        }
    } else {
        None
    }
}

/// `--export-proj-weights DIR`: dump the projection weights for offline LoRA training (ADR 0033)
/// and exit. No-op without the flag.
pub(crate) fn export_proj_weights_and_exit(args: &Args, decoder: Option<&ActionDecoder>) {
    // Export projection weights for offline LoRA training and exit (ADR 0033).
    if let Some(ref export_dir) = args.export_proj_weights {
        let dec = decoder.expect("--export-proj-weights requires --action");
        std::fs::create_dir_all(export_dir).expect("failed to create export dir");
        let weights = dec.proj_weights_flat();
        // MLP: W1=[640,LM_HIDDEN] then W2=[LM_HIDDEN,N_REGIONS*LM_HIDDEN];
        // linear: W1=[640,N_REGIONS*LM_HIDDEN] only.
        // gaps.md §39: this used to hardcode the Qwen2.5-era 4480 (5*896) for the linear-proj
        // case — silently wrong under Qwen3's 1024-dim LM (5*1024=5120). The default MLP path
        // was never affected (w2 was always "everything remaining").
        let out_dim = action::N_REGIONS * action::LM_HIDDEN;
        let w1_len = if args.linear_proj {
            640 * out_dim
        } else {
            640 * action::LM_HIDDEN
        };
        let w1_bytes: Vec<u8> = weights[..w1_len]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let w2_bytes: Vec<u8> = weights[w1_len..]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        std::fs::write(export_dir.join("w1.bin"), &w1_bytes).expect("w1.bin write failed");
        std::fs::write(export_dir.join("w2.bin"), &w2_bytes).expect("w2.bin write failed");
        println!("exported projection weights to {}", export_dir.display());
        println!(
            "  w1.bin  {} f32  shape=[640,{}]",
            w1_len,
            if args.linear_proj {
                out_dim
            } else {
                action::LM_HIDDEN
            }
        );
        println!(
            "  w2.bin  {} f32  shape=[{},{out_dim}]",
            weights.len() - w1_len,
            if args.linear_proj {
                0
            } else {
                action::LM_HIDDEN
            }
        );
        std::process::exit(0);
    }
}

/// Open (append) the `--metrics-file` NDJSON sink written after every sleep cycle, if requested.
pub(crate) fn open_metrics_sink(args: &Args) -> Result<Option<std::fs::File>> {
    let sink: Option<std::fs::File> = if let Some(ref path) = args.metrics_file {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        info!(path = %path.display(), "metrics file opened");
        Some(f)
    } else {
        None
    };
    Ok(sink)
}

/// Background workers that keep slow per-episode work off the event loop: the Qwen gain worker
/// (needs `--action` and `--action-gain`) and the wake-reconstruction worker (`--wake-recon-async`,
/// created after the perception weights are restored so it starts from them).
pub(crate) fn spawn_workers(
    args: &Args,
    decoder: &mut Option<ActionDecoder>,
    perception: &PerceptionProcessor,
) -> (Option<GainWorker>, Option<WakeReconWorker>) {
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

    let wake_recon_worker: Option<WakeReconWorker> = if args.wake_recon_async {
        info!("wake-recon worker spawned — wake_reconstruct_step running in background");
        Some(perception.create_wake_recon_worker())
    } else {
        None
    };
    (gain_worker, wake_recon_worker)
}

/// Load the language embedder when `--language` is set; a missing model is a hard error.
pub(crate) fn init_language_embedder(args: &Args) -> Result<Option<LangEmbedder>> {
    let lang_embedder: Option<LangEmbedder> = if args.language {
        match LangEmbedder::new(&args.models_dir, &args.action_model_prefix) {
            Ok(le) => Some(le),
            Err(e) => anyhow::bail!("language embedder init failed: {e}"),
        }
    } else {
        None
    };
    Ok(lang_embedder)
}

/// Assemble the warm/cold recollection layer and apply the HNSW search width.
pub(crate) fn build_recall(
    args: &Args,
    semantic: SemanticStore,
    cold: ColdArchive,
    effective_max_warm: usize,
) -> RecollectionLayer {
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
    recall
}

/// Whisper ASR decoder (Stage B, ADR 0013). Soft failure: a missing decoder ONNX just disables ASR.
pub(crate) fn init_whisper(args: &Args) -> Option<WhisperDecoder> {
    if args.language {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// `--language` is a default-on flag with no CLI off switch, so tests turn it off on the struct.
    fn args(extra: &[&str], language: bool) -> Args {
        let mut v = vec!["somnus"];
        v.extend_from_slice(extra);
        let mut a = Args::try_parse_from(v).unwrap();
        a.language = language;
        a
    }

    #[test]
    fn model_files_not_required_without_media_or_language() {
        let dir = tempfile::tempdir().unwrap();
        let a = args(&["--models-dir", dir.path().to_str().unwrap()], false);
        assert!(check_model_files(&a).is_ok());
    }

    #[test]
    fn missing_encoder_for_media_input_names_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let a = args(
            &[
                "--models-dir",
                dir.path().to_str().unwrap(),
                "--file",
                "clip.mp4",
            ],
            false,
        );
        let err = check_model_files(&a).unwrap_err().to_string();
        assert!(err.contains("whisper_tiny_encoder.onnx"), "{err}");
    }

    #[test]
    fn present_encoders_pass_and_missing_language_model_files_fail() {
        let dir = tempfile::tempdir().unwrap();
        for f in [
            "whisper_tiny_encoder.onnx",
            "clap_audio_encoder.onnx",
            "efficientat_mn10_encoder_fp16.onnx",
            "dinov2_small.onnx",
        ] {
            std::fs::write(dir.path().join(f), b"x").unwrap();
        }
        let d = dir.path().to_str().unwrap();
        assert!(check_model_files(&args(&["--models-dir", d, "--file", "c.mp4"], false)).is_ok());
        let err = check_model_files(&args(&["--models-dir", d], true))
            .unwrap_err()
            .to_string();
        assert!(err.contains("language model file not found"), "{err}");
    }

    #[test]
    fn build_perception_applies_cli_settings() {
        let a = args(&["--settle-steps", "3", "--prior-beta", "0.25"], false);
        let (p, regions) = build_perception(&a).unwrap();
        assert!(regions.is_none());
        assert_eq!(p.prior_betas(), [0.25; 5]);
    }
}
