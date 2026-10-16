//! Startup stages of `run()`, pulled out of the monolithic entry point: logging setup, model-file
//! preflight and perception-processor construction. Each takes the parsed CLI and has no hidden
//! state, so they can be read (and tested) on their own.

use anyhow::Result;
use burn::tensor::Device;
use somnus_perception::PerceptionProcessor;
use tracing::info;

use crate::{cli::Args, parse_prior_beta_regions};

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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args(extra: &[&str]) -> Args {
        let mut v = vec!["somnus"];
        v.extend_from_slice(extra);
        Args::try_parse_from(v).unwrap()
    }

    #[test]
    fn model_files_not_required_without_media_or_language() {
        let dir = tempfile::tempdir().unwrap();
        let a = args(&[
            "--models-dir",
            dir.path().to_str().unwrap(),
            "--language",
            "false",
        ]);
        assert!(check_model_files(&a).is_ok());
    }

    #[test]
    fn missing_encoder_for_media_input_names_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let a = args(&[
            "--models-dir",
            dir.path().to_str().unwrap(),
            "--file",
            "clip.mp4",
        ]);
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
        assert!(check_model_files(&args(&[
            "--models-dir",
            d,
            "--file",
            "c.mp4",
            "--language",
            "false"
        ]))
        .is_ok());
        let err = check_model_files(&args(&["--models-dir", d, "--language", "true"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("language model file not found"), "{err}");
    }

    #[test]
    fn build_perception_applies_cli_settings() {
        let a = args(&["--settle-steps", "3", "--prior-beta", "0.25"]);
        let (p, regions) = build_perception(&a).unwrap();
        assert!(regions.is_none());
        assert_eq!(p.prior_betas(), [0.25; 5]);
    }
}
