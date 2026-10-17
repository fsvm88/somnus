//! Command-line arguments for the `somnus` binary.

use clap::Parser;
use std::path::PathBuf;

use crate::ONNX_INFERENCE_CROSSOVER_UNITS;

#[derive(Debug, Parser)]
#[command(about = "somnus sensory + encoding + perception + memory + sleep pipeline — Phase 5")]
pub(crate) struct Args {
    /// One or more media files to use as input. May be specified multiple times.
    /// In loop mode (--file-loop) the files cycle in order indefinitely.
    /// Without --file-loop each file is played once and then the process exits.
    #[arg(long, num_args = 1..)]
    pub(crate) file: Vec<PathBuf>,

    #[arg(long)]
    pub(crate) mic: bool,

    #[arg(long, default_value = "0")]
    pub(crate) webcam: u32,

    #[arg(long, default_value = "models")]
    pub(crate) models_dir: PathBuf,

    /// Seed burn's global RNG (parameter init, dropout-style sampling) so runs are
    /// reproducible. Unset = OS entropy, as before. Logged at startup either way
    /// for the explicit case only; A/B scripts should always pass it.
    #[arg(long)]
    pub(crate) seed: Option<u64>,

    /// L2 norm of the encoder column given to each unit added by seeded region growth. Default
    /// 0.01 is the historical value; original units sit at ~0.5-0.6, so grown units start ~50x
    /// weaker and mostly win k-WTA slots via the duty-cycle boost with near-zero activations
    /// (gaps.md §50). Experiment knob; do not change the default without an A/B.
    #[arg(long, default_value = "0.01")]
    pub(crate) grown_unit_init_scale: f32,

    /// Size of the global rayon pool that burn-flex's CPU kernels run on. 0 = auto: half the
    /// logical cores. The encoder/ASR ONNX sessions share the machine with it; with every library
    /// at "all cores" the wake loop ran 2x slower (ADR 0049 pilot, gaps.md Q14/R5).
    #[arg(long, default_value = "0")]
    pub(crate) rayon_threads: usize,

    /// Intra-op threads for each audio/video encoder ONNX session. 0 = auto: one eighth of the
    /// logical cores, at least 2 (the GPU does the heavy lifting; the threads only run fallback
    /// ops and pre/post-processing).
    #[arg(long, default_value = "0")]
    pub(crate) encoder_ort_threads: usize,

    #[arg(long)]
    pub(crate) embeddings_out: Option<PathBuf>,

    /// Maximum number of episodes in the hot episodic buffer.
    #[arg(long, default_value = "4096")]
    pub(crate) episodic_capacity: usize,

    /// Directory for memory files (cold archive, snapshots).
    #[arg(long, default_value = "memory")]
    pub(crate) memory_dir: PathBuf,

    /// Number of nearest episodes to retrieve on each query (logged at debug level).
    #[arg(long, default_value = "5")]
    pub(crate) top_k: usize,

    /// Fraction of episodic capacity that triggers a sleep cycle (0.0–1.0).
    #[arg(long, default_value = "0.8")]
    pub(crate) sleep_high_water: f64,

    /// Maximum wake duration in seconds before a sleep cycle is forced.
    #[arg(long, default_value = "600")]
    pub(crate) max_wake_secs: u64,

    /// Top-down prior blend weight β. Small non-zero base so cosines are always
    /// measured from episode 1 (ADR 0024 S4 chicken-and-egg fix — `blend_with_prior`
    /// returns no cosine at all when this is exactly 0.0, freezing S2's settle EWMA
    /// at cold start); blend stays negligible until S4 scales it up once predictions
    /// mature (3-cycle cosine ring all >0.65). Pass 0.0 explicitly to fully disable.
    #[arg(long, default_value = "0.01")]
    pub(crate) prior_beta: f32,

    /// Per-region prior blend weights as five comma-separated floats:
    /// speech,env,visual,symbolic,language (e.g. "0.05,0.04,0.03,0.05,0.01").
    /// When set, overrides --prior-beta for each region independently and
    /// disables the S4 global controller (static mode).
    #[arg(long)]
    pub(crate) prior_beta_regions: Option<String>,

    /// Number of settling iterations per episode (0 = disabled). Default 1 — the
    /// value every real campaign has run since 2026-05-31 (ADR 0047 hardened this
    /// into the default 2026-08-16); `SETTLE_EPS` is calibrated for it (see
    /// `somnus-perception::SETTLE_EPS` doc comment). A cold-start WARN still fires
    /// with no prior snapshot (random output heads distort early settling inputs,
    /// ADR 0011) — harmless noise, not a functional gate.
    #[arg(long, default_value = "1")]
    pub(crate) settle_steps: usize,

    /// Settling blend weight β: fraction of output-head prediction mixed into
    /// settling inputs. 0.0 = raw only (no-op loop); 1.0 = pure prediction.
    #[arg(long, default_value = "0.5")]
    pub(crate) settle_beta: f32,

    /// Cosine similarity gate for settling: settle fires when prior_cosine < this
    /// threshold. Raise toward 1.0 to reduce settle frequency. Default 0.85.
    #[arg(long, default_value = "0.85")]
    pub(crate) settle_gate: f32,

    /// Run settling only every N episodes (1 = every episode). Subsamples settle
    /// to reduce GPU cost. E.g. --settle-every 2 halves settle overhead.
    #[arg(long, default_value = "1")]
    pub(crate) settle_every: usize,

    /// Working-memory ring size: number of recent episode vectors to carry as
    /// cross-attention context in the binding area (ADR 0010/10d). Default 8 —
    /// validated (roadmap.md §10d): WM=8 vs WM=0 on the same snapshot, output_mse
    /// 40-50% lower with WM (ADR 0047 hardened this into the default 2026-08-16).
    /// 0 disables the ring entirely. Cleared at sleep entry (wake-local by design).
    #[arg(long, default_value = "8")]
    pub(crate) working_memory_size: usize,

    /// Cap on the number of records in the warm HNSW index. When warm reaches
    /// this limit, the index is rebuilt during sleep keeping only the most recent
    /// N records from cold. Default 200_000 (~500 MB at 640-dim f32) — the value
    /// every real campaign has run since 2026-06-17 (ADR 0047 hardened this into
    /// the default 2026-08-16). 0 = auto-derive from free RAM instead
    /// (`free_ram_bytes×0.35/2500`, clamped [1_000, 1_000_000], ADR 0024 S5) —
    /// pass 0 explicitly to opt back into the auto formula.
    #[arg(long, default_value = "200000")]
    pub(crate) max_warm_records: usize,

    /// ef parameter for HNSW warm-memory queries. Higher ef = more accurate recall
    /// but slower search. Calibrated default 16 (ADR 0024 S5) — ~4× faster than 64
    /// with negligible recall loss at scale.
    #[arg(long, default_value = "16")]
    pub(crate) recall_ef: usize,

    /// Run wake_reconstruct_step on a background worker thread instead of
    /// blocking the ingest loop (saves ~1–3ms/ep amortized, eliminates 5–15ms
    /// latency spikes on high-salience episodes). Head weights merge back into
    /// the main model at sleep entry; EMA prediction weights stream back
    /// continuously with a few-episode lag.
    #[arg(long, default_value = "true")]
    pub(crate) wake_recon_async: bool,

    /// Cap on the number of records in the cold archive. When the archive exceeds
    /// this limit during sleep, the oldest records are dropped and the file is
    /// rewritten keeping only the most recent N. 0 = unlimited (default).
    /// Suggested: 500_000 for multi-day runs (~330 MB at 640-dim INT8).
    #[arg(long, default_value = "0")]
    pub(crate) max_cold_records: usize,

    /// Enable ActionDecoder: per-episode LM forward for gain modulation + stdin generation (ADR 0012).
    /// Requires qwen3_lm.onnx, qwen3_embed_table.safetensors, qwen3_tokenizer.json in --models-dir.
    #[arg(long, default_value = "true")]
    pub(crate) action: bool,

    /// Enable per-region gain modulation from LM attention (ADR 0012).
    /// Only effective when --action is set.
    #[arg(long, default_value = "true")]
    pub(crate) action_gain: bool,

    /// Run the gain update every N episodes instead of every episode.
    /// Gain changes slowly; N=20 is sufficient while keeping the background worker busy < 50%.
    #[arg(long, default_value = "20")]
    pub(crate) gain_interval: usize,

    /// Run wake_reconstruct_step every N episodes. Higher values reduce CPU overhead at
    /// the cost of less frequent output-head gradient updates. N=1 fires every episode.
    #[arg(long, default_value = "5")]
    pub(crate) wake_reconstruct_interval: usize,

    /// Encode only every Nth video frame through DINOv2. N=1 encodes all frames;
    /// N=2 halves video encoder load with negligible scene-understanding loss.
    #[arg(long, default_value = "1")]
    pub(crate) video_stride: usize,

    /// Skip video encoder inference when a frame's mean absolute byte
    /// difference from the last *encoded* frame falls below this threshold
    /// (range 0.0-1.0). 0.0 (default) disables gating — every stride-selected
    /// frame is encoded, matching prior behavior. Applied after `--video-stride`.
    #[arg(long, default_value = "0.0")]
    pub(crate) video_change_threshold: f32,

    /// Force a video encode at least every N stride-selected frames even if
    /// `--video-change-threshold` keeps reporting no change, so a fully static
    /// scene still produces periodic embeddings instead of none at all.
    #[arg(long, default_value = "30")]
    pub(crate) video_change_max_skip: usize,

    /// Maximum tokens to generate per query (ADR 0012).
    #[arg(long, default_value = "25")]
    pub(crate) action_max_tokens: usize,

    /// Number of past episodes to retrieve from warm memory and prepend as recall
    /// context before each generation query (ADR 0028, R1). Set to 0 to disable.
    #[arg(long, default_value = "3")]
    pub(crate) recall_k: usize,

    /// Skip autoregressive generation; only run LM hidden-state distillation.
    /// Text from stdin feeds lang_rx (distillation) but not query_rx (generation).
    /// Use during training campaigns where generation output is not needed.
    #[arg(long, default_value = "false")]
    pub(crate) action_distill_only: bool,

    /// Use qwen3_lm_fixed.onnx (GPU, fixed seq_len=1024) for autoregressive generation
    /// instead of the default CPU path (qwen3_lm.onnx, dynamic seq_len).
    /// NOTE: on gfx1201 (RDNA 4) the GPU path is ~14× slower than CPU at typical seq_lens
    /// due to immature MIGraphX support. Re-evaluate when ROCm matures for RDNA 4.
    #[arg(long, default_value = "false")]
    pub(crate) gpu_generation: bool,

    /// Use a single linear layer (EPISODE_DIM → N_REGIONS*LM_HIDDEN) for episode projection
    /// instead of the 2-layer MLP (ADR 0032). A/B control arm: pass to both runs, compare
    /// mean_proj_loss. Default (MLP) is the treatment arm.
    #[arg(long, default_value = "false")]
    pub(crate) linear_proj: bool,

    /// Filename prefix for action-decoder model artifacts in --models-dir
    /// (expects `{prefix}_gain.onnx`, `{prefix}_lm.onnx`, `{prefix}_embed_table.safetensors`,
    /// `{prefix}_tokenizer.json`, plus optional `{prefix}_text_hidden.onnx` /
    /// `{prefix}_lm_fixed.onnx` / `{prefix}_lm_{prefill,decode}_kv*.onnx`).
    /// LM_HIDDEN is a compile-time constant (1024, Qwen3-0.6B), so a given build only
    /// works with a matching hidden_size family of files. Kept parameterized so a future
    /// model swap can't repeat the cross-contamination that a manual file-swap caused
    /// during the original Qwen3 A/B (see ai-docs/a_b_qwen3_findings.md).
    #[arg(long, default_value = "qwen3")]
    pub(crate) action_model_prefix: String,

    /// Append (episode, text) pairs to a JSONL file during each sleep cycle (ADR 0033).
    /// Used to collect training data for offline LoRA fine-tuning. Only text-bearing
    /// traces (those with a text_target from ASR or text-bypass) are written.
    /// Format: {"episode":[...640 f32...], "text":"decoded text string"}
    #[arg(long)]
    pub(crate) export_training_data: Option<PathBuf>,

    /// Export MLP projection weights (W1, W2) as raw f32-LE binary files for
    /// offline LoRA training (ADR 0033). Writes <dir>/w1.bin and <dir>/w2.bin,
    /// then exits. Requires --action (needs loaded decoder weights).
    #[arg(long)]
    pub(crate) export_proj_weights: Option<PathBuf>,

    /// Use LM forward-pass hidden states for language region input instead of
    /// the frozen embed-table mean-pool. Requires --action (loads qwen3_lm.onnx);
    /// falls back to the embed-table baseline if the decoder is unavailable.
    /// Each text line runs a single non-autoregressive LM forward pass; hidden
    /// states are mean-pooled and L2-normalised to produce the 1024-dim embedding.
    /// Default true — the A/B this flag existed to run concluded in its favor
    /// (the 4x p_lang improvement, gaps.md item 10) and every real campaign has
    /// run with it on; hardened into the default 2026-08-16 (ADR 0047).
    #[arg(long, default_value = "true")]
    pub(crate) lang_use_lm: bool,

    /// L1 sparsity penalty coefficient on SAE activations, added to the NREM
    /// reconstruction loss.  0.0 = off (default).  Positive values encourage
    /// sparser codes; use A/B runs to find a value that lowers MSE without
    /// destabilising prior cosines.
    #[arg(long, default_value = "0.0")]
    pub(crate) sae_l1_coeff: f32,

    /// Use batch-quantile k-WTA threshold instead of per-sample top-k during NREM.
    /// true (default) = batch-level quantile threshold; false = per-sample top-k.
    /// Changes duty-cycle update semantics (order-independent). Deployed via A/B validation.
    #[arg(long, default_value = "true")]
    pub(crate) batch_topk: bool,

    /// DER++ Phase 3 (ai-docs/gaps.md item 7): auto-adapt the weight on the output-head
    /// wake_loss term (MSE against stale stored outputs) via a running-variance estimate,
    /// instead of the implicit λ=1 unweighted sum. false (default) = off. Not yet A/B'd.
    #[arg(long, default_value = "false")]
    pub(crate) der_auto_lambda: bool,

    /// Loop the input file indefinitely. Restarts from the beginning each time
    /// the file ends, keeping warm memory alive across iterations. Useful for
    /// calibration runs where a single file is too short to trigger multiple
    /// sleep cycles.
    #[arg(long, default_value = "false")]
    pub(crate) file_loop: bool,

    /// Enable language region (ADR 0013, Stage A). Reads text from stdin; each
    /// line is tokenized via Qwen2.5 BPE, mean-pooled through the frozen embed
    /// table, and ingested into the language k-WTA region.
    /// Requires qwen3_embed_table.safetensors and qwen3_tokenizer.json in
    /// --models-dir. When combined with --action, the same stdin line is also
    /// routed to the query channel for generation.
    #[arg(long, default_value = "true")]
    pub(crate) language: bool,

    /// Write tracing logs directly to this file instead of stderr.
    /// Avoids the broken-pipe issue when running under `cargo run` and
    /// redirecting stderr (cargo's internal pipe breaks when cargo exits).
    /// Prefer `scripts/run.sh` which builds and execs the binary directly.
    #[arg(long)]
    pub(crate) log_file: Option<PathBuf>,

    /// Exit cleanly after this many sleep cycles (0 = run indefinitely).
    /// Useful for deterministic calibration runs.
    #[arg(long, default_value = "0")]
    pub(crate) max_sleep_cycles: usize,

    /// Session index written to every NDJSON metrics line.
    /// Used by train_campaign.sh to number consecutive runs in a multi-session
    /// campaign; 0 means "not part of a campaign" and is omitted from the line.
    #[arg(long, default_value = "0")]
    pub(crate) session: usize,

    /// Append one NDJSON metrics line per sleep cycle to this file.
    /// Contains: cycle, ts, traces_replayed, mean_loss, mean_output_mse,
    /// warm_recall_l2, was_interrupted, region_sizes, grew, wake_episodes,
    /// asr_count, lang_ingest_count, mean_output_mse_wake, prior_cosine_mean.
    #[arg(long)]
    pub(crate) metrics_file: Option<PathBuf>,

    /// Write the process PID to this file on startup (for status scripts).
    #[arg(long)]
    pub(crate) pid_file: Option<PathBuf>,

    /// Fire a one-shot generation query after the first episode is recorded.
    /// Requires --action. Useful for non-interactive evaluation runs.
    #[arg(long)]
    pub(crate) query: Option<String>,

    /// If > 0.0, trigger sleep early when the per-episode prior_cosine EWMA drops
    /// more than this amount below the healthy baseline captured at wake start.
    /// Set to 0.0 (default) to disable; calibrate from observed prior_cosine ranges
    /// before enabling. Trigger is logged as "perf_degradation".
    #[arg(long, default_value = "0.0")]
    pub(crate) perf_sleep_threshold: f64,

    /// NREM exits early when loss EWMA drops below this value (0.0 = run all traces).
    /// Useful when consolidation converges quickly on small or familiar buffers.
    /// Exit is logged as "loss_target_met" in the sleep summary.
    #[arg(long, default_value = "0.0")]
    pub(crate) sleep_loss_target: f32,

    /// Hard cap on total sleep-cycle wall time in seconds (0 = no cap).
    /// When exceeded the cycle is treated as interrupted (no spill, no snapshot).
    /// Exit is logged as "time_budget" in the sleep summary.
    #[arg(long, default_value = "0")]
    pub(crate) sleep_max_secs: u64,

    /// Fraction of grown units to prune per region during REM (0.0 = disabled).
    /// Only units above the N_UNITS=1024 baseline are candidates. Pruning fires
    /// when gist_recall_rate ≥ 0.8 (familiar content; system not struggling).
    #[arg(long, default_value = "0.0")]
    pub(crate) prune_fraction: f32,

    /// Minimum number of grown units a region must have before pruning is attempted.
    /// Avoids thrashing on small growth steps.
    #[arg(long, default_value = "64")]
    pub(crate) prune_min_grown: usize,

    /// After a growth event, suppress further growth for this many sleep cycles (default 3).
    /// c1 gist is structurally low due to cross-session NREM drift, causing growth to fire
    /// nearly every cycle without a cooldown. Set to 0 to disable (original behaviour).
    #[arg(long, default_value = "3")]
    pub(crate) growth_cooldown: usize,

    /// Number of traces processed per forward+backward pass during NREM replay.
    /// Higher values amortise optimizer overhead and improve BLAS utilisation.
    /// 0 or 1 = sequential (original behaviour). Default: 128.
    #[arg(long, default_value = "128")]
    pub(crate) replay_batch_size: usize,

    /// Minimum number of NREM traces replayed before the salience gate can
    /// interrupt sleep. Prevents early-exit on the very first batch when
    /// diverse/long clips produce many novel embeddings immediately.
    /// 0 = no floor (original behaviour); try 500–1000 with long-clip training.
    #[arg(long, default_value = "0")]
    pub(crate) nrem_min_traces: usize,

    /// Pending-embedding count above which NREM is interrupted to allow live
    /// input to be processed. 0 = disabled (default; correct for file-mode
    /// campaigns where encoders run faster than real-time and would otherwise
    /// abort sleep prematurely). Set to ~16 for live ingest mode.
    #[arg(long, default_value = "0")]
    pub(crate) nrem_salience_threshold: usize,

    /// Minimum character count for an ASR transcription to be forwarded to the
    /// language ingest path. Filters noise/silence artifacts from non-speech
    /// clips (Whisper often outputs 1–7 char strings on urban/nature audio).
    /// 0 = accept all non-empty strings. Default: 20.
    #[arg(long, default_value = "20")]
    pub(crate) asr_min_chars: usize,

    /// How many extra times language-tagged traces are repeated at the front of
    /// the NREM replay queue (in addition to the one baseline pass).
    /// 0 = no oversampling (default). 9 = replay language traces 10× total.
    /// Increases the projection training budget without adding new data.
    #[arg(long, default_value = "0")]
    pub(crate) lang_oversample: usize,

    /// Number of complete passes through the NREM buffer per sleep cycle (default 4).
    /// arXiv:2505.12512; validated on somnus 2026-07-06 (ai-docs/gaps.md item 2): -70.9%
    /// mean_output_mse vs N=1 for +16.6% total wall-clock, no reversals across 5 regions.
    #[arg(long, default_value = "4")]
    pub(crate) nrem_replay_epochs: usize,

    /// Live-episode binding training in NREM (ADR 0041, gaps.md §33): each NREM batch
    /// additionally recomputes episodes live through the binding area and trains the
    /// output heads against the regional targets on them, so gradients reach the
    /// binding weights (which otherwise never train — grown units' binding-projection
    /// rows stay zero forever). Default off; the A/B this change is gated on compares
    /// otherwise-identical runs with and without this flag.
    #[arg(long, default_value = "false")]
    pub(crate) binding_replay: bool,

    /// Cap on `binding_replay` batches per sleep cycle (ADR 0041 adoption path, gaps.md §35):
    /// the pilot's isolated A/B showed the unbounded pass costs +34.5% NREM wall-clock and
    /// displaces ~30% of replay traces, driving episode-space drift faster than warm-store
    /// turnover absorbs it. Capping the count per cycle bounds both. Only meaningful when
    /// `--binding-replay` is set; ignored otherwise. Default 1.
    #[arg(long, default_value = "1")]
    pub(crate) binding_replay_steps: usize,

    /// Learning-rate multiplier on `binding_replay`'s optimizer step (gaps.md §33, candidate
    /// (a) of the three queued after step-count capping — §36's `--binding-replay-steps 1`
    /// pilot — improved wall-clock but made quality *worse*, not better). Reduces step
    /// *magnitude* instead of step *count*, on the theory that a full-strength gradient
    /// update through the live binding path is what's driving the same-cycle MSE blowup and
    /// salience/settle cascade, not the number of times it fires.
    ///
    /// **Was a loss-scale multiplier (applied before `.backward()`) until 2026-08-16** — that
    /// pilot (`engineering-log.md` §44) was a no-op by construction: `binding_replay_step_batch`
    /// steps an AdamW optimizer, which normalizes each update by its running second-moment
    /// estimate, so scaling the loss tensor cancels out almost entirely before it reaches the
    /// parameters. §44's near-identical-to-§36 result (aggregate `mean_output_mse` +440.6% vs.
    /// §36's +445%) is consistent with this — not, as previously recorded, evidence that dosage
    /// doesn't matter. Now scales the learning rate passed to `optimizer.step()` directly, which
    /// actually changes the update magnitude. Only meaningful when `--binding-replay` is set.
    /// Default 1.0 (no scaling, byte-identical to pre-this-flag behavior).
    #[arg(long, default_value = "1.0")]
    pub(crate) binding_replay_lr_scale: f32,

    /// Freeze the five output heads during `binding_replay`'s optimizer step (gaps.md §33
    /// candidate (d), 2026-08-16, added after both variants of candidate (a) — loss-scale and
    /// LR-scale — failed with the same dosage-invariant `mean_output_mse` regression). The
    /// heads are shared, unguarded state: `fused_replay_wake_step_batch` trains them on
    /// *stored* episodes (the source of the production `mean_output_mse` metric) while
    /// `binding_replay_step_batch` trains the same weights on *live* episodes from the
    /// currently-training binder, in the same NREM batch, through the same optimizer — a
    /// plausible direct cause of the regression, independent of update magnitude. When set,
    /// gradient still flows *through* the heads into the live episode tensor and on into the
    /// binding weights (so binding still trains), but the heads' own weights are excluded from
    /// this step's `optimizer.step()` — they're updated only by the fused path. Only
    /// meaningful when `--binding-replay` is set. Default false (heads train normally,
    /// byte-identical to pre-this-flag behavior).
    #[arg(long, default_value = "false")]
    pub(crate) binding_replay_freeze_heads: bool,

    /// Sleep-cycle cadence for `sync_wake_binding` — how often the wake-time binding
    /// snapshot (`PerceptionModel::wake_binding`) is hard-copied from the live
    /// `binding_replay`-trained copy (ADR 0048, gaps.md §33 candidate (e), 2026-09-13).
    /// Added after the confirming diagnostic (engineering-log.md §47) disconfirmed the
    /// leading hypothesis that ORT re-export cadence was the cause of every prior
    /// pilot's `mean_output_mse` regression — the real, backend-independent mechanism is
    /// that wake-time inference (burn or ORT) always reads the *currently-training*
    /// binder, so every sleep cycle forms and stores episodes through a newer generation
    /// than the ones already stored. This flag decouples "how often the binder trains"
    /// (`binding_replay_steps`/`binding_replay_lr_scale`, unaffected) from "how often
    /// wake sees a new binder generation." Applies regardless of `--binding-replay`:
    /// `homeostatic_decay` mutates the trained binding tensors every cycle
    /// unconditionally, so the sync must run on this cadence even with
    /// `--binding-replay` off, or `wake_binding` would freeze at its initial values
    /// forever on the default/production path. Default 1 (sync every cycle —
    /// byte-identical to pre-this-flag behavior, the control-arm equivalent for
    /// this ADR's own validation pilot).
    #[arg(long, default_value = "1")]
    pub(crate) binding_wake_lag_cycles: usize,

    /// Enable ORT Chrome-trace profiling for CPU sessions (Qwen LM).
    /// Writes `<dir>/qwen3_lm_<timestamp>.json` — open in https://ui.perfetto.dev
    /// or chrome://tracing. Profiling data is flushed before process exit.
    /// GPU sessions (Whisper encoder, CLAP, DINOv2, Qwen gain) are covered by
    /// rocprof instead; see scripts/profile.sh.
    #[arg(long)]
    pub(crate) ort_profile_dir: Option<PathBuf>,

    /// Directory where ONNX files are written and loaded from, when ORT binding
    /// inference (Phase 1 GPU unification, ADR 0023/0037) is active.
    /// Default: `<memory-dir>/onnx_models` — no need to set this explicitly.
    /// Whether ORT is actually *used* is decided automatically by the
    /// `--onnx-crossover-units` threshold, not by whether this is set (see below);
    /// this flag only overrides *where* the graphs live if the default is unwanted.
    #[arg(long)]
    pub(crate) onnx_models_dir: Option<PathBuf>,

    /// Path to scripts/export_perception_graph.py (default: auto-detected relative
    /// to the binary location).
    #[arg(long)]
    pub(crate) onnx_script: Option<PathBuf>,

    /// Use MIGraphX GPU EP for ORT binding session (requires AMD GPU + MIGraphX).
    /// When false, uses CPU EP (for testing / validation).
    #[arg(long, default_value = "true")]
    pub(crate) onnx_migraphx: bool,

    /// Disable automatic ORT/MIGraphX perception inference entirely — always use
    /// burn on CPU, regardless of region size. For testing, reproducing older runs
    /// (all growth pilots predate ADR 0037's auto-switch), or machines without a
    /// working Python3+onnx+onnxruntime environment or MIGraphX GPU.
    #[arg(long)]
    pub(crate) force_cpu_inference: bool,

    /// Run the video encoder (DINOv3/v2) on the plain CPU execution provider instead of
    /// MIGraphX. Isolated measurement: 23 ms vs 54 ms per frame (gaps.md §49 R5). Default off
    /// until an A/B on the live pipeline confirms it.
    #[arg(long, default_value = "false")]
    pub(crate) video_encoder_cpu: bool,

    /// Region size (units) at or above which ORT/MIGraphX perception inference is
    /// automatically enabled (ADR 0037; empirical CPU/GPU crossover from ADR 0023's
    /// scaling analysis, research/gpu-unification-design.md §2). Once any region
    /// crosses this, ORT stays active for the rest of the process — growth only
    /// ever increases region size, so there's no reason to switch back to CPU.
    #[arg(long, default_value_t = ONNX_INFERENCE_CROSSOVER_UNITS)]
    pub(crate) onnx_crossover_units: usize,
}
