use crate::util::argmax;
/// ActionDecoder: region-token projection + frozen Qwen3-0.6B (ADR 0012).
///
/// Dual function:
///   1. Every episode: project episode vector → 4 region tokens → LM forward →
///      hidden-state norms → per-region gain multipliers for the next k-WTA step.
///   2. On query (stdin): run full autoregressive generation; emit JSON to stdout.
///
/// Training (NREM): surrogate MSE loss between mean(proj_tokens) and
/// mean(text_target_embeddings) from text-bypass queries.
use anyhow::{bail, Result};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{AdamWConfig, GradientsParams, ModuleOptimizer},
    tensor::{activation, Device, Tensor, TensorData},
};
use somnus_perception::TensorF32Ext;
use ndarray::{Array2, Array3, ArrayView4};
use ort::{
    session::Session,
    value::{DynValueTypeMarker, TensorRef, ValueRef},
};
use safetensors::SafeTensors;
use std::io::Write as _; // for stdout().flush() in streaming generation
use tokenizers::Tokenizer;

/// LM hidden dimension for Qwen3-0.6B (ADR 0035). Must match the model files loaded
/// via --action-model-prefix and somnus-perception's LANG_HIDDEN/REGION_DIMS.
pub const LM_HIDDEN: usize = 1024;
pub const N_REGIONS: usize = 5;
const EPISODE_DIM: usize = 640;
const PROJ_LR: f64 = 1e-4;
// Qwen3 base stops on <|im_end|> (verified against its tokenizer's eos_token_id).
const EOS_TOKEN_ID: u32 = 151645;
/// Fixed sequence length for the GPU text-hidden-state path (qwen3_text_hidden.onnx).
/// Must match MAX_TEXT_SEQ in scripts/export_models.py.
const MAX_TEXT_SEQ: usize = 128;
/// Fixed sequence length for the GPU generation path (qwen3_lm_fixed.onnx).
/// Covers recalled (3×5) + current (5) + history (256) + query (~50) + generated (200) ≈ 526.
/// Must match MAX_GEN_SEQ in scripts/export_models.py.
const MAX_GEN_SEQ: usize = 1024;
/// Token budget for conversation history across turns (R2).
/// Oldest turns are evicted when the total exceeds this limit.
const CONV_HISTORY_BUDGET: usize = 256;
/// Static KV cache size for the split prefill/decode generation path (ADR 0034).
/// KV_SIZE=2048 covers a 400-token prefix plus 1600 generated tokens.
const KV_SIZE: usize = 2048;
const N_KV_LAYERS: usize = 24;
const N_KV_HEADS: usize = 2;
const KV_HEAD_DIM: usize = 64;

// Pre-computed ORT key name slices — eliminates format!() heap allocs in the decode hot loop.
static PAST_KEY_NAMES: [&str; N_KV_LAYERS] = [
    "past_key_0",
    "past_key_1",
    "past_key_2",
    "past_key_3",
    "past_key_4",
    "past_key_5",
    "past_key_6",
    "past_key_7",
    "past_key_8",
    "past_key_9",
    "past_key_10",
    "past_key_11",
    "past_key_12",
    "past_key_13",
    "past_key_14",
    "past_key_15",
    "past_key_16",
    "past_key_17",
    "past_key_18",
    "past_key_19",
    "past_key_20",
    "past_key_21",
    "past_key_22",
    "past_key_23",
];
static PAST_VALUE_NAMES: [&str; N_KV_LAYERS] = [
    "past_value_0",
    "past_value_1",
    "past_value_2",
    "past_value_3",
    "past_value_4",
    "past_value_5",
    "past_value_6",
    "past_value_7",
    "past_value_8",
    "past_value_9",
    "past_value_10",
    "past_value_11",
    "past_value_12",
    "past_value_13",
    "past_value_14",
    "past_value_15",
    "past_value_16",
    "past_value_17",
    "past_value_18",
    "past_value_19",
    "past_value_20",
    "past_value_21",
    "past_value_22",
    "past_value_23",
];
static NEW_KEY_NAMES: [&str; N_KV_LAYERS] = [
    "new_key_0",
    "new_key_1",
    "new_key_2",
    "new_key_3",
    "new_key_4",
    "new_key_5",
    "new_key_6",
    "new_key_7",
    "new_key_8",
    "new_key_9",
    "new_key_10",
    "new_key_11",
    "new_key_12",
    "new_key_13",
    "new_key_14",
    "new_key_15",
    "new_key_16",
    "new_key_17",
    "new_key_18",
    "new_key_19",
    "new_key_20",
    "new_key_21",
    "new_key_22",
    "new_key_23",
];
static NEW_VALUE_NAMES: [&str; N_KV_LAYERS] = [
    "new_value_0",
    "new_value_1",
    "new_value_2",
    "new_value_3",
    "new_value_4",
    "new_value_5",
    "new_value_6",
    "new_value_7",
    "new_value_8",
    "new_value_9",
    "new_value_10",
    "new_value_11",
    "new_value_12",
    "new_value_13",
    "new_value_14",
    "new_value_15",
    "new_value_16",
    "new_value_17",
    "new_value_18",
    "new_value_19",
    "new_value_20",
    "new_value_21",
    "new_value_22",
    "new_value_23",
];

/// KV-cache state for the split prefill/decode generation path (ADR 0034).
struct KvCache {
    /// 24 key buffers and 24 value buffers, alternating k0,v0,k1,v1,...
    /// Each buffer holds [N_KV_HEADS, KV_SIZE, KV_HEAD_DIM] floats in row-major order.
    bufs: Vec<Vec<f32>>,
    /// Attention mask [KV_SIZE + 1]: 0.0 = attend, -1e4 = masked.
    /// Position KV_SIZE is the current token's self-attention slot (always 0.0).
    mask: Vec<f32>,
    /// Next write position in the cache.
    pos: usize,
    /// Reusable logits buffer [vocab_size] — avoids 607 KB allocation per decode step.
    logits_scratch: Vec<f32>,
    /// Per-layer scratch for new K/V output [N_KV_LAYERS*2][N_KV_HEADS*KV_HEAD_DIM].
    /// Eliminates 48 heap allocations per decode step.
    new_kv_scratch: Vec<Vec<f32>>,
}

impl KvCache {
    fn new() -> Self {
        let buf_len = N_KV_HEADS * KV_SIZE * KV_HEAD_DIM;
        let mut mask = vec![-1e4f32; KV_SIZE + 1];
        mask[KV_SIZE] = 0.0; // self-attention slot is always valid
        Self {
            bufs: vec![vec![0.0f32; buf_len]; N_KV_LAYERS * 2],
            mask,
            pos: 0,
            logits_scratch: Vec::new(),
            new_kv_scratch: Vec::new(),
        }
    }

    /// Allocate scratch buffers. Called once after vocab_size is known.
    fn init_scratch(&mut self, vocab_size: usize) {
        self.logits_scratch = vec![0.0f32; vocab_size];
        self.new_kv_scratch = vec![vec![0.0f32; N_KV_HEADS * KV_HEAD_DIM]; N_KV_LAYERS * 2];
    }

    fn reset(&mut self) {
        for buf in &mut self.bufs {
            buf.fill(0.0);
        }
        self.mask.fill(-1e4);
        self.mask[KV_SIZE] = 0.0;
        self.pos = 0;
    }
}

/// One (query, response) turn stored for conversation history (R2).
struct ConvTurn {
    query_ids: Vec<u32>,
    response_ids: Vec<u32>,
}

type ProjOptim = ModuleOptimizer;

/// MLP hidden dimension for the 2-layer episode projection (ADR 0032).
const PROJ_HIDDEN: usize = 1024;

#[derive(Module, Debug)]
struct ProjModel {
    fc1: Linear, // [EPISODE_DIM, PROJ_HIDDEN]
    fc2: Linear, // [PROJ_HIDDEN, N_REGIONS * LM_HIDDEN]
}

impl ProjModel {
    fn new(device: &Device, linear_proj: bool) -> Self {
        let (fc1, fc2) = if linear_proj {
            // Control arm: single linear layer EPISODE_DIM → N_REGIONS*LM_HIDDEN, fc2 unused (1→1 dummy).
            let fc1 = LinearConfig::new(EPISODE_DIM, N_REGIONS * LM_HIDDEN)
                .with_bias(false)
                .init(device);
            let fc2 = LinearConfig::new(1, 1).with_bias(false).init(device);
            (fc1, fc2)
        } else {
            let fc1 = LinearConfig::new(EPISODE_DIM, PROJ_HIDDEN)
                .with_bias(false)
                .init(device);
            let fc2 = LinearConfig::new(PROJ_HIDDEN, N_REGIONS * LM_HIDDEN)
                .with_bias(false)
                .init(device);
            (fc1, fc2)
        };
        Self { fc1, fc2 }
    }
}

/// Re-wraps every parameter as a fresh autodiff leaf (used after loading weights, whose tensors
/// carry no autodiff association).
struct Retrack;

impl burn::module::ModuleMapper for Retrack {
    fn map_float<const D: usize>(
        &mut self,
        param: burn::module::Param<Tensor<D>>,
    ) -> burn::module::Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        let fresh = tensor.inner().autodiff().require_grad();
        burn::module::Param::from_mapped_value(id, fresh, mapper)
    }
}

pub struct ActionDecoder {
    model: Option<ProjModel>,
    optimizer: ProjOptim,
    /// Frozen embedding table: flat [vocab_size * LM_HIDDEN], row-major f32.
    embed_table: Vec<f32>,
    /// Transformer-only (no LM head), static [1,5,1024].
    /// Moved to GainWorker via `create_gain_worker`; None afterwards.
    gain_session: Option<Session>,
    /// Full model with LM head, dynamic seq_len — CPU fallback for generate().
    lm_session: Session,
    /// Base model only (no LM head), static [1, MAX_TEXT_SEQ, 1024] — MIGraphX-eligible.
    /// Used by text_hidden_states() when present; falls back to lm_session on CPU if absent.
    text_hidden_session: Option<Session>,
    /// Base model only (no LM head), static [1, MAX_GEN_SEQ, 1024] — MIGraphX-eligible.
    /// GPU generation path: Rust extracts hidden[seq_len-1] and computes logits via
    /// dot(hidden, embed_table^T). Works because Qwen2.5-0.5B ties lm_head == embed_tokens.
    /// Falls back to lm_session (CPU) when absent or when use_gpu_gen is false.
    lm_fixed_session: Option<Session>,
    /// Whether to use the GPU generation path (lm_fixed_session) when present.
    /// Opt-in via --gpu-generation: on gfx1201 the fixed-1024 path is ~14× slower than
    /// CPU at typical seq_lens; re-evaluate when ROCm/MIGraphX matures for RDNA 4.
    use_gpu_gen: bool,
    /// Prefill model for KV-cache generation: inputs_embeds [1,n,1024] → logits + 48 KV tensors.
    /// Loaded from qwen3_lm_prefill_kv.onnx when present.
    prefill_kv_session: Option<Session>,
    /// Single-step decode model for KV-cache generation: [1,1,1024] + pos + mask + 48 KV → logits + 48 new KV.
    /// Loaded from qwen3_lm_decode_kv.onnx when present.
    decode_kv_session: Option<Session>,
    /// KV cache state for generate_kv(). Reset at each generate() call.
    kv_cache: KvCache,
    /// Use a single linear layer (EPISODE_DIM → N_REGIONS*LM_HIDDEN) instead of the
    /// 2-layer MLP. A/B control arm for ADR 0032 validation (--linear-proj flag).
    linear_proj: bool,
    tokenizer: Tokenizer,
    device: Device,
    /// Cross-cycle LM hidden-state cache. Clips loop so token sequences repeat; after the first
    /// cycle each unique text pays the ORT cost exactly once for the lifetime of the session.
    lm_hidden_cache: std::collections::HashMap<Vec<u32>, Vec<f32>>,
    /// Conversation history ring buffer (R2). Evicted oldest-first when total tokens exceed
    /// CONV_HISTORY_BUDGET. Cleared on sleep entry (wake-local by design).
    conversation: std::collections::VecDeque<ConvTurn>,
    /// Running total of tokens across all turns in `conversation`.
    conv_token_count: usize,
}

impl ActionDecoder {
    /// The model is `None` only while moved into an optimizer step (`take()` / put-back);
    /// any access in that window is a logic bug, so fail with a message that says so.
    #[inline]
    fn model_ref(&self) -> &ProjModel {
        self.model
            .as_ref()
            .expect("ActionDecoder.model accessed while taken for an optimizer step")
    }

    /// Load all model artifacts from `models_dir`.
    /// Returns Err if any required file is absent.
    ///
    /// `ort_profile_dir`: when `Some`, enables ORT Chrome-trace profiling for
    /// lm_session (CPU). Call `end_profiling()` before process exit to flush.
    ///
    /// `model_prefix`: filename prefix for all action-decoder model artifacts
    /// (normally "qwen3") — kept parameterized so future model swaps can never
    /// cross-contaminate files the way the qwen3 A/B file-swap incident once did
    /// (see ai-docs/a_b_qwen3_findings.md).
    pub fn new(
        models_dir: &std::path::Path,
        device: Device,
        ort_profile_dir: Option<&std::path::Path>,
        gpu_generation: bool,
        linear_proj: bool,
        model_prefix: &str,
    ) -> Result<Self> {
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::ActionDecoderInit,
            Self::new_inner(
                models_dir,
                device,
                ort_profile_dir,
                gpu_generation,
                linear_proj,
                model_prefix,
            )
        );
        result
    }

    fn new_inner(
        models_dir: &std::path::Path,
        device: Device,
        ort_profile_dir: Option<&std::path::Path>,
        gpu_generation: bool,
        linear_proj: bool,
        model_prefix: &str,
    ) -> Result<Self> {
        // Training state lives on the autodiff-enabled device (autodiff is a device property in burn 0.22).
        let device = device.autodiff();
        let gain_path = models_dir.join(format!("{model_prefix}_gain.onnx"));
        let lm_path = models_dir.join(format!("{model_prefix}_lm.onnx"));
        let embed_path = models_dir.join(format!("{model_prefix}_embed_table.safetensors"));
        let tok_path = models_dir.join(format!("{model_prefix}_tokenizer.json"));
        for p in [&gain_path, &lm_path, &embed_path, &tok_path] {
            if !p.exists() {
                bail!("action decoder model file not found: {}", p.display());
            }
        }

        let gain_session = Some(crate::migraphx_session(&gain_path)?);
        let lm_profile_prefix = ort_profile_dir.map(|d| d.join(format!("{model_prefix}_lm")));
        let lm_session = crate::cpu_session(&lm_path, lm_profile_prefix.as_deref())?;
        let text_hidden_path = models_dir.join(format!("{model_prefix}_text_hidden.onnx"));
        let text_hidden_session = if text_hidden_path.exists() {
            tracing::info!(
                "loading {model_prefix}_text_hidden.onnx — GPU path active for distillation targets"
            );
            Some(crate::migraphx_session(&text_hidden_path)?)
        } else {
            tracing::info!(
                "{model_prefix}_text_hidden.onnx absent — text_hidden_states() uses CPU lm_session"
            );
            None
        };
        let lm_fixed_path = models_dir.join(format!("{model_prefix}_lm_fixed.onnx"));
        let lm_fixed_session = if gpu_generation && lm_fixed_path.exists() {
            tracing::info!(
                "loading {model_prefix}_lm_fixed.onnx — GPU generation path active (--gpu-generation)"
            );
            Some(crate::migraphx_session(&lm_fixed_path)?)
        } else if gpu_generation {
            tracing::info!(
                "{model_prefix}_lm_fixed.onnx absent — --gpu-generation ignored, using CPU lm_session"
            );
            None
        } else {
            tracing::info!("generate() using CPU lm_session (pass --gpu-generation to use {model_prefix}_lm_fixed.onnx)");
            None
        };

        let prefill_kv_int8 = models_dir.join(format!("{model_prefix}_lm_prefill_kv_int8.onnx"));
        let decode_kv_int8 = models_dir.join(format!("{model_prefix}_lm_decode_kv_int8.onnx"));
        let prefill_kv_fp32 = models_dir.join(format!("{model_prefix}_lm_prefill_kv.onnx"));
        let decode_kv_fp32 = models_dir.join(format!("{model_prefix}_lm_decode_kv.onnx"));
        let (prefill_kv_session, decode_kv_session) = if prefill_kv_int8.exists()
            && decode_kv_int8.exists()
        {
            tracing::info!("loading KV-cache INT8 models — fast generate() path active (Q-B)");
            (
                Some(crate::cpu_session(&prefill_kv_int8, None)?),
                Some(crate::cpu_session(&decode_kv_int8, None)?),
            )
        } else if prefill_kv_fp32.exists() && decode_kv_fp32.exists() {
            tracing::info!("loading KV-cache FP32 models — fast generate() path active (ADR 0034)");
            (
                Some(crate::cpu_session(&prefill_kv_fp32, None)?),
                Some(crate::cpu_session(&decode_kv_fp32, None)?),
            )
        } else {
            tracing::info!("KV-cache models absent — generate() using slow lm_session path");
            (None, None)
        };

        let bytes = std::fs::read(&embed_path)?;
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("safetensors load failed: {e}"))?;
        let tv = st.tensor("embed_tokens.weight").map_err(|e| {
            anyhow::anyhow!(
                "embed_tokens.weight missing in {}: {e}",
                embed_path.display()
            )
        })?;
        let embed_table: Vec<f32> = tv
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();

        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

        let model = ProjModel::new(&device, linear_proj);
        let optimizer: ProjOptim = AdamWConfig::new().init();
        let kv_cache = {
            let vocab_size = embed_table.len() / LM_HIDDEN;
            let mut kv = KvCache::new();
            kv.init_scratch(vocab_size);
            kv
        };

        Ok(Self {
            model: Some(model),
            optimizer,
            embed_table,
            gain_session,
            lm_session,
            text_hidden_session,
            use_gpu_gen: lm_fixed_session.is_some(),
            lm_fixed_session,
            prefill_kv_session,
            decode_kv_session,
            kv_cache,
            linear_proj,
            tokenizer,
            device,
            lm_hidden_cache: std::collections::HashMap::new(),
            conversation: std::collections::VecDeque::new(),
            conv_token_count: 0,
        })
    }

    /// Autoregressive greedy generation given a current episode vector and a text query.
    ///
    /// Sequence layout (R1 + R2):
    ///   `[recalled_0_tokens | ... | recalled_k_tokens | current_tokens | hist_turn_0 | ... | hist_turn_N | query_tokens]`
    /// Recalled episodes are furthest-first (closest sits immediately before current episode).
    /// Conversation history is oldest-first within CONV_HISTORY_BUDGET tokens.
    /// Pass an empty slice for `recalled_episodes` to skip recall.
    /// Pushes (query_ids, response_ids) into the conversation buffer after each successful call.
    /// Returns the generated text (decoded, EOS stripped).
    pub fn generate(
        &mut self,
        episode: &[f32],
        recalled_episodes: &[Vec<f32>],
        query: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::ActionGenerate,
            self.generate_inner(episode, recalled_episodes, query, max_tokens)
        );
        result
    }

    fn generate_inner(
        &mut self,
        episode: &[f32],
        recalled_episodes: &[Vec<f32>],
        query: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let query_ids = self.tokenize(query)?;
        let query_embeds = self.embed_flat(&query_ids)?; // [n_query * LM_HIDDEN]

        // Collect conversation history turns that fit within the token budget.
        // We need to split the borrow: collect turn IDs first, then embed them.
        let history_ids: Vec<(&[u32], &[u32])> = {
            let mut remaining = CONV_HISTORY_BUDGET;
            let mut turns: Vec<(&[u32], &[u32])> = Vec::new();
            // Walk oldest→newest; include turns that fit.
            for turn in &self.conversation {
                let n = turn.query_ids.len() + turn.response_ids.len();
                if n <= remaining {
                    turns.push((&turn.query_ids, &turn.response_ids));
                    remaining -= n;
                }
                // Don't break: a larger older turn may block a smaller newer one; accept
                // the simplicity of first-fit oldest-first.
            }
            turns
        };
        let history_turn_count = history_ids.len();

        // Prepend recalled episode tokens (furthest first), then current episode tokens.
        let mut all_embeds: Vec<f32> = Vec::new();
        for rec_ep in recalled_episodes {
            all_embeds.extend(self.project_flat(rec_ep)?);
        }
        all_embeds.extend(self.project_flat(episode)?); // current episode [N_REGIONS * LM_HIDDEN]

        // Append conversation history (oldest first).
        let mut history_token_count: usize = 0;
        for (q_ids, r_ids) in &history_ids {
            all_embeds.extend(self.embed_flat(q_ids)?);
            all_embeds.extend(self.embed_flat(r_ids)?);
            history_token_count += q_ids.len() + r_ids.len();
        }

        all_embeds.extend_from_slice(&query_embeds);
        let mut seq_len =
            (1 + recalled_episodes.len()) * N_REGIONS + history_token_count + query_ids.len();

        tracing::info!(
            history_turns = history_turn_count,
            history_tokens = history_token_count,
            recalled = recalled_episodes.len(),
            seq_len,
            "generation starting"
        );

        let mut generated: Vec<u32> = Vec::new();

        if self.prefill_kv_session.is_some() && seq_len <= KV_SIZE {
            generated = self.generate_kv(all_embeds, seq_len, max_tokens)?;
        } else if self.use_gpu_gen {
            // GPU path: fixed [1, MAX_GEN_SEQ, LM_HIDDEN] buffer.
            // Causal attention ensures hidden[seq_len-1] is unaffected by zero-padding at
            // positions seq_len..MAX_GEN_SEQ. Logits computed on CPU via embed_table dot.
            let mut gpu_buf = vec![0.0f32; MAX_GEN_SEQ * LM_HIDDEN];
            let copy_len = all_embeds.len().min(gpu_buf.len());
            gpu_buf[..copy_len].copy_from_slice(&all_embeds[..copy_len]);

            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..max_tokens {
                if seq_len > MAX_GEN_SEQ {
                    break;
                }
                // Inner block: sess + outputs borrow released before logits_from_hidden
                // borrows self immutably. use_gpu_gen guarantees lm_fixed_session is Some.
                let hidden = {
                    let sess = self
                        .lm_fixed_session
                        .as_mut()
                        .expect("use_gpu_gen set but lm_fixed_session is None");
                    let input =
                        Array3::from_shape_vec((1, MAX_GEN_SEQ, LM_HIDDEN), gpu_buf.clone())
                            .map_err(|e| anyhow::anyhow!("ndarray shape error: {e}"))?;
                    let tensor = TensorRef::from_array_view(&input)?;
                    let outputs = sess.run(ort::inputs!["inputs_embeds" => tensor])?;
                    let (_, hidden_data) =
                        outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
                    let start = (seq_len - 1) * LM_HIDDEN;
                    hidden_data[start..start + LM_HIDDEN].to_vec()
                    // outputs + sess dropped here
                };
                let next_id = {
                    let mut logits = self.logits_from_hidden(&hidden);
                    for &id in &generated {
                        logits[id as usize] /= 1.3;
                    }
                    argmax(&logits)
                };

                generated.push(next_id);
                if next_id == EOS_TOKEN_ID {
                    break;
                }
                if seq_len < MAX_GEN_SEQ {
                    let next_embed = self.embed_flat(&[next_id])?;
                    gpu_buf[seq_len * LM_HIDDEN..(seq_len + 1) * LM_HIDDEN]
                        .copy_from_slice(&next_embed);
                }
                seq_len += 1;
            }
        } else {
            // CPU fallback: dynamic seq_len via lm_session.
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..max_tokens {
                let next_id = {
                    let input = Array3::from_shape_vec((1, seq_len, LM_HIDDEN), all_embeds.clone())
                        .map_err(|e| anyhow::anyhow!("ndarray shape error: {e}"))?;
                    let tensor = TensorRef::from_array_view(&input)?;
                    let outputs = self
                        .lm_session
                        .run(ort::inputs!["inputs_embeds" => tensor])?;
                    let (_shape, logits_data) = outputs["logits"].try_extract_tensor::<f32>()?;
                    let logits_flat = logits_data.to_vec();
                    let vocab_size = logits_flat.len() / seq_len;
                    let last_logits = &logits_flat[(seq_len - 1) * vocab_size..];
                    let mut penalized: Vec<f32> = last_logits.to_vec();
                    for &id in &generated {
                        penalized[id as usize] /= 1.3;
                    }
                    argmax(&penalized)
                };

                generated.push(next_id);
                if next_id == EOS_TOKEN_ID {
                    break;
                }
                let next_embed = self.embed_flat(&[next_id])?;
                all_embeds.extend_from_slice(&next_embed);
                seq_len += 1;
            }
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

        // Push this turn into the conversation buffer (R2).
        // Strip trailing EOS so it's never embedded as mid-context end-of-sequence.
        let mut response_ids = generated;
        if response_ids.last() == Some(&EOS_TOKEN_ID) {
            response_ids.pop();
        }
        let turn_tokens = query_ids.len() + response_ids.len();
        self.conv_token_count += turn_tokens;
        self.conversation.push_back(ConvTurn {
            query_ids,
            response_ids,
        });
        // Evict oldest turns until within budget.
        while self.conv_token_count > CONV_HISTORY_BUDGET {
            if let Some(front) = self.conversation.pop_front() {
                let n = front.query_ids.len() + front.response_ids.len();
                self.conv_token_count = self.conv_token_count.saturating_sub(n);
            } else {
                break;
            }
        }

        Ok(text)
    }

    /// Clear the conversation history buffer (R2). Called at sleep entry.
    pub fn clear_conversation(&mut self) {
        self.conversation.clear();
        self.conv_token_count = 0;
    }

    /// Surrogate projection training step (ADR 0012, NREM).
    ///
    /// Loss: cosine MSE — MSE between L2-normalised mean(projected region tokens) and
    /// normalised mean(text token embeddings). Bounded in [0, 4]; magnitude-invariant.
    #[allow(dead_code)]
    pub fn projection_step(&mut self, episode: &[f32], target_ids: &[u32]) -> Result<f32> {
        if target_ids.is_empty() {
            return Ok(0.0);
        }

        // Pure-Rust: mean-pool + L2-normalize text embeddings (frozen, no grad).
        let mut text_mean = vec![0.0f32; LM_HIDDEN];
        for &id in target_ids {
            let row = &self.embed_table[id as usize * LM_HIDDEN..(id as usize + 1) * LM_HIDDEN];
            for (a, b) in text_mean.iter_mut().zip(row) {
                *a += b;
            }
        }
        let n = target_ids.len() as f32;
        for v in &mut text_mean {
            *v /= n;
        }
        let text_norm: f32 = (text_mean.iter().map(|x| x * x).sum::<f32>() + 1e-8).sqrt();
        for v in &mut text_mean {
            *v /= text_norm;
        }

        // Burn forward pass (with grad tracking).
        let ep_data = TensorData::new(episode.to_vec(), [1, EPISODE_DIM]);
        let ep_t: Tensor<2> = Tensor::from_data(ep_data, &self.device);

        let model = self.model_ref();
        let proj_out = if self.linear_proj {
            model.fc1.forward(ep_t) // [1, N_REGIONS * LM_HIDDEN]
        } else {
            let h = activation::relu(model.fc1.forward(ep_t)); // [1, PROJ_HIDDEN]
            model.fc2.forward(h) // [1, N_REGIONS * LM_HIDDEN]
        };
        let proj_out = proj_out.reshape([N_REGIONS, LM_HIDDEN]); // [N_REGIONS, LM_HIDDEN]
        let proj_sum = proj_out.sum_dim(0); // [1, LM_HIDDEN]
        let sq_sum = proj_sum.clone().powf_scalar(2.0).sum_dim(1); // [1, 1]
        let norm = (sq_sum + 1e-8f32).sqrt(); // [1, 1]
        let proj_unit = proj_sum / norm; // [1, LM_HIDDEN]

        // Detached text target tensor.
        let text_data = TensorData::new(text_mean, [1, LM_HIDDEN]);
        let text_unit: Tensor<2> = Tensor::from_data(text_data, &self.device);

        let loss = (proj_unit - text_unit).powf_scalar(2.0).mean();
        let loss_val: f32 = loss.clone().into_scalar::<f32>();

        // Take/replace pattern for the optimizer step.
        let model = self.model.take().unwrap();
        let grads = loss.backward();
        let grads_p = GradientsParams::from_grads(grads, &model);
        let model = self.optimizer.step(PROJ_LR, model, grads_p);
        self.model = Some(model);

        Ok(loss_val)
    }

    /// Compute L2-normalised mean LM hidden state for a token sequence.
    ///
    /// GPU path (qwen3_text_hidden.onnx, MIGraphX): pads to MAX_TEXT_SEQ, runs the base
    /// transformer without LM head, mean-pools the first `n` real positions. Causal attention
    /// guarantees end-padding positions do not affect hidden states at real positions, so no
    /// attention_mask is needed. Sequences longer than MAX_TEXT_SEQ are truncated.
    ///
    /// CPU fallback (qwen3_lm.onnx): original dynamic-seq-len path, used when the GPU model
    /// is absent.
    pub fn text_hidden_states(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Ok(vec![0.0; LM_HIDDEN]);
        }
        if let Some(cached) = self.lm_hidden_cache.get(token_ids) {
            return Ok(cached.clone());
        }

        // Truncate to MAX_TEXT_SEQ before embedding; embed_flat borrows self before session.
        let n = token_ids.len().min(MAX_TEXT_SEQ);
        let embeds = self.embed_flat(&token_ids[..n])?;

        // embed_flat borrows embed_table only; compute before borrowing any session.
        let hidden_vec = if let Some(sess) = self.text_hidden_session.as_mut() {
            // GPU path: pad embeds to [1, MAX_TEXT_SEQ, LM_HIDDEN] with zeros.
            let mut padded = embeds;
            padded.resize(MAX_TEXT_SEQ * LM_HIDDEN, 0.0);
            let input = Array3::from_shape_vec((1, MAX_TEXT_SEQ, LM_HIDDEN), padded)
                .map_err(|e| anyhow::anyhow!("shape error: {e}"))?;
            let tensor = TensorRef::from_array_view(&input)?;
            let outputs = sess.run(ort::inputs!["inputs_embeds" => tensor])?;
            let (_, hidden_data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
            hidden_data.to_vec()
            // outputs and sess borrow drop here
        } else {
            // CPU fallback: dynamic seq_len via lm_session.
            let input = Array3::from_shape_vec((1, n, LM_HIDDEN), embeds)
                .map_err(|e| anyhow::anyhow!("shape error: {e}"))?;
            let tensor = TensorRef::from_array_view(&input)?;
            let outputs = self
                .lm_session
                .run(ort::inputs!["inputs_embeds" => tensor])?;
            let (shape, hidden_data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
            if shape.len() < 3 {
                bail!("unexpected last_hidden_state shape: {:?}", shape);
            }
            hidden_data.to_vec()
        };

        // Mean-pool the first n real positions (GPU path: n ≤ MAX_TEXT_SEQ; CPU: n = full len).
        let mut mean = vec![0.0f32; LM_HIDDEN];
        for tok in 0..n {
            for d in 0..LM_HIDDEN {
                mean[d] += hidden_vec[tok * LM_HIDDEN + d];
            }
        }
        let scale = 1.0 / n as f32;
        for v in &mut mean {
            *v *= scale;
        }

        // L2-normalise.
        let norm = (mean.iter().map(|x| x * x).sum::<f32>() + 1e-8).sqrt();
        let result: Vec<f32> = mean.iter().map(|x| x / norm).collect();
        self.lm_hidden_cache
            .insert(token_ids.to_vec(), result.clone());
        Ok(result)
    }

    /// Batched surrogate projection training step (embed-table mean target).
    /// Superseded by `projection_step_batch_distill`; retained for reference.
    #[allow(dead_code)]
    pub fn projection_step_batch(&mut self, pairs: &[(&[f32], &[u32])]) -> Result<f32> {
        let valid: Vec<_> = pairs.iter().filter(|(_, ids)| !ids.is_empty()).collect();
        if valid.is_empty() {
            return Ok(0.0);
        }
        let n = valid.len();

        // Compute normalized text means per trace (pure Rust, no grad).
        let mut text_targets = vec![0.0f32; n * LM_HIDDEN];
        for (i, (_, ids)) in valid.iter().enumerate() {
            let mut mean = vec![0.0f32; LM_HIDDEN];
            for &id in *ids {
                let row = &self.embed_table[id as usize * LM_HIDDEN..(id as usize + 1) * LM_HIDDEN];
                for (a, b) in mean.iter_mut().zip(row) {
                    *a += b;
                }
            }
            let scale = 1.0 / ids.len() as f32;
            for v in &mut mean {
                *v *= scale;
            }
            let norm = (mean.iter().map(|x| x * x).sum::<f32>() + 1e-8).sqrt();
            for (j, v) in mean.iter().enumerate() {
                text_targets[i * LM_HIDDEN + j] = v / norm;
            }
        }

        // Stack episodes: [N, EPISODE_DIM].
        let flat_eps: Vec<f32> = valid
            .iter()
            .flat_map(|(ep, _)| ep.iter().copied())
            .collect();
        let ep_t: Tensor<2> =
            Tensor::from_data(TensorData::new(flat_eps, [n, EPISODE_DIM]), &self.device);

        let model = self.model_ref();
        let proj_out = if self.linear_proj {
            model.fc1.forward(ep_t) // [N, N_REGIONS * LM_HIDDEN]
        } else {
            let h = activation::relu(model.fc1.forward(ep_t)); // [N, PROJ_HIDDEN]
            model.fc2.forward(h) // [N, N_REGIONS * LM_HIDDEN]
        };
        let proj_out = proj_out.reshape([n, N_REGIONS, LM_HIDDEN]); // [N, N_REGIONS, LM_HIDDEN]
        let proj_sum = proj_out.sum_dim(1).reshape([n, LM_HIDDEN]); // [N, LM_HIDDEN]
        let sq_sum = proj_sum.clone().powf_scalar(2.0).sum_dim(1); // [N, 1]
        let norm = (sq_sum + 1e-8f32).sqrt(); // [N, 1]
        let proj_unit = proj_sum / norm; // [N, LM_HIDDEN]

        let text_data = TensorData::new(text_targets, [n, LM_HIDDEN]);
        let text_unit: Tensor<2> = Tensor::from_data(text_data, &self.device);

        let loss = (proj_unit - text_unit).powf_scalar(2.0).mean();
        let loss_val: f32 = loss.clone().into_scalar::<f32>();

        let model = self.model.take().unwrap();
        let grads = loss.backward();
        let grads_p = GradientsParams::from_grads(grads, &model);
        let model = self.optimizer.step(PROJ_LR, model, grads_p);
        self.model = Some(model);

        Ok(loss_val)
    }

    /// Distillation-based projection training step (Gap 6).
    ///
    /// Targets are precomputed LM hidden states (from `text_hidden_states`),
    /// already mean-pooled and L2-normalised to [LM_HIDDEN]. This replaces the
    /// static embed-table mean target used by `projection_step_batch`: Qwen's
    /// transformer contextualises the text tokens before pooling, so the target
    /// encodes inter-token relationships rather than a bag-of-embeddings sum.
    ///
    /// `pairs`: (episode_vec [EPISODE_DIM], lm_hidden_target [LM_HIDDEN])
    pub fn projection_step_batch_distill(&mut self, pairs: &[(&[f32], &[f32])]) -> Result<f32> {
        let valid: Vec<_> = pairs.iter().filter(|(_, h)| !h.is_empty()).collect();
        if valid.is_empty() {
            return Ok(0.0);
        }
        let n = valid.len();

        let flat_hidden: Vec<f32> = valid.iter().flat_map(|(_, h)| h.iter().copied()).collect();
        let flat_eps: Vec<f32> = valid
            .iter()
            .flat_map(|(ep, _)| ep.iter().copied())
            .collect();
        let ep_t: Tensor<2> =
            Tensor::from_data(TensorData::new(flat_eps, [n, EPISODE_DIM]), &self.device);

        let model = self.model_ref();
        let proj_out = if self.linear_proj {
            model.fc1.forward(ep_t) // [N, N_REGIONS * LM_HIDDEN]
        } else {
            let h = activation::relu(model.fc1.forward(ep_t)); // [N, PROJ_HIDDEN]
            model.fc2.forward(h) // [N, N_REGIONS * LM_HIDDEN]
        };
        let proj_out = proj_out.reshape([n, N_REGIONS, LM_HIDDEN]);
        let proj_sum = proj_out.sum_dim(1).reshape([n, LM_HIDDEN]);
        let sq_sum = proj_sum.clone().powf_scalar(2.0).sum_dim(1);
        let norm = (sq_sum + 1e-8f32).sqrt();
        let proj_unit = proj_sum / norm; // [N, LM_HIDDEN], L2-normalised

        // Targets are already normalised; load as detached tensor.
        let target_t: Tensor<2> =
            Tensor::from_data(TensorData::new(flat_hidden, [n, LM_HIDDEN]), &self.device);

        let loss = (proj_unit - target_t).powf_scalar(2.0).mean();
        let loss_val: f32 = loss.clone().into_scalar::<f32>();

        let model = self.model.take().unwrap();
        let grads = loss.backward();
        let grads_p = GradientsParams::from_grads(grads, &model);
        let model = self.optimizer.step(PROJ_LR, model, grads_p);
        self.model = Some(model);

        Ok(loss_val)
    }

    /// Flush the ORT Chrome-trace profile for lm_session.
    /// Must be called before `process::exit(0)` when `--ort-profile-dir` is set;
    /// the profile file is only written on explicit `end_profiling` or normal drop,
    /// neither of which happens with the MIGraphX-workaround exit.
    /// Returns the path written, or an error if profiling was not enabled.
    pub fn end_profiling(&mut self) -> Result<String> {
        self.lm_session
            .end_profiling()
            .map_err(|e| anyhow::anyhow!("end_profiling failed: {e}"))
    }

    /// Decode token IDs back to a UTF-8 string. Used by R3 introspection.
    pub fn decode_ids(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Frobenius norm of the projection weight matrices (monitoring, Phase 8).
    pub fn proj_weight_norm(&self) -> Result<f32> {
        let model = self.model_ref();
        let w1 = model.fc1.weight.val().to_f32_vec();
        let norm_sq: f32 = if self.linear_proj {
            w1.iter().map(|x| x * x).sum()
        } else {
            let w2 = model.fc2.weight.val().to_f32_vec();
            w1.iter().chain(w2.iter()).map(|x| x * x).sum()
        };
        Ok(norm_sq.sqrt())
    }

    /// Move the gain ORT session to a background `GainWorker` thread.
    ///
    /// Call once at startup (after weights are loaded). The session is not duplicated —
    /// it moves entirely to the worker thread (no second GPU allocation).
    /// Returns None if the gain session was already moved or was never loaded.
    pub fn create_gain_worker(&mut self) -> Option<GainWorker> {
        let sess = self.gain_session.take()?;
        let proj_w = self.proj_weights_flat();
        Some(GainWorker::spawn(sess, proj_w, self.linear_proj))
    }

    /// Snapshot projection weights for GainWorker.
    ///
    /// MLP mode (default): [W1_flat | W2_flat]
    ///   W1: [EPISODE_DIM, PROJ_HIDDEN], W2: [PROJ_HIDDEN, N_REGIONS*LM_HIDDEN]
    /// Linear mode (--linear-proj): [W1_flat] only
    ///   W1: [EPISODE_DIM, N_REGIONS*LM_HIDDEN]
    /// gain_from_episode() receives `linear_proj` separately to interpret the layout.
    pub fn proj_weights_flat(&self) -> Vec<f32> {
        let model = self.model_ref();
        let w1 = model.fc1.weight.val().detach().to_f32_vec();
        if self.linear_proj {
            return w1;
        }
        let w2 = model.fc2.weight.val().detach().to_f32_vec();
        let mut out = Vec::with_capacity(w1.len() + w2.len());
        out.extend(w1);
        out.extend(w2);
        out
    }

    /// Save projection weights as safetensors (`<path>.safetensors`).
    pub fn save_weights(&self, path: &std::path::Path) -> Result<()> {
        use burn_store::{ModuleSnapshot, SafetensorsStore};
        // Write beside the target and rename: a kill mid-write must not truncate the only copy.
        let dst = path.with_extension("safetensors");
        let tmp = somnus_memory::fsutil::tmp_path(&dst);
        let mut store = SafetensorsStore::from_file(&tmp);
        self.model_ref()
            .save_into(&mut store)
            .map_err(|e| anyhow::anyhow!("save_weights failed: {e:?}"))?;
        somnus_memory::fsutil::commit_tmp(&tmp, &dst)?;
        Ok(())
    }

    /// Load projection weights from `<path>.safetensors`. On failure the model is replaced by
    /// a fresh one so callers remain functional (cold-starting the projection).
    pub fn load_weights(&mut self, path: &std::path::Path) -> Result<()> {
        use burn_store::{ModuleSnapshot, SafetensorsStore};
        let mut store = SafetensorsStore::from_file(path.with_extension("safetensors"));
        let result = self
            .model
            .as_mut()
            .expect("ActionDecoder.model accessed while taken for an optimizer step")
            .load_from(&mut store)
            .map_err(|e| anyhow::anyhow!("{e:?}"))
            .and_then(|r| {
                if r.errors.is_empty() {
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("{:?}", r.errors))
                }
            });
        match result {
            Ok(()) => {
                // Loaded tensors carry no autodiff association: re-wrap every parameter as a
                // fresh tracked leaf, and reset the optimizer state to match.
                let model = self.model.take().expect("model present");
                self.model = Some(model.map(&mut Retrack));
                self.optimizer = AdamWConfig::new().init();
                Ok(())
            }
            Err(e) => {
                tracing::warn!("decoder weight restore failed (cold-starting projection): {e}");
                self.model = Some(ProjModel::new(&self.device, self.linear_proj));
                self.optimizer = AdamWConfig::new().init();
                Err(anyhow::anyhow!("load_weights failed: {e}"))
            }
        }
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// KV-cache generation path (ADR 0034). Runs prefill once, then decodes token-by-token.
    ///
    /// `prefix_embeds`: flat [prefix_len * LM_HIDDEN] embeddings for the full prefix.
    /// `prefix_len`: number of prefix tokens.
    /// Returns raw generated token IDs (including EOS if hit).
    fn generate_kv(
        &mut self,
        prefix_embeds: Vec<f32>,
        prefix_len: usize,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        // ── Step 1: Prefill ──────────────────────────────────────────────────
        // prefill_sess is a local — outputs borrows it, not self, so we can write
        // to self.kv_cache.bufs while outputs is alive. No intermediate Vec needed.
        let mut prefill_sess = self.prefill_kv_session.take().unwrap();
        let first_logits: Vec<f32> = {
            let arr = Array3::from_shape_vec((1, prefix_len, LM_HIDDEN), prefix_embeds)
                .map_err(|e| anyhow::anyhow!("ndarray: {e}"))?;
            let tensor = TensorRef::from_array_view(&arr)?;
            let outputs = prefill_sess.run(ort::inputs!["inputs_embeds" => tensor])?;

            // try_extract_tensor returns (&Shape, &[T]) — data is a direct slice, no Option.
            let (_, logit_data) = outputs["logits"].try_extract_tensor::<f32>()?;
            let vocab = logit_data.len() / prefix_len;
            let first_logits: Vec<f32> = logit_data[(prefix_len - 1) * vocab..].to_vec();

            // Scatter KV directly into cache — eliminates ~7 MB intermediate Vec<Vec<f32>>.
            // KV shape from prefill: [1, N_KV_HEADS, prefix_len, KV_HEAD_DIM].
            self.kv_cache.reset();
            for i in 0..N_KV_LAYERS {
                let (_, k_src) = outputs[PAST_KEY_NAMES[i]].try_extract_tensor::<f32>()?;
                let (_, v_src) = outputs[PAST_VALUE_NAMES[i]].try_extract_tensor::<f32>()?;
                for h in 0..N_KV_HEADS {
                    let src_off = h * prefix_len * KV_HEAD_DIM;
                    let dst_off = h * KV_SIZE * KV_HEAD_DIM;
                    self.kv_cache.bufs[2 * i][dst_off..dst_off + prefix_len * KV_HEAD_DIM]
                        .copy_from_slice(&k_src[src_off..src_off + prefix_len * KV_HEAD_DIM]);
                    self.kv_cache.bufs[2 * i + 1][dst_off..dst_off + prefix_len * KV_HEAD_DIM]
                        .copy_from_slice(&v_src[src_off..src_off + prefix_len * KV_HEAD_DIM]);
                }
            }
            // outputs dropped here; prefill_sess borrow ends.
            first_logits
        };
        self.prefill_kv_session = Some(prefill_sess);

        // Open mask for prefix positions (once, not per-layer).
        for p in 0..prefix_len {
            self.kv_cache.mask[p] = 0.0;
        }
        self.kv_cache.pos = prefix_len;

        // ── Step 2: Greedy decode ────────────────────────────────────────────
        let mut generated: Vec<u32> = Vec::new();
        {
            // First token from prefill logits (scratch not used here — first_logits is small).
            self.kv_cache.logits_scratch.copy_from_slice(&first_logits);
            // No prior generated tokens yet, skip rep-penalty.
            let mut next_id = argmax(&self.kv_cache.logits_scratch);
            generated.push(next_id);
            if next_id != EOS_TOKEN_ID {
                if let Ok(piece) = self.tokenizer.decode(&[next_id], false) {
                    print!("{piece}");
                    let _ = std::io::stdout().flush();
                }
            }

            let mut decode_sess = self.decode_kv_session.take().unwrap();
            for _ in 1..max_tokens {
                if next_id == EOS_TOKEN_ID || self.kv_cache.pos >= KV_SIZE {
                    break;
                }
                let pos = self.kv_cache.pos;

                // Embed the previous token (borrows embed_table only, safe while sess is out).
                let embed_vec = self.embed_flat(&[next_id])?;

                // Zero-copy views into KV buffers — no clone needed (decode_sess already taken out).
                // Block ensures views drop before we write back to self.kv_cache.bufs below.
                let outputs = {
                    let k_views: Vec<ArrayView4<f32>> = (0..N_KV_LAYERS)
                        .map(|i| {
                            ArrayView4::from_shape(
                                (1, N_KV_HEADS, KV_SIZE, KV_HEAD_DIM),
                                &self.kv_cache.bufs[2 * i],
                            )
                            .map_err(|e| anyhow::anyhow!("ndarray: {e}"))
                        })
                        .collect::<Result<_>>()?;
                    let v_views: Vec<ArrayView4<f32>> = (0..N_KV_LAYERS)
                        .map(|i| {
                            ArrayView4::from_shape(
                                (1, N_KV_HEADS, KV_SIZE, KV_HEAD_DIM),
                                &self.kv_cache.bufs[2 * i + 1],
                            )
                            .map_err(|e| anyhow::anyhow!("ndarray: {e}"))
                        })
                        .collect::<Result<_>>()?;
                    let mask_view =
                        ArrayView4::from_shape((1, 1, 1, KV_SIZE + 1), &self.kv_cache.mask)
                            .map_err(|e| anyhow::anyhow!("ndarray: {e}"))?;

                    let emb_arr = Array3::from_shape_vec((1, 1, LM_HIDDEN), embed_vec)
                        .map_err(|e| anyhow::anyhow!("ndarray: {e}"))?;
                    let pos_arr = Array2::from_shape_vec((1, 1), vec![pos as i64])
                        .map_err(|e| anyhow::anyhow!("ndarray: {e}"))?;

                    // Use &str keys to avoid 96 String heap allocs per token.
                    let mut feed: Vec<(&str, ValueRef<'_, DynValueTypeMarker>)> =
                        Vec::with_capacity(3 + N_KV_LAYERS * 2);
                    feed.push((
                        "inputs_embeds",
                        TensorRef::from_array_view(&emb_arr)?.into_dyn(),
                    ));
                    feed.push((
                        "position_ids",
                        TensorRef::from_array_view(&pos_arr)?.into_dyn(),
                    ));
                    feed.push((
                        "attention_mask",
                        TensorRef::from_array_view(mask_view)?.into_dyn(),
                    ));
                    for i in 0..N_KV_LAYERS {
                        feed.push((
                            PAST_KEY_NAMES[i],
                            TensorRef::from_array_view(k_views[i])?.into_dyn(),
                        ));
                        feed.push((
                            PAST_VALUE_NAMES[i],
                            TensorRef::from_array_view(v_views[i])?.into_dyn(),
                        ));
                    }
                    decode_sess.run(feed)?
                };

                // Copy logits into scratch — avoids 607 KB allocation per step.
                // try_extract_tensor returns (&Shape, &[T]) — data is already a slice.
                let (_, logit_data) = outputs["logits"].try_extract_tensor::<f32>()?;
                self.kv_cache.logits_scratch.copy_from_slice(logit_data);

                // Copy new KV into scratch — avoids 48 Vec allocs per step.
                // new_key shape: [1, N_KV_HEADS, 1, KV_HEAD_DIM].
                for i in 0..N_KV_LAYERS {
                    let (_, k) = outputs[NEW_KEY_NAMES[i]].try_extract_tensor::<f32>()?;
                    let (_, v) = outputs[NEW_VALUE_NAMES[i]].try_extract_tensor::<f32>()?;
                    self.kv_cache.new_kv_scratch[2 * i].copy_from_slice(k);
                    self.kv_cache.new_kv_scratch[2 * i + 1].copy_from_slice(v);
                }
                // outputs dropped here; no longer borrowed.

                // Write new KV from scratch into cache at position `pos`.
                // Field destructuring lets the borrow checker see new_kv_scratch and bufs
                // as disjoint borrows of the same struct.
                {
                    let KvCache {
                        ref new_kv_scratch,
                        ref mut bufs,
                        ..
                    } = self.kv_cache;
                    for i in 0..N_KV_LAYERS {
                        let k_new = &new_kv_scratch[2 * i];
                        let v_new = &new_kv_scratch[2 * i + 1];
                        for h in 0..N_KV_HEADS {
                            let src = h * KV_HEAD_DIM;
                            let dst = h * KV_SIZE * KV_HEAD_DIM + pos * KV_HEAD_DIM;
                            bufs[2 * i][dst..dst + KV_HEAD_DIM]
                                .copy_from_slice(&k_new[src..src + KV_HEAD_DIM]);
                            bufs[2 * i + 1][dst..dst + KV_HEAD_DIM]
                                .copy_from_slice(&v_new[src..src + KV_HEAD_DIM]);
                        }
                    }
                }
                self.kv_cache.mask[pos] = 0.0;
                self.kv_cache.pos += 1;

                // Greedy next token with repetition penalty.
                for &id in &generated {
                    self.kv_cache.logits_scratch[id as usize] /= 1.3;
                }
                next_id = argmax(&self.kv_cache.logits_scratch);
                generated.push(next_id);

                // Stream each token immediately — first token appears in ~40 ms.
                if next_id != EOS_TOKEN_ID {
                    if let Ok(piece) = self.tokenizer.decode(&[next_id], false) {
                        print!("{piece}");
                        let _ = std::io::stdout().flush();
                    }
                }
            }
            self.decode_kv_session = Some(decode_sess);
        }
        println!(); // end token stream on its own line before the JSON response

        Ok(generated)
    }

    /// Compute logits for every vocabulary token given a single hidden-state vector.
    ///
    /// `logit[v] = dot(hidden, embed_table[v])` — valid when lm_head and embed_tokens are
    /// tied (Qwen2.5-0.5B always ties them). Called by the GPU generation path to avoid
    /// shipping vocab_size * LM_HIDDEN floats off the GPU per decode step.
    fn logits_from_hidden(&self, hidden: &[f32]) -> Vec<f32> {
        let vocab = self.embed_table.len() / LM_HIDDEN;
        let mut out = vec![0.0f32; vocab];
        for (v, logit) in out.iter_mut().enumerate() {
            let row = &self.embed_table[v * LM_HIDDEN..(v + 1) * LM_HIDDEN];
            *logit = hidden.iter().zip(row).map(|(h, e)| h * e).sum();
        }
        out
    }

    /// Project episode → flat Vec<f32> of length [N_REGIONS * LM_HIDDEN].
    fn project_flat(&self, episode: &[f32]) -> Result<Vec<f32>> {
        let model = self.model_ref();
        let ep_data = TensorData::new(episode.to_vec(), [1, EPISODE_DIM]);
        let ep_t: Tensor<2> = Tensor::from_data(ep_data, &self.device);
        let proj = if self.linear_proj {
            model.fc1.forward(ep_t) // [1, N_REGIONS * LM_HIDDEN]
        } else {
            let h = activation::relu(model.fc1.forward(ep_t)); // [1, PROJ_HIDDEN]
            model.fc2.forward(h) // [1, N_REGIONS * LM_HIDDEN]
        };
        let flat = proj.to_f32_vec();
        Ok(flat)
    }

    /// Embed token IDs → flat Vec<f32> of length [n * LM_HIDDEN].
    fn embed_flat(&self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let mut out = Vec::with_capacity(ids.len() * LM_HIDDEN);
        for &id in ids {
            let start = id as usize * LM_HIDDEN;
            out.extend_from_slice(&self.embed_table[start..start + LM_HIDDEN]);
        }
        Ok(out)
    }
}

// ── GainWorker ────────────────────────────────────────────────────────────────

enum GainMsg {
    Compute(Vec<f32>),
    UpdateWeights(Vec<f32>),
}

/// Offloads per-episode Qwen gain updates from the tokio event loop to a dedicated OS thread.
///
/// The main loop sends episode vectors non-blocking and polls for results non-blocking.
/// Projection weights are refreshed after each NREM cycle via `update_weights`.
pub struct GainWorker {
    tx: std::sync::mpsc::SyncSender<GainMsg>,
    rx: std::sync::mpsc::Receiver<[f32; 5]>,
}

impl GainWorker {
    /// Spawn the worker thread. Takes ownership of the gain ORT session (no duplicate load).
    pub fn spawn(session: Session, proj_weights: Vec<f32>, linear_proj: bool) -> Self {
        let (tx, req_rx) = std::sync::mpsc::sync_channel::<GainMsg>(2);
        let (resp_tx, rx) = std::sync::mpsc::sync_channel::<[f32; 5]>(2);

        std::thread::Builder::new()
            .name("gain-worker".into())
            .spawn(move || {
                let mut proj = proj_weights;
                let mut session = session;
                while let Ok(msg) = req_rx.recv() {
                    match msg {
                        GainMsg::UpdateWeights(w) => proj = w,
                        GainMsg::Compute(ep) => {
                            let gain = gain_from_episode(&mut session, &proj, &ep, linear_proj)
                                .unwrap_or_else(|e| {
                                    tracing::warn!(?e, "gain computation failed");
                                    [1.0f32; 5]
                                });
                            let _ = resp_tx.send(gain);
                        }
                    }
                }
            })
            .expect("failed to spawn gain-worker thread");

        Self { tx, rx }
    }

    /// Non-blocking: queue an episode for gain computation.
    /// Returns false and drops the request if the worker is busy (channel full).
    pub fn request_gain(&self, episode: Vec<f32>) -> bool {
        self.tx.try_send(GainMsg::Compute(episode)).is_ok()
    }

    /// Non-blocking: return the latest computed gain if ready.
    pub fn try_recv_gain(&self) -> Option<[f32; 5]> {
        self.rx.try_recv().ok()
    }

    /// Refresh projection weights after NREM training. Non-blocking; drops if channel full.
    pub fn update_weights(&self, weights: Vec<f32>) {
        let _ = self.tx.try_send(GainMsg::UpdateWeights(weights));
    }
}

/// Raw gain computation: project episode → region tokens → ORT forward → norms → gain.
///
/// MLP layout (default): proj_w = [W1_flat | W2_flat]
///   W1: [EPISODE_DIM, PROJ_HIDDEN], W2: [PROJ_HIDDEN, N_REGIONS*LM_HIDDEN]
/// Linear layout (linear_proj=true): proj_w = [W1_flat]
///   W1: [EPISODE_DIM, N_REGIONS*LM_HIDDEN]
/// Burn stores Linear weights as [n_in, n_out], so y = x @ W (no transpose).
fn gain_from_episode(
    session: &mut Session,
    proj_w: &[f32],
    episode: &[f32],
    linear_proj: bool,
) -> Result<[f32; 5]> {
    // Runs on GainWorker's dedicated background OS thread — somnus_metrics::METRICS.record()
    // (inside timed_core!) is a plain atomic fetch_add, safe cross-thread (same reasoning as
    // somnus-perception's WakeWorkerTrainStep).
    let (result, _) = somnus_metrics::timed_core!(
        somnus_metrics::Metric::GainCompute,
        gain_from_episode_inner(session, proj_w, episode, linear_proj)
    );
    result
}

fn gain_from_episode_inner(
    session: &mut Session,
    proj_w: &[f32],
    episode: &[f32],
    linear_proj: bool,
) -> Result<[f32; 5]> {
    let n_out = N_REGIONS * LM_HIDDEN;

    let proj_flat = if linear_proj {
        // Single linear layer: proj_flat[k] = sum_i episode[i] * W[i * n_out + k]
        let mut out = vec![0.0f32; n_out];
        for (i, &xi) in episode.iter().enumerate() {
            let row = &proj_w[i * n_out..(i + 1) * n_out];
            for (k, &wik) in row.iter().enumerate() {
                out[k] += xi * wik;
            }
        }
        out
    } else {
        // Two-layer MLP: h = relu(episode @ W1), proj_flat = h @ W2
        let w1_len = EPISODE_DIM * PROJ_HIDDEN;
        let w1 = &proj_w[..w1_len];
        let w2 = &proj_w[w1_len..];

        // Layer 1: h[j] = max(0, sum_i episode[i] * W1[i * PROJ_HIDDEN + j])
        let mut h = vec![0.0f32; PROJ_HIDDEN];
        for (i, &xi) in episode.iter().enumerate() {
            let row = &w1[i * PROJ_HIDDEN..(i + 1) * PROJ_HIDDEN];
            for (j, &wij) in row.iter().enumerate() {
                h[j] += xi * wij;
            }
        }
        for v in &mut h {
            *v = v.max(0.0);
        } // ReLU

        // Layer 2: proj_flat[k] = sum_j h[j] * W2[j * n_out + k]
        let mut out = vec![0.0f32; n_out];
        for (j, &hj) in h.iter().enumerate() {
            if hj == 0.0 {
                continue;
            }
            let row = &w2[j * n_out..(j + 1) * n_out];
            for (k, &wjk) in row.iter().enumerate() {
                out[k] += hj * wjk;
            }
        }
        out
    };

    let inputs = Array3::from_shape_vec((1, N_REGIONS, LM_HIDDEN), proj_flat)
        .map_err(|e| anyhow::anyhow!("ndarray shape: {e}"))?;
    let tensor = TensorRef::from_array_view(&inputs)?;
    let outputs = session.run(ort::inputs!["inputs_embeds" => tensor])?;
    let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
    if shape.len() < 3 || shape[1] != N_REGIONS as i64 {
        bail!("unexpected last_hidden_state shape: {:?}", shape);
    }
    let hidden = data.to_vec();
    let norms: [f32; 5] = std::array::from_fn(|i| {
        let row = &hidden[i * LM_HIDDEN..(i + 1) * LM_HIDDEN];
        row.iter().map(|x| x * x).sum::<f32>().sqrt()
    });
    let max_n = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: [f32; 5] = norms.map(|n| ((n - max_n) / 10.0).exp());
    let sum_e: f32 = exp.iter().sum();
    let attn = exp.map(|e| e / sum_e);
    Ok(attn.map(|a| (a * 5.0).clamp(0.5, 2.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ProjModel safetensors round trip: tensor names are the on-disk contract with snapshots
    /// converted from burn 0.21 (`fc1.weight`, `fc2.weight`), values survive, and the reloaded
    /// params are fresh autodiff leaves (a gradient reaches them).
    #[test]
    fn proj_model_safetensors_round_trip_and_retrack() {
        use burn_store::{ModuleSnapshot, SafetensorsStore};
        let device = Device::flex().autodiff();
        let src = ProjModel::new(&device, true);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("proj.safetensors");
        src.clone()
            .save_into(&mut SafetensorsStore::from_file(&path))
            .unwrap();
        let header_len = u64::from_le_bytes(std::fs::read(&path).unwrap()[..8].try_into().unwrap());
        let header =
            String::from_utf8(std::fs::read(&path).unwrap()[8..8 + header_len as usize].to_vec())
                .unwrap();
        assert!(header.contains("\"fc1.weight\"") && header.contains("\"fc2.weight\""));

        let mut dst = ProjModel::new(&device, true);
        let r = dst
            .load_from(&mut SafetensorsStore::from_file(&path))
            .unwrap();
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        let dst = dst.map(&mut Retrack);
        let a = src.fc1.weight.val().to_f32_vec();
        let b = dst.fc1.weight.val().to_f32_vec();
        assert_eq!(a, b);

        let x: Tensor<2> = Tensor::from_data(
            TensorData::new(vec![0.5f32; EPISODE_DIM], [1, EPISODE_DIM]),
            &device,
        );
        let loss = dst.fc1.forward(x).powf_scalar(2.0).mean();
        let grads = loss.backward();
        assert!(
            dst.fc1.weight.val().grad(&grads).is_some(),
            "no gradient reached reloaded param"
        );
    }

    #[test]
    fn gain_clamp_range() {
        // Verify that softmax → scale → clamp stays in [0.5, 2.0]
        let norms = [1.0f32, 2.0, 3.0, 4.0];
        let max_n = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: [f32; 4] = norms.map(|n| ((n - max_n) / 10.0).exp());
        let sum_e: f32 = exp.iter().sum();
        let attn = exp.map(|e| e / sum_e);
        let gain = attn.map(|a| (a * 4.0).clamp(0.5, 2.0));
        for g in &gain {
            assert!(*g >= 0.5 && *g <= 2.0);
        }
    }
}
