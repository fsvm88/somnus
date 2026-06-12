/// ActionDecoder: region-token projection + frozen Qwen2.5-0.5B (ADR 0012).
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
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        Autodiff,
    },
    module::Module,
    nn::{Linear, LinearConfig},
    optim::adaptor::OptimizerAdaptor,
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Tensor, TensorData},
};
use ndarray::Array3;
use ort::{session::Session, value::TensorRef};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

pub const LM_HIDDEN: usize = 896; // Qwen2.5-0.5B hidden dimension
pub const N_REGIONS: usize = 5;
const EPISODE_DIM: usize = 640;
const PROJ_LR: f64 = 1e-4;
const EOS_TOKEN_ID: u32 = 151643; // Qwen2.5 base EOS (<|endoftext|>)

type TrainB = Autodiff<NdArray>;
type ProjOptim = OptimizerAdaptor<AdamW, ProjModel<TrainB>, TrainB>;

#[derive(Module, Debug)]
struct ProjModel<B: burn::tensor::backend::Backend> {
    projection: Linear<B>,
}

pub struct ActionDecoder {
    model: Option<ProjModel<TrainB>>,
    optimizer: ProjOptim,
    /// Frozen embedding table: flat [vocab_size * LM_HIDDEN], row-major f32.
    embed_table: Vec<f32>,
    /// Transformer-only (no LM head), static [1,5,896].
    /// Moved to GainWorker via `create_gain_worker`; None afterwards.
    gain_session: Option<Session>,
    /// Full model with LM head, dynamic seq_len — used by generate().
    lm_session: Session,
    tokenizer: Tokenizer,
    /// Current gain multipliers: set by update_gain, applied to the next episode's k-WTA inputs.
    #[allow(dead_code)]
    pub gain: [f32; 5],
    device: NdArrayDevice,
    /// Cross-cycle LM hidden-state cache. Clips loop so token sequences repeat; after the first
    /// cycle each unique text pays the ORT cost exactly once for the lifetime of the session.
    lm_hidden_cache: std::collections::HashMap<Vec<u32>, Vec<f32>>,
}

impl ActionDecoder {
    /// Load all model artifacts from `models_dir`.
    /// Returns Err if any required file is absent.
    ///
    /// `ort_profile_dir`: when `Some`, enables ORT Chrome-trace profiling for
    /// lm_session (CPU). Call `end_profiling()` before process exit to flush.
    pub fn new(
        models_dir: &std::path::Path,
        device: NdArrayDevice,
        ort_profile_dir: Option<&std::path::Path>,
    ) -> Result<Self> {
        let gain_path = models_dir.join("qwen25_gain.onnx");
        let lm_path = models_dir.join("qwen25_lm.onnx");
        let embed_path = models_dir.join("qwen25_embed_table.safetensors");
        let tok_path = models_dir.join("qwen25_tokenizer.json");
        for p in [&gain_path, &lm_path, &embed_path, &tok_path] {
            if !p.exists() {
                bail!("action decoder model file not found: {}", p.display());
            }
        }

        let gain_session = Some(crate::migraphx_session(&gain_path)?);
        let lm_profile_prefix = ort_profile_dir.map(|d| d.join("qwen25_lm"));
        let lm_session = crate::cpu_session(&lm_path, lm_profile_prefix.as_deref())?;

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
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

        let projection = LinearConfig::new(EPISODE_DIM, N_REGIONS * LM_HIDDEN)
            .with_bias(false)
            .init::<TrainB>(&device);
        let model = ProjModel { projection };
        let optimizer: ProjOptim = AdamWConfig::new().init::<TrainB, ProjModel<TrainB>>();

        Ok(Self {
            model: Some(model),
            optimizer,
            embed_table,
            gain_session,
            lm_session,
            tokenizer,
            gain: [1.0; 5],
            device,
            lm_hidden_cache: std::collections::HashMap::new(),
        })
    }

    /// Project episode vector → 4 region tokens → LM forward →
    /// update `self.gain` from hidden-state L2 norms (ADR 0012).
    ///
    /// No-ops if gain_session was moved to GainWorker via `create_gain_worker`.
    #[allow(dead_code)]
    pub fn update_gain(&mut self, episode: &[f32]) -> Result<()> {
        if self.gain_session.is_none() {
            return Ok(());
        }
        // Compute projection before borrowing gain_session to avoid self re-borrow conflict.
        let proj_flat = self.project_flat(episode)?;

        let sess = self.gain_session.as_mut().unwrap();
        let inputs = Array3::from_shape_vec((1, N_REGIONS, LM_HIDDEN), proj_flat)
            .map_err(|e| anyhow::anyhow!("ndarray shape error: {e}"))?;
        let tensor = TensorRef::from_array_view(&inputs)?;
        // Extract hidden data inside a block so outputs (and its borrow of sess) drops
        // before we write to self.gain.
        let norms: [f32; 5] = {
            let outputs = sess.run(ort::inputs!["inputs_embeds" => tensor])?;
            let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
            if shape.len() < 3 || shape[1] != N_REGIONS as i64 {
                bail!("unexpected last_hidden_state shape: {:?}", shape);
            }
            let h = data.to_vec();
            std::array::from_fn(|i| {
                let row = &h[i * LM_HIDDEN..(i + 1) * LM_HIDDEN];
                row.iter().map(|x| x * x).sum::<f32>().sqrt()
            })
        };

        let max_n = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: [f32; 5] = norms.map(|n| ((n - max_n) / 10.0).exp());
        let sum_e: f32 = exp.iter().sum();
        let attn = exp.map(|e| e / sum_e);
        self.gain = attn.map(|a| (a * 5.0).clamp(0.5, 2.0));
        tracing::info!(gain = ?self.gain, norms = ?norms, "action gain updated");
        Ok(())
    }

    /// Autoregressive greedy generation given a current episode vector and a text query.
    ///
    /// Prepends the N_REGIONS projected region tokens as a soft prefix before the query tokens.
    /// Returns the generated text (decoded, EOS stripped).
    pub fn generate(&mut self, episode: &[f32], query: &str, max_tokens: usize) -> Result<String> {
        let proj_flat = self.project_flat(episode)?; // [N_REGIONS * LM_HIDDEN]

        let query_ids = self.tokenize(query)?;
        let query_embeds = self.embed_flat(&query_ids)?; // [n_query * LM_HIDDEN]

        // Concatenate region tokens + query embeds → [N_REGIONS + n_query, LM_HIDDEN]
        let mut all_embeds: Vec<f32> = proj_flat;
        all_embeds.extend_from_slice(&query_embeds);
        let mut seq_len = N_REGIONS + query_ids.len();

        let mut generated: Vec<u32> = Vec::new();

        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..max_tokens {
            // Scope the ORT call so SessionOutputs (and its internal borrow of lm_session)
            // is dropped before embed_flat borrows self again.
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
                // Repetition penalty: divide logit by 1.3 for each previously generated token.
                // Prevents degenerate loops common in greedy decoding.
                let mut penalized: Vec<f32> = last_logits.to_vec();
                for &id in &generated {
                    penalized[id as usize] /= 1.3;
                }
                argmax(&penalized)
                // outputs dropped here; lm_session borrow released
            };

            generated.push(next_id);
            if next_id == EOS_TOKEN_ID {
                break;
            }
            let next_embed = self.embed_flat(&[next_id])?;
            all_embeds.extend_from_slice(&next_embed);
            seq_len += 1;
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;
        Ok(text)
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
        let ep_t: Tensor<TrainB, 2> = Tensor::from_data(ep_data, &self.device);

        let model = self.model.as_ref().unwrap();
        let proj_out = model.projection.forward(ep_t); // [1, N_REGIONS * LM_HIDDEN]
        let proj_out = proj_out.reshape([N_REGIONS, LM_HIDDEN]); // [N_REGIONS, LM_HIDDEN]
        let proj_sum = proj_out.sum_dim(0); // [1, LM_HIDDEN]
        let sq_sum = proj_sum.clone().powf_scalar(2.0).sum_dim(1); // [1, 1]
        let norm = (sq_sum + 1e-8f32).sqrt(); // [1, 1]
        let proj_unit = proj_sum / norm; // [1, LM_HIDDEN]

        // Detached text target tensor.
        let text_data = TensorData::new(text_mean, [1, LM_HIDDEN]);
        let text_unit: Tensor<TrainB, 2> = Tensor::from_data(text_data, &self.device);

        let loss = (proj_unit - text_unit).powf_scalar(2.0).mean();
        let loss_val: f32 = loss.clone().into_scalar();

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
    /// Embeds `token_ids` via the frozen embed table, runs them through the full
    /// LM transformer (qwen25_lm.onnx, `last_hidden_state` output), and returns
    /// the mean-pooled, L2-normalised [LM_HIDDEN] vector.
    ///
    /// Used as the distillation target for projection training: richer than the
    /// static embed-table mean because Qwen contextualises the tokens before
    /// pooling, encoding inter-token relationships.
    pub fn text_hidden_states(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Ok(vec![0.0; LM_HIDDEN]);
        }
        if let Some(cached) = self.lm_hidden_cache.get(token_ids) {
            return Ok(cached.clone());
        }
        let embeds = self.embed_flat(token_ids)?;
        let n = token_ids.len();
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
        let hidden = hidden_data.to_vec();
        // Mean-pool over token positions.
        let mut mean = vec![0.0f32; LM_HIDDEN];
        for tok in 0..n {
            for d in 0..LM_HIDDEN {
                mean[d] += hidden[tok * LM_HIDDEN + d];
            }
        }
        let scale = 1.0 / n as f32;
        for v in &mut mean {
            *v *= scale;
        }
        // L2-normalise.
        let norm = (mean.iter().map(|x| x * x).sum::<f32>() + 1e-8).sqrt();
        for v in &mut mean {
            *v /= norm;
        }
        self.lm_hidden_cache
            .insert(token_ids.to_vec(), mean.clone());
        Ok(mean)
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
        let ep_t: Tensor<TrainB, 2> =
            Tensor::from_data(TensorData::new(flat_eps, [n, EPISODE_DIM]), &self.device);

        let model = self.model.as_ref().unwrap();
        let proj_out = model.projection.forward(ep_t); // [N, N_REGIONS * LM_HIDDEN]
        let proj_out = proj_out.reshape([n, N_REGIONS, LM_HIDDEN]); // [N, N_REGIONS, LM_HIDDEN]
        let proj_sum = proj_out.sum_dim(1).reshape([n, LM_HIDDEN]); // [N, LM_HIDDEN]
        let sq_sum = proj_sum.clone().powf_scalar(2.0).sum_dim(1); // [N, 1]
        let norm = (sq_sum + 1e-8f32).sqrt(); // [N, 1]
        let proj_unit = proj_sum / norm; // [N, LM_HIDDEN]

        let text_data = TensorData::new(text_targets, [n, LM_HIDDEN]);
        let text_unit: Tensor<TrainB, 2> = Tensor::from_data(text_data, &self.device);

        let loss = (proj_unit - text_unit).powf_scalar(2.0).mean();
        let loss_val: f32 = loss.clone().into_scalar();

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
        let ep_t: Tensor<TrainB, 2> =
            Tensor::from_data(TensorData::new(flat_eps, [n, EPISODE_DIM]), &self.device);

        let model = self.model.as_ref().unwrap();
        let proj_out = model.projection.forward(ep_t); // [N, N_REGIONS * LM_HIDDEN]
        let proj_out = proj_out.reshape([n, N_REGIONS, LM_HIDDEN]);
        let proj_sum = proj_out.sum_dim(1).reshape([n, LM_HIDDEN]);
        let sq_sum = proj_sum.clone().powf_scalar(2.0).sum_dim(1);
        let norm = (sq_sum + 1e-8f32).sqrt();
        let proj_unit = proj_sum / norm; // [N, LM_HIDDEN], L2-normalised

        // Targets are already normalised; load as detached tensor.
        let target_t: Tensor<TrainB, 2> =
            Tensor::from_data(TensorData::new(flat_hidden, [n, LM_HIDDEN]), &self.device);

        let loss = (proj_unit - target_t).powf_scalar(2.0).mean();
        let loss_val: f32 = loss.clone().into_scalar();

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

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Frobenius norm of the projection weight matrix (monitoring, Phase 8).
    #[allow(dead_code)]
    pub fn proj_weight_norm(&self) -> Result<f32> {
        let model = self.model.as_ref().unwrap();
        let w_data = model.projection.weight.val().into_data();
        let flat = w_data.convert::<f32>().to_vec::<f32>().unwrap();
        Ok(flat.iter().map(|x| x * x).sum::<f32>().sqrt())
    }

    /// Move the gain ORT session to a background `GainWorker` thread.
    ///
    /// Call once at startup (after weights are loaded). The session is not duplicated —
    /// it moves entirely to the worker thread (no second GPU allocation).
    /// Returns None if the gain session was already moved or was never loaded.
    pub fn create_gain_worker(&mut self) -> Option<GainWorker> {
        let sess = self.gain_session.take()?;
        let proj_w = self.proj_weights_flat();
        Some(GainWorker::spawn(sess, proj_w))
    }

    /// Snapshot the projection linear weights as a flat Vec<f32> (shape [n_in, n_out]).
    /// Used to initialize or refresh the GainWorker after NREM training.
    pub fn proj_weights_flat(&self) -> Vec<f32> {
        self.model
            .as_ref()
            .unwrap()
            .projection
            .weight
            .val()
            .detach()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap()
    }

    /// Save projection weights to a burn binary file.
    pub fn save_weights(&self, path: &std::path::Path) -> Result<()> {
        let model_clone = self.model.as_ref().unwrap().clone();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 path"))?;
        model_clone
            .save_file(path_str, &recorder)
            .map_err(|e| anyhow::anyhow!("save_weights failed: {e}"))
    }

    /// Load projection weights from a burn binary file.
    pub fn load_weights(&mut self, path: &std::path::Path) -> Result<()> {
        // Extract path string before taking the model so a bad-path error
        // doesn't leave self.model = None.
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 path"))?;
        let model = self.model.take().unwrap();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        match model.load_file(path_str, &recorder, &self.device) {
            Ok(loaded) => {
                self.model = Some(loaded);
                Ok(())
            }
            Err(e) => {
                // load_file consumes the model; restore a fresh one so callers
                // remain functional after a failed weight restore.
                let projection = LinearConfig::new(EPISODE_DIM, N_REGIONS * LM_HIDDEN)
                    .with_bias(false)
                    .init::<TrainB>(&self.device);
                self.model = Some(ProjModel { projection });
                Err(anyhow::anyhow!("load_weights failed: {e}"))
            }
        }
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Project episode → flat Vec<f32> of length [N_REGIONS * LM_HIDDEN].
    fn project_flat(&self, episode: &[f32]) -> Result<Vec<f32>> {
        let model = self.model.as_ref().unwrap();
        let ep_data = TensorData::new(episode.to_vec(), [1, EPISODE_DIM]);
        let ep_t: Tensor<TrainB, 2> = Tensor::from_data(ep_data, &self.device);
        let proj = model.projection.forward(ep_t); // [1, N_REGIONS * LM_HIDDEN]
        let flat = proj.into_data().convert::<f32>().to_vec::<f32>().unwrap();
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
    pub fn spawn(session: Session, proj_weights: Vec<f32>) -> Self {
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
                            let gain =
                                gain_from_episode(&mut session, &proj, &ep).unwrap_or_else(|e| {
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
fn gain_from_episode(session: &mut Session, proj_w: &[f32], episode: &[f32]) -> Result<[f32; 5]> {
    // project_flat: episode [EPISODE_DIM] @ proj_w [EPISODE_DIM, N_REGIONS * LM_HIDDEN]
    // Burn stores Linear weights as [n_in, n_out] (no transpose needed for y = x @ W).
    let n_out = N_REGIONS * LM_HIDDEN;
    let mut proj_flat = vec![0.0f32; n_out];
    for (i, &xi) in episode.iter().enumerate() {
        let row = &proj_w[i * n_out..(i + 1) * n_out];
        for (j, &wij) in row.iter().enumerate() {
            proj_flat[j] += xi * wij;
        }
    }

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

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3, 0.9, 0.2]), 3);
        assert_eq!(argmax(&[1.0]), 0);
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
