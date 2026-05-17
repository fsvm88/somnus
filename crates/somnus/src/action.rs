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
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear_no_bias, AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use ndarray::Array3;
use ort::{session::Session, value::TensorRef};
use tokenizers::Tokenizer;

pub const LM_HIDDEN: usize = 896; // Qwen2.5-0.5B hidden dimension
pub const N_REGIONS: usize = 5;
const EPISODE_DIM: usize = 640;
const PROJ_LR: f64 = 1e-4;
const EOS_TOKEN_ID: u32 = 151643; // Qwen2.5 base EOS (<|endoftext|>)

pub struct ActionDecoder {
    projection: Linear,
    varmap: VarMap,
    optimizer: AdamW,
    /// Frozen embedding table [vocab_size, LM_HIDDEN] for training surrogate loss + generation.
    embed_table: Tensor,
    lm_session: Session,
    tokenizer: Tokenizer,
    /// Current gain multipliers: set by update_gain, applied to the next episode's k-WTA inputs.
    pub gain: [f32; 5],
    device: Device,
}

impl ActionDecoder {
    /// Load all model artifacts from `models_dir`.
    /// Returns Err if any required file is absent.
    pub fn new(models_dir: &std::path::Path, device: Device) -> Result<Self> {
        let lm_path = models_dir.join("qwen25_lm.onnx");
        let embed_path = models_dir.join("qwen25_embed_table.safetensors");
        let tok_path = models_dir.join("qwen25_tokenizer.json");
        for p in [&lm_path, &embed_path, &tok_path] {
            if !p.exists() {
                bail!("action decoder model file not found: {}", p.display());
            }
        }

        let lm_session = Session::builder()?.commit_from_file(&lm_path)?;

        let mut embed_tensors = candle_core::safetensors::load(&embed_path, &device)?;
        let embed_table = embed_tensors.remove("embed_tokens.weight").ok_or_else(|| {
            anyhow::anyhow!("embed_tokens.weight missing in {}", embed_path.display())
        })?;

        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let projection = linear_no_bias(EPISODE_DIM, N_REGIONS * LM_HIDDEN, vb.pp("projection"))?;
        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: PROJ_LR,
                ..Default::default()
            },
        )?;

        Ok(Self {
            projection,
            varmap,
            optimizer,
            embed_table,
            lm_session,
            tokenizer,
            gain: [1.0; 5],
            device,
        })
    }

    /// Project episode vector → 4 region tokens → LM forward →
    /// update `self.gain` from hidden-state L2 norms (ADR 0012).
    pub fn update_gain(&mut self, episode: &[f32]) -> Result<()> {
        let proj_flat = self.project_flat(episode)?; // [4 * LM_HIDDEN]

        let inputs = Array3::from_shape_vec((1, N_REGIONS, LM_HIDDEN), proj_flat)
            .map_err(|e| anyhow::anyhow!("ndarray shape error: {e}"))?;
        let tensor = TensorRef::from_array_view(&inputs)?;
        let outputs = self
            .lm_session
            .run(ort::inputs!["inputs_embeds" => tensor])?;
        let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        if shape.len() < 3 || shape[1] != N_REGIONS as i64 {
            bail!("unexpected last_hidden_state shape: {:?}", shape);
        }
        let hidden = data.to_vec();

        // L2 norm of each region position's hidden vector → softmax → clipped gain
        let norms: [f32; 5] = std::array::from_fn(|i| {
            let row = &hidden[i * LM_HIDDEN..(i + 1) * LM_HIDDEN];
            row.iter().map(|x| x * x).sum::<f32>().sqrt()
        });
        let max_n = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: [f32; 5] = norms.map(|n| ((n - max_n) / 10.0).exp()); // temperature = 10
        let sum_e: f32 = exp.iter().sum();
        let attn = exp.map(|e| e / sum_e);
        self.gain = attn.map(|a| (a * 5.0).clamp(0.5, 2.0));
        tracing::info!(gain = ?self.gain, norms = ?norms, "action gain updated");
        Ok(())
    }

    /// Autoregressive greedy generation given a current episode vector and a text query.
    ///
    /// Prepends the 4 projected region tokens as a soft prefix before the query tokens.
    /// Returns the generated text (decoded, EOS stripped).
    pub fn generate(&mut self, episode: &[f32], query: &str, max_tokens: usize) -> Result<String> {
        let proj_flat = self.project_flat(episode)?; // [4 * LM_HIDDEN]

        let query_ids = self.tokenize(query)?;
        let query_embeds = self.embed_flat(&query_ids)?; // [n_query * LM_HIDDEN]

        // Concatenate region tokens + query embeds → [4 + n_query, LM_HIDDEN]
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
    /// Loss: MSE between mean(projected region tokens) and mean(text token embeddings).
    /// Gradients flow through the projection only; embed_table is frozen (no Var).
    pub fn projection_step(&mut self, episode: &[f32], target_ids: &[u32]) -> Result<f32> {
        if target_ids.is_empty() {
            return Ok(0.0);
        }
        let dev = self.device.clone();
        let ep_t = Tensor::from_slice(episode, (1, EPISODE_DIM), &dev)?;
        let proj_out = self
            .projection
            .forward(&ep_t)?
            .reshape((N_REGIONS, LM_HIDDEN))?; // [4, LM_HIDDEN]
        let proj_mean = proj_out.mean(0)?; // [LM_HIDDEN]

        // Frozen text embeddings; no gradient
        let idx = Tensor::from_vec(target_ids.to_vec(), (target_ids.len(),), &dev)?;
        let text_embeds = self.embed_table.index_select(&idx, 0)?.detach(); // [n, LM_HIDDEN]
        let text_mean = text_embeds.mean(0)?; // [LM_HIDDEN]

        let loss = (proj_mean - text_mean)?.sqr()?.mean_all()?;
        let loss_val = loss.to_scalar::<f32>()?;
        self.optimizer.backward_step(&loss)?;
        Ok(loss_val)
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
        let data = self.varmap.data().lock().unwrap();
        let w = data["projection.weight"].as_tensor();
        Ok(w.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt())
    }

    /// Save projection weights to a safetensors file.
    pub fn save_weights(&self, path: &std::path::Path) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    /// Load projection weights from a safetensors file and rebuild the Linear.
    pub fn load_weights(&mut self, path: &std::path::Path) -> Result<()> {
        self.varmap.load(path)?;
        let w = {
            let data = self.varmap.data().lock().unwrap();
            data["projection.weight"].as_tensor().clone()
        };
        self.projection = Linear::new(w, None);
        Ok(())
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Project episode → flat Vec<f32> of length [4 * LM_HIDDEN].
    fn project_flat(&self, episode: &[f32]) -> Result<Vec<f32>> {
        let ep_t = Tensor::from_slice(episode, (1, EPISODE_DIM), &self.device)?;
        Ok(self.projection.forward(&ep_t)?.squeeze(0)?.to_vec1()?)
    }

    /// Embed token IDs → flat Vec<f32> of length [n * LM_HIDDEN].
    fn embed_flat(&self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let idx = Tensor::from_vec(ids.to_vec(), (ids.len(),), &self.device)?;
        let embeds = self.embed_table.index_select(&idx, 0)?; // [n, LM_HIDDEN]
        Ok(embeds.flatten_all()?.to_vec1()?)
    }
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
