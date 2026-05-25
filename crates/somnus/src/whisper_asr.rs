/// WhisperDecoder: greedy ASR from Whisper-tiny encoder hidden states (ADR 0013).
///
/// Takes the raw (1500, 384) encoder hidden state produced by the Whisper encoder ONNX,
/// runs the Whisper-tiny decoder with a fixed English-transcription prefix, and returns
/// decoded text.  The decoder ONNX projects only the last token position (last-token-only
/// wrapper, ~10× faster than full-sequence LM head).
use anyhow::Result;
use ndarray::{Array2, Array3};
use ort::{session::Session, value::TensorRef};
use tokenizers::Tokenizer;

pub const WHISPER_ENC_SEQ: usize = 1500;
pub const WHISPER_ENC_DIM: usize = 384;

// Must match WHISPER_DECODER_MAX_SEQ in scripts/export_models.py.
// decoder_input_ids is always padded to this length; logits read from position (real_len-1).
const DECODER_MAX_SEQ: usize = 204; // 4-token prefix + 200 max generated tokens

// Whisper special token IDs (multilingual model, whisper-tiny)
const SOT: i64 = 50258;
const EOT: i64 = 50257;
const EN: i64 = 50259;
const TRANSCRIBE: i64 = 50359;
const NO_TIMESTAMPS: i64 = 50363;

pub struct WhisperDecoder {
    session: Session,
    tokenizer: Tokenizer,
}

impl WhisperDecoder {
    pub fn new(models_dir: &std::path::Path) -> Result<Self> {
        let dec_path = models_dir.join("whisper_tiny_decoder.onnx");
        let tok_path = models_dir.join("whisper_tokenizer.json");
        if !dec_path.exists() || !tok_path.exists() {
            anyhow::bail!(
                "whisper decoder files not found (need whisper_tiny_decoder.onnx + whisper_tokenizer.json)"
            );
        }
        // MIGraphX does not support encoder-decoder cross-attention; use CPU.
        let session = crate::cpu_session(&dec_path)?;
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("whisper tokenizer load failed: {e}"))?;
        Ok(Self { session, tokenizer })
    }

    /// Greedy decode from encoder hidden states.
    ///
    /// `encoder_hidden`: flat slice of length 1500 * 384 = 576 000.
    /// Returns decoded text (trimmed), or empty string for silence/noise.
    ///
    /// The decoder ONNX expects static shape (1, DECODER_MAX_SEQ) for decoder_input_ids.
    /// We maintain a padded buffer; at each step we read logits from position (real_len-1).
    /// Causal attention means padding zeros after real_len have no effect on earlier positions.
    pub fn decode_greedy(&mut self, encoder_hidden: &[f32], max_tokens: usize) -> Result<String> {
        let max_tokens = max_tokens.min(DECODER_MAX_SEQ - 4); // 4 = prefix length

        let enc_arr = Array3::from_shape_vec(
            (1, WHISPER_ENC_SEQ, WHISPER_ENC_DIM),
            encoder_hidden.to_vec(),
        )
        .map_err(|e| anyhow::anyhow!("encoder hidden shape error: {e}"))?;

        // Padded token buffer: prefix tokens + zeros for the rest.
        let mut padded = vec![0i64; DECODER_MAX_SEQ];
        padded[0] = SOT; padded[1] = EN; padded[2] = TRANSCRIBE; padded[3] = NO_TIMESTAMPS;
        let mut generated: Vec<u32> = Vec::new();

        let enc_tensor_base = enc_arr.clone();

        for real_len in 4..4 + max_tokens {
            let dec_arr = Array2::from_shape_vec((1, DECODER_MAX_SEQ), padded.clone())
                .map_err(|e| anyhow::anyhow!("decoder ids shape error: {e}"))?;

            let enc_tensor = TensorRef::from_array_view(&enc_tensor_base)?;
            let dec_tensor = TensorRef::from_array_view(&dec_arr)?;

            let outputs = self.session.run(ort::inputs![
                "encoder_hidden_states" => enc_tensor,
                "decoder_input_ids"    => dec_tensor,
            ])?;

            // logits shape: (1, DECODER_MAX_SEQ, vocab_size) — read position (real_len - 1)
            let (_shape, logits_data) = outputs["logits"].try_extract_tensor::<f32>()?;
            let logits_flat: Vec<f32> = logits_data.to_vec();
            let vocab_size = logits_flat.len() / DECODER_MAX_SEQ;
            let step_logits = &logits_flat[(real_len - 1) * vocab_size..real_len * vocab_size];

            // Repetition penalty: divide logit by 1.3 for each previously generated token.
            let mut penalized: Vec<f32> = step_logits.to_vec();
            for &id in &generated {
                penalized[id as usize] /= 1.3;
            }
            let next_id = argmax(&penalized) as i64;

            if next_id == EOT {
                break;
            }
            padded[real_len] = next_id;
            if next_id < EOT {
                generated.push(next_id as u32);
            }
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("whisper decode failed: {e}"))?;
        Ok(text.trim().to_string())
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
