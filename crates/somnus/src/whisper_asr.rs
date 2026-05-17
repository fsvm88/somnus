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
        let session = Session::builder()?.commit_from_file(&dec_path)?;
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("whisper tokenizer load failed: {e}"))?;
        Ok(Self { session, tokenizer })
    }

    /// Greedy decode from encoder hidden states.
    ///
    /// `encoder_hidden`: flat slice of length 1500 * 384 = 576 000.
    /// Returns decoded text (trimmed), or empty string for silence/noise.
    pub fn decode_greedy(&mut self, encoder_hidden: &[f32], max_tokens: usize) -> Result<String> {
        let enc_arr =
            Array3::from_shape_vec((1, WHISPER_ENC_SEQ, WHISPER_ENC_DIM), encoder_hidden.to_vec())
                .map_err(|e| anyhow::anyhow!("encoder hidden shape error: {e}"))?;

        // Fixed English-transcription prefix: SOT, EN, TRANSCRIBE, NO_TIMESTAMPS
        let mut tokens: Vec<i64> = vec![SOT, EN, TRANSCRIBE, NO_TIMESTAMPS];
        let mut text_ids: Vec<u32> = Vec::new();

        for _ in 0..max_tokens {
            let seq_len = tokens.len();
            let dec_arr = Array2::from_shape_vec((1, seq_len), tokens.clone())
                .map_err(|e| anyhow::anyhow!("decoder ids shape error: {e}"))?;

            let enc_tensor = TensorRef::from_array_view(&enc_arr)?;
            let dec_tensor = TensorRef::from_array_view(&dec_arr)?;

            let outputs = self.session.run(ort::inputs![
                "encoder_hidden_states" => enc_tensor,
                "decoder_input_ids"    => dec_tensor,
            ])?;

            // logits shape: (1, 1, vocab_size) — last position only
            let (_shape, logits_data) = outputs["logits"].try_extract_tensor::<f32>()?;
            let logits: Vec<f32> = logits_data.to_vec();
            let next_id = argmax(&logits) as i64;

            if next_id == EOT {
                break;
            }
            tokens.push(next_id);
            // Only accumulate regular text tokens (id < EOT = 50257)
            if next_id < EOT {
                text_ids.push(next_id as u32);
            }
        }

        let text = self
            .tokenizer
            .decode(&text_ids, true)
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
