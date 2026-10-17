/// WhisperDecoder: greedy ASR from Whisper-tiny encoder hidden states (ADR 0013).
///
/// Takes the raw (1500, 384) encoder hidden state produced by the Whisper encoder ONNX,
/// runs the Whisper-tiny decoder with a fixed English-transcription prefix, and returns
/// decoded text.  The decoder ONNX projects only the last token position (last-token-only
/// wrapper, ~10× faster than full-sequence LM head).
use crate::util::argmax;
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
        // Whisper decoder is moved into a spawn_blocking thread; we can't call
        // end_profiling() on it before exit, so ORT profiling is disabled here.
        // Use the flamegraph (perf) for CPU cost analysis of the ASR path.
        let session = crate::cpu_session(&dec_path, None)?;
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
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::AsrDecode,
            self.decode_greedy_inner(encoder_hidden, max_tokens)
        );
        result
    }

    fn decode_greedy_inner(&mut self, encoder_hidden: &[f32], max_tokens: usize) -> Result<String> {
        let max_tokens = max_tokens.min(DECODER_MAX_SEQ - 4); // 4 = prefix length

        let enc_arr = Array3::from_shape_vec(
            (1, WHISPER_ENC_SEQ, WHISPER_ENC_DIM),
            encoder_hidden.to_vec(),
        )
        .map_err(|e| anyhow::anyhow!("encoder hidden shape error: {e}"))?;

        // Padded token buffer: prefix tokens + zeros for the rest.
        let mut padded = vec![0i64; DECODER_MAX_SEQ];
        padded[0] = SOT;
        padded[1] = EN;
        padded[2] = TRANSCRIBE;
        padded[3] = NO_TIMESTAMPS;
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

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn models_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
    }

    /// Skip-with-message when the decoder artifacts aren't on disk (gitignored
    /// models tree, same pattern as somnus-encoding's encode_models.rs).
    fn decoder() -> Option<WhisperDecoder> {
        let dir = models_dir();
        if !dir.join("whisper_tiny_decoder.onnx").exists()
            || !dir.join("whisper_tokenizer.json").exists()
        {
            eprintln!("skipping: whisper decoder artifacts not present (gitignored)");
            return None;
        }
        Some(WhisperDecoder::new(&dir).expect("WhisperDecoder::new"))
    }

    // ── argmax (pure) ────────────────────────────────────────────────────────

    // ── decoder (model-gated) ────────────────────────────────────────────────

    /// Drift pin (§17/§22/§24 bug class): the hardcoded special-token IDs must
    /// match the actual tokenizer file on disk. A re-exported model with a
    /// different vocabulary would otherwise fail silently at runtime by decoding
    /// garbage.
    #[test]
    fn special_token_ids_match_tokenizer() {
        let Some(dec) = decoder() else { return };
        let expect = [
            (SOT, "<|startoftranscript|>"),
            (EOT, "<|endoftext|>"),
            (EN, "<|en|>"),
            (TRANSCRIBE, "<|transcribe|>"),
            (NO_TIMESTAMPS, "<|notimestamps|>"),
        ];
        for (id, name) in expect {
            let got = dec.tokenizer.id_to_token(id as u32);
            assert_eq!(got.as_deref(), Some(name), "token id {id} drifted");
        }
    }

    /// max_tokens = 0 never runs the decoder loop: empty text out, no ORT call.
    #[test]
    fn decode_max_tokens_zero_is_empty() {
        let Some(mut dec) = decoder() else { return };
        let hidden = vec![0.0f32; WHISPER_ENC_SEQ * WHISPER_ENC_DIM];
        let text = dec.decode_greedy(&hidden, 0).expect("decode max_tokens=0");
        assert_eq!(text, "");
    }

    /// Wrong-length encoder hidden must be a clean error, not a panic or a
    /// mis-shaped ORT call (encode_speech always produces exactly 1500×384).
    #[test]
    fn decode_wrong_hidden_length_errors() {
        let Some(mut dec) = decoder() else { return };
        assert!(dec.decode_greedy(&vec![0.0f32; 100], 8).is_err());
        assert!(dec
            .decode_greedy(&vec![0.0f32; WHISPER_ENC_SEQ * WHISPER_ENC_DIM + 1], 8)
            .is_err());
    }

    /// Full decode loop against the real CPU decoder session on synthetic input:
    /// must not error and must be deterministic (frozen model, greedy search).
    #[test]
    fn decode_zeros_hidden_deterministic() {
        let Some(mut dec) = decoder() else { return };
        let hidden = vec![0.0f32; WHISPER_ENC_SEQ * WHISPER_ENC_DIM];
        let a = dec.decode_greedy(&hidden, 8).expect("decode run 1");
        let b = dec.decode_greedy(&hidden, 8).expect("decode run 2");
        assert_eq!(a, b, "ASR decode not deterministic on identical input");
    }

    /// End-to-end critical path: real encoder hidden states (from the actual
    /// whisper encoder ONNX) fed to the decoder — exactly the shape of the data
    /// AudioProcessor sends to the ASR thread. Content is input-dependent (a
    /// sine may transcribe or hallucinate), so only determinism is asserted.
    #[test]
    fn decode_real_encoder_hidden_deterministic() {
        let dir = models_dir();
        // AudioEncoder::new loads both the whisper encoder and the env encoder.
        #[cfg(not(feature = "mn10_env"))]
        let env_file = "clap_audio_encoder_fp16.onnx";
        #[cfg(feature = "mn10_env")]
        let env_file = "efficientat_mn10_encoder_fp16.onnx";
        if !dir.join("whisper_tiny_encoder.onnx").exists() || !dir.join(env_file).exists() {
            eprintln!("skipping: whisper/env encoder artifacts not present (gitignored)");
            return;
        }
        let Some(mut dec) = decoder() else { return };

        let mut enc = somnus_encoding::AudioEncoder::new(&dir).expect("AudioEncoder::new");
        let pcm: Vec<f32> = (0..480_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let (_emb, hidden) = enc
            .encode_speech(&pcm, somnus_sensory::Timestamp::now())
            .expect("encode_speech");
        assert_eq!(hidden.len(), WHISPER_ENC_SEQ * WHISPER_ENC_DIM);

        let a = dec.decode_greedy(&hidden, 16).expect("decode run 1");
        let b = dec.decode_greedy(&hidden, 16).expect("decode run 2");
        assert_eq!(a, b, "ASR decode not deterministic on real encoder output");
    }

    /// Stress battery for gaps.md §40 (intermittent ORT panic "GetElementType is not implemented"):
    /// hostile / extreme encoder hidden states through the real decoder, each under `catch_unwind`.
    /// Reports every panic with its message, then checks the poisoning question: does a known-good
    /// input still decode to its pre-battery output afterwards?
    /// `cargo test -p somnus --release whisper_asr::tests::stress -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn stress_decode_battery_for_ort_panic() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let Some(mut dec) = decoder() else { return };
        let n = WHISPER_ENC_SEQ * WHISPER_ENC_DIM;
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = move || {
            // xorshift64* -> roughly gaussian via sum of 4 uniforms
            let mut g = 0.0f32;
            for _ in 0..4 {
                seed ^= seed >> 12;
                seed ^= seed << 25;
                seed ^= seed >> 27;
                let u = (seed.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32;
                g += u - 0.5;
            }
            g * 1.7
        };
        let good: Vec<f32> = (0..n).map(|_| rnd() * 0.5).collect();
        let baseline = dec.decode_greedy(&good, 16).expect("baseline decode");
        println!("STRESS baseline text: {baseline:?}");

        let mut cases: Vec<(String, Vec<f32>)> = vec![];
        for scale in [1e-6f32, 1e-3, 1.0, 10.0, 1e3, 1e6, 1e12] {
            cases.push((
                format!("gauss x{scale:e}"),
                (0..n).map(|_| rnd() * scale).collect(),
            ));
        }
        cases.push(("all NaN".into(), vec![f32::NAN; n]));
        cases.push(("all +Inf".into(), vec![f32::INFINITY; n]));
        cases.push(("all -Inf".into(), vec![f32::NEG_INFINITY; n]));
        cases.push(("one NaN".into(), {
            let mut v = good.clone();
            v[12345] = f32::NAN;
            v
        }));
        cases.push(("one Inf".into(), {
            let mut v = good.clone();
            v[n / 2] = f32::INFINITY;
            v
        }));
        cases.push(("denormals".into(), vec![1e-40f32; n]));
        cases.push((
            "alt +/-1e30".into(),
            (0..n)
                .map(|i| if i % 2 == 0 { 1e30 } else { -1e30 })
                .collect(),
        ));
        cases.push(("constant 1".into(), vec![1.0; n]));
        cases.push((
            "sparse spikes".into(),
            (0..n)
                .map(|i| if i % 997 == 0 { 50.0 } else { 0.0 })
                .collect(),
        ));
        for k in 0..40 {
            cases.push((format!("gauss run {k}"), (0..n).map(|_| rnd()).collect()));
        }
        let (mut panics, mut errs, mut oks) = (0, 0, 0);
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        for (name, hidden) in &cases {
            for max_tokens in [1usize, 8, 64, 200] {
                match catch_unwind(AssertUnwindSafe(|| dec.decode_greedy(hidden, max_tokens))) {
                    Ok(Ok(_)) => oks += 1,
                    Ok(Err(e)) => {
                        errs += 1;
                        println!("STRESS err   [{name}, max_tokens={max_tokens}]: {e}");
                    }
                    Err(p) => {
                        panics += 1;
                        let msg = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default();
                        println!("STRESS PANIC [{name}, max_tokens={max_tokens}]: {msg}");
                    }
                }
            }
        }
        std::panic::set_hook(prev_hook);
        let after = catch_unwind(AssertUnwindSafe(|| dec.decode_greedy(&good, 16)));
        println!("STRESS summary: ok={oks} err={errs} panic={panics}; post-battery decode of known-good input: {:?} (baseline {baseline:?})", after.as_ref().map(|r| r.as_ref().ok()));
        if let Ok(Ok(t)) = after {
            assert_eq!(
                t, baseline,
                "decoder state poisoned: known-good output changed after the battery"
            );
        }
    }

    /// §40 concurrency hypothesis: two decoder sessions decoding on separate threads at once
    /// (the live pipeline runs the ASR thread next to the audio/video encoders and the LM).
    /// `cargo test -p somnus --release whisper_asr::tests::stress_concurrent -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn stress_concurrent_decoders_for_ort_panic() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let dir = models_dir();
        if !dir.join("whisper_tiny_decoder.onnx").exists() {
            return;
        }
        let n = WHISPER_ENC_SEQ * WHISPER_ENC_DIM;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        let handles: Vec<_> = (0..3u64)
            .map(|t| {
                let dir = dir.clone();
                std::thread::spawn(move || {
                    let mut dec = WhisperDecoder::new(&dir).unwrap();
                    let mut x = 0x1234_5678_9ABC_DEF0u64 ^ (t << 32);
                    let (mut ok, mut panics, mut errs) = (0u32, 0u32, 0u32);
                    while std::time::Instant::now() < deadline {
                        let hidden: Vec<f32> = (0..n)
                            .map(|_| {
                                x ^= x << 13;
                                x ^= x >> 7;
                                x ^= x << 17;
                                ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
                            })
                            .collect();
                        match catch_unwind(AssertUnwindSafe(|| dec.decode_greedy(&hidden, 32))) {
                            Ok(Ok(_)) => ok += 1,
                            Ok(Err(_)) => errs += 1,
                            Err(_) => panics += 1,
                        }
                    }
                    (ok, errs, panics)
                })
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            let (ok, errs, panics) = h.join().unwrap();
            println!("STRESS_CONC thread {i}: ok={ok} err={errs} panic={panics}");
            assert_eq!(panics, 0);
        }
    }
}
