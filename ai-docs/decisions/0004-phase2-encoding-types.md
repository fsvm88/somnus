# 0004. Phase 2 encoding pipeline: concrete types and interfaces

- **Status:** accepted
- **Date:** 2026-04-28

> **Decision:** EncoderOutput struct + ORT session types for the encoding pipeline; ort 2.0 against system ORT 1.24.4; parity-tested against HuggingFace Python reference (≤1e-4).  
> **Keywords:** EncoderOutput, ORT session, ort crate, encoding pipeline, speech_enc, env_enc, visual_enc, ONNX, parity test

## Context

ADR 0003 selected and pinned the three frozen encoders (Whisper-tiny speech, CLAP-HTSAT-unfused env, DINOv2-S/14 visual) and exported them to ONNX. This ADR fixes the concrete Rust types for `crates/somnus-encoding`: the new library crate wrapping OnnxRuntime inference and delivering typed embeddings.

**Constraints:** OnnxRuntime 1.24.4 installed system-wide (`/usr/lib/libonnxruntime.so`, dynamic linking, no bundled ORT). `ort` 2.0.0-rc.12 + `load-dynamic` confirmed against system ORT 1.24.4, resolves `.onnx.data` external-data correctly (`experiments/ort-smoke`). `ndarray="0.17"` required by ort's ndarray feature. `somnus-encoding` is sync-only (binary bridges with `spawn_blocking`, same pattern as `somnus-sensory`). Audio preprocessing (resampling + mel spectrogram) must run in Rust — ONNX models start at the mel-spectrogram boundary, raw PCM isn't accepted. `Timestamp` reused directly from `somnus-sensory` — no shared-core crate yet.

### Encoder input/output (from ADR 0003)

| Model | Input shape | Output shape | Dim |
|-------|-------------|--------------|-----|
| `whisper_tiny_encoder.onnx` | `(1, 80, 3000)` f32 — mel spec 30 s @ 16 kHz, 80 bins | `(1, 1500, 384)` f32 → mean-pooled to `(384,)` | 384 |
| `clap_audio_encoder.onnx` | `(1, 1, 1001, 64)` f32 — mel spec 10 s @ 48 kHz, 64 bins | `(1, 512)` f32 | 512 |
| `dinov2_small.onnx` | `(1, 3, 224, 224)` f32 — ImageNet-normalised RGB | `(1, 257, 384)` f32 → patch tokens `[1:]` mean-pooled to `(384,)` | 384 |

### Mel-spectrogram preprocessing parameters

All parameters verified against HuggingFace source
(`feature_extraction_whisper.py`, `feature_extraction_clap.py`, `audio_utils.py`).

**Whisper-tiny** (`WhisperFeatureExtractor` defaults, `_np_extract_fbank_features`):
- Sample rate: 16 000 Hz
- n_fft: 400, window: Hann, hop_length: 160
- n_mels: 80, fmin: 0 Hz, fmax: 8 000 Hz
- Mel scale: Slaney (HF default for Whisper)
- Pipeline:
  1. **Power** spectrogram: `|stft|²` (`power=2.0`)
  2. Mel filterbank → `(80, T)` where T ≈ 3001 for 480 000-sample input
  3. Floor: `max(mel, 1e-10)`
  4. Log: `log10(mel)` (not `10 * log10`)
  5. Trim last frame: `[:, :-1]` → `(80, 3000)` exactly
  6. Dynamic range: `maximum(log_spec, log_spec.max() - 8.0)`
  7. Normalise: `(log_spec + 4.0) / 4.0`
- Chunk: non-overlapping 30 s windows = 480 000 samples @ 16 kHz.

**CLAP-HTSAT-unfused** (`ClapFeatureExtractor` defaults, `_np_extract_fbank_features`):
- Sample rate: 48 000 Hz
- n_fft: 1024, window: Hann, hop_length: 480
- n_mels: 64, **fmin: 50 Hz**, fmax: 14 000 Hz
- **Mel scale: Slaney (area-normalised triangular filters, `mel_filters_slaney`).**
  The ADR originally said HTK with `truncation="fusion"` — this was wrong.
  The actual default is `truncation="rand_trunc"`, which uses `mel_filters_slaney`
  (librosa Slaney) not HTK. Verified by fixture parity tests.
- Pipeline:
  1. **Power** spectrogram: `|stft|²` (`power=2.0`)
  2. Mel filterbank → `(64, T)`
  3. Floor: `max(mel, 1e-10)`
  4. `power_to_db`: `10 * log10(mel / reference)` with `reference=1.0` → `10 * log10(mel)`
  5. Transpose: `(T, 64)` → final shape `(1001, 64)` for a 10 s chunk
  6. **No** per-utterance mean/std normalisation; no `db_range` clipping (defaults).
- Chunk: non-overlapping 10 s windows = 480 000 samples @ 48 kHz.

**DINOv2-S/14** (ImageNet normalisation):
- Resize frame to 224 × 224 (bilinear).
- Divide by 255.0 then subtract mean `[0.485, 0.456, 0.406]`, divide by std
  `[0.229, 0.224, 0.225]` — per-channel, channel-first layout.

## Decision

### Crate dependency additions (`crates/somnus-encoding/Cargo.toml`)

```toml
[dependencies]
anyhow          = "1"
somnus-sensory     = { path = "../somnus-sensory" }    # for Timestamp
ndarray         = "0.17"
ort             = { version = "2.0.0-rc.12", features = ["load-dynamic", "ndarray"] }
rubato          = "0.16"   # sinc resampler; needed to hit 16 kHz and 48 kHz targets
rustfft         = "6"      # FFT for mel spectrogram computation
thiserror       = "2"
tracing         = "0.1"
```

`rubato` is the resampler; `rustfft` underpins the mel computation. Both are
pure-Rust and compile without system headers.

### Module layout (`crates/somnus-encoding/src/`)

```
lib.rs            — re-exports; EncodingError, EncodingResult
audio.rs          — AudioEncoder, AudioEmbedding, AudioEncoderSource
video.rs          — VideoEncoder, VideoEmbedding
preprocess/
    mel.rs        — mel spectrogram: WhisperMel, ClapMel
    resample.rs   — SincResampler wrapper around rubato
    image.rs      — resize + ImageNet normalise
```

### Embedding types

```rust
// audio.rs
pub struct AudioEmbedding {
    pub source:    AudioEncoderSource,
    pub values:    Vec<f32>,         // len = 384 (Speech) or 512 (Env)
    pub timestamp: Timestamp,        // start of the audio chunk
}

pub enum AudioEncoderSource {
    Speech,   // Whisper-tiny, 384-dim, speech content
    Env,      // CLAP-HTSAT, 512-dim, environmental sound
}

// video.rs
pub struct VideoEmbedding {
    pub values:    Vec<f32>,         // len = 384, DINOv2 mean-pooled patches
    pub timestamp: Timestamp,
}
```

Single `AudioEmbedding` type with a source discriminant is preferred over two
separate types because the downstream channel topology (a single audio-embedding
channel fanning to two regional concept areas) is simpler to implement and the
discriminant carries the routing key explicitly.

### Encoder handles

```rust
// audio.rs
pub struct AudioEncoder {
    speech_session: ort::session::Session,  // whisper_tiny_encoder.onnx
    env_session:    ort::session::Session,  // clap_audio_encoder.onnx
    speech_mel:     WhisperMel,             // pre-built mel filterbank
    env_mel:        ClapMel,
}

impl AudioEncoder {
    /// Build both sessions from the given model directory.
    pub fn new(models_dir: &Path) -> Result<Self, EncodingError>;

    /// Compute a Whisper speech embedding for exactly 480 000 samples @ 16 kHz.
    ///
    /// Caller is responsible for resampling to 16 kHz before calling.
    /// Returns one embedding. Panics in debug mode if len != 480 000.
    pub fn encode_speech(&mut self, pcm_16k: &[f32], ts: Timestamp)
        -> Result<AudioEmbedding, EncodingError>;

    /// Compute a CLAP env embedding for exactly 480 000 samples @ 48 kHz.
    ///
    /// Caller is responsible for resampling to 48 kHz before calling.
    pub fn encode_env(&mut self, pcm_48k: &[f32], ts: Timestamp)
        -> Result<AudioEmbedding, EncodingError>;
}

// video.rs
pub struct VideoEncoder {
    session: ort::session::Session,   // dinov2_small.onnx
}

impl VideoEncoder {
    pub fn new(models_dir: &Path) -> Result<Self, EncodingError>;

    /// Encode one RGB frame (any size, any PixelFormat with u8 data).
    ///
    /// Internally: convert to RGB24 if needed, resize to 224×224,
    /// ImageNet-normalise, run DINOv2, mean-pool patch tokens.
    pub fn encode(&mut self, frame: &VideoFrame) -> Result<VideoEmbedding, EncodingError>;
}
```

Resampling is the *caller's* responsibility. The binary (or a thin wrapper in
`audio.rs`) manages the `rubato::SincFixedIn` resampler instances and accumulation
buffers that collect incoming `AudioFrame`s until a full 30 s / 10 s chunk is
available.

### Error type

```rust
// lib.rs
#[derive(Debug, thiserror::Error)]
pub enum EncodingError {
    #[error("OnnxRuntime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("expected {expected} samples, got {got}")]
    WrongLength { expected: usize, got: usize },

    #[error("unexpected output shape {got:?}, expected {expected:?}")]
    ShapeMismatch { expected: Vec<i64>, got: Vec<i64> },

    #[error("unsupported pixel format: {0:?}")]
    UnsupportedPixelFormat(PixelFormat),

    #[error("image processing error: {0}")]
    Image(String),
}

pub type EncodingResult<T> = Result<T, EncodingError>;
```

### ORT initialisation

`ort::init_from("/usr/lib/libonnxruntime.so")?.commit()` is called once by the
binary before any encoder is constructed. Encoder construction does not call
`init_from` itself (it would fail silently if called after the first initialisation).

### Mean-pooling (Whisper and DINOv2)

```rust
// Whisper: (1, 1500, 384) → (384,)
fn mean_pool_2d(tensor: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    for t in 0..seq_len {
        for d in 0..dim {
            out[d] += tensor[t * dim + d];
        }
    }
    let n = seq_len as f32;
    out.iter_mut().for_each(|v| *v /= n);
    out
}

// DINOv2: patch tokens = tokens[1..] → (256, 384), mean-pool → (384,)
// Token 0 is the CLS token and is skipped per ADR 0003.
```

## Implementation notes (from parity testing)

1. **`LOGSTEP = ln(6.4)/27` exactly** — the Slaney mel scale constant (≈0.068751775); `hz_to_mel_slaney` and `mel_to_hz_slaney` must both use it or filterbank centre frequencies drift and parity fails.
2. **f64 throughout the mel pipeline** (Hann window, STFT, filterbank matrix, mel matmul), casting to f32 only at the final output — f32 in the filterbank matmul accumulates rounding error past 1e-4 tolerance, especially near-floor.
3. **Bilinear resize in float space** — convert u8→f32/255 *before* resizing (matches PyTorch's `F.interpolate`); rounding back to u8 first causes quantisation errors on high-frequency patterns (checkerboard) exceeding 1e-4.
4. **Near-floor STFT values are FFT-implementation-dependent** (numpy pocketfft vs. rustfft differ by a few ULPs at ~-90dB sidelobes) — after the f64 rewrite (point 2), max error stays within tolerance.

## Consequences

- `somnus-encoding` becomes a workspace member alongside `somnus-sensory`/`somnus`; binary gains two `spawn_blocking` sites (audio chunks, video frames).
- `mel.rs`/`resample.rs` are the crate's most complex modules — the ORT smoke test confirms integration is solid but doesn't validate mel correctness.
- **Parity fixture tests required, gating for Phase 2 completion:** Python-venv-dumped reference log-mel arrays (zeros, 440Hz sine, a real clip) in `tests/fixtures/`, Rust output within 1e-4 absolute tolerance; same approach for DINOv2 image preprocessing — silent embedding drift isn't detectable at the ORT level alone.
- Invalidated by: swapping any encoder model (preprocessing params change), moving to GPU execution providers (`ort` feature set + `init_from` change), or extracting `Timestamp` into a shared crate.

## Alternatives considered

- **Two distinct audio embedding types** (`SpeechEmbedding`/`EnvEmbedding`). Rejected — a single type with a source discriminant keeps the channel topology simpler and the discriminant is cheap.
- **Async API in `somnus-encoding`.** Rejected — inference is CPU-bound; `spawn_blocking` from the binary is the right split, keeps the library free of executor deps.
- **`ORT_DYLIB_PATH` env var instead of `init_from`.** Both work; `init_from` in code is more explicit.
- **`libsamplerate` (C FFI) instead of `rubato`.** `rubato` is pure Rust, proven for audio, avoids the FFI dependency.
