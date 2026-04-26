# Encoders

Research note for somnus. Covers the frozen pretrained encoder stack: audio (speech-focused + environment-focused) and visual. All numbers are **engineering estimates** from model architecture specs and published benchmarks; exact figures depend on driver/kernel versions. Refine to Option A (cited measurements) before making final encoder commitments.

Hardware target: 14 GB VRAM (16 GB minus ~2 GB desktop overhead), 56 GB usable RAM.

See `architecture.md` "Frontend: encoders and bypass" for the design commitment. Encoder choice becomes the first ADR.

---

## Why encoders anchor the architecture

Encoder output dimensions set embedding sizes for every downstream component:

- Regional concept areas store vectors of encoder output dimension.
- The binding area composes encoder outputs — if they differ in size, a projection step is needed.
- The HNSW index in the recollection layer is built over these vectors.
- Memory tier capacity scales linearly with embedding dimension.

Choosing a smaller encoder is not just a latency decision — it multiplies through memory cost across the entire hierarchy.

---

## Audio encoders

somnus needs two types of audio feature stream (see `architecture.md` "Output discipline: multi-source"):

- **Speech-recognition features** — frame-level or token-level representations tied to spoken content.
- **Environmental-sound features** — multi-label embeddings for ambient events (vehicle noise, footsteps, etc.).

These can come from one multi-task encoder or two specialized encoders.

### Whisper (speech)

OpenAI Whisper is an encoder-decoder trained on 680k hours of supervised audio-text data. The encoder alone provides high-quality speech features without running the decoder.

| Variant        | Params | FP16 size | Enc hidden dim | Enc layers | Enc output shape          |
|----------------|--------|-----------|----------------|------------|---------------------------|
| tiny / tiny.en | 39M    | ~75 MB    | 384            | 4          | 1500 × 384 per 30s chunk  |
| base / base.en | 74M    | ~148 MB   | 512            | 6          | 1500 × 512 per 30s chunk  |
| small / small.en | 244M | ~488 MB   | 768            | 12         | 1500 × 768 per 30s chunk  |

**Inference latency (encoder-only forward pass):**
- tiny: ~10–25 ms per 30 s chunk on consumer GPU (est.)
- base: ~20–45 ms
- small: ~60–120 ms

Full transcription (encoder + autoregressive decoder) is 10–20× slower. somnus uses encoder-only for features.

**Tradeoffs:**
- Best in class for phoneme-level and word-level speech representations.
- Operates on 30-second fixed chunks — introduces ~30 s of latency if used naively. Mitigated by streaming with overlapping windows (at added compute cost).
- *Not* designed for environmental sound; performance on non-speech audio is poor.
- ONNX export is straightforward via `transformers`; runs under `ort` without issues.

### HuBERT / wav2vec 2.0 (speech features)

Self-supervised speech representation models. HuBERT generally produces better phoneme-aligned representations than wav2vec2 at the same model size.

| Model             | Params | FP16 size | Hidden dim | Output shape               |
|-------------------|--------|-----------|------------|----------------------------|
| wav2vec2-base     | 95M    | ~190 MB   | 768        | ~50 frames/s × 768         |
| wav2vec2-large    | 317M   | ~634 MB   | 1024       | ~50 frames/s × 1024        |
| HuBERT-base       | 95M    | ~190 MB   | 768        | ~50 frames/s × 768         |
| HuBERT-large      | 317M   | ~634 MB   | 1024       | ~50 frames/s × 1024        |
| WavLM-base        | 95M    | ~190 MB   | 768        | ~50 frames/s × 768         |

**Inference latency (base variants, GPU):**
- ~30–80 ms per 10 s audio on GPU (est.)
- Frame-level outputs: more granular than Whisper's 30s-chunk pooling.

**Tradeoffs vs. Whisper:**
- Continuous streaming-friendly (no fixed chunk window).
- Richer prosody and phoneme features — better for learning word-level concepts in context.
- Not trained for environmental sound.
- No built-in text supervision — requires a downstream head if word identity is needed.
- WavLM adds masked speech prediction with utterance mixing; slightly stronger on downstream tasks.

### CLAP (environmental + some speech)

LAION-CLAP: dual audio-text encoder trained contrastively on a large audio-caption dataset. Bridges environmental sound and natural language descriptions.

| Model               | Params (audio enc) | FP16 size | Output dim |
|---------------------|--------------------|-----------|------------|
| LAION-CLAP (HTSAT)  | ~75–86M            | ~155 MB   | 512        |
| Microsoft CLAP      | ~86M               | ~172 MB   | 512        |

Audio encoder uses HTSAT (Hierarchical Token-Semantic Audio Transformer, a Swin Transformer variant over mel spectrograms).

**Inference latency:**
- ~20–50 ms per 10 s clip on GPU (est.)
- Operates over variable-length audio; no fixed chunk constraint.

**Tradeoffs:**
- Strong on environmental sound classification and retrieval.
- Shared 512-dim embedding space with text — enables text-to-audio similarity directly, useful for symbolic-region grounding.
- Speech recognition quality is lower than Whisper or HuBERT — CLAP was not trained primarily for speech features.
- HTSAT architecture is less common in Rust/ONNX toolchains; exporting to ONNX requires care.

### Audio encoder summary and recommended pairing

Running two specialized encoders is preferred (see `architecture.md` "Output discipline"):

| Pairing                  | Total params | VRAM (FP16) | Speech quality | Env quality | Streaming |
|--------------------------|--------------|-------------|----------------|-------------|-----------|
| Whisper-tiny + CLAP      | ~114M        | ~230 MB     | Good           | Good        | Chunked   |
| HuBERT-base + CLAP       | ~170M        | ~345 MB     | Better         | Good        | Yes       |
| WavLM-base + CLAP        | ~170M        | ~345 MB     | Best           | Good        | Yes       |

**Phase 1 recommendation (estimate):** Whisper-tiny (encoder-only) + CLAP-audio. Smallest VRAM footprint, proven ONNX paths, covers both streams. Upgrade speech encoder to WavLM-base if phoneme quality matters for downstream concept learning.

---

## Visual encoders

somnus needs a frozen visual encoder emitting continuous frame embeddings for the visual regional concept area.

### DINOv2

Meta's self-supervised ViT family. Currently best-in-class for dense visual features and semantic patch representations without supervision.

| Variant     | Params | FP16 size | Output dim | Patch | Notes                  |
|-------------|--------|-----------|------------|-------|------------------------|
| ViT-S/14    | 21M    | ~42 MB    | 384        | 14px  | Fastest, good features |
| ViT-B/14    | 86M    | ~172 MB   | 768        | 14px  | Better spatial detail  |
| ViT-L/14    | 307M   | ~614 MB   | 1024       | 14px  | High quality, heavy    |
| ViT-g/14    | 1.1B   | ~2.2 GB   | 1536       | 14px  | Research use only      |

**Inference latency (224×224 input, GPU):**
- ViT-S/14: ~4–8 ms/frame (est.)
- ViT-B/14: ~12–18 ms/frame (est.)
- ViT-L/14: ~35–55 ms/frame (est.)

**Tradeoffs:**
- Dense patch tokens available — can retrieve spatial feature maps, not just the [CLS] pooled embedding. Useful for object-level binding.
- No language alignment — embeddings are not text-comparable without an additional head.
- ONNX export available; runs under `ort`.

### SigLIP / SigLIP 2

Google's CLIP variant using sigmoid loss. Strong language-vision alignment; becoming the encoder of choice for multimodal LLMs (Gemma 3, Qwen3-VL).

| Variant       | Params (vision enc) | FP16 size | Output dim | Notes                         |
|---------------|---------------------|-----------|------------|-------------------------------|
| SigLIP-B/16   | ~87M                | ~174 MB   | 768        | Good text-aligned features    |
| SigLIP-So400M | ~400M               | ~800 MB   | 1152       | Very strong, large            |
| SigLIP 2-B    | ~93M                | ~186 MB   | 768        | Multilingual, dense features  |

**Tradeoffs:**
- Better language alignment than DINOv2 — text-grounded visual concepts land near text embeddings. Useful for the symbolic region.
- SigLIP 2 adds dense (patch-level) features; better spatial resolution than original SigLIP.
- Slightly behind DINOv2 on pure vision tasks (e.g., semantic segmentation).
- If symbolic region uses CLAP's 512-dim text embeddings and visual region uses SigLIP-B's 768-dim, a projection is needed for cross-modal binding — or both encoders must be chosen to share a projection space.

### MobileViT / EfficientViT

Lightweight models for minimal VRAM footprint.

| Model          | Params | FP16 size | Output dim | Notes                           |
|----------------|--------|-----------|------------|---------------------------------|
| MobileViT-S    | 5.6M   | ~11 MB    | 640        | Very fast, mobile-optimized     |
| MobileViT-XS   | 2.3M   | ~5 MB     | 384        | Tiny                            |
| EfficientViT-M5| 12.5M  | ~25 MB    | 512        | Efficient, modern               |

**Tradeoffs:**
- Dramatically smaller VRAM footprint — relevant if VRAM is tight.
- Noticeably lower feature quality than DINOv2 or SigLIP on downstream tasks.
- Good for Phase 1 where we are testing the pipeline, not yet optimizing feature quality.

### Visual encoder summary

| Encoder       | Params | FP16 size | Dim  | Best for                              |
|---------------|--------|-----------|------|---------------------------------------|
| DINOv2-S/14   | 21M    | ~42 MB    | 384  | Dense spatial features, low VRAM      |
| DINOv2-B/14   | 86M    | ~172 MB   | 768  | Better features, still affordable     |
| SigLIP-B/16   | 87M    | ~174 MB   | 768  | Language-aligned, text-visual binding |
| MobileViT-S   | 5.6M   | ~11 MB    | 640  | Prototyping, minimal VRAM             |

**Phase 1 recommendation:** DINOv2-S/14 (42 MB FP16, 384-dim). Matches Whisper-tiny's embedding size (384-dim) — avoids a projection head at the binding area. Upgrade to DINOv2-B or SigLIP-B if alignment with the symbolic region requires language grounding.

---

## Text bypass encoder

The text bypass encoder must produce embeddings in the same representational space as the speech-recognition output (see `architecture.md` "Text bypass"). Options:

- **CLIP text encoder / SigLIP text encoder:** project text into a shared text-vision space. If visual encoder is SigLIP, the text encoder is a natural match. Params: ~63–93M, ~126–186 MB FP16, 512–768-dim.
- **Sentence-transformers (all-MiniLM-L6):** 22M params, ~44 MB FP16, 384-dim. Very fast, general-purpose text embeddings. Not aligned to audio space without fine-tuning.
- **Fine-tuned joint encoder:** trained to align text embeddings with HuBERT/WavLM speech representations. Does not exist off-the-shelf; requires a research step.

The fundamental tension: audio speech encoders (HuBERT, Whisper) and text encoders do not natively share embedding spaces. CLAP partially bridges this (512-dim joint audio-text space). If text bypass must map to the same space as the speech encoder, either:
1. Use CLAP as the speech encoder (gives a shared 512-dim space), or
2. Train a small projection head.

---

## Budget analysis

All encoders in VRAM simultaneously (wake mode):

| Scenario         | Encoders                              | Total VRAM (FP16)  | VRAM remaining (of 14 GB) |
|------------------|---------------------------------------|--------------------|-----------------------------|
| Minimal          | Whisper-tiny + CLAP + DINOv2-S        | ~267 MB            | ~13.7 GB                    |
| Moderate         | WavLM-base + CLAP + DINOv2-B          | ~517 MB            | ~13.5 GB                    |
| Rich             | WavLM-base + CLAP + SigLIP-B + CLIP-text | ~700 MB        | ~13.3 GB                    |

Encoder VRAM is not the bottleneck. Forward-pass activation buffers (roughly 2–4× model size for a single frame) add another ~200–700 MB in the worst case, still leaving >12 GB for episodic memory and cognitive layers.

**Key constraint is latency, not memory.** Running multiple encoders in parallel at video frame rates (25 fps) and audio chunk rates requires:
- Visual: ~8 ms/frame for DINOv2-S = ~125 fps theoretical max → not the bottleneck at 25 fps.
- Audio: Whisper operates on 30s chunks, not per-frame. CLAP at ~30 ms/10s clip → real-time at continuous audio.

---

## Open questions → future ADR

1. One multi-task audio encoder (CLAP alone) or two specialized? Depends on how important rich phoneme features are for the symbolic region in Phase 1.
2. DINOv2-S (384-dim, matches Whisper-tiny) vs. SigLIP-B (768-dim, better language alignment)? If text bypass matters in early phases, SigLIP-B is better.
3. How to handle text bypass's representational alignment with the speech encoder — shared CLAP space, projection head, or deferred?
4. Streaming vs. chunked audio processing — Whisper's 30s window may introduce unacceptable latency; evaluate WavLM for streaming.

---

## Sources

- Radford et al., 2022. "Robust Speech Recognition via Large-Scale Weak Supervision." OpenAI Whisper. arXiv:2212.04356.
- Hsu et al., 2021. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." arXiv:2106.07447.
- Chen et al., 2022. "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing." arXiv:2110.13900.
- Wu et al., 2023. "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation." LAION-CLAP. arXiv:2211.06687.
- Oquab et al., 2023. "DINOv2: Learning Robust Visual Features without Supervision." arXiv:2304.07193.
- Zhai et al., 2023. "Sigmoid Loss for Language Image Pre-Training." SigLIP. arXiv:2303.15343.
- Tschannen et al., 2025. "SigLIP 2." arXiv:2502.14786.
- Mehta & Rastegari, 2021. "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer." arXiv:2110.02178.
- OpenAI Whisper GPU benchmarks: github.com/openai/whisper/discussions/530, /918.
