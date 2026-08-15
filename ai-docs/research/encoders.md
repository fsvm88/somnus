# Encoders

> **Contents:** Frozen pretrained encoder options and their VRAM/latency/dimension tradeoffs — speech (Whisper-tiny 384-dim ✓ committed), environment (CLAP-HTSAT 512-dim ✓ committed), visual (DINOv2-S/14 384-dim ✓ committed). Alternatives compared: HuBERT, SigLIP, MobileViT.
> **When to read:** Considering a replacement encoder; understanding why the committed trio was chosen; checking embedding dimensions for a new region.

Research note covering the frozen pretrained encoder stack: audio (speech + environment) and visual. All numbers are **engineering estimates** from architecture specs and published benchmarks — exact figures depend on driver/kernel versions; refine to cited measurements before final commitments.

Hardware target: 14GB VRAM (16GB minus ~2GB desktop overhead), 56GB usable RAM. See `architecture.md` "Frontend: encoders and bypass" for the design commitment.

---

## Why encoders anchor the architecture

Encoder output dimensions set embedding sizes for every downstream component: regional concept areas store encoder-dim vectors, the binding area needs a projection step if sizes differ, the recollection-layer HNSW index is built over these vectors, and memory-tier capacity scales linearly with dimension. A smaller encoder isn't just a latency decision — it multiplies through memory cost across the whole hierarchy.

---

## Audio encoders

somnus needs two feature streams (`architecture.md` "Output discipline"): speech-recognition features and environmental-sound features — from one multi-task encoder or two specialized ones.

### Whisper (speech)

Encoder-decoder trained on 680k hours of supervised audio-text; encoder alone gives high-quality speech features without the decoder.

| Variant | Params | FP16 size | Enc hidden dim | Enc layers | Enc output shape |
|---|---|---|---|---|---|
| tiny/tiny.en | 39M | ~75MB | 384 | 4 | 1500×384 per 30s chunk |
| base/base.en | 74M | ~148MB | 512 | 6 | 1500×512 per 30s chunk |
| small/small.en | 244M | ~488MB | 768 | 12 | 1500×768 per 30s chunk |

Encoder-only latency (est.): tiny ~10-25ms/30s chunk, base ~20-45ms, small ~60-120ms. Full transcription (decoder included) is 10-20× slower — somnus uses encoder-only. Best-in-class phoneme/word-level speech features; 30s fixed chunks (mitigate via overlapping-window streaming); poor on non-speech audio; ONNX export straightforward via `transformers`.

### HuBERT / wav2vec 2.0 (speech features)

Self-supervised; HuBERT generally beats wav2vec2 at matched size.

| Model | Params | FP16 size | Hidden dim | Output shape |
|---|---|---|---|---|
| wav2vec2-base | 95M | ~190MB | 768 | ~50 frames/s × 768 |
| wav2vec2-large | 317M | ~634MB | 1024 | ~50 frames/s × 1024 |
| HuBERT-base | 95M | ~190MB | 768 | ~50 frames/s × 768 |
| HuBERT-large | 317M | ~634MB | 1024 | ~50 frames/s × 1024 |
| WavLM-base | 95M | ~190MB | 768 | ~50 frames/s × 768 |

Est. ~30-80ms/10s clip on GPU (base variants); frame-level output, more granular than Whisper's chunk pooling. Streaming-friendly (no fixed window), richer prosody/phoneme features, but no environmental-sound training and no built-in text supervision. WavLM adds masked speech prediction with utterance mixing, slightly stronger downstream.

### CLAP (environmental + some speech)

Dual audio-text encoder, contrastive, bridges environmental sound and language.

| Model | Params (audio enc) | FP16 size | Output dim |
|---|---|---|---|
| LAION-CLAP (HTSAT) | ~75-86M | ~155MB | 512 |
| Microsoft CLAP | ~86M | ~172MB | 512 |

HTSAT = Hierarchical Token-Semantic Audio Transformer (Swin variant over mel spectrograms). Est. ~20-50ms/10s clip; variable-length input, no fixed chunk. Strong on env classification/retrieval; shared 512-dim audio-text space (useful for symbolic-region grounding); weaker speech recognition than Whisper/HuBERT; HTSAT less common in Rust/ONNX toolchains, needs care to export.

**Pairing summary:**

| Pairing | Total params | VRAM (FP16) | Speech quality | Env quality | Streaming |
|---|---|---|---|---|---|
| Whisper-tiny + CLAP | ~114M | ~230MB | Good | Good | Chunked |
| HuBERT-base + CLAP | ~170M | ~345MB | Better | Good | Yes |
| WavLM-base + CLAP | ~170M | ~345MB | Best | Good | Yes |

**Phase 1 recommendation (estimate):** Whisper-tiny (encoder-only) + CLAP — smallest VRAM, proven ONNX paths, covers both streams. Upgrade to WavLM-base if phoneme quality matters downstream.

---

## Visual encoders

### DINOv2

Meta's self-supervised ViT family — best-in-class dense visual/semantic patch features without supervision.

| Variant | Params | FP16 size | Output dim | Patch | Notes |
|---|---|---|---|---|---|
| ViT-S/14 | 21M | ~42MB | 384 | 14px | Fastest, good features |
| ViT-B/14 | 86M | ~172MB | 768 | 14px | Better spatial detail |
| ViT-L/14 | 307M | ~614MB | 1024 | 14px | High quality, heavy |
| ViT-g/14 | 1.1B | ~2.2GB | 1536 | 14px | Research use only |

Est. latency (224×224, GPU): S ~4-8ms/frame, B ~12-18ms, L ~35-55ms. Dense patch tokens available (spatial feature maps, not just CLS pooled — useful for object-level binding); no language alignment; ONNX export runs fine under `ort`.

### SigLIP / SigLIP 2

Google's sigmoid-loss CLIP variant, strong language-vision alignment, becoming the encoder of choice for multimodal LLMs (Gemma 3, Qwen3-VL).

| Variant | Params (vision enc) | FP16 size | Output dim | Notes |
|---|---|---|---|---|
| SigLIP-B/16 | ~87M | ~174MB | 768 | Good text-aligned features |
| SigLIP-So400M | ~400M | ~800MB | 1152 | Very strong, large |
| SigLIP 2-B | ~93M | ~186MB | 768 | Multilingual, dense features |

Better language alignment than DINOv2 (text-grounded concepts land near text embeddings — useful for the symbolic region); SigLIP 2 adds dense patch-level features; slightly behind DINOv2 on pure vision tasks (e.g. segmentation). If symbolic region uses CLAP's 512-dim text and visual uses SigLIP-B's 768-dim, cross-modal binding needs a projection unless both share a space.

### MobileViT / EfficientViT

Lightweight, minimal VRAM.

| Model | Params | FP16 size | Output dim | Notes |
|---|---|---|---|---|
| MobileViT-S | 5.6M | ~11MB | 640 | Very fast, mobile-optimized |
| MobileViT-XS | 2.3M | ~5MB | 384 | Tiny |
| EfficientViT-M5 | 12.5M | ~25MB | 512 | Efficient, modern |

Dramatically smaller footprint, noticeably lower quality than DINOv2/SigLIP — fine for pipeline testing, not feature-quality optimization.

**Visual summary:**

| Encoder | Params | FP16 size | Dim | Best for |
|---|---|---|---|---|
| DINOv2-S/14 | 21M | ~42MB | 384 | Dense spatial features, low VRAM |
| DINOv2-B/14 | 86M | ~172MB | 768 | Better features, still affordable |
| SigLIP-B/16 | 87M | ~174MB | 768 | Language-aligned, text-visual binding |
| MobileViT-S | 5.6M | ~11MB | 640 | Prototyping, minimal VRAM |

**Phase 1 recommendation:** DINOv2-S/14 (42MB FP16, 384-dim) — matches Whisper-tiny's dim, avoids a binding-area projection head. Upgrade to DINOv2-B or SigLIP-B if symbolic-region language grounding is needed.

---

## Text bypass encoder

Must land in the same representational space as the speech-recognition output (`architecture.md` "Text bypass"). Options: **CLIP/SigLIP text encoder** (~63-93M, ~126-186MB FP16, 512-768-dim — natural match if visual is SigLIP); **sentence-transformers all-MiniLM-L6** (22M, ~44MB, 384-dim, fast general-purpose, not audio-aligned without fine-tuning); **fine-tuned joint encoder** aligned to HuBERT/WavLM (doesn't exist off-the-shelf, would need a research step).

Fundamental tension: audio speech encoders and text encoders don't natively share embedding spaces. CLAP partially bridges this (512-dim joint space) — either use CLAP as the speech encoder, or train a small projection head.

---

## Budget analysis

All encoders in VRAM simultaneously (wake mode):

| Scenario | Encoders | Total VRAM (FP16) | VRAM remaining (of 14GB) |
|---|---|---|---|
| Minimal | Whisper-tiny + CLAP + DINOv2-S | ~267MB | ~13.7GB |
| Moderate | WavLM-base + CLAP + DINOv2-B | ~517MB | ~13.5GB |
| Rich | WavLM-base + CLAP + SigLIP-B + CLIP-text | ~700MB | ~13.3GB |

Not VRAM-bound — activation buffers add another ~200-700MB worst case, still >12GB free. **Latency is the real constraint:** DINOv2-S at ~8ms/frame → ~125fps theoretical max, not the bottleneck at 25fps; Whisper operates on 30s chunks not per-frame; CLAP ~30ms/10s clip → real-time at continuous audio.

---

## Open questions → future ADR

1. One multi-task audio encoder (CLAP alone) or two specialized? Depends on how much phoneme richness the symbolic region needs in Phase 1.
2. DINOv2-S (384-dim, matches Whisper-tiny) vs. SigLIP-B (768-dim, better language alignment)?
3. Text bypass's representational alignment with the speech encoder — shared CLAP space, projection head, or deferred?
4. Streaming vs. chunked audio — Whisper's 30s window may be unacceptable latency; evaluate WavLM for streaming.

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
