---
status: accepted
date: 2026-04-28
deferred:
  - id: enc-whisper-upgrade
    what: Upgrade Whisper-tiny to Whisper-base (512-dim) or WavLM-base if audio-speech concept quality is insufficient
    when: When Phase 3 concept learning shows clear degradation attributable to speech encoder quality
  - id: enc-sliding-window
    what: Migrate Whisper to overlapping sliding window or WavLM-base to reduce 30s audio-speech chunk latency
    when: If real-time dialogue or sub-30s concept binding is required
  - id: enc-siglip-visual
    what: Swap DINOv2-S for SigLIP-B when symbolic-region grounding requires language-aligned visual embeddings
    when: When symbolic region grounding work begins in earnest (Phase 7+)
---

# 0003. Phase 2 encoder selection: Whisper-tiny + CLAP + DINOv2-S/14

- **Status:** accepted
- **Date:** 2026-04-28

> **Decision:** Whisper-tiny (speech, 384-dim) + CLAP-HTSAT-unfused (env, 512-dim) + DINOv2-S/14 (visual, 384-dim) as frozen pretrained encoders, all run via ORT.  
> **Keywords:** Whisper, CLAP, DINOv2, frozen encoder, 384-dim, 512-dim, ORT, speech, environment, visual

## Context

Phase 2 wires frozen pretrained encoders between `somnus-sensory` and the regional concept areas. Three streams required: audio-speech (phoneme/word-level), audio-env (environmental-sound), visual (patch-level).

Resolves four open questions from `architecture.md`/`research/encoders.md`: (1) one multi-task audio encoder vs. two specialized? (2) which models, at which sizes? (3) where do ONNX artifacts live? (4) how does Whisper's 30s chunk window interact with a continuous stream?

**Constraints:** VRAM budget 14GB usable (16GB minus ~2GB desktop overhead), encoder VRAM must leave ≥12GB for later-phase memory/cognitive layers. Latency: each encoder completes within one frame period of its modality at real-time rates. ONNX+`ort` for all encoders (decouples inference from Python). Frozen only — no Phase 2 fine-tuning, weights pinned to a fixed revision.

## Decision

### Encoders

| Stream       | Model                    | HF repo                    | HF revision (SHA)                        | FP16 size | Output dim |
|--------------|--------------------------|----------------------------|------------------------------------------|-----------|------------|
| Audio-speech | Whisper-tiny (enc. only) | `openai/whisper-tiny`      | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | ~75 MB    | 384        |
| Audio-env    | CLAP-HTSAT-unfused       | `laion/clap-htsat-unfused` | `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` | ~155 MB   | 512        |
| Visual       | DINOv2-ViT-S/14          | `facebook/dinov2-small`    | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | ~42 MB    | 384        |

Total frozen encoder VRAM (FP16): ~272 MB. Forward-pass activation buffers add
roughly 2–3× model size per encoder; peak simultaneous activation ≤ ~1 GB, well
within budget.

### Why two audio encoders, not one

Whisper-tiny gives sequential frame-level features tied to spoken content (what audio-speech needs); CLAP gives a multi-label continuous embedding over the whole clip (what audio-env needs) — running both lets the signals coexist without one collapsing the other ("engine overtakes voice"). CLAP alone doesn't produce phoneme/word-aligned representations Whisper's encoder does. Cost of both: ~230MB VRAM, ~30-70ms combined forward pass — affordable.

### Why Whisper-tiny, not base or small

384-dim output matches DINOv2-S/14 — audio-speech and visual can compose at the binding area without a projection head (base=512-dim, small=768-dim would need one). Phase 2's goal is "embeddings flowing in real time," not optimized speech understanding — tiny's phoneme quality is sufficient; upgrade to base if downstream concept learning shows clear degradation.

### Why DINOv2-S/14, not SigLIP-B

DINOv2 gives strong general-purpose spatial features; SigLIP-B's language-alignment advantage matters for symbolic-region grounding, which isn't Phase 2's focus. DINOv2-S/14 is also the smallest option (42MB FP16), keeping VRAM clear for later phases.

### Whisper 30-second chunk strategy

Whisper's encoder needs a fixed 30s mel spectrogram — **Phase 2: non-overlapping 30s chunks, zero-padded at stream end.** `somnus-sensory`'s AudioFrames accumulate to 480,000 samples @ 16kHz, resampled if needed, converted to an 80×3000 mel spectrogram, forwarded through the encoder → 1500×384 frame-level tokens, mean-pooled to one 384-dim vector/chunk for the Phase 2 binding area (per-token routing deferred to Phase 3). Introduces up to 30s latency — acceptable since the goal is end-to-end flow, not real-time dialogue; migrate to WavLM-base or a sliding window if Phase 3 needs lower latency. CLAP and DINOv2 have no fixed-chunk constraint, operate at their natural (per-clip, per-frame) rates.

### ONNX artifact storage

Model weights **not committed** — the trio totals ~272MB FP16 (git history of updates would compound fast) and the Codeberg repo has no git-lfs equivalent configured.

Convention:

```
models/                   # .gitignored; populated by scripts/export_models.py
  whisper_tiny_encoder.onnx
  clap_audio_encoder.onnx
  dinov2_small.onnx
scripts/
  export_models.py        # committed; exports ONNX from HF revisions pinned above
```

`scripts/export_models.py` pins the exact HF revisions from the table above and uses
`optimum` + `transformers` + `torch` to export encoder-only ONNX graphs. Running this
script is the first step of Phase 2 setup. It is idempotent: re-running skips files
already present.

A `models/README.md` (committed) records the expected file names, sizes (MB), and
SHA-256 checksums of the canonical exports so anyone can verify their artifacts.

### New crate: `somnus-encoding`

Handles: loading ONNX models at startup via `ort`; `AudioEmbedding`/`VideoEmbedding` types; a sync encoding API matching `somnus-sensory`'s pull/push style. Binary wires `somnus-sensory`→`somnus-encoding` via `spawn_blocking`, same pattern as Phase 1. Pure library crate, no async/tokio.

## Alternatives considered

- **Single audio encoder (CLAP only).** Rejected — not strong at phoneme/word-level speech; the 512-dim/384-dim mismatch with DINOv2-S would also need a projection.
- **WavLM-base + CLAP.** WavLM (95M params, ~190MB FP16) gives richer phoneme features than Whisper-tiny (39M, ~75MB) but outputs 768-dim/~50fps, needing a projection to align with DINOv2-S. Deferred as the natural upgrade if Phase 3 audio-speech quality proves insufficient.
- **SigLIP-B/16 instead of DINOv2-S/14.** Language alignment (768-dim shared text-visual space) matters for the symbolic region, not visual alone; would need projection heads (audio 384-dim vs. visual 768-dim). Deferred until the symbolic region is in scope.
- **MobileViT-S as visual encoder.** Lightest (5.6M, ~11MB FP16), noticeably lower quality than DINOv2-S — acceptable fallback if DINOv2-S's ONNX export fails, not the target.

## Consequences

- Embedding dims: audio-speech 384, audio-env 512, visual 384 — binding area handles a 1280-dim concatenated input in Phase 2, or projects to a common dim (Phase 3 decision).
- Phase 2 setup needs a Python env (`torch`/`transformers`/`optimum[onnxruntime]`) for `scripts/export_models.py`, plus system OnnxRuntime.
- Audio-speech latency is 30s/chunk — acceptable in Phase 2, review once Phase 3 consumes embeddings in real time.
- CLAP fed the same 30s windows as Whisper for Phase 2 (simplest shared chunking); finer-grained CLAP inference is a Phase 3 refinement.

## What would invalidate this decision

- Whisper-tiny phoneme quality proving insufficient for Phase 3 concept learning → upgrade to Whisper-base or WavLM-base (larger projection either way).
- 30s audio latency becoming a hard blocker → migrate to WavLM-base (~50fps, no chunk constraint).
- VRAM budget shrinking significantly (Phase 4 memory/cognitive layers exhausting 12GB) → downgrade DINOv2-S to MobileViT-S, or drop CLAP for WavLM-only audio-env.
- Symbolic-region grounding needing language-aligned visual embeddings (Phase 4+) → swap DINOv2-S for SigLIP-B or add a frozen CLIP text encoder.
- `ort` ONNX export breaking/going unmaintained for any of the three → fall back to `candle` direct-from-HF-weights, or a Python sidecar via gRPC.
