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

Phase 2 wires frozen pretrained encoders between the sensory pipeline (`somnus-sensory`)
and the regional concept areas. Three encoder streams are required by the architecture:

- **Audio-speech:** phoneme/word-level representations for the audio-speech region.
- **Audio-env:** environmental-sound representations for the audio-env region.
- **Visual:** patch-level visual representations for the visual region.

The ADR resolves four open questions from `architecture.md` and `research/encoders.md`:

1. One multi-task audio encoder (CLAP alone) or two specialized encoders?
2. Which specific models, at which sizes?
3. Where do ONNX artifacts live in the repo?
4. How does Whisper's 30-second chunk window interact with a continuous audio stream?

Constraints:

- **VRAM budget:** 14 GB usable (16 GB minus ~2 GB desktop overhead). Encoder VRAM
  must leave ≥ 12 GB for episodic memory and cognitive layers in later phases.
- **Latency target:** each encoder completes its forward pass within one frame period
  of its respective modality at real-time rates.
- **ONNX + `ort`:** all encoders are run via ONNX Runtime (`ort` crate) so inference
  is decoupled from Python and can run in a Rust process.
- **Frozen only:** no fine-tuning in Phase 2. Weights are loaded once at startup,
  pinned to a fixed revision for reproducibility.

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

`architecture.md` notes that two specialized encoders compose better than one
multi-task encoder:

- Whisper-tiny produces **sequential frame-level features** tied to spoken content —
  the representation the audio-speech region needs.
- CLAP produces a **multi-label continuous embedding** over the whole audio clip —
  the representation the audio-env region needs.
- Running both in parallel lets both signals coexist without one collapsing the other
  (the "engine overtakes voice" failure mode).

CLAP alone, even though it covers some speech, does not produce the kind of phoneme-
and word-aligned representations Whisper's encoder provides. The cost of running both
is ~230 MB VRAM and ~30–70 ms combined forward-pass time, which is affordable.

### Why Whisper-tiny, not base or small

Whisper-tiny's 384-dim output matches DINOv2-S/14's 384-dim. At the binding area,
audio-speech and visual embeddings can be concatenated or otherwise composed without
a projection head. Whisper-base (512-dim) and Whisper-small (768-dim) would require
a projection.

Whisper-tiny's phoneme-level quality is sufficient for Phase 2, whose goal is "embeddings
flowing in real time", not optimized speech understanding. Upgrade to Whisper-base if
downstream concept learning shows clear quality degradation.

### Why DINOv2-S/14, not SigLIP-B

DINOv2 produces strong general-purpose spatial features; SigLIP-B's advantage (language-
aligned embeddings) becomes relevant when symbolic-region grounding requires text-visual
similarity. The symbolic region is not the focus of Phase 2. DINOv2-S/14 at 42 MB FP16
is also the smallest choice, keeping VRAM clear for later phases.

### Whisper 30-second chunk strategy

Whisper's encoder operates over a fixed 30-second mel spectrogram. For a continuous audio
stream this requires buffering:

**Phase 2 strategy: non-overlapping 30-second chunks, zero-padded at end of stream.**

- The audio stream from `somnus-sensory` accumulates `AudioFrame`s until 30 seconds of
  samples are buffered (at 16 kHz, the rate Whisper expects: 480 000 samples).
- The accumulated buffer is resampled to 16 kHz if the source rate differs, converted
  to a mel spectrogram (80 bins × 3000 frames), then forwarded through the encoder.
- The encoder outputs 1500 × 384 frame-level tokens per chunk.
- For the binding area in Phase 2, the 1500 tokens are mean-pooled to a single 384-dim
  vector per chunk. Finer temporal resolution (per-token routing) is deferred to Phase 3.
- This introduces up to 30 seconds of latency for the audio-speech stream. This is
  acceptable for Phase 2, where the goal is end-to-end embedding flow, not real-time
  dialogue. If Phase 3 concept learning needs lower latency, migrate to WavLM-base or
  add a sliding-window with overlap.

CLAP and DINOv2 have no fixed-chunk constraint and operate at their natural rates
(per-clip and per-frame respectively).

### ONNX artifact storage

Model weights are **not committed to the repository**. Reasons:
- Whisper-tiny + CLAP + DINOv2-S/14 total ~272 MB FP16; git history of model updates
  would compound that cost quickly.
- SHA-256 repo on Codeberg has no git-lfs equivalent configured.

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

A `crates/somnus-encoding` library crate handles:

- Loading ONNX models at startup via `ort`.
- Defining `AudioEmbedding` and `VideoEmbedding` frame types.
- Exposing a sync encoding API matching the pull/push style of `somnus-sensory`.

The binary `somnus` wires `somnus-sensory` → `somnus-encoding` via `spawn_blocking` bridges,
matching the pattern established in Phase 1.

`somnus-encoding` is a pure library crate; no async, no tokio dependency.

## Alternatives considered

**Single audio encoder (CLAP only).** Simpler — one model, 512-dim for all audio.
Rejected: CLAP is not strong at phoneme/word-level speech representations. Concepts
grounded in spoken words would be underserved. The 512-dim mismatch with DINOv2-S's
384-dim also requires a projection at the binding area.

**WavLM-base + CLAP** (the "best speech quality" pairing from `encoders.md`).
WavLM-base is 95M params vs. Whisper-tiny's 39M; it produces richer phoneme features
but at ~190 MB FP16 vs. ~75 MB. WavLM also outputs ~50 frames/s (768-dim), not 384-dim,
requiring a projection to align with DINOv2-S. Deferred as the natural upgrade path if
Phase 3 concept quality for audio-speech proves insufficient.

**SigLIP-B/16 instead of DINOv2-S/14.** SigLIP provides language alignment (768-dim
shared text-visual space) which is valuable for the symbolic region but not for the
visual region alone. SigLIP-B at ~174 MB FP16 and 768-dim would require projection
heads at the binding area (DINOv2 audio: 384-dim, SigLIP visual: 768-dim). Deferred
until the symbolic region is in scope.

**MobileViT-S as visual encoder.** Lightest option (5.6M params, ~11 MB FP16), good for
prototyping. Feature quality is noticeably lower than DINOv2-S on dense vision tasks.
Acceptable as a temporary stand-in if ONNX export of DINOv2-S proves difficult, but not
the intended target.

## Consequences

- **Embedding dimensions in the system:** audio-speech 384, audio-env 512, visual 384.
  The binding area must handle a 384+512+384 = 1280-dim concatenated input in Phase 2,
  or project all streams to a common dim. This is a Phase 3 decision; Phase 2 stops at
  emitting the individual embeddings.
- **Phase 2 setup prerequisite:** Python environment with `torch`, `transformers`,
  `optimum[onnxruntime]` to run `scripts/export_models.py`. OnnxRuntime C++ shared
  library installed system-wide (Arch: `onnxruntime` from community repo or AUR).
- **Audio-speech latency is 30s per chunk.** Acceptable in Phase 2. Review when Phase 3
  starts consuming embeddings in real time.
- **CLAP input rate:** CLAP expects variable-length audio clips; Phase 2 will feed it
  the same 30s windows used for Whisper (simplest shared chunking). Finer-grained CLAP
  inference (e.g., per-second) is a Phase 3 refinement.

## What would invalidate this decision

- **Whisper-tiny phoneme quality proves insufficient for concept learning in Phase 3.**
  Upgrade to Whisper-base (512-dim, requires binding-area projection) or WavLM-base
  (768-dim, larger projection).
- **30-second audio latency becomes a hard blocker** for a Phase 3 use case. Migrate the
  speech encoder to WavLM-base, which operates at ~50 frames/s with no chunk constraint.
- **VRAM budget shrinks significantly** (e.g., large episodic memory + cognitive layer
  in Phase 4 exhausts 12 GB). Downgrade DINOv2-S to MobileViT-S (42 MB → 11 MB)
  or drop CLAP in favour of single audio-env features from WavLM.
- **Symbolic region grounding requires language-aligned visual embeddings** (Phase 4+).
  Swap DINOv2-S for SigLIP-B or add a frozen CLIP text encoder alongside.
- **`ort` ONNX export for any of the three models proves broken or unmaintained.**
  Fall back to `candle` inference from HF weights directly, or to a Python sidecar via
  gRPC.
