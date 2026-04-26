# Roadmap

> **Status:** coarse phasing. Use as orientation, not a commitment. Update when reality diverges.

The phases below are approximate and overlap. We do not need to finish a phase before touching the next; the listing imposes priority, not strict order. Each phase has a single concrete *exit criterion* — when we can answer that question, we have made meaningful progress.

## Phase 0 — Scaffolding ✓

- Project structure, conventions, and behavioral instructions in place.
- Tech-stack survey, architectural sketch, research notes, ADRs for Phase 1.
- No production code yet.

**Exit criterion:** ✓ a new collaborator can answer "what is this project, and how do we work on it?" by reading `ai-docs/`.

## Phase 1 — Sensory capture

Get raw audio and video flowing through Rust, end-to-end. Defined in ADRs 0001 and 0002.

- **Input sources:**
  - Microphone → `AudioFrame` stream (cpal, crossbeam callback bridge).
  - Webcam → `VideoFrame` stream (nokhwa, pull model).
  - Media file → both streams (ffmpeg-next `MediaFile`; primary test path since no camera is available).
  - Mixed: mic + file audio fan-in via `Sender::clone` (e.g., movie + person speaking).
- **Frame types:** `AudioFrame` (f32 PCM, interleaved channels), `VideoFrame` (packed pixels), dual `Timestamp` (monotonic + wall-clock).
- **Backpressure:** bounded crossbeam channels, drop-oldest on overflow, per-source drop counter.
- **Binary:** `--file <path>` / `--mic` / `--webcam <idx>` flags; tokio async main loop; periodic metrics log; Ctrl+C graceful shutdown.
- **No encoding in this phase.** Frames are consumed by the logging loop only.

**Exit criterion:** a 10-minute run on a real video file completes without OOM, measurable frame-rate drift, or process leak. Audio and video drop rates logged.

## Phase 2 — Encoding (frozen pretrained models)

Frames → continuous embeddings via frozen pretrained encoders.

- Load and run frozen ONNX models via `ort`:
  - **Audio-speech:** Whisper-tiny encoder (39 M params, 384-dim, ~10–25 ms per 30 s chunk).
  - **Audio-env:** CLAP audio encoder (~86 M, 512-dim, parallel to speech encoder).
  - **Visual:** DINOv2-S/14 (~21 M, 384-dim, ~4–8 ms/frame).
- `AudioEmbedding` and `VideoEmbedding` types in `somnus-encoding`.
- VRAM budget check: all three encoders + overhead ≤ 2 GB, leaving ≥ 12 GB for cognitive layers and episodic memory.
- ADR for final encoder selection (one multi-task audio encoder vs. two specialized).

**Exit criterion:** embeddings flowing for all three encoder streams in real time; latency ≤ the frame period for each modality.

## Phase 3 — Perception (sparse coding + regional concept areas)

Embeddings → sparse activations per region; first pass of multimodal binding.

- Regional concept areas (audio-speech, audio-env, visual, symbolic) implemented as k-WTA layers in `candle`.
- `SparseActivation { indices: Vec<u16>, values: Vec<f32>, region: RegionId }`.
- Simple binding area: concatenation + linear projection (Phase 3), upgradeable to cross-attention (Phase 4).
- SAE reconstruction error wired as the growth-trigger signal and replay-priority signal.
- `somnus-perception` crate created.

**Exit criterion:** sparse activations produced per frame; binding area outputs a fixed-size episode vector; reconstruction error is observable as a metric.

## Phase 4 — Memory hierarchy

Episodic buffer, semantic store, cold archive, and recollection index.

- **Hot (episodic):** bounded VRAM ring of recent episode vectors; `EpisodicBuffer`.
- **Warm (semantic):** region-sharded sparse-coded representations in RAM; `usearch` HNSW index for retrieval; `SemanticStore`.
- **Cold (archive):** INT8-quantized vectors written to disk via `memmap2`; `ColdArchive`.
- Recollection layer: HNSW query returns top-k candidates; cold→warm promotion on cache miss.
- `somnus-memory` crate created.

**Exit criterion:** given a cue (e.g., a replayed audio segment), the system retrieves the top-k most similar episodes from semantic memory.

## Phase 5 — Sleep cycle

First wake/sleep transition; replay-driven consolidation.

- Sleep triggers: episodic buffer high-water mark, reconstruction-error degradation, schedule fallback.
- **NREM-like:** replay episodes from episodic buffer → update semantic memory weights; homeostatic weight downscaling; compress and spill old episodes to cold archive.
- **REM-like:** generative self-probe; cascade-correlation growth where gist recall fails; HNSW index rebuild if tombstone fraction exceeds threshold.
- Adaptive sleep duration: short (≈ 10–20 min GPU) if no growth triggered; long (≈ 1–4 h GPU) if growth runs.
- `somnus-sleep` crate created.

**Exit criterion:** replay during sleep measurably changes wake-time reconstruction error on a held-out set of episodes. Growth triggered and confirmed by unit count increase in at least one region.

## Phase 6+ — TBD

Placeholders, not commitments. Each gets researched in `research/` before being scoped:

- Cross-attention binding (upgrade from concat) and bidirectional region ↔ binding top-down priors.
- Neuromodulation as global gain control and learning-rate gate.
- SAE reconstruction error → full predictive-coding settling loop (Phase 4+ of PC adoption).
- Sample-level audio mixing (mic + file waveform summing with resampling).
- Action loop, if and when we add one.
- Language as an additional modality.
