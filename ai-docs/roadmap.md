# Roadmap

> **Status:** coarse phasing. Use as orientation, not a commitment. Update when reality diverges.

The phases below are approximate and overlap. We do not need to finish a phase before touching the next; the listing imposes priority, not strict order. Each phase has a single concrete *exit criterion* — when we can answer that question, we have made meaningful progress.

## Phase 0 — Scaffolding (current)

- Project structure, conventions, and behavioral instructions in place.
- Initial tech-stack survey written.
- Architectural sketch (revisable).
- No production code yet.

**Exit criterion:** a new collaborator can answer "what is this project, and how do we work on it?" by reading `ai-docs/`.

## Phase 1 — Sensory I/O

Get raw audio and video flowing through Rust, end-to-end, on this machine.

- Microphone → audio frame stream.
- Webcam → video frame stream.
- A binary that runs both for an extended session without leaks or drift.
- Basic preprocessing: resampling for audio, downsampling/cropping for video.
- Telemetry: timestamps, frame rates, drop counts.

**Exit criterion:** a stable, observable real-time sensory pipeline.

## Phase 2 — Toy perception + working memory

A small system that does *something* with the sensory streams.

- A self-supervised feature extractor per modality (likely starting with pretrained encoders via `ort`).
- A simple working-memory buffer (bounded ring of recent percepts, with attention/gating).
- A first crude form of multimodal binding.

**Exit criterion:** the system can answer "what just happened?" over a short window of recent input.

## Phase 3 — Episodic memory + first sleep cycle

- Episodic store: write percepts with timestamps and context.
- A first sleep cycle: pause sensory ingestion, replay episodes through a learner, write a consolidated representation.
- The first non-trivial wake/sleep transition.

**Exit criterion:** replay during sleep changes the system's behavior on wake. Measurable, however toy.

## Phase 4 — Semantic memory + continuous learning

- A second-tier memory that accumulates abstractions across episodes.
- Continuous online updates during wake; consolidation during sleep.
- A concrete catastrophic-forgetting experiment with a metric.

**Exit criterion:** we can articulate a forgetting trade-off curve for our system.

## Phase 5+ — TBD

Placeholders, not commitments. Each gets researched in `research/` before being scoped:

- Neuromodulation as a global learning-rate / attention gate.
- World model and predictive errors as a first-class signal.
- Multiple sleep stages (NREM-like, REM-like) with distinct dynamics.
- Action loop, if and when we add one.
- Language as an additional modality, if we add one.
- Embodiment beyond audio/video (motor, proprioception).
