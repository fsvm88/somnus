# Glossary

Single source of truth for terminology. Define terms here; link from elsewhere.

When you introduce a new term in a doc or in code, add it here first — even if just one line.

## Core operating modes

- **Wake.** The real-time operating mode: sensory ingestion, perception, online learning, working memory, control.
- **Sleep.** The offline operating mode: replay, consolidation, abstraction, pruning, homeostatic adjustment. Wall-clock-decoupled.
- **Mode transition.** A controlled switch between wake and sleep. Triggered by neuromodulatory signals, schedules, or both.

## Learning

- **Continuous learning.** Online updating from streaming experience without a separate training phase. Persists across sessions; resists catastrophic forgetting.
- **Catastrophic forgetting.** The tendency of neural systems trained sequentially on different tasks/data to lose earlier learning. A central problem we expect to own.
- **Replay.** Re-presenting (real or generatively reconstructed) past experience to a learning subsystem, typically during sleep, to consolidate or stabilize representations.
- **Consolidation.** Transformation of recent, episode-bound memories into more abstract, generalized representations. Loosely, hippocampus → neocortex.
- **Self-supervised learning.** Learning without external reward or labels, from structure within the input stream itself (predictive, contrastive, reconstructive).

## Memory subsystems

- **Working memory.** Capacity-limited, attention-gated, short-lived state that holds the current "context" for cognition.
- **Episodic memory.** Bound representations of specific events, with temporal and contextual indexing. Fast write, source of replay material.
- **Semantic memory.** Generalized, schema-like knowledge abstracted from many episodes. Slow write, recipient of consolidation.

## Perception

- **Modality.** A sensory channel. Initial modalities: audio, video. Each has its own preprocessing pipeline before fusion.
- **Multimodal binding.** Linking concurrent features across modalities into a unified percept (e.g., a sound and its visual source).
- **Predictive error.** The mismatch between an expected and actual signal. A learning signal in predictive-processing architectures.

## Architecture

- **Subsystem.** A functionally coherent module (e.g., `working_memory`, `sensory.audio`). Replaceable; communicates via defined interfaces.
- **Neuromodulation.** Global, slowly-varying state (arousal, valence, novelty/surprise) that gates learning rate, attention, and mode transitions.
- **World model.** An internal generative model of the environment used for prediction, replay, and (eventually) planning.
- **Observation log.** A structured, persistent record of subsystem events, used for debugging, replay, and post-hoc analysis.

---

*This list is intentionally short. Grow it as terms get used in more than one place.*
