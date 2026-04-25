# Vision

**somnus** is a research project toward an AI system that resembles, in its information-processing structure, a human brain.

## Defining commitments

1. **Continuous learning.** No fixed train/inference boundary. The system updates from experience as it runs and across sessions. Catastrophic forgetting is a problem we own, not a problem we sidestep.
2. **Sleep and wake.** The system has explicit offline phases for memory consolidation, abstraction, and replay. Sleep is not a metaphor — it is a first-class operating mode with its own dynamics.
3. **Multimodal sensory input.** Audio and video as primary streams, ingested in real time. Perception is grounded; the system does not operate on disembodied tokens.
4. **Brain-inspired architecture.** Subsystems and dynamics correspond, *in function*, to known brain systems (working memory, hippocampus, neocortex, basal ganglia, neuromodulators), at an algorithmic level — not a neuron-level simulation. The system has multiple **specialized regions** (per perceptual modality and per concept domain) plus an explicit **integrative / binding** area; it is *not* a single monolithic network.
5. **Bottom-up growth.** The system starts small — a "baby brain" — and expands its capacity in response to experience. New capacity is added when the system saturates against its current environment; sleep cycles drive the consolidation that produces this growth. We do not allocate a large network up front and we do not pretrain on a curated corpus. Saturation is detected behaviorally, not by a numerical threshold: if the system, post-sleep, cannot reconstruct the *gist* of recent episodes — getting categorical content right even when individual details may differ, dream-style — it is full and grows.
6. **Grounded concepts.** Every concept the system holds is anchored to perceptual exemplars — instances of what was seen or heard. Memories are not free-floating symbols; they are tied to experienced events through the modality channels that originally carried them. The architecture must support this binding from the start.

## What we are *not* building

- A chatbot. Language is a possible later modality, not the substrate.
- A neuron-level brain simulation. We are not NEST, NEURON, Brian, or Spaun.
- A foundation model trained once on an internet-scale corpus. The defining property is continuous, environment-driven adaptation from a small initial state.
- A system that requires cluster-scale compute. somnus must run on a single consumer machine.
- A productionized agent. This is research; deployment, scaling, multi-tenancy are out of scope.

## Levels of brain mimicry

It is useful to distinguish three levels at which a system can imitate a brain:

- **Mechanistic** — same substrate (neurons, ion channels, etc.). *Not our goal.*
- **Algorithmic** — same information-processing principles (sparse coding, predictive errors, replay-based consolidation, neuromodulatory gating). **This is our goal.**
- **Functional** — same observable input/output behavior, regardless of mechanism. We accept this as evidence of progress, but it is not the target on its own.

When a design choice is justified by appeal to biology, it should be at the algorithmic level — and accompanied by an engineering reason that stands on its own.

## Operating constraints

- **Local execution.** Target hardware is a single consumer machine: ~14 GB GPU VRAM (16 GB physical, minus ~2 GB reserved for the host desktop) and ~56 GB system RAM (64 GB physical, minus ~8 GB host overhead), plus disk for cold long-term memory. Cluster-only designs are wrong for somnus.
- **Multi-tier memory.** Memory is organized as a hierarchy — hot (in VRAM, currently active), warm (in system RAM, recently used), cold (compressed on disk, archived) — with a recollection layer that indexes across tiers. Long-term memory is sharded by region (modality / concept domain) and pulled into warm/hot storage when cued.
- **Frozen pretrained components are allowed; pretrained learning subsystems are not.** A pretrained perceptual encoder (audio or video) may serve as a fixed prior that bridges raw signal to features. The continuous-learning subsystems that sit on top of those features start small and grow with experience — that is what we mean by "no internet-scale pretraining of the core system."

## Why Rust

A long-running, memory-resident, real-time multimodal system benefits from predictable performance, the absence of GC pauses, strong concurrency primitives, and the ability to deploy as a single binary. We accept that the Rust ML/audio/vision ecosystem is less mature than Python's; see `tech-stack.md` for how we navigate that.

## Inspirations (not commitments)

These are starting points for `research/`, not architectural commitments:

- Predictive processing / active inference (Friston, Clark).
- Complementary Learning Systems (McClelland, McNaughton, O'Reilly): hippocampus-fast, neocortex-slow.
- Sparse distributed representations (Hawkins, HTM lineage).
- Replay and memory consolidation, sharp-wave ripples, NREM/REM dynamics.
- Free-energy and efficient-coding views of cortex.
- Neuromodulation as global gain control (dopamine, acetylcholine, noradrenaline).
