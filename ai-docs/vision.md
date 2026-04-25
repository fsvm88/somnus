# Vision

**somnus** is a research project toward an AI system that resembles, in its information-processing structure, a human brain.

## Defining commitments

1. **Continuous learning.** No fixed train/inference boundary. The system updates from experience as it runs and across sessions. Catastrophic forgetting is a problem we own, not a problem we sidestep.
2. **Sleep and wake.** The system has explicit offline phases for memory consolidation, abstraction, and replay. Sleep is not a metaphor — it is a first-class operating mode with its own dynamics.
3. **Multimodal sensory input.** Audio and video as primary streams, ingested in real time. Perception is grounded; the system does not operate on disembodied tokens.
4. **Brain-inspired architecture.** Subsystems and dynamics correspond, *in function*, to known brain systems (working memory, hippocampus, neocortex, basal ganglia, neuromodulators), at an algorithmic level — not a neuron-level simulation.

## What we are *not* building

- A chatbot. Language is a possible later modality, not the substrate.
- A neuron-level brain simulation. We are not NEST, NEURON, Brian, or Spaun.
- A foundation model trained once and frozen. The defining property is continuous adaptation.
- A productionized agent. This is research; deployment, scaling, multi-tenancy are out of scope.

## Levels of brain mimicry

It is useful to distinguish three levels at which a system can imitate a brain:

- **Mechanistic** — same substrate (neurons, ion channels, etc.). *Not our goal.*
- **Algorithmic** — same information-processing principles (sparse coding, predictive errors, replay-based consolidation, neuromodulatory gating). **This is our goal.**
- **Functional** — same observable input/output behavior, regardless of mechanism. We accept this as evidence of progress, but it is not the target on its own.

When a design choice is justified by appeal to biology, it should be at the algorithmic level — and accompanied by an engineering reason that stands on its own.

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
