# Architecture

> **Status:** initial sketch. Subject to substantial change as research matures. Update this doc when subsystem boundaries shift; record the *why* in an ADR.

somnus is decomposed by **function**, in loose correspondence with brain systems. The carve-up is meant to be revisable; what matters at this stage is that subsystems are *replaceable*, and that interfaces between them are roughly right.

## Top-level subsystems

```
                ┌──────────── sensory ────────────┐
                │   audio pipeline   video pipeline│
                └─────────────────┬────────────────┘
                                  │ feature streams
                ┌─────────────────▼────────────────┐
                │            perception            │  per-modality features,
                │   (recognition, multimodal bind) │  predictive errors
                └─────────────────┬────────────────┘
                                  │ percepts
                ┌─────────────────▼────────────────┐
                │          working memory          │  attention-gated,
                │                                  │  capacity-limited
                └──┬──────────────┬────────────────┘
                   │              │
                   ▼              ▼
              ┌─────────┐   ┌──────────┐
              │episodic │   │ semantic │
              │ memory  │◀═▶│  memory  │   ◀═▶ via consolidation (sleep)
              └────┬────┘   └────┬─────┘
                   │             │
                   └──────┬──────┘
                          ▼
                ┌──────────────────────────────────┐
                │       cognition / control        │  attention, retrieval,
                │      (incl. world model)         │  (eventually) action
                └──────────────────────────────────┘

                          ↕   neuromodulation (global gain & state)
                          ↕   sleep/wake controller (mode)
```

## Subsystem responsibilities (sketch)

- **Sensory.** Real-time ingestion of audio (microphone or file) and video (camera or file). Per-modality preprocessing → low-level feature streams. Stable, well-defined output rate.
- **Perception.** Per-modality feature extraction, cross-modal binding, generation of predictive-error signals. Bridges raw features to percepts.
- **Working memory.** Holds the current behavioral/cognitive context. Attention-gated read/write. Bounded capacity; explicit forgetting policy.
- **Episodic memory.** Stores event-bound, contextualized representations with rapid (one-shot) writes. Indexed by time/context. Source of replay material during sleep.
- **Semantic memory.** Slowly-updated, generalized knowledge. Recipient of consolidation from episodic memory. Read during wake, written mostly during sleep.
- **Cognition / control.** Selects what to attend to, what to retrieve, what to (eventually) act on. Hosts the world model used for prediction and replay.
- **Neuromodulation.** Global scalar/vector signals (arousal, valence, novelty/surprise) that gate learning rate, attention, and mode transitions. *Not a subsystem in the data-flow sense — it modulates other subsystems.*
- **Sleep/wake controller.** Manages mode transitions. During sleep: drives replay, runs consolidation passes, applies homeostatic adjustments (synaptic downscaling, pruning).

## Cross-cutting concerns

- **Time.** A monotonic logical clock plus wall-clock are both available. Subsystems run on their own cadence; the wake loop is real-time, sleep is not.
- **Persistence.** Episodic and semantic stores survive shutdown. Working memory and short-lived caches do not. Storage backend is deliberately unspecified — see ADRs as they land.
- **Telemetry / observation log.** Each subsystem emits structured events; the observation log is the basis for debugging, replay, and post-hoc analysis. Should be cheap enough to leave on always.
- **Configuration.** Hyperparameters and runtime settings live in versioned config files; reproducibility depends on this.

## Open architectural questions

Tracked here so they don't get lost. Resolutions become ADRs.

- **Perception ↔ working memory coupling.** Push, pull, or both? Is there a predictive top-down path from working memory into perception?
- **Action loop.** Does the system have an action loop in early phases, or is it perception-only? If actions exist, what's their effector — internal state only, or external (audio out, motor)?
- **Language.** Late-stage modality, or absent entirely?
- **Scheduling.** Async tasks, dedicated threads, an event loop, or some hybrid? How do real-time wake constraints coexist with batch sleep work?
- **Tensor representation.** Single shared tensor type across subsystems, or each picks its own? Trades off interop vs. flexibility.
- **State boundaries.** Where exactly does "working memory" end and "episodic memory" begin? Is there a continuum?
- **Sleep granularity.** One monolithic sleep mode, or multiple (NREM-like, REM-like) with different dynamics?
