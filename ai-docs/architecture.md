# Architecture

> **Status:** working sketch. Reflects decisions through the vision discussion (Q1/Q2/Q3 + sub-Q1). Subject to further change as research matures. Update this doc when subsystem boundaries shift; record the *why* in an ADR.

somnus is decomposed by **function**, in loose correspondence with brain systems. Four shaping commitments:

- **Regional specialization** per perceptual modality and per concept domain, *not* a monolithic network.
- **Explicit binding** of cross-region information into unified episodes.
- **Multi-tier memory** (hot / warm / cold) with a cross-tier recollection index.
- **Behavioral, growth-driven saturation** detected during sleep.

The carve-up is revisable; what matters is that subsystems are *replaceable* and interfaces between them are roughly right.

## Top-level structure

```
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │ audio input  │  │ video input  │  │ text bypass  │
   │ (mic / file) │  │ (cam / file) │  │   (debug)    │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          ▼                 ▼                 │
   ┌──────────────┐  ┌──────────────┐         │
   │ audio enc.   │  │ visual enc.  │         │
   │ (speech +    │  │   (frozen)   │         │
   │  env. sound; │  └──────┬───────┘         │
   │  frozen)     │         │                 │
   └──────┬───────┘         │                 │
          │                 │                 │
          ▼                 ▼                 ▼
   ╔══════════════════════════════════════════════════╗
   ║   Regional concept areas (continuous embeddings) ║
   ║                                                  ║
   ║  audio-speech │ audio-env │ visual │ symbolic    ║
   ╚════════════════════════╤═════════════════════════╝
                            │
                            ▼
                  ┌────────────────────┐
                  │  Integrative /     │  cross-modal
                  │  binding area      │  binding
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │  Episodic memory   │  HOT (VRAM)
                  │  (recent events)   │  one-shot writes
                  └─────────┬──────────┘
                            │ replay during sleep
                            ▼
   ┌──────────────────┐  ┌─────────────────────┐
   │  Recollection    │◄─┤  Semantic memory    │  WARM (system RAM)
   │  layer           │  │  (consolidated;     │  growable;
   │  (cross-tier     │  │  sparsely coded)    │  region-sharded
   │   index)         │  └─────────┬───────────┘
   └────────┬─────────┘            │ compress during sleep
            │ pulls cold→warm      ▼
            │             ┌─────────────────────┐
            └────────────►│  Long-term archive  │  COLD (disk)
                          │  (compressed;       │  sharded by
                          │  region × concept)  │  region × concept
                          └─────────────────────┘

           ↕ neuromodulation (global gain & state)
           ↕ sleep/wake controller (mode + growth + compression)
```

## Frontend: encoders and bypass

All frontend models are **frozen pretrained**. The continuous-learning system sits *above* them. Frozen-encoder + grow-on-top is the architectural answer to "no internet-scale pretraining of the core system" while still leveraging existing perceptual models.

- **Visual encoder.** Frozen pretrained model sized to the budget (e.g., DINO-small / SigLIP-small / MobileViT). Outputs continuous embeddings into the visual region. Specific model: future ADR.
- **Audio encoder(s).** Frozen, dual-function:
  - **Speech-recognition features** — outputs tied to spoken content (the text-equivalent representation).
  - **Environmental-sound features** — distinguishes vehicle noise, voice-in-crowd, ambient sound, etc.
  - May be **one multi-task encoder** (CLAP-style) or **two specialized encoders run in parallel** (e.g., Whisper-tiny + CLAP/AST). Implementation choice; future ADR. The architectural commitment is that *both* feature streams are produced.
- **Text bypass.** Debug/research channel that injects text-derived embeddings directly into the symbolic region, bypassing speech-to-text. The text encoder must produce embeddings in the **same representational space** as the audio encoder's speech-recognition output, so a concept reached via "I heard 'cat'" and a concept reached via "I read 'cat'" land in the same place.
  - **Limitation worth flagging:** text bypass loses prosody, speaker identity, and environmental context — perceptual richness real audio carries. Treat as research/debug, not a primary input.

**Output discipline: multi-source, not hard-classified.** Encoders emit rich continuous representations (embeddings, or per-class probability vectors), *not* a single dominant label. If an audio frame contains both an engine and a voice, both signals are present in the output — even if the engine is louder. Downstream regions are responsible for parsing co-occurring sources; the encoder does not collapse them. Same for visual: simultaneous objects in a frame all contribute to the embedding. This is the architectural answer to the "engine roaring overtakes the voice" failure mode.

This commitment slightly favors two specialized audio encoders (Whisper-style transcription + CLAP/AST-style environmental classification) over one multi-task encoder, since they emit different kinds of signals (sequential text + multi-label embedding) that compose without collapsing into each other.

## Regional concept areas

Each region holds **continuous-embedding clusters** — *not* discrete symbols. See `vision.md` commitment 6 (grounded concepts). Sparsity in coding minimizes interference and makes growth local.

Initial regions (revisable, see open questions):

- **audio-speech** — concepts grounded in spoken language.
- **audio-env** — concepts grounded in environmental sound (objects, events).
- **visual** — concepts grounded in visual experience.
- **symbolic** — concepts that are linguistically *named*, regardless of which sense brought them in. Receives input from audio-speech and from text bypass.

Discretization (e.g., a VQ-style codebook layer over continuous embeddings, or an emergent symbolic-discrete sub-region) is **out of scope for early phases**. It can be added later when reasoning over abstract concepts becomes a research target — biology and modern ML both suggest discrete symbols emerge from continuous representations, not the other way around.

## Integrative / binding area

Composes concurrent regional embeddings into a single unified representation: "what is happening now". This is the gateway from the regional layer to episodic memory.

Plays a binding role analogous to hippocampal/entorhinal binding in mammalian memory.

**Bidirectional with the regions.** The integrative area is not just a downstream consumer — it also feeds back into the regions to bias their interpretation using cross-modal priors. Concrete case: if visual mouth-motion is present, the integrative area raises the prior on "speech" in the audio-speech region, surfacing a faint voice signal that bottom-up audio processing alone would have missed under engine noise. Mechanistically this is predictive-processing / Bayesian-brain top-down influence; phenomenologically it covers things like the McGurk effect and cocktail-party-with-visual-cues disambiguation. The principal-flow diagram above shows bottom-up arrows for clarity; the top-down paths back into the regions are implicit and live here.

## Memory hierarchy

Memory is organized along **two orthogonal axes**:

- **Regional axis** — which concept area a memory belongs to.
- **Tiered axis** — how recently / actively it is being used.

| Tier | Where      | Contents                                                              |
|------|------------|-----------------------------------------------------------------------|
| HOT  | VRAM       | Episodic memory: recent unified episodes, source of replay.           |
| WARM | system RAM | Semantic memory: consolidated, sparsely-coded, growable, region-shard.|
| COLD | disk       | Long-term archive: compressed older content, sharded by region/concept. |

The **recollection layer** is the cross-tier index. Given a partial cue (from current perception or from internal state during sleep), it identifies relevant memories anywhere in the hierarchy and materializes them into warmer tiers as needed.

Conceptually:

- Episodic memory ≈ short-term hippocampal trace.
- Semantic memory ≈ active neocortical knowledge.
- Long-term archive ≈ remote / consolidated cortical memory.
- Recollection layer ≈ hippocampal retrieval mechanism.

## Sleep, growth, and compression

During sleep, the system runs four interleaved processes:

1. **Replay.** Episodes from episodic memory are replayed into the semantic memory of the regions they involve, driving consolidation.
2. **Self-probe (gist-level).** The system picks recent episodes and attempts to *reconstruct* them through the semantic + recollection path. The reconstruction is compared at **gist level** — categorical content matches, even when details differ (dream-style: town and weather right, color and order of houses wrong is acceptable).
3. **Growth.** If gist-level reconstruction fails for a meaningful share of recent episodes in a region, that region's semantic memory **grows**. Growth is local — adding units, not whole layers — and likely cascade-correlation-style (new units trained on residual error). See open questions.
4. **Compression.** Under memory pressure, oldest / least-cued semantic content is quantized, pruned, or distilled, then spilled to the cold archive.

This is the operationalization of `vision.md` commitment 5 (bottom-up growth) and the behavioral saturation criterion stated there.

## Cross-cutting

- **Time.** Monotonic logical clock + wall-clock. Wake is real-time; sleep is wall-clock-decoupled.
- **Persistence.** Episodic, semantic, and archive all survive shutdown. Episodic and semantic via snapshots; archive lives on disk by definition.
- **Telemetry / observation log.** Every subsystem emits structured events. Always-on, structured, replayable; basis for debugging, post-hoc analysis, and the system's own self-probe.
- **Configuration.** Hyperparameters and runtime settings in versioned config files; reproducibility depends on this.
- **Neuromodulation.** Global signals (arousal, valence, novelty/surprise) modulate learning rates, attention, and mode transitions. Threaded through subsystems as scalars/vectors, not a separate subsystem in the data-flow sense.

## Open architectural questions

Tracked here so they don't get lost. Resolutions become ADRs.

- **Initial regional set.** Is the minimal viable set audio-speech + symbolic, with audio-env and visual added later? Or commit to all four regions from Phase 1?
- **Encoder specifics.** Single multi-task audio encoder (CLAP-style) vs. two specialized (Whisper + CLAP/AST). Visual encoder choice.
- **Growth mechanism.** Cascade-Correlation-style unit addition? At what granularity (single unit, group, sub-cluster)? How do new units integrate with sparse codes?
- **Recollection-layer implementation.** Vector ANN (e.g., `hnsw_rs`) over the archive? Hopfield-style associative memory? Hybrid?
- **Compression strategy.** Quantization, structured pruning, knowledge distillation, or layered combination? When is content moved cold vs. dropped?
- **Sparse-coding scheme.** k-WTA, top-k, learned sparsity penalty? Per-region or global?
- **Replay budget within a sleep cycle.** Split between replay (consolidation), self-probe, growth, and compression.
- **Sleep granularity.** One sleep mode or multiple (NREM-like for consolidation, REM-like for generative replay / dreaming)?
- **Action loop.** Does the system have one in early phases? Effector — internal state only, or external (audio out, motor)?
- **Scheduling.** Async tasks, dedicated threads, event loop? How do real-time wake constraints coexist with batch sleep work?
- **Tensor representation.** Single shared tensor type across subsystems, or each subsystem picks its own?
