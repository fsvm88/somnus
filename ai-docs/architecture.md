# Architecture

> **Contents:** High-level system structure — encoders, 5 regional concept areas (audio-speech, audio-env, visual, symbolic, language), binding area (cross-attention + GRU), 3-tier memory (hot/warm/cold), sleep cycle (NREM+REM), action decoder (Qwen3-0.6B). Implementation state as of 2026-06-07 included inline.  
> **When to read:** Understanding how subsystems connect; checking which open architectural questions are resolved vs. open.
>
> **Status:** working sketch. Reflects decisions through the vision discussion (Q1/Q2/Q3 + sub-Q1). Subject to further change as research matures. Update this doc when subsystem boundaries shift; record the *why* in an ADR.
>
> **Implementation state (2026-07-11):** Phases 1-14 + architectural roundup + gap fixes + Item 6 (DINOv3-ViT-S, deployed) + Item 7 Phase 2 (DER++ stale blocking) + Item 5 (Qwen3 adopted, ADR 0036) complete. `somnus-sensory` captures audio/video. `somnus-encoding` runs Whisper-tiny (384-dim), CLAP-HTSAT (512-dim), DINOv3-ViT-S (384-dim, active; DINOv2-S/14 fallback) via ORT. `somnus-perception` runs five k-WTA layers (n=1024, k=20) — audio-speech, audio-env, visual, symbolic, language (`LANG_HIDDEN`=1024, Qwen3-0.6B) — plus a 4-head cross-attention binding area with GRU temporal context (GRU_HIDDEN=256) → 640-dim episode vectors + SAE decoders. `somnus-memory` implements the three-tier hierarchy: `EpisodicBuffer` (bounded ring, hot), `SemanticStore` (hnsw_rs HNSW, warm), `ColdArchive` (INT8-quantised append-only, cold), `RecollectionLayer` (warm query + cold→warm promotion). The `somnus` binary wires everything with a full sleep cycle (NREM+REM, DER++ stale-output detection), text bypass, auto-grounding, action decoder (Qwen3-0.6B) and eval harness. A/B on a speech-dense corpus favored Qwen3 over Qwen2.5 on proj_loss (~22% lower, consistent, 5-session campaign); adopted as the sole LM 2026-07-05 (ADR 0036, Qwen2.5 removed), confirmed same-day via direct `generate()` comparison (coherent English vs. incoherent token garbage).
>
> **Since 2026-07-05:** two A/B rejections kept opt-in-only (DER++ auto-λ, engineering-log.md §3; EfficientAT-mn10, §9), one candidate scoped and not pursued (M2D-CLAP, §8) — no default changes. The growth mechanism (`grow_region_seeded`, ADR 0024 §2c) was characterized empirically for the first time via two 75-cycle pilots (2026-07-09/11): ~1 growth event/4.7 cycles, gate-limited not signal-limited, `per_region_mse` improving alongside growth, no memorization-plateau even against a 145-clip diverse corpus (`growth-mechanisms.md` "Empirical validation", engineering-log.md §16). Both pilots ran in isolated scratch dirs — production `memory/` unchanged.
>
> **Since 2026-07-12:** `memory/` wiped, full 10x growth campaign launched (target ~9,660 units/region, honeybee-order vs. the fruit-fly-order ~1,024-unit baseline above) — **completed 2026-08-15: all five regions at 9,821 units (101.7%)** after 791 cycles/188 growth events, 0 panics/segfaults, output MSE −74% first-vs-last-10 (engineering-log.md §34). The "n=1024, k=20" above describes the architectural *baseline*, not production size (production k is `scaled_k(9821)`). Alongside: K_ACTIVE scales with growth (`region::scaled_k`, §17), optimizer resets scoped to the grown region only (§18), growth batch size is novelty-proportional (§19). Perception inference auto-switches to ORT/MIGraphX past the CPU/GPU crossover with no manual flag (ADR 0037, superseding Phase 14's opt-in flag) — confirmed live with zero fallback and a genuine ~3-4× speedup once two post-launch dimension-drift bugs were fixed (engineering-log.md §24).

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

                  ┌──────────────────────────────────────┐
                  │  ActionDecoder                        │
                  │  projection (640 → 5×1024) +          │
                  │  frozen Qwen3-0.6B                    │
                  │                                      │
                  │  every episode: attn weights →        │
                  │    per-region gain (k-WTA input)      │
                  │  on query (stdin): autoregressive     │
                  │    text → stdout JSON                 │
                  └──────────────────────────────────────┘
```

## Frontend: encoders and bypass

All frontend models are **frozen pretrained**; the continuous-learning system sits *above* them — frozen-encoder + grow-on-top is the architectural answer to "no internet-scale pretraining of the core system" while still leveraging existing perceptual models.

- **Visual encoder.** Frozen pretrained model sized to budget (DINO-small/SigLIP-small/MobileViT). Outputs into the visual region.
- **Audio encoder(s).** Frozen, dual-function: speech-recognition features (spoken content, text-equivalent) and environmental-sound features (vehicle noise, voice-in-crowd, ambient sound). One multi-task encoder (CLAP-style) or two specialized ones (Whisper-tiny + CLAP/AST) — implementation choice; the architectural commitment is that *both* streams are produced.
- **Text bypass.** Debug/research channel injecting text-derived embeddings directly into the symbolic region, bypassing speech-to-text — must land in the *same representational space* as the audio encoder's speech output, so "heard 'cat'" and "read 'cat'" land in the same place. **Limitation:** loses prosody/speaker identity/environmental context — research/debug channel, not primary input.

**Output discipline: multi-source, not hard-classified.** Encoders emit rich continuous representations, not a single dominant label — an audio frame with both an engine and a voice keeps both signals even if the engine is louder; downstream regions parse co-occurring sources, the encoder doesn't collapse them. Same for simultaneous visual objects. This is the architectural answer to "engine roaring overtakes the voice." Slightly favors two specialized audio encoders over one multi-task encoder, since sequential text + multi-label embedding compose without collapsing into each other.

## Regional concept areas

Each region holds **continuous-embedding clusters**, not discrete symbols (`vision.md` commitment 6). Sparsity minimizes interference and makes growth local.

Regions: **audio-speech** (spoken language), **audio-env** (environmental sound), **visual** (visual experience), **symbolic** (linguistically-named concepts regardless of sense, fed from audio-speech + text bypass), **language** (Qwen3 embed table, 1024-dim mean-pooled, from stdin text + Whisper ASR — trains the action-decoder projection during NREM).

Discretization (VQ-style codebook, emergent symbolic sub-region) is out of scope for early phases — biology and ML both suggest discrete symbols emerge from continuous representations, not the reverse.

## Integrative / binding area

Composes concurrent regional embeddings into one unified "what is happening now" representation — the gateway from regions to episodic memory, analogous to hippocampal/entorhinal binding.

**Bidirectional with the regions.** Not just a downstream consumer — feeds back to bias regional interpretation via cross-modal priors. E.g. visual mouth-motion raises the prior on "speech" in audio-speech, surfacing a faint voice under engine noise. Predictive-processing/Bayesian-brain top-down influence; covers McGurk-effect-style and cocktail-party-with-visual-cues disambiguation. The diagram above shows bottom-up arrows for clarity — top-down paths back into the regions are implicit and live here.

## Memory hierarchy

Memory is organized along **two orthogonal axes**: regional (which concept area) and tiered (how recently/actively used).

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

## Sleep cycle and wake/sleep control

### Wake (live mode)

Real-time: **reads** semantic memory + cold archive (via recollection) for recognition/retrieval; **writes** integrated episodes from the binding area to episodic memory; **does not update** semantic memory (all consolidation deferred to sleep, keeping wake real-time-cheap and preventing incoming experience from interfering with consolidated knowledge mid-stream); **working memory** holds transient context, no long-term updates.

### Sleep stages

Two-stage, loosely modeled on mammalian NREM/REM.

**NREM-like — consolidation, homeostasis, compression.** Replay: recent episodic memories replayed into the semantic memory of the regions they involve. Homeostatic downscaling: gentle global decay of semantic weights with selective protection of replay-reinforced ones, restoring plasticity headroom (Tononi's synaptic homeostasis hypothesis). Compression: oldest/least-cued semantic content quantized/pruned/distilled, spilled to cold.

**REM-like — generative replay, self-probe, growth.** Generative replay: reconstruct recent episodes through the semantic+recollection path. Self-probe (gist-level): compare reconstruction to original categorically — right scene/objects/events even if details differ (dream-style: town/weather right, house color/order wrong is fine). Growth: if enough recent episodes in a region fail gist-level recall, that region's semantic memory grows locally (new units, not whole layers — cascade-correlation-style, trained on residual error). **Future scope (Phase 3+):** creative recombination — stitching episode fragments into synthetic novel episodes for replay; the architectural locus for recombinational creativity.

Operationalizes `vision.md` commitments 2 (sleep/wake) and 5 (bottom-up growth + behavioral saturation).

### Entry triggers and duration

Triggers (any): exhaustion (episodic buffer high-water mark), wake-performance degradation (recognition/predictive accuracy drops), schedule fallback (max wake-time cap). Duration adaptive: run until consolidation goals met (replay covers recent episodes, gist-recall stabilizes, compression clears pressure) or a max-sleep-time cap, whichever first.

### Interruptibility

Interruptible by high-salience external events (loud novel sound, motion onset, alert breaching a salience threshold). Sleep operations are idempotent/checkpointed so partial work survives interruption; the system wakes, and the next sleep cycle resumes from the checkpoint.

## Cross-cutting

- **Time.** Monotonic logical clock + wall-clock; wake is real-time, sleep is wall-clock-decoupled.
- **Persistence.** Episodic/semantic/archive all survive shutdown — episodic and semantic via snapshots, archive lives on disk by definition.
- **Telemetry.** Every subsystem emits structured events, always-on, replayable — basis for debugging, post-hoc analysis, and the system's own self-probe.
- **Configuration.** Hyperparameters/runtime settings in versioned config files — reproducibility depends on it.
- **Neuromodulation.** Global signals (arousal, valence, novelty/surprise) modulate learning rates, attention, mode transitions — threaded through subsystems as scalars/vectors, not a separate subsystem.

## Open architectural questions

Tracked here so they don't get lost. Resolutions become ADRs.

- ~~**Initial regional set.**~~ Resolved in ADR 0005: four regions from Phase 3 (audio-speech, audio-env, visual, symbolic). Phase 6 (ADR 0013) added the language region (index 4, 1024-dim Qwen3 embed, originally 896-dim Qwen2.5 — see ADR 0036). Five regions total.
- ~~**Encoder specifics.**~~ Resolved in ADR 0003: Whisper-tiny (speech, 384-dim) + CLAP-HTSAT-unfused (env, 512-dim) + DINOv2-S/14 or DINOv3-ViT-S (visual, 384-dim, swappable via model file).
- ~~**Growth mechanism.**~~ Resolved in ADR 0024 §2c ("Growth controller — within-session signal"): `grow_region_seeded()` (`crates/somnus-perception/src/lib.rs`), a seeded weight-tensor resize (new columns seeded from normalised high-residual inputs, not a cascade-correlation candidate-training pool as originally sketched in `growth-mechanisms.md`), triggered by `within_session_gist_rate` with a 3-cycle cooldown backstop. Empirically characterized via two 75-cycle pilots (2026-07-09/11, `gaps.md` §16, `growth-mechanisms.md` "Empirical validation"): gate-limited cadence (~1 event/4.7 cycles), no memorization-plateau risk even against a diverse 145-clip corpus. Full 10x-scale campaign not yet run.
- ~~**Recollection-layer implementation.**~~ Resolved in ADR 0006: `hnsw_rs` HNSW (L2 distance) for warm tier; cold→warm promotion stubbed for Phase 5.
- ~~**Compression strategy.**~~ Resolved: INT8 quantization (640 bytes/record) and Product Quantization (27.5× reduction) both implemented — see `ai-docs/research/memory-compression.md`. `ColdArchive` is INT8-quantised (implementation-state paragraph above).
- ~~**Sparse-coding scheme.**~~ Resolved in ADR 0005: k-WTA with per-unit duty-cycle boosting, n=1024, k=20 (2% sparsity). Per-region weights; shared hyperparameters.
- **Sleep-cycle time budget (narrowed).** ADR 0024's S1–S4 controllers + growth controller (§2c) now govern most of the *adaptive triggering* this question originally gestured at (settle-rate, wakefulness pressure, prior-beta scale-up, growth cooldown) — that part is resolved. What remains genuinely open: no explicit NREM-vs-REM wall-clock allocation exists; the two stages run to their own completion criteria rather than sharing a budget. Revisit only if sleep duration becomes a practical bottleneck (not observed yet).
- ~~**Action loop.**~~ Resolved in ADR 0012: region-token projection (640→5×1024) + frozen Qwen3-0.6B (adopted over Qwen2.5-0.5B, ADR 0036, following the ADR 0035 compile-time-choice A/B infrastructure). Per-episode LLM attention weights drive per-region gain modulation; full autoregressive decode on stdin query. NREM projection training uses LM contextualised hidden-state distillation targets (ADR 0012 amendment). A/B favored Qwen3 on proj_loss (~22% lower, 2026-07-05), confirmed with a direct generation-quality comparison the same day (`ai-docs/a_b_qwen3_findings.md`).
- **Scheduling.** Async tasks, dedicated threads, event loop? How do real-time wake constraints coexist with batch sleep work?
- **Tensor representation.** Single shared tensor type across subsystems, or each subsystem picks its own?
