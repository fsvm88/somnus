# Sleep and consolidation

> **Contents:** NREM/REM biology mapped to somnus's two-stage cycle (NREM=replay+homeostasis+compression, REM=generative replay+gist probe+growth). Covers sharp-wave ripples, synaptic homeostasis hypothesis, memory budget estimates for the hardware target.  
> **When to read:** Understanding why two sleep stages; tuning sleep triggers (high-water, schedule, perf-degradation); checking biology-to-engineering mapping for a sleep-cycle change.

Literature behind somnus's two-stage sleep cycle. The commitment lives in `vision.md` (commitment 2); the operationalization is in `architecture.md`'s "Sleep cycle and wake/sleep control" section. This note captures the biology and ML evidence that motivated the choices, and is honest about where somnus diverges.

## Summary of somnus's choice

Two-stage sleep, loosely modeled on mammalian NREM/REM:

- **NREM-like:** replay-driven consolidation, homeostatic weight downscaling, compression to cold storage.
- **REM-like:** generative replay, gist-level self-probe of recent episodes, growth of region-level capacity where recall fails.

Triggered by behavioral signals (exhaustion of episodic memory, wake-performance degradation), with a schedule fallback. Adaptive duration with a max-cap. Interruptible by high-salience external events.

## NREM sleep — what biology says

**Replay during sharp-wave ripples (SWRs).** Hippocampal place-cell sequences from prior wake are reactivated during slow-wave sleep, often time-compressed and structured around the ~100–250 Hz SWRs. The reactivations transfer information to neocortex and are widely accepted as a candidate substrate for systems-level memory consolidation. Recent work shows NREM micro-structure further partitions: contracted-pupil substates handle initial consolidation of recent experience, dilated-pupil substates handle maintenance of older memories.

**Homeostatic synaptic downscaling.** Tononi & Cirelli's **Synaptic Homeostasis Hypothesis (SHY)**, originally 2003 (*Brain Research Bulletin*), refined 2014 ("Sleep and the price of plasticity", *Neuron*). Wake produces a net increase in synaptic strength (potentiation from learning). NREM sleep — specifically the UP states of cortical slow waves and hippocampal SWRs — drives a *systematic but selective* downscaling: weak synapses lose strength faster than strong ones. The hypothesized functions: energy conservation, signal-to-noise improvement, avoidance of synaptic saturation, and restoration of plasticity headroom for the next wake cycle.

**Engram broadcasting.** Network-wide disinhibition during NREM allows engram-driven bursts; this strengthens connections from the original engram neurons to less-active downstream neurons, distributing the memory.

## REM sleep — what biology says

**Theta and gamma coherence** between hippocampus and neocortex during REM, facilitating cross-region information exchange.

**Competitive refinement / pruning.** REM raises network inhibition; only the most active excitatory neurons keep firing. The result is competitive strengthening of strong memory traces and pruning of weak ones — a *refinement* dynamic distinct from NREM's broadcasting.

**Emotional and schema integration.** REM is implicated in emotional memory consolidation and schema-level integration of recent experience into existing knowledge.

**Generative content (dreams).** REM is when vivid dream content is most reported, often a creative recombination of recent experience and stored memory. The functional role of dreaming itself is contested — proposals range from "byproduct" to "active rehearsal" to "creative-recombination engine".

## Mapping to somnus

| Biology (NREM/REM)                         | somnus (NREM-like / REM-like)                                                       |
|--------------------------------------------|----------------------------------------------------------------------------------|
| Replay via SWRs → cortex                   | Replay episodic → semantic, region by region                                     |
| SHY: global synaptic downscaling           | Homeostatic weight decay with selective protection of replay-reinforced weights |
| Engram broadcasting                        | Episode → region's semantic memory updates                                       |
| REM theta-gamma coherence                  | Generative replay through semantic + recollection path                           |
| REM competitive refinement / pruning       | Compression: quantize / prune / distill, spill to cold archive                  |
| Dream-style generative recombination       | (Phase 3+) creative recombination of episodes for replay                         |
| —                                          | **Cascade-correlation-style growth in REM-like phase**                          |

The last row is the **honest divergence**: REM in mammals does not add new neurons. somnus's REM-like phase does (cascade-correlation unit addition where gist-level recall fails). This is an engineering choice, not a biological model — using the offline window when consolidation has just exposed which regions are saturated is a convenient time to grow. Adult neurogenesis happens in mammalian hippocampus but is not sleep-stage-locked; somnus is freely re-purposing the REM phase.

## Sleep entry triggers — biology and somnus

In mammals, sleep is regulated by two interacting processes (Borbély, 1982):
- **Process S (homeostatic)** — sleep pressure rises with wake duration; molecularly tracked by adenosine accumulation.
- **Process C (circadian)** — ~24 h rhythm modulating propensity to sleep/wake.

somnus does not simulate sleep pressure or circadian rhythm. We use behavioral proxies:
- Episodic memory passing a high-water mark (analogue of Process S).
- Wake-performance degradation on incoming inputs.
- Schedule fallback (loose analogue of Process C — pragmatic, not principled).

## Replay in ML continual learning

ML continual learning has converged on replay as the most robust class of methods empirically. See `continuous-learning.md`. Generative replay (Shin et al., 2017) explicitly cites the hippocampal generative role.

A consistent gap between ML and biology: biological replay has features ML usually doesn't replicate — time-compression, time-reversal (replay both forward and backward), prioritization of high-novelty/high-reward sequences, and replay of *non-experienced* trajectories (sequences the animal could plausibly take but hasn't). These are research opportunities for somnus over time.

## Open questions for somnus

- **Sleep-cycle ratio.** Mammals oscillate in ~90-min NREM/REM cycles, mostly NREM early in the night and REM later. somnus has no such constraint — what's the right ratio for our purposes?
- **Total sleep fraction.** Mammals sleep 20–30% of the day. somnus is unconstrained; can be optimized for whatever produces best wake performance.
- **Interruption cost.** Sleep deprivation impairs consolidation in mammals. What's the analogous cost in somnus? Specifically, how much partial-sleep work is recoverable after interruption?
- **Creative recombination's value.** Phase-3+ feature. Will REM-style dream recombination produce useful generalization, or just noise?

## Hardware budget and time tradeoffs

> This section was added as a tradeoffs supplement. The biology and design rationale above are unchanged.
>
> Hardware target: 14 GB VRAM, 56 GB RAM.

### Per-operation time estimates (one sleep cycle)

All estimates assume 14 GB VRAM GPU; cognitive layers ~10M params; 10K recent episodes in episodic memory.

| Sleep operation | Estimated wall-clock time | Notes |
|----------------|--------------------------|-------|
| **NREM-like: replay consolidation** (10K episodes) | 20–50 s | ~2–5 ms/episode forward+backward |
| **NREM-like: homeostatic downscaling** (1 pass over weights) | <1 s | One multiply-add per param |
| **NREM-like: compression + cold spill** (100K vectors, INT8 quantize + write) | ~2–5 min | Includes NVMe write |
| **REM-like: generative replay + self-probe** (10K episodes) | 1–5 min | Reconstruction + cosine compare |
| **REM-like: growth** (1 unit per region, 4 regions) | **40–160 min** | Cascade-correlation candidate training |
| **REM-like: HNSW index rebuild** (if needed) | 5–30 min | Triggered by tombstone accumulation |
| **Total sleep cycle (no growth)** | ~10–20 min | Replay + homeostasis + probe + compression |
| **Total sleep cycle (with growth)** | **~1–4 hours** | Growth dominates |

**Key insight:** growth is 1–2 orders of magnitude more expensive than all other sleep operations combined. Sleep duration should be adaptive based on whether growth is triggered.

### NREM/REM time budget recommendation

| Growth triggered? | Recommended sleep structure |
|-------------------|-----------------------------|
| No | Short sleep: 90% NREM replay+homeostasis+compression, 10% REM probe |
| Yes | Long sleep: 50% NREM, 50% REM with growth; cap at max-sleep-time |

The mammalian NREM-heavy early sleep / REM-heavy late sleep pattern is a convenient default; for somnus the right split depends on whether growth is needed, not circadian rhythm.

### Memory budget during sleep

Sleep operations use VRAM for:
- Replay forward+backward passes: ~200–500 MB activation memory (frees after each episode).
- Growth candidate training: one candidate unit's activations + gradients: <10 MB. Fine to run 20 candidates in parallel if VRAM allows.
- Reconstruction for self-probe: same as replay.

The encoders (frozen) can be swapped to RAM during sleep if their VRAM is needed for sleep operations. Encoder swap cost: ~500 MB transfer at ~16 GB/s PCIe 4.0 = ~30 ms. Negligible.

### Sleep entry threshold calibration

Sleep entry triggers from `architecture.md`:
- **Episodic memory high-water mark:** the warm/cold compression creates pressure sooner than VRAM fills (VRAM can hold ~12M episodes; the right trigger is probably ~10K–100K episodes to limit growth work per cycle).
- **Wake-performance degradation:** reconstruction error on recent inputs exceeds a baseline. This is the most direct signal that consolidation is needed.
- **Schedule fallback:** recommended max wake duration = 30–60 min of active experience before forcing sleep; this keeps each sleep cycle's growth load manageable.

## Sources

- Tononi & Cirelli, 2003. "Sleep and synaptic homeostasis: a hypothesis." *Brain Research Bulletin*.
- Tononi & Cirelli, 2014. "Sleep and the price of plasticity." *Neuron*.
- McClelland, McNaughton, & O'Reilly, 1995. *Psychological Review*, 102: 419–457.
- Shin, Lee, Kim, & Kim, 2017. NeurIPS. arXiv:1705.08690.
- Borbély, 1982. "A two process model of sleep regulation." *Human Neurobiology* — referenced for the conceptual framing of sleep pressure; not exhaustively reviewed here.
- Reviews of NREM/REM functional split: see `pmc.ncbi.nlm.nih.gov` review articles on systems memory consolidation during sleep (oscillations, neuromodulators, synaptic remodeling).
