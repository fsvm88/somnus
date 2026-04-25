# Sleep and consolidation

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

## Sources

- Tononi & Cirelli, 2003. "Sleep and synaptic homeostasis: a hypothesis." *Brain Research Bulletin*.
- Tononi & Cirelli, 2014. "Sleep and the price of plasticity." *Neuron*.
- McClelland, McNaughton, & O'Reilly, 1995. *Psychological Review*, 102: 419–457.
- Shin, Lee, Kim, & Kim, 2017. NeurIPS. arXiv:1705.08690.
- Borbély, 1982. "A two process model of sleep regulation." *Human Neurobiology* — referenced for the conceptual framing of sleep pressure; not exhaustively reviewed here.
- Reviews of NREM/REM functional split: see `pmc.ncbi.nlm.nih.gov` review articles on systems memory consolidation during sleep (oscillations, neuromodulators, synaptic remodeling).
