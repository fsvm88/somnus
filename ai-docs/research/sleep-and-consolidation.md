# Sleep and consolidation

> **Contents:** NREM/REM biology mapped to somnus's two-stage cycle (NREM=replay+homeostasis+compression, REM=generative replay+gist probe+growth). Covers sharp-wave ripples, synaptic homeostasis hypothesis, memory budget estimates for the hardware target.
> **When to read:** Understanding why two sleep stages; tuning sleep triggers (high-water, schedule, perf-degradation); checking biology-to-engineering mapping for a sleep-cycle change.

Literature behind somnus's two-stage sleep cycle. Commitment in `vision.md` (2), operationalization in `architecture.md`'s "Sleep cycle and wake/sleep control." Captures the biology/ML evidence behind the choices and is honest about divergences.

## Summary of somnus's choice

Two-stage sleep, loosely modeled on mammalian NREM/REM: **NREM-like** — replay-driven consolidation, homeostatic weight downscaling, cold-storage compression. **REM-like** — generative replay, gist-level self-probe, region-capacity growth where recall fails. Triggered by behavioral signals (episodic-memory exhaustion, wake-performance degradation) with a schedule fallback; adaptive duration with a max-cap; interruptible by high-salience external events.

## NREM sleep — what biology says

**Replay during sharp-wave ripples (SWRs).** Hippocampal place-cell sequences from prior wake reactivate during slow-wave sleep, time-compressed, structured around ~100-250Hz SWRs — transfers information to neocortex, a widely-accepted substrate for systems-level consolidation. Recent work shows NREM micro-structure further partitions: contracted-pupil substates handle initial consolidation of recent experience, dilated-pupil substates handle maintenance of older memories.

**Homeostatic synaptic downscaling.** Tononi & Cirelli's Synaptic Homeostasis Hypothesis (SHY, 2003, refined 2014 "Sleep and the price of plasticity," *Neuron*): wake nets a synaptic-strength increase (learning potentiation); NREM (cortical slow-wave UP states, hippocampal SWRs) drives systematic-but-selective downscaling — weak synapses lose strength faster than strong ones. Hypothesized functions: energy conservation, signal-to-noise improvement, avoiding synaptic saturation, restoring plasticity headroom for the next wake.

**Engram broadcasting.** Network-wide NREM disinhibition allows engram-driven bursts, strengthening connections from original engram neurons to less-active downstream neurons — distributing the memory.

## REM sleep — what biology says

**Theta/gamma coherence** between hippocampus and neocortex facilitates cross-region exchange. **Competitive refinement/pruning:** REM raises network inhibition, only the most active excitatory neurons keep firing — competitive strengthening of strong traces, pruning of weak ones, distinct from NREM's broadcasting dynamic. **Emotional/schema integration:** REM is implicated in emotional memory consolidation and schema-level integration into existing knowledge. **Generative content (dreams):** vivid dream content is most reported during REM, often a creative recombination of recent + stored experience — dreaming's functional role is contested (byproduct to active rehearsal to creative-recombination engine).

## Mapping to somnus

| Biology (NREM/REM) | somnus (NREM-like/REM-like) |
|---|---|
| Replay via SWRs → cortex | Replay episodic → semantic, region by region |
| SHY: global synaptic downscaling | Homeostatic weight decay with selective protection of replay-reinforced weights |
| Engram broadcasting | Episode → region's semantic memory updates |
| REM theta-gamma coherence | Generative replay through semantic + recollection path |
| REM competitive refinement/pruning | Compression: quantize/prune/distill, spill to cold |
| Dream-style generative recombination | (Phase 3+) creative recombination of episodes for replay |
| — | **Cascade-correlation-style growth in REM-like phase** |

The last row is the **honest divergence**: mammalian REM doesn't add neurons; somnus's REM-like phase does (cascade-correlation addition where gist-recall fails) — an engineering choice, not a biological model. Using the offline window right after consolidation exposes which regions are saturated is simply a convenient time to grow. Adult neurogenesis happens in mammalian hippocampus but isn't sleep-stage-locked — somnus is freely re-purposing the REM phase.

## Sleep entry triggers — biology and somnus

Mammalian sleep regulation (Borbély, 1982): **Process S** (homeostatic — sleep pressure rises with wake duration, tracked via adenosine) and **Process C** (circadian, ~24h rhythm). somnus simulates neither directly — uses behavioral proxies: episodic memory passing a high-water mark (Process S analogue), wake-performance degradation on incoming inputs, schedule fallback (a loose, pragmatic Process C analogue).

## Replay in ML continual learning

ML CL has converged on replay as the most robust method class empirically (`continuous-learning.md`). Generative replay (Shin et al. 2017) explicitly cites the hippocampal generative role. Consistent ML/biology gap: biological replay has features ML usually doesn't replicate — time-compression, time-reversal, prioritization of high-novelty/high-reward sequences, replay of *non-experienced* plausible trajectories. Research opportunities for somnus over time.

## Open questions for somnus

- **Sleep-cycle ratio.** Mammals oscillate ~90-min NREM/REM cycles, NREM-heavy early, REM-heavy late — somnus has no such constraint, what's the right ratio here?
- **Total sleep fraction.** Mammals sleep 20-30%/day; somnus is unconstrained, could optimize for best wake performance.
- **Interruption cost.** Sleep deprivation impairs mammalian consolidation — what's the analogous somnus cost, and how much partial-sleep work is recoverable after interruption?
- **Creative recombination's value** (Phase 3+): useful generalization, or just noise?

## Hardware budget and time tradeoffs

> Tradeoffs supplement — biology/design rationale above unchanged. Hardware target: 14GB VRAM, 56GB RAM.

**Per-operation time estimates** (14GB VRAM GPU, ~10M-param cognitive layers, 10K recent episodes):

| Sleep operation | Estimated wall-clock | Notes |
|---|---|---|
| NREM: replay consolidation (10K episodes) | 20-50s | ~2-5ms/episode fwd+bwd |
| NREM: homeostatic downscaling (1 pass) | <1s | one multiply-add/param |
| NREM: compression + cold spill (100K vectors, INT8) | ~2-5min | includes NVMe write |
| REM: generative replay + self-probe (10K episodes) | 1-5min | reconstruction + cosine compare |
| REM: growth (1 unit/region, 4 regions) | **40-160min** | cascade-correlation candidate training |
| REM: HNSW index rebuild (if needed) | 5-30min | triggered by tombstone accumulation |
| **Total sleep cycle (no growth)** | ~10-20min | replay+homeostasis+probe+compression |
| **Total sleep cycle (with growth)** | **~1-4 hours** | growth dominates |

**Key insight (pre-implementation estimate — see `growth-mechanisms.md`'s empirical-validation section for what actually shipped and its very different real cost):** growth was projected as 1-2 orders of magnitude more expensive than everything else combined — sleep duration should be adaptive on whether growth triggers.

**NREM/REM time budget recommendation:** no growth → short sleep, 90% NREM (replay+homeostasis+compression) / 10% REM probe. Growth triggered → long sleep, 50% NREM / 50% REM with growth, capped at max-sleep-time. The mammalian NREM-heavy-early/REM-heavy-late pattern is a convenient default; somnus's right split depends on whether growth is needed, not circadian rhythm.

**Memory budget during sleep:** replay fwd+bwd ~200-500MB activation memory (frees per episode); growth candidate training ~<10MB/candidate (fine to run 20 in parallel if VRAM allows); self-probe reconstruction same as replay. Frozen encoders can swap to RAM during sleep if their VRAM is needed (~500MB transfer at ~16GB/s PCIe 4.0 ≈ 30ms, negligible).

**Sleep entry threshold calibration:** episodic high-water mark — warm/cold compression creates pressure sooner than VRAM fills (VRAM holds ~12M episodes; right trigger is probably ~10K-100K to limit per-cycle growth work). Wake-performance degradation — reconstruction error on recent inputs past a baseline, the most direct consolidation-needed signal. Schedule fallback — recommended max wake duration 30-60min of active experience, keeping per-cycle growth load manageable.

## Sources

- Tononi & Cirelli, 2003. "Sleep and synaptic homeostasis: a hypothesis." *Brain Research Bulletin*.
- Tononi & Cirelli, 2014. "Sleep and the price of plasticity." *Neuron*.
- McClelland, McNaughton, & O'Reilly, 1995. *Psychological Review*, 102: 419-457.
- Shin, Lee, Kim, & Kim, 2017. NeurIPS. arXiv:1705.08690.
- Borbély, 1982. "A two process model of sleep regulation." *Human Neurobiology* — conceptual framing only, not exhaustively reviewed.
- Reviews of NREM/REM functional split: pmc.ncbi.nlm.nih.gov review articles on systems memory consolidation during sleep.
