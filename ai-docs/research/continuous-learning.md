# Continuous learning

> **Contents:** CL families (replay/rehearsal, regularization, parameter isolation, growth), CLS dual-memory theory, where somnus's approach sits, hardware budget tradeoffs, and open failure modes.
> **When to read:** Designing the sleep-cycle learning update; evaluating catastrophic forgetting risk; understanding why CLS+replay+growth was chosen.

Literature behind somnus's continuous-learning design. Commitments live in `vision.md` (1, 5); operationalization in `architecture.md`. Records what the literature says, what it doesn't, and where somnus diverges.

## Summary of somnus's approach

**CLS-style dual memory + replay-driven consolidation + cascade-correlation-style growth + sparse codes**, with frozen pretrained encoders beneath. Only the cognitive layers (regional concept areas + integrative area + semantic memory) learn continually; the encoder stack is fixed.

## Families of continuous-learning methods

**Replay/rehearsal.** Store past experience, re-present during learning — experience replay (raw buffered samples) or generative replay (a trained generator resynthesizes past data, bounding storage). Foundational: Shin, Lee, Kim, & Kim, 2017, "Deep Generative Replay" (NeurIPS, arXiv:1705.08690) — generator (originally GAN) + solver, generator produces past-like samples interleaved with new data; cites the hippocampus's generative role. **somnus use:** episodic memory is the raw-replay buffer during sleep; generative replay enters as scale grows and storage tightens, reusing the same generator for REM self-probing.

**Regularization (weight protection).** Penalize weight changes damaging prior learning, via per-weight importance. EWC (Kirkpatrick et al. 2017, PNAS — quadratic penalty toward old values, Fisher-Information importance, inspired by neocortical synaptic consolidation), Synaptic Intelligence (Zenke, Poole, Ganguli 2017, ICML), Memory Aware Synapses (Aljundi et al. 2018, ECCV). **somnus use:** not central — these need task IDs and full-dataset retraining, a poor fit for task-free streaming. A weak per-region decay does appear (homeostatic downscaling, `sleep-and-consolidation.md`), but replay+growth is the primary forgetting-resistance mechanism.

**Architecture — parameter isolation.** Fresh parameters per task, old ones frozen (Progressive Networks, Rusu et al. 2016; PackNet, Mallya & Lazebnik 2018). **somnus use:** none — these assume discrete task IDs; somnus has one continuous unlabeled stream.

**Architecture — growth (somnus's core capacity mechanism).** Add capacity as data arrives. Cascade-Correlation (Fahlman & Lebiere 1990, NIPS) — adds hidden units one at a time trained to correlate with current residual error, frozen input weights post-insertion, no task IDs needed. Growing Neural Gas (Fritzke 1994, NIPS) — topological relations via Hebbian updates + Competitive Hebbian Learning, no time-decaying params, continuous learning. Dynamically Expandable Networks (Yoon et al. 2018) — task-aware growth. **somnus use:** cascade-correlation-style addition is the planned per-region growth mechanism — REM self-probe finds gist-recall failing, new units train on the region's residual reconstruction error, task-free by construction. GNG informs the topological-learning side; may borrow its competitive-Hebbian connection-pruning for the recollection-layer index later.

**Sparse/orthogonal codes.** Sparse representations so different concepts use different units, minimizing interference algorithmically — HTM (Hawkins/Numenta, sparse distributed representations), k-WTA layers, sparse autoencoders, learned sparsity penalties. **somnus use:** regional concept areas are sparsely-coded — sparsity makes growth local (new units occupy different sparsity patterns, reducing interference with consolidated knowledge). Exact mechanism was an open question at the time this note was written (now resolved: k-WTA, ADR 0005).

**Dual-memory/Complementary Learning Systems (CLS).** Foundational: McClelland, McNaughton, & O'Reilly, 1995, *Psychological Review* 102:419-457. Hippocampus = sparse, pattern-separated, fast one-shot episode learning; neocortex = distributed, overlapping, slow cross-episode integration extracting statistical structure; memories store hippocampally first, offline reinstatement produces small neocortical synaptic changes, remote memory accumulates over many reinstatements — avoids the catastrophic interference rapid neocortical integration would cause. **somnus use:** episodic memory ≈ hippocampus (fast, one-shot, replay source), semantic memory ≈ neocortex (slow, generalist, growable), sleep-time replay is the consolidation channel — CLS is the architectural template the rest of the system extends.

**Meta-continual learning.** "Learn to learn without forgetting" — needs substantial pre-experience, the opposite of a baby-brain start. **Not used.**

## What's solved vs. open

**Solid:** replay works — the most empirically robust CL family; CLS is well-supported architecturally. **Active/open:** task-free online budget-bounded growth (most growing-network methods are task-aware); generative-replay drift over many cycles; sparse-codes+growth interaction (adding units while preserving sparsity isn't a settled methodology).

## Where somnus diverges from the literature

Most literature assumes fixed capacity + a discrete-task curriculum; somnus is fixed-encoder + growable-cognitive-layers on one continuous unlabeled stream. Most CL benchmarks (Permuted MNIST, Split CIFAR) are task-discrete — somnus's evaluation needs to be different, likely streaming benchmarks built around the system's own self-probe.

## Hardware budget and performance tradeoffs

> Tradeoffs supplement — theory above unchanged. Hardware target: 14GB VRAM, 56GB RAM.

**What fits at what scale:**

| Component | Params (est.) | FP16 VRAM |
|---|---|---|
| Per region (384-dim input, 2000 units) | 384×2000×2=1.5M | 3MB |
| 4 regions total | 6M | 12MB |
| Binding area (cross-attention, 4 regions) | ~4M | 8MB |
| **Total cognitive layers** | **~10M** | **~20MB** |

Cognitive layers are tiny relative to 14GB VRAM — growing 10× (20K units/region, ~100M params) would still cost only ~200MB, well within budget. The ceiling on growth is **compute during sleep** (time to train new units), not memory.

**CL performance:** experience replay — one fwd+bwd pass/replayed episode, 10M-param net batch=1 ~2-5ms/episode (est.), 10K episodes/cycle ~20-50s GPU — **not the bottleneck.** Cascade-correlation growth — one unit ~10-40min GPU (see `growth-mechanisms.md`), 10 units across all regions/sleep ~100-400min — **this was projected to be the bottleneck** (later found not to hold in practice — the shipped mechanism is a seeded resize, not candidate-pool training; see `growth-mechanisms.md`'s empirical-validation section). **Forgetting rate without replay:** standard catastrophic forgetting ~30-60% loss of earlier-task performance after a new task, no replay (Kirkpatrick et al. 2017); with sparse codes (2-5%, 2000 units) expected inter-concept overlap ~0.4 units/pair → interference/update ~0.02% of stored patterns, dramatically reduced; with replay covering recent N episodes, forgetting drops near zero for recently-replayed content, older unreplayed content decays at the homeostatic rate (~1-5%/cycle, an open hyperparameter).

**Catastrophic forgetting detection:** same self-probe as the growth trigger — if gist-recall of *old* episodes (not just recent) drops, forgetting has outpaced replay's repair rate. Monitor as a system health metric from Phase 3 on. Cost: one forward pass/probed episode, ~2-5s for a 1K random sample — run once per sleep cycle in the self-probe phase.

## Sources

- Fahlman & Lebiere, 1990. "The Cascade-Correlation Learning Architecture." NIPS.
- Fritzke, 1994. "A Growing Neural Gas Network Learns Topologies." NIPS.
- Kirkpatrick et al., 2017. "Overcoming catastrophic forgetting in neural networks." *PNAS*. arXiv:1612.00796.
- McClelland, McNaughton, & O'Reilly, 1995. *Psychological Review*, 102: 419-457.
- Shin, Lee, Kim, & Kim, 2017. "Continual Learning with Deep Generative Replay." NeurIPS. arXiv:1705.08690.
- Aljundi et al., 2018. "Memory Aware Synapses." ECCV.
- Zenke, Poole, & Ganguli, 2017. "Continual Learning Through Synaptic Intelligence." ICML.
- Rusu et al., 2016. "Progressive Neural Networks." arXiv:1606.04671.
- Mallya & Lazebnik, 2018. "PackNet." CVPR.
- Yoon et al., 2018. "Lifelong Learning with Dynamically Expandable Networks." ICLR.
