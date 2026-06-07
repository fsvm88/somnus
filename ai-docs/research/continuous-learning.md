# Continuous learning

> **Contents:** CL families (replay/rehearsal, regularization, parameter isolation, growth), CLS dual-memory theory, where somnus's approach sits, hardware budget tradeoffs, and open failure modes.  
> **When to read:** Designing the sleep-cycle learning update; evaluating catastrophic forgetting risk; understanding why CLS+replay+growth was chosen.

Literature behind somnus's continuous-learning design. The commitments live in `vision.md` (commitments 1, 5); the operationalization is in `architecture.md`. This note records what the literature actually says, what it doesn't say, and where somnus diverges.

## Summary of somnus's approach

**CLS-style dual memory + replay-driven consolidation + cascade-correlation-style growth + sparse codes**, with frozen pretrained encoders beneath. Only the cognitive layers (regional concept areas + integrative area + semantic memory) learn continually; the encoder stack is fixed.

## Families of continuous-learning methods

### Replay / rehearsal

Store past experience, re-present during learning. Two flavors:

- **Experience replay** — keep raw samples in a buffer.
- **Generative replay** — train a generator to *resynthesize* past data, so storage is bounded.

Foundational ML reference: **Deep Generative Replay**, Shin, Lee, Kim, & Kim, 2017 (NeurIPS, arXiv:1705.08690). A "generator" (originally a GAN) plus a "solver" model; the generator produces past-like samples interleaved with new-task data. Explicitly cites the hippocampus's generative role.

**somnus use:** episodic memory is the buffer for raw experience replay during sleep. Generative replay enters as scale grows and storage tightens; the same generator is reused for self-probing during REM-like sleep.

### Regularization (weight protection)

Penalize weight changes that damage prior learning; per-weight importance estimates slow updates on important weights.

- **Elastic Weight Consolidation (EWC)** — Kirkpatrick et al., 2017, *PNAS*. Quadratic penalty pulling weights toward old values; importance via the Fisher Information Matrix. Biologically inspired by neocortical synaptic consolidation.
- **Synaptic Intelligence (SI)** — Zenke, Poole, Ganguli, 2017, ICML.
- **Memory Aware Synapses (MAS)** — Aljundi et al., 2018, ECCV.

**somnus use:** not the central tool — these methods typically require task IDs and full retraining on whole datasets, which doesn't fit a task-free streaming setting. A weak per-region weight decay does appear during sleep (homeostatic downscaling, see `sleep-and-consolidation.md`), but the primary forgetting-resistance mechanism is replay + growth.

### Architecture — parameter isolation

Allocate fresh parameters per task; freeze old ones.

- Progressive Networks — Rusu et al., 2016.
- PackNet — Mallya & Lazebnik, 2018.

**somnus use:** none. These methods assume discrete tasks with identifiers; we have one continuous unlabeled stream.

### Architecture — growth (the core somnus mechanism for capacity expansion)

Add capacity to the network as data comes in.

- **Cascade-Correlation** — Fahlman & Lebiere, 1990 (NIPS). Adds hidden units one at a time; each new unit is trained to correlate with the network's current residual error; once added, the unit's input weights are frozen and outputs cascade forward. Crucially, no task IDs required.
- **Growing Neural Gas (GNG)** — Fritzke, 1994 (NIPS). Incremental network learning topological relations via Hebb-like weight updates and Competitive Hebbian Learning for connections. No time-decaying parameters; can learn continuously.
- **Dynamically Expandable Networks (DEN)** — Yoon et al., 2018. Task-aware growth.

**somnus use:** cascade-correlation-style addition is the planned growth mechanism for the semantic memory of each region — when REM-like self-probe finds gist-level recall failing, new units are trained on the residual reconstruction error of the existing units in that region. Task-free by construction. GNG is a good reference for the topological-learning side; we may borrow its competitive-Hebbian connection-pruning idea for the recollection-layer index later.

### Sparse / orthogonal codes

Make representations sparse so different concepts use different units, minimizing interference at the algorithmic level.

- HTM (Hierarchical Temporal Memory) — Hawkins / Numenta lineage. Strong commitment to sparse distributed representations.
- k-WTA layers, sparse autoencoders, learned sparsity penalties.

**somnus use:** regional concept areas use sparsely-coded continuous embeddings. Sparsity makes growth local — new units occupy different sparsity patterns from existing ones, reducing interference with consolidated knowledge. The exact sparsity mechanism is an open architectural question.

### Dual-memory / Complementary Learning Systems (CLS)

Foundational paper: **McClelland, McNaughton, & O'Reilly, 1995**, "Why there are complementary learning systems in the hippocampus and neocortex...", *Psychological Review*, 102: 419–457.

The framework:

- **Hippocampus** = sparse, pattern-separated, fast one-shot learning of episodes.
- **Neocortex** = distributed, overlapping, slow integration across episodes; extracts statistical structure.
- Memories are first stored hippocampally; reinstatement during offline periods produces small synaptic changes in neocortex; remote memory accumulates over many reinstatements.
- Avoids catastrophic interference — rapid integration into neocortex would damage existing structured knowledge.

**somnus use:** episodic memory ≈ hippocampus (fast, one-shot, source of replay). Semantic memory ≈ neocortex (slow, generalist, growable). Sleep-time replay is the consolidation channel between them. CLS is the architectural template the rest of the system extends.

### Meta-continual learning

"Learn to learn without forgetting" — meta-learning applied to CL. Requires substantial pre-experience, which is the opposite of a baby brain. **Not used.**

## What's solved vs. open

**Solid:**
- Replay works. It's the most robust family of CL methods empirically.
- CLS is well-supported as an architectural pattern.

**Active:**
- **Task-free, online, budget-bounded growth** — what we want. Most growing-network methods are task-aware.
- **Generative-replay drift** — over many cycles, the generator can degrade; how to keep it honest.
- **Sparse-codes + growth interaction** — adding units while preserving sparsity is not a settled methodology.

## Where somnus diverges from the literature

- The literature mostly assumes a fixed-capacity network and a curriculum of discrete tasks. somnus is fixed-encoder + growable-cognitive-layers operating on a single continuous stream with no task labels.
- Most CL benchmarks (Permuted MNIST, Split CIFAR, etc.) are task-discrete. somnus's evaluation will need to be different — likely streaming benchmarks built around the system's own self-probe.

## Hardware budget and performance tradeoffs

> This section was added as a tradeoffs supplement. The theory above is unchanged.
>
> Hardware target: 14 GB VRAM, 56 GB RAM.

### What fits at what scale

Regional concept areas (sparse-coded, SAE or k-WTA) hold the learnable part of the system. Encoder weights are frozen and much larger. Budget analysis:

| Component | Params (est.) | FP16 VRAM |
|-----------|--------------|-----------|
| Per region (384-dim input, 2000 units) | 384×2000×2 = 1.5M | 3 MB |
| 4 regions total | 6M | 12 MB |
| Binding area (cross-attention, 4 regions) | ~4M | 8 MB |
| **Total cognitive layers** | **~10M** | **~20 MB** |

The cognitive layers are tiny relative to the 14 GB VRAM budget. Growing them 10× (to 20K units per region, ~100M cognitive params) would still cost only ~200 MB VRAM — well within budget. The ceiling on growth is **compute during sleep** (time to train new units), not memory.

### CL performance: forgetting and replay cost

**Experience replay during NREM-like sleep:**
- One forward + backward pass through the cognitive layers per replayed episode.
- 10M-param network, batch=1: ~2–5 ms/episode on GPU (est.).
- 10K episodes replayed per sleep cycle: ~20–50 s of GPU time.
- **Not the bottleneck.**

**Growth via cascade-correlation:**
- Adding one unit: candidate training + output weight update = ~10–40 min GPU (see `growth-mechanisms.md`).
- Adding 10 units across all regions per sleep: ~100–400 min GPU.
- **This is the bottleneck.** Growth time dominates the sleep budget, not replay.

**Forgetting rate without replay:**
- Standard catastrophic forgetting: a fixed-capacity network loses ~30–60% of earlier-task performance after a new task with no replay (Kirkpatrick et al., 2017).
- With sparse codes (2–5% sparsity, 2000 units): expected inter-concept overlap ~0.4 units per concept pair → interference per update ~0.02% of all stored patterns — dramatically reduced.
- With replay covering recent N episodes: forgetting rate drops near zero for recently replayed content; older unreplayed content decays at the homeostatic downscaling rate (~1–5% per sleep cycle — an open hyperparameter).

### Catastrophic forgetting detection

Forgetting is detectable via the same self-probe used for growth triggering: if gist-level recall of *old* episodes (not just recent ones) drops, the forgetting rate has exceeded the repair rate of replay. This should be monitored as a system health metric from Phase 3 onward.

**Compute cost of forgetting probe:** same as growth trigger — one forward pass per probed episode. Probing a random sample of 1K old episodes: ~2–5 s. Run this once per sleep cycle as part of the self-probe phase.

## Sources

- Fahlman & Lebiere, 1990. "The Cascade-Correlation Learning Architecture." NIPS.
- Fritzke, 1994. "A Growing Neural Gas Network Learns Topologies." NIPS.
- Kirkpatrick et al., 2017. "Overcoming catastrophic forgetting in neural networks." *PNAS*. arXiv:1612.00796.
- McClelland, McNaughton, & O'Reilly, 1995. *Psychological Review*, 102: 419–457.
- Shin, Lee, Kim, & Kim, 2017. "Continual Learning with Deep Generative Replay." NeurIPS. arXiv:1705.08690.
- Aljundi et al., 2018. "Memory Aware Synapses." ECCV.
- Zenke, Poole, & Ganguli, 2017. "Continual Learning Through Synaptic Intelligence." ICML.
- Rusu et al., 2016. "Progressive Neural Networks." arXiv:1606.04671.
- Mallya & Lazebnik, 2018. "PackNet." CVPR.
- Yoon et al., 2018. "Lifelong Learning with Dynamically Expandable Networks." ICLR.
