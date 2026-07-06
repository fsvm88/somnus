# Brain-scale comparison and capacity-efficiency techniques

> **Contents:** Quantitative neuron/synapse comparison between the human brain and somnus's current implementation, what the compute-speed-vs-scale tradeoff actually buys, and a survey of capacity/compute-efficiency techniques not yet covered in other research notes (MoE/conditional routing, weight sharing, low-rank factorization, event-driven computation, adaptive computation time, local attention, multi-timescale processing, developmental-style pruning-after-growth). Cross-references existing notes for topics already covered in depth (growth, compression, sparse coding, EWC).
> **When to read:** Reasoning about long-term capacity scaling; deciding what to prioritize next among efficiency techniques; calibrating expectations for emergent capability given current scale.

Research note for somnus. Grew out of a 2026-07-06 conversation reasoning about how far a single-developer, single-GPU project can realistically get toward brain-like scale, and what techniques (biological or engineering) help close the gap per unit of effort. All numbers are **order-of-magnitude engineering estimates**, not precise counts — the goal is calibration, not precision.

---

## Quantitative comparison

**Human brain** (Herculano-Houzel's isotropic fractionator method is the standard modern source for the neuron count; synapse estimates are less settled):

- ~86 billion neurons total (~16B cortical, ~69B cerebellar — mostly granule cells, structurally simple and densely packed).
- ~100–150 trillion synapses in cortex alone; whole-brain estimates run toward 10^15 once cerebellar granule-cell connectivity is included (cerebellum has the highest synapse-per-neuron density in the brain).

**somnus, current implementation (2026-07-05 state)**, counting "neuron-equivalent units" as hidden+intermediate activation width × layer count for transformer/CNN components, and actual unit count for the sparse regions:

| Component | Basis | ≈ neuron-equivalent units | Params |
|---|---|---|---|
| Qwen3-0.6B (frozen action-decoder LM) | 28 layers × (1024 hidden + 3072 MLP) | ~114,700 | ~600M |
| Whisper-tiny (frozen, speech) | 8 layers × (384 + ~1536) | ~15,400 | ~39M |
| DINOv3-ViT-S (frozen, visual) | 12 layers × (384 + ~1536) | ~23,000 | ~21M |
| CLAP-HTSAT (frozen, audio-env) | Swin-tiny-scale audio encoder | ~15,000–20,000 | ~30M |
| somnus's own growable core: 5× k-WTA regions (n=1024, k=20 active) + GRU (256) + binding | direct count | ~5,400 | ~10–15M |
| **Total** | | **~175,000–180,000** | **~700M** |

Ratios: **~480,000× fewer neuron-equivalent units** than the brain (in the same order as a fruit fly's ~150k neurons); **~140,000× fewer connections** (total params vs. ~10^14 synapses). Isolating just somnus's *own* growable machinery (not the frozen encoders/LM riding on top) against the brain: **~6–7 million× fewer connections**.

### Does raw clock speed close the gap?

Silicon clock speed (~10^9 Hz) vs. biological neuron firing rate (~1–200 Hz) looks like a 10^7–10^9 advantage, but two things eat most of it:

- A biological synapse is not a scalar multiply — dendritic nonlinearities, local plasticity, and neuromodulatory gating mean a single synapse does more computational work per "clock" than a single float weight. Compute-equivalent estimates for the brain (Moravec 1997; Kurzweil 2005 — both contested, cited here as order-of-magnitude anchors, not settled science) land around 10^15–10^18 ops/sec, meaning clock speed buys back roughly 2–4 orders of magnitude, not the full 7–9.
- Emergent capability in both biological and artificial networks tracks **scale** (units/params/data), not clock speed — the empirical basis being LLM scaling laws (Kaplan et al. 2020, "Scaling Laws for Neural Language Models," arXiv:2001.08361; Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" / Chinchilla, arXiv:2203.15556). A fast fruit-fly-scale network is still a fruit-fly-scale network, just a quick one.

**Implication for somnus:** the ~5–6 order-of-magnitude gap will not close through hardware speed. It only narrows through (a) the growth mechanism actually shipping, so the growable core stops being frozen at its current ~5,400 units, and (b) algorithmic efficiency techniques that extract more effective capacity per unit/parameter than dense architectures do — since out-scaling frontier labs in raw count is not on the table for a single-GPU project.

**On consciousness specifically:** neuron/synapse count is a weak proxy. Theories that attempt to quantify it in a principled way — e.g. Integrated Information Theory's Φ (Tononi et al., "Integrated information theory: from consciousness to its physical substrate," *Nature Reviews Neuroscience* 2016) — are about integrated causal structure, not raw count. A honeybee (~1M neurons) plausibly has richer integrated behavior than a larger but shallowly-connected network. This comparison calibrates expected *scale of capability*, not consciousness directly.

---

## Efficiency techniques already covered elsewhere (don't duplicate — link)

- **Growth mechanism** (cascade-correlation, GNG, online cloning) — [`growth-mechanisms.md`](growth-mechanisms.md).
- **Memory quantization & compression** (INT8 ✓ implemented, Product Quantization ✓ implemented at 27.5× reduction, structured/magnitude pruning, distillation) — [`memory-compression.md`](memory-compression.md).
- **Sparse coding** (k-WTA, top-k, SAE/L1, BatchTopK) — [`sparse-coding.md`](sparse-coding.md), [`gaps.md`](../gaps.md) §1.
- **Elastic Weight Consolidation** (protect important weights instead of only growing new units) — [`continuous-learning.md`](continuous-learning.md).
- **REM competitive pruning** (biological precedent for "grow then prune") — [`sleep-and-consolidation.md`](sleep-and-consolidation.md).
- **Vector retrieval at scale** (HNSW, IVF-PQ, flat search) — [`vector-retrieval.md`](vector-retrieval.md).
- **Replay prioritization** (uniform, recency-weighted, PER, reservoir sampling) — [`replay-strategies.md`](replay-strategies.md).

## New candidate techniques (not yet documented)

### Conditional computation / Mixture-of-Experts routing

A gating network selects a small subset of sub-networks to run per input, rather than running all of them every step. This is the formalized version of what somnus's 5 regions already do informally (each region only fires on its relevant modality). The win compounds as more specialized sub-networks are added later without paying full cost on every step.

- Shazeer et al., 2017. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." arXiv:1701.06538. Foundational MoE routing paper; only a handful of experts activate per token.
- Fedus, Zoph, Shazeer, 2021. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." arXiv:2101.03961. Simplified top-1 routing; directly relevant if somnus's binding area ever needs to pick among more than 5 regions.

**somnus implication:** speculative — only pays off once the region count grows past what informal per-modality routing handles cleanly.

### Weight sharing across regions

A single "canonical microcircuit" parameterization reused across regions with different inputs, instead of 5 independently-parameterized k-WTA layers. Biology does this structurally — neocortical columns are near-identical in local circuitry across sensory modalities (Mountcastle, 1978, "An organizing principle for cerebral function: The unit module and the distributed system," in *The Mindful Brain*, MIT Press — the original "canonical cortical column" argument).

**somnus implication:** speculative, worth a small experiment — would directly cut connection count if the 5 regions turn out not to need fully independent parameterizations at n=1024, k=20.

### Low-rank factorization of region/binding weights

Express a dense N×M weight matrix as two smaller matrices (rank r ≪ min(N,M)) when the effective rank of the learned matrix is lower than its dimensions — free connection reduction if true.

- Hu et al., 2021. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09614. Framed for fine-tuning, but the underlying low-rank-approximates-a-trained-matrix argument applies to any dense weight, not just adapters.

**somnus implication:** speculative — testable directly: SVD somnus's existing region encoder/decoder matrices post-training and check singular value decay; if it falls off fast, low-rank reparameterization is close to free.

### Event-driven / spiking-style computation

Recompute only on change, not on a fixed clock. Biological neurons are mostly silent — computation happens on spikes, and only where/when there's a spike, which is the single biggest efficiency mechanism real neural tissue uses (vs. dense per-frame recomputation in standard ANNs).

- Roy, Jaiswal, Panda, 2019. "Towards spiking neural networks: A survey." *Nature* 575, 607–617. Survey of spiking NN approaches and neuromorphic hardware (e.g., Intel Loihi) — a hardware-software co-design direction, not immediately actionable on the current ORT/Rust stack but worth tracking.

**somnus implication:** a cheap, non-speculative version is available *now* without spiking hardware: a "did this frame/chunk change enough to warrant re-encoding" gate in front of the audio/video encoders, skipping ORT inference entirely on near-static input. Directly reduces wake-path compute.

### Adaptive computation time / early exit

Stop propagating through a network once a confidence threshold is met, rather than always running the full depth. Direct engineering analog to the brain's fast subcortical shortcuts (e.g., amygdala threat response bypassing full cortical processing) and to dual-process (System-1/System-2) accounts of cognition.

- Graves, 2016. "Adaptive Computation Time for Recurrent Neural Networks." arXiv:1603.08983. Original ACT mechanism — a per-step halting unit.

**somnus implication:** speculative — most applicable to the action-decoder's autoregressive generation loop, where a confident early token distribution could skip remaining layers.

### Local / sparse attention in the binding area

Restrict the binding area's cross-attention to spatially/temporally nearby content instead of full all-to-all, mirroring retinotopic/tonotopic local connectivity in real cortex (nearby cortical locations represent nearby sensory space; long-range connections are the exception, not the rule).

- Beltagy, Peters, Cohan, 2020. "Longformer: The Long-Document Transformer." arXiv:2004.05150. Local-plus-global attention pattern reducing O(n²) attention cost.

**somnus implication:** speculative — currently moot at 5 regions (attention cost is trivial), but becomes relevant if region count grows significantly, or if per-region content is ever tokenized into multiple sub-vectors rather than one embedding per region.

### Multi-timescale processing

Different sub-networks/clocks for fast sensory vs. slow conceptual processing, avoiding full-network cost on every tick for content that changes slowly. Loose analog to cortical processing hierarchies operating at different oscillatory bands (theta ~4–8 Hz for episodic/contextual binding, gamma ~30–80 Hz for local sensory processing).

- Kiebel, Daunizeau, Friston, 2008. "A hierarchy of time-scales and the brain." *PLoS Computational Biology* 4(11):e1000209. Predictive-processing account of cortical hierarchies operating at different temporal scales.

**somnus implication:** speculative — most relevant to the binding area's GRU, which currently updates every episode; a slower-updating "context" state layered on top of it (updating every N episodes) could reduce redundant computation without losing temporal binding.

---

## Summary judgment

Of the above, the two with the best (effort, payoff) ratio for a single-developer project, in priority order:

1. **Change-gated encoder inference** (event-driven, no architecture change) — cheapest to implement, directly cuts wake-path compute, no research risk. ✅ Implemented 2026-07-06 for the video path only (audio silence-gating is a different, less contained problem, left for a separate pass if warranted) — see `ai-docs/gaps.md` §13 for the implementation. Deployed opt-in (`--video-change-threshold`, default 0.0/disabled); actual wake-path CPU savings not yet A/B'd.
2. **Post-training SVD check on region/binding weights** (low-rank factorization) — a half-day experiment that tells you immediately whether there's free connection reduction on the table.

Everything else in the "new candidate techniques" section is speculative and lower-priority until region count or binding complexity actually grows past what the current design handles — consistent with CLAUDE.md's "don't design for hypothetical future requirements."
