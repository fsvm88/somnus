# Brain-scale comparison and capacity-efficiency techniques

> **Contents:** Quantitative neuron/synapse comparison between the human brain and somnus's current implementation, what the compute-speed-vs-scale tradeoff actually buys, and a survey of capacity/compute-efficiency techniques not yet covered in other research notes (MoE/conditional routing, weight sharing, low-rank factorization, event-driven computation, adaptive computation time, local attention, multi-timescale processing, developmental-style pruning-after-growth). Cross-references existing notes for topics already covered in depth (growth, compression, sparse coding, EWC).
> **When to read:** Reasoning about long-term capacity scaling; deciding what to prioritize next among efficiency techniques; calibrating expectations for emergent capability given current scale.

Research note. Grew out of a 2026-07-06 conversation reasoning about how far a single-developer, single-GPU project can realistically get toward brain-like scale, and what techniques help close the gap per unit of effort. All numbers are **order-of-magnitude engineering estimates**, not precise counts — calibration, not precision.

---

## Quantitative comparison

**Human brain** (Herculano-Houzel's isotropic fractionator method is the standard modern neuron-count source; synapse estimates are less settled): ~86 billion neurons total (~16B cortical, ~69B cerebellar, mostly simple densely-packed granule cells); ~100-150 trillion synapses in cortex alone, whole-brain toward 10^15 once cerebellar granule-cell connectivity is included (cerebellum has the brain's highest synapse-per-neuron density).

**somnus, current implementation (2026-07-05 state)** — "neuron-equivalent units" = hidden+intermediate activation width × layer count for transformer/CNN components, actual unit count for sparse regions:

| Component | Basis | ≈ neuron-equivalent units | Params |
|---|---|---|---|
| Qwen3-0.6B (frozen action-decoder LM) | 28 layers × (1024 hidden + 3072 MLP) | ~114,700 | ~600M |
| Whisper-tiny (frozen, speech) | 8 layers × (384 + ~1536) | ~15,400 | ~39M |
| DINOv3-ViT-S (frozen, visual) | 12 layers × (384 + ~1536) | ~23,000 | ~21M |
| CLAP-HTSAT (frozen, audio-env) | Swin-tiny-scale audio encoder | ~15,000–20,000 | ~30M |
| somnus's own growable core: 5× k-WTA (n=1024,k=20) + GRU(256) + binding | direct count | ~5,400 | ~10–15M |
| **Total** | | **~175,000–180,000** | **~700M** |

Ratios: **~480,000× fewer neuron-equivalent units** than the brain (same order as a fruit fly's ~150k neurons); **~140,000× fewer connections** (total params vs. ~10^14 synapses). Isolating somnus's *own* growable machinery against the brain: **~6-7 million× fewer connections.**

### Does raw clock speed close the gap?

Silicon (~10^9 Hz) vs. biological neuron firing (~1-200 Hz) looks like a 10^7-10^9 advantage, but two things eat most of it: a biological synapse isn't a scalar multiply (dendritic nonlinearities, local plasticity, neuromodulatory gating do more work per "clock" than one float weight) — compute-equivalent brain estimates (Moravec 1997, Kurzweil 2005 — both contested, cited as order-of-magnitude anchors) land ~10^15-10^18 ops/sec, so clock speed buys back ~2-4 orders of magnitude, not the full 7-9. And emergent capability tracks **scale** (units/params/data), not clock speed — the empirical basis being LLM scaling laws (Kaplan et al. 2020, arXiv:2001.08361; Hoffmann et al. 2022/Chinchilla, arXiv:2203.15556). A fast fruit-fly-scale network is still fruit-fly-scale, just quicker.

**Implication for somnus:** the ~5-6 order-of-magnitude gap won't close through hardware speed — only through (a) the growth mechanism actually shipping so the growable core stops being frozen at ~5,400 units, and (b) algorithmic efficiency techniques extracting more effective capacity per unit/param than dense architectures — out-scaling frontier labs in raw count isn't on the table for a single-GPU project.

**On consciousness specifically:** neuron/synapse count is a weak proxy. Theories quantifying it more principledly — e.g. Integrated Information Theory's Φ (Tononi et al., *Nature Reviews Neuroscience* 2016) — are about integrated causal structure, not raw count; a honeybee (~1M neurons) plausibly has richer integrated behavior than a larger shallowly-connected network. This comparison calibrates expected *scale of capability*, not consciousness directly.

---

## Efficiency techniques already covered elsewhere (link, don't duplicate)

Growth mechanism (cascade-correlation, GNG, online cloning) — `growth-mechanisms.md`. Memory quantization/compression (INT8 ✓, PQ ✓ at 27.5×, structured/magnitude pruning, distillation) — `memory-compression.md`. Sparse coding (k-WTA, top-k, SAE/L1, BatchTopK) — `sparse-coding.md`, `gaps.md` §1. Elastic Weight Consolidation — `continuous-learning.md`. REM competitive pruning (biological "grow then prune" precedent) — `sleep-and-consolidation.md`. Vector retrieval at scale (HNSW, IVF-PQ, flat) — `vector-retrieval.md`. Replay prioritization (uniform, recency-weighted, PER, reservoir) — `replay-strategies.md`.

## New candidate techniques (not yet documented)

**Conditional computation/MoE routing.** A gating network runs a small sub-network subset per input instead of all of them — the formalized version of what somnus's 5 regions already do informally (each fires only on its relevant modality); the win compounds as more specialized sub-networks are added later without paying full cost every step. Shazeer et al. 2017, arXiv:1701.06538 (foundational MoE, a handful of experts activate per token); Fedus, Zoph, Shazeer 2021, arXiv:2101.03961 (Switch Transformers, top-1 routing — relevant if binding ever needs to pick among >5 regions). **somnus implication:** speculative, pays off once region count grows past what informal per-modality routing handles cleanly.

**Weight sharing across regions.** A single "canonical microcircuit" reused across regions with different inputs, instead of 5 independently-parameterized k-WTA layers — biology does this structurally (near-identical neocortical column circuitry across modalities, Mountcastle 1978, "An organizing principle for cerebral function," *The Mindful Brain*, MIT Press). **somnus implication:** speculative, worth a small experiment — would cut connection count directly if the 5 regions don't need fully independent parameterization at n=1024,k=20.

**Low-rank factorization of region/binding weights.** Express a dense N×M matrix as two smaller matrices (rank r≪min(N,M)) if effective rank is lower than dimensions — free reduction if true. Hu et al. 2021, LoRA, arXiv:2106.09614 (framed for fine-tuning, but the low-rank-approximates-a-trained-matrix argument applies to any dense weight). **somnus implication:** directly testable via SVD on trained region matrices, checking singular-value decay. **Checked 2026-07-06 — null result** (see Summary judgment below).

**Event-driven/spiking-style computation.** Recompute only on change, not a fixed clock — biological neurons are mostly silent, computation happens on spikes, the single biggest efficiency mechanism real neural tissue uses vs. dense per-frame ANN recomputation. Roy, Jaiswal, Panda 2019, *Nature* 575:607-617 (spiking NN/neuromorphic hardware survey, e.g. Intel Loihi — hardware-software co-design, not immediately actionable on the current ORT/Rust stack). **somnus implication:** a cheap non-speculative version is available now without spiking hardware — a "did this frame/chunk change enough" gate before encoder inference, skipping ORT entirely on near-static input. **Implemented 2026-07-06** (see Summary judgment below).

**Adaptive computation time/early exit.** Stop propagating once a confidence threshold is met, rather than always running full depth — engineering analog to the brain's fast subcortical shortcuts (amygdala threat response bypassing full cortical processing) and System-1/System-2 cognition. Graves 2016, arXiv:1603.08983 (original ACT, a per-step halting unit). **somnus implication:** speculative, most applicable to the action decoder's autoregressive loop, where a confident early token distribution could skip remaining layers.

**Local/sparse attention in the binding area.** Restrict cross-attention to spatially/temporally nearby content instead of full all-to-all, mirroring retinotopic/tonotopic local cortical connectivity (nearby locations represent nearby sensory space, long-range connections are the exception). Beltagy, Peters, Cohan 2020, Longformer, arXiv:2004.05150 (local-plus-global attention, reduces O(n²) cost). **somnus implication:** speculative, moot at 5 regions (attention cost trivial), relevant if region count grows or per-region content gets tokenized into multiple sub-vectors.

**Multi-timescale processing.** Different sub-networks/clocks for fast sensory vs. slow conceptual processing, avoiding full-network cost every tick for slowly-changing content — loose analog to cortical hierarchies operating at different oscillatory bands (theta ~4-8Hz episodic/contextual binding, gamma ~30-80Hz local sensory). Kiebel, Daunizeau, Friston 2008, *PLoS Computational Biology* 4(11):e1000209. **somnus implication:** speculative, most relevant to binding's GRU (currently updates every episode) — a slower-updating context layer (every N episodes) could reduce redundant computation without losing temporal binding.

---

## Summary judgment

Best (effort, payoff) ratio for a single-developer project, in priority order:

1. **Change-gated encoder inference** (event-driven, no architecture change) — cheapest, directly cuts wake-path compute, no research risk. ✅ Implemented 2026-07-06, video path only (audio silence-gating is a separate, less contained problem, left for a future pass) — `gaps.md` §13. Opt-in (`--video-change-threshold`, default 0.0/disabled); wake-path CPU savings not yet A/B'd.
2. **Post-training SVD check on region/binding weights** (low-rank factorization) — a half-day experiment settling whether free connection reduction is on the table. ✅ Checked 2026-07-06 — **null result**: all 21 region/binding matrices near-full-rank (mean r95/full_rank=0.772) on the most-trained checkpoint available — not pursued. `gaps.md` §14.

Everything else in "new candidate techniques" is speculative and lower-priority until region count or binding complexity grows past what the current design handles — per CLAUDE.md's "don't design for hypothetical future requirements."
