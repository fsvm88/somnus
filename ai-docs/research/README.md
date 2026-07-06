# research/

> **Contents:** Topic-organized research notes — theory (continuous learning, sleep, predictive coding), component engineering (encoders, ML runtimes, vector retrieval, sparse coding, memory compression), and algorithms (growth, replay, binding). Each file cites primary sources and distinguishes findings from somnus-specific implications.  
> **When to read:** Looking for the evidence behind a design choice; finding alternatives that were considered.

Topic-organized research notes that inform design decisions.

## Conventions

- One file per topic (e.g., `predictive-coding.md`, `replay-and-consolidation.md`, `catastrophic-forgetting.md`). Group per topic, not per source.
- Lead with a one-paragraph summary: what is this, and why do we care?
- Cite primary sources: paper title, authors, year, link, one-line takeaway.
- Distinguish *findings* (what the literature says) from *implications* (what it might mean for somnus). Mark speculation as such.
- Negative results count. If we tried something and it did not work, write it up so we don't repeat the experiment.

## Naming

`kebab-case.md`, named for the topic.

## Current notes

### Theory / design rationale
- [`brain-scale-comparison.md`](brain-scale-comparison.md) — quantitative neuron/synapse comparison (brain vs. somnus), compute-speed-vs-scale tradeoff, survey of capacity-efficiency techniques (MoE routing, weight sharing, low-rank factorization, event-driven computation, adaptive computation time, local attention, multi-timescale processing).
- [`continuous-learning.md`](continuous-learning.md) — CL families (replay, regularization, growth, CLS), where somnus sits, + hardware budget tradeoffs.
- [`sleep-and-consolidation.md`](sleep-and-consolidation.md) — NREM/REM biology, mapping to somnus's two-stage cycle, + sleep time/memory budget estimates.
- [`predictive-coding.md`](predictive-coding.md) — Friston FEP / PC framework; partial adoption rationale; what it costs to implement fully vs. what somnus already borrows.

### Component engineering (memory / latency / compute tradeoffs)
- [`encoders.md`](encoders.md) — Frozen pretrained audio (Whisper, CLAP, HuBERT) and visual (DINOv2, SigLIP, MobileViT) encoder options; VRAM, latency, embedding dimension, pairing recommendations.
- [`ml-runtimes.md`](ml-runtimes.md) — `ort`, `candle`, `burn`, `tch-rs`; inference vs. training use cases; recommended split runtime.
- [`vector-retrieval.md`](vector-retrieval.md) — Recollection layer options: flat search, HNSW, IVF-PQ, modern Hopfield; memory/query latency by scale.
- [`sparse-coding.md`](sparse-coding.md) — k-WTA, top-k, SAE/L1; compute overhead, dead-unit problem, interaction with growth.
- [`memory-compression.md`](memory-compression.md) — FP16/INT8/INT4 quantization, PQ, pruning, distillation; warm→cold compression pipeline.

### Algorithms (compute / quality tradeoffs)
- [`growth-mechanisms.md`](growth-mechanisms.md) — Cascade-correlation, GNG, online cloning; cost per unit added, growth trigger detection.
- [`replay-strategies.md`](replay-strategies.md) — Uniform, recency-weighted, PER, reservoir sampling, generative replay; compute overhead and episodic buffer sizing.
- [`multimodal-binding.md`](multimodal-binding.md) — Concatenation, gated fusion, cross-attention, contrastive, PoE; latency and top-down feedback mechanisms.

## Suggested future topics

- Neuromodulation as gain control in deep networks.
- Self-supervised learning from video and audio (foundation for Phase 2 encoding).
- Scheduling and concurrency: real-time wake + batch sleep on one GPU.
