# Replay strategies

> **Contents:** Replay sampling options — uniform, recency-weighted, prioritized experience replay (PER), reservoir, generative. Covers compute overhead, episodic buffer sizing recommendations, and why uniform+synthetic was chosen for somnus.
> **When to read:** Tuning the NREM replay loop; implementing PER or recency weighting; understanding synthetic replay (ADR 0019).

Research note covering how episodic memory content is selected and replayed into semantic memory during NREM-like sleep. All numbers are **engineering estimates**. See `architecture.md` "Sleep stages" (NREM: "Replay") and `sleep-and-consolidation.md` — replay is the primary hot→warm consolidation channel.

---

## What replay does

Wake writes unified episodes to the hot tier. During NREM-like sleep: sample episodes from episodic memory → replay each back through the cognitive layer (regions + binding) to update semantic weights, using a self-supervised reconstruction target or cross-episode Hebbian reinforcement. Goal: transfer the gist of recent experience into the semantic layer without catastrophically interfering with older consolidated patterns.

---

## Selection strategies

**Uniform random replay.** Zero bookkeeping, unbiased, good CL theoretical properties (uniform mixing prevents recency bias). Inefficient if the buffer has lots of redundant/near-identical content; doesn't prioritize novel/surprising experiences. O(1) sampling, no maintenance. **Recommended starting point (Phase 3)** — robust, simple, well-understood.

**Recency-weighted replay.** `p(i) ∝ exp(-λ·age(i))`. Recent episodes stay relevant to current perception, models the mammalian early-sleep recency bias, still simple (one timestamp/episode). Old episodes get abandoned (accelerated forgetting); λ has no obvious principled value. Sampling: reservoir O(n) or alias method O(1)-after-O(n)-build.

**Prioritized Experience Replay (PER).** Schaul et al. 2015 — priority ∝ reconstruction error under current semantic memory (somnus's TD-error analogue); optionally combined with recency: `p(i) ∝ recon_error(i)×recency(i)`. Focuses sleep time on information-dense episodes, surfaces novel concepts, connects directly to the growth trigger (high-error episodes are both replay-prioritized and growth candidates). Needs maintained reconstruction errors (non-trivial overhead); sampling bias starves low-priority episodes (mitigate with importance-sampling weights, standard PER fix); priority goes stale the moment weights update, needs periodic refresh. Cost: one decoder forward pass/episode for priority (~2ms/ep, ~3min for a 100K full refresh — do incrementally instead); sampling O(log n) via sum-tree. **Overhead vs. uniform: ~20-30%** more replay-phase compute plus maintenance.

**Reservoir sampling.** Buffer management, not selection per se: each new episode has equal probability of replacing any existing one — the buffer stays a uniform sample of all history regardless of size, no recency bias. Useful if episodic VRAM is bounded — preserves a representative historical sample rather than forgetting old episodes entirely. Trades temporal density for breadth (1M-buffer/100M-seen = 1% sample; individual events preserved, temporal order may be lost).

**Generative replay.** Train a generator (VAE/diffusion) to synthesize plausible episodes interleaved with real ones. Storage bounded by generator size, not history length; can produce novel recombinations. Generator drift risk (degraded generator → hallucinated "memories," not real ones); expensive to train (a full generative model over the input distribution); needs careful joint/alternated training with the cognitive layer to avoid forgetting the generator itself. **Phase 5+** — not needed until episodic storage becomes binding (won't happen soon at 14GB VRAM); the binding area's associative-completion capability is already a lightweight version of this without a separate generator.

**Time-reversed replay.** Mammalian hippocampal SWRs replay both forward and reverse (Diba & Buzsáki 2007) — reverse has theoretical temporal-credit-assignment advantages. Interesting future experiment (forward for sequence-learning, backward for reward/credit propagation if a reward signal exists) — not applicable Phase 3-4 (no stored temporal sequences, no reward signal).

---

## What to replay: episodes vs. features

**Raw episode (binding output):** unified 512-dim "what happened" representation, ~1KB/episode (1M episodes = 1GB VRAM). **Encoder-level features (per-region):** finer-grained, enables region-specific replay, 4× larger/episode (1M = 4GB). **Raw sensory data:** mel/video frames, orders of magnitude larger, not practical for long histories.

**Decision:** binding-area outputs as the primary replay unit; per-region features only for high-priority episodes (high reconstruction error or salience-marked at recording); raw sensory data only for a short ring buffer (last few minutes), not the full episodic memory.

---

## Budget analysis

Episodic capacity: 512-dim FP16 binding output = 1KB/episode; ~12GB available VRAM / 1KB ≈ **~12M episodes fit in VRAM**. At 1 episode/second of interesting events, that's ~140 days of continuous experience — VRAM isn't the binding constraint for episodic capacity in any near-term phase. Sleep replay throughput: uniform, 10K episodes/cycle @ 2ms/episode ≈ 20s GPU; PER full 100K-episode priority refresh ≈ 3min.

---

## Open questions → future ADR

1. PER vs. uniform for the first sleep-cycle implementation? PER is more principled, uniform is robust and simpler.
2. Episodic buffer policy: keep all until sleep, or a bounded ring buffer? 12M-episode capacity means a ring buffer isn't forced, but has simpler semantics.
3. Handling PER's generative-drift problem — do stale priorities cause replay waste?
4. Episode granularity: per second? per salience-gated event? per input frame? Affects buffer fill rate and replay quality.

---

## Sources

- Shin, Lee, Kim, & Kim, 2017. "Continual Learning with Deep Generative Replay." NeurIPS. arXiv:1705.08690.
- Schaul et al., 2015. "Prioritized Experience Replay." arXiv:1511.05952.
- McClelland, McNaughton, & O'Reilly, 1995. *Psychological Review*, 102: 419-457. (CLS theory.)
- Diba & Buzsáki, 2007. "Forward and reverse hippocampal place-cell sequences during ripples." *Nature Neuroscience*.
- Vitter, 1985. "Random sampling with a reservoir." *ACM Transactions on Mathematical Software*.
