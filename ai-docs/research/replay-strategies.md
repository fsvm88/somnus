# Replay strategies

Research note for somnus. Covers how episodic memory content is selected and replayed into semantic memory during NREM-like sleep. All numbers are **engineering estimates**.

See `architecture.md` "Sleep stages" (NREM-like: "Replay") and `sleep-and-consolidation.md`. The replay mechanism is the primary consolidation channel from hot (episodic) to warm (semantic) memory.

---

## What replay does

During wake, the system writes unified episodes to the hot tier (episodic memory in VRAM). During NREM-like sleep:
1. Episodes are sampled from episodic memory.
2. Each sampled episode is "replayed" — passed back through the cognitive layer (regional concept areas + binding area) to update semantic memory weights.
3. The update signal comes from the episode itself (self-supervised reconstruction target) or from cross-episode patterns (hebbian reinforcement of co-occurring features).

Goal: transfer the gist of recent experience into the semantic layer without catastrophic interference with older consolidated patterns.

---

## Selection strategies

### Uniform random replay

Sample episodes uniformly at random from the episodic buffer.

**Pros:**
- Zero bookkeeping. No priority scores, no data structures beyond the buffer.
- Unbiased — every episode gets equal replay probability.
- Good theoretical properties for CL (uniform mixing prevents recency bias in semantic updates).

**Cons:**
- Inefficient if the buffer is large and only a small fraction of episodes are informative (e.g., many redundant frames with nearly identical content).
- Does not prioritize novel or surprising experiences — which are typically the ones most worth consolidating.

**Compute cost:**
- Sampling: O(1) per sample (random index into array).
- No maintenance overhead.

**Recommended for Phase 3 (first sleep cycle):** start here. It is robust, simple, and well-understood.

---

### Recency-weighted replay (exponential decay)

Give higher replay probability to more recent episodes using an exponential decay schedule: `p(i) ∝ exp(−λ × age(i))`.

**Pros:**
- Recent experiences are more likely to still be relevant to current perception.
- Models the mammalian bias toward replaying recent events during early sleep.
- Still simple — requires only storing a timestamp per episode.

**Cons:**
- Old episodes get very low probability and are effectively abandoned — eventual forgetting of older content is accelerated.
- The decay rate λ is a hyperparameter with no obvious principled value.

**Compute cost:**
- Weight maintenance: one float per episode (timestamp → decay weight).
- Sampling from a non-uniform distribution: reservoir sampling O(n) or alias method O(1) after O(n) build.

---

### Prioritized Experience Replay (PER)

Schaul et al., 2015 (PER). Priority ∝ TD error (in RL) or, for somnus, ∝ reconstruction error (how surprising/hard to recall the episode is).

**For somnus:**
- Priority p(i) = reconstruction error of episode i under the current semantic memory.
- High-priority = episodes the semantic layer currently fails to explain = highest value for consolidation.
- Combined with recency: p(i) ∝ reconstruction_error(i) × recency_weight(i).

**Pros:**
- Focuses sleep time on the most information-dense episodes.
- Naturally surfaces novel concepts (high reconstruction error) for preferential consolidation.
- Connects directly to the growth trigger: episodes with high reconstruction error are both replay-prioritized and gist-failure candidates for growth.

**Cons:**
- Requires computing and maintaining reconstruction errors — non-trivial overhead.
- PER introduces a sampling bias: low-priority episodes may be starved, which can cause their representations to degrade (a subtle form of forgetting).
  - Mitigation: importance sampling weights to correct for the bias (standard PER solution).
- The priority score is stale the moment semantic memory weights update — needs periodic refresh.

**Compute cost:**
- Priority maintenance: one forward pass through the decoder per episode to compute reconstruction error. For 100K episodes at ~2 ms/episode: ~3 min per full priority refresh.
- Sampling: O(log n) with a sum-tree (standard PER data structure).
- Priority refresh should happen incrementally, not all at once: update priorities for recently replayed episodes only.

**Overhead vs. uniform: ~20–30%** more compute for the replay phase (est.), plus priority maintenance.

---

### Reservoir sampling (bounded buffer with replacement)

Not a sampling strategy per se, but a buffer management strategy: maintain a fixed-size episodic buffer where each new episode has an equal probability of replacing any existing episode.

**Result:** the buffer is a uniform random sample of all episodes seen so far, regardless of buffer size. No recency bias.

**Use case in somnus:** if episodic memory VRAM is bounded (e.g., only keep the most recent N episodes), reservoir sampling ensures that the buffer does not forget old episodes entirely — it maintains a representative sample of the full history.

**Tradeoff:** trades temporal density for temporal breadth. For a 1M-episode buffer with 100M total seen: the buffer holds a 1% sample of history. Individual events are preserved; their temporal order information may be lost.

---

### Generative replay

Instead of storing raw episodes and replaying them, train a generator (VAE or diffusion model) to synthesize plausible episodes, interleaved with real ones.

**Pros:**
- Storage is bounded by the generator's parameter count, not the history length.
- Generator can produce novel combinations of seen patterns (useful for robustness).

**Cons:**
- Generator quality must be maintained — if the generator drifts or degrades, replayed "memories" are hallucinations, not memories. Generative drift is a known failure mode.
- Training the generator is expensive: it is a full generative model over the input distribution.
- Generator and cognitive-layer network must be trained jointly or carefully alternated to avoid catastrophic forgetting of the generator itself.

**Phase for somnus:** Phase 5+. Not needed until episodic memory storage becomes the binding constraint (which won't happen for a long time given 14 GB VRAM budget). The binding area's associative-completion capability is a lightweight version of this that does not require a separate generator.

---

### Time-reversed replay

Mammalian hippocampal SWRs replay sequences both forward and in reverse (Diba & Buzsáki, 2007). Reverse replay has theoretical advantages for temporal credit assignment.

**For somnus:** an interesting future experiment. During NREM-like sleep, replay an episode sequence both forwards (for sequence-learning) and backwards (for reward/credit propagation, if a reward signal exists). Not applicable in Phase 3–4 (no temporal sequences stored as such, no reward signal).

---

## What to replay: episodes vs. features

**Raw episode (binding area output):**
- Store the unified 512-dim representation from the binding area. This is the "what happened" representation.
- Replay: pass this stored vector back through the cognitive update path.
- Size: ~1 KB per episode. 1M episodes = 1 GB VRAM.

**Encoder-level features (region-specific):**
- Store per-region encoder outputs (before binding). Finer-grained; enables region-specific replay.
- Size: 4× larger per episode (4 regions × ~512 dims each). 1M episodes = 4 GB VRAM.

**Raw sensory data:**
- Store mel spectrogram frames or video frames alongside the episode. Enables full replay through the frozen encoder stack.
- Size: orders of magnitude larger. Not practical for long histories.

**Decision for somnus:** store binding-area outputs as the primary replay unit. Store per-region features only for episodes flagged as high-priority (high reconstruction error, or salience-marked at recording time). Raw sensory data only for a short ring buffer (last few minutes), not for the full episodic memory.

---

## Budget analysis

**Episodic memory capacity (VRAM, binding outputs at FP16):**
- 512-dim FP16 binding output: 1 KB/episode.
- Available VRAM after encoders + runtime overhead: ~12 GB.
- 12 GB / 1 KB = ~12M episodes → **~12 million episodes** fit in VRAM.

At 1 episode per second of interesting events: 12M seconds ≈ 140 days of continuous experience. The VRAM is not the binding constraint for episodic memory capacity in any near-term phase.

**Sleep replay throughput:**
- Uniform replay, 10K episodes per sleep cycle, 2 ms/episode semantic update: ~20 s of GPU time.
- PER with priority maintenance (full refresh of 100K episodes): ~3 min.
- Sleep cycle target: consolidate enough episodes that episodic memory can safely drop old content.

---

## Open questions → future ADR

1. PER vs. uniform for the first sleep cycle implementation (Phase 3)? PER is more principled but adds complexity; uniform is robust and correct.
2. What is the right episodic buffer policy — keep all episodes until sleep, or use a bounded ring buffer? Given the 12M episode capacity, a ring buffer is not forced; but bounded buffers have simpler semantics.
3. How to handle the PER generative-drift problem — if reconstruction errors are stale, do stale priorities cause replay waste?
4. Episode granularity: one episode per second? Per event (salience-gated)? Per input frame? Granularity affects buffer fill rate and replay quality.

---

## Sources

- Shin, Lee, Kim, & Kim, 2017. "Continual Learning with Deep Generative Replay." NeurIPS. arXiv:1705.08690.
- Schaul et al., 2015. "Prioritized Experience Replay." arXiv:1511.05952.
- McClelland, McNaughton, & O'Reilly, 1995. *Psychological Review*, 102: 419–457. (CLS theory, replay as consolidation channel.)
- Diba & Buzsáki, 2007. "Forward and reverse hippocampal place-cell sequences during ripples." *Nature Neuroscience*. (Time-reversed replay.)
- Vitter, 1985. "Random sampling with a reservoir." *ACM Transactions on Mathematical Software*. (Reservoir sampling.)
