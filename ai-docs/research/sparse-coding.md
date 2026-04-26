# Sparse coding

Research note for somnus. Covers sparse representation schemes for the regional concept areas: k-winner-take-all (k-WTA), top-k activation, and sparse autoencoders (SAE/L1-penalized). All numbers are **engineering estimates**.

See `architecture.md` "Regional concept areas" for the design commitment. Sparsity is a core mechanism for reducing interference between concepts and making growth local — new units can occupy different activation patterns without overwriting consolidated ones.

---

## Why sparsity

In dense representations, every new concept update touches all weights, causing interference with existing concepts (catastrophic forgetting). Sparse codes make different concepts use largely non-overlapping subsets of units:

- If concept A activates units {1, 5, 12, 33} and concept B activates {2, 8, 17, 44}, a weight update for B does not affect A's representation.
- With k active units out of n total, the expected overlap between two random concepts is k²/n. For k=20, n=1000: expected overlap = 0.4 units — negligible.
- This is the HTM / sparse distributed representation insight (Hawkins/Numenta lineage).

Sparsity also enables local growth: adding new units to a region only affects representations that use them.

---

## Metrics for sparsity

- **Sparsity level:** fraction of active units = k/n. Typical useful range: 1–10% (k-WTA with k/n = 0.02–0.10).
- **Lifetime sparsity:** average activation frequency of a single unit over many inputs. A unit that is active on 5% of inputs is more useful than one that fires on 90% or 0.01%.
- **Population sparsity:** average fraction of active units per input. This is the k/n metric above.

Good codes have both: each input activates few units, and each unit activates on few inputs.

---

## Schemes

### k-WTA (k-winner-take-all)

A hard competitive constraint: given a layer of n neurons with raw activations, keep the top-k highest activations and zero out the rest.

**How it works:**
1. Compute raw activations: z = Wx + b (standard linear layer + optional nonlinearity).
2. Find the k-th largest value in z.
3. Zero all activations below the k-th largest. Optionally apply ReLU to the survivors.

**Compute cost:**
- Finding the k-th largest: `O(n)` using QuickSelect (partial sort), or `O(n log k)` using a min-heap of size k.
- GPU: `torch.topk` (or equivalent) is a fast CUDA primitive. For n=1000, k=20: negligible overhead (~0.01 ms).
- Backward pass: gradient flows only through the k active units. Backprop cost scales with k, not n — slightly cheaper than dense.

**Memory:**
- No additional memory beyond the weight matrix — the mask is computed in-place.
- Sparse activations can be stored as (index, value) pairs: k × 2 values instead of n. For k=20, n=1000: 20 floats + 20 indices vs. 1000 floats — 49× storage reduction for activations.

**Tradeoffs:**
- Dead units: units that never win across many inputs receive no gradient and cease to learn. Require mitigation (boosting: temporarily increase bias of underused units to force them to win occasionally).
- Hard threshold has discontinuous gradient — not ideal for standard gradient-based training, but workable with straight-through estimator or soft approximations.
- Fixed k means fixed representational capacity — if the input complexity grows, k/n may be too small.
- Simple to implement; widely used in HTM and spiking NN literature.

**Recommended use in somnus:** k-WTA with boosting on the regional concept area linear layers. Start with k/n ≈ 0.02–0.05 (2–5% sparsity). Monitor lifetime sparsity to detect dead units.

---

### Top-k activation (soft variant)

A differentiable approximation of k-WTA. Several variants:

- **Sparsemax:** projects the activation vector onto the probability simplex with a sparse solution. Differentiable, outputs exactly k non-zero values in expectation. Slower than top-k (requires a sort + thresholding pass).
- **Entmax(α):** generalization of softmax and sparsemax with tunable sparsity. α=1.5 gives a good middle ground.
- **Straight-through estimator on top-k:** apply hard top-k in the forward pass, use gradient of the full activation in the backward pass. Simple, works well in practice.

**Compute cost:**
- Sort: O(n log n) for exact sort, O(n) for approximate with bucketing.
- For n=1000, k=20: sort cost ~10–20 µs on CPU, <1 µs on GPU with CUDA `topk`.

**Tradeoffs:**
- Differentiable → cleaner gradient flow, better integration with standard optimizers.
- Less interpretable than hard k-WTA.
- `entmax` requires a custom kernel (not in standard CUDA libs); straight-through top-k is simplest.

---

### Sparse autoencoder (SAE) / L1-penalized sparse coding

Train an autoencoder (encoder + decoder) where the bottleneck layer is encouraged to be sparse via an L1 penalty on activations: `loss = reconstruction_loss + λ × ||z||₁`.

At inference, the encoder maps input → sparse code; the decoder reconstructs.

**Compute cost (training):**
- Full encoder forward + reconstruction + L1 penalty computation: ~2–3× cost of a plain autoencoder.
- Training to convergence on the regional concept area may take thousands of replay steps during sleep.

**Compute cost (inference):**
- Encoder forward pass only; same cost as a standard linear layer of the same size.
- No sorting step needed — sparsity is emergent from the trained weights.

**Memory:**
- Model weights: one encoder matrix (d_input × n_hidden) + one decoder matrix (n_hidden × d_input).
- For d_input=512 (from encoder), n_hidden=2000 units: encoder = 512×2000×4 bytes = 4 MB FP32. Small.
- Dead units still occur but are partially mitigated by the L1 penalty (units with never-used codes get zero gradient from L1).

**Tradeoffs:**
- Naturally learns which features are informative — the coding space is data-driven, not manually imposed.
- Reconstruction loss provides a natural self-probe signal: high reconstruction error on an input → the concept is not well-covered by current codes → candidate for growth.
- Training requires a reasonable amount of replay data to converge. In early phases (before much experience), codes will be underdeveloped.
- λ tuning is sensitive: too high → nearly all units inactive (useless); too low → dense codes (no sparsity benefit).
- Dictionary learning (e.g., online dictionary learning, KSVD) is a non-gradient alternative that directly optimizes for sparse codes — may be more efficient for small concept areas.

**Recommended use in somnus:** SAE over the regional concept area is the most principled approach and gives reconstruction error as a free signal for growth triggering. k-WTA is simpler and adequate for Phase 1–2; SAE can replace it in Phase 3–4 when the replay pipeline is working.

---

### Comparison table

| Scheme        | Sparsity type | Gradient | Compute overhead | Self-probe signal | Complexity |
|---------------|---------------|----------|------------------|--------------------|------------|
| k-WTA         | Hard, fixed k | Discontinuous (needs STE) | Minimal (topk call) | None directly | Low |
| Top-k / STE   | Hard, fixed k | Straight-through | Minimal (topk call) | None directly | Low |
| Sparsemax     | Soft, dynamic | Differentiable | Sort + project | None directly | Medium |
| SAE (L1)      | Soft, emergent | Clean backprop | 2–3× training | Reconstruction error | Medium |
| Dictionary learning | Hard, learned | Alternating minimization | Per-iteration | Reconstruction error | Medium-High |

---

## Interaction with growth

The growth trigger (cascade-correlation unit addition when gist-level recall fails) works naturally with sparse codes:

- New units start with random or zero weights; they are not yet active for any existing patterns.
- Cascade-correlation trains new units to correlate with the current reconstruction residual, activating them for inputs that are currently not well-represented.
- Because existing patterns use sparse codes, new units can be added without disrupting them — they fill in new "slots" in the activation space.
- SAE reconstruction error directly quantifies "which inputs need more capacity" — the growth trigger condition.

---

## Budget analysis

A regional concept area with n=2000 units, k=40 active (2% sparsity), d_input=384 (from DINOv2-S):

- **Weight matrix:** 384 × 2000 × 4 bytes (FP32) = 3.1 MB. Negligible.
- **Per-episode activation storage (sparse):** 40 floats + 40 indices = 320 bytes. For 1M episodes: 320 MB RAM.
- **Per-episode activation storage (dense, for comparison):** 2000 × 4 bytes = 8 KB. For 1M episodes: 8 GB.

Sparse activation storage is 25× smaller for 2% sparsity. At larger scales (n=10K, k=200), the sparse format is still 50× smaller than dense.

Across 4 regions (audio-speech, audio-env, visual, symbolic) + binding area: total cognitive-layer weight matrices < 100 MB even at n=5000 per region. Entirely negligible relative to the 14 GB VRAM budget.

---

## Open questions → future ADR

1. Start with k-WTA (simpler) or SAE (more principled) for Phase 2 regional areas? SAE gives the reconstruction signal that the growth mechanism needs — worth the added complexity.
2. Shared sparsity level across all regions, or per-region tuning? Visual and audio concepts likely differ in intrinsic dimensionality.
3. How to handle dead units in k-WTA — boosting threshold, or explicit reinitialisation?
4. Binding area sparsity: does the integrative area benefit from sparsity, or is a dense representation more appropriate there?

---

## Sources

- Olshausen & Field, 1996. "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." *Nature* 381: 607–609.
- Hawkins, Ahmad, & Cui, 2017. "A Theory of How Columns in the Neocortex Enable Learning the Structure of the World." *Frontiers in Neural Circuits*. HTM sparse distributed representations.
- Makhzani & Frey, 2013. "k-Sparse Autoencoders." arXiv:1312.5663.
- Elhage et al., 2022. "Toy Models of Superposition." Anthropic. (Motivates SAE for disentangled features.) Transformer Circuits Thread.
- Bricken et al., 2023. "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." Anthropic. (Sparse autoencoders applied to transformer features.)
- Martins & Astudillo, 2016. "From Softmax to Sparsemax." ICML. arXiv:1602.02068.
- Peters et al., 2019. "Sparse Sequence-to-Sequence Models." ACL. Entmax.
