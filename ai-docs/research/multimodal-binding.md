# Multimodal binding

Research note for somnus. Covers how the integrative/binding area composes concurrent regional embeddings into a single unified episode, and how the bidirectional top-down prior paths work. All numbers are **engineering estimates**.

See `architecture.md` "Integrative / binding area" for the design commitment. The binding area is the gateway from the regional layer to episodic memory, and provides the cross-modal disambiguation (McGurk-style) through its top-down feedback to regions.

---

## What binding must do

**Bottom-up (required in all phases):**
- Input: concurrent regional embeddings (audio-speech, audio-env, visual, symbolic) — at least 2–4 vectors of dimension 384–768.
- Output: one unified episode embedding for episodic memory storage.
- Constraint: must be fast enough for real-time operation during wake (<5 ms).

**Top-down (required for cross-modal disambiguation):**
- Given a partial or ambiguous cue in one region, use the binding area state to send a prior back to other regions that biases their interpretation.
- Example: visual mouth-motion present → binding area raises prior on "speech present" in audio-speech region, surfacing a faint voice under engine noise.
- Constraint: must be differentiable (for sleep-cycle learning updates) and bidirectional.

---

## Schemes

### Concatenation + linear projection

The simplest possible binding: concatenate the regional embeddings and project to the output dimension with a linear layer.

```
z = Linear([e_speech; e_env; e_visual; e_symbolic], d_out)
```

For 4 regions at 384-dim each: input = 1536 dims → project to 512-dim output.
Linear layer: 1536 × 512 × 4 bytes = 3.1 MB FP32. Trivial.

**Forward pass latency:** one matrix-vector multiply. For batch=1: <0.1 ms on GPU. Negligible.

**Pros:**
- Near-zero compute overhead.
- Differentiable by construction.
- Simple to implement, debug, and reason about.

**Cons:**
- No cross-region interaction in the binding layer itself — all interaction must come from the linear weights.
- Hard to implement top-down feedback: a simple linear layer has no natural mechanism for feeding a partial state back to individual regions.
- May not learn meaningful compositional structure without strong regularization.

**Decision for somnus:** use concatenation + projection as the Phase 2 baseline. It is correct by simplicity. Upgrade only when cross-modal interaction fails in practice.

---

### Gated fusion

Each region's embedding is modulated by a learned gate that depends on a global context signal (e.g., from a gating network that sees all regional embeddings):

```
gate_i = σ(W_gate × [e_1; e_2; e_3; e_4])  # one gate per region
z = sum_i(gate_i × Linear(e_i))
```

**Pros:**
- The binding layer can learn to up- or down-weight regions based on their current content (e.g., silence → zero weight on audio-speech, high weight on visual).
- Still simple: one gating network + per-region projections. Total params: ~4×(384×512) + (1536×4) ≈ 790K params. Tiny.
- Differentiable. Latency: ~1–2 ms (est.) for 4 regions.

**Cons:**
- Gates computed from concatenated inputs — does not capture pairwise region interactions.
- Top-down path still requires a separate mechanism.

**Role in somnus:** good upgrade from plain concatenation in Phase 3 when the system can learn gate values from replay.

---

### Cross-attention (transformer-style)

Model the binding area as a multi-head cross-attention module. Each region's embedding attends to the others, computing weighted combinations based on learned queries/keys.

```
# Each region e_i queries the other regions
attended_i = softmax(Q_i(e_i) · K_j(e_j)^T / sqrt(d_k)) · V_j(e_j)
z = Linear(concat[attended_i for all i])
```

**Pros:**
- Explicit pairwise region interaction — audio-speech attends to visual, etc.
- **Natural mechanism for top-down feedback**: the query for a region is a function of the current binding state; by computing attention iteratively (multiple steps), the binding area can refine regional representations.
- Differentiable, gradient flows cleanly.
- Well-understood (standard transformer self-attention).

**Cons:**
- Compute: O(n_regions² × d_head). For 4 regions, d_head=64: 16 × 64 = 1024 multiply-adds per head. With 8 heads: still <1 ms on GPU.
- More parameters: Q/K/V projection matrices per region, ~4×3×(384×64×8) = ~3M params. Small.
- Hyperparameters: n_heads, d_head, n_layers. Need to tune.
- Overkill for Phase 2 — the cross-attention pattern cannot be learned until there is substantial multi-episode experience.

**Decision for somnus:** target binding architecture for Phase 3+ once replay enables learning. The iterative attention formulation naturally implements the top-down feedback to regions.

---

### Contrastive binding (CLIP-style)

Train the binding area and the region encoders jointly with a contrastive loss: episodes with concurrent audio+visual should have their cross-modal embeddings pulled together; episodes from different moments should be pushed apart.

**Pros:**
- Forces the binding layer to learn a shared semantic space for all modalities.
- Cross-modal retrieval (audio cue → find matching visual episode) falls out naturally.

**Cons:**
- Requires paired multimodal training data — we need audio and video from the same moment, which we have, but the pairing signal is implicit (temporal co-occurrence).
- In early phases, temporal co-occurrence is a noisy pairing signal (audio and video may not be semantically related even when simultaneous).
- Training is more complex (requires careful batch construction for negatives).

**Role in somnus:** interesting for Phase 4+ once the system has accumulated enough varied experience for the contrastive pairing signal to be meaningful. Not needed for Phase 2–3.

---

### Product of Experts (PoE)

Model each regional embedding as a "soft constraint" on the latent space, and take the product:

```
p(z | e_1, e_2, ..., e_n) ∝ ∏_i p(z | e_i)
```

In Gaussian form: the product of Gaussians is a Gaussian with precision = sum of precisions; mean = precision-weighted average.

**Pros:**
- Natural uncertainty handling: if a region is low-confidence (high variance Gaussian), it contributes less to the bound episode.
- Handles missing modalities gracefully: missing regions contribute no constraint.

**Cons:**
- Requires the regional encoders to output distribution parameters (mean + variance), not just point embeddings — a significant architectural change.
- Product collapse: if one region is highly confident, it dominates and the others are ignored.
- More complex training (variational objective).

**Role in somnus:** deferred to Phase 5+. The missing-modality handling is valuable eventually (one ear vs. two), but the architecture change is too disruptive for Phase 2–3.

---

## Top-down feedback mechanism

The bidirectional design (binding area → regions) requires an explicit feedback pathway. Options:

### Broadcast (simplest)

After computing the binding output z, broadcast it back to each region as an additive bias:

```
e_i_refined = e_i + Linear(z, d_region)
```

Each region's representation is shifted by a learned function of the global state. The shift biases interpretation without overwriting the bottom-up signal.

**Latency:** one linear layer per region per feedback step. <0.5 ms total. Negligible.
**Tradeoff:** all regions receive the same global signal — cannot send modality-specific priors (e.g., only update audio-speech region, not audio-env). Can be addressed with region-specific projection matrices.

### Iterative refinement (attention-based)

Run the cross-attention binding several times (2–3 steps): each step takes the current regional embeddings, computes a binding state, feeds back to refine the regional embeddings, then recomputes binding.

**Latency:** scales linearly with number of steps. For 3-step cross-attention: ~3 ms total (est.). Acceptable.
**Tradeoff:** more powerful but requires more training signal to learn the refinement dynamics. Good target for Phase 4+.

---

## Comparison table

| Scheme                 | Phase | Compute (est.) | Top-down | Cross-modal interaction | Complexity |
|------------------------|-------|----------------|----------|------------------------|------------|
| Concatenation + linear | 2     | <0.1 ms        | Manual   | None (linear only)     | Low        |
| Gated fusion           | 3     | ~0.5 ms        | Manual   | Global gate            | Low–Med    |
| Cross-attention        | 3–4   | ~1–2 ms        | Natural  | Pairwise               | Medium     |
| Contrastive binding    | 4+    | ~2–5 ms        | No       | Shared space           | High       |
| Product of Experts     | 5+    | ~2–5 ms        | Natural  | Uncertainty-weighted   | High       |

---

## Budget analysis

Binding area parameter count (Phase 3, cross-attention with 4 regions at 384-dim, 8 heads, d_head=64):
- Q/K/V projections: 4 regions × 3 matrices × (384 × 64 × 8) = ~3.0M params.
- Output projection: 4 × (512 × 512) = ~1.0M params.
- **Total: ~4M params, ~8 MB FP16. Negligible.**

Binding area activation memory per forward pass (4 regions × 1 episode):
- Attention scores: 4 × 4 × 8_heads = 128 floats.
- Hidden states: 4 × 512 = 2048 floats.
- ~20 KB per forward pass — entirely negligible.

The binding area is not a memory or compute bottleneck at any foreseeable scale.

---

## Open questions → future ADR

1. Start with concatenation+linear (Phase 2) and upgrade to gated fusion or cross-attention (Phase 3)? The concatenation baseline is fast to implement and allows testing the rest of the pipeline without committing to a binding architecture.
2. How many cross-attention steps for the iterative top-down refinement? 2–3 steps is typically sufficient in practice.
3. Should the binding area embed temporal context (episode ordering, timestamps) or operate purely on the current snapshot?
4. Handling missing modalities: if video is unavailable (dark room), the visual region produces a zero/noise embedding. Should the binding area have a "present/absent" flag per region?

---

## Sources

- Radford et al., 2021. "Learning Transferable Visual Models From Natural Language Supervision." CLIP. arXiv:2103.00020.
- Vaswani et al., 2017. "Attention Is All You Need." arXiv:1706.03762. (Multi-head attention mechanism.)
- Wu & Goodman, 2018. "Multimodal Generative Models for Scalable Weakly-Supervised Learning." NeurIPS. (Product of Experts for multimodal VAEs.)
- McGurk & MacDonald, 1976. "Hearing lips and seeing voices." *Nature*. (McGurk effect — empirical case for top-down cross-modal priors.)
- Rao & Ballard, 1999. "Predictive coding in the visual cortex." *Nature Neuroscience*. (Bidirectional information flow in perception.)
