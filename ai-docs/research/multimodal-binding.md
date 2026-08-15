# Multimodal binding

> **Contents:** Binding mechanism options (concatenation, gated fusion, cross-attention ✓ committed, contrastive, PoE), top-down prior feedback, latency constraints. Covers 4-head cross-attention + GRU temporal context rationale.
> **When to read:** Understanding the BindingArea design; adding a new region; evaluating alternatives to cross-attention.

Research note covering how the integrative/binding area composes concurrent regional embeddings into a unified episode, and how bidirectional top-down priors work. All numbers are **engineering estimates**. See `architecture.md` "Integrative / binding area" — the gateway from regions to episodic memory, and the source of cross-modal (McGurk-style) disambiguation via top-down feedback.

---

## What binding must do

**Bottom-up (all phases):** input = concurrent regional embeddings (audio-speech, audio-env, visual, symbolic — 2-4 vectors, 384-768 dim); output = one unified episode embedding for episodic memory; constraint = real-time wake operation (<5ms).

**Top-down (cross-modal disambiguation):** given a partial/ambiguous cue in one region, send a prior back to bias other regions' interpretation — e.g. visual mouth-motion → raised "speech present" prior in audio-speech, surfacing a faint voice under engine noise. Must be differentiable (sleep-cycle learning) and bidirectional.

---

## Schemes

**Concatenation + linear projection.** `z = Linear([e_speech; e_env; e_visual; e_symbolic], d_out)`. 4×384-dim → 512-dim: 1536×512×4B=3.1MB FP32, <0.1ms/call. Near-zero overhead, differentiable by construction, simple. No cross-region interaction beyond the linear weights; no natural top-down mechanism; may not learn compositional structure without regularization. **Phase 2 baseline** — correct by simplicity, upgrade only when cross-modal interaction fails in practice.

**Gated fusion.** `gate_i = σ(W_gate·[e_1;e_2;e_3;e_4])`, `z = Σ gate_i·Linear(e_i)`. Learns to up/down-weight regions by content (silence→zero audio-speech weight). ~790K params, differentiable, ~1-2ms/4 regions. Gates only see the concatenated input (no pairwise interaction); top-down still needs a separate mechanism. Good Phase 3 upgrade once the system can learn gates from replay.

**Cross-attention (transformer-style).** `attended_i = softmax(Q_i(e_i)·K_j(e_j)ᵀ/√d_k)·V_j(e_j)`, `z=Linear(concat[attended_i])`. Explicit pairwise interaction; natural top-down mechanism (query is a function of binding state, iterative attention refines regions); differentiable, well-understood. `O(n_regions²×d_head)` — <1ms on GPU for 4 regions/8 heads; ~3M params (Q/K/V per region); needs tuning (n_heads/d_head/n_layers); overkill until there's enough multi-episode replay experience to learn the pattern. **Target architecture for Phase 3+**, with iterative attention naturally implementing top-down feedback.

**Contrastive binding (CLIP-style).** Joint training of binding + region encoders with a contrastive loss (concurrent audio+visual pulled together, different moments pushed apart) — forces a shared cross-modal semantic space, cross-modal retrieval falls out naturally. Needs paired multimodal data (temporal co-occurrence is a noisy signal early on); more complex training (negative-batch construction). Phase 4+, once accumulated experience makes the pairing signal meaningful.

**Product of Experts (PoE).** `p(z|e_1..e_n) ∝ ∏ p(z|e_i)`; Gaussian form: product precision = sum of precisions, mean = precision-weighted average. Natural uncertainty handling (low-confidence region contributes less); graceful missing-modality handling (no constraint at all). Needs regions to output distribution params (mean+variance), not point embeddings — a significant architectural change; a highly-confident region can dominate and silence others; more complex (variational) training. Deferred to Phase 5+ — missing-modality handling is valuable eventually but the architecture change is too disruptive for Phase 2-3.

---

## Top-down feedback mechanism

**Broadcast (simplest).** `e_i_refined = e_i + Linear(z, d_region)` — each region shifted by a learned function of global state, biasing without overwriting the bottom-up signal. <0.5ms total. All regions get the same global signal (no modality-specific priors) unless region-specific projections are added.

**Iterative refinement (attention-based).** Run cross-attention binding 2-3 steps: each computes a binding state, refines regional embeddings, recomputes. ~3ms total for 3 steps — acceptable, but needs more training signal to learn the refinement dynamics. Good Phase 4+ target.

---

## Comparison table

| Scheme | Phase | Compute (est.) | Top-down | Cross-modal interaction | Complexity |
|---|---|---|---|---|---|
| Concatenation + linear | 2 | <0.1ms | Manual | None (linear only) | Low |
| Gated fusion | 3 | ~0.5ms | Manual | Global gate | Low-Med |
| Cross-attention | 3-4 | ~1-2ms | Natural | Pairwise | Medium |
| Contrastive binding | 4+ | ~2-5ms | No | Shared space | High |
| Product of Experts | 5+ | ~2-5ms | Natural | Uncertainty-weighted | High |

---

## Budget analysis

Phase 3 cross-attention, 4 regions @ 384-dim, 8 heads, d_head=64: Q/K/V ≈ 3.0M params, output projection ≈ 1.0M — **total ~4M params, ~8MB FP16**, negligible. Activation memory/forward pass: attention scores 4×4×8=128 floats, hidden states 4×512=2048 floats — ~20KB, entirely negligible. Not a memory or compute bottleneck at any foreseeable scale.

---

## Open questions → future ADR

1. Start with concatenation+linear (Phase 2), upgrade to gated fusion or cross-attention (Phase 3)? Concatenation is fast to implement and lets the rest of the pipeline be tested without committing to a binding architecture.
2. How many cross-attention steps for iterative top-down refinement? 2-3 typically sufficient.
3. Should binding embed temporal context (episode ordering/timestamps) or operate on the current snapshot only?
4. Missing modalities (dark room → zero/noise visual embedding): should binding have a present/absent flag per region?

---

## Sources

- Radford et al., 2021. "Learning Transferable Visual Models From Natural Language Supervision." CLIP. arXiv:2103.00020.
- Vaswani et al., 2017. "Attention Is All You Need." arXiv:1706.03762.
- Wu & Goodman, 2018. "Multimodal Generative Models for Scalable Weakly-Supervised Learning." NeurIPS. (PoE for multimodal VAEs.)
- McGurk & MacDonald, 1976. "Hearing lips and seeing voices." *Nature*.
- Rao & Ballard, 1999. "Predictive coding in the visual cortex." *Nature Neuroscience*.
