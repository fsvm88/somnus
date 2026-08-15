# Predictive coding

> **Contents:** Friston's FEP/PC framework, partial adoption rationale, what PC costs to implement fully vs. what somnus borrows (top-down priors, settling loop, reconstruction error as learning signal). Concludes with what is and isn't PC in somnus.
> **When to read:** Evaluating a PC-inspired feature; justifying or questioning the settling loop (ADR 0011); checking engineering vs. biology appeal.

Research note covering Friston's free energy principle and predictive processing, and whether/how to adopt it. All numbers are **engineering estimates**. CLAUDE.md principle applied here: "every biological appeal must also have an engineering justification; if the engineering doesn't stand on its own, the biology is decoration."

---

## What predictive coding claims

The **free energy principle** (Friston, 2005-present): biological brains minimize "variational free energy," a bound on sensory-observation surprise given an internal generative model. In predictive coding form: every layer predicts the layer below's activity via a top-down model; computes prediction error = observed − predicted; errors propagate up, predictions propagate down; learning updates the top-down model to reduce error; perception = inference = minimizing prediction error = updating beliefs. A hierarchical Bayesian inference scheme implemented as recurrent cross-layer computation.

**Relationship to standard deep learning:** not the same as backprop, but related — in a linear PC network, the per-layer weight update is local (needs only that layer's activity + error, no full-network backprop); non-linear PC (PC-Net, Millidge et al. 2022) can approximate backprop in feedforward nets; the inference phase (settling prediction errors) adds forward-backward iterations per input, like running an optimization loop before committing to a representation.

---

## Engineering tradeoffs

**What PC buys:**

| Benefit | Engineering mechanism | Applicable to somnus? |
|---|---|---|
| Bidirectional information flow | Top-down predictions + bottom-up errors | Yes — maps directly to binding↔regions |
| Local learning rules | Each layer updates from local error | Partially — useful for sleep updates without full backprop |
| Robustness to missing data | Top-down prediction fills absent modalities | Yes — graceful missing audio/video |
| Surprise as a learning signal | Prediction error = what's worth learning | Yes — directly useful as replay priority + growth trigger |
| Online inference | Model adapts belief per input via settling | Maybe — adds latency |

**The most directly useful part isn't the full FEP framework but one piece: prediction error as a signal.** SAE reconstruction error (`sparse-coding.md`) already is a form of prediction error — the decoder predicts the input, error drives learning and growth triggers. somnus uses this without adopting the full PC formalism.

**What PC costs:** settling iterations — feedforward is 1 pass; PC iterates the error-update rule to convergence, typically 5-20 steps, ~5-20× compute/input. For a region at d=384/500 units, one settling step ≈ one fwd+bwd ≈ ~1ms GPU, 10 steps ≈ ~10ms/episode; at 25fps video + continuous audio, ~250ms settling/second of experience — faster than real-time but a narrow margin. **Extra params:** bidirectional weights double region param count (384×500: 2×(384×500×4B)=1.5MB/region, still negligible) but need careful init to avoid runaway feedback instability. **Complexity:** the iterative settling loop is structurally different from forward-pass+gradient; neither `candle` nor `burn` has built-in PC support, needs a custom training loop; existing Python/PyTorch PC implementations exist as reference, porting is non-trivial but feasible.

---

## Partial adoption vs. full adoption

**Full adoption** (replace standard feedforward+backprop throughout with PC settling+local rules): gains local learning, top-down priors, surprise signal; costs 5-20× inference compute/input, custom training infra, significant complexity. **Verdict: not justified for Phase 1-3** — wake-time compute cost is the primary concern on a real-time system.

**Partial adoption** (borrow specific mechanisms) — the recommended approach:

| PC idea | Adopted as | Where | Phase |
|---|---|---|---|
| Prediction error as learning signal | SAE reconstruction error | Regional concept areas | 3 |
| Top-down prior from binding→regions | Gated feedback embedding | Binding ↔ regions | 3 |
| Surprise detection | Reconstruction error > threshold → high-salience episode | Episodic memory, replay priority | 3 |
| Generative model per region | SAE decoder | Regional concept areas | 3 |
| Homeostatic adjustment | SHY-inspired weight decay | Sleep cycle | Already in design |

Captures most of the engineering value without the inference-iteration overhead.

---

## Friston's active inference (not for early phases)

Full FEP includes active inference: actions taken to minimize *expected* free energy, not just current error — goal-directed behavior, exploration, decision-making. Eventual Phase 5+ consideration once an action loop exists; irrelevant with no action space yet.

---

## Predictive coding and the sleep cycle

Underexplored connection: **the sleep cycle as free-energy minimization at a slower timescale.** Wake = fast inference, no weight updates (errors accumulate unresolved). NREM = replay-driven updates reducing accumulated wake-time errors. REM = generative replay, the model predicting without input, testing its own generative model. Growth = adding capacity where predictions consistently fail (high free energy, insufficient capacity to reduce it). Consistent with both the FEP literature and somnus's existing design without architectural change — a conceptual lens, not an engineering commitment.

---

## Relationship to other notes

`sparse-coding.md`: SAE reconstruction error is prediction error, the decoder is the regional generative model. `multimodal-binding.md`: binding→regions top-down feedback is the PC top-down prior, iterative cross-attention is one PC-settling implementation. `replay-strategies.md`: reconstruction error as replay priority = surprise-driven consolidation. `growth-mechanisms.md`: growth triggered by persistent high reconstruction error = growth triggered by persistent high free energy.

---

## Open questions

1. Implement an explicit PC settling loop in Phase 4+ (less compute-constrained), or stay partial-adoption throughout?
2. Does the local learning rule (no full-network backprop) actually buy anything at somnus's scale? At <10M params/region, full backprop isn't the bottleneck.
3. Is SAE reconstruction error sufficient for replay prioritization, or does a dedicated prediction-error architecture add measurable value?

---

## Sources

- Friston, 2005. "A theory of cortical responses." *Philosophical Transactions of the Royal Society B*.
- Friston et al., 2010. "Action and behavior: a free-energy formulation." *Biological Cybernetics*.
- Rao & Ballard, 1999. "Predictive coding in the visual cortex." *Nature Neuroscience*.
- Millidge, Tschantz, & Buckley, 2022. "Predictive Coding Approximates Backprop along Arbitrary Computation Graphs." *Neural Computation*. arXiv:2006.04182.
- Millidge et al., 2021. "Neural Kalman Filtering." arXiv:2102.10021.
- Clark, 2013. "Whatever next? Predictive brains, situated agents, and the future of cognitive science." *Behavioral and Brain Sciences*.
