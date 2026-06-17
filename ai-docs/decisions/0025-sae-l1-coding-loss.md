# ADR 0025 — SAE L1 Coding Loss

**Date:** 2026-06-17
**Status:** Closed — L1 not adopted

## Context

ADR 0005 described a "full differentiable SAE" combining reconstruction loss with a coding (sparsity) loss. The original system trained the SAE with reconstruction-only loss (MSE) while enforcing sparsity via hard k-WTA selection (K_ACTIVE=20 out of N_UNITS=1024). The encoder weights do receive gradients via the straight-through estimator — the mask is applied after the linear projection, and the backward pass flows through the unmasked z values.

Item D on the backlog was to add the L1 coding term (option a) or replace k-WTA with ReLU+L1 (option b). After Item B (LM-contextualized language embeddings) raised p_lang from 0.557 to 0.85–0.91, the prior cosine metrics are saturating and the remaining headroom is in reconstruction MSE (~0.03).

The decision to attempt option (a) rather than option (b) rests on the observation that B's win was upstream of the encoder — the encoder successfully turned improved inputs into good codes, suggesting k-WTA is not the bottleneck. Option (b) would replace k-WTA with ReLU+L1, which eliminates duty-cycle dead-unit prevention and introduces a new hyperparameter (λ) that must be tuned to recover equivalent sparsity. The cost/risk ratio is unfavorable.

## Decision

Add an optional L1 penalty on the sparse activations to the NREM SAE reconstruction loss:

```
total_loss = MSE(x, decoder(sparse_z)) + λ · |sparse_z|₁ / (b · N_UNITS)
```

where `sparse_z = z_batch * mask` (mask is the k-WTA binary mask). The L1 term is added per-region and summed, identical structure to how MSE terms are accumulated.

Implementation: `--sae-l1-coeff` flag (default 0.0 = off) stored as a runtime field on `PerceptionProcessor`; no function signature changes to `run_sleep_cycle` or `fused_replay_wake_step_batch`.

k-WTA, duty-cycle boosting, and all other encoder mechanics are unchanged.

## Alternatives considered

**Option (b): Replace k-WTA with ReLU+L1.** Requires dead-unit mitigation, adds λ-tuning overhead just to match current sparsity, and removes the principled fixed-k guarantee. Skipped unless option (a) yields meaningful results and a follow-up investigation makes sense.

**Option (a) with L1 on pre-mask z.** Would penalize all 1024 activations including the masked-off ones, pulling the network toward small activations globally. Not what we want — the penalty should target the active units' magnitudes.

**No L1 at all.** MSE of ~0.03 with all prior cosines at 0.76–0.98 leaves little obvious room; this experiment is a bounded falsification of "coding loss helps reconstruction."

## A/B Validation

Pre-stated win threshold: sessions 3–5 cycle-average `mean_output_mse` drops >10% vs control.
Test λ: 0.1 (estimated L1 contribution ~6% of MSE at typical active magnitudes).
Fresh cold memory for both arms; same clips, same session count (5 sessions × 2 cycles).

**Result (2026-06-17):**

| Arm | s3–5 avg MSE |
|-----|-------------|
| Control (λ=0) | 0.3145 |
| Treatment (λ=0.1) | 0.3981 |

Δ = **−27%** (treatment worse). Prior cosines were comparable between arms (not degraded by L1).
Win threshold not met. L1 at λ=0.1 actively impairs cold-start reconstruction learning.

The prior cosines being healthy in both arms rules out "L1 is too high, try smaller" — the harm is in MSE convergence rate, not representation quality. Smaller λ would likely be negligible at the steady-state MSE of ~0.03 already achieved by the main campaign (sessions 40+).

**Conclusion:** L1 coding loss on k-WTA sparse activations does not help. Item D closed. The `--sae-l1-coeff` flag remains in the codebase at default 0.0 for future experimentation.

## What would reopen this

- Evidence that dead units accumulate at scale and duty-cycle fails to prevent it → revisit option (b).
- A specific reconstruction bottleneck identified at higher session counts → re-test L1 at λ ≤ 0.01 on a warm memory store, not cold.
