# ADR 0025 — SAE L1 Coding Loss

**Date:** 2026-06-17
**Status:** Closed — L1 not adopted

## Context

ADR 0005 described a "full differentiable SAE" combining reconstruction loss with a coding (sparsity) loss. The original system trained the SAE reconstruction-only (MSE), enforcing sparsity via hard k-WTA (K_ACTIVE=20/N_UNITS=1024) — encoder weights already get gradients via the straight-through estimator (mask applied post-projection, backward flows through unmasked z). Backlog item D: add the L1 coding term (option a) or replace k-WTA with ReLU+L1 (option b). After item B (LM-contextualized language embeddings) raised p_lang 0.557→0.85-0.91, prior cosines were saturating with remaining headroom in reconstruction MSE (~0.03).

Chose option (a) over (b): B's win was upstream of the encoder — it successfully turned improved inputs into good codes, suggesting k-WTA isn't the bottleneck. Option (b) would eliminate duty-cycle dead-unit prevention and introduce a new λ hyperparameter to retune for equivalent sparsity — unfavorable cost/risk.

## Decision

Optional L1 penalty on sparse activations added to NREM SAE reconstruction loss: `total_loss = MSE(x, decoder(sparse_z)) + λ·|sparse_z|₁/(b·N_UNITS)`, where `sparse_z = z_batch*mask` (k-WTA binary mask) — same accumulation structure as MSE, per-region and summed. `--sae-l1-coeff` flag (default 0.0=off), runtime field on `PerceptionProcessor`, no signature changes to `run_sleep_cycle`/`fused_replay_wake_step_batch`. k-WTA, duty-cycle boosting, and other encoder mechanics unchanged.

## Alternatives considered

- **Option (b): replace k-WTA with ReLU+L1.** Needs dead-unit mitigation, adds λ-tuning just to match current sparsity, removes the principled fixed-k guarantee. Skipped unless (a) yields meaningful results worth a follow-up.
- **Option (a) with L1 on pre-mask z.** Would penalize all 1024 activations including masked-off ones, pulling toward global small activations — not the target (should penalize active units' magnitudes only).
- **No L1 at all.** MSE~0.03 with priors at 0.76-0.98 leaves little obvious room — this experiment is a bounded falsification of "coding loss helps reconstruction."

## A/B Validation

Pre-stated win: s3-5 avg `mean_output_mse` drops >10% vs. control. λ=0.1 tested (est. ~6% MSE contribution at typical active magnitudes). Fresh cold memory both arms, same clips/session count (5×2).

| Arm | s3-5 avg MSE |
|---|---|
| Control (λ=0) | 0.3145 |
| Treatment (λ=0.1) | 0.3981 |

Δ=**−27% (treatment worse)**. Prior cosines comparable between arms (not degraded by L1) — rules out "L1 too high, try smaller," since the harm is in MSE convergence rate, not representation quality. A smaller λ would likely be negligible at the ~0.03 steady-state MSE the main campaign already achieved (sessions 40+).

**Conclusion:** L1 coding loss on k-WTA sparse activations doesn't help. Item D closed. `--sae-l1-coeff` stays in the codebase at default 0.0 for future experimentation.

## What would reopen this

Evidence that dead units accumulate at scale and duty-cycle fails to prevent it → revisit option (b). A specific reconstruction bottleneck identified at higher session counts → re-test L1 at λ≤0.01 on a warm memory store, not cold.
