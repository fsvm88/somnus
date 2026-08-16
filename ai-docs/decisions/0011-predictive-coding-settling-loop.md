# ADR 0011 — Predictive-Coding Settling Loop

**Date:** 2026-05-11
**Status:** Accepted

> **Decision:** After try_bind, run a fixed-budget settling loop; each iteration re-runs encoder+binding with prior-blended inputs until prior_cosine converges or budget exhausted. ORT enc+bind used when sessions active.
> **Keywords:** settle, settling loop, settle_episode, prior_cosine, SETTLE_EPS, SETTLE_COSINE_GATE, settle_steps, ORT settle, Path A/B/C

## Context

ADR 0010's top-down priors bias the *next* episode's encoder inputs with predictions from the *current* episode — a one-step, inter-episode path, not an iterative refinement of the *current* episode. Predictive coding (Friston FEP, `research/predictive-coding.md`) suggests perception is iterative — the brain converges on a representation minimizing prediction error within a single sensory moment, not just across moments. ADR 0010 explicitly deferred this ("iterative refinement is an incremental upgrade"); this ADR implements it.

## Decision

After `try_bind` produces an episode vector, run a fixed-budget settling loop before storing the episode or updating inter-episode priors. Each iteration: (1) feed the current episode vector through the five output heads (inference mode, no grad) for per-region predictions; (2) blend each prediction with the raw encoder input, `settle_input_i = (1-settle_β)×raw_i + settle_β×pred_i`; (3) run each region's k-WTA on `settle_input_i` via **`forward_no_dc`** (new — skips the duty-cycle EMA update, since settling must not update duty cycles: calling `forward` N times/input would inflate the effective learning rate N× and corrupt homeostasis); (4) re-bind the new sparse activations → refined episode vector; (5) compute `‖ep_new-ep_old‖₂`, log at INFO for calibration, stop early if `delta<SETTLE_EPS` (default 1.0).

The *settled* vector feeds `update_prior_predictions` (ADR 0010), episodic-memory storage, and `wake_reconstruct_step`'s target derivation. `regional_inputs` (raw encoder inputs) are **not** affected by settling — sleep training (`replay_step`, SAE MSE) always trains against raw inputs, avoiding a circular-target problem (output heads must predict raw inputs, not inputs already shaped by their own predictions).

**New params:** `settle_steps` (usize, default 0 at the time — max settling iterations, 0 disables the loop entirely); `settle_beta` (f32, default 0.5 — prediction fraction blended into settling input, 0.0=raw-only/1-step no-op, 1.0=pure prediction). Both CLI args; default 0 preserved existing behavior at the time, no snapshot incompatibility. **Amendment (2026-08-16, ADR 0047):** `settle_steps` default raised 0→1 — every real campaign has run at 1 since `f44d5ee` (2026-05-31), and `SETTLE_EPS`'s calibration (see Consequences below) is itself tuned for 1-iteration convergence, not 0. The cold-start WARN this section describes still fires (metrics-only, not a functional gate) and now applies to any snapshot-less `somnus` invocation, not just campaign runs.

**Cold-start gate:** with `settle_steps>0` but no prior snapshot (random output heads), the binary logs a startup WARN — random predictions at `settle_beta>0` would distort k-WTA inputs during settling, potentially slowing the heads' own convergence. Recommended ramp: `settle_steps=0` until `output_mse` stabilises, then enable settling.

## Consequences

**Duty-cycle integrity:** settling uses `forward_no_dc` throughout; the primary ingest-path `forward` call (before settling) still updates duty cycles once per input — homeostasis unaffected. **Latency:** each step runs 5 output-head forwards + 5 k-WTA linear forwards + binding attention (incl. GRU context), all CPU, ~0.5-2ms/iteration at default sizes — `settle_steps=3` adds ≤6ms/episode, acceptable for the current non-real-time loop. **No new weights** — `forward_no_dc` is a pure read path. **`regional_inputs` vs. episode-vector divergence:** the settled vector may not be exactly reconstructible from stored raw inputs via one forward pass, since settling applies blended inputs — intentional, the settled representation is the *refined belief* while `regional` preserves the *ground-truth observation* for training (parallels ADR 0010's wake/sleep inconsistency, acceptable at small β). **SETTLE_EPS calibration:** 4.0 in the recommended campaign config (typical 640-dim episode-vector L2 norm is higher than the original 512-dim estimate) — session metrics show settling typically converges in 1-2 iterations at this threshold.

## Alternatives considered

- **Post-k-WTA injection** (modify `to_dense_n` output instead of k-WTA input). Avoids re-running the linear layer but loses k-WTA sparsification of the feedback signal — the sparsification is the point, feedback should survive the same competitive selection raw input does.
- **Gradient-based settling** (run `backward` through the loop to update weights per episode). Too slow per-step, couples settling to the training loop — settling is inference-time, gradient flow stays in sleep.
- **No early stopping.** Wastes compute when already converged — delta logging gives the calibration data to know whether early stopping fires in practice.
