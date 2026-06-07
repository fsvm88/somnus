# ADR 0011 — Predictive-Coding Settling Loop

**Date:** 2026-05-11  
**Status:** Accepted

> **Decision:** After try_bind, run a fixed-budget settling loop; each iteration re-runs encoder+binding with prior-blended inputs until prior_cosine converges or budget exhausted. ORT enc+bind used when sessions active.  
> **Keywords:** settle, settling loop, settle_episode, prior_cosine, SETTLE_EPS, SETTLE_COSINE_GATE, settle_steps, ORT settle, Path A/B/C

## Context

ADR 0010 introduced top-down priors that bias the *next* episode's encoder inputs with predictions derived from the *current* episode. This is a one-step, inter-episode feedback path. It does not allow the system to iteratively refine the *current* episode by cycling feedback through the k-WTA layers multiple times before committing to a representation.

Predictive coding (Friston FEP; see `ai-docs/research/predictive-coding.md`) suggests that perception is an iterative inference process: the brain converges on a representation that minimises prediction error within a single sensory moment, not just across moments. A settling loop is the straightforward algorithmic analogue.

ADR 0010 explicitly deferred this: "iterative refinement … is an incremental upgrade to this mechanism." This ADR implements it.

## Decision

After `try_bind` produces an episode vector, run a fixed-budget settling loop before storing the episode or updating the inter-episode priors (ADR 0010). Each iteration:

1. Feed the *current* episode vector through the five output heads (inference mode, no grad) to generate per-region predictions.
2. Blend each prediction with the corresponding raw encoder input:

   ```
   settle_input_i = (1 − settle_β) × raw_i + settle_β × pred_i
   ```

3. Run each region's k-WTA layer on `settle_input_i` using **`forward_no_dc`** — a new method that skips the duty-cycle moving average update. Settling iterations must not update duty cycles; calling `forward` N times per input would inflate the effective learning rate by N× and corrupt homeostasis.
4. Re-bind the new sparse activations → refined episode vector.
5. Compute the L2 norm of the delta `‖ep_new − ep_old‖₂`. Log it at **INFO** for calibration. If `delta < SETTLE_EPS` (default 1.0), stop early.

The *settled* episode vector is returned to the caller and used for:
- `update_prior_predictions` (ADR 0010 inter-episode priors)
- Storage in episodic memory (the episode vector agents recall against)
- `wake_reconstruct_step` (output-head training target derivation)

The `regional_inputs` (raw encoder inputs) are **not affected** by settling. Sleep training (`replay_step`, SAE MSE) always trains against raw inputs. This avoids the circular-target problem: output heads must predict raw inputs, not inputs that were already shaped by their own predictions.

### New parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `settle_steps` | `usize` | `0` | Max settling iterations per episode. `0` disables the loop (existing behaviour). |
| `settle_beta` | `f32` | `0.5` | Fraction of output-head prediction blended into settling input. `0.0` = raw only (loop converges in 1 step, no-op). `1.0` = pure prediction. |

Both are CLI args. `settle_steps = 0` default preserves all existing behaviour; no snapshot incompatibility.

### Cold-start gate

When `settle_steps > 0` but no prior snapshot exists (output heads are still random), the binary logs a startup WARN. Random output head predictions injected at `settle_beta > 0` will distort k-WTA inputs during the settling loop, potentially slowing convergence of the output heads themselves. The recommended ramp: run with `settle_steps=0` until `output_mse` has stabilised, then enable settling.

## Consequences

**Duty-cycle integrity.** Settling uses `forward_no_dc` throughout; the primary `forward` call (from the ingest path, before settling begins) still updates duty cycles once per encoder input. Homeostasis is unaffected.

**Latency.** Each settling step runs: 5 × output head forward + 5 × k-WTA linear forward + binding attention forward (including GRU temporal context). All on CPU; estimated ~0.5–2 ms per iteration at default sizes. With `settle_steps = 3` this adds ≤6 ms per episode — acceptable for the current non-real-time research loop.

**No new weights.** `forward_no_dc` is a pure read path on existing k-WTA weights. No new parameters.

**`regional_inputs` vs. episode vector divergence.** The settled episode vector may not be exactly reconstructible from the stored `regional` (raw encoder) inputs via a single forward pass, because settling applies blended (not raw) inputs. This is intentional: the settled representation is the *refined belief*, while `regional` preserves the *ground-truth observation* for training. This parallels the ADR 0010 wake/sleep inconsistency (acceptable at small β).

**SETTLE_EPS calibration.** Calibrated value is 4.0 (`SETTLE_EPS=4.0` in the recommended campaign config). Typical episode vector L2 norm is higher for 640-dim than the original 512-dim estimate. Session metrics show settling typically converges in 1–2 iterations at this threshold.

## Alternatives considered

- **Post–k-WTA injection** (modify `to_dense_n` output instead of k-WTA input): avoids re-running the linear layer but loses the k-WTA sparsification of the feedback signal. The sparsification is the point — feedback should survive the same competitive selection that the raw input does.
- **Gradient-based settling** (run `backward` through the loop to update k-WTA weights per episode): too slow per-step and couples settling to the training loop. Settling is inference-time; gradient flow stays in sleep.
- **No early stopping** (always run `settle_steps` iterations regardless of convergence): wastes compute when the system is already converged. Delta logging gives the calibration data to know whether early stopping ever fires in practice.
