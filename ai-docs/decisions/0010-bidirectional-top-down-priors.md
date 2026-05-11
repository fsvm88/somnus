# ADR 0010 — Bidirectional Top-Down Priors

**Date:** 2026-05-11  
**Status:** Accepted

## Context

The architecture doc commits to a bidirectional binding area: the integrative layer both composes regional inputs into an episode (bottom-up) and sends top-down signals back to the regions to bias their interpretation of the next input. The canonical example is McGurk-style disambiguation: visual evidence of mouth motion should raise the prior on "speech" in the audio-speech region, surfacing a faint voice that bottom-up audio alone would suppress under noise.

ADR 0009 added output heads that map the episode vector (512-dim) back to each region's encoder dimension — exactly the feedback signal needed. The top-down prior mechanism reuses those heads at no extra parameter cost.

## Decision

After each episode is produced, run the four output heads in inference mode to generate **predicted encoder embeddings** for each region. On the *next* encoder input, blend the raw embedding with the stored prediction:

```
biased_input_i = (1 - β) * raw_encoder_i + β * prior_prediction_i
```

Feed `biased_input_i` to the k-WTA layer. Store `raw_encoder_i` in `EpisodeTrace.regional` for SAE and output-head training targets.

`β` (prior blend weight) is a **CLI arg defaulting to 0.0** (mechanism present, off by default). Enable only after `output_mse` has converged in a baseline run — the output heads start with random weights and their predictions are noise until trained. Injecting 20% random noise from the start would pollute encoder inputs during the period the heads need to learn from them.

### Observable: prior fidelity

When β > 0 and a prior exists, log the cosine similarity between the prior prediction and the raw encoder input at INFO per ingest call. A rising `prior_cosine` over a session indicates the heads are learning useful predictions. This is the primary metric for evaluating whether the top-down path is doing anything meaningful.

### No new weights

The four output heads are the feedback path. `prior_predictions: [Option<Vec<f32>>; 4]` is added as a field on `PerceptionProcessor` and updated after every episode. Plain `Vec<f32>` — detached from the computation graph, no gradient tracking needed.

### Timing

Prior is always one episode behind (from episode t−1 to bias encoding at t). This is correct for predictive coding: the current belief (last episode) generates a prediction; the prediction biases processing of the next observation; the prediction error updates the model. No settling loop — single-pass.

## Consequences

**Wake/sleep k-WTA inconsistency (known, acceptable).**  
During wake, k-WTA sees `biased_input` (when β > 0). During sleep, `replay_step` trains k-WTA + SAE on `raw_encoder` (stored in `EpisodeTrace.regional`). Same layer, two input regimes. At β = 0.2 the bias is mild; the gap is dominated by the direct input signal. The alternative — store biased inputs and train against biased targets — was rejected because it makes the output head loss circular (predict the biased input that was itself derived from the prior prediction). Training on raw is the ground-truth anchor.

**Cold-start gate.** β = 0.0 is the safe default. The prior mechanism ships in the binary; tuning it on is a deliberate one-line flag change after a baseline observation.

**No settling loop.** The iterative refinement scheme from `multimodal-binding.md` (run binding + feedback 2–3 times per input) is not implemented. Single-pass is consistent with real-time wake operation and avoids per-input iteration overhead. If iterative refinement proves necessary for disambiguation quality, it is an incremental upgrade to this mechanism.

**Snapshot compatibility.** `prior_predictions` is in-memory state (not saved). On restore, it starts as `[None; 4]` and warms up within one episode — no snapshot format change.

## Alternatives considered

- **Separate feedback weight matrices** (not reusing output heads): more expressive (can learn different features for feedback vs. reconstruction), adds ~852K params, loses the output-head-as-feedback elegance. Deferred.
- **Post–k-WTA injection** (bias the dense activation inside `to_dense_n` before binding): avoids the wake/sleep training inconsistency but requires structural changes to `BindingArea::forward` (which currently takes `&[&SparseActivation]`). More invasive; deferred.
- **Gating** (learned scalar gate per region, not additive blend): requires training a gating network; overkill given that β accomplishes the same effect as a fixed gate. Can revisit once the additive blend is calibrated.
