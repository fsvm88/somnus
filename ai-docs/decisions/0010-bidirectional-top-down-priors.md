---
status: accepted
date: 2026-05-11
deferred:
  - id: post-kwta-prior-injection
    what: Inject prior feedback after k-WTA (into to_dense_n output) rather than before, to survive the same competitive selection as raw input
    when: If pre-k-WTA injection shows poor prior uptake — ADR 0011 settling loop may make this unnecessary
  - id: per-region-gating
    what: Replace fixed β blend weight with a learned per-region scalar gate, allowing different regions to accept different amounts of top-down influence
    when: After prior_cosine per region is stable across sessions and per-region β tuning shows clear benefit
---

# ADR 0010 — Bidirectional Top-Down Priors

**Date:** 2026-05-11
**Status:** Accepted

> **Decision:** After each episode, output heads predict next encoder embeddings; on next ingest, blend raw input with stored prediction (RECALL_BLEND=0.5); prior_cosine logged per region.
> **Keywords:** top-down priors, prior blend, blend_with_prior, prior_cosine, RECALL_BLEND, predictive coding, bidirectional, region prior

## Context

`architecture.md` commits to a bidirectional binding area: composes regional inputs bottom-up, and sends top-down signals back to bias interpretation of the next input — canonically, McGurk-style disambiguation (visual mouth-motion raises the audio-speech "speech" prior, surfacing a faint voice noise would otherwise suppress). ADR 0009's output heads (episode vector→each region's encoder dim) are exactly the needed feedback signal — the top-down prior mechanism reuses them at no extra parameter cost.

## Decision

After each episode, run the five output heads in inference mode for **predicted encoder embeddings** per region. On the *next* encoder input, blend raw with the stored prediction: `biased_input_i = (1-β)*raw_encoder_i + β*prior_prediction_i`, fed to k-WTA; `raw_encoder_i` stays stored in `EpisodeTrace.regional` for SAE/output-head training targets.

`β` is a CLI arg, **default 0.0** (mechanism present, off) — enable only after `output_mse` converges in a baseline run, since random output-head weights produce noise predictions that would pollute encoder inputs during the period the heads need to learn from them.

**Observable — prior fidelity:** when β>0 and a prior exists, log cosine similarity between prior prediction and raw encoder input at INFO per ingest — a rising `prior_cosine` over a session is the primary metric for whether the top-down path does anything meaningful.

**No new weights:** the five output heads are the feedback path. `prior_predictions: [Option<Vec<f32>>;5]` on `PerceptionProcessor`, updated after every episode — plain `Vec<f32>`, detached, no gradient tracking. Language head (index 4) participates but stays None until an episode with language input occurs.

**Timing:** prior is always one episode behind (t-1 biases encoding at t) — correct for predictive coding (current belief generates a prediction, prediction biases next observation processing, prediction error updates the model). No settling loop, single-pass.

## Consequences

**Wake/sleep k-WTA inconsistency (known, acceptable):** wake sees `biased_input` (β>0); sleep's `replay_step` trains on `raw_encoder` (stored). Same layer, two input regimes — at β=0.2 the bias is mild, dominated by the direct signal. Storing/training against biased inputs was rejected — it would make the output-head loss circular (predicting the biased input that was itself derived from the prior prediction); training on raw is the ground-truth anchor. **Cold-start gate:** β=0.0 default is safe, tuning on is a deliberate one-line flag change after a baseline observation. **No settling loop:** the iterative refinement scheme (`multimodal-binding.md`) isn't implemented — single-pass matches real-time wake operation, avoids per-input iteration overhead; an incremental upgrade if disambiguation quality needs it (done — ADR 0011). **Snapshot compatibility:** `prior_predictions` is in-memory only, not saved — restores as `[None;4]`, warms up within one episode, no snapshot format change.

## Alternatives considered

- **Separate feedback weight matrices** (not reusing output heads). More expressive (different features for feedback vs. reconstruction), adds ~852K params, loses the output-head-as-feedback elegance. Deferred.
- **Post-k-WTA injection** (bias the dense activation inside `to_dense_n` before binding). Avoids the wake/sleep inconsistency but needs structural changes to `BindingArea::forward` (takes `&[&SparseActivation]`) — more invasive. Deferred.
- **Gating** (learned scalar gate per region, not additive blend). Needs a gating network — overkill when β accomplishes the same effect as a fixed gate; revisit once the additive blend is calibrated.
