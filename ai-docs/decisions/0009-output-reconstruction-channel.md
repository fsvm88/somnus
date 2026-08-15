---
status: accepted
date: 2026-05-11
deferred:
  - id: separate-output-head-optimizer
    what: Give output heads their own AdamW with independent LR schedule, separate from the cognitive layer optimizer
    when: If output heads and k-WTA/binding weights show conflicting gradient directions in the shared optimizer
---

# ADR 0009 — Output / Reconstruction Channel

**Date:** 2026-05-11
**Status:** Accepted

> **Decision:** Five EmaOutputHeads (EPISODE_DIM → region_dim per region) trained with MSE during wake; EMA weights synced from burn after each update cycle.
> **Keywords:** EmaOutputHeads, output heads, reconstruction, wake_reconstruct, MSE, linear_no_bias, EPISODE_DIM, region_dim, EMA

## Context

The system had no generative/predictive path — binding produces a 640-dim episode vector during wake but nothing uses it to reconstruct the inputs that produced it. Output heads achieve two things: (1) a second training signal on the episode vector — binding must produce representations that can reconstruct each region's encoder input, not just distinguish episodes (the predictive-coding loop at the integrative level); (2) foundation for mental imagery — a retrieved episode vector can decode into approximate encoder-space representations, a precursor needed before bidirectional priors become tractable.

## Decision

Five `linear_no_bias(EPISODE_DIM, region_dim)` output heads on `PerceptionProcessor`, one per region, trained with MSE during wake (one optimizer step/episode) — gradients flow through the heads and back through binding.

```
episode vector [640] → output_heads[i] (linear_no_bias, 640→region_dim_i) → predicted encoder embedding → MSE vs. actual
```

| Head | Input | Output | Params |
|---|---|---|---|
| output_heads[0] (speech) | 640 | 384 | 245,760 |
| output_heads[1] (env) | 640 | 512 | 327,680 |
| output_heads[2] (visual) | 640 | 384 | 245,760 |
| output_heads[3] (symbolic) | 640 | 384 | 245,760 |
| output_heads[4] (language) | 640 | 896 | 573,440 |
| **Total** | | | **1,638,400** |

Language head (index 4) added in Phase 6 (ADR 0013); empty-region masking applies — it contributes zero loss when no text was ingested that episode.

**Wake training (`wake_reconstruct_step`):** called once/episode, right after `try_bind` produces an episode vector, using the raw (pre-blend) episode vector + stored `regional_inputs`. Non-empty regions contribute equally to the loss, empty regions masked to zero. Shared optimizer — same AdamW instance for SAE, binding, and output heads.

**NREM integration (amendment 2026-06-14):** `EpisodeTrace` was found to already store both `regional` encoder inputs *and* the episode vector (`record.values`, set at push time), making re-running k-WTA+binding unnecessary. `fused_replay_wake_step_batch` (`a1c43cb`/`cd4364e`) passes stored episode vectors directly alongside regional inputs, accumulating both SAE-decoder and output-head losses into a single backward pass — no extra forward-pass cost, so the O(replay×forward_pass) cost the original deferral worried about doesn't apply. **Trade-off:** stored episode vectors reflect binding weights at wake time, not replay time — as NREM updates SAE weights, output heads learn to reconstruct from slightly stale episode representations. Expected and acceptable.

## Alternatives considered

- **Single concatenated head (640→2560).** Simpler but makes per-region loss masking awkward and mixes modalities with different scales (Whisper vs. CLAP embeddings differ in magnitude).
- **Bias in heads.** No clear benefit for a regression target that varies with input — consistent with `linear_no_bias` used throughout region projections.
- **Separate optimizer for output heads.** Would allow independent per-group LR tuning — deferred, the shared AdamW is sufficient for Phase 5 and the NREM neuromodulation schedule already acts on the whole cognitive layer.

## Consequences

Wake logs now include `output_mse`/episode — an observable learning signal confirming the heads are learning. The episode vector is now shaped by two objectives (SAE regional reconstruction in sleep, output-head reconstruction in wake) — a multi-objective gradient for binding, expected and desirable. VarMap gains 5 tensors (`output_heads.{0..4}.weight`) — old snapshots missing these get fresh-initialized heads (warning logged); the language head (index 4) is missing from pre-Phase-6 snapshots and starts from scratch. `rebuild_linears` extended for the 5 keys.
