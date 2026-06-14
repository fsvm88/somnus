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

The system currently has no generative or predictive path. The binding area produces a 640-dim episode vector during wake, but nothing uses that vector to reconstruct the sensory inputs that produced it. Adding output heads achieves two things:

1. **Second training signal on the episode vector.** The binding area now must produce representations that can reconstruct each region's encoder input — not just distinguish episodes. This is the predictive-coding loop at the integrative level.
2. **Foundation for mental imagery.** A retrieved episode vector can be decoded into approximate encoder-space representations, enabling "what would this recall sound/look like?" queries. This precursor is needed before bidirectional priors (ADR-TBD) become tractable.

## Decision

Add **five `linear_no_bias(EPISODE_DIM, region_dim)` output heads** — one per region — to `PerceptionProcessor`. Train with MSE loss during wake (one optimizer step per episode). Gradients flow through the heads and back through the binding area.

### Architecture

```
episode vector [640]
  ↓  output_heads[i] (linear_no_bias, 640 → region_dim_i)
  ↓  predicted encoder embedding [region_dim_i]
  ↓  MSE vs. actual encoder embedding [region_dim_i]
```

| Head | Input | Output | Params |
|---|---|---|---|
| output_heads[0] (speech)   | 640 | 384 | 245,760 |
| output_heads[1] (env)      | 640 | 512 | 327,680 |
| output_heads[2] (visual)   | 640 | 384 | 245,760 |
| output_heads[3] (symbolic) | 640 | 384 | 245,760 |
| output_heads[4] (language) | 640 | 896 | 573,440 |
| **Total**                  |     |     | **1,638,400** |

The language head (index 4, 896-dim) was added in Phase 6 (ADR 0013). Empty-region masking applies: the language head contributes zero loss when no text was ingested on that episode.

### Wake training (`wake_reconstruct_step`)

Called once per episode, immediately after `try_bind` produces an episode vector, using the raw (pre-blend) episode vector and the stored `regional_inputs`. Non-empty regions contribute equally to the loss; empty regions are masked out (zero contribution). The optimizer is shared with the rest of the cognitive layer; the same AdamW instance is used for SAE, binding, and output heads.

### NREM integration: implemented (amendment 2026-06-14)

`EpisodeTrace` was later found to store both `regional` encoder inputs **and** the episode vector (`record.values`), which was set at push time. This made re-running k-WTA+binding unnecessary. The implementation (`fused_replay_wake_step_batch`, committed in `a1c43cb` / `cd4364e`) passes the stored episode vectors directly alongside regional inputs, accumulating both SAE-decoder and output-head losses into a single backward pass — no extra forward pass cost. The O(replay × forward_pass) cost noted in the original deferral does not apply because the episode vector is already available.

**Trade-off vs. the deferred design:** stored episode vectors reflect the binding-area weights at wake time, not at replay time. As NREM updates SAE weights, the output heads learn to reconstruct from slightly stale episode representations. This is expected and acceptable at this stage.

## Alternatives considered

- **Single concatenated head (640 → 2560):** simpler, but makes per-region loss masking awkward and mixes modalities with different scales (Whisper embeddings vs. CLAP embeddings are on different magnitude scales).
- **Bias in heads:** no clear benefit for a regression target that varies with input; consistent with existing region projections (`linear_no_bias` throughout).
- **Separate optimizer for output heads:** would allow independent LR tuning per group. Deferred; the shared AdamW is sufficient for Phase 5 and the NREM neuromodulation schedule already acts on the whole cognitive layer.

## Consequences

- Wake logs now include `output_mse` per episode — an observable learning signal that can be tracked to verify the heads are learning.
- Episode vector is now shaped by two objectives: SAE regional reconstruction (sleep) and output head reconstruction (wake). If these pull in conflicting directions, the binding area will face a multi-objective gradient — expected and desirable at this stage.
- VarMap gains 5 weight tensors (`output_heads.{0,1,2,3,4}.weight`). Old snapshots missing these keys get fresh-initialized heads (warning logged). The language head (index 4) was added in Phase 6; pre-Phase-6 snapshots are missing it and start from scratch.
- `rebuild_linears` extended for the 5 keys.
