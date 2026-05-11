# ADR 0009 — Output / Reconstruction Channel

**Date:** 2026-05-11  
**Status:** Accepted

## Context

The system currently has no generative or predictive path. The binding area produces a 512-dim episode vector during wake, but nothing uses that vector to reconstruct the sensory inputs that produced it. Adding output heads achieves two things:

1. **Second training signal on the episode vector.** The binding area now must produce representations that can reconstruct each region's encoder input — not just distinguish episodes. This is the predictive-coding loop at the integrative level.
2. **Foundation for mental imagery.** A retrieved episode vector can be decoded into approximate encoder-space representations, enabling "what would this recall sound/look like?" queries. This precursor is needed before bidirectional priors (ADR-TBD) become tractable.

## Decision

Add **four `linear_no_bias(EPISODE_DIM, region_dim)` output heads** — one per modality region — to `PerceptionProcessor`. Train with MSE loss during wake (one optimizer step per episode). Gradients flow through the heads and back through the binding area.

### Architecture

```
episode vector [512]
  ↓  output_heads[i] (linear_no_bias, 512 → region_dim_i)
  ↓  predicted encoder embedding [region_dim_i]
  ↓  MSE vs. actual encoder embedding [region_dim_i]
```

| Head | Input | Output | Params |
|---|---|---|---|
| output_heads[0] (speech)   | 512 | 384 | 196,608 |
| output_heads[1] (env)      | 512 | 512 | 262,144 |
| output_heads[2] (visual)   | 512 | 384 | 196,608 |
| output_heads[3] (symbolic) | 512 | 384 | 196,608 |
| **Total**                  |     |     | **851,968** |

### Wake training (`wake_reconstruct_step`)

Called once per episode, immediately after `try_bind` produces an episode vector, using the raw (pre-blend) episode vector and the stored `regional_inputs`. Non-empty regions contribute equally to the loss; empty regions are masked out (zero contribution). The optimizer is shared with the rest of the cognitive layer; the same AdamW instance is used for SAE, binding, and output heads.

### NREM integration: deferred

During NREM replay, only regional encoder inputs are stored in `EpisodeTrace` — the episode vector is not. Training output heads during sleep would require re-running the full k-WTA + binding forward pass for each trace to re-produce the episode vector. This is feasible but changes `replay_step` significantly; deferred to a subsequent step to keep this change focused.

## Alternatives considered

- **Single concatenated head (512 → 1664):** simpler, but makes per-region loss masking awkward and mixes modalities with different scales (Whisper embeddings vs. CLAP embeddings are on different magnitude scales).
- **Bias in heads:** no clear benefit for a regression target that varies with input; consistent with existing region projections (`linear_no_bias` throughout).
- **Separate optimizer for output heads:** would allow independent LR tuning per group. Deferred; the shared AdamW is sufficient for Phase 5 and the NREM neuromodulation schedule already acts on the whole cognitive layer.

## Consequences

- Wake logs now include `output_mse` per episode — an observable learning signal that can be tracked to verify the heads are learning.
- Episode vector is now shaped by two objectives: SAE regional reconstruction (sleep) and output head reconstruction (wake). If these pull in conflicting directions, the binding area will face a multi-objective gradient — expected and desirable at this stage.
- VarMap gains 4 weight tensors (`output_heads.{0,1,2,3}.weight`). Old snapshots missing these keys get fresh-initialized heads via `varmap.load` (warning logged, training resumes from scratch for the heads only).
- `rebuild_linears` extended for the 4 new keys.
