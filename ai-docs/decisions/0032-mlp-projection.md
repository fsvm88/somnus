# ADR 0032 — MLP projection (640→1024→5×896, ADR 0012 item M)

**Date:** 2026-06-21
**Status:** Accepted

## Context

The episode→LM-space projection maps a 640-dim episode vector to 5 region tokens (each 896-dim), used for:
1. Gain computation (GainWorker): episode → 5 region tokens → Qwen forward → per-region activation norms → gain multipliers.
2. Generation (ActionDecoder): recalled/current episode → 5 region tokens prepended to the LM context.
3. Training target (NREM distillation): projection weights trained by matching mean-pooled region token to mean LM hidden state of the corresponding text target.

The linear projection (640→4480, ~2.9M params) converged at `mean_proj_loss ≈ 0.0007` across sessions 46–50 of the mixed campaign, with no meaningful decline in the last 4 sessions (range 0.00066–0.00115, oscillating). This matches the plateau trigger from ADR 0012 item M.

## Decision

Replace the single `Linear` in `ProjModel` with a two-layer MLP:

```
W1: [640, 1024]  (655,360 params)
act: ReLU (element-wise max(0, x))
W2: [1024, 4480] (4,587,520 params)
Total: 5,242,880 params
```

**Why ReLU and not GELU:** GainWorker reimplements the projection as a hand-coded matmul (no Burn context, no library). ReLU is `max(0, x)` — trivial to replicate correctly. GELU would require a polynomial approximation or erf, with correctness risk.

**Why 1024 hidden dim:** 1024 > EPISODE_DIM (640) forces the first layer to expand before compressing in W2, increasing representational capacity. Keeps within ~6M params.

**Cold start, not warm start:** The old linear W [640, 4480] cannot be directly repurposed — W2 expects a [1024]-dim input. Warm-start heuristics (padding, pseudoinverse) add complexity without clear benefit given how quickly a 5M-param MLP re-learns on fresh NREM steps. Old `decoder_weights.bin` files are incompatible; `load_weights` failure silently resets to fresh (existing behaviour) with a specific log message.

**What M can and cannot fix:** Better projection capacity → lower `proj_loss` → more faithful episode→token mapping. Persistent hallucination is *not* expected to change: Qwen is frozen and has never seen our prefix token distribution. Hallucination is L's domain (LoRA fine-tuning). Judge M by `proj_loss` only.

## Changes required

1. `ProjModel`: add `fc1: Linear<B>` (640→1024) and `fc2: Linear<B>` (1024→4480); remove `projection`.
2. `project_flat()`: forward is `relu(fc1(episode))` → `fc2`.
3. `projection_step*()`: training loops use the same forward path, no change needed beyond model definition.
4. `proj_weights_flat()`: return `[W1_flat | W2_flat]` concatenated. W1_flat: 655,360 f32; W2_flat: 4,587,520 f32.
5. `gain_from_episode()`: 2-layer forward:
   - `h[j] = max(0, Σ_i episode[i] * W1[i * 1024 + j])` for j in 0..1024
   - `proj_flat[k] = Σ_j h[j] * W2[j * n_out + k]` for k in 0..4480
6. `load_weights`/`save_weights`: Burn serialises by field name; a file with `projection.*` keys will fail to load into a model with `fc1.*`/`fc2.*` keys — existing fallback (fresh init + warn) handles this correctly. Log the architecture change explicitly.

## Alternatives considered

**GELU activation:** More expressive, matches Qwen's MLP. Rejected: non-trivial to hand-implement correctly in gain_from_episode without a library.

**Larger hidden dim (2048 or 4096):** More capacity, more params. 4480-output already constrains expressivity; 1024 hidden is sufficient to exceed the linear capacity. Deferred.

**Warm start via SVD:** Decompose old W ≈ W2 @ W1 using truncated SVD. Principled but adds a scipy dependency for a one-off init. The linear converged at 0.0007; the MLP will converge at least as fast from cold.

## Consequences

- `decoder_weights.bin` from any prior session becomes incompatible; first-run after deploy resets proj weights silently and logs a warning.
- GainWorker receives a larger `proj_w` vec (5.24M f32 = ~21 MB vs 2.87M f32 = ~11 MB). No structural change to the worker channel.
- Training time per NREM step increases slightly (one extra matmul); negligible vs ORT forward pass.
- Evaluate via `mean_proj_loss` trend across 3–5 sessions post-deploy; expect it to recover to ~0.0007 within 1 session and then potentially improve further.
