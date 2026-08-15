# ADR 0032 — MLP projection (640→1024→5×896, ADR 0012 item M)

**Date:** 2026-06-21
**Status:** Accepted

## Context

The episode→LM-space projection maps a 640-dim episode vector to 5 region tokens (896-dim each), used for: (1) gain computation (GainWorker: episode→tokens→Qwen forward→per-region activation norms→gain multipliers), (2) generation (ActionDecoder: recalled/current episode→tokens prepended to LM context), (3) training target (NREM distillation: projection trained by matching mean-pooled region tokens to the mean LM hidden state of the corresponding text target). The linear projection (640→4480, ~2.9M params) converged at `mean_proj_loss≈0.0007` across sessions 46-50 of the mixed campaign, no meaningful decline in the last 4 (range 0.00066-0.00115, oscillating) — matching ADR 0012 item M's plateau trigger.

## Decision

Replace the single `Linear` in `ProjModel` with a two-layer MLP: `W1:[640,1024]` (655,360 params) → `ReLU` → `W2:[1024,4480]` (4,587,520 params), total 5,242,880 params.

**Why ReLU not GELU:** GainWorker reimplements the projection as a hand-coded matmul (no Burn, no library) — ReLU (`max(0,x)`) is trivial to replicate correctly, GELU would need a polynomial approximation or erf with correctness risk. **Why 1024 hidden:** exceeds EPISODE_DIM(640), forcing the first layer to expand before W2 compresses — more representational capacity while staying within ~6M params. **Cold start, not warm:** the old linear `W[640,4480]` can't be repurposed (W2 expects a [1024]-dim input); warm-start heuristics (padding, pseudoinverse) add complexity with no clear benefit given how fast a 5M-param MLP re-learns on fresh NREM steps. Old `decoder_weights.bin` becomes incompatible — `load_weights` failure silently resets to fresh (existing behavior) with a specific log message.

**What M can/can't fix:** better projection capacity → lower `proj_loss` → more faithful episode→token mapping. Persistent hallucination is *not* expected to change — Qwen is frozen and has never seen the prefix token distribution (that's item L's, LoRA fine-tuning, domain). Judge M by `proj_loss` only.

## Changes required

`ProjModel` gains `fc1:Linear<B>`(640→1024)+`fc2:Linear<B>`(1024→4480), removes `projection`. `project_flat()` = `relu(fc1(episode))→fc2`. `projection_step*()` uses the same forward path, no other change. `proj_weights_flat()` returns `[W1_flat|W2_flat]` (655,360 + 4,587,520 f32). `gain_from_episode()` 2-layer forward: `h[j]=max(0,Σ_i episode[i]*W1[i*1024+j])` for j in 0..1024, `proj_flat[k]=Σ_j h[j]*W2[j*n_out+k]` for k in 0..4480. `load_weights`/`save_weights`: Burn serialises by field name — a file with `projection.*` keys fails to load into `fc1.*`/`fc2.*` keys, existing fresh-init+warn fallback handles it, architecture change logged explicitly.

## Alternatives considered

- **GELU activation.** More expressive, matches Qwen's MLP — rejected, non-trivial to hand-implement correctly in `gain_from_episode` without a library.
- **Larger hidden dim (2048/4096).** More capacity — the 4480-output already constrains expressivity, 1024 hidden is sufficient to exceed linear capacity. Deferred.
- **Warm start via SVD** (decompose old W≈W2@W1 via truncated SVD). Principled but adds a scipy dependency for a one-off init — the linear converged at 0.0007, the MLP converges at least as fast from cold.

## Consequences

`decoder_weights.bin` from any prior session becomes incompatible — first run after deploy resets proj weights silently, logs a warning. GainWorker receives a larger `proj_w` vec (5.24M f32≈21MB vs. 2.87M f32≈11MB) — no structural channel change. Training time per NREM step increases slightly (one extra matmul), negligible vs. ORT forward. Evaluate via `mean_proj_loss` trend across 3-5 post-deploy sessions — expect recovery to ~0.0007 within 1 session, potentially improving further.
