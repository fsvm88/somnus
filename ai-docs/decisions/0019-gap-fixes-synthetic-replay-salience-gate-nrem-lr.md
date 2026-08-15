---
status: accepted
date: 2026-05-30
deferred:
  - id: lang-last-token-pooling
    what: Replace LangEmbedder mean-pool of embed table with Qwen forward pass + last-token hidden state for richer language input
    when: Medium effort; worthwhile when language region prior_cosine plateaus and better input representation is the bottleneck; see ADR 0014
---

# ADR 0019 — Gap fixes: synthetic replay, salience-gated wake reconstruct, salience → NREM LR

**Date:** 2026-05-30
**Status:** Accepted

> **Decision:** Three gradient-quality improvements: (1) synthetic NREM replay (REM recombinations pushed into hot buffer), (2) salience-gated wake reconstruct (WAKE_RECONSTRUCT_SALIENCE=0.8 threshold), (3) salience-scaled NREM LR (LR_SALIENCE_CAP=2.0).
> **Keywords:** synthetic replay, salience gate, NREM LR, WAKE_RECONSTRUCT_SALIENCE, LR_SALIENCE_CAP, LR_SALIENCE_MAX_SCALE, prior_beta, ep_salience

## Context

Three low-effort improvements identified after the architectural roundup (A-G commits), all targeting gradient signal quality without architecture changes. Item 4 (language last-token pooling) was dropped: `LangEmbedder` does a static embed-table lookup, not a Qwen forward pass — "last token of a causal LM has full context" doesn't apply; mean-pool is strictly better for a context-free table over 1-3 token fragments. A real last-token version would need running `qwen25_lm.onnx` per line (medium effort, separate decision).

## Decisions

**1 — Synthetic NREM replay.** REM creative recombinations (blended episode pairs) now push into the hot episodic buffer after `episodic.clear()`, so the next NREM cycle trains on them — previously they were produced, measured for novelty MSE, and discarded (the measurement confirmed novelty but the network never trained on it). Blended regional inputs use the same `REM_SYNTH_ALPHA=0.5` as episode synthesis, giving NREM consistent encoder-level targets. Salience = the source-pair L2 distance that qualified the pair for synthesis — a natural novelty proxy (widely-separated parents→high salience→replayed first next cycle). Episode vectors L2-normalised before push, matching real-episode storage. Synthetics enter the warm/cold recall corpus on the next spill — intentional, creative combinations surviving NREM consolidation should influence future recollection. **Would invalidate:** synthetic pollution measurably degrading `gist_recall_rate`/`warm_recall_l2` over multi-session runs — reduce `REM_SYNTH_MAX_N` or add a disable-synthetic-cold-spill flag.

**2 — Salience-gated wake reconstruction.** `wake_reconstruct_step` skips when an episode's warm-recall L2 is below `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD=0.8` — familiar frames (low L2, close to many prior episodes) produce near-zero gradient (output heads already predict them well), wasting backward passes and potentially reinforcing stale predictions. Salience source: warm-recall L2 from `record_episode`, now called *before* `wake_reconstruct_step` at all three call sites (audio/video/language) — avoids a double warm query, one consistent salience definition throughout wake. Episodes with no warm neighbours (fresh memory) return `f32::MAX`, always passing the gate. Threshold recalibrated 2.0→0.8 after L2-normalisation was added to `record_episode` (unit vectors have max L2=2.0, so 2.0 passed everything) — within L2=0.8 frames are effectively identical, above it reconstruction gradients carry genuine information.

**3 — Salience → NREM LR.** `nrem_lr` now accepts `batch_salience:f32`, applying a bounded 1×-2× multiplier on top of the existing cosine×trend_scale schedule — novel traces (high warm-recall L2) get more gradient, familiar ones (already well-represented, converge quickly) avoid over-training. Sentinel handling: `f32::MAX` (no warm neighbour, fresh-run first cycle) maps to 0.5 novelty (neutral ~1.5× multiplier) — caught by the `>=f32::MAX/2.0` guard before normalisation, avoiding the inf/NaN risk if all traces are sentinel. Bounds: `LR_SALIENCE_CAP=2.0` normalises raw L2 to [0,1] (recalibrated from 10.0 for L2-normalised unit vectors, max distance 2.0); `LR_SALIENCE_MAX_SCALE=2.0` caps the multiplier — combined max LR = `LR_MAX×LR_SALIENCE_MAX_SCALE=2e-3`, well within stable AdamW range at these batch sizes. Traces are already sorted salience-descending before NREM, so high-salience batches land early in the cycle when cosine annealing is also at peak LR — the two effects compound deliberately.
