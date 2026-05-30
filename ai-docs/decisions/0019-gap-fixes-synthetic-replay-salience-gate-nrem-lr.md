# ADR 0019 — Gap fixes: synthetic replay, salience-gated wake reconstruct, salience → NREM LR

**Date:** 2026-05-30  
**Status:** Accepted

## Context

Three low-effort improvements identified after the architectural roundup (A–G commits). All target gradient signal quality without changing the model architecture.

Item 4 from the original gap list (language last-token pooling) was dropped: `LangEmbedder` does a static embed-table lookup, not a Qwen forward pass, so "last token of a causal LM has full context" does not apply. Mean-pool is strictly better for a context-free table over 1–3 token fragments. The real last-token version would require running `qwen25_lm.onnx` per line (medium effort, separate decision).

---

## Decisions

### 1 — Synthetic NREM replay

**What:** REM creative recombinations (blended episode pairs) are now pushed into the hot episodic buffer after `episodic.clear()`, so the next NREM cycle trains on them.

**Why:** Previously synthetics were produced, measured for novelty MSE, and discarded — the measurement confirmed they were novel but the network never trained on them.

**Design choices:**
- Blended regional inputs are computed with the same `REM_SYNTH_ALPHA = 0.5` used for episode synthesis, giving NREM consistent encoder-level targets.
- Salience = source-pair L2 distance (the distance that qualified the pair for synthesis). This is a natural novelty proxy: widely-separated parents → high salience → replayed first in the next NREM cycle.
- Episode vectors are L2-normalised before push, matching how real episodes are stored by `record_episode`.
- Synthetics enter the warm/cold recall corpus on the next sleep spill. This is intentional: creative combinations that survive NREM consolidation should influence future recollection.

**What would invalidate this:** if synthetic pollution measurably degrades gist_recall_rate or warm_recall_l2 trajectories over multi-session runs, reduce `REM_SYNTH_MAX_N` or add a flag to disable synthetic spill to cold.

### 2 — Salience-gated wake reconstruction

**What:** `wake_reconstruct_step` is skipped when the episode's warm-recall L2 distance is below `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD = 2.0`.

**Why:** Familiar frames (low warm-recall L2 = close to many prior episodes) produce near-zero gradient: the output heads already predict them well. Spending backward passes on them wastes compute and can reinforce stale predictions if the model has already converged on that content.

**Salience source:** warm-recall L2 computed inside `record_episode`, which is now called *before* `wake_reconstruct_step` at all three call sites (audio, video, language). This avoids a double warm query and uses a single consistent salience definition throughout wake processing. Episodes with no warm neighbours (first run, fresh memory dir) return `f32::MAX`, which always passes the gate.

**Threshold:** 2.0 is below the same-content p95 of 4.1 (calibrated 2026-05-14). Content within L2=2.0 is effectively identical across frames; above that, reconstruction gradients carry genuine information.

### 3 — Salience → NREM LR

**What:** `nrem_lr` now accepts `batch_salience: f32` and applies a bounded salience multiplier (1×–2×) on top of the existing cosine × trend_scale schedule.

**Why:** Novel traces (high warm-recall L2) benefit from more gradient. Familiar traces are already well-represented in warm memory and converge quickly; over-training them wastes NREM time.

**Sentinel handling:** `f32::MAX` (no warm neighbour — first cycle of a fresh run) maps to 0.5 novelty (neutral multiplier ~1.5×). This avoids the inf/NaN risk the advisor flagged: if all traces are sentinel, salience = f32::MAX, which is caught by the `>= f32::MAX / 2.0` guard before normalisation.

**Bounds:** `LR_SALIENCE_CAP = 10.0` normalises raw L2 to [0, 1]; `LR_SALIENCE_MAX_SCALE = 2.0` caps the multiplier. Combined max LR = `LR_MAX × LR_SALIENCE_MAX_SCALE = 2e-3`, still well within a stable AdamW range at these batch sizes.

**Interaction with sort order:** traces are already sorted salience-descending before NREM, so high-salience batches appear early in the cycle when cosine annealing is also at peak LR. The two effects compound deliberately.
