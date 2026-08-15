# A/B Validation: DINOv3-ViT-S vs DINOv2-S/14

**Date:** 2026-07-05
**Duration:** 2 sleep cycles per session, 5 sessions each model (`scripts/ab_dinov3.sh`)
**Prerequisite:** Fixed the CWD-relative external-data bug that previously made DINOv3 look broken (gaps.md §6) and the register-token pooling bug in `somnus-encoding/src/video.rs`.

## Results

**Reconstruction error** (`mean_output_mse_wake`, lower=better): DINOv2-S (control) 0.3573 vs. DINOv3-ViT-S (treatment) 0.0971, mean across 10 cycles — **−73%**.

**Per-region MSE:**

| Region | DINOv2 | DINOv3 | Delta |
|---|---|---|---|
| speech | 0.686 | 0.666 | ~flat (expected, encoder unrelated) |
| env | 0.041 | 0.045 | ~flat (expected) |
| **visual** | **2.183** | **0.097** | **−95.6%** |
| symbolic | 0.167 | 0.046 | −72% (downstream of visual grounding) |
| language | 0.0001 | 0.0001 | flat (expected) |

Improvement concentrated exactly where a visual-encoder swap should land — visual MSE drops ~22×, with a smaller knock-on in symbolic (binds across modalities including visual). Speech/env/language unaffected — the expected no-confound pattern.

**Memory retrieval** (`warm_recall_l2`, higher=better): DINOv2 0.2994 vs. DINOv3 0.4835, mean across 9 cycles (cycle 1 of session 1 has no prior warm memory) — **+61.5%**.

**Cross-modal gist entropy:** both ~2.3219, stable — binding/cross-modal integration unaffected by the encoder swap, as expected.

## Verdict

**DINOv3-ViT-S is a clear, large improvement over DINOv2-S/14** on both reconstruction quality and memory retrieval, effect correctly localized to the visual region. No crashes, shape errors, or lang/action regressions across all 10 sessions — the genuine "true drop-in" result the 2026-06-28 A/B was supposed to test, before the (since-fixed) CWD/external-data bug forced that run onto a DINOv2/DINOv2 comparison.

## Status

DINOv3 is already the active encoder — `VideoEncoder::new` prefers `dinov3_small.onnx` whenever present in `models/` (now is, single-file, embedded weights). No further action needed to "turn on" DINOv3 — this A/B validates what's already running. gaps.md item 6: validated/adopted, not just unblocked.

## Files

Control: `memory/dinov3_ab_control/metrics.jsonl`. Treatment: `memory/dinov3_ab_treatment/metrics.jsonl`. Script: `scripts/ab_dinov3.sh` (rewritten 2026-07-05 to use an isolated `--memory-dir` per arm and `--metrics-file` NDJSON output, matching the Qwen3 A/B's validated pattern — the previous version shared memory state between arms and only tail-parsed stdout).
