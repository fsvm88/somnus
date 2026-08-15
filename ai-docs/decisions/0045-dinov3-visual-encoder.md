# ADR 0045 — Adopt DINOv3-ViT-S as the visual encoder; drop DINOv2-S/14

**Date:** 2026-07-05
**Status:** Implemented

## Context

DINOv3 (Meta, Aug 2025, arXiv:2508.10104) is a candidate drop-in replacement for the DINOv2-S/14 visual encoder: same parameter count (~21M), same 384-dim output, and a higher ImageNet linear-probe score (81.40% vs DINOv2-S's 81.1%). A same-dim swap is the cheap case — no cross-modal projection resizing — so it only needed correctness + quality validation before adoption.

A first attempt (2026-06-28) concluded DINOv3 was broken on the MIGraphX (CPU) backend — "can't compile DINOv3's attention graph" — and the A/B that followed was forced onto a DINOv2/DINOv2 comparison instead. That conclusion was wrong.

**The MIGraphX bug, and the lesson in it.** `torch.onnx.export` wrote DINOv3 in external-data format: the weights live in a sibling `.data` file, referenced from the `.onnx` by a relative path. The MIGraphX execution provider resolves that relative path against the **process's current working directory**, not the model file's directory — so any invocation outside `models/` failed to locate the weights and surfaced as an opaque compile/load error. That error was misread as an attention-graph incompatibility, a plausible-sounding but false diagnosis that stalled adoption for a week. The actual fix (`export_models.py`'s `export_dinov3()`) re-packs the export with embedded weights, matching DINOv2's single-file layout — no CWD dependency, no external-data path to resolve at all. Lesson: an opaque backend load/compile failure is not evidence of a capability gap until the error is root-caused; "backend can't do X" is a conclusion of last resort, not first resort.

**Shape difference (not a bug).** DINOv3-ViT-S/16 outputs `[1, 201, 384]` (1 CLS token + 4 register tokens + 196 patch tokens), vs. DINOv2-S/14's `[1, 257, 384]` (1 CLS + 256 patches) — different patch size and DINOv3's register-token architecture, not a defect. `VideoEncoder` previously hardcoded "skip 1 token, pool 256" for DINOv2; it now tracks `prefix_tokens`/`total_tokens` per model instance so both shapes pool correctly.

## Decision

Adopt DINOv3-ViT-S as the only visual encoder, replacing DINOv2-S/14 (same pattern as ADR 0036: single-developer research repo, no backward-compat requirement, A/B question answered — no feature flag kept around).

**Implementation:**
- `export_models.py::export_dinov3()` exports with embedded weights (fixes the CWD/external-data bug above).
- `crates/somnus-encoding/src/video.rs`: `VideoEncoder::new` prefers `models/dinov3_small.onnx` when present, else falls back to DINOv2; added `prefix_tokens`/`total_tokens` fields per instance in place of hardcoded pooling constants, so both encoders' token layouts are handled correctly.
- DINOv3 is already the active encoder in `models/` (single-file, embedded weights) — no further "turn on" step.

## Alternatives considered

- **Keep DINOv2 as default, DINOv3 behind a flag** (ADR 0035/0036's compile-time-feature pattern): rejected — same-dim drop-in with a clean, large, correctly-localized A/B win; no reason to carry two encoders once the comparison is settled.
- **Debug MIGraphX further before re-attempting the A/B**: this *was* the path taken, and it's the reason the first A/B attempt (2026-06-28) was invalid — recorded here as the "what actually happened" rather than an alternative not taken.

## A/B validation

**Setup:** `scripts/ab_dinov3.sh`, 2 sleep cycles/session × 5 sessions per arm, isolated `--memory-dir` per arm, NDJSON `--metrics-file` output (fixed from an earlier version of the script that shared memory state between arms and only tail-parsed stdout). Full write-up: `ai-docs/a_b_dinov3_findings.md`.

**Results** (mean across cycles; DINOv2-S control vs. DINOv3-ViT-S treatment):

| Metric | DINOv2 | DINOv3 | Delta |
|---|---|---|---|
| `mean_output_mse_wake` (overall) | 0.3573 | 0.0971 | −73% |
| `per_region_mse[visual]` | 2.183 | 0.097 | **−95.6%** |
| `per_region_mse[symbolic]` | 0.167 | 0.046 | −72% (downstream, binds cross-modally with visual) |
| `warm_recall_l2` | 0.2994 | 0.4835 | **+61.5%** |
| `per_region_mse[speech/env/language]` | — | — | flat (expected — unrelated regions) |
| cross-modal gist entropy | ~2.3219 | ~2.3219 | flat (expected) |

Effect is large, concentrated exactly where a visual-encoder swap should land, with no regressions in unrelated regions and no confounds in binding/cross-modal metrics. No crashes or shape errors across 10 sessions.

## Consequences

- Visual-region reconstruction quality and memory-retrieval quality both improve substantially with no code cost beyond the pooling generalization above (same 384-dim output means no cross-modal projection changes were needed).
- `VideoEncoder`'s token-layout handling is now model-generic (`prefix_tokens`/`total_tokens`) rather than DINOv2-hardcoded, which is a prerequisite for any future ViT-family swap, not just this one.
- Old DINOv2 export path (`export_models.py`) is left in place, not required to be deleted.
- The MIGraphX external-data/CWD gotcha applies to any future ONNX export using external-data format on this backend — worth checking first if a new model "won't load" opaquely.

## What would invalidate this

- A regression in visual-region MSE or `warm_recall_l2` on a longer multi-session campaign that the 5-session A/B didn't have power to catch.
- A future MIGraphX/ONNX-runtime upgrade that changes external-data path resolution — would remove the reason the embedded-weights export was needed, but wouldn't itself argue against DINOv3.
- A visual encoder with materially better linear-probe/retrieval numbers at the same or lower cost (same review bar as this ADR: same-dim drop-in gets a lightweight A/B; different-dim gets the ADR-0035-style compile-time-choice treatment).

## Metrics

No new metrics were required for this change: `mean_output_mse_wake`, `per_region_mse[*]`, `warm_recall_l2`, and cross-modal gist entropy are pre-existing Core/Fine-tier metrics (`somnus-metrics`, ADR 0039) that already cover encoder-swap evaluation — the A/B used them as-is via `--metrics-file` NDJSON output, no instrumentation gap to fill.

## Source

`engineering-log.md` §6 (primary record); full A/B detail in `a_b_dinov3_findings.md`.
