---
status: in-progress
date: 2026-06-17
---

# ADR 0027 — Per-Region Prior Beta (Item G)

**Date:** 2026-06-17  
**Status:** In progress — plumbing complete; static A/B pending

## Context

ADR 0010 deferred a per-region scalar gate with trigger: "prior_cosine stable per region
AND per-region β tuning shows clear benefit." Per-region cosines at steady state (s43–50
mixed_campaign):

| Region | prior_cosine mean |
|---|---|
| speech | 0.973 |
| symbolic | 0.941 |
| env | 0.909 |
| visual | 0.885 |
| language | 0.843 |

The S4 controller (ADR 0024) already adapts β globally: scales from `--prior-beta` (0.01)
to 0.05 when global cosine > 0.65 for 3 consecutive cycles. At steady state, global cosine
≈ 0.91, so S4 fires and the effective β is 0.05 for all five regions.

The hypothesis under test: the 0.05 ceiling is appropriate for strong-prior regions (speech,
symbolic) where prior predictions are reliable, but applies too much influence in language
(0.843) where the prior is a weaker predictor. Blending a poor prior into language input
at β=0.05 pushes it toward a less accurate prediction.

**Direction:** boost strong-prior regions, leave weak-prior regions at base. The expected
mechanism is that language encoding (and downstream episode binding) improves when the
language prior doesn't contaminate the raw embedding.

**Scope:** This is a static per-region heuristic, not a gradient-learned gate. "Learned"
in item G means session-to-session cosine-adaptive adjustment of per-region β. This ADR
tests whether static per-region β is beneficial before adding that dynamics layer.

## Decision (pre-stated, before running A/B)

### Plumbing

Replace `prior_beta: f32` with `prior_betas: [f32; 5]` in `PerceptionProcessor`. Add
`--prior-beta-regions "s,e,v,y,l"` CLI arg (comma-separated, 5 floats). When set, S4's
global controller is bypassed (static mode). `set_prior_beta(β)` remains as set-all
convenience for S4 backward compatibility.

### A/B design

Fresh cold memory, 5 sessions × 2 cycles.

- **Control:** `--prior-beta 0.01` (S4 adaptive; reaches 0.05 globally by ≈ s2)
- **Treatment:** `--prior-beta-regions "0.05,0.04,0.03,0.05,0.01"` (static; no S4)
  - speech: 0.05, env: 0.04, visual: 0.03, symbolic: 0.05, language: 0.01
  - Rationale: cosine ≥ 0.94 → 0.05; 0.90–0.93 → 0.04; 0.88–0.90 → 0.03; < 0.88 → 0.01

### Pre-stated win threshold (s3–5 avg)

Treatment wins if ALL hold:

1. `output_mse` ≤ control × 1.10 (primary; must not regress by > 10%)
2. `warm_recall_l2` within **±15%** of control
3. `gist_recall_rate` within **±0.05** of control

Evaluating on s3–5 because s1–2 in the treatment arm start with static high β before the
encoder heads have converged; some transient noise is expected.

## What happens on win

Enable per-region beta in `train_campaign.sh`. Upgrade S4 to per-region rings (each region
independently tracked). File the adaptive dynamics as a follow-on deferred item.

## What happens on loss

Accept that uniform β (via S4) is adequate. Item G closed. The ADR 0010 deferred item for
a gradient-learned gate remains open only if a concrete motivation surfaces.
