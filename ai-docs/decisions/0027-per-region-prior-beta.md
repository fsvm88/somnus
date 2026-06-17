---
status: in-progress
date: 2026-06-17
---

# ADR 0027 — Per-Region Prior Beta (Item G)

**Date:** 2026-06-17  
**Status:** Closed — FAIL (gist regression). Per-region static beta not enabled. Uniform β via S4 retained.

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

## Cold-start A/B result (2026-06-17) — FAIL

5 sessions × 2 cycles, fresh cold memory. Both arms built from the same binary.

| Metric | Control (s3–5 avg) | Treatment (s3–5 avg) | Δ | Threshold | Verdict |
|---|---|---|---|---|---|
| mean_output_mse_wake | 0.8014 | 0.5366 | −33.0% | ≤ctrl×1.10 | PASS |
| warm_recall_l2 | 0.5162 | 0.4456 | −13.7% | ±15% | PASS |
| gist_recall_rate | 0.8229 | 0.7617 | −0.061 | ±0.05 | **FAIL** |
| prior cosine — speech | 0.639 | 0.844 | +0.205 | ±0.05 | FAIL† |
| prior cosine — env | 0.807 | 0.869 | +0.062 | ±0.05 | FAIL† |
| prior cosine — visual | 0.488 | 0.543 | +0.054 | ±0.05 | FAIL† |
| prior cosine — symbolic | 0.736 | 0.745 | +0.009 | ±0.05 | PASS |
| prior cosine — language | 0.819 | 0.929 | +0.110 | ±0.05 | FAIL† |

† Prior cosine FAILs are all in the positive direction (treatment uniformly higher). Not
a quality regression; the higher cosines are partly self-reinforcing (blending prior into
input raises cos(input, prior) by construction). Not counted against the verdict.

**Gist regression is real and in the wrong direction.** Treatment gist 6.1 points lower
than control — the system consolidates warm episodes less effectively when per-region high
betas are active. Plausible mechanism: high-β prior injection biases the k-WTA inputs
away from the raw signal; the resulting sparse activations stored in warm memory are
slightly different from what the NREM replay loop trained on (which uses raw encoder
targets per ADR 0010). The mismatch accumulates over 5 sessions and degrades gist.

**Confound: S4 never fired in the control arm.** The control ran at β=0.01 for all 10
cycles; the intended comparison was "S4-scaled 0.05 uniform" vs "per-region 0.05/0.04/
0.03/0.05/0.01." The actual comparison was "0.01 uniform" vs "0.05 per-region" — a 5×
difference in effective beta for four of five regions. The MSE improvement in treatment
(−33%) reflects that high-β prior injection from session 1 accelerates convergence in
cold start; this is a cold-start effect, not a per-region distribution effect.

**The intended comparison was not tested.** The question "does distributing a 0.05 budget
unevenly across regions improve on the uniform 0.05 that S4 eventually delivers?" remains
unanswered. This is not a reason to run another A/B — uniform 0.05 (via S4) is the
current campaign behavior and not a proven bottleneck. A per-region distribution buys
meaningful improvement only if the budget allocation matters, and there is no evidence
of that from this run.

**Verdict:** FAIL by pre-stated rule (gist threshold breached, wrong direction). Uniform
β via S4 is retained as-is. Per-region beta plumbing ships in the binary as a
`--prior-beta-regions` flag for future experimentation but is not hardcoded in
`train_campaign.sh`.
