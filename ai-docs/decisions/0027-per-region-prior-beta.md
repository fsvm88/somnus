---
status: in-progress
date: 2026-06-17
---

# ADR 0027 — Per-Region Prior Beta (Item G)

**Date:** 2026-06-17
**Status:** Closed — FAIL (gist regression). Per-region static beta not enabled. Uniform β via S4 retained.

## Context

ADR 0010 deferred a per-region scalar gate, triggered when "prior_cosine stable per region AND per-region β tuning shows clear benefit." Per-region cosines at steady state (s43-50 mixed_campaign): speech 0.973, symbolic 0.941, env 0.909, visual 0.885, language 0.843.

S4 (ADR 0024) already adapts β globally: `--prior-beta`(0.01)→0.05 when global cosine>0.65 for 3 consecutive cycles. At steady state global cosine≈0.91, so S4 fires and effective β=0.05 for all five regions.

**Hypothesis:** the 0.05 ceiling suits strong-prior regions (speech, symbolic) but over-influences language (0.843), where the prior is a weaker predictor — blending a poor prior into language input pushes it toward a less accurate prediction. **Direction:** boost strong-prior regions, leave weak-prior regions at base. **Scope:** static per-region heuristic, not a gradient-learned gate — "learned" (item G) would mean session-to-session cosine-adaptive per-region β; this ADR tests whether static per-region β helps before adding that layer.

## Decision (pre-stated, before running A/B)

**Plumbing:** `prior_beta: f32`→`prior_betas: [f32;5]` in `PerceptionProcessor`; new `--prior-beta-regions "s,e,v,y,l"` CLI arg bypasses S4's global controller when set (static mode); `set_prior_beta(β)` kept as a set-all convenience for S4 compat.

**A/B:** fresh cold memory, 5 sessions × 2 cycles. Control: `--prior-beta 0.01` (S4 adaptive, reaches 0.05 globally by ≈s2). Treatment: `--prior-beta-regions "0.05,0.04,0.03,0.05,0.01"` (static, no S4) — speech 0.05, env 0.04, visual 0.03, symbolic 0.05, language 0.01, per cosine≥0.94→0.05 / 0.90-0.93→0.04 / 0.88-0.90→0.03 / <0.88→0.01.

**Pre-stated win** (s3-5 avg, all must hold): `output_mse` ≤ control×1.10; `warm_recall_l2` within ±15%; `gist_recall_rate` within ±0.05. Evaluated on s3-5 since s1-2 treatment starts with static high β before encoder heads converge, expected transient noise.

**On win:** enable per-region beta in `train_campaign.sh`, upgrade S4 to per-region rings, file the adaptive-dynamics layer as a follow-on. **On loss:** accept uniform β via S4 is adequate, close item G; ADR 0010's gradient-learned-gate deferred item stays open only if a concrete motivation surfaces.

## Cold-start A/B result (2026-06-17) — FAIL

| Metric | Control (s3-5 avg) | Treatment (s3-5 avg) | Δ | Threshold | Verdict |
|---|---|---|---|---|---|
| mean_output_mse_wake | 0.8014 | 0.5366 | −33.0% | ≤ctrl×1.10 | PASS |
| warm_recall_l2 | 0.5162 | 0.4456 | −13.7% | ±15% | PASS |
| gist_recall_rate | 0.8229 | 0.7617 | −0.061 | ±0.05 | **FAIL** |
| prior cosine — speech | 0.639 | 0.844 | +0.205 | ±0.05 | FAIL† |
| prior cosine — env | 0.807 | 0.869 | +0.062 | ±0.05 | FAIL† |
| prior cosine — visual | 0.488 | 0.543 | +0.054 | ±0.05 | FAIL† |
| prior cosine — symbolic | 0.736 | 0.745 | +0.009 | ±0.05 | PASS |
| prior cosine — language | 0.819 | 0.929 | +0.110 | ±0.05 | FAIL† |

† All positive-direction — treatment uniformly higher, partly self-reinforcing (blending prior into input raises cos(input,prior) by construction), not counted against the verdict.

**Gist regression is real and wrong-direction:** treatment 6.1 points lower — the system consolidates warm episodes less effectively with per-region high betas active. Plausible mechanism: high-β prior injection biases k-WTA inputs away from raw signal, so the sparse activations stored in warm differ from what NREM replay trained on (raw encoder targets, ADR 0010) — the mismatch accumulates over 5 sessions.

**Confound: S4 never fired in control** — it ran at β=0.01 for all 10 cycles. The intended comparison ("S4-scaled 0.05 uniform" vs. "per-region 0.05/0.04/0.03/0.05/0.01") became "0.01 uniform" vs. "0.05 per-region," a 5× difference for four of five regions. Treatment's −33% MSE improvement reflects high-β prior injection accelerating cold-start convergence, not a per-region distribution effect.

**The intended comparison was never tested.** "Does distributing a 0.05 budget unevenly improve on the uniform 0.05 S4 eventually delivers?" remains unanswered — not a reason to re-run, since uniform 0.05 via S4 is current campaign behavior and not a proven bottleneck; there's no evidence the allocation matters.

**Verdict: FAIL** by pre-stated rule (gist threshold breached, wrong direction). Uniform β via S4 retained. Per-region beta plumbing ships as `--prior-beta-regions` for future experimentation but isn't hardcoded in `train_campaign.sh`.
