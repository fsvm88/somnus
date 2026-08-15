# ADR 0046 — Audio Env Encoder Candidates: CLAP-HTSAT Retained

**Date:** 2026-07-09
**Status:** Closed — status quo confirmed. CLAP-HTSAT remains the default audio-environmental encoder. Both surveyed replacements were not adopted.

## Context

CLAP-HTSAT is the pre-existing default encoder for the audio-environmental region (`REGION_DIMS[1]`, 512-dim). Two candidate replacements were surveyed for possible adoption: M2D-CLAP (higher paper-benchmark quality, feasibility question only) and EfficientAT-mn10 (much smaller footprint, taken all the way to an A/B). This ADR records why neither displaced CLAP-HTSAT as default.

## Decision

Retain CLAP-HTSAT as the default audio-environmental encoder. Neither candidate is adopted as default.

## Candidate 1 — M2D-CLAP: not pursued (feasibility only)

**Source:** NTT, March 2025, arXiv:2503.22104, `nttcslab/m2d`. ~86M params (ViT-B), AudioSet linear-probe mAP 49.0 vs CLAP-HTSAT ~45-46 (+3 mAP), 768-dim (up from 512).

No implementation was attempted. Feasibility research surfaced two blockers before any export work started:

- **No ONNX export precedent anywhere** — dynamic positional-embedding interpolation plus an `nnAudio` STFT frontend, the same fragility class the mn10 candidate (below) was specifically chosen to avoid.
- **License** — NTT's non-commercial custom "Evaluation" agreement, not MIT/Apache.

The expected gain (+3 mAP on a paper benchmark) was judged small relative to that risk, and candidate 2's A/B result (below) is concrete precedent that paper-benchmark quality and somnus-pipeline reconstruction quality can diverge in the wrong direction. Not implemented.

**Revisit if:** an ONNX export appears elsewhere, or CLAP-HTSAT becomes a measured bottleneck worth the licensing/export risk.

## Candidate 2 — EfficientAT-mn10: implemented, A/B run, rejected as default

**Source:** `fschmid56/EfficientAT`. ~5-20M params (MobileNetV3, width_mult=1.0), pure CNN, no attention/ScatterElements.

**Implementation (kept, opt-in):** `Mn10Mel` (`crates/somnus-encoding/src/preprocess/mel.rs`) — a Rust port of `AugmentMelSTFT`'s Kaldi-style filterbank, parity-tested at 1e-4. `export_efficientat.py` exports only the MobileNetV3 backbone (mel frontend excluded, to avoid reintroducing an STFT op) — verified rel L2 ~9e-7, zero STFT ops. Gated behind the opt-in `mn10_env` Cargo feature (`crates/somnus-encoding/Cargo.toml`, `crates/somnus-perception/Cargo.toml`, forwarded from `crates/somnus/Cargo.toml`): `REGION_DIMS[1]` 512→960, separate env-encoder session/mel/rate (CLAP 48kHz vs mn10 32kHz).

**A/B (5 sessions × 2 cycles, fresh cold memory, h3h3/shorts corpus).** Region-1 MSE isn't a valid cross-arm comparison — CLAP targets are contrastive/unit-norm, mn10 targets are raw log-mel, different distributions. Verdict instead comes from the other four regions (identical architecture in both arms):

| Metric | Control (CLAP-HTSAT) | Treatment (mn10) | Δ |
|---|---|---|---|
| avg MSE, 4 comparable regions | 0.011966 | 0.030588 | **+155.6% worse** |
| cycles regressed | — | — | 9/10, no reversals |
| warm_recall_l2 | — | — | +52.6% (favors mn10) |
| gist_recall_rate | — | — | +30.7% (favors mn10) |

The 2.5× MSE regression, consistent across 9 of 10 cycles with no reversals, is the controlling result. The two metrics that moved in mn10's favor don't override it — a 2.5× reconstruction-quality regression this consistent is treated as a real effect, and the recall-metric improvement as a likely scale-distortion artifact of mn10's raw log-mel target range rather than genuine encoding quality.

**Decision: reject as default.** CLAP-HTSAT stays default. `mn10_env` code is **kept**, not deleted — clean, parity-tested, opt-in — because it may suit deployment-size-constrained use later, or serve as a fallback if M2D-CLAP (or another candidate) is revisited and also fails. "Rejected as default" and "code removed" are different outcomes here: only the former applies.

## Cross-candidate methodological point

mn10's AudioSet paper benchmark carries no explicit quality claim over CLAP-HTSAT (it's a size/efficiency play), yet even there somnus-pipeline MSE regressed sharply while a benchmark-adjacent recall metric moved the "wrong" way (favorably) — paper/benchmark-style metrics and somnus-pipeline reconstruction quality diverged. M2D-CLAP's headline advantage over CLAP-HTSAT (+3 AudioSet mAP) is the same *kind* of paper-benchmark signal. Given the mn10 result, that signal was treated as weak evidence for actual somnus-pipeline gain — reinforcing the decision to not spend export/licensing effort chasing it without a working A/B to confirm it transfers.

## What would invalidate this

- An ONNX export of M2D-CLAP appears elsewhere (removing the do-it-yourself export risk) — worth a real A/B at that point, independent of the mAP number.
- CLAP-HTSAT becomes a measured bottleneck (latency, size) that these candidates would address, materially changing the risk/reward calculus for M2D-CLAP's licensing and export risk.
- A future mn10-family or other candidate A/B shows reconstruction-MSE parity or improvement, not just favorable recall-metric movement — the recall-only signal from this A/B is explicitly not sufficient on its own.

## Metrics

Both candidate evaluations are pre-`somnus-metrics`-two-tier-instrumentation decisions about which *encoder* to run, not measured production code paths in themselves. M2D-CLAP: no code was built, so no metrics apply. mn10: the A/B used existing region MSE, `warm_recall_l2`, and `gist_recall_rate` metrics (pre-existing evaluation-run metrics, not new Core/Fine additions) to compare arms; no new data-handling operation was introduced by this ADR's decision (reject-as-default, keep-as-opt-in) that would require new Core or Fine instrumentation. If `mn10_env` is ever promoted to default, that promotion is the point requiring its own Core-tier accounting (encoder-selection counter, mel-frontend timing) per the standing metrics policy.
