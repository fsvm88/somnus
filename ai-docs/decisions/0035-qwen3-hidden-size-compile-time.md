# ADR 0035 — LM hidden size (Qwen2.5 vs Qwen3) is a compile-time choice, not runtime

**Date:** 2026-07-05
**Status:** Superseded by [ADR 0036](0036-adopt-qwen3-drop-qwen25.md) — the A/B this feature flag
existed to run concluded in Qwen3's favor, so the flag was removed in favor of a single hardcoded
build. This document remains the historical record of the infra work and the A/B methodology.

## Context

A 2026-06-27 commit ("prepare for Qwen3-0.6B") bumped `LM_HIDDEN` in `crates/somnus/src/action.rs`
from 896 (Qwen2.5-0.5B) to 1024 (Qwen3-0.6B) as a single global constant, while
`ActionDecoder::new` kept loading `qwen25_embed_table.safetensors` — Qwen2.5's real 896-wide
table — by hardcoded filename. `embed_flat()` sliced that table at stride 1024, silently
misaligning every token embedding fed into the action decoder. A parallel copy of the same bug
existed in `crates/somnus-perception` (`REGION_DIMS[4]`, `CROSS_MODAL_DIM`, and ~6 more scattered
`896`/`1024` literals for the language region) and in `crates/somnus/src/main.rs` (`LANG_DIM`).

This wasn't caught for over a week because the 2026-06-28 Qwen3 A/B and the 2026-07-05 DINOv3 A/B
both ran with `--action-distill-only`, which skips the one code path (`generate()`) that calls the
corrupted `lm_session` directly, and `--lang-use-lm` prefers the separate, unaffected
`qwen25_text_hidden.onnx` for text embedding. Found 2026-07-05 while investigating unrelated
`ort-smoke` output. Full incident: `ai-docs/a_b_qwen3_findings.md`, `.claude/session-state.md`.

On top of the corruption bug, the original A/B's Qwen3 arm never got matching GPU-path exports
(`qwen3_gain.onnx`/`qwen3_text_hidden.onnx`/`qwen3_lm_fixed.onnx`) and instead ran via a
`cp qwen3_lm_base.onnx → qwen25_lm.onnx` file-swap trick in `ab_qwen3.sh`, whose restore step
silently failed and left the wrong model on disk under the Qwen2.5 filename.

## Decision

`LM_HIDDEN` (896 vs 1024) is architecturally load-bearing throughout the perception/action
pipeline — it sizes the language k-WTA layer, the SAE decoder, the cross-modal projection input,
and the episode-projection output. A cold-started build's projection/output-head weights are
shape-locked to whichever value compiled in. This makes it a **compile-time architecture choice**,
not something to hot-swap at runtime by pointing at a different file — the previous "hardcode one
value globally, swap files by convention" approach is exactly what caused the corruption above.

Implementation:

- Added a `qwen3` Cargo feature to the `somnus` crate (propagates `somnus-perception/qwen3_hidden`).
- `action::LM_HIDDEN`, `main::LANG_DIM`, `somnus_perception::LANG_HIDDEN` (new constant —
  `REGION_DIMS`/`CROSS_MODAL_DIM` and every other language-region size now derive from it) are
  `#[cfg(feature = "qwen3")]`-gated: 896 without the feature, 1024 with it.
- `action::EOS_TOKEN_ID` is also cfg-gated: Qwen2.5 base stops on `<|endoftext|>` (151643), Qwen3
  base on `<|im_end|>` (151645) — verified against each tokenizer's `eos_token_id`.
- `ActionDecoder::new` and `LangEmbedder::new` take a `model_prefix`/`--action-model-prefix`
  argument instead of hardcoded `qwen25_*` filenames, so control/treatment runs load
  `{prefix}_gain.onnx`, `{prefix}_lm.onnx`, `{prefix}_embed_table.safetensors`, etc. This also
  removed a live footgun in `LangEmbedder`: it previously **auto-detected** "prefer qwen3 files if
  present," which — once both file sets permanently coexist on disk — would silently pick the
  wrong file set for whichever binary was actually running.
- Two release binaries built from the same source: `somnus-qwen25-control` (default features) and
  `somnus-qwen3-treatment` (`--features qwen3`), each paired with its own `--action-model-prefix`
  default so the common invocation can't cross the streams.
- `export_qwen3()` in `scripts/export_models.py` rewritten to mirror `export_qwen25()` exactly
  (`qwen3_gain.onnx`, `qwen3_text_hidden.onnx`, `qwen3_lm.onnx`, `qwen3_lm_fixed.onnx`), fixing a
  second bug found in passing: it had loaded Qwen3's checkpoint into `Qwen2ForCausalLM` (wrong
  architecture — Qwen3 has QK-norm layers Qwen2 doesn't) instead of `Qwen3ForCausalLM`.

## Consequences

- No more implicit coupling between a file on disk and a compiled-in dimension constant — the
  binary and the model files it can load are fixed together at build time.
- Switching the default model family requires a rebuild (`cargo build --release --features
  qwen3`), not a config flag. Acceptable for a single-developer research repo; would need
  revisiting if this ever needed to be a runtime/user-facing choice.
- Adding a third LM with a different hidden size means adding a third feature value (or
  generalizing past a boolean feature) — not attempted here since only two models exist.

## A/B result (2026-07-05, `ai-docs/a_b_qwen3_findings.md`)

First attempt (standard 4-clip corpus, 10 sessions × 5 cycles/arm) was inconclusive — too little
speech content for `lang_ingest`/`decoder_trainable` events to fire more than ~once per session.
Second attempt on a speech-dense `clips/long/` lecture/podcast corpus (`scripts/
ab_qwen3_longclips.sh`, 5 sessions × 4 cycles/arm) gave both arms identical exposure (73
`lang_ingest` events, ~40 `decoder_trainable` events each) and showed Qwen3's `mean_proj_loss`
consistently ~22% lower than Qwen2.5's by the end of the campaign, with no reversals across any
sample. Recommendation: adopt Qwen3-0.6B for the action-decoder LM, pending a longer confirmation
run or a direct generation-quality test — not yet the default build.
