# ADR 0035 — LM hidden size (Qwen2.5 vs Qwen3) is a compile-time choice, not runtime

**Date:** 2026-07-05
**Status:** Superseded by [ADR 0036](0036-adopt-qwen3-drop-qwen25.md) — the A/B this feature flag existed to run concluded in Qwen3's favor, so the flag was removed for a single hardcoded build. Kept as the historical record of the infra work and A/B methodology.

## Context

A 2026-06-27 commit bumped `LM_HIDDEN` (`action.rs`) 896(Qwen2.5-0.5B)→1024(Qwen3-0.6B) as a single global constant, while `ActionDecoder::new` kept loading Qwen2.5's real 896-wide `qwen25_embed_table.safetensors` by hardcoded filename — `embed_flat()` sliced it at stride 1024, silently misaligning every token embedding. A parallel copy of the same bug lived in `somnus-perception` (`REGION_DIMS[4]`, `CROSS_MODAL_DIM`, ~6 more scattered 896/1024 literals) and `main.rs` (`LANG_DIM`).

Undetected for over a week because the 2026-06-28 Qwen3 A/B and 2026-07-05 DINOv3 A/B both ran with `--action-distill-only` (skips the one path, `generate()`, that calls the corrupted `lm_session`) and `--lang-use-lm` (prefers the separate unaffected `qwen25_text_hidden.onnx` for text embedding). Found 2026-07-05 while investigating unrelated `ort-smoke` output.

On top of that, the original A/B's Qwen3 arm never got matching GPU-path exports — it ran via a `cp qwen3_lm_base.onnx → qwen25_lm.onnx` file-swap trick whose restore step silently failed, leaving the wrong model on disk under the Qwen2.5 filename.

## Decision

`LM_HIDDEN` (896 vs 1024) is architecturally load-bearing throughout perception/action — it sizes the language k-WTA layer, SAE decoder, cross-modal projection input, and episode-projection output. A cold-started build's projection/output-head weights are shape-locked to whichever value compiled in — this makes it a **compile-time architecture choice**, not a runtime file-swap; the previous "hardcode one value globally, swap files by convention" approach is exactly what caused the corruption above.

**Implementation:** new `qwen3` Cargo feature on `somnus` (propagates `somnus-perception/qwen3_hidden`). `action::LM_HIDDEN`, `main::LANG_DIM`, new `somnus_perception::LANG_HIDDEN` (`REGION_DIMS`/`CROSS_MODAL_DIM` and every other language-region size derive from it) are `#[cfg(feature="qwen3")]`-gated: 896 without, 1024 with. `action::EOS_TOKEN_ID` also cfg-gated: Qwen2.5 base stops on `<|endoftext|>`(151643), Qwen3 base on `<|im_end|>`(151645), verified against each tokenizer's `eos_token_id`. `ActionDecoder::new`/`LangEmbedder::new` take a `model_prefix`/`--action-model-prefix` arg instead of hardcoded `qwen25_*` filenames — also removes a live footgun where `LangEmbedder` auto-detected "prefer qwen3 files if present," which would silently pick the wrong set once both permanently coexist on disk. Two release binaries from the same source: `somnus-qwen25-control` (default features) and `somnus-qwen3-treatment` (`--features qwen3`), each with its own `--action-model-prefix` default so a common invocation can't cross the streams. `export_qwen3()` rewritten to mirror `export_qwen25()` exactly, fixing a second bug found in passing: it had loaded Qwen3's checkpoint into `Qwen2ForCausalLM` (wrong architecture, missing QK-norm layers) instead of `Qwen3ForCausalLM`.

## Consequences

No more implicit coupling between an on-disk file and a compiled-in dimension constant — binary and loadable model files are fixed together at build time. Switching the default model family needs a rebuild, not a config flag — acceptable for a single-developer research repo, would need revisiting if this ever became a runtime/user-facing choice. A third LM with a different hidden size needs a third feature value (or generalizing past a boolean feature) — not attempted since only two models exist.

## A/B result (2026-07-05, `a_b_qwen3_findings.md`)

First attempt (standard 4-clip corpus, 10 sessions × 5 cycles/arm) was inconclusive — too little speech for `lang_ingest`/`decoder_trainable` to fire more than ~once/session. Second attempt on a speech-dense `clips/long/` corpus (`ab_qwen3_longclips.sh`, 5 sessions × 4 cycles/arm) gave identical exposure both arms (73 `lang_ingest`, ~40 `decoder_trainable` each) and showed Qwen3's `mean_proj_loss` consistently ~22% lower than Qwen2.5's by campaign end, no reversals. Recommendation: adopt Qwen3-0.6B, pending a longer confirmation or direct generation-quality test — not yet the default build at this point.
