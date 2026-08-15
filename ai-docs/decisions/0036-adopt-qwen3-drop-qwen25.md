# ADR 0036 — Adopt Qwen3-0.6B as the action-decoder LM; drop Qwen2.5

**Date:** 2026-07-05
**Status:** Implemented

## Context

ADR 0035 made the Qwen2.5-vs-Qwen3 hidden-size choice (896 vs 1024) an explicit compile-time `qwen3` feature so both could be built and A/B'd safely. The A/B (`a_b_qwen3_findings.md` "redo take 2," speech-dense `clips/long/` corpus, 5 sessions × 4 cycles/arm, identical exposure) showed Qwen3 consistently ~22% lower `mean_proj_loss` than Qwen2.5 across the whole campaign, no reversals, directionally consistent `per_region_mse[language]` gap.

## Decision

Adopt Qwen3-0.6B as the only action-decoder LM — drop Qwen2.5 entirely rather than keep it behind a feature flag (single-developer research repo, no backward-compat requirement per CLAUDE.md; the A/B question the flag existed for is answered). Also drops the previously-floated "Option D: Qwen2.5-0.5B-Instruct" backlog item, contingent on Qwen2.5 remaining baseline.

**Implementation:** removed the `qwen3` feature from `somnus/Cargo.toml` and `qwen3_hidden` from `somnus-perception/Cargo.toml` — one build. `action::LM_HIDDEN`/`somnus_perception::LANG_HIDDEN` hardcoded `1024`, no `#[cfg]`. `action::EOS_TOKEN_ID` hardcoded to Qwen3's `<|im_end|>`(151645). `main::LANG_DIM` hardcoded `1024`. `--action-model-prefix` defaults to `"qwen3"` unconditionally (was cfg-gated) — the flag itself stays parameterized, not tied to this A/B, so a future model swap still can't repeat ADR 0035's file cross-contamination. Fixed a latent bug found while removing the cfg gates: `main.rs`'s `--language` startup file-existence check had hardcoded `qwen25_embed_table.safetensors`/`qwen25_tokenizer.json` regardless of `--action-model-prefix`, unlike every other model-loading site — now derives filenames from `args.action_model_prefix` consistently. Doc-comment example filenames updated to `qwen3_*`. Left the two cumulative cross-modal offset tables in `somnus-perception/src/lib.rs` (`[(0,384),(384,512),(896,384),(1280,LANG_HIDDEN)]`) untouched — that `896` is `384+512`, a coincidental collision, not a Qwen2.5 reference.

Not touched: `qwen25_*.onnx`/`.safetensors` on disk (still re-exportable via `export_qwen25()` if ever needed) — deleting is optional cleanup, not required.

## Consequences

Single binary, single model family — no more `somnus-qwen25-control`/`somnus-qwen3-treatment` split. Cross-modal projection input dim `1280+LANG_HIDDEN` goes from 2176(896) to 2304(1024) as the *only* value (previously feature-toggled) — `architecture.md`/`architecture.svg` updated to match. Re-adopting Qwen2.5 or any other hidden-size LM later means re-adding a compile-time choice, ADR 0035's approach as the template. Confirmed via direct `generate()` comparison the same day (`git worktree` at the last dual-feature commit, one binary/arm against its take-2 trained memory dir): Qwen3 produced coherent grammatical English, Qwen2.5 consistently incoherent mixed-script token garbage — confirms the take-2 proj_loss gap reflects real generation quality, closing the open confirmation item. Full writeup: `a_b_qwen3_findings.md` ("Take 3"). A longer multi-session confirmation campaign remains optional future work, not blocking.
