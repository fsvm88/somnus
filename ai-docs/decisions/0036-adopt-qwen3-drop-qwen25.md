# ADR 0036 — Adopt Qwen3-0.6B as the action-decoder LM; drop Qwen2.5

**Date:** 2026-07-05
**Status:** Implemented

## Context

ADR 0035 made the Qwen2.5-vs-Qwen3 hidden-size choice (896 vs 1024) an explicit compile-time
`qwen3` Cargo feature so both could be built and A/B'd safely. The A/B itself
(`ai-docs/a_b_qwen3_findings.md`, "redo take 2", speech-dense `clips/long/` corpus, 5 sessions ×
4 cycles/arm, identical exposure both arms) showed Qwen3 with a consistently ~22% lower
`mean_proj_loss` than Qwen2.5 across the whole campaign, no reversals at any sample, and a
directionally consistent `per_region_mse[language]` gap.

## Decision

Adopt Qwen3-0.6B as the only action-decoder LM. Drop Qwen2.5 entirely rather than keep it behind
a feature flag — this is a research repo with one developer and no backward-compatibility
requirement (`CLAUDE.md`), and the A/B question this flag existed to let us ask has been answered.
This also drops the previously-floated "Option D: Qwen2.5-0.5B-Instruct" backlog item, since it
was contingent on Qwen2.5 remaining the baseline.

Implementation:

- Removed the `qwen3` feature from `somnus/Cargo.toml` and `qwen3_hidden` from
  `somnus-perception/Cargo.toml`. There is now exactly one build.
- `action::LM_HIDDEN` (`crates/somnus/src/action.rs`) and `somnus_perception::LANG_HIDDEN`
  (`crates/somnus-perception/src/lib.rs`) are hardcoded `1024`, no `#[cfg]`.
- `action::EOS_TOKEN_ID` is hardcoded to Qwen3's `<|im_end|>` (151645).
- `main::LANG_DIM` is hardcoded `1024`.
- `--action-model-prefix` defaults to `"qwen3"` unconditionally (was previously cfg-gated to
  match the build). The flag itself stays parameterized — not tied to this A/B — so a future
  model swap still can't repeat the file cross-contamination ADR 0035 fixed.
- Fixed a latent bug found while removing the cfg gates: the `--language` startup file-existence
  check in `main.rs` had hardcoded `qwen25_embed_table.safetensors`/`qwen25_tokenizer.json`
  regardless of `--action-model-prefix`, unlike every other model-loading call site. Now derives
  the filenames from `args.action_model_prefix` like the rest.
- Doc comments referencing example filenames (`qwen25_lm.onnx`, `qwen25_text_hidden.onnx`, etc.)
  updated to `qwen3_*`.
- Left the two cumulative cross-modal offset tables in `somnus-perception/src/lib.rs`
  (`[(0,384),(384,512),(896,384),(1280,LANG_HIDDEN)]`) untouched — that `896` is `384+512`, not
  the old LM hidden dim; a coincidental collision, not a reference to Qwen2.5.

Not touched: `qwen25_*.onnx`/`.safetensors` files on disk (still re-exportable via
`scripts/export_models.py::export_qwen25()` if ever needed again) — deleting them is optional
cleanup, not required by this change.

## Consequences

- Single binary, single model family. No more `somnus-qwen25-control` / `somnus-qwen3-treatment` split —
  back to one `somnus` binary.
- Cross-modal projection input dimension changes: `1280 + LANG_HIDDEN` goes from `2176`
  (896) to `2304` (1024) as the *only* value now (previously toggled by feature flag).
  `ai-docs/architecture.md` and `ai-docs/architecture.svg` updated to match.
- Re-adopting Qwen2.5 (or any other hidden-size LM) later means re-adding a compile-time
  choice — ADR 0035's approach is the template if that happens.
- Confirmation via a direct `generate()` quality comparison was run the same day (a `git worktree`
  at the last commit with both Cargo features, one binary per arm, each against its take-2 trained
  memory dir): Qwen3 produced coherent, grammatical English; Qwen2.5 produced incoherent
  mixed-script token garbage, consistently. This confirms the take-2 `mean_proj_loss` gap reflects
  real generation quality, closing out the open confirmation item. Full writeup:
  `ai-docs/a_b_qwen3_findings.md` ("Take 3"). A longer multi-session confirmation campaign remains
  optional future work, not blocking.
