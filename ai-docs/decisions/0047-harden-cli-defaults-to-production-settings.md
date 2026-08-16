# 0047. Harden four CLI defaults to match already-adopted production settings

- **Status:** accepted, shipped as default.
- **Date:** 2026-08-16

## Context

A CLI-default sweep (`constants.md`, 2026-08-16) compared every flag's bare `default_value` against what `scripts/train_campaign.sh` — the only harness that has run every real campaign, including the completed 10x growth run — unconditionally passes. `--prior-beta` turned out to be an outright bug: ADR 0024's S4 fix set the *campaign script's* env-var default to 0.01, but the Rust flag's own default was left at 0.0, so any direct (non-campaign-script) invocation got the disabled/broken behavior the fix was meant to eliminate. Fixed same-day.

The sweep found four more flags where the CLI default and the value every real campaign has actually run at have been silently different — some since as far back as 2026-05-31. Each is independently justified by existing evidence already in the docs; this ADR is the paper trail for hardening all four at once, since they're the same class of fix (sync the default to the value already adopted in practice) found by the same sweep.

## Decision

| Flag | Old default | New default | Evidence |
|---|---|---|---|
| `--working-memory-size` | 0 (disabled) | **8** | roadmap.md §10d: WM=8 vs WM=0 on the same snapshot, output_mse 40-50% lower with WM (ADR 0010/10d). Every campaign has run at WM=8 since `train_campaign.sh`'s working-memory support landed. |
| `--settle-steps` | 0 (disabled) | **1** | `somnus-perception::SETTLE_EPS`'s own calibration comment: "train_campaign.sh uses `--settle-steps 1` to match this reality" — `SETTLE_EPS=4.0` is calibrated against 1-iteration convergence, not 0. Campaign has run at steps=1 since `f44d5ee` (2026-05-31). |
| `--max-warm-records` | 0 (auto-derive from free RAM, ADR 0024 S5) | **200000** | Campaign has hardcoded 200,000 since `f891e21` (2026-06-17, three days *after* S5's auto-formula shipped) and the completed 10x campaign ended at 151,725 warm records — comfortably under the cap, no evidence it was ever a bad fit. `0` still opts back into the S5 auto formula. |
| `--lang-use-lm` | false | **true** | gaps.md item 10: this flag drives "the 4× p_lang improvement." Its own doc comment called it an "A/B flag... before hardening" — the A/B concluded in its favor and every campaign has run with it on. Falls back to the embed-table baseline automatically if `--action` (and its `qwen3_lm.onnx` load) isn't enabled. |

Also fixed in passing: `--max-warm-records`'s doc comment said "0 = unlimited" — stale since ADR 0024 S5 changed `0`'s meaning to "auto-derive from free RAM"; corrected to describe the actual current behavior.

## Alternatives considered

1. **Leave the defaults as-is, keep flagging the discrepancy in `constants.md`.** Rejected — per CLAUDE.md, "when the ecosystem has a gap... don't hand-roll," and more generally this project prefers small reversible fixes over an open paper trail; there's no scenario where a bare `somnus` invocation *should* silently diverge from the one configuration that's actually been validated at scale. `constants.md`'s sweep table already named the confidence level for each; none of the four had a stated reason to stay at the old default.
2. **Only fix `--working-memory-size` and `--settle-steps`** (the two with direct A/B/pilot evidence), leave `--max-warm-records` and `--lang-use-lm` flagged (weaker evidence: no fresh A/B, just "campaign always does this and nothing broke"). Rejected on user instruction — asked explicitly for all four to move together, with the reasoning captured here rather than deferred.
3. **Change `train_campaign.sh` to stop passing these flags explicitly, relying on the new defaults.** Not done — the script's explicit `--working-memory-size 8` / `--settle-steps 1` / `--recall-ef 16`-style passes are harmless now that they match the default, and leaving them explicit means the campaign's actual configuration stays self-documenting in its own invocation rather than depending on `somnus`'s current default (which could change again). No functional difference either way; left alone to minimize this change's diff.

## Consequences

- Any direct `somnus` invocation without these four flags now behaves like a real campaign run instead of the old research-safe/disabled baseline. Interactive/one-off/debug invocations that relied on the old defaults (e.g. expecting `--lang-use-lm` off, or `--max-warm-records` auto-sizing to available RAM) will now get the campaign-tuned behavior unless they pass the flag explicitly.
- `--settle-steps=1`'s cold-start WARN (ADR 0011: "settling active but no prior snapshot — output heads are random") now fires on every from-scratch `somnus` run with no snapshot, not just campaign runs — this is pre-existing, harmless (metrics-only) behavior, just newly visible outside the campaign script.
- `--max-warm-records`'s ADR 0024 S5 auto-formula (`free_ram×0.35/2500`, clamped `[1_000, 1_000_000]`) is no longer exercised by default — still available via `--max-warm-records 0`.

## What would invalidate this

- `--max-warm-records=200000` becomes a binding constraint at a scale beyond the completed 10x campaign (e.g. a future 100x campaign) — re-derive from the S5 auto formula or raise the hardcoded value.
- `--lang-use-lm`'s "4× p_lang improvement" claim doesn't hold under a future LM/encoder swap (gaps.md item 10 already flags that a language-encoder migration must move together with the LM) — re-A/B before assuming this default still applies.
- `--settle-steps` is ever raised above 1 in practice — `SETTLE_EPS` needs recalibrating first per its own doc comment (ADR 0011).

## Metrics

No new metric needed — these are default-value changes to already-instrumented, already-Core-tier paths (`working_memory`, settle loop, warm-store cap, language embedding source all already emit their existing NDJSON fields regardless of these flags' values). Nothing about changing a default requires new instrumentation to evaluate or monitor going forward.

## Source

`constants.md`'s "CLI-default sweep (2026-08-16)" section; `crates/somnus/src/lib.rs` (`working_memory_size`, `settle_steps`, `max_warm_records`, `lang_use_lm` doc comments). `cargo test -p somnus` (101 tests), `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check` all clean after the change.
