# 0039. Unified two-tier metrics instrumentation (`somnus-metrics`)

- **Status:** implemented.
- **Date:** 2026-07-14

## Context

While scoping ADR 0038's benchmarking prerequisite (measure the k-WTA scoring matmul's real
cost), it became clear the codebase had no shared instrumentation mechanism at all: every timed
call site hand-rolled its own `let t = Instant::now(); ...; t.elapsed()` pair, duplicated dozens
of times across `crates/somnus-perception/src/lib.rs` (6 ingest phases), `crates/somnus/src/sleep.rs`
(11 wall-clock phases), `crates/somnus-encoding/src/{audio,video}.rs` (3 encoder timers), and
`crates/somnus/src/main.rs` (4 more sites, two of them *duplicate* measurements of a call already
timed inside `somnus-encoding`). No Criterion setup, no timing macro, no shared registry — each site
independently computed a duration and fed it into its own local accumulator or `tracing::info!`
field, then a single ~90-field `serde_json::json!{}` literal in `main.rs` assembled everything
into the campaign's NDJSON metrics file by hand.

The user wanted two things layered on top of this: (1) much finer-grained counters — "every
operation which handles data gets a counter" — gated behind a Cargo feature so a normal build
stays exactly as lean as today, and (2) once that existed, a single pass converting every
*existing* hand-rolled timer to the same mechanism, so there is one way of doing metrics instead
of two (the old ad hoc pattern plus a new opt-in one bolted alongside it).

## Decision

New crate `crates/somnus-metrics`, with metrics recorded in two tiers sharing one registry:

- **Core** (`timed_core!`/`counted_core!`): unconditionally compiled and always recording — this
  *is* the "big item" tier, i.e. every metric this codebase has always reported (ingest phases,
  sleep wall-clock, encoder timers, main-loop ingest/query timers), just funneled through one
  macro instead of copy-pasted boilerplate at each site.
- **Fine** (`timed!`/`counted!`): gated behind the `fine-metrics` Cargo feature (off by default).
  With the feature disabled, these macros expand to a bare passthrough of their body — the
  `Instant`/atomic-counter code is textually absent, not a runtime-skipped branch. Reserved for
  granular sub-breakdowns not worth paying for on every build (currently: `KwtaScoreMatmul`/
  `KwtaSelectTopK`, the sub-phases inside the k-WTA ingest step that ADR 0038 needs).

Both tiers write into a single fixed-size atomic `Registry` (one `AtomicU64` count/total-nanosecond
pair per `Metric` enum variant — mirrors `RegionId::as_index()`'s existing dense-array-over-enum
pattern rather than a `HashMap`), drained once per sleep cycle via `Registry::snapshot_and_reset()`
into a new `"fine_metrics"` key in the existing metrics NDJSON line. All pre-existing top-level
JSON fields (`perc_kwta_us`, `nrem_wall_ms`, `speech_encode_total_ms`, etc.) are populated exactly
as before — from the same `Duration` values, now returned by the macro instead of a separately
maintained `Instant` — so `campaign_status.py`/`eval_metrics.py` need no changes.

**Full conversion, not just new call sites.** Every existing hand-rolled timer across
`somnus-perception`, `somnus-encoding`, and `somnus`'s event loop / sleep cycle now goes through
`timed_core!` (or a direct `somnus_metrics::METRICS.record()` call where a span is too structurally
entangled — multiple mutable locals escaping scope, nested loops with early `break`s — to cleanly
wrap as a single macro-body expression; see "Alternatives" below). Two genuinely duplicate
timers were found and removed in the process: `main.rs` was independently re-timing
`encode_speech`/`encode_env`/video-frame `encode()` calls that `somnus-encoding` already timed and
returned via `encode_ms` — main.rs now logs that field directly instead of measuring the same call
twice.

## Alternatives considered

1. **Wrap every span in `timed_core!`'s block-expression idiom uniformly.** Works cleanly for
   self-contained bodies (a single `match`/`if`/function call with no bindings needed outside the
   macro). Rejected for a handful of structurally awkward spans — sleep.rs's full NREM loop
   (`SleepNrem`), its two non-contiguous REM sub-sections (`SleepRem`), and the whole-cycle
   wall-clock (`SleepTotal`) — because each has locals computed inside the "old span" that are
   consumed far later in the function (e.g. `mean_train_loss` feeds the final `SleepStats`
   construction hundreds of lines later). Forcing these into a macro body would either leak scope
   incorrectly or require a large, risky restructuring for no behavioral benefit. For these,
   `somnus_metrics::METRICS.record(Metric::X, duration)` is called directly at the same point the
   duration was already being computed — same registry, same `Metric` enum, same output, just
   without the macro's return-value sugar. This is a deliberate, documented exception, not an
   inconsistency: the goal is one *registry*, not that every call site must look identical.
2. **`tracing` spans instead of a bespoke registry.** `tracing` is already a workspace dependency
   but used purely for line-logging (`info!`/`warn!`), never `#[instrument]`/spans, and no
   subscriber layer for structured span export is configured. Spans add per-call
   allocation/dispatch overhead the user explicitly wanted to avoid, and the desired output
   (aggregated count + total time per operation, drained once per cycle) is a poor fit for a
   line-per-event log stream. Rejected.
3. **`HashMap<String, ...>` registry instead of a fixed enum-indexed array.** Would avoid having
   to enumerate `Metric` variants up front, but costs a hash + allocation per record call — this
   codebase already has one precedent for exactly this trade-off (`RegionId::as_index()` over a
   dense array) and follows it here for the same reason: cheap enough to call unconditionally in
   the Core tier.
4. **Keep the pilot's dual-write pattern** (separate `Instant::now()` for the old field, plus a
   `timed!` wrap for the new registry) **permanently, instead of converting fully.** This was the
   state after the first half of this session's work (see gaps.md's write-up) and is explicitly
   what the user asked to eliminate — "one way of doing metrics."

## Consequences

- New always-on runtime cost: one `Instant::now()`/`.elapsed()` pair (already present everywhere
  before this change) plus two atomic `fetch_add`s per Core-tier call site, on every build. This
  is strictly less code executed than before at sites that previously computed a duration by hand
  and then also fed it into a `tracing` field — the macro replaces both.
- `crates/somnus-metrics` is now a dependency of `somnus-perception`, `somnus-memory`, `somnus-sensory`,
  `somnus-encoding`, and `somnus` — a genuine cross-cutting concern shared by 5 crates, matching
  `CLAUDE.md`'s bar for a new crate ("split crates along architectural boundaries... only once
  those boundaries are real").
- `main.rs`'s `#![recursion_limit]` had to rise 256→512 to add the one new `"fine_metrics"` key to
  the already-large `json!{}` macro invocation — noted here since it's an easy thing to trip over
  again if more fields are added to that literal in the future.
- `somnus-memory` currently has no Core-tier call sites converted (it has no pre-existing hand-rolled
  timers — its I/O operations were never individually timed at all). Adding instrumentation there
  is a Fine-tier, opportunistic addition for a future investigation, not part of this conversion
  (per the crate's own doc-comment guardrail: add `Metric` variants as specific questions need
  them, not as an upfront exhaustive inventory).

## What would invalidate this

- If Core-tier overhead is ever measured to matter (unlikely given `Instant::now()`'s ~20-40ns
  cost on Linux and the operation-granularity guardrail this crate documents), the two-tier split
  would need a third, even-cheaper "sampled" tier, or Core itself would need to move behind a
  runtime flag instead of being unconditional.
- If a genuine cross-crate structured-tracing need emerges (span-based distributed tracing,
  OpenTelemetry export), this registry-based approach would likely be superseded rather than
  extended — it was deliberately scoped to "cheap always-on counters," not general observability.
