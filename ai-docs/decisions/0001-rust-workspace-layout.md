# 0001. Rust workspace layout

- **Status:** accepted
- **Date:** 2026-04-26

> **Decision:** Single Cargo workspace; crates split along real architectural boundaries only, not speculatively.
> **Keywords:** cargo, workspace, somnus-sensory, somnus-encoding, somnus-perception, somnus-memory, crate layout

## Context

Starting from scratch in Rust — the crate/module structure needs fixing before any code so work proceeds consistently. Phase 1 delivers a working sensory pipeline (audio+video capture, framed, timestamped, fed into a consuming loop — no encoding, memory, or sleep). Future phases add encoding (`somnus-encoding`), perception (`somnus-perception`), memory (`somnus-memory`), sleep (`somnus-sleep`).

**Key pressures:** don't over-split (an empty `somnus-types`/`somnus-telemetry` crate for Phase 1 anticipates a dependency graph that doesn't exist yet — split when a second crate actually needs the shared piece). Don't add a `SensorSource<F>` trait (the "three concrete uses → one abstraction" rule — Phase 1 has zero callers). cpal callbacks can't be async — the capture bridge design must be built around cpal callback→sync channel→async bridge task. Split the ML runtime structurally from day 1 (`ort` for frozen encoder inference, `candle` for cognitive-layer training) — these live in Phase 2+ crates, Phase 1 uses neither.

## Decision

**Phase 1 workspace:**

```
somnus/                         ← git repo root
  Cargo.toml                 ← [workspace] only; no package here
  crates/
    somnus-sensory/             ← library: capture, frame types, channels, errors
      Cargo.toml
      src/
        lib.rs
        frame.rs
        audio.rs
        video.rs
        error.rs
        metrics.rs
    somnus/                     ← binary: wires everything; tracing init; config; main loop
      Cargo.toml
      src/
        main.rs
```

No `SensorSource<F>` trait — `AudioCapture`/`VideoCapture` are concrete structs with similar but not identical APIs.

**Reserved for Phase 2+** (created only when that phase begins, interface designed in a companion ADR at that time):

| Crate | Phase | Purpose |
|---|---|---|
| `somnus-encoding` | 2 | Frozen encoder inference via `ort`; `AudioFrame→AudioEmbedding`, `VideoFrame→VideoEmbedding` |
| `somnus-perception` | 3 | Regional concept areas (k-WTA sparse coding via `candle`); `Embedding→SparseActivation` |
| `somnus-memory` | 4 | Episodic (VRAM), semantic (RAM), cold archive (disk); HNSW recollection index |
| `somnus-sleep` | 5 | Sleep cycle orchestration; replay, homeostasis, compression, growth |

Shared dependencies go in root `Cargo.toml`'s `[workspace.dependencies]` for version consistency.

## Alternatives considered

- **Single flat crate (lib+binary in one).** Simpler for Phase 1, but Phase 2's `somnus-encoding` split would need restructuring — the workspace layout avoids that churn now.
- **`somnus-types` shared crate.** Natural home for `AudioFrame`/`VideoFrame` once `somnus-encoding` needs them, but Phase 1 has only one consumer (`somnus-sensory` itself) — premature. Extract when `somnus-encoding` actually needs it.
- **`somnus-telemetry` crate.** Wrapping `tracing` for Phase 1 is pure ceremony — `tracing`'s API is already stable and well-scoped. Add only if project-specific telemetry conventions get complex enough to justify it (Phase 4+).
- **`SensorSource<F>` trait.** Would unify `AudioCapture`/`VideoCapture` — not needed, Phase 1 has one binary wiring both explicitly and their APIs already differ enough (audio callback-push, video frame-pull) that unifying would force a false abstraction.

## Consequences

The workspace is established once, not restructured until Phase 2. Phase 2's `somnus-encoding` may need `somnus-sensory`'s public frame types factored into a shared crate — an ADR decides that when it arrives. The binary (`crates/somnus/`) owns configuration, tracing init, top-level error handling — the only place using `anyhow`; library crates use `thiserror`.

## What would invalidate this

A second Phase-1 library crate needing `AudioFrame`/`VideoFrame` would force an early `somnus-types` extraction. If Phase 2 encoding work begins before Phase 1's capture pipeline works, revisit the crate-split schedule.
