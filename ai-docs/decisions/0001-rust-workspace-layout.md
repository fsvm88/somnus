# 0001. Rust workspace layout

- **Status:** accepted
- **Date:** 2026-04-26

## Context

The project is starting from scratch in Rust. Before writing any code, the crate
and module structure needs to be fixed so that work can proceed in a consistent
direction. Phase 1 delivers a working sensory pipeline: audio and video capture,
framed, timestamped, and fed into a consuming loop. No encoding, no memory, no
sleep. Future phases add encoding (`somnus-encoding`), perception (`somnus-perception`),
memory (`somnus-memory`), and sleep (`somnus-sleep`).

Key pressures:

- **Do not over-split.** An empty `somnus-types` or `somnus-telemetry` crate for Phase 1
  anticipates a dependency graph that does not yet exist. Split when a second crate
  needs the shared piece, not before.
- **Do not add a `SensorSource<F>` trait.** The "three concrete uses → one
  abstraction" rule applies. Phase 1 has zero callers of such a trait.
- **cpal callbacks cannot be async.** The capture bridge must be designed around
  this constraint: cpal callback → sync channel → async bridge task.
- **Split ML runtime from day 1** (structurally). `ort` handles frozen encoder
  inference; `candle` handles cognitive-layer training. These live in Phase 2+
  crates; Phase 1 does not use either.

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

No `SensorSource<F>` trait in Phase 1. `AudioCapture` and `VideoCapture` are
concrete structs with similar but not identical APIs.

**Reserved crate structure for Phase 2+** (do not create until the phase begins):

| Crate | Phase | Purpose |
|-------|-------|---------|
| `somnus-encoding` | 2 | Frozen encoder inference via `ort`; `AudioFrame → AudioEmbedding`, `VideoFrame → VideoEmbedding` |
| `somnus-perception` | 3 | Regional concept areas (k-WTA sparse coding via `candle`); `Embedding → SparseActivation` |
| `somnus-memory` | 4 | Episodic (VRAM), semantic (RAM), cold archive (disk); HNSW recollection index |
| `somnus-sleep` | 5 | Sleep cycle orchestration; replay, homeostasis, compression, growth |

Each crate is created when the phase begins, not before. Its interface will be
designed in a companion ADR at that time.

**Shared dependencies** go in `[workspace.dependencies]` in the root `Cargo.toml`
to keep versions consistent.

## Alternatives considered

**Single flat crate (lib + binary in one).** Simpler for Phase 1, but the
`somnus-encoding` split in Phase 2 would require restructuring. Workspace layout now
avoids that churn.

**`somnus-types` shared crate.** Natural place for `AudioFrame`, `VideoFrame`, etc.,
once `somnus-encoding` needs them. But for Phase 1 there is only one consumer
(`somnus-sensory` itself), so the split is premature. When `somnus-encoding` is created,
*if* it needs these types, extract them then.

**`somnus-telemetry` crate.** Wrapping `tracing` in a project crate for Phase 1 is
pure ceremony. `tracing` is already a stable, well-scoped API. Add
`somnus-telemetry` only if project-specific telemetry conventions become complex
enough to justify a crate (Phase 4+).

**`SensorSource<F>` trait.** Would unify `AudioCapture` and `VideoCapture` behind
a common interface. Not needed: Phase 1 has one binary that wires them both
explicitly, and their APIs are already different enough (audio is callback-push,
video is frame-pull) that a unified trait would force a false abstraction.

## Consequences

- The workspace is established once and not restructured until Phase 2.
- Phase 2 creates `somnus-encoding`; at that point `somnus-sensory`'s public frame types
  may need to be factored into a shared crate. An ADR will decide that when it
  arrives.
- The binary (`crates/somnus/`) owns configuration, tracing initialization, and
  top-level error handling. It is the only place that uses `anyhow`. Library
  crates use `thiserror`.

## What would invalidate this

- A second library crate in Phase 1 that needs `AudioFrame` or `VideoFrame` would
  force an early `somnus-types` extraction.
- If Phase 2 encoding work begins before the Phase 1 capture pipeline is working,
  the crate split schedule should be revisited.
