# ADR 0017 — Phase 12b: HNSW warm-index persistence

**Status:** accepted  
**Date:** 2026-05-18

> **Decision:** Dump warm HNSW index to disk after each sleep (hnsw.file_dump + records.bin sidecar + warm/version tag); load on restart to avoid cold-rebuild (~3min → <5s).  
> **Keywords:** HNSW persistence, warm index, dump, load, Box::leak, warm/version, SMNWARM, restart time, SemanticStore

## Context

On every restart, `load_snapshot` rebuilds the warm HNSW index by iterating the entire
cold archive and inserting every record. At 164 k records this takes ~3 minutes, which
makes the binary unusable for quick iteration and will worsen linearly with archive size.

The hnsw_rs 0.3.4 crate provides `Hnsw::file_dump(dir, basename)` which writes two
files (`.hnsw.data`, `.hnsw.graph`) and `HnswIo::load_hnsw()` which reads them back.
The loaded index is not `'static` — its lifetime is tied to the `HnswIo` reader — but
the current `SemanticStore` type requires `Hnsw<'static, f32, DistL2>`.

## Decision

### Lifetime resolution

Hold `HnswIo` alongside `SemanticStore` as a lifetime parent, stored in the struct
itself.  Concretely:

```rust
pub struct SemanticStore {
    _io: Option<Box<HnswIo>>,          // keeps the allocation alive
    index: Hnsw<'???, f32, DistL2>,    // borrows from _io when loaded
    ...
}
```

This requires self-referential storage, which Rust does not support natively.  The
practical resolution is `Box::leak`: leak the `HnswIo` allocation after loading so the
returned `Hnsw` is effectively `'static`.  The leaked allocation is a one-time fixed
cost per restart (~a few KB of reader state); it is reclaimed when the process exits.
This is safe because:
- We load exactly once per process lifetime (at startup).
- The `HnswIo` object is otherwise unused after loading.
- There is no way to trigger a second load that would double-leak.

### Sidecar records

hnsw_rs only persists the graph structure, not the associated `EpisodeRecord` data.
The `records: HashMap<u64, EpisodeRecord>` sidecar must be serialised separately using
`bincode` (already a workspace dependency).  The sidecar file lives next to the HNSW
files: `<memory_dir>/warm/records.bin`.

### Persistence points

- **On sleep**: after hot→cold spill and before snapshot write, call
  `SemanticStore::dump(memory_dir/warm/)`.  This keeps HNSW state in sync with cold.
- **On startup**: if the warm dump files exist and their version tag matches, load them
  instead of rebuilding from cold.  If absent, version-mismatched, or corrupt, fall
  back to cold-rebuild (existing path) and log a warning.

### Version tag

Write a 4-byte magic + 2-byte version (u16 LE) to `warm/version` alongside the HNSW
files.  Bump the version when `EPISODE_DIM`, `HNSW_M`, or `HNSW_MAX_LAYERS` change.
On mismatch, cold-rebuild and overwrite the dump.

### File layout

```
memory/
  warm/
    version          # 6 bytes: magic 'SMNW' (4 B) + u16 version (2 B)
    index.hnsw.data
    index.hnsw.graph
    records.bin      # bincode-encoded HashMap<u64, EpisodeRecord>
```

## Alternatives considered

- **Lifetime-parametric `SemanticStore<'a>`** — viral change: propagates into
  `RecollectionLayer`, `main.rs`, every call site.  Not worth it for one startup use.
- **Arc<Mutex<HnswIo>>** — adds runtime overhead for something that is only touched
  once; does not resolve the lifetime mismatch cleanly.
- **Rewrite SemanticStore to use a different ANN library** — hnsw_rs is working and
  well-tested; switching crates is not justified by a startup-time problem.

## Exit criterion

Restart time < 5 s for a store with ≥ 100 k records (vs. current ~3 min from cold-rebuild).

## What would invalidate this decision

If hnsw_rs releases a version where `load_hnsw` returns `Hnsw<'static, ...>`, the
`Box::leak` can be removed without touching the rest of the design.
If the sidecar grows large enough to make bincode serialisation slow (> a few seconds),
switch to a memory-mapped approach (similar to `ColdArchive`).
