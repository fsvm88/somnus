# ADR 0017 — Phase 12b: HNSW warm-index persistence

**Status:** accepted
**Date:** 2026-05-18

> **Decision:** Dump warm HNSW index to disk after each sleep (hnsw.file_dump + records.bin sidecar + warm/version tag); load on restart to avoid cold-rebuild (~3min → <5s).
> **Keywords:** HNSW persistence, warm index, dump, load, Box::leak, warm/version, SMNWARM, restart time, SemanticStore

## Context

On every restart, `load_snapshot` rebuilds the warm HNSW index by iterating the entire cold archive — ~3min at 164k records, worsening linearly, making the binary unusable for quick iteration. `hnsw_rs` 0.3.4 provides `Hnsw::file_dump(dir, basename)` (writes `.hnsw.data`/`.hnsw.graph`) and `HnswIo::load_hnsw()` (reads them back) — but the loaded index isn't `'static` (lifetime tied to the `HnswIo` reader), while `SemanticStore` requires `Hnsw<'static, f32, DistL2>`.

## Decision

**Lifetime resolution:** hold `HnswIo` alongside `SemanticStore` as a lifetime parent (self-referential storage, which Rust doesn't support natively) — practical resolution is `Box::leak`: leak the `HnswIo` allocation after loading so the returned `Hnsw` is effectively `'static`. A one-time fixed cost per restart (~a few KB), reclaimed at process exit. Safe because: loaded exactly once per process lifetime (startup); `HnswIo` is otherwise unused after loading; no way to trigger a second load that would double-leak.

**Sidecar records:** `hnsw_rs` only persists graph structure, not `EpisodeRecord` data — the `records: HashMap<u64, EpisodeRecord>` sidecar is serialised separately via `bincode`, at `<memory_dir>/warm/records.bin`.

**Persistence points:** on sleep — after hot→cold spill, before snapshot write, `SemanticStore::dump(memory_dir/warm/)` (keeps HNSW in sync with cold). On startup — if warm dump files exist and their version tag matches, load instead of cold-rebuild; if absent/mismatched/corrupt, fall back to cold-rebuild and log a warning.

**Version tag:** 4-byte magic + u16 LE version at `warm/version`, bumped when `EPISODE_DIM`/`HNSW_M`/`HNSW_MAX_LAYERS` change — mismatch triggers cold-rebuild and overwrite.

**File layout:**
```
memory/warm/
  version          # 6 bytes: magic 'SMNW'(4B) + u16 version(2B)
  index.hnsw.data
  index.hnsw.graph
  records.bin      # bincode-encoded HashMap<u64, EpisodeRecord>
```

## Alternatives considered

- **Lifetime-parametric `SemanticStore<'a>`.** Viral — propagates into `RecollectionLayer`, `main.rs`, every call site — not worth it for one startup use.
- **`Arc<Mutex<HnswIo>>`.** Runtime overhead for something touched once, doesn't cleanly resolve the lifetime mismatch.
- **Switch ANN library.** `hnsw_rs` works and is well-tested — a startup-time problem doesn't justify a crate swap.

## Exit criterion

Restart time <5s for ≥100k records (vs. ~3min cold-rebuild).

## What would invalidate this decision

`hnsw_rs` releasing a version where `load_hnsw` returns `Hnsw<'static,...>` — `Box::leak` could be removed without touching the rest of the design. The sidecar growing large enough to make bincode serialisation slow (>a few seconds) — switch to a memory-mapped approach like `ColdArchive`.
