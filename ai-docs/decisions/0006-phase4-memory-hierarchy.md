# ADR 0006 — Phase 4: memory hierarchy (`somnus-memory`)

**Status:** accepted  
**Date:** 2026-05-10

## Context

Phase 3 produced 512-dim episode vectors from the binding area. Phase 4 makes those vectors retrievable: given a cue, the system should return the most similar past episodes. This is the first half of the recollection mechanism described in `architecture.md`.

The architecture specifies three tiers:

- **Hot (VRAM/episodic)** — bounded ring of recent episodes, source of replay.
- **Warm (RAM/semantic)** — HNSW index for similarity retrieval.
- **Cold (disk/archive)** — quantised long-term store.

Phase 4 implements all three tiers plus the recollection layer, with the cold→warm promotion path stubbed. The exit criterion is: given a cue episode vector, the system returns the top-k most similar episodes from semantic memory.

## Decisions

### 1. HNSW library: `hnsw_rs` 0.3

**Alternatives considered:**
- `usearch` — C++ core with Rust bindings; fastest, but any build or debugging issue crosses a language boundary.
- `hnsw_rs` — pure Rust; somewhat slower than usearch but the gap is irrelevant at the scales we expect (< 100k episodes before growth).
- Roll a simple flat-scan — O(n) per query; acceptable for tiny n but doesn't scale.

**Chosen:** `hnsw_rs`. The pure-Rust constraint avoids C++ FFI debugging overhead during active research. Can be swapped for `usearch` or a custom implementation once the interface is proven.

**Parameters:** M=16, max_elements=100_000, max_layers=16, ef_construction=400, search_ef=64. These are conservative HNSW defaults. Tune when scale warrants it.

**Distance:** L2 (`DistL2`), not cosine. Episode vectors come from a `linear_no_bias` projection with no normalisation; magnitudes vary. L2 is honest about this. Cosine would silently ignore magnitude differences, which in Phase 3 carry information about activation strength.

### 2. Eager insert policy

In Phase 4, every episode from the binding area is inserted into both the episodic buffer and the HNSW index immediately on perception. There is no sleep gate.

Sleep-gated consolidation (replay → selective semantic update, homeostatic downscaling, eviction to cold) is Phase 5 work. Eager insert satisfies the Phase 4 exit criterion without requiring the sleep machinery.

### 3. Cold archive: single file, INT8 per-vector quantisation

The cold archive is `memory/episodes.cold` — a single append-only binary file. Not sharded by region.

**Why not region-sharded?** Region sharding in `architecture.md` refers to per-region sparse codes (the k-WTA activations), not the unified episode vectors. Episode vectors are cross-region by construction (binding area concatenates all four regions). Introducing region-sharding for episodes in Phase 4 would be a premature and wrong decomposition.

**Format (little-endian per record):**
```
magic:     b"SMNCOLD\0"   (8 B, header only)
id:        u64            (8 B)
timestamp: f64            (8 B, seconds since Unix epoch)
scale:     f32            (4 B, dequant: v = q_i8 * scale)
values:    [i8; 640]      (640 B)  ← was 512 at Phase 4; grew to 640 with Phase 6 language region
─────────────────────────
total per record: 660 B  ← was 532 B at Phase 4
```

**Quantisation:** `scale = max(|values|) / 127.0`; per-vector, not per-archive. Guards against the all-zero case with `scale = 1.0` fallback. Expected error < 0.8% of max magnitude per element.

**Reading:** `memmap2` over the file; iterate at fixed stride. No random-access index in Phase 4 — promotion is stubbed.

### 4. `ColdArchive::append` and `RecollectionLayer::promote` implemented but not called

Both methods have their full signatures and bodies in Phase 4:
- `ColdArchive::append` — quantises and appends; tested.
- `RecollectionLayer::promote` — returns `Ok(())` immediately.

This locks the interface now so Phase 5 can fill the bodies without touching call sites in the `somnus` binary.

### 5. Episode id: session-local monotonic u64

`id` is assigned by `EpisodicBuffer::push` from a monotonic counter starting at 0 each run. No UUID, no persistence across restarts in Phase 4. Persistence (snapshot/restore) is a Phase 5 concern tied to the sleep-loop checkpoint mechanism.

## Consequences

- `crates/somnus-memory` added to workspace. New workspace deps: `hnsw_rs = "0.3"`, `memmap2 = "0.9"`.
- `somnus` binary gains `--episodic-capacity`, `--memory-dir`, `--top-k` CLI args.
- Every episode is inserted into both tiers; memory grows unbounded within a session (EpisodicBuffer evicts oldest; SemanticStore does not — bounded by `max_elements=100_000`).
- Cold archive file persists across runs; Phase 5 will leverage this for warm-up.
- Retrieval is log-only at `RUST_LOG=debug` in Phase 4; no retrieval output is surfaced to the user in normal operation.

## What this does not address

- Sleep-gated consolidation (Phase 5).
- Cold→warm promotion logic (Phase 5).
- Per-region sparse-code storage (a separate concern from episode storage; Phase 5+).
- Snapshot/restore of the HNSW index across restarts (Phase 5).
- Eviction from SemanticStore (Phase 5, tied to homeostatic downscaling).
