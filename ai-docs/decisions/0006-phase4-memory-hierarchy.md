---
status: accepted
date: 2026-05-10
deferred:
  - id: per-region-sparse-storage
    what: Store per-region sparse activation codes (k-WTA indices + values) separately from episode vectors for fine-grained replay and introspection
    when: When replay analysis shows episode vectors alone are insufficient for diagnostic queries
  - id: salience-weighted-eviction
    what: Replace recency-based cold truncation with salience-weighted eviction once salience metadata is stored in cold records
    when: DONE 2026-06-14 — salience added to EpisodeRecord + cold format (SMNCOLD3); evict_lowest_salience() replaces truncate_to_recent() in sleep.rs
  - id: cold-warm-promotion-index
    what: Build a cold archive index to support O(1) promotion by id (currently O(N) linear scan)
    when: When cold archive exceeds ~500k records and promotion latency is measurable
---

# ADR 0006 — Phase 4: memory hierarchy (`somnus-memory`)

**Status:** accepted
**Date:** 2026-05-10

> **Decision:** EpisodicBuffer (bounded ring, hot) + SemanticStore (hnsw_rs HNSW L2, warm) + ColdArchive (INT8 memmap2, cold) + RecollectionLayer; warm==cold union maintained at all times.
> **Keywords:** EpisodicBuffer, SemanticStore, ColdArchive, RecollectionLayer, hnsw_rs, HNSW, hot, warm, cold, memmap2, somnus-memory

## Context

Phase 3 produced 512-dim episode vectors from binding. Phase 4 makes them retrievable — given a cue, return the most similar past episodes, the first half of `architecture.md`'s recollection mechanism: **hot** (VRAM/episodic, bounded ring, replay source), **warm** (RAM/semantic, HNSW similarity retrieval), **cold** (disk/archive, quantised long-term). Phase 4 implements all three tiers + recollection, with cold→warm promotion stubbed. Exit criterion: given a cue vector, return top-k most similar episodes from semantic memory.

## Decisions

**1. HNSW library: `hnsw_rs` 0.3.** Alternatives: `usearch` (C++ core, fastest, but any build/debug issue crosses a language boundary); flat-scan (O(n)/query, fine at tiny n, doesn't scale). Chosen `hnsw_rs` — pure Rust avoids C++ FFI debugging overhead during active research, swappable later once the interface is proven. Params: M=16, max_elements=100,000, max_layers=16, ef_construction=400, search_ef=64 — conservative defaults, tune when scale warrants. Distance: L2 (`DistL2`), not cosine — episode vectors come from an unnormalised `linear_no_bias` projection with meaningful magnitude variance; cosine would silently discard that.

**2. Eager insert policy.** Every episode inserts into both the episodic buffer and HNSW immediately on perception, no sleep gate — sleep-gated consolidation (replay→selective update, homeostatic downscaling, cold eviction) is Phase 5 work; eager insert satisfies the Phase 4 exit criterion without the sleep machinery.

**3. Cold archive: single file, INT8 per-vector quantisation.** `memory/episodes.cold`, append-only, not region-sharded (region sharding in `architecture.md` refers to per-region sparse codes, not unified cross-region episode vectors — sharding episodes here would be a premature, wrong decomposition). Format (LE per record): magic `SMNCOLD\0`(8B header)+id(u64,8B)+timestamp(f64,8B)+scale(f32,4B)+values([i8;640],640B) = **660B/record** (was 532B at Phase 4, before the 640-dim language-region grow). Quantisation: `scale=max(|values|)/127.0` per-vector (not per-archive), `scale=1.0` fallback for all-zero, expected error <0.8% of max magnitude/element. Reading: `memmap2`, fixed-stride iteration, no random-access index (promotion stubbed).

**4. `ColdArchive::append`/`RecollectionLayer::promote` implemented but not called.** `append` quantises and appends (tested); `promote` returns `Ok(())` immediately — locks the interface now so Phase 5 fills the bodies without touching `somnus` binary call sites.

**5. Episode id: session-local monotonic u64.** Assigned by `EpisodicBuffer::push` from a counter starting at 0 each run — no UUID, no cross-restart persistence in Phase 4 (tied to the Phase 5 sleep-loop checkpoint mechanism).

## Consequences

`somnus-memory` added to the workspace; new deps `hnsw_rs="0.3"`, `memmap2="0.9"`. `somnus` gains `--episodic-capacity`/`--memory-dir`/`--top-k`. Every episode inserts into both tiers — memory grows unbounded within a session (EpisodicBuffer evicts oldest, SemanticStore doesn't, bounded by max_elements=100,000). Cold archive persists across runs for Phase 5 warm-up. Retrieval is debug-log-only in Phase 4, not surfaced to the user.

## What this does not address

Sleep-gated consolidation, cold→warm promotion logic, per-region sparse-code storage, HNSW snapshot/restore across restarts, SemanticStore eviction (all Phase 5+).
