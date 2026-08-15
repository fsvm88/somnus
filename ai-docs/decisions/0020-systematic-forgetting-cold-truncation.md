---
status: accepted
date: 2026-05-30
---

# ADR 0020 — Systematic forgetting: bounded cold archive

**Status:** accepted
**Date:** 2026-05-30

> **Decision:** Add --max-cold-records N; during sleep spill, truncate cold archive to most recent N records (FIFO) to prevent unbounded growth.
> **Keywords:** cold archive, max-cold-records, truncation, systematic forgetting, SMNCOLD2, bounded archive, FIFO

## Context

`episodes.cold` is append-only with no bound — on a continuously running machine it would exceed hundreds of GB within days, and replay never revisits early episodes anyway. The warm HNSW index already caps at `--max-warm-records` (rebuilt from cold's tail on overflow); cold had no equivalent.

## Decision

New `--max-cold-records N` (default 0=unlimited). During each sleep spill, after the warm-HNSW rebuild check, if `cold.len()>N` the archive truncates to the most recent N records via `ColdArchive::truncate_to_recent(n)` — reads only the tail slice (no full scan), rewrites the file as `MAGIC+kept_bytes`, updates the in-memory length. No temp file — overwrite-in-place is atomic enough for a research setting (a crash mid-write leaves a partial file, detectable by size check on reopen). Fixed-size records (`RECORD_BYTES=661` at `COLD_DIM=640`: 8B id+8B timestamp+4B scale+640B quantised values+1B ingest_source) make "last N" a direct byte-offset calculation, no parsing needed.

**Suggested value:** `--max-cold-records 500000` ≈ 330MB on disk — at ~1,000 episodes/cycle, retains ~500 cycles of history, far more than any current run.

## Alternatives considered

- **LRU/salience-based eviction.** Keeps high-salience old episodes instead of blindly dropping oldest — correct in principle, but needed salience tags not yet stored in cold. Deferred until richer per-record metadata exists (done — see amendment).
- **Compaction into a denser archive.** No encoding exists for this yet, out of scope.
- **Leave cold unbounded.** Acceptable short-term, becomes a practical problem after multi-day long-clip runs.

## What would invalidate this

If salience metadata is added to cold records, revisit whether recency-based truncation is still right or salience-weighted eviction is worth the complexity.

## Amendment: salience-weighted eviction (2026-06-14)

Salience (`f32`) added to `EpisodeRecord`, persisted in cold under the `SMNCOLD3` magic (RECORD_BYTES 661→665). `EpisodeTrace.salience` removed — the canonical value lives in `EpisodeRecord.salience` (set at `push()` from warm-recall L2 distance; `f32::MAX` sentinel for first-session records where warm is empty). `evict_lowest_salience(n)` replaces `truncate_to_recent(n)` in the sleep loop — selects the top-n records by salience (descending), rewrites chronologically. Bytes are copied into a `Vec<u8>` before truncating to avoid SIGBUS from mmapping a just-truncated file. `COLD_RECORD_BYTES` in `main.rs` updated 661→665 for the disk-cap formula.
