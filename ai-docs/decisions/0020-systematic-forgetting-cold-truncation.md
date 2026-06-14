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

The cold archive (`episodes.cold`) is append-only. Without a bound, it grows indefinitely:
each sleep cycle spills the hot buffer into cold (~seconds of episodes per cycle). On a
machine running continuously, the archive would exceed hundreds of GB within days and
replay would never revisit early episodes anyway.

The warm HNSW index already has a size cap (`--max-warm-records`), rebuilt from cold's
tail each time it overflows. Cold itself had no equivalent bound.

## Decision

Add `--max-cold-records N` (default 0 = unlimited). During each sleep spill, after the
warm-HNSW rebuild check, if `cold.len() > N` the archive is truncated to its most recent N
records via `ColdArchive::truncate_to_recent(n)`.

`truncate_to_recent` reads only the tail slice (no full scan), rewrites the file as
`MAGIC + kept_bytes`, and updates the in-memory length. No temporary file needed — the
rewrite is atomic enough for a research setting (the old file is overwritten in place;
a crash mid-write leaves a partial file detectable by size check on reopen).

The cold archive format uses fixed-size records (`RECORD_BYTES = 661` bytes at `COLD_DIM=640`: 8B id + 8B timestamp + 4B scale + 640B quantised values + 1B ingest_source),
so "last N records" is a direct byte-offset calculation with no parsing required.

## Suggested value

`--max-cold-records 500000` ≈ 330 MB on disk. At a typical rate of ~1 000 episodes/cycle
this retains ~500 cycles of history, far more than any current training run.

## Alternatives considered

- **LRU / salience-based eviction:** keeps high-salience old episodes instead of
  blindly dropping the oldest. Correct in principle; complex to implement (requires
  reading salience tags not currently stored in cold). Deferred to Gap 8+ once we have
  richer per-record metadata.
- **Compaction into a smaller archive:** merge old records into a denser representation
  before discarding. No encoding for this yet; out of scope.
- **Leave cold unbounded:** acceptable short-term but becomes a practical problem after
  multi-day runs with the long clips.

## What would invalidate this

If we add salience metadata to cold records (Gap 8), revisit whether recency-based
truncation is still the right policy or whether salience-weighted eviction is worth
the complexity.

## Amendment: salience-weighted eviction (2026-06-14)

Salience (`f32`) is now added to `EpisodeRecord` and persisted in the cold archive under the `SMNCOLD3` magic (RECORD_BYTES 661 → 665). `EpisodeTrace.salience` was removed; the canonical value lives in `EpisodeRecord.salience` (set at `push()` time from the warm-recall L2 distance). `f32::MAX` is the sentinel for first-session records where warm memory is empty.

`evict_lowest_salience(n)` in `ColdArchive` replaces `truncate_to_recent(n)` in the sleep loop. It selects the top-n records by salience (descending) and rewrites them in chronological order. The bytes are copied into a `Vec<u8>` before truncating the file to avoid SIGBUS from mmapping a file that was just truncated.

`COLD_RECORD_BYTES` in `main.rs` updated from 661 to 665 for the disk-cap formula.
