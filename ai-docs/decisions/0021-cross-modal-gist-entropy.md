---
status: accepted
date: 2026-05-30
deferred:
  - id: cross-session-ingest-source
    what: Preserve ingest_source through cold→warm promotion so the gist entropy metric covers cross-session mixing, not just within-session
    when: DONE 2026-06-14 — warm store already holds full EpisodeRecord (incl. ingest_source) via rebuild_from_recent; gist probe now also bins a bounded warm sample (GIST_SAMPLE records) into cluster_sources after the hot-tier loop
  - id: salience-weighted-cluster-assignment
    what: Weight cluster assignment by episode salience in the gist entropy calculation to de-emphasise repetitive familiar content
    when: DONE 2026-06-14 — cluster_sources now accumulates f32 weights per slot; weight = salience.clamp(0.01, 2.0), f32::MAX sentinel → 1.0
---

# ADR 0021 — Cross-modal gist entropy: ingest_source and concept probe

**Status:** accepted
**Date:** 2026-05-30

> **Decision:** Add IngestSource enum on EpisodeRecord to track modality; add cross_modal_gist_entropy metric (Shannon entropy over ingest_source distribution per cycle).
> **Keywords:** IngestSource, gist entropy, cross_modal_gist_entropy, concept probe, modality balance, SMNHOT02, asr_count, blend_count

## Context

The existing gist probe (ADR 0007) measures episodic clustering quality — does the system recall episodes from the same k-means cluster as their nearest warm-memory neighbor? That says whether the episode space is well-organised, not whether the same concept is represented consistently across senses. If a dog seen and a dog heard land in different gist clusters, there's no cross-modal binding for "dog"; same cluster is evidence of grounding. Without a modality tag per episode there's no way to measure this.

## Decision

**1. `IngestSource` enum on `EpisodeRecord`** (Speech=0, Env=1, Visual=2, Language=3, Mixed=4, Unknown=255), derived at push time from which regional encoder inputs were non-empty: region 0 only→Speech, 1→Env, 2→Visual, 3→Unknown (Symbolic is a derived convergence layer, not a raw input), 4→Language, 2+→Mixed, none→Unknown.

**2. Cold archive schema bump (SMNCOLD2).** `ingest_source` appended as 1 byte/record, `RECORD_BYTES` 660→661, magic bumped `SMNCOLD1`→`SMNCOLD2` so old archives fail loudly (`BadMagic`) rather than silently misreading. Old archives incompatible — use a fresh `--memory-dir` when upgrading.

**3. Cross-modal gist entropy metric.** During the REM pre-spill probe, after k-means: assign each hot-tier episode to its nearest gist centroid, build a per-cluster `ingest_source` histogram (5 known-source bins, Unknown excluded), compute Shannon entropy (bits) per cluster with ≥2 known-source episodes, report `cross_modal_gist_entropy` = mean entropy across occupied clusters. 0 bits = single modality per cluster (no cross-modal binding); log2(5)≈2.32 bits = maximum uniform mix; rising over sessions = growing multi-modal grouping. **Diagnostic only** — no growth/pruning decisions gate on it, informs architecture/LR schedule decisions once the typical range is understood.

## Limitations

Hot-tier only at introduction — cold→warm promotion didn't carry `ingest_source`, so the metric reflected within-session mixing only (fixed by the 2026-06-14 amendment below). Short clips → few modalities per session → trivially low entropy until multi-modal clips are used. `Unknown` excluded from entropy — audio-only runs read as 0 bits by construction, a configuration artefact not a failure.

## What would invalidate this

Cross-session source tracking added → extend the probe to sample warm too (done, see amendment). Salience-weighted cluster assignment added → adjust histogram weights accordingly (done, see amendment).

## Amendment: cross-session gist histogram (2026-06-14)

The warm store's `records: HashMap<usize, EpisodeRecord>` already carries `ingest_source`/`salience` for cold→warm-promoted records via `rebuild_from_recent`. New `SemanticStore::sample_records(max_n)` returns an evenly-sampled slice; after the hot-tier loop, a bounded warm sample (`GIST_SAMPLE` records) is also binned into `cluster_sources` via `nearest_centroid`, giving `cross_modal_gist_entropy` a real cross-session signal once cold history is non-trivial. Hot-tier records also present in warm are double-counted — negligible bias at production scale (warm≫hot), not worth filtering.

## Amendment: proportional regional attribution + -0.0 fix (2026-06-15)

The original `IngestSource`-slot approach collapsed all multi-modal (video+audio) episodes into `Mixed`, keeping `cross_modal_gist_entropy` always ~0 and hitting an IEEE 754 `-0.0` artefact (`-1.0*log2(1.0)=-0.0`). Two fixes (`sleep.rs`):

1. **Proportional regional attribution:** `cluster_sources` is now filled by decomposing each episode vector into 5×(EPISODE_DIM/5) regional slices, L2-norming each, normalising to sum=1, and distributing the episode's salience weight proportionally across all 5 slots — meaningful entropy even for Mixed-tagged episodes, since the regional breakdown is embedded in the vector itself. New helper `regional_weights(ep_vec)`. `IngestSource` no longer feeds the histogram (still stored on `EpisodeRecord` for episode counting/other uses).
2. **-0.0 fix:** per-term entropy returns `0.0` explicitly when `p>=1.0` instead of evaluating `-p*p.log2()`.

`cluster_sources` stays `Vec<[f32;6]>` (slot 5 unused/zero, no type change needed). **Long-term note:** norm-based attribution is a proxy for true regional presence (norms reflect binding projections, not raw k-WTA activations) — if finer attribution is needed, store a `regional_presence: u8` bitmask on `EpisodeRecord` (SMNCOLD4 bump) and use explicit per-region flags instead.

## Amendment: salience-weighted cluster assignment (2026-06-14)

`cluster_sources` changed `Vec<[u32;6]>`→`Vec<[f32;6]>` — each episode contributes `weight=salience.clamp(0.01,2.0)` (or 1.0 if `salience==f32::MAX`, first-session/empty-warm) instead of a flat 1. Entropy computed on f32 weighted totals throughout; cluster-occupied threshold now `>=2.0` (weighted sum). `traces_for_rem` extended to 4 elements `(ep_vec, regional, ingest_source, salience)`; `collect_high_residual_inputs` in `somnus-perception` updated to `<T,U>` to stay generic over the trailing two elements.
