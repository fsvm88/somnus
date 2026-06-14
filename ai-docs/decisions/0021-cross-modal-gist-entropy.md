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

The existing gist probe (ADR 0007) measures episodic clustering quality: does the system
recall episodes from the same k-means cluster as their nearest warm-memory neighbor? This
tells us whether the episode space is well-organised, but not whether the same concept
is represented consistently across different senses.

If a dog seen in video and a dog heard barking end up in different gist clusters,
the system has no cross-modal binding for "dog". If they land in the same cluster,
that's evidence of grounding. Without a modality tag per episode, there is no way to
measure this.

## Decision

### 1. `IngestSource` enum on `EpisodeRecord`

Add an `IngestSource: u8` enum (Speech=0, Env=1, Visual=2, Language=3, Mixed=4,
Unknown=255) to `EpisodeRecord`. Derived at push time from which regional encoder inputs
were non-empty:

- region 0 only → Speech
- region 1 only → Env
- region 2 only → Visual
- region 3 only → Unknown (Symbolic is a derived convergence layer, not a raw encoder input)
- region 4 only → Language
- 2+ non-empty → Mixed
- none non-empty → Unknown

### 2. Cold archive schema bump (SMNCOLD2)

`ingest_source` is appended as 1 byte per record, extending `RECORD_BYTES` from 660 to
661. The magic bytes are bumped from `SMNCOLD1` to `SMNCOLD2` so old archives fail
gracefully with `BadMagic` rather than silently misreading data.

Old archives are incompatible. Always use a fresh `--memory-dir` when upgrading.

### 3. Cross-modal gist entropy metric

During the REM pre-spill probe, after k-means clustering:

- Assign each hot-tier episode to its nearest gist centroid.
- Build a per-cluster histogram of `ingest_source` values (5 known-source bins; Unknown
  excluded from entropy calculation).
- For each cluster with ≥2 known-source episodes, compute Shannon entropy (bits) over
  the source distribution.
- Report `cross_modal_gist_entropy`: mean entropy across occupied clusters.

**Interpretation:**
- 0 bits: all episodes in each cluster came from the same modality (no cross-modal binding).
- log2(5) ≈ 2.32 bits: maximum — perfectly uniform source mix per cluster.
- A value rising over sessions indicates the system is grouping multi-modal experiences
  into shared representational clusters.

This is a **diagnostic metric only**. No growth or pruning decisions are gated on it.
Once we have multi-session data and understand the typical range, it can inform
architectural changes or learning-rate schedules.

## Limitations

- Hot-tier only: the gist probe runs on the current session's episodes before spill.
  Episodes from prior sessions (in warm/cold) do not contribute source tags because
  cold→warm promotion does not currently carry `ingest_source` back through the warm
  HNSW. The metric therefore reflects within-session cross-modal mixing, not
  cross-session consolidation.
- With short training clips, a single session may produce few modalities, making entropy
  trivially low. The metric becomes informative once multi-modal clips are used.
- `Unknown` is excluded from entropy. If the system runs in audio-only mode, entropy will
  be 0 bits by construction — not a failure, just a configuration artefact.

## What would invalidate this

If we add cross-session source tracking (cold→warm promotion preserves `ingest_source`),
the probe should be extended to sample from warm as well. If salience-weighted cluster
assignment is added, the histogram weights should be adjusted accordingly.

## Amendment: cross-session gist histogram (2026-06-14)

The warm store's `records: HashMap<usize, EpisodeRecord>` already carries `ingest_source`
(and `salience`) for all cold→warm–promoted records via `rebuild_from_recent`. A new
`SemanticStore::sample_records(max_n)` method returns an evenly-sampled slice of those
records. After the hot-tier loop in the gist probe, a bounded warm sample (`GIST_SAMPLE`
records) is also binned into `cluster_sources` via `nearest_centroid`, giving
`cross_modal_gist_entropy` a cross-session signal once cold history is non-trivial.

Note: hot-tier records that are also in warm are counted twice; at production scales (warm
≫ hot) this is a negligible bias and not worth the filtering cost.

## Amendment: salience-weighted cluster assignment (2026-06-14)

`cluster_sources` changed from `Vec<[u32; 6]>` to `Vec<[f32; 6]>`. Each episode now
contributes `weight = salience.clamp(0.01, 2.0)` to its cluster/source slot instead of 1.
Episodes with `salience == f32::MAX` (first-session, warm was empty) use weight 1.0.
The entropy computation operates on `f32` weighted totals throughout; the cluster-occupied
threshold is now `>= 2.0` (weighted sum). `traces_for_rem` tuple extended to 4 elements
`(ep_vec, regional, ingest_source, salience)` and `collect_high_residual_inputs` in
`somnus-perception` updated to `<T, U>` to remain generic over the trailing two elements.
