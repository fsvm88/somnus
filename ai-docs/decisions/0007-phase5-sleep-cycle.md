---
status: accepted
date: 2026-05-10
deferred:
  - id: adaptive-sleep-duration
    what: Replace schedule/capacity sleep trigger with wakefulness pressure accumulator — see ADR 0024 S3
    when: ADR 0024 Phase 2 implementation
  - id: nrem-output-head-training
    what: Train output heads during NREM by re-running full k-WTA+binding forward pass on stored regional traces; see ADR 0009
    when: After output head wake-training converges and NREM replay is stable (adds significant NREM cost)
---

# ADR 0007 — Phase 5: sleep cycle + consolidation (`somnus-sleep`)

**Status:** accepted
**Date:** 2026-05-10

> **Decision:** NREM replay (AdamW + cosine LR, homeostatic decay) + REM gist probe (k-means, growth when gist_recall_rate < 0.5) + snapshot (weights + episodic + region_sizes); sleep triggered by high-water mark, schedule, or perf-degradation.
> **Keywords:** NREM, REM, sleep, replay, consolidation, growth, snapshot, AdamW, cosine LR, homeostatic, gist_recall_rate, sleep_trigger, high_water, schedule, perf_degradation

## Context

Phase 4 left the system in eager-insert mode: every episode lands in both hot and warm tiers immediately, no consolidation/weight updates/compression, k-WTA and SAE weights still random. Phase 5 adds the sleep machinery that makes hot→warm→cold meaningful and starts training perception weights via experience replay.

Three smoke tests verified design assumptions first: candle autograd flows through a k-WTA straight-through mask (gradient reaches encoder Linear weights); `VarMap` must be kept alive after construction for optimizer/save/load access; growth via `Tensor::cat`+`Var::from_tensor`+Linear rebuild is feasible, optimizer state resets on growth.

## Decisions

**1. `EpisodeTrace` — hot buffer stores encoder inputs alongside the episode vector.** Upgraded from `EpisodeRecord` (id+timestamp+vector) to `EpisodeTrace{record, regional:[Vec<f32>;5]}` (speech 384/env 512/visual 384/symbolic 384/language 896 — was `[4]` at Phase 5, language added Phase 6/ADR 0013). `regional[i]` is the raw encoder output fed into region i's k-WTA, needed for the SAE reconstruction training loss during sleep replay. Cold archive keeps storing `EpisodeRecord` only — per-region inputs are transient hot-tier data, not worth cold storage. `EpisodicBuffer::push` gains a `regional` param; `PerceptionOutput` gains `regional_inputs: Option<[Vec<f32>;4]>`.

**2. `PerceptionProcessor` stores VarMap + optimizer.** Holds `varmap: VarMap` (no longer let out of scope); AdamW created from `varmap.all_vars()`, also stored. Enables weight save/load (`varmap.save`/`load`), optimizer step during sleep, post-growth optimizer reset. Also stores `latest_inputs: [Option<Vec<f32>>;4]` so `try_bind` can assemble `regional_inputs`.

**3. k-WTA/SAE training during NREM replay.** Per trace, per region: forward `z=encoder_linear.forward(regional[i])` (tracked) → straight-through top-K mask (untracked constant) → `sparse_z=z*mask` (gradient flows through z) → SAE decode → `mse=mean((regional[i]-recon)²)` → accumulate across regions → optimizer step. Duty-cycle boosting runs during wake inference only, not on the replay training forward. AdamW default lr=1e-3, tunable.

**4. Homeostatic downscaling.** After replay, all k-WTA/SAE decoder weights × `(1-DECAY_RATE)` (default 1e-4/cycle); rows for units that appeared in at least one replay step are protected from decay. Implements the "replay protection" variant of Tononi's synaptic homeostasis hypothesis.

**5. Compression: episodic→cold spill.** End of NREM: `cold.append(&trace.record)` for the whole hot buffer — fills `ColdArchive::append` (implemented but unconnected at Phase 4). Warm HNSW isn't evicted in Phase 5 (100k cap is the limit; region-sharded eviction/pruning is Phase 6+).

**6. Cold→warm promotion implemented.** `promote(id)` scans the cold archive (memmap) for the record, dequantises, inserts into warm. Linear scan acceptable at Phase 5 scale (<10k cold records); an index is Phase 6 work.

**7. REM stage: generative replay + growth trigger.** No weight updates — per trace, mean SAE MSE per region; if `mean_mse[region] > GROWTH_THRESHOLD (0.1)` for ≥`GROWTH_MIN_SAMPLES(16)` traces, flag for growth. **Growth (simplified CC):** `GROWTH_BATCH=64` new units per flagged region — `new_rows=Tensor::randn(0,GROWTH_INIT_STD,(64,n_in))`, `cat` onto the encoder weight, new `Var`, rebuild `KWtaLayer.linear`, extend `duty_cycle` with zeros, expand the SAE decoder similarly (zero-init new columns), recreate the optimizer (resets Adam moments, acceptable). Full candidate-pool pre-training deferred to Phase 6.

**Phase 5 limitation — growth is session-local.** `VarMap::save`/`load` requires exact shape match; a grown region's shape differs from the initial `N_UNITS`, so restoring from a pre-growth snapshot fails silently (grown vars missing). Growth isn't persisted across restarts in Phase 5 — always restores to the `N_UNITS` baseline. Durable growth is Phase 6 work.

**8. Sleep controller.** Synchronous function in `sleep.rs` (not a separate crate), blocks the main loop. **Entry triggers** (checked in the main `select!` loop): `episodic.len() >= HIGH_WATER` (80% capacity default) or `wake_elapsed >= MAX_WAKE_DURATION` (10min default). **Interruptibility:** an `AtomicBool` flag set on ctrl-c or a high-salience event (stub returning false in Phase 5) — each replay step checks it, exits early on set, resumes from a checkpoint index on `EpisodicBuffer` next cycle. **Duration:** no adaptive cap in Phase 5 — full buffer always replayed (or interrupted).

**9. Snapshot/restore.** `memory/snapshot.tmp/` written atomically (rename to `memory/snapshot/`, same-filesystem atomic on Linux): `episodic.bin` (u64 count + per-trace `EpisodeRecord` bytes + 4×`[u16 len; f32 values]` regional embeddings), `weights.safetensors` (`varmap.save`). Restore on startup: load episodic.bin, rebuild warm `SemanticStore` from cold archive + any unspilled hot records, load weights, `next_id = max(restored_ids)+1`.

**Why rebuild instead of persisting the HNSW file?** `hnsw_rs::Hnsw::load_hnsw`'s lifetime is tied to its `HnswIo` loader (for mmap support) — storing `Hnsw<'static>` while persisting across calls would need unsafe lifetime erasure. Rebuilding from the (already-durable) cold archive sidesteps this: O(N log N), fast enough at Phase 5 scale (<10K records). Worth the unsafe complexity only past ~100K+ records — deferred to that milestone.

**Quantization note:** cold vectors are INT8-quantised with per-vector scale — dequantised L2 differs slightly from f32. Episode vectors are 640-dim (Phase 5 was 512, grew with Phase 6's language region); `RECALL_BLEND_THRESHOLD=5.5` stays valid for L2-normalised 640-dim vectors.

**10. Eager insert removed from wake path.** Phase 4's eager SemanticStore insert is gone — wake only writes hot `EpisodicBuffer`; warm is populated by sleep replay, matching `architecture.md`'s "no semantic memory updates during wake."

## Consequences

- `somnus-perception`: `PerceptionProcessor` gains `varmap`/`optimizer`/`latest_inputs`; new `train_region`/`homeostatic_decay`/`grow_region`; `PerceptionOutput` gains `regional_inputs`.
- `somnus-memory`: `EpisodeTrace` added; `EpisodicBuffer` stores traces; `EpisodeRecord` unchanged (cold); `RecollectionLayer::promote` implemented; snapshot/restore added.
- `somnus/src/sleep.rs`: new module, `run_sleep_cycle`. `somnus/src/main.rs`: sleep trigger in main loop, snapshot on exit.
- New workspace dep: `safetensors` (VarMap save/load). `all_vars()` is sufficient for optimizer construction, no extra introspection needed.

## What this does not address

Region-sharded warm eviction (Phase 6); salience-gated sleep interruptibility (Phase 6); adaptive sleep duration (Phase 6, ADR 0024 S3+P1b); neuromodulation (ADR 0024); GPU execution (CPU-only until model scale warrants CUDA).

---

## Amendment — Cascade-correlation growth init implemented (2026-06-14, cbd8f63)

The deferred `cascade-correlation-training` item shipped as **seeded growth init** (P2b) instead of full candidate-pool training: `collect_high_residual_inputs<T>()` scores hot-buffer traces by per-region SAE reconstruction MSE, returns the top `2×GROWTH_BATCH` sorted descending; `grow_region_seeded()` seeds each new encoder column with a normalized seed×0.01 (decoder rows stay zero-init), falling back to `grow_region()` on empty seeds; `sleep.rs` collects seeds before REM growth and passes them through. A lightweight CC approximation — new units bias toward inputs the model already fails on, rather than random init. Full candidate-pool pre-training remains deferred until growth-quality metrics stabilize.
