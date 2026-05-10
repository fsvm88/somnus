# ADR 0007 — Phase 5: sleep cycle + consolidation (`somnus-sleep`)

**Status:** accepted  
**Date:** 2026-05-10

## Context

Phase 4 left the system in eager-insert mode: every episode vector lands in both the hot (EpisodicBuffer) and warm (SemanticStore) tiers immediately, without any consolidation, weight updates, or compression. The k-WTA and SAE weights from Phase 3 are still random. Phase 5 adds the sleep machinery that makes the hot→warm→cold pipeline meaningful and starts training the perception weights via experience replay.

Three smoke tests verified the design assumptions before this ADR was finalised:
- Candle autograd flows through a k-WTA straight-through mask (gradient reaches encoder Linear weights).
- VarMap must be kept alive after construction for optimizer, save, and load access.
- Growth via `Tensor::cat` + `Var::from_tensor` + Linear rebuild is feasible; optimizer state resets on growth.

## Decisions

### 1. EpisodeTrace — hot buffer stores encoder inputs alongside episode vector

The hot EpisodicBuffer is upgraded from `EpisodeRecord` (id + timestamp + episode vector) to `EpisodeTrace`:

```
EpisodeTrace {
    record:   EpisodeRecord,         // id, timestamp_secs, episode vector (512-dim)
    regional: [Vec<f32>; 4],         // encoder embeddings: [speech(384), env(512), visual(384), symbolic(384)]
}
```

`regional[i]` is the raw encoder output that was fed into region `i`'s k-WTA layer. It is needed to compute a training loss (SAE reconstruction MSE given the k-WTA output) during sleep replay.

The cold archive (ColdArchive) continues to store EpisodeRecord only. Per-region encoder inputs are transient hot-tier data; they are not worth the cold storage cost.

**Consequence:** `EpisodicBuffer::push` gains a `regional: [Vec<f32>; 4]` parameter. The `PerceptionOutput` struct gains `regional_inputs: Option<[Vec<f32>; 4]>` — populated whenever an episode vector is produced. The `somnus` binary passes this through to the buffer push.

### 2. PerceptionProcessor stores VarMap + optimizer

`PerceptionProcessor` is restructured to hold `varmap: VarMap`. The constructor no longer lets it go out of scope. The optimizer (AdamW) is created from `varmap.all_vars()` and also stored on the struct.

This enables:
- Weight save/load via `varmap.save(path)` / `varmap.load(path)`.
- Optimizer step during sleep via `opt.backward_step(&loss)`.
- Post-growth optimizer reset (recreate from updated vars).

The `PerceptionProcessor` also stores `latest_inputs: [Option<Vec<f32>>; 4]` — the most recent encoder input per region — so that `try_bind` can assemble `regional_inputs` alongside the episode vector.

### 3. k-WTA and SAE weight training during NREM-like replay

During the NREM-like stage, each EpisodeTrace in the hot buffer is replayed. For each region `i`:

1. Forward: `z = encoder_linear.forward(regional[i])` — tracked tensor.
2. Straight-through mask: top-K selection on `z`, create 0/1 mask tensor (untracked constant).
3. Sparse activation: `sparse_z = z * mask` — gradient flows through `z`.
4. SAE decode: `recon = sae_decoder.forward(sparse_z)`.
5. Loss: `mse = mean((regional[i] - recon)²)`.
6. Accumulate loss over all regions in the trace.
7. Optimizer step.

Duty-cycle boosting continues to run during wake-time inference. During replay, the boosting scalar is not applied to the training forward pass (we train on the raw linear output, not the boosted one).

**Learning rate:** AdamW with default params (lr=1e-3); tunable in config.

### 4. Homeostatic downscaling

After replay, all k-WTA Linear and SAE decoder weights are multiplied by `1 - DECAY_RATE` (default: `1e-4` per sleep cycle). Units whose indices appeared in at least one replay step during the NREM stage are protected from decay (their weight rows are not scaled down).

Mechanistically: after the replay loop collects the set of active unit indices per region, the weight decay skips those rows. This implements the "replay protection" variant of Tononi's synaptic homeostasis hypothesis: consolidation reinforces replayed content while reducing overall weight magnitude.

### 5. Compression: episodic → cold spill

At the end of NREM, cold-archive the entire hot buffer:

```
for trace in episodic.iter() {
    cold.append(&trace.record)?;
}
```

This fills `ColdArchive::append`, which was implemented but unconnected in Phase 4.

The SemanticStore (warm HNSW) is not evicted in Phase 5. Its 100k cap is the limit. Region-sharded eviction and warm→cold pruning are Phase 6+ work.

### 6. Cold→warm promotion: RecollectionLayer::promote() implemented

`promote(id)` now:
1. Iterates the cold archive (via memmap) to find the record with the given id.
2. Dequantises and inserts into `recall.warm` (SemanticStore).

A linear scan over the cold archive is acceptable in Phase 5 (cold len < 10k). An index is Phase 6 work.

### 7. REM-like stage: generative replay + growth trigger

After NREM, the REM stage measures reconstruction quality without updating weights:

1. For each trace in the hot buffer, compute mean SAE MSE per region (no optimizer step).
2. If `mean_mse[region] > GROWTH_THRESHOLD` (default: 0.1) for at least `GROWTH_MIN_SAMPLES=16` traces: flag that region for growth.

**Growth (simplified CC):**

For each flagged region, add `GROWTH_BATCH=64` new units:

1. Get current k-WTA encoder weight: shape `[N_UNITS, n_in]`.
2. Create `new_rows = Tensor::randn(0.0, GROWTH_INIT_STD, (GROWTH_BATCH, n_in))`.
3. `expanded_w = Tensor::cat(&[old_w, new_rows], 0)` → `[N_UNITS + GROWTH_BATCH, n_in]`.
4. Create `new_var = Var::from_tensor(&expanded_w)`.
5. Replace var in varmap; rebuild `KWtaLayer.linear` from `Linear::new(new_var.as_tensor(), None)`.
6. Extend `duty_cycle` vec with `GROWTH_BATCH` zeros.
7. Similarly expand SAE decoder weight from `[n_in, N_UNITS]` to `[n_in, N_UNITS + GROWTH_BATCH]` (add zero-initialised columns).
8. Recreate optimizer from `varmap.all_vars()` (resets Adam moments — acceptable).

Full cascade-correlation candidate-pool training (training new units on residual error before insertion) is deferred to Phase 6.

**Phase 5 limitation — growth is session-local:** `save_weights` / `load_weights` uses `VarMap::save` / `VarMap::load`, which requires that the saved and loaded var shapes match exactly. A grown region has a different encoder/decoder shape than the initial `N_UNITS`. This means weight restores from a pre-growth snapshot will fail silently (the grown vars are missing). In Phase 5, growth is not persisted across restarts — the system always restores to the `N_UNITS` baseline. Durable growth (saving the new shape + reloading into a resized layer) is Phase 6 work.

### 8. Sleep controller

The sleep controller is a synchronous function in `crates/somnus/src/sleep.rs` (not a separate crate). It blocks the main loop.

**Entry triggers (checked in the main select! loop after every perception event):**
- `episodic.len() >= HIGH_WATER` (default: 80% of episodic_capacity).
- `wake_elapsed >= MAX_WAKE_DURATION` (default: 10 minutes).

**Interruptibility:** An `AtomicBool` interrupt flag is set on `ctrl-c` or on a high-salience event (not implemented in Phase 5 — the flag is wired but the salience detector is a stub returning false). Each replay step checks the flag; if set, the sleep cycle exits early and the next wake cycle will resume replay from where it left off (tracked by a checkpoint index stored on `EpisodicBuffer`).

**Duration:** The cycle runs until all traces in the hot buffer have been replayed (or interrupted). No adaptive duration cap in Phase 5 — the full buffer is always replayed.

### 9. Snapshot / restore

Snapshot directory: `memory/snapshot/`. Write atomically:

1. Write to `memory/snapshot.tmp/`:
   - `episodic.bin` — custom binary: u64 count, then per-trace: EpisodeRecord bytes (same layout as cold archive per-record) + 4 × `[u16 len; f32 values...]` for regional embeddings.
   - `semantic.hnsw` — `hnsw_rs::Hnsw::file_dump("semantic", dir)`.
   - `weights.safetensors` — `varmap.save(path)`.
2. `std::fs::rename("memory/snapshot.tmp", "memory/snapshot")` — atomic on Linux (same filesystem).

On startup, if `memory/snapshot/` exists, restore:
1. Load `episodic.bin` into `EpisodicBuffer`.
2. Load `semantic.hnsw` via `hnsw_rs::Hnsw::load("semantic", dir)`.
3. Load `weights.safetensors` via `varmap.load(path)`.

`EpisodeTrace` ids from the previous session are restored verbatim; the `next_id` counter is set to `max(restored_ids) + 1`.

**Note on `hnsw_rs::Hnsw::file_dump/load`:** these methods write/read two files per call: `name.hnsw.data` and `name.hnsw.graph`. Both are written to the snapshot directory.

### 10. Eager insert removed from wake path

The Phase 4 eager insert into SemanticStore is removed. During wake, episodes go only into the hot EpisodicBuffer. The warm tier is populated by the sleep replay step. This aligns with the architecture.md spec: "does not update semantic memory during wake — all consolidation is deferred to sleep."

## Consequences

- `crates/somnus-perception`: `PerceptionProcessor` gains `varmap`, `optimizer`, `latest_inputs`; new `train_region`, `homeostatic_decay`, `grow_region` methods; `PerceptionOutput` gains `regional_inputs`.
- `crates/somnus-memory`: `EpisodeTrace` added; `EpisodicBuffer` stores traces; `EpisodeRecord` unchanged (cold tier); `RecollectionLayer::promote` implemented; snapshot/restore logic added.
- `crates/somnus/src/sleep.rs`: new module with `run_sleep_cycle`.
- `crates/somnus/src/main.rs`: sleep trigger in main loop; snapshot on exit.
- Workspace deps added: `safetensors` (for VarMap.save/load); no new crates needed.
- The `all_vars` call on the VarMap is sufficient for optimizer construction; no additional introspection APIs are needed.

## What this does not address

- Region-sharded warm-tier eviction (Phase 6).
- Cascade-correlation candidate training for growth (Phase 6).
- Salience-gated sleep interruptibility (Phase 6).
- Adaptive sleep duration (Phase 6).
- Neuromodulation signals (future).
- GPU execution (currently CPU; CUDA is a future concern when model scales warrant it).
