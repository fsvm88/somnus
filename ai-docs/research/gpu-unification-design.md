# GPU Unification Design

> **Status:** design complete, not yet implemented
> **Date:** 2026-06-02
> **Companion ADR:** decisions/0023-gpu-unification.md

---

## 1. Problem and motivation

The current architecture runs two GPU frameworks in the same process:

- **ORT + MIGraphX** — frozen encoder models (Whisper-tiny, CLAP-HTSAT, DINOv2-S, Qwen2.5)
- **burn NdArray (CPU)** — all perception inference and training (k-WTA, binding, SAE, output heads)

This split was workable at current scale (1K units/region, batch-1) because perception takes only ~35ms/episode and hides behind the GPU encode path (~96ms). However two problems compound as the system scales:

1. **Memory bandwidth wall on CPU.** At current scale, ~19 MB of weights must be read from system RAM per episode. DDR5 bandwidth is ~70 GB/s; VRAM bandwidth on the RX 9070 XT is ~640 GB/s — a 9× difference. The perception path is already memory-bandwidth-bound (confirmed in Perf G: "bottleneck is memory bandwidth").

2. **wgpu conflict (commit 5dbf9c9).** A previous attempt to run burn on the GPU failed because MIGraphX had already claimed the AMD device. Two GPU frameworks cannot coexist on this GPU. The only resolution is to consolidate on one framework — ORT/MIGraphX is the right choice because the frozen encoders cannot move off ORT.

The design goal is: **all GPU-bound compute runs through ORT/MIGraphX; burn remains on CPU for training only; weights are synced at sleep→wake boundaries.**

---

## 2. Scaling analysis

### Memory-bandwidth model

GPU vs CPU perception latency, modeled as memory-bandwidth-bound (dominant cost is reading weight matrices):

| Scale | Wt (MB) | CPU (ms) | GPU ms | Winner | Speedup |
|---|---|---|---|---|---|
| 1× current (1K units, 384 enc) | 19 | 0.26 | 0.67 | CPU | 0.4× |
| 4× units (4K units, 512 enc) | 129 | 1.84 | 0.84 | **GPU** | **2.2×** |
| 16× units (16K units, 1K enc) | 1,393 | 19.9 | 2.82 | **GPU** | **7.1×** |
| 64× units (64K units, 4K enc) | 22,280 | 318 | 35.5 | **GPU** | **9.0×** |

With micro-batching (batch=8, which amortizes kernel launch overhead):

| Scale | CPU (ms) | GPU (ms) | Speedup |
|---|---|---|---|
| 1× current | 0.26 | 0.11 | **2.4×** |
| 4× units | 1.84 | 0.28 | **6.5×** |
| 16× units | 19.9 | 2.26 | **8.8×** |

Assumptions: CPU DDR5 70 GB/s, VRAM 640 GB/s (RX 9070 XT), GPU kernel launch ~40 µs, 16 kernels per full perception pass.

Note: these estimates are conservative — burn's NdArray backend is not BLAS-optimized, so measured CPU times (kwta 2.1ms/ep at current scale) are already ~8× slower than the bandwidth model predicts. GPU wins are therefore likely larger than shown.

### Crossover point

At batch=1 with n_in=1024: GPU overtakes CPU at n_units ≈ **1,230** — just 20% above current 1,024. With micro-batching, GPU wins at current scale already.

### Human-brain-inspired scaling trajectory

| Horizon | N_UNITS | Regions | EPISODE_DIM | Weights | GPU win |
|---|---|---|---|---|---|
| Today | 1,024 | 5 | 640 | 19 MB | 0.4× (batch-1) |
| 1 year | 4,096 | 10 | 2,048 | ~500 MB | 4–6× |
| 2–3 years | 16,384 | 20 | 8,192 | ~8 GB | 7–9× |
| Long-term | 64K+ | 50 | 32K+ | ~100 GB+ | ~9× (memory-BW limited) |

The system should be designed for this trajectory. Locking in the ORT/MIGraphX path now avoids a much costlier migration later when training data and model capacity have grown.

---

## 3. Architecture boundary

The fundamental split is **frozen-during-wake vs. live-during-wake**.

### Exported to ORT (GPU inference graph)

These operations have stable shapes within a wake cycle (no weight updates, no shape changes) and are the computationally expensive parts at scale:

| Component | What it does | Weight shape |
|---|---|---|
| KWtaLayer.linear (×5) | Encoder embedding → region activations | [n_in, N_UNITS] |
| SaeDecoder.decoder (×5) | Sparse activations → reconstructed input | [N_UNITS, n_in] |
| cross_modal_proj | [speech\|env\|visual\|lang] concat → symbolic input | [CROSS_MODAL_DIM, 384] |
| BindingArea.region_i (×5) | N_UNITS → D_PROJ projection | [N_UNITS, D_PROJ] |
| BindingArea.w_q/w_k/w_v | Cross-attention QKV | [D_PROJ, D_PROJ] |
| BindingArea.temporal_gru | One GRU step for temporal context | GRU_HIDDEN=256 |
| BindingArea.temporal_proj | GRU hidden → EPISODE_DIM | [GRU_HIDDEN, EPISODE_DIM] |
| EmaOutputHeads (×5) | Episode → per-region reconstruction pred | [EPISODE_DIM, d_out] |

### Stays on CPU

These are either mutable-during-wake, part of the training graph, or cheap scalar operations:

| Component | Reason |
|---|---|
| Duty-cycle boosting + top-k selection | Mutable state per episode; O(n_units) sort — small op |
| Duty-cycle update | Stateful write, per episode |
| Backward pass + AdamW optimizer | All of training; burn provides autodiff |
| EMA weight update | Writes to EMA tensors after wake_reconstruct_step |
| Growth / prune operations | Shape-changing; triggers graph rebuild |
| Snapshot save / load | I/O, not on hot path |
| Telemetry emission | CPU-side always |
| Prior blend (blend_with_prior) | Scalar ops on Vec<f32>, negligible |

### Settling loop

Currently settle_steps=1 in the campaign (effectively disabled). When enabled, the loop is iterative with a data-dependent trip count. For the initial GPU implementation, keep the settle loop on CPU (re-call the ORT session for each settle iteration). Re-evaluate when settle_steps > 1 is actually used in practice.

---

## 4. ONNX graph design (two-graph split)

### Why two graphs, not one

A single fused graph was designed first and validated against numpy (7.93e-09 diff on CPU EP). MIGraphX crashed it with "invalid input for conversion to integer" (migraphx_execution_provider_utils.h:351, `ToInteger`). Binary search located the trigger: **≥4 ScatterElements nodes whose outputs are exposed as graph outputs** cause MIGraphX to fail at runtime on RDNA4/gfx1201. This is a MIGraphX bug (not an ONNX spec issue — the same ops pass in isolation and in graphs with ≤3 such outputs).

The two-graph split eliminates all ScatterElements from both graphs. Topk and masking live on CPU between the two calls. Neither graph hits the bug.

**Host↔device cost**: 5×[1, 1024] read + 5×[1, 1024] write = ~40 KB per episode. Negligible vs. the weight-read cost at any scale.

### Graph 1: Encoder (`perception_encoder.onnx`)

```
Inputs:
  speech_enc   f32[1, 384]
  env_enc      f32[1, 512]
  visual_enc   f32[1, 384]
  language_enc f32[1, 896]

Internal:
  cross_cat    = concat(speech, env, visual, language)   [1, 2176]
  symbolic_enc = cross_cat @ W_cross                     [1, 384]
  z_i          = enc_i @ W_enc_i                         [1, N_UNITS]

Outputs:
  z_0 .. z_4   f32[1, N_UNITS]   raw pre-topk activations
```

Ops: Concat, MatMul only. No risk on MIGraphX.

### CPU step (between Graph 1 and Graph 2)

```
for i in 0..5:
    boosted = z[i] + BOOST * (target_dc - dc[i])
    topk_idx = argpartition(boosted, -K_ACTIVE)[-K_ACTIVE:]
    mask[topk_idx] = 1.0; sparse_z[i] = z[i] * mask
    dc[i] *= (1 - ALPHA); dc[i][topk_idx] += ALPHA
```

O(5 × N_UNITS) ≈ 0.1ms at N_UNITS=1024.

### Graph 2: Binding (`perception_binding.onnx`)

```
Inputs:
  sp_0 .. sp_4   f32[1, N_UNITS]    sparse activations (after CPU step)
  prev_episode   f32[1, EPISODE_DIM]
  gru_hidden     f32[1, 1, GRU_HIDDEN]

Internal:
  proj_i         = sp_i @ W_region_i    [1, D_PROJ]
  x              = concat([proj_0..4])  [5, D_PROJ]
  gru_hidden_out = GRU(prev_episode, W_gru, R_gru; init=gru_hidden)
  ctx            = reshape(gru_hidden_out, [1, GH]) @ W_temporal_proj  [1, EPISODE_DIM]
  ctx_tokens     = reshape(ctx, [5, D_PROJ])
  x_ext          = concat([x, ctx_tokens])            [10, D_PROJ]
  Q = x @ W_q,   K = x_ext @ W_k,   V = x_ext @ W_v
  multi-head scaled-dot-product (4 heads, seq=5, kv_seq=10)
  episode        = reshape(attn_out, [1, EPISODE_DIM])

Outputs:
  episode        f32[1, EPISODE_DIM]
  gru_hidden_out f32[1, 1, GRU_HIDDEN]
```

Ops: MatMul, GRU, Concat, Reshape, Transpose, Softmax, Mul. All confirmed working on MIGraphX.

### MIGraphX operator coverage (empirically verified)

| Op | Status | How tested |
|---|---|---|
| MatMul | ✓ | Probes + full graph |
| GRU (linear_before_reset=1) | ✓ 1.79e-07 diff | `probe.py` + full graph |
| Softmax + MatMul (cross-attention) | ✓ 1.86e-08 diff | `probe.py` + full graph |
| TopK | ✓ correct set, different order | `probe.py` |
| ScatterElements → internal node | ✓ | `probe.py` |
| ScatterElements → graph output ≥4 | ✗ MIGraphX bug | `probe.py` threshold test |
| INT64 graph outputs (topk indices) | ✗ MIGraphX bug | `probe.py` |

**Hard constraint**: keep all topk/masking/ScatterElements on CPU. Never expose them as ONNX graph outputs.

### N_UNITS is static per export

Shape changes from `grow_region` / `prune_region` invalidate both graphs. Rebuild at the next sleep→wake boundary.

---

## 5. Weight sync protocol

### Principle

Burn is the **source of truth** for all weights. The ORT graph is a **derived read-only copy** valid for one wake period. No weights flow GPU→burn; the sync is strictly burn→GPU.

### Sync points

| Event | Action |
|---|---|
| Sleep → wake boundary | Export current burn weights → rebuild ORT session |
| grow_region or prune_region called | Mark graph as stale; rebuild at sleep→wake |
| EMA head update (every ~N episodes) | Update EMA initializers in graph (see §5.3) |

### Export mechanism

Burn's `BinFileRecorder` serializes weights to disk. At sync time:

1. Extract all weight tensors from burn model as `Vec<f32>` (already done by `save_weights`)
2. Load the **static ONNX graph skeleton** (graph structure without initializer data) from a template file
3. Patch the ONNX protobuf: write extracted weight blobs into the graph's initializer list
4. Write patched ONNX to a tempfile
5. Create new ORT session from tempfile with MIGraphX EP

The ONNX skeleton is generated once by a Python script (`scripts/export_perception_skeleton.py`) that constructs the graph topology from scratch using the `onnx` Python library. The skeleton has correctly typed but zeroed initializers. Rust patches them at runtime using the ONNX protobuf schema (via `prost` + the `.proto` files from the ONNX spec, already in tree as part of ORT).

### EMA head sync

EMA weights are updated on every `wake_reconstruct_step` (~655 times per 3276-episode cycle). Options:

- **Option A (initial):** keep EMA heads off-graph; run inference via the existing flat `Vec<f32>` matmul. The ONNX graph outputs the raw episode vector; EMA predict stays on CPU. Cost at current scale: ~0.1ms/episode (Perf G: "within noise").
- **Option B (when N_UNITS ≥ 4096):** include EMA heads as dynamic inputs to the graph, re-bound after each update. This eliminates the CPU readback of the episode vector for EMA inference.

Initial implementation uses Option A. Graduate to Option B when EMA weights exceed ~50 MB.

### Rebuild cost

At current scale, ONNX init is empirically ~0.5–2s (estimated from ORT session startup). Amortized over 602s wake periods this is <0.3%. At larger scales the graph is larger but the wake period is proportionally longer. Rebuild is triggered at most once per sleep cycle; it is acceptable latency.

---

## 6. Duty-cycle management

The duty cycle is a per-unit float that boosted activation scores (`boosted_z = z + BOOST_STRENGTH * (target_dc - dc)`). It is updated every episode:

```
for dc in duty_cycle:  dc *= (1 - DUTY_CYCLE_ALPHA)   -- ALPHA = 0.001
for i in top_k_indices: duty_cycle[i] += DUTY_CYCLE_ALPHA
```

This update is cheap (O(N_UNITS) on CPU, ~4 KB write at current scale) and has no autodiff dependency. It runs entirely on CPU in the split between Graph 1 and Graph 2:

1. Graph 1 returns z[i] (raw FLOAT, no topk performed inside the graph)
2. CPU selects top-K from boosted z, builds sparse mask, updates dc
3. Graph 2 receives sparse_z as input

This design avoids all MIGraphX issues (no INT64 outputs, no ScatterElements in graph outputs). The CPU step adds ~0.1ms at N_UNITS=1024 — negligible.

---

## 7. Micro-batching

Micro-batching is the mechanism that makes GPU win even at current N_UNITS=1024 (GPU speedup 2.4× at batch=8). It requires buffering encoder outputs before running perception.

### Current flow (batch-1, sequential)

```
for each video frame:
    speech_enc = Whisper(audio_chunk)   \
    env_enc    = CLAP(audio_chunk)       | concurrent GPU threads
    visual_enc = DINOv2(frame)          /
    -> ingest_speech → ingest_env → ingest_visual → bind
    -> one episode per frame
```

### Micro-batched flow (batch=B_MICRO)

```
accumulate B_MICRO frames of (speech_enc, env_enc, visual_enc, lang_enc)
-> single ORT call: perception_graph(batch=B_MICRO inputs)
-> B_MICRO episodes in one pass
```

### Implications

- **Latency tradeoff**: at B_MICRO=8 and 25fps video, perception latency increases by 8×(1/25s) = 320ms. This is acceptable for a system that already has >1s end-to-end latency.
- **GRU temporal state**: the GRU processes one frame at a time sequentially within a batch (frames are ordered). The GRU hidden state update across frames within a batch is still sequential — export as ONNX sequence. The cached `gru_hidden` across batches is the hidden state after the last frame of the previous batch.
- **Duty-cycle update**: applied per-sample in order, same as single-frame case.
- **Working memory**: the WM deque update happens after all B_MICRO episodes are produced, appending them in order. The `prev_episode` GRU input for frame i+1 within a batch is the episode from frame i — included in the graph as an internal recurrence.

B_MICRO is a runtime parameter (default 1, set via `--micro-batch-size`). Setting it to 1 reproduces current behavior exactly.

---

## 8. GPU contention resolution

The previous wgpu attempt failed because wgpu and MIGraphX are separate device owners — both try to claim the AMD GPU exclusively. The ORT solution avoids this entirely:

- Both encoders and perception use ORT sessions
- All sessions share the same ORT runtime instance (`ort::init_from(...)`)
- MIGraphX is the execution provider for all sessions
- ORT internally manages GPU resource scheduling within a single runtime

This is the key structural win: the GPU contention disappears because there is only one GPU framework in the process. The existing `migraphx_session()` function in `main.rs` is already the right entry point; perception ONNX sessions would be built through the same path.

Encoder sessions and perception sessions will share the GPU sequentially (ORT queues work). At current scale this is fine — the serialization cost is negligible. At larger scales, ORT's parallel execution mode can be used if needed.

---

## 9. Growth and re-export

`grow_region(region, n_new)` changes `KWtaLayer.n_units`, which changes the shape of:
- `W_region_i` in binding area: [old_n_units, D_PROJ] → [new_n_units, D_PROJ]
- `W_enc_i` in k-WTA layer: [n_in, old_n_units] → [n_in, new_n_units]
- `W_dec_i` in SAE decoder: [old_n_units, n_in] → [new_n_units, n_in]
- All `dc_i` state vectors
- All `topk_indices_i` outputs

Shape changes invalidate the ONNX graph. The protocol:

1. `grow_region` / `prune_region` sets a `graph_stale: bool` flag in `PerceptionProcessor`
2. At the sleep→wake transition, if `graph_stale`, the graph is rebuilt with the new shapes
3. The Python skeleton generator is parameterized by `N_UNITS` — it must be re-run, or the Rust export code must handle variable shapes (the latter is preferable for a self-contained system)

D_PROJ also changes when N_UNITS grows (currently fixed at 128; at scale should be N_UNITS/8 or configurable). D_PROJ changes also require a graph rebuild.

### Protobuf patching vs full skeleton regeneration

Two strategies:
- **Patch only weights** (fast): assume graph topology is fixed, only update initializer blobs. Requires the graph skeleton was exported with the correct shapes. Works only when no growth has occurred.
- **Full skeleton regeneration** (robust): always regenerate from the current model shapes. Requires Python script to be runnable at sleep→wake boundary, or a pure-Rust ONNX graph builder.

Recommendation: **pure-Rust ONNX builder** — a helper module `crates/somnus-perception/src/onnx_export.rs` that constructs the ONNX protobuf programmatically using `prost` + the ONNX proto schema. This makes the system fully self-contained with no Python runtime dependency post-training. The schema is small and stable.

---

## 10. Training path: no change

All training (NREM replay, wake_reconstruct, fused_replay_wake) stays entirely on CPU with burn. The GPU path is **inference-only during wake**. The interface between the two is:

```
wake (GPU inference, ORT):  run_perception_graph(encodings) → (episodes, gru_state, dc_updates, prior_preds)
sleep (CPU training, burn): fused_replay_wake_step_batch(regional, episodes) → (sae_loss, wake_loss)
sleep→wake sync:            export_weights(burn_model) → rebuild_ort_session()
```

No gradient information flows through the ORT session. No burn tensor leaves the CPU.

---

## 11. Phased implementation

### Phase 0 — validate graph equivalence (no behavior change)

**Status: partially complete (2026-06-02).** Two-graph design validated numpy↔ONNX↔MIGraphX. MIGraphX bug isolated and worked around. Weight exporter implemented.

Remaining gate: **burn ↔ ONNX episode match on one live episode** (with non-uniform dc to exercise the boost path). The numpy↔ONNX check validated graph structure; burn↔ONNX closes the GRU gate-order and Linear transpose convention risks against real trained weights.

**Deliverables:** `scripts/export_perception_graph.py` (two-graph builder + numpy validator), `PerceptionProcessor::export_weights_for_onnx()`, `experiments/migraphx-op-probe/probe.py`.

**Kill criterion:** if burn↔ONNX episode MSE > 1e-3 after correcting GRU conventions, fall back to CPU-ORT EP for the binding graph (no perf win, but validates rest of pipeline).

### Phase 1 — wake inference on GPU

Replace burn inference in the wake path with the ORT graph. Keep burn inference for settle (settle_steps=1, rarely fires) and all sleep-time passes. Add weight sync at sleep→wake boundary.

**Deliverable:** modified `PerceptionProcessor` with `OrtInferenceSession` field; `sync_ort_weights()` method; updated `ingest_*` functions.

### Phase 2 — micro-batching

Add `--micro-batch-size N` flag. Default 1 (identical to Phase 1). When N>1, buffer encoder outputs and call the ORT graph in batches.

**Deliverable:** encoder output buffer, batched ORT call, B_MICRO sequential duty-cycle updates, GRU intra-batch recurrence.

### Phase 3 — EMA heads in graph (when N_UNITS ≥ 4096)

Graduate EMA output heads into the ORT graph as dynamic initializer inputs, updated at each `wake_reconstruct_step`. Remove the CPU flat-matmul path.

**Deliverable:** modified graph export, `update_ema_in_graph()` method.

### Phase 4 — Rust ONNX builder

Replace Python skeleton generator with pure-Rust `onnx_export.rs` using `prost`. Makes the system self-contained; required before growth is frequent enough that re-running a Python script at every sleep→wake becomes awkward.

---

## 12. Kill criteria

This design should be abandoned or revised if:

1. MIGraphX produces incorrect results for any of: GRU op, TopK, multi-head attention reshapes — and no workaround exists. (Mitigation: fall back to CPU-ORT EP for validation; accept no perf win until MIGraphX fixes the op.)
2. ORT session initialization time at grown N_UNITS exceeds 10% of the wake period budget. (Mitigation: cache the session across sleep cycles if shapes haven't changed.)
3. The encoder sessions and perception session contend on MIGraphX in a way that ORT's internal scheduler cannot resolve. (Mitigation: run perception on CPU-ORT EP; accept the bandwidth cost but gain the unified-framework benefit.)
4. The research direction shifts away from region-scaling before Phase 1 is complete. At current N_UNITS=1024, batch-1 CPU is faster — the unification is only worth the implementation cost if scaling is actually on the roadmap.

---

## 13. Open questions

- **D_PROJ scaling policy**: currently fixed at 128. Should scale with N_UNITS to maintain representational capacity. Decide formula (N_UNITS/8? fixed schedule?) before Phase 1 — it affects EPISODE_DIM.
- **EPISODE_DIM when region count grows**: currently 640 = 5 × 128. If we add regions, EPISODE_DIM must grow. The memory snapshot format will break. Design a migration path.
- **Settle loop at scale**: currently settle_steps=1. If settle is re-enabled, iterative ORT calls for settle are expensive. Consider exporting the settle loop as a scan/loop ONNX op, or keeping settle on CPU with weight copies.
- **ORT parallel execution**: ORT has an inter-op parallelism setting. At larger N, it may be worth running encoder sessions and perception in separate ORT streams. Profile before adding complexity.
