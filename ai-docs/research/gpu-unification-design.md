# GPU Unification Design

> **Contents:** Full design for moving perception inference (k-WTA, cross_modal_proj, BindingArea GRU) from burn/CPU to ORT+MIGraphX/GPU. Covers scaling analysis, two-graph ONNX split (encoder + binding), duty-cycle management, weight sync protocol, MIGraphX operator constraints, phased implementation. Phase 0+1 complete; Phases 2-4 prospective.
> **When to read:** Understanding the ORT two-graph architecture; debugging MIGraphX issues (ScatterElements/INT64 constraint); planning micro-batching or a Rust ONNX builder.
> **Status:** Phase 0 (gate validation) and Phase 1 (wake inference on GPU) implemented as of 2026-06-07. Phases 2-4 (micro-batching, EMA in graph, Rust ONNX builder) prospective.
> **Date:** 2026-06-02 (design); 2026-06-07 (Phase 1 complete)
> **Companion ADR:** decisions/0023-gpu-unification.md
> **Superseding change (2026-08-12):** §4's Graph 2 description below is stale as written — see
> the note inline. `decisions/0040-sparse-gather-binding-projection.md` rewrote the binding graph's
> region-projection step from a dense matmul to a Gather over active indices only; not part of the
> original phased plan below, but a real change to the deployed graph. `decisions/0038-sparse-encoder-scoring.md`
> (`proposed`, not started) is the analogous encoder-side direction this design's Phase 4 didn't
> anticipate.

---

## 1. Problem and motivation

Two GPU frameworks currently run in the same process: **ORT+MIGraphX** for frozen encoders (Whisper-tiny, CLAP-HTSAT, DINOv2-S, Qwen2.5), and **burn NdArray (CPU)** for all perception inference/training (k-WTA, binding, SAE, output heads). Workable at current scale (1K units/region, batch-1 — perception ~35ms/episode hides behind the ~96ms GPU encode path), but two problems compound with scale:

1. **CPU memory-bandwidth wall.** ~19MB of weights read per episode at current scale. DDR5 ~70GB/s vs. RX 9070 XT VRAM ~640GB/s — 9× gap. Already memory-bandwidth-bound (Perf G).
2. **wgpu conflict (5dbf9c9).** A prior attempt to run burn on GPU failed — MIGraphX had already claimed the AMD device; two GPU frameworks can't coexist on it. Only resolution: consolidate on ORT/MIGraphX (the frozen encoders can't move off it).

**Design goal:** all GPU-bound compute runs through ORT/MIGraphX; burn stays CPU-only for training; weights sync at sleep→wake boundaries.

---

## 2. Scaling analysis

**Memory-bandwidth model** (dominant cost = reading weight matrices):

| Scale | Wt (MB) | CPU (ms) | GPU ms | Winner | Speedup |
|---|---|---|---|---|---|
| 1× current (1K units, 384 enc) | 19 | 0.26 | 0.67 | CPU | 0.4× |
| 4× units (4K, 512 enc) | 129 | 1.84 | 0.84 | **GPU** | **2.2×** |
| 16× units (16K, 1K enc) | 1,393 | 19.9 | 2.82 | **GPU** | **7.1×** |
| 64× units (64K, 4K enc) | 22,280 | 318 | 35.5 | **GPU** | **9.0×** |

With micro-batching (batch=8, amortizes kernel launch):

| Scale | CPU (ms) | GPU (ms) | Speedup |
|---|---|---|---|
| 1× current | 0.26 | 0.11 | **2.4×** |
| 4× units | 1.84 | 0.28 | **6.5×** |
| 16× units | 19.9 | 2.26 | **8.8×** |

Assumptions: DDR5 70GB/s, VRAM 640GB/s, kernel launch ~40µs, 16 kernels/pass. Conservative — burn's NdArray backend isn't BLAS-optimized, so measured CPU times (kwta 2.1ms/ep) already run ~8× slower than the model predicts, meaning real GPU wins are likely larger.

**Crossover:** batch=1, n_in=1024 → GPU wins at n_units ≈ **1,230** (20% above current 1,024). With micro-batching, GPU already wins at current scale.

**Scaling trajectory:**

| Horizon | N_UNITS | Regions | EPISODE_DIM | Weights | GPU win |
|---|---|---|---|---|---|
| Today | 1,024 | 5 | 640 | 19MB | 0.4× (batch-1) |
| 1 year | 4,096 | 10 | 2,048 | ~500MB | 4-6× |
| 2-3 years | 16,384 | 20 | 8,192 | ~8GB | 7-9× |
| Long-term | 64K+ | 50 | 32K+ | ~100GB+ | ~9× (BW-limited) |

Locking in ORT/MIGraphX now avoids a costlier migration once training data and capacity have grown.

---

## 3. Architecture boundary

Split is **frozen-during-wake vs. live-during-wake**.

**Exported to ORT** (stable shapes within a wake cycle, the expensive part at scale): `KWtaLayer.linear`×5 (`[n_in,N_UNITS]`), `SaeDecoder.decoder`×5 (`[N_UNITS,n_in]`), `cross_modal_proj` (`[CROSS_MODAL_DIM,384]`), `BindingArea.region_i`×5 (`[N_UNITS,D_PROJ]`), `BindingArea.w_q/w_k/w_v` (`[D_PROJ,D_PROJ]`), `BindingArea.temporal_gru` (GRU_HIDDEN=256), `BindingArea.temporal_proj` (`[GRU_HIDDEN,EPISODE_DIM]`), `EmaOutputHeads`×5 (`[EPISODE_DIM,d_out]`).

**Stays on CPU:** duty-cycle boosting+top-k (mutable per-episode state, O(n_units) sort — small), duty-cycle update (stateful write), backward+AdamW (all training, burn autodiff), EMA weight update, growth/prune (shape-changing, triggers rebuild), snapshot save/load (I/O), telemetry, prior blend (scalar ops on `Vec<f32>`, negligible).

**Settling loop:** currently `settle_steps=1` (effectively disabled). Data-dependent trip count when enabled — keep on CPU initially (re-call the ORT session per iteration); revisit if `settle_steps>1` sees real use.

---

## 4. ONNX graph design (two-graph split)

**Why two graphs:** a single fused graph validated against numpy (7.93e-09 diff, CPU EP) but crashed MIGraphX ("invalid input for conversion to integer," `ToInteger`). Binary search found the trigger: **≥4 `ScatterElements` nodes exposed as graph outputs** crash on RDNA4/gfx1201 — a MIGraphX bug (same ops pass in isolation and with ≤3 such outputs). The two-graph split eliminates all `ScatterElements` from both graphs (topk/masking live on CPU between calls). Host↔device cost: 5×[1,1024] read + write ≈ 40KB/episode, negligible vs. weight-read cost at any scale.

**Graph 1 — Encoder** (`perception_encoder.onnx`): inputs `speech_enc[1,384]`/`env_enc[1,512]`/`visual_enc[1,384]`/`language_enc[1,896]`; internal `cross_cat=concat(...)[1,2176]` → `symbolic_enc=cross_cat@W_cross[1,384]`, `z_i=enc_i@W_enc_i[1,N_UNITS]`; outputs `z_0..z_4[1,N_UNITS]` (raw pre-topk). Ops: Concat, MatMul only — no MIGraphX risk.

**CPU step (between graphs), O(5×N_UNITS)≈0.1ms at N_UNITS=1024:**
```
for i in 0..5:
    boosted = z[i] + BOOST * (target_dc - dc[i])
    topk_idx = argpartition(boosted, -K_ACTIVE)[-K_ACTIVE:]
    mask[topk_idx] = 1.0; sparse_z[i] = z[i] * mask
    dc[i] *= (1 - ALPHA); dc[i][topk_idx] += ALPHA
```

**Graph 2 — Binding** (`perception_binding.onnx`): originally designed with dense `sp_0..sp_4[1,N_UNITS]` inputs (sparse-valued but full-length) feeding `proj_i=sp_i@W_region_i[1,D_PROJ]`. **Superseded 2026-08-12 (ADR 0040):** the region-projection step was found to be 75-76% of this graph's per-call cost at campaign scale — a dense matmul reading the full `W_region_i` regardless of input sparsity, memory-bandwidth-bound. Current inputs are `active_idx_0..4[k_active_i]` (int64) + `active_val_0..4[1,k_active_i]` (float) pairs instead, with `proj_i = Gather(W_region_i, active_idx_i, axis=0)` (`[k,D_PROJ]`) followed by a `[1,k]@[k,D_PROJ]` weighted-sum matmul — numerically identical to the dense form (every non-gathered row contributes exactly zero), 54-59% cheaper. `k_active_i` is baked into the graph's static shape at export, sourced from `weights_manifest.json` (never re-derived from `n_units` in Python — the §17/§22/§24 drift class this design didn't anticipate). From this point on, unchanged: `x=concat(proj_0..4)[5,D_PROJ]`, GRU step → `ctx` via `W_temporal_proj[1,EPISODE_DIM]` → `ctx_tokens[5,D_PROJ]` → `x_ext=concat([x,ctx_tokens])[10,D_PROJ]` → 4-head scaled-dot-product attention (Q from `x`, K/V from `x_ext`) → `episode[1,EPISODE_DIM]`; outputs `episode`, `gru_hidden_out`. Ops: MatMul, Gather, GRU, Concat, Reshape, Transpose, Softmax, Mul — all confirmed on MIGraphX (the Gather rewrite also surfaced and worked around a `simplify_reshapes` compiler bug — see ADR 0040).

**MIGraphX operator coverage (empirically verified):** MatMul ✓; GRU (linear_before_reset=1) ✓ 1.79e-07 diff; Softmax+MatMul (cross-attention) ✓ 1.86e-08 diff; TopK ✓ correct set, different order; ScatterElements→internal node ✓; ScatterElements→graph output ≥4 ✗ (MIGraphX bug); INT64 graph outputs (topk indices) ✗ (MIGraphX bug). **Hard constraint: keep all topk/masking/ScatterElements on CPU, never expose as ONNX graph outputs.**

**N_UNITS is static per export** — `grow_region`/`prune_region` invalidate both graphs; rebuild at the next sleep→wake boundary.

---

## 5. Weight sync protocol

**Principle:** burn is the source of truth for all weights; the ORT graph is a derived read-only copy valid for one wake period. Sync is strictly burn→GPU, never the reverse.

**Sync points:** sleep→wake boundary (export burn weights, rebuild ORT session); `grow_region`/`prune_region` called (mark graph stale, rebuild at next sleep→wake); EMA head update (update EMA initializers, see below).

**Export mechanism:** extract weight tensors from burn as `Vec<f32>` (already done by `save_weights`) → load the static ONNX graph skeleton (topology only, zeroed initializers, generated once by `scripts/export_perception_skeleton.py` via the Python `onnx` library) → patch the protobuf's initializer list with the extracted blobs (Rust, via `prost` + ONNX proto schema already in tree with ORT) → write to tempfile → create a new ORT session with the MIGraphX EP.

**EMA head sync** (updated ~655×/3276-episode cycle): **Option A (initial)** — keep EMA heads off-graph, inference via the existing flat `Vec<f32>` matmul; ONNX outputs the raw episode, EMA predict stays on CPU (~0.1ms/ep, within noise). **Option B (N_UNITS≥4096)** — include EMA heads as dynamic graph inputs, re-bound after each update, eliminating the CPU episode-vector readback. Start with A, graduate to B once EMA weights exceed ~50MB.

**Rebuild cost:** ~0.5-2s at current scale (ORT session startup), <0.3% amortized over a 602s wake period. Larger scales get proportionally longer wake periods too. At most once per sleep cycle — acceptable.

---

## 6. Duty-cycle management

Per-unit float boosting activation scores: `boosted_z = z + BOOST_STRENGTH*(target_dc - dc)`, updated every episode (`dc *= (1-ALPHA)` then `dc[top_k] += ALPHA`, ALPHA=0.001). Cheap (O(N_UNITS), ~4KB write at current scale), no autodiff dependency — runs entirely on CPU between Graph 1 and Graph 2: Graph 1 returns raw `z[i]` (no topk inside), CPU selects top-K from boosted z / builds the sparse mask / updates dc, Graph 2 receives `sparse_z`. Avoids all MIGraphX issues (no INT64 outputs, no ScatterElements graph outputs); the CPU step adds ~0.1ms at N_UNITS=1024.

---

## 7. Micro-batching

The mechanism that makes GPU win even at current N_UNITS=1024 (2.4× at batch=8) — requires buffering encoder outputs before running perception.

**Current (batch-1, sequential):** per frame, Whisper/CLAP/DINOv2 run concurrently on GPU threads → ingest each modality → bind → one episode/frame.

**Micro-batched (batch=B_MICRO):** accumulate B_MICRO frames of encoder outputs → one ORT call over the batch → B_MICRO episodes per pass.

**Implications:** latency tradeoff — at B_MICRO=8, 25fps, perception latency grows by 8×(1/25s)=320ms, acceptable given the system's existing >1s end-to-end latency. GRU processes frames sequentially within a batch (ordered) — exported as an ONNX sequence; cached `gru_hidden` across batches is the hidden state after the previous batch's last frame. Duty-cycle applies per-sample in order, same as single-frame. WM deque updates after all B_MICRO episodes are produced, in order; `prev_episode` for frame i+1 within a batch comes from frame i, included as an internal graph recurrence. `B_MICRO` is a runtime parameter (`--micro-batch-size`, default 1 = exact current behavior).

---

## 8. GPU contention resolution

The prior wgpu attempt failed because wgpu and MIGraphX are separate device owners, both trying to claim the GPU exclusively. ORT avoids this structurally: both encoders and perception use ORT sessions sharing one runtime instance (`ort::init_from(...)`), MIGraphX as the EP for all of them, ORT manages GPU scheduling internally — there's only one GPU framework in the process. The existing `migraphx_session()` in `main.rs` is already the right entry point for perception sessions too. Encoder and perception sessions share the GPU sequentially (ORT queues work) — fine at current scale; ORT's parallel-execution mode is available if larger scales need it.

---

## 9. Growth and re-export

`grow_region(region, n_new)` changes `KWtaLayer.n_units`, reshaping `W_region_i` (binding), `W_enc_i` (k-WTA), `W_dec_i` (SAE decoder), all `dc_i` state, all `topk_indices_i` outputs — invalidating the ONNX graph. Protocol: `grow_region`/`prune_region` sets `graph_stale: bool` on `PerceptionProcessor`; at the next sleep→wake transition, if stale, rebuild with new shapes (the Python skeleton generator is parameterized by `N_UNITS` and must be re-run, or the Rust export code handles variable shapes — preferable for self-containment). `D_PROJ` (fixed 128 currently; should scale with N_UNITS eventually, e.g. N_UNITS/8) also forces a rebuild when it changes.

**Patching strategy:** patch-only-weights (fast, assumes fixed topology, breaks after growth) vs. full skeleton regeneration (robust, needs Python runnable at sleep→wake, or a pure-Rust builder). **Recommendation: pure-Rust ONNX builder** (`crates/somnus-perception/src/onnx_export.rs`, `prost` + the ONNX proto schema) — makes the system self-contained with no post-training Python dependency; the schema is small and stable.

---

## 10. Training path: no change

All training (NREM replay, wake_reconstruct, fused_replay_wake) stays entirely CPU/burn. GPU is inference-only during wake:

```
wake (GPU inference, ORT):  run_perception_graph(encodings) → (episodes, gru_state, dc_updates, prior_preds)
sleep (CPU training, burn): fused_replay_wake_step_batch(regional, episodes) → (sae_loss, wake_loss)
sleep→wake sync:            export_weights(burn_model) → rebuild_ort_session()
```

No gradient flows through the ORT session; no burn tensor leaves the CPU.

---

## 11. Phased implementation

**Phase 0 — validate graph equivalence, no behavior change. Complete (2026-06-07).** Two-graph design validated numpy↔ONNX↔MIGraphX; the ScatterElements/INT64 bug isolated and worked around; weight exporter implemented; burn↔ONNX episode MSE <2e-8 on live episodes. Deliverables: `scripts/export_perception_graph.py`, `PerceptionProcessor::export_weights_for_onnx()`, `experiments/migraphx-op-probe/probe.py`.

**Phase 1 — wake inference on GPU. Complete (2026-06-07).** burn inference in the wake path replaced with the ORT graph; `settle_episode` also uses ORT encoder+binding (no burn kwta during settle); all 5 ingest paths skip burn forward when ORT is active; `cross_modal_proj` uses flat Vec matmul; weight sync at every sleep→wake boundary. **Benchmark** (campaign_20260607_185345): kwta=0.000ms/ep, sym=0.30ms/ep, bind=1.6-2.6ms/ep, total perception ~17-19ms/ep vs. 18ms burn-only baseline.

**Phase 2 — micro-batching.** `--micro-batch-size N` (default 1 = Phase 1 behavior). N>1 buffers encoder outputs, calls the graph in batches. Deliverable: encoder output buffer, batched ORT call, sequential intra-batch duty-cycle updates, GRU intra-batch recurrence.

**Phase 3 — EMA heads in graph (N_UNITS≥4096).** Graduate EMA output heads into the graph as dynamic initializer inputs, updated each `wake_reconstruct_step`; remove the CPU flat-matmul path. Deliverable: modified graph export, `update_ema_in_graph()`.

**Phase 4 — Rust ONNX builder.** Replace the Python skeleton generator with `onnx_export.rs` (`prost`) — self-contained, needed once growth is frequent enough that re-running Python at every sleep→wake becomes awkward.

---

## 12. Kill criteria

Abandon or revise if: (1) MIGraphX produces wrong results for GRU/TopK/multi-head-attention-reshapes with no workaround (mitigation: fall back to CPU-ORT EP, accept no perf win until fixed); (2) ORT session init at grown N_UNITS exceeds 10% of the wake budget (mitigation: cache the session across cycles if shapes are unchanged); (3) encoder and perception sessions contend on MIGraphX beyond what ORT's scheduler resolves (mitigation: run perception on CPU-ORT EP, accept the bandwidth cost, keep the unified-framework benefit); (4) the research direction moves away from region-scaling before Phase 1 completes (at N_UNITS=1024 batch-1, CPU is still faster — unification is only worth it if scaling is actually on the roadmap).

---

## 13. Open questions

- **D_PROJ scaling policy:** fixed at 128 — should scale with N_UNITS (formula TBD, e.g. N_UNITS/8) before Phase 1 grows it further; affects EPISODE_DIM. **Still unresolved as of 2026-08-15:** the 10x growth campaign (engineering-log.md §34) took N_UNITS 1,024→9,821 with D_PROJ still fixed at 128 (confirmed current) — this question was never revisited despite growth well past the original "before Phase 1 grows it further" framing. Similarly, §5's EMA-head-sync "Option B" trigger (N_UNITS≥4096) has been crossed by ~2.4× with EMA heads still off-graph (Option A) — not confirmed broken, just confirmed not acted on.
- **EPISODE_DIM when region count grows:** currently 640=5×128; adding regions grows it and breaks the memory snapshot format — needs a migration path.
- **Settle loop at scale:** iterative ORT calls are expensive if settle is re-enabled (`settle_steps>1`) — consider exporting as a scan/loop ONNX op, or keeping settle on CPU with weight copies.
- **ORT parallel execution:** inter-op parallelism setting — may be worth separate ORT streams for encoder vs. perception at larger N. Profile before adding the complexity.
