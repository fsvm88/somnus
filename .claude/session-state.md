# Session state — somnus

> **Last updated:** 2026-06-07
> **Last commit on main:** 321cb8e — GainWorker: offload Qwen gain ORT call to dedicated background thread
> **Branch:** `main` (not yet pushed)

## Status: GainWorker COMPLETE. Phase 1.5 encoder ORT live (unvalidated). Next: benchmark + duty-cycle validation.

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| train_campaign.sh ONNX defaults | 590f7e6 | ONNX_MODELS_DIR / ONNX_SCRIPT env vars; no more EXTRA_ARGS required |
| Phase 1.5: encoder ORT | 590f7e6 | All 5 encoder matmuls + cross-modal proj move to GPU; topk+dc stays in Rust |
| GainWorker (Phase 2 redirect) | 321cb8e | Qwen gain ORT call moved to dedicated OS thread; ~54ms stall eliminated from tokio loop |

---

## GainWorker design (321cb8e)

- `ActionDecoder.gain_session` is `Option<Session>`; `create_gain_worker()` takes it via `Option::take()`.
- `GainWorker` holds one `std::sync::mpsc::SyncSender<GainMsg>` (cap=2) + `Receiver<[f32; 5]>` (cap=2).
- Audio/video arms: `gw.request_gain(ep.clone())` — fire-and-forget, drops if channel full.
- Language arm: same, via `LangIngestResult.last_episode` field.
- Main loop top: `gw.try_recv_gain()` → `perception.set_region_gain(g)` — non-blocking poll.
- After NREM save_weights: `gw.update_weights(dec.proj_weights_flat())` — keeps worker's projection current.
- `gain_interval` default raised to 20 (from 10); `gain_update_total_ms` removed from metrics (field kept as literal 0).
- `update_gain` and `gain` field kept on `ActionDecoder` as `#[allow(dead_code)]` fallback.

---

## GPU unification status: Phase 1.5 COMPLETE (unvalidated)

**Phase 0 (burn↔ONNX gate):** All checks pass at <2e-08.

**Phase 1 (ORT binding session live):** 2.4–2.8× speedup confirmed (campaign_20260602_2308).

**Phase 1.5 (ORT encoder session live — commit 590f7e6):**
- `enable_ort_inference` now auto-attaches `perception_encoder.onnx` if it exists in
  the same directory as `perception_binding.onnx`.
- In `try_bind`: when encoder ORT is active, runs all 5 linears + cross-modal proj on GPU
  → 5 z vectors → Rust topk+dc-update via `KWtaLayer::topk_from_z` → ORT binding.
- `compute_symbolic_cross_modal` uses `forward_no_dc` (no burn dc update) when encoder ORT
  active — duty-cycle update happens once in try_bind via topk_from_z.
- `refresh_language_cache` skipped when encoder ORT active (language z from ORT in try_bind).
- Fallback to burn activations on ORT encoder error (WARN logged).

**Duty-cycle cadence change (validate before relying on ORT-trained weights):**
- Burn path: each region's dc updates when its ingest_* fires (~2 updates/binding: triggering + symbolic).
- ORT encoder path: all 5 regions dc-update once per binding event.
- Behavioral effect unknown — run burn-only vs. ORT campaign, same seed+clip, compare
  loss trajectory and per-region dc distributions.

**Campaign usage with ORT (Phase 1.5):**
```bash
SESSIONS=3 CYCLES=3 CAPACITY=4096 \
MEMORY_DIR=memory/campaign_$(date +%Y%m%d_%H%M) \
ONNX_MODELS_DIR=memory/onnx_models \
ONNX_SCRIPT=scripts/export_perception_graph.py \
EXTRA_ARGS="--lang-oversample 9 --asr-min-chars 5 --max-warm-records 200000" \
scripts/train_campaign.sh
```

**Phase 1.5 not yet benchmarked.** Expected: encoder matmuls (~2.1+3.1ms/ep) move to GPU;
total bind_ms should drop further from the 0.78–0.98ms/ep Phase 1 baseline.

---

## Next steps (priority order)

1. **Benchmark Phase 1.5 + GainWorker**: run a campaign with ORT enabled, compare bind_ms and
   wall-time per episode vs. the Phase 1 baseline (0.78–0.98ms/ep). Confirm GainWorker keeps
   the tokio loop event latency flat even during gain updates.
2. **Duty-cycle validation**: run burn-only vs. ORT encoder campaign (same seed+clip), compare
   loss trajectory and per-region dc distributions. Gate any weight-sharing on this passing.
3. **Identify next bottleneck** after GainWorker: check `perc_prior_us` and `perc_bind_us`
   from the metrics — if those dominate, the GRU binding (0.78ms sequential) is the wall.

---

## MIGraphX constraint (hard)

**Never expose ScatterElements outputs as ONNX graph outputs.** MIGraphX on RDNA4/gfx1201 crashes at runtime with "invalid input for conversion to integer" when ≥4 such outputs are present. Root cause unknown; workaround is two-graph split with topk/masking on CPU.

Also: **INT64 graph outputs trigger the same bug**. Keep topk indices off the output list.

---

## Campaign results (memory/campaign_20260602_1921)

3 sessions × 3 cycles, 29K warm records:
- s3c3 loss: 0.136 (healthy)
- synthetics_produced=8 every cycle
- bind_ms flat 2.1-2.5ms/ep (burn, no ORT)
- All nrem_exit_reason=traces_exhausted (clean run)

---

## Key facts (unchanged)

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--working-memory-size 8` hardcoded in `train_campaign.sh`
- `--asr-min-chars 5` for noisy-audio clips
- `--max-warm-records 200000` recommended for long runs
- Wake schedule trigger at 602s/cycle
- N_UNITS=1024, K_ACTIVE=20, D_PROJ=128, EPISODE_DIM=640
- GRU_INPUT=640, GRU_HIDDEN=256
- CROSS_MODAL_DIM=2176
- burn GRU: reset_after=True (default) → ONNX linear_before_reset=1
- burn Linear weight shape: [n_in, n_out] (transpose of ONNX/PyTorch convention)
- DUTY_CYCLE_ALPHA=0.001, BOOST_STRENGTH=1.0, TARGET_DC=20/1024≈0.0195
