# Session state — somnus

> **Last updated:** 2026-06-02
> **Last commit on main:** f4813c5 — GPU unification Phase 0
> **Branch:** `main` (not yet pushed to Codeberg)

## Status: GPU unification Phase 0 partially complete. Two-graph ONNX design validated on MIGraphX. Burn↔ONNX numeric gate still to close.

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| Fresh campaign | campaign_20260602_1921 | 3 sessions × 3 cycles, 29K warm records, s3c3 loss=0.136, synthetics=8/cycle |
| GPU unification spec | f0b7921 | ADR 0023 + ai-docs/research/gpu-unification-design.md |
| MIGraphX op probes | f4813c5 | GRU/TopK/cross-attention all confirmed on RDNA4/gfx1201 |
| MIGraphX bug isolation | f4813c5 | ScatterElements→graph-output ≥4 crashes MIGraphX (ToInteger bug) |
| Two-graph ONNX design | f4813c5 | encoder graph + binding graph, both validated numpy↔ONNX↔MIGraphX |
| Weight exporter (Rust) | f4813c5 | PerceptionProcessor::export_weights_for_onnx() |

---

## GPU unification status

**Phase 0 gate: burn ↔ ONNX numeric match on one live episode**

What's done:
- `experiments/migraphx-op-probe/probe.py` — all ops confirmed
- `scripts/export_perception_graph.py` — builds `perception_encoder.onnx` + `perception_binding.onnx`
- `PerceptionProcessor::export_weights_for_onnx()` — dumps weights to `.f32` bundle
- numpy↔CPU EP↔MIGraphX validated at <5e-9 with non-uniform dc

What remains for Phase 0 completion:
1. Export actual burn weights via `export_weights_for_onnx()`
2. Build ONNX graphs from those weights via Python script
3. Run one burn inference episode + same episode through ONNX
4. Assert episode diff < 1e-4

Likely risks for burn↔ONNX match:
- GRU gate-order (burn: update/reset/new = z/r/h — same as ONNX, but needs runtime confirmation)
- Linear weight transpose (burn stores [in, out]; ONNX GRU W expects [H, I] per gate — Python script transposes)
- Non-uniform dc: boost term is `z + BOOST*(TARGET_DC - dc)` — make sure ONNX argpartition matches burn's select_top_k with same boosted scores

After Phase 0:
- Phase 1: replace burn wake inference with ORT session calls in PerceptionProcessor
- Phase 2: micro-batching (--micro-batch-size flag)

---

## MIGraphX constraint (hard)

**Never expose ScatterElements outputs as ONNX graph outputs.** MIGraphX on RDNA4/gfx1201 crashes at runtime with "invalid input for conversion to integer" when ≥4 such outputs are present. Root cause unknown; workaround is two-graph split with topk/masking on CPU.

Also: **INT64 graph outputs trigger the same bug**. Keep topk indices off the output list.

---

## Campaign results (memory/campaign_20260602_1921)

3 sessions × 3 cycles, 29K warm records:
- s3c3 loss: 0.136 (healthy)
- synthetics_produced=8 every cycle (no NREM salience interrupts)
- bind_ms flat 2.1-2.5ms/ep (autodiff refresh holding)
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
