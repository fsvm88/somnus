# Session state — somnus

> **Last updated:** 2026-06-02
> **Last commit on main:** ca4ab59 — GPU unification Phase 0 complete
> **Branch:** `main` (not yet pushed to Codeberg)

## Status: GPU unification Phase 0 COMPLETE. burn↔ONNX gate closed. Ready for Phase 1 (ORT inference in wake path).

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

## GPU unification status: Phase 0 COMPLETE

**All burn↔ONNX checks pass (CPU EP and MIGraphX EP):**
- Encoder z vectors: exact match (<1.5e-06 on MIGraphX)
- Sparse_z sets: same top-K selected
- Episode (call 1, attention path): 5.59e-09 / 1.58e-08
- Episode (call 2, GRU with non-zero input): 3.73e-09 / 1.68e-08
- GRU hidden state: 6.52e-09 / 2.24e-08

Bug found and fixed: `cross_modal_proj` has `bias=true` — the bias wasn't being exported. Added `W_cross_bias` export and `Add` node in encoder ONNX graph.

**Validation toolchain:**
- `cargo run --release -p somnus-perception --bin onnx_validate_prepare -- <dir>` — exports weights + reference data
- `python3 experiments/onnx-validate/validate.py --data-dir <dir> --migraphx` — validates

**Phase 1: replace burn wake inference with ORT**

Concrete changes needed:
1. Add `OnnxPerceptionSession` struct in `crates/somnus-perception/src/onnx_session.rs`:
   - Holds two ORT sessions (encoder + binding)
   - `run(&mut self, feeds) -> (episode, gru_h_out)` method
   - `rebuild(weights_dir, onnx_dir)` — calls Python script + creates sessions
2. Add to `PerceptionProcessor`:
   - Optional `ort_session: Option<OnnxPerceptionSession>`
   - `enable_ort_inference(weights_dir, onnx_dir)` — exports weights, builds graphs, loads sessions
   - Modified `ingest_*` functions: if ort_session is Some, use ORT path; else burn path
3. In `crates/somnus/src/main.rs`:
   - `--onnx-models-dir` flag for where to store the ONNX files
   - On startup (if flag set): call `enable_ort_inference`
   - At sleep→wake boundary: rebuild ORT session if model changed

The ORT session needs:
- `ort` crate dependency in `somnus-perception`
- `migraphx_session` helper (currently in `somnus/src/main.rs`) needs to move or be duplicated

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
