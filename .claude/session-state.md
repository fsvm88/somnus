# Session state — somnus

> **Last updated:** 2026-06-02
> **Last commit on main:** 56008a0 — GPU unification Phase 1 complete
> **Branch:** `main` (not yet pushed to Codeberg)

## Status: GPU unification Phase 1 COMPLETE. ORT binding session live on MIGraphX. Ready for campaign comparison.

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

## GPU unification status: Phase 1 COMPLETE

**Phase 0 (burn↔ONNX gate):** All checks pass on CPU EP and MIGraphX EP at <2e-08.

**Phase 1 (ORT binding session live):**
- `--onnx-models-dir <dir>` flag enables ORT binding inference
- Session built at startup, rebuilt at each sleep→wake boundary
- Uses MIGraphX EP by default; `--no-onnx-migraphx` falls back to CPU EP
- 401 live wake episodes confirmed via ORT/MIGraphX in 1-cycle test
- Fallback to burn on ORT error (logged as WARN)

**What Phase 1 does NOT change:**
- Burn k-WTA (encoder matmuls) still run for duty-cycle tracking
- Only binding area (GRU + cross-attention) is on GPU via ORT
- EMA heads still on CPU (burn flat matmul)

**Campaign usage with ORT:**
```bash
SESSIONS=3 CYCLES=3 CAPACITY=4096 \
MEMORY_DIR=memory/campaign_$(date +%Y%m%d_%H%M) \
EXTRA_ARGS="--lang-oversample 9 --asr-min-chars 5 --max-warm-records 200000 \
  --onnx-models-dir memory/onnx_models --onnx-script scripts/export_perception_graph.py" \
scripts/train_campaign.sh
```

**Next steps:**
- Phase 1.5: add encoder ORT session (replace burn k-WTA matmuls) 
- Measure bind_ms improvement in campaign (expect ~7ms → <1ms for binding)
- Phase 2: micro-batching (--micro-batch-size)

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
