# Session state — somnus

> **Last updated:** 2026-06-07
> **Last commit on main:** 590f7e6 — GPU unification Phase 1.5 + train_campaign.sh ONNX defaults
> **Branch:** `main` (not yet pushed)

## Status: Phase 1.5 COMPLETE (encoder ORT live). Needs benchmark validation. Phase 2 (micro-batching) has a fundamental issue — see below.

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| Phase 1.5: encoder ORT | 590f7e6 | All 5 encoder matmuls + cross-modal proj move to GPU; topk+dc stays in Rust |
| train_campaign.sh ONNX defaults | 590f7e6 | ONNX_MODELS_DIR / ONNX_SCRIPT env vars; no more EXTRA_ARGS required |

---

## GPU unification status: Phase 1.5 COMPLETE (unvalidated)

**Phase 0 (burn↔ONNX gate):** All checks pass at <2e-08.

**Phase 1 (ORT binding session live):** 2.4–2.8× speedup confirmed (campaign_20260602_2308).

**Phase 1.5 (ORT encoder session live — Phase 1.5 commit 590f7e6):**
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

## Phase 2 (micro-batching) — BLOCKED on design issue

**Problem:** The GRU binding is sequential (episode n feeds its hidden state into episode n+1).
Batching the binding across episodes is not possible in the wake path. Only the encoder can be
batched (B independent matmuls → B z vectors), but the encoder is now cheap (GPU).

**Conclusion:** "Phase 2 micro-batching for compounding GPU benefit" is mostly false — the
binding (the actual bottleneck) cannot be batched. Building a wake-path buffering refactor
to batch only the encoder is low ROI.

**Options for what Phase 2 becomes:**
- Redirect to next real bottleneck (Qwen embeddings, NREM replay, something measured)
- Defer / shelve — focus on validation + next research step
- User decides

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
