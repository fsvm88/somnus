# Session state — somnus

> **Last updated:** 2026-06-07
> **Last commit on main:** c965118 — Update session state: GainWorker complete
> **Branch:** `main` (not yet pushed)

## Status: Phase 1.5 + GainWorker COMPLETE and VALIDATED. Duty-cycle hypothesis falsified.

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| train_campaign.sh ONNX defaults | 590f7e6 | ONNX_MODELS_DIR / ONNX_SCRIPT env vars |
| Phase 1.5: encoder ORT | 590f7e6 | All 5 encoder matmuls + cross-modal proj → GPU |
| GainWorker | 321cb8e | Qwen gain ORT offloaded to dedicated OS thread |

---

## Benchmark results (warm-cycle averages, CAPACITY=2048)

| Component | Burn-only | Phase 1.5 | Phase 1 (CAPACITY=4096) |
|---|---|---|---|
| bind_ms/ep | 2.90 | 1.53 | 0.84 |
| kwta_ms/ep | 1.49 | 1.76 | 2.00 |
| sym_ms/ep | 3.83 | 3.04 | 2.86 |
| prior_ms/ep | 5.34 | 5.93 | 3.24* |
| **perc_ms/ep** | **18.08** | **16.83** | **19.12*** |

*Phase 1 prior/perc numbers flattered by larger capacity (more within-cycle warmup).

Phase 1.5 beats burn-only by **1.25ms/ep** on total perception wall time.

---

## Duty-cycle validation: DONE — hypothesis FALSIFIED

Burn-only control campaign (same CAPACITY=2048, no ONNX) showed:
- settle/ep = 0.94 (Phase 1.5 = 0.93, Phase 1 = 0.73)
- prior_ms/ep = 5.34 (Phase 1.5 = 5.93)

**The higher settle rate vs Phase 1 is from smaller capacity (1630 vs 3268 ep/cycle),
not from the ORT encoder duty-cycle cadence change.** Phase 1.5 is clean.

The 0.6ms/ep residual gap (Phase 1.5 vs burn-only prior) is minor overhead from having
ORT sessions active — not systemic.

---

## Current bottleneck breakdown (Phase 1.5, warm CAPACITY=2048)

| Component | ms/ep | Notes |
|---|---|---|
| **prior_us** | **5.93** | #1 — settle loop + prior update in burn |
| sym_us | 3.04 | symbolic cross-modal in burn |
| kwta_us | 1.76 | topk + dc update in Rust |
| bind_us | 1.53 | encoder ORT + binding ORT |

Separate threads (not in perception path):
- video_encode: ~6ms/ep warm (DINOv2)
- speech_encode: ~5ms/ep warm (Whisper)

---

## GPU unification status

**Phase 0 (burn↔ONNX gate):** All checks pass at <2e-08.
**Phase 1 (ORT binding):** 2.4–2.8× speedup confirmed.
**Phase 1.5 (ORT encoder):** Validated — 1.25ms/ep better than burn-only.
**GainWorker:** No stall impact confirmed.

---

## MIGraphX constraint (hard)

**Never expose ScatterElements outputs as ONNX graph outputs.** MIGraphX on RDNA4/gfx1201 crashes
with "invalid input for conversion to integer" when ≥4 such outputs are present.
Also: **INT64 graph outputs trigger the same bug**. Keep topk indices off the output list.

---

## Key facts

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--working-memory-size 8` hardcoded in `train_campaign.sh`
- `--asr-min-chars 5` for noisy-audio clips
- `--max-warm-records 200000` recommended for long runs
- Wake schedule trigger at 602s/cycle
- N_UNITS=1024, K_ACTIVE=20, D_PROJ=128, EPISODE_DIM=640
- GRU_INPUT=640, GRU_HIDDEN=256, CROSS_MODAL_DIM=2176
- burn GRU: reset_after=True (default) → ONNX linear_before_reset=1
- burn Linear weight shape: [n_in, n_out] (transpose of ONNX/PyTorch convention)
- DUTY_CYCLE_ALPHA=0.001, BOOST_STRENGTH=1.0, TARGET_DC=20/1024≈0.0195
