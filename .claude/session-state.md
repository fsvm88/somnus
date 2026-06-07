# Session state — somnus

> **Last updated:** 2026-06-07
> **Last commit on main:** ea9ed9a — ORT encoder in settle: replace burn kwta with encoder ORT + topk_no_dc
> **Branch:** `main` (not yet pushed)

## Status: ORT settle fully optimised. Cumulative −3.98ms/ep (22%) over burn-only.

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| train_campaign.sh ONNX defaults | 590f7e6 | ONNX_MODELS_DIR / ONNX_SCRIPT env vars |
| Phase 1.5: encoder ORT | 590f7e6 | 5 encoder matmuls → GPU |
| GainWorker | 321cb8e | Qwen gain ORT → dedicated OS thread |
| Instrument prior_us | de26f6c | settle_us + prior_update_us sub-timers in metrics |
| ORT binding in settle | 86ca92e | burn GRU in settle → ORT binding; hidden saved/restored |
| ORT encoder in settle | ea9ed9a | burn kwta in settle → ORT encoder + topk_from_z_no_dc |

---

## Benchmark progression (warm cycles, CAPACITY=2048)

| Stage | perc_total | Δ |
|---|---|---|
| Burn-only (control) | 18.08 ms/ep | — |
| Phase 1.5 (ORT enc+bind) | 16.83 ms/ep | −1.25 |
| + ORT binding in settle | 14.59 ms/ep | −2.24 |
| + ORT encoder in settle | **14.10 ms/ep** | −0.49 |
| **vs burn-only** | | **−3.98 ms/ep (−22%)** |

### Settle bottleneck: CLOSED

| | ms/trigger | settle/ep |
|---|---|---|
| Burn settle | ~5.9 ms | 0.84 |
| ORT bind settle | 4.59 ms | 0.985 |
| ORT enc settle | **1.75 ms** | 0.985 |

Residual 1.75ms = encoder ORT (~0.7ms) + binding ORT (~0.85ms) + topk_no_dc (~0.2ms).
Cannot go lower without reducing settle rate. prior_update_us is now the floor (~0.35ms).

---

## Current bottleneck breakdown (ORT enc settle, warm CAPACITY=2048)

| Component | ms/ep | Notes |
|---|---|---|
| **sym_us** | **~2.5** | #1 — symbolic cross-modal (burn cross_modal_proj + kwta) |
| prior_us | ~2.1 | settle (1.73) + prior_update (0.35) |
| bind_us | ~1.5 | ORT encoder + ORT binding |
| **kwta_us** | **~1.5** | #2 — topk+dc for triggering region (burn) |

Settle bottleneck is closed. Remaining burn costs: sym_us and kwta_us.

---

## settle_episode execution paths (ea9ed9a)

Three paths selected at runtime:
- **Path A** (encoder+binding active): ORT encoder → topk_from_z_no_dc → ORT binding
- **Path B** (binding only): burn kwta → ORT binding
- **Path C** (burn only): burn kwta → burn binding

Hidden state saved before loop, restored after each ORT binding call (settle must not
advance cross-episode GRU state). topk_from_z_no_dc added to KWtaLayer (wraps
private select_top_k_no_dc — no dc update during refinement passes).

---

## Duty-cycle validation: DONE — duty-cycle cadence change is NOT the cause of higher
settle rate vs Phase 1. Burn-only control had same settle rate (0.94/ep). Root cause:
smaller CAPACITY (2048 vs 4096 in Phase 1) → fewer episodes per cycle → less warmup.

---

## GPU unification status

- Phase 0 (gate): passes <2e-08
- Phase 1 (ORT binding): 2.4–2.8× speedup confirmed
- Phase 1.5 (ORT encoder): validated, +1.25ms/ep over Phase 1
- GainWorker: no stall impact confirmed
- ORT settle (enc+bind): +2.73ms/ep over Phase 1.5

---

## MIGraphX constraint (hard)

Never expose ScatterElements or INT64 outputs. Workaround: two-graph split, topk/masking on CPU.

---

## Key facts

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--working-memory-size 8` hardcoded in `train_campaign.sh`
- `--asr-min-chars 5` for noisy-audio clips
- `--max-warm-records 200000` recommended for long runs
- Wake schedule trigger at 602s/cycle; N_UNITS=1024, K_ACTIVE=20, D_PROJ=128, EPISODE_DIM=640
- GRU_INPUT=640, GRU_HIDDEN=256, CROSS_MODAL_DIM=2176
- burn GRU: reset_after=True → ONNX linear_before_reset=1
- burn Linear weight: [n_in, n_out] (transpose of ONNX/PyTorch)
- DUTY_CYCLE_ALPHA=0.001, BOOST_STRENGTH=1.0, TARGET_DC=20/1024≈0.0195
