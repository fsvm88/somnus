# Session state — somnus

> **Last updated:** 2026-06-08
> **Last commit on main:** 47a0101 — Update session state: three perf opts committed, metric fix documented
> **Branch:** `main` (clean, up to date with origin)

## Status: Perf analysis complete. Three next targets queued.

---

## Completed this session (9b5e3bf)

| Item | What changed |
|------|--------------|
| Fix `perception_total_ms` metric | Video timer now wraps only `ingest_visual`; was including `record_episode` + `wake_reconstruct_step` (~12ms inflation) |
| SIMD EMA matmuls | `EmaOutputHeads::predict` → `matrixmultiply::sgemm` (AVX2/FMA); expected ~0.15–0.20ms/ep savings |
| `--settle-gate FLOAT` | Expose `SETTLE_COSINE_GATE` as CLI arg (default 0.85); raises gate to reduce settle trigger rate |
| `--settle-every N` | Subsample settle to every N episodes (default 1); `--settle-every 4` saves ~2.25ms/ep |

Previous sessions: ORT encoder+binding, GainWorker, flat sym matmul, burn kwta elimination.

---

## What the "12ms gap" was

The old video `t_perc` timer wrapped the entire Ok() block. The excess came from:
- **`record_episode` → HNSW `query_warm_with_dist`**: ~1–15ms depending on warm index size
- **`wake_reconstruct_step`**: full burn autodiff backward, 5–15ms/call @ every 5 eps = ~1–3ms amortized
- **JSON episode write** (640 floats): ~0.5ms

Cycle 3's 4.93ms ≈ components (4.75ms) was already clean. Cycles 1–2 were inflated by these triggers.
After metric fix, all cycles should show ~5–6ms/ep perception.

---

## True hot path breakdown (post-fix, ONNX active)

| Component | ms/ep | Timer |
|---|---|---|
| bind_us (ORT enc + ORT bind) | ~1.6–2.6 | inside perception |
| settle_us (ORT enc + topk + ORT bind) | ~2–3.5 × settle_every rate | inside perception |
| prior_update_us (5× sgemm heads) | ~0.15 | inside perception |
| sym_us (2176→384 Vec matmul) | ~0.30 | inside perception |
| **HNSW query_warm_with_dist** | ~1–15 (grows with warm index) | outside, untracked |
| **wake_reconstruct_step** | ~1–3 amortized (5–15ms/call × 1/5 ep) | outside, untracked |
| JSON episode write (640 f32) | ~0.5 | outside, untracked |

With `--settle-every 4`: perception ≈ **3.4ms/ep**. Total frame latency also depends on HNSW + wake_reconstruct.

---

## Queued next (smallest → biggest)

### 1. `--recall-ef N` (15 min)
`SEARCH_EF=64` hardcoded in `crates/somnus-memory/src/semantic.rs:23`. Expose as CLI arg.
- Wire through `RecollectionLayer` → `SemanticStore::query_with_dist`
- At `--recall-ef 16`: ~4× faster HNSW queries; growing warm index is the main long-run bottleneck
- Tradeoff: lower recall accuracy — validate with `eval_metrics.py`

### 2. Lazy `regional_inputs` in `try_bind` (30 min)
`crates/somnus-perception/src/lib.rs:818–825`: 5 `latest_inputs.clone()` calls every episode (~10KB).
Only needed for `episodic.push` (always) and `wake_reconstruct_step` (every 5 eps × salience gate).
Approach: clone `regional` only when episode is Some (already gated), which it is — nothing to change there.
Real fix: avoid storing full 5-vec array in `PerceptionOutput`; instead pass it lazily only when needed.
Note: `episodic.push` stores all 5 for sleep replay, so we can't skip the clone for push — only for wake_reconstruct.
Actually minor (~0.05ms/ep) — may not be worth the refactor. Re-evaluate after --recall-ef.

### 3. Async `wake_reconstruct_step` (2–3 hr)
Move the blocking autodiff backward pass off the main ingest loop onto a dedicated worker thread.
Model: identical to `GainWorker` pattern in `crates/somnus/src/action.rs`.
- Worker thread owns a clone of the model + optimizer
- Main loop sends `(episode: Vec<f32>, regional: [Vec<f32>; 5])` over a channel (non-blocking)
- Worker does backward + AdamW step + EMA update, sends updated EMA weights back
- Main loop applies updated EMA weights non-blockingly
Concern: model desync between perception's training model and wake_reconstruct's model copy. Need to periodically sync. Same issue GainWorker has with stale weights — acceptable for research.
Saves: ~1–3ms/ep amortized, and eliminates latency spikes on high-salience episodes.

---

## Performance knobs

| Flag | Default | Effect |
|---|---|---|
| `--settle-steps 0` | 0 | Disable settle entirely; saves ~3ms/ep |
| `--settle-every N` | 1 | Subsample settle; `--settle-every 4` → ~3.4ms/ep perception |
| `--settle-gate F` | 0.85 | Raise to reduce settle trigger rate as priors improve |
| `--wake-reconstruct-interval N` | 5 | Fire wake_reconstruct every N episodes |
| `--video-stride N` | 1 | Encode every Nth video frame; halving visual load with N=2 |

---

## settle_episode execution paths

- **Path A** (ORT encoder + binding active): ORT encoder → topk_from_z_no_dc → ORT binding
- **Path B** (binding only): burn kwta → ORT binding
- **Path C** (burn only): burn kwta → burn binding

GRU hidden saved before settle loop, restored after each binding call.
settle_beta=0.5 (default): blends raw input with 50% EMA prediction before re-encoding.

---

## MIGraphX constraint (hard)

Never expose ScatterElements or INT64 outputs. Two-graph split, topk/masking on CPU.

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
- settle_beta=0.5, settle_gate=0.85, settle_every=1 (all defaults)
- SEARCH_EF=64 (hardcoded in somnus-memory/src/semantic.rs:23)
- wake_reconstruct_interval=5 (default), salience gate=0.8
