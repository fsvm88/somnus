# Session state — somnus

> **Last updated:** 2026-06-12
> **Last commit on main:** b7c4c8f — perf: async wake_reconstruct_step via WakeReconWorker
> **Branch:** `main` (clean)

## Status: Perf queue from 2026-06-08 cleared. Next: validate with a campaign run.

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| `--recall-ef N` | de4a49e | `SEARCH_EF` now a `SemanticStore` field, set via CLI (default 64). Propagates through `rebuild_from_recent`. `--recall-ef 16` ≈ 4× faster HNSW queries; validate recall with `eval_metrics.py` |
| cargo fmt pass | 93daa34 | rustfmt normalization across 10 files, no code changes |
| Async wake_reconstruct | b7c4c8f | `--wake-recon-async` (off by default): `WakeReconWorker` in `somnus-perception/src/wake_worker.rs` |

**Item 2 from the old queue (lazy `regional_inputs`) was skipped** — re-evaluated as planned: ~0.05ms/ep for a `PerceptionOutput` refactor; not worth it. Drop unless profiling says otherwise.

---

## WakeReconWorker design (b7c4c8f)

Key insight: `wake_reconstruct_step` trains **only the 5 output heads**, and wake inference reads **only `ema_heads.flat`** — so the worker needs just the heads, not the full model.

- Worker thread owns: copy of 5 head Linears, its own AdamW, the authoritative EMA state during wake.
- Wake: `request_train` non-blocking (drops when backlogged → `wake_recon_dropped` in cycle NDJSON); EMA flat weights stream back, applied via `apply_ema_flat` at the poll site next to the gain-worker poll.
- Sleep entry: `fetch_weights` (blocking handshake) merges wake-trained heads into the main model before NREM, preserving ParamIds so main AdamW moments stay attached.
- Wake re-entry: `set_weights` pushes NREM-trained heads back; worker AdamW moments stale by one sleep cycle (accepted).
- Covered by `wake_worker::tests::round_trip_train_and_weight_sync`.
- Known approximations: EMA application lags a few episodes; main optimizer's head moments idle during wake.

---

## Queued next

1. **Validation campaign run** with the new flags:
   - `--recall-ef 16` → check `warm_recall_l2` / `gist_recall_rate` don't degrade unacceptably
   - `--wake-recon-async` → confirm `mean_output_mse_wake` stays comparable to sync baseline, `wake_recon_dropped` ≈ 0, and per-episode latency spikes are gone
2. Pre-existing clippy `--tests` lints in `somnus-encoding/tests/mel_parity.rs` and `somnus-memory/src/pq.rs` (identity_op, assertions_on_constants) — not from this session; fix opportunistically. Note: project gate is `cargo clippy -- -D warnings` without `--tests`, which is clean.

---

## True hot path breakdown (ONNX active)

| Component | ms/ep | Notes |
|---|---|---|
| bind_us (ORT enc + ORT bind) | ~1.6–2.6 | |
| settle_us | ~2–3.5 × settle_every rate | |
| prior_update_us (5× sgemm heads) | ~0.15 | |
| sym_us (2176→384 Vec matmul) | ~0.30 | |
| HNSW query_warm_with_dist | ~1–15 → ~0.25–4 with `--recall-ef 16` | grows with warm index |
| wake_reconstruct_step | ~0 on main loop with `--wake-recon-async` | was ~1–3 amortized |
| JSON episode write (640 f32) | ~0.5 | |

With `--settle-every 4` + `--recall-ef 16` + `--wake-recon-async`: main-loop cost ≈ **4ms/ep** total expected.

---

## Performance knobs

| Flag | Default | Effect |
|---|---|---|
| `--settle-steps 0` | 0 | Disable settle entirely; saves ~3ms/ep |
| `--settle-every N` | 1 | Subsample settle; `--settle-every 4` → ~3.4ms/ep perception |
| `--settle-gate F` | 0.85 | Raise to reduce settle trigger rate as priors improve |
| `--recall-ef N` | 64 | Lower → faster HNSW recall queries, less accurate |
| `--wake-recon-async` | false | Offload wake_reconstruct backward to worker thread |
| `--wake-reconstruct-interval N` | 5 | Fire wake_reconstruct every N episodes |
| `--video-stride N` | 1 | Encode every Nth video frame |

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
- SEARCH_EF default 64 → now `--recall-ef` (somnus-memory/src/semantic.rs)
- wake_reconstruct_interval=5 (default), salience gate=0.8
