# Session state — somnus

> **Last updated:** 2026-06-12
> **Last commit on main:** 2e9d1f6 — fix: wake-recon worker heads were frozen (missing require_grad)
> **Branch:** `main` (clean)

## Status: Perf queue cleared AND validated via A/B campaign runs. Async wake-recon works after require_grad fix.

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| `--recall-ef N` | de4a49e | `SEARCH_EF` now a `SemanticStore` field, set via CLI (default 64). Propagates through `rebuild_from_recent` |
| cargo fmt pass | 93daa34 | rustfmt normalization across 10 files, no code changes |
| Async wake_reconstruct | b7c4c8f | `--wake-recon-async` (off by default): `WakeReconWorker` in `somnus-perception/src/wake_worker.rs` |
| require_grad fix | 2e9d1f6 | **Critical**: `Param::initialized` inherits `require_grad` from the tensor; `from_data` tensors don't track gradients. Both `set_linear_weight` (worker) and `set_head_weights_flat` (sleep merge) produced silently frozen heads. Caught by A/B run. Also req channel 4→16, test now asserts weights move |

**Item 2 from the old queue (lazy `regional_inputs`) was skipped** — re-evaluated as planned: ~0.05ms/ep for a `PerceptionOutput` refactor; not worth it.

---

## A/B validation results (2026-06-12)

Setup: 2 sessions × 3 cycles, capacity 2048, fresh memory dirs, ONNX active.
A = baseline (`memory/ab_baseline/campaign/`), B2 = `--recall-ef 16 --wake-recon-async` after fix (`memory/ab_newflags/campaign/`).

- **Async learning confirmed**: B2 `mean_output_mse_wake` 0.67→0.46, prior cosines climb to [0.55, 0.53, 0.52, –, 0.27]. First B run (frozen heads) had flat mse ~0.7 and cosines ~0.
- **B2's sleep cycles split**: c1 sleep aborts before REM (pending embeddings interrupt — pre-existing dynamic, also seen in 2026-06-07 baseline c3), re-enters immediately as an empty "c2" that finishes REM+spill. So B2 effectively got 2 training cycles/session vs A's 3 — compare B2 c3 against A c2.
- **At matched progress, B2 ≈ A**: mse 0.456–0.468 vs A c2's 0.467–0.487; warm_recall_l2 1.00–1.05 vs 0.93–1.05; gist_recall_rate 0.35–0.39 vs 0.32–0.53. No recall-ef-16 degradation visible at this index size (~10k records) — the real test needs a 100k+ warm index.
- **Wake cycles faster**: B2 ~28–31s vs A ~31–46s (confounded with cycle splitting; directionally right).
- **Open issue: drop rate 25–35%** (78–122/cycle) even with 16-deep queue — worker throughput (backward + AdamW + EMA + 6.5MB flat clone per step at ~13 req/s) can't absorb bursts. Worker still trains on ~200+ eps/cycle and converges fine. Options if it matters: batch drained requests in the worker, throttle EMA responses (every Nth step), or accept.

---

## WakeReconWorker design (b7c4c8f + 2e9d1f6)

Key insight: `wake_reconstruct_step` trains **only the 5 output heads**, and wake inference reads **only `ema_heads.flat`** — so the worker needs just the heads, not the full model.

- Worker thread owns: copy of 5 head Linears, its own AdamW, the authoritative EMA state during wake.
- Wake: `request_train` non-blocking (drops when backlogged → `wake_recon_dropped` in cycle NDJSON); EMA flat weights stream back, applied via `apply_ema_flat` at the poll site next to the gain-worker poll.
- Sleep entry: `fetch_weights` (blocking handshake) merges wake-trained heads into the main model before NREM, preserving ParamIds so main AdamW moments stay attached.
- Wake re-entry: `set_weights` pushes NREM-trained heads back; worker AdamW moments stale by one sleep cycle (accepted).
- Covered by `wake_worker::tests::round_trip_train_and_weight_sync`.
- Known approximations: EMA application lags a few episodes; main optimizer's head moments idle during wake.

---

## Queued next

> Recommended order (discussed end of 2026-06-12 session): start with **#3 sleep-split fix** — it taxes every file-mode campaign one cycle/session, small scope. Then launch **#1 at-scale run** in the background. Caveat for #3: the interrupt exists so live input isn't starved during sleep; fix must not break live ingest.

1. **recall-ef verdict at scale**: re-run the A/B on a grown warm index (100k+ records, e.g. long-clips campaign with `--max-warm-records 200000`) — at ~10k records HNSW is cheap either way, so the small run couldn't show the recall/speed tradeoff.
2. **Worker drop rate** (25–35%): decide whether to batch drained Train requests in the worker, throttle EMA responses (every Nth step), or accept. Only matters if dropped high-salience episodes hurt head convergence — current evidence says no.
3. **Sleep-split dynamic**: pending-embeddings interrupt aborts REM, causing an empty follow-up cycle (pre-existing, not async-specific; also in 2026-06-07 baseline). Costs one effective training cycle per session in file mode. Look at the interrupt condition in `run_sleep_cycle`.
4. Pre-existing clippy `--tests` lints in `somnus-encoding/tests/mel_parity.rs` and `somnus-memory/src/pq.rs` (identity_op, assertions_on_constants). Project gate (`cargo clippy -- -D warnings`, no `--tests`) is clean.

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
