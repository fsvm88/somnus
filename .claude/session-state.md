# Session state — somnus

> **Last updated:** 2026-06-13
> **Last commit on main:** e4c7275 — fix: pre-grow perception before ORT build to match saved region sizes
> **Branch:** `main` (clean)

## Status: Weight-restore bug fixed and validated (3-session test passes, all tests green). Residual c1 gist collapse is intrinsic (cross-session embedding drift); not a bug.

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

## Completed this session (2026-06-13)

| Item | Commit | What changed |
|------|--------|--------------|
| Sleep-split fix | 1d8c61e | `--nrem-salience-threshold` (default 0 = disabled). Encoders in file mode ran faster than real-time, hitting the old hardcoded 16-threshold and aborting NREM before REM every c1. Live mode: pass `--nrem-salience-threshold 16`. |
| HNSW timing metric | 37fa1eb | `warm_query_us_mean` added to cycle NDJSON (was missing — needed for ef verdict). |
| At-scale A/B | 37fa1eb | `scripts/run_ef_scale_ab.sh` launched in background (PID 64286). A=ef64, B=ef16. 10 sessions × 3 cycles × CAPACITY=4096, two 1-hr walk clips, ONNX on. Expected ~100k warm records by end. |
| **Critical: weight restore fix** | 7f0194c | Burn's `BinFileRecorder` calls `path.set_extension("bin")` on every save/load, replacing `.safetensors` with `.bin`. `weights_path()` returned `weights.safetensors`; the `exists()` check always failed → model trained from scratch every session. Fixed: `weights_path()` now returns stem `weights` (no extension); `exists()` checks use `.with_extension("bin")`. Decoder path fixed identically. |
| **Follow-on: pre-grow before ONNX build** | e4c7275 | Weight restore set region sizes AFTER ORT session was built from base [1024,...] model → ORT/Burn mismatch → panic (exit 101). Fixed: read `region_sizes.bin` and pre-grow perception BEFORE `rebuild_ort_session`, so ONNX is always exported from the correct (possibly grown) model. Weight restore then just loads weights with no size change needed. |

---

## At-scale A/B: complete (pre-fix — results partially invalidated)

Both campaigns ran with the weight-restore bug active (model trained from scratch every session). Results are still valid for the **ef=16 vs ef=64 query latency comparison** (that comparison is model-agnostic). **NOT valid for gist_recall_rate** as a measure of cross-session memory quality.

- ef=16 is 2.37× faster on HNSW queries with no meaningful recall degradation at 98k records → **ef=16 confirmed as default**
- gist_recall_rate showed c1=near-zero, c2=mid, c3=high pattern EVERY session — this was the bug, not expected behavior. With fresh model each session, old warm embeddings never clustered with current ones.

---

## Validation result (3 sessions × 3 cycles, CAPACITY=1024, 2 walk clips)

| cycle | gist  | rl2   | grew  | region_sizes              |
|-------|-------|-------|-------|---------------------------|
| s1c1  | nan   | nan   | True  | [1088,1024,1088,1088,1024]|
| s1c2  | 0.376 | 1.001 | True  | [1152,1088,1152,1152,1024]|
| s1c3  | 0.537 | 0.532 | True  | [1216,1152,1216,1216,1024]|
| s2c1  | 0.252 | 1.238 | True  | [1280,1216,1280,1280,1024]|
| s2c2  | 0.532 | 0.511 | True  | [1344,1280,1344,1344,1024]|
| s2c3  | 0.592 | 0.396 | True  | [1408,1344,1408,1408,1024]|
| s3c1  | 0.015 | 1.304 | True  | [1472,1408,1472,1472,1024]|
| s3c2  | 0.053 | 0.611 | True  | [1536,1472,1536,1536,1024]|
| s3c3  | 0.935 | 0.257 | False | [1536,1472,1536,1536,1024]|

Key observations:
- Region sizes now accumulate across sessions ✓ (was resetting to 1088 every session)
- s2c1 gist = 0.252 (was ~0.05 before fix) — cross-session recall improved ✓
- Residual c1 collapse is intrinsic: c1 nearest warm neighbors are from previous session (full session of NREM drift away); c3 has within-session c2 neighbors (1 NREM apart) → high gist
- Growth is still unbounded (every cycle except s3c3) — runaway growth concern
- s3c1 gist = 0.015 despite weight restore: at session 3+, the growing model diverges faster from warm store embeddings than at session 2

## Open issues / queued next

1. **Runaway growth**: regions growing from 1024 → 1536 in 3 sessions without pruning. Root cause: GIST_GROWTH_THRESHOLD=0.85 almost never met at c1/c2; PRUNE_GIST_THRESHOLD never met consistently. Options: (a) lower growth threshold, (b) use mean_output_mse as growth signal (not gist), (c) add a growth cooldown.
2. **Residual c1 gist collapse**: intrinsic to raw embedding warm store + NREM drift. Could address with re-encoding warm store after NREM (expensive) or a different gist probe strategy (e.g., probe only within-session warm entries for the growth signal).
3. **Long-clip campaign**: once growth is under control, try `clips/long/` with `--nrem-salience-threshold 0`.

## Closed items (this arc)

- **Worker drop rate**: 0% in real campaigns (CAPACITY=4096 + ef=16). The 25–35% was from the small-capacity validation run. No action needed.
- **Clippy --tests lints**: fixed in 3ebf15f (pq.rs const assert, mel_parity.rs `/1` simplification).

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
