# Session state — somnus

> **Last updated:** 2026-06-14
> **Last commit on main:** 039b1a7 — feat(S1): adaptive salience gate — EWMA replaces fixed threshold (ADR 0024)
> **Branch:** `main` (clean)

## Status: S1 (adaptive salience gate) implemented and committed. S2 (settle-rate modulation) is next.

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| ADR 0024 written | 98828ed | Architecture for neuromodulatory self-regulation. Bucket S split into S-query/S-safety/S-flags/S-throttle/S-ops. YAML front-matter deferred items across all 14 ADRs. |
| **S1 — Adaptive salience gate** | 039b1a7 | Replaced `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD=0.8` with EWMA-based threshold. `sal_gate(mu, sigma, n)` = clamp(μ+0.5σ, 0.3, 3.0) once n≥100, else 0.8. State persisted in `snapshot/sal_ewma.bin` (16B: μ+σ+n). Updated all 4 call sites (audio/video/2×lang). New NDJSON fields: `sal_ewma_mean`, `sal_ewma_std`, `sal_threshold`. |

---

## ADR 0024 implementation sequence

| Phase | Signal | Status | Replaces |
|-------|--------|--------|---------|
| 1a | S1 — Adaptive salience gate | **DONE** (039b1a7) | `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD=0.8` |
| 1b | S2 — Settle-rate modulation | **NEXT** | `--settle-gate 0.85` (static prior_cosine threshold) |
| 2 | S3 — Wakefulness pressure homeostat | pending | `--sleep-trigger-fraction` + `--max-wake-secs` |
| 2b | S4 — Prior-beta auto-enable | pending | `--prior-beta 0.0` |
| — | Growth controller | deferred | prerequisite: within-session gist probe |

---

## S2 spec (ADR 0024 §S2 — settle-rate modulation)

**What**: replace static `--settle-gate 0.85` with per-session prior_cosine EWMA gate.
- EWMA of per-episode prior_cosine mean (across 5 regions), α=0.01
- Settle fires when current cosine < C[t] − 0.1  (i.e., episode is below-trend)
- Cold-start: settle always fires for first 100 episodes (fallback)
- Persist in `snapshot/settle_cosine_ewma.bin` (12B: mu f32 + n u64)
- NDJSON: `settle_ewma_mean`, `settle_gate` (the actual C[t]−0.1 value)
- No minimum settle gate (can go to 0 if priors are all low)

Current code path for settle in main.rs:
- `settle_iters > 0` check and trigger logic lives inside `PerceptionProcessor` (somnus-perception)
- `--settle-gate` is passed to `perception.settle_gate` at startup
- Each episode: `out.settle_iters` is returned; in main loop we track `settle_trigger_count`

Implementation:
1. `snapshot.rs`: add `save_settle_ewma(mu: f32, n: u64, ...)` / `load_settle_ewma(...) -> (f32, u64)`
2. `lib.rs`: export
3. `main.rs`:
   - Load at startup
   - Helper `settle_gate_val(mu: f32, n: u64) -> Option<f32>` — returns `Some(mu - 0.1)` when n≥100, `None` for cold-start (always fire)
   - After each episode with `out.prior_cosines`: update EWMA with `episode_cosine_mean()`
   - Pass dynamic gate to perception (need to check how `--settle-gate` is currently consumed)
   - Save after sleep cycle
   - Emit NDJSON fields

Need to read `--settle-gate` consumption path in perception before implementing.

---

## Previous session history (key items)

- `--recall-ef N` (de4a49e): `SEARCH_EF` now a `SemanticStore` field, default 64 → **ef=16 confirmed as default**
- Async wake_reconstruct (b7c4c8f + 2e9d1f6): `WakeReconWorker` in `somnus-perception/src/wake_worker.rs`
- Sleep-split fix (1d8c61e): `--nrem-salience-threshold` (default 0 = disabled)
- Weight restore fix (7f0194c): `weights_path()` returns stem without extension
- Pre-grow before ONNX build (e4c7275): read `region_sizes.bin` before `rebuild_ort_session`

---

## Open issues / queued next

1. **S2 — Settle-rate modulation** (immediate next): Replaces `--settle-gate 0.85`. Spec above.
2. **A/B validate S1** (can do after S2 or independently): Run 2s×3c to confirm `sal_threshold` moves from 0.8 and `high_salience_episodes` changes vs baseline.
3. **S3 — Wakefulness pressure homeostat**: After S2 stable.
4. **S4 — Prior-beta auto-enable**: After S3.
5. **Within-session gist probe**: Prerequisite for growth controller.

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
- settle_beta=0.5, settle_gate=0.85 (static, to be replaced by S2), settle_every=1 (all defaults)
- SEARCH_EF default → `--recall-ef 16` (confirmed at scale)
- wake_reconstruct_interval=5 (default)
- SAL_EWMA_COLD_START=0.8, α=0.01, cold-start until n≥100

---

## Performance knobs

| Flag | Default | Effect |
|---|---|---|
| `--settle-steps 0` | 0 | Disable settle entirely; saves ~3ms/ep |
| `--settle-every N` | 1 | Subsample settle; `--settle-every 4` → ~3.4ms/ep perception |
| `--settle-gate F` | 0.85 | Raise to reduce settle trigger rate as priors improve (S2 will make this adaptive) |
| `--recall-ef N` | 64 | Lower → faster HNSW recall queries, less accurate |
| `--wake-recon-async` | false | Offload wake_reconstruct backward to worker thread |
| `--wake-reconstruct-interval N` | 5 | Fire wake_reconstruct every N episodes |
| `--video-stride N` | 1 | Encode every Nth video frame |

---

## MIGraphX constraint (hard)

Never expose ScatterElements or INT64 outputs. Two-graph split, topk/masking on CPU.
