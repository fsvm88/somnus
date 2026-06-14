# Session state — somnus

> **Last updated:** 2026-06-14
> **Last commit on main:** 7214d16 (within-session gist probe)
> **Branch:** `main` (clean)

## Status: ADR 0024 complete (S1–S4 + growth controller 2c, docs updated). Next: S5 (startup-queryable priors) or S6 (adaptive sleep ceilings).

---

## Completed this session

| Item | What changed |
|------|-------------|
| ADR 0024 S1–S4 implemented | S1–S4 committed in prior session (039b1a7–dfcdd7c) |
| S4 bug found | `blend_with_prior` returns `None` cosine when `prior_beta <= 0.0` → chicken-and-egg: S4 ring buffer never fills, S2 settle EWMA freezes at cold start |
| S4 fix | `--prior-beta` default changed 0.0 → 0.01; S4 guard changed to `args.prior_beta < 0.05`; on scale-down resets to `args.prior_beta` (base) not 0.0 |
| Campaign script | `PRIOR_BETA` env var added (default 0.01); replaces hardcoded `--prior-beta 0.1` |
| ADR 0024 updated | S4 section rewritten as scale-up controller; mapping table updated; amendment with validation data appended; s5/s6 deferred triggers marked met |
| Confirmatory run | 2s×3c fresh memory, PRIOR_BETA=0.01 — all four controllers confirmed ✅ |
| Within-session gist probe | `SemanticStore::records_since()`, `SleepStats::within_session_gist_rate`, `session_start_secs` param — committed 7214d16 |

### Confirmatory run key metrics (memory/adr0024_ab2/campaign_20260614_114140)

| c | s | trigger | wake_eps | sal_threshold | settle_gate_dynamic | pressure | prior_beta_eff |
|---|---|---------|----------|--------------|-------------------|----------|---------------|
| 1 | 1 | high_water | 3276 | 0.800 | 0.638 | 0.000 | 0.010 |
| 2 | 1 | pressure | 500 | 1.023 | 0.576 | 2.796 | 0.010 |
| 3 | 1 | pressure | 120 | 0.825 | 0.618 | 2.611 | 0.010 |
| 1 | 2 | pressure | 287 | 0.345 | 0.697 | 0.732 | 0.010 |
| 2 | 2 | pressure | 120 | 0.424 | 0.567 | 2.504 | 0.010 |
| 3 | 2 | pressure | 120 | 0.519 | 0.593 | 2.645 | **0.050** |

S4 scaled up at S2C3 — prior cosines exceeded 0.65 for 3 consecutive cycles.

### Within-session gist probe design notes
- `recall.warm.records_since(session_start_secs)` filters by `timestamp_secs >= session_start_secs`
- `session_start_secs` = `wall_secs(&SystemTime::now())` at process start — same epoch as episode timestamps
- Probe runs k-means + brute-force NN exclusively on within-session records (both centroids AND neighbor pool)
- Emitted as `"within_session_gist_rate"` in per-cycle NDJSON
- In session 1 cycle 1: warm is empty → `None`. In session N cycle 1: warm has only prior-session records → `None`. Metric only populated from cycle 2 onward within each session.
- To validate: S1C1 should give `None`; `gist_recall_rate` and `within_session_gist_rate` should be close in a fresh-memory run (no cross-session drift yet).

---

## ADR 0024 implementation sequence

| Phase | Signal | Status | Notes |
|-------|--------|--------|-------|
| 1a | S1 — Adaptive salience gate | **DONE** | |
| 1b | S2 — Settle-rate modulation | **DONE** | |
| 2 | S3 — Wakefulness pressure homeostat | **DONE** | |
| 2b | S4 — Prior-beta scale-up controller | **DONE** (bug fixed) | 0.01→0.05, not toggle |
| prereq | Within-session gist probe | **DONE** | `within_session_gist_rate` in NDJSON |
| — | Growth controller | **DONE** | ws→gist→MSE priority; validated 5s×3c; S5C2 false positive eliminated |

---

## Open issues / queued next

1. **S5 — Startup-queryable priors**: Trigger met (S1+S2 validated). Query RAM/HNSW at startup to auto-seed `--max-warm-records` and `--recall-ef`.
3. **S6 — Adaptive safety ceilings**: Trigger met (S3 validated). Widen `--max-wake-secs` to 3× rolling mean wake duration.
4. **Feature flag defaults** (noted in ADR 0024): `--enable-language`, `--enable-action-decoder`, `--wake-recon-async` should default to `true`. Low priority until pipeline is stable.

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
- settle_beta=0.5, settle_every=1 (defaults)
- SEARCH_EF → `--recall-ef 16` (confirmed at scale)
- wake_reconstruct_interval=5 (default)
- SAL_EWMA_COLD_START=0.8, α=0.01 for S1/S2 EWMAs
- S3: P_THRESHOLD=1.0, P_MAX=3.0, P_COLD_START_N=500, P_MIN_EPISODES=120
- S4: base=0.01 (default), scale target=0.05, enable threshold=0.65 (3 cycles), disable threshold=0.40
- `PRIOR_BETA=0.01` is now the campaign default (was 0.1 hardcoded, then parametrized)

---

## Performance knobs

| Flag | Default | Effect |
|---|---|---|
| `--settle-steps 0` | 0 | Disable settle entirely |
| `--settle-every N` | 1 | Subsample settle |
| `--settle-gate F` | 0.85 | Cold-start prior for S2 adaptive gate |
| `--recall-ef N` | 64 | Lower → faster HNSW queries |
| `--wake-recon-async` | false | Offload wake_reconstruct to worker |
| `--wake-reconstruct-interval N` | 5 | Fire wake_reconstruct every N episodes |
| `--video-stride N` | 1 | Encode every Nth video frame |

---

## MIGraphX constraint (hard)

Never expose ScatterElements or INT64 outputs. Two-graph split, topk/masking on CPU.
