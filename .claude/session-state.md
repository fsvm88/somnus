# Session state — somnus

> **Last updated:** 2026-06-14
> **Last commit on main:** dfcdd7c — feat(S4): prior-beta auto-enable — cosine-gated controller (ADR 0024)
> **Branch:** `main` (clean)

## Status: ADR 0024 Phase 1+2 complete. S1–S4 all implemented. Growth controller deferred (needs within-session gist probe). Next: A/B validate the controllers, then tackle within-session gist probe.

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| ADR 0024 written | 98828ed | Architecture for neuromodulatory self-regulation. Bucket S split into S-query/S-safety/S-flags/S-throttle/S-ops. YAML front-matter deferred items across all 14 ADRs. |
| **S1 — Adaptive salience gate** | 039b1a7 | Replaced `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD=0.8` with EWMA-based threshold. State in `snapshot/sal_ewma.bin`. NDJSON: `sal_ewma_mean`, `sal_ewma_std`, `sal_threshold`. |
| **S2 — Settle-rate modulation** | 880391d | `--settle-gate 0.85` replaced by prior-cosine EWMA gate C[t]−0.1. State in `snapshot/settle_ewma.bin`. NDJSON: `settle_ewma_mean`, `settle_gate_dynamic`. |
| **S3 — Wakefulness pressure homeostat** | 83a94d5 | Novelty accumulator P ∈ [0, 3.0]; fires sleep at P≥1.0 (after sal_n≥500, ≥120 eps guard). Dissipated by 0.5×gist_recall_rate per NREM. NDJSON: `wakefulness_pressure`. |
| **S4 — Prior-beta auto-enable** | dfcdd7c | 3-cycle ring buffer of global prior_cosine_mean. Enable β=0.05 when all 3 > 0.65; disable when any < 0.40. NDJSON: `prior_beta_effective`. |

---

## ADR 0024 implementation sequence

| Phase | Signal | Status | Replaces |
|-------|--------|--------|---------|
| 1a | S1 — Adaptive salience gate | **DONE** (039b1a7) | `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD=0.8` |
| 1b | S2 — Settle-rate modulation | **DONE** (880391d) | `--settle-gate 0.85` |
| 2 | S3 — Wakefulness pressure homeostat | **DONE** (83a94d5) | capacity fill + schedule (additive, not replacing) |
| 2b | S4 — Prior-beta auto-enable | **DONE** (dfcdd7c) | `--prior-beta 0.0` |
| — | Growth controller | **DEFERRED** | prerequisite: within-session gist probe |

---

## Open issues / queued next

1. **A/B validate S1–S4**: Run 2s×3c fresh-memory run to confirm:
   - `sal_threshold` moves away from 0.8 cold-start after ~100 eps
   - `settle_gate_dynamic` tracks below `settle_ewma_mean`
   - `wakefulness_pressure` accumulates and triggers sleep (or high_water still fires in warm-up)
   - `prior_beta_effective` remains 0.0 (priors likely < 0.65 at 6 total cycles)
2. **Within-session gist probe**: Filter the REM gist probe to only current-session warm entries. Needed before growth controller can be implemented. Separate from ADR 0024 sequence.
3. **Feature flag defaults** (noted in ADR 0024): `--enable-language`, `--enable-action-decoder`, `--wake-recon-async` should default to `true`. Low priority until pipeline is stable.

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
- S4: enable threshold=0.65 (3 cycles), disable threshold=0.40

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
