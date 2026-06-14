# Session state — somnus

> **Last updated:** 2026-06-14
> **Last commit on main:** ef4682d (items 4+5: grown-units binding exposure)
> **Branch:** `main` (clean)

## Status: All deferred items complete. Queue empty.

---

## Completed this session

| Item | What changed |
|------|-------------|
| ADR scan | All 24 ADRs scanned for pending/deferred items; prioritized P1 and P2 for implementation |
| P1a — auto cold cap | `--max-cold-records 0` → auto-derive from `free_disk × 0.20 / 661`; clamp `[10k, 10M]`; falls back to unlimited if disk lookup returns 0 (btrfs/NFS); emits `max_cold_records_effective` |
| P1b — auto sleep cap | `--max-sleep-secs 0` → rolling 5-cycle window: `max(300, 3 × mean_sleep_secs)`; C1 gets 0 (no cap); emits `sleep_max_secs_effective` |
| S7 — wake-interval throttle | MSE EWMA (α=0.3); stagnation = improvement < 1% for 3 cycles → decrement `effective_wake_interval` toward 1; emits `wake_interval_effective`, `s7_mse_ewma` |
| P2b — cascade-correlation init | `collect_high_residual_inputs()` ranks traces by SAE MSE; `grow_region_seeded()` seeds new encoder cols with normalized high-residual inputs × 0.01; decoder rows zero-init |
| Validation run | 2S×3C fresh memory confirms: P1b cold-start disabled ✅, P1b C2/C3 = 300s ✅, S7 tracking ✅ (no interval change — expected in short run), no crashes, correct line count |
| Disk fallback fix | P1a: when `sysinfo::Disks` can't find mount point (returns 0 bytes), warn and fall back to unlimited (0) instead of clamping to 10k minimum |
| ADR 0024 updated | S-query/S-safety/S-throttle tables updated; amendment section added for P1a/P1b/S7 |
| ADR 0007 updated | `cascade-correlation-training` removed from deferred frontmatter; amendment added |

### Commits
- `7964eb3` — `feat(P1+P2a): auto max-cold-records, auto sleep-max-secs, S7 wake-interval (ADR 0024)`
- `cbd8f63` — `feat(P2b): cascade-correlation growth init (ADR 0007)`
- `0afc38a` — `docs: reconcile items 1+2; ADR 0009 NREM amendment`
- `34fbc3e` — `feat(item3): salience-weighted cold eviction (ADR 0006, 0020)`
- `76e156d` — `feat(item7): salience-weighted gist cluster histogram (ADR 0021)`
- `9c4731f` — `feat(item6): cross-session ingest-source in gist entropy (ADR 0021)`

---

## ADR 0024 implementation sequence

| Phase | Signal | Status | Notes |
|-------|--------|--------|-------|
| 1a | S1 — Adaptive salience gate | **DONE** | |
| 1b | S2 — Settle-rate modulation | **DONE** | |
| 2 | S3 — Wakefulness pressure homeostat | **DONE** | |
| 2b | S4 — Prior-beta scale-up controller | **DONE** (bug fixed) | 0.01→0.05, not toggle |
| prereq | Within-session gist probe | **DONE** | `within_session_gist_rate` in NDJSON |
| 2c | Growth controller | **DONE** | ws→gist→MSE priority; validated 5s×3c; S5C2 false positive eliminated |
| S5 | Startup-queryable priors | **DONE** | `--max-warm-records 0` → auto; `--recall-ef` default 16 |
| S6 | Adaptive wake ceiling | **DONE** | wake_ewma (α=0.3/cycle); max_wake = max(floor, 3×μ); persisted |
| P1a | Auto cold cap | **DONE** | `--max-cold-records 0` → disk-derived; fallback=unlimited |
| P1b | Auto sleep cap | **DONE** | rolling 5-cycle window; floor 300s |
| S7 | Wake-interval throttle | **DONE** | MSE stagnation → decrement interval toward 1 |

---

## Open issues / queued next

Deferred items from ADR scan (Tier 1 + Tier 2, in priority order):

| # | Item | Source ADR | Status |
|---|------|-----------|--------|
| ~~1~~ | ~~Growth controller~~ | ADR 0024 | **DONE** — wired to `within_session_gist_rate` (e30916c) |
| ~~2~~ | ~~NREM output-head training~~ | ADR 0009 | **DONE** — `fused_replay_wake_step_batch` uses stored episode vectors; no k-WTA re-run needed (a1c43cb / cd4364e). ADR 0009 amended. |
| ~~3~~ | ~~Salience-weighted cold eviction~~ | ADR 0006, 0020 | **DONE** — `salience: f32` added to `EpisodeRecord`; SMNCOLD3 (665B/record); `evict_lowest_salience()` in sleep.rs. |
| 4 | **Grown-units binding exposure** — allow grown units above N_UNITS=1024 into the binding area (currently clamped by `to_dense_n`) | ADR 0005, 0008 | Multi-commit subsystem change — see notes below. |
| 5 | **Per-region projection resize** — resize per-region linear projections when regions grow | ADR 0008 | Blocked by item 4. |
| ~~6~~ | ~~Cross-session ingest-source through promotion~~ | ADR 0021 | **DONE** — `sample_records()` on warm store; bounded warm sample binned into `cluster_sources` after hot-tier loop (9c4731f). |
| ~~7~~ | ~~Salience-weighted cluster assignment~~ | ADR 0021 | **DONE** — `cluster_sources` changed to `Vec<[f32;6]>`; weight = `salience.clamp(0.01,2.0)`; sentinel f32::MAX → 1.0 (76e156d). |
| ~~4~~ | ~~Grown-units binding exposure~~ | ADR 0005, 0008 | **DONE** — `region_n:[usize;5]` skip field in BindingArea; `forward()` uses `region_n[i]` not N_UNITS; `grow_region`/`grow_region_seeded`/`prune_region` now resize binding projection in sync; `load_weights` reconstructs `region_n` from weight dims (ef4682d). |
| ~~5~~ | ~~Per-region projection resize~~ | ADR 0008 | **DONE** — implemented with item 4 above (ef4682d). |

### Item 4+5 scope note (resolved)

Items 4+5 are a multi-commit subsystem change, **not** a "one functional commit" unit:
- `try_bind` in `binding.rs` uses `act.to_dense_n(N_UNITS)` — clamps every region at 1024 regardless of actual grown size
- `binding.rs` projections are `LinearConfig::new(N_UNITS, D_PROJ)` fixed-size
- ORT path (`onnx_session.rs`) builds `Array2::from_shape_vec((1, N_UNITS), …)` and feeds fixed `[1, N_UNITS]` tensors to a compiled graph — changing region width means re-exporting the ONNX model under MIGraphX constraints
- Snapshot format bump (VarMap `[N_UNITS, D_PROJ]` projection weights) + optimizer-state migration for grown rows
- Prerequisite: regions actually need to exceed 1024 in practice — to be verified before spending time on this

**Resolved:** confirmed via campaign logs — regions reached up to 1920 units. Items 4+5 implemented in ef4682d.

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
- wake_reconstruct_interval=5 (default); S7 decrements toward 1 on MSE stagnation
- SAL_EWMA_COLD_START=0.8, α=0.01 for S1/S2 EWMAs
- S3: P_THRESHOLD=1.0, P_MAX=3.0, P_COLD_START_N=500, P_MIN_EPISODES=120
- S4: base=0.01 (default), scale target=0.05, enable threshold=0.65 (3 cycles), disable threshold=0.40
- `PRIOR_BETA=0.01` is now the campaign default
- S7: S7_MSE_ALPHA=0.3, stagnation threshold=1% improvement, consecutive=3, decrement step=1
- COLD_RECORD_BYTES=665 (8+8+4+640+1+4); SMNCOLD3; P1a uses 20% of free disk; salience stored per record

---

## Performance knobs

| Flag | Default | Effect |
|---|---|---|
| `--settle-steps 0` | 0 | Disable settle entirely |
| `--settle-every N` | 1 | Subsample settle |
| `--settle-gate F` | 0.85 | Cold-start prior for S2 adaptive gate |
| `--recall-ef N` | 16 | Lower → faster HNSW queries |
| `--wake-recon-async` | true | Offload wake_reconstruct to worker |
| `--wake-reconstruct-interval N` | 5 | S7 adapts this downward on MSE stagnation |
| `--video-stride N` | 1 | Encode every Nth video frame |

---

## MIGraphX constraint (hard)

Never expose ScatterElements or INT64 outputs. Two-graph split, topk/masking on CPU.
