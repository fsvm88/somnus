# Session state — somnus

> **Last updated:** 2026-05-29
> **Last commit on main:** a1540bc — Cap ONNX intra/inter-op threads dynamically
> **Branch:** `main` (uncommitted changes — see below)

## Status: Campaign running + wake optimizations implemented (uncommitted)

---

## campaign_20260526_231637 — COMPLETED (sessions 1–10)

Sessions 1–8 complete as of this session. Session 9 in progress when last checked, session 10 pending.

Parameters: `SESSIONS=10 CYCLES=5 CAPACITY=8192 --lang-oversample 9 --asr-min-chars 5 --replay-batch-size 128`

Trend: `warm_recall_l2` converges cleanly each session (1.8→0.18 by cycle 5). `mean_loss` declining across sessions. All good.

---

## Wake phase bottleneck investigation — DONE

Profiled session 7 log (52.5 min total):
- **Real bottleneck: `wake_reconstruct_step` backward+optimizer** — 315ms median per frame, ~4190 frames/session ≈ 22 min/session.
- Settle iter cost: only 7.8ms median (misread in initial bucket analysis due to multi-thread log interleaving).
- NREM: only 1.9 min/session — not worth optimizing.

---

## Wake optimizations — IMPLEMENTED, NOT COMMITTED

### Change 1: Parallelize 5 output heads with rayon (Task #1)
**Files:** `crates/somnus-perception/src/lib.rs`, `crates/somnus-perception/Cargo.toml`, `Cargo.toml`  
**What:** `settle_episode` and `update_prior_predictions` now use `rayon::par_iter` for the 5 head forward passes and 5 SAE forward_no_dc passes.  
**Expected impact: LOW** — settle is only 7ms per iter × 3267 iters ≈ 16s per session. Advisor flagged this as likely a no-op or regression vs. burn-ndarray's existing multi-thread pool. Should benchmark before shipping, potentially revert.

### Change 2: `--wake-reconstruct-interval N` (Task #2)
**Files:** `crates/somnus/src/main.rs`  
**What:** New CLI arg (default=1 = original behavior). At N=4, runs backward+optimizer only every 4th episode instead of every frame. Counter resets each sleep cycle.  
**Expected impact: HIGH** — at N=4, saves ~75% of 22 min/session = ~16 min → session from ~42 min to ~26 min.  
**Caveat:** Changes learning dynamics (fewer head gradient updates). `mean_output_mse_wake` will rise. Don't compare pre/post campaigns directly on that metric.

---

## Next queued work

1. **Stop current campaign once done** (it will finish on its own; no action needed now).
2. **Commit the two changes** — run `cargo test` first, then commit.
3. **Benchmark Task #1** to verify rayon actually helps:
   - `cargo run --release -p somnus-perception --bin bench_nrem_batch` exists but tests NREM, not settle.
   - Add a tiny bench that times `settle_episode` serial vs parallel, or just time a short session.
   - If flat/negative, revert the rayon parallelization and remove the dep.
4. **Run a new campaign with the new binary** at `--wake-reconstruct-interval 4` to measure speed/quality trade-off.
5. **Task #3 (future):** Port `PerceptionModel` to `burn-wgpu` backend to GPU-accelerate the backward pass. This targets the 315ms/frame `wake_reconstruct_step`. Write ADR first — wgpu+MIGraphX compete for the same GPU (RDNA4/gfx1201). Experiment in `experiments/` first.

---

## Key facts

- `--replay-batch-size 128` is hardcoded in `train_campaign.sh` — do NOT pass it in `EXTRA_ARGS`
- `burn-ndarray` features need listing in each crate's `Cargo.toml` individually
- `--asr-min-chars 5` needed for noisy-audio clips (default 20 too high)
- Dynamic ONNX thread cap: on 32-thread machine → intra=16, inter=8
- `wake_reconstruct_step` backward pass: 315ms median, 712ms mean (heavy outliers)
- Real settle cost: 7.8ms median per iter — burn-ndarray's own multi-thread already handles it
