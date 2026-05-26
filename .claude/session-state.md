# Session state — somnus

> **Last updated:** 2026-05-26
> **Last commit on main:** (pending commit — NREM speed optimizations)
> **Branch:** `main`

## Status: NREM speed optimizations implemented

### campaign_no_walks — COMPLETED (15 sessions × 3 cycles)
- Memory dir: `memory/gpu_campaign_mem3`
- Final metrics: output_mse≈0.40, warm_recall_l2≈0.84, gist_recall_rate≈0.61
- **No language traces generated** (decoder_trainable=0 in last sessions — ASR likely not triggering at 20-char threshold on these clips)

### NREM speed investigation + fixes (this session)

#### Root cause of slow NREM (21 tr/s in campaign_no_walks)
1. `burn-ndarray` features fix (prev session): fixed the primary bottleneck; raised peak to ~168 tr/s
2. **`projection_step` per-trace sequential backward**: each call does a full AdamW step on the 640→4480 (2.87M param) projection MLP. At ~10% lang density (campaign_no_walks), that's 6.4 calls × ~427ms = 2.7s wasted per batch of 64, vs 299ms for replay+wake combined.

#### Fixes applied (uncommitted)
1. **`projection_step_batch`** added to `crates/somnus/src/action.rs`: batches all lang-tagged traces in a replay batch into one forward+backward+optimizer call instead of N sequential calls. Identical training semantics (mini-batch SGD instead of per-sample SGD; with N=6-12, this is negligible difference).
2. **Sleep.rs updated**: NREM loop now calls `dec.projection_step_batch(&proj_pairs)` per batch instead of per-trace loop.
3. **Per-phase timing instrumentation** added to `run_sleep_cycle` in `sleep.rs`: logs `replay_ms_avg`, `wake_ms_avg`, `proj_ms_avg`, `total_ms` at end of each NREM cycle.
4. **Default `replay_batch_size` bumped 64→128** in `main.rs`: bench shows 55% more throughput; production test showed 14% improvement (168→239 tr/s without projection).

#### Measured throughput (no projection, talking_head.mp4, 1638 traces)
- batch=64: replay=168ms, wake=131ms, total=7804ms → **210 tr/s**
- batch=128: replay=289ms, wake=238ms, total=6860ms → **239 tr/s**

#### Projected throughput with projection (estimate, 10% lang density, batch=128)
- Old (per-trace): 21 tr/s
- New (batched, N=12.8 per batch): estimated 130-160 tr/s

### Build status (current, uncommitted)
- `cargo build --release` — clean
- `cargo clippy -p somnus -p somnus-perception -- -D warnings` — clean

## Next queued work
1. **Commit the NREM speed changes** (projection_step_batch + timing + batch_size=128)
2. **Run new campaign** with optimized binary to verify projection speedup with real lang traces
   - Use `--lang-oversample 9` and `--asr-min-chars 5` to ensure lang traces are generated
   - Watch for `NREM phase timings` log: `proj_ms_avg` should drop dramatically
3. **Phase 9b** (mic/webcam live capture) — pending
4. **Phase 11b** (creative REM) — pending

## Key facts
- `burn-ndarray` features need to be listed in each crate's `Cargo.toml` (workspace dep is just a template)
- Both `crates/somnus/Cargo.toml` and `crates/somnus-perception/Cargo.toml` now have `burn-ndarray = { workspace = true }`
- `projection_step` (per-trace) kept with `#[allow(dead_code)]` — useful for debugging
- No region growth occurred in campaign_no_walks (gist thresholds not triggered)
