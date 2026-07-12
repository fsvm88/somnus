# Session state — somnus

> **Last updated:** 2026-07-12 (later still)
> **🔄 IN PROGRESS: full 10x growth campaign running unattended (restarted twice, see below).**
> `scripts/campaign_status.py` is the fast path back into context — run it first on resume, before
> reading anything else in this file. If it's still running, no action needed. If finished, verify
> final region sizes vs. the 10x target, check for plateau/regression in `per_region_mse`, write up
> the result as a new `gaps.md` section, and commit.
>
> **If the campaign has been stopped** (crash, reboot, manual shutdown) and needs restarting:
> `scripts/resume_growth_campaign.sh` — self-contained, reads `SESSION_START`/`SESSIONS`/`CYCLES`/
> `CLIPS`/etc. from the campaign's own `params.txt` + `metrics.jsonl` (not from this file or any
> session-specific state), computes the correct `SESSION_START`/`SESSIONS` to resume exactly where
> it left off, rebuilds `somnus` first by default (`SKIP_BUILD=1` to skip), and refuses to run if a
> `train_campaign.sh` is already active. `DRY_RUN=1` to preview the computed invocation without
> launching. To stop the campaign cleanly at any point: `scripts/stop_graceful.sh` (stops the
> wrapper, then SIGTERMs `somnus` and waits up to 60s for a clean exit before SIGKILL; `--dry-run` to
> preview, `--wait N` to change the timeout — state is snapshotted every cycle, so this loses at
> most a few seconds).
>
> **What's live:** §17/§18/§19 (K_ACTIVE scaling, optimizer-reset scoping, novelty-proportional
> batch), §21 (fixed a warm-store HNSW dump leak — was leaking ~24-26MB/cycle unboundedly, also
> silently broke cross-restart index reload), and §22/ADR 0037 (ORT/MIGraphX perception inference
> now auto-enables past 1,230 units/region instead of requiring a flag — confirmed live,
> `onnx=active` in `campaign_status.py`, ~2.1× faster perception step measured). `memory/` was
> wiped completely at campaign start per the user's explicit call (all production state + old
> A/B/pilot scratch dirs, ~17GB, gitignored/untracked, nothing lost that wasn't reproducible) —
> regrowing from scratch on the current mechanism. `CAMPAIGN_DIR=memory/campaign_growth10x`
> throughout; `CLIPS`=full 145-file h3h3+shorts+podcasts corpus (`.mp4` only). Launched via
> `scripts/train_campaign.sh` in the background each time — task IDs don't survive a session
> restart; use `pgrep -fa train_campaign.sh` or `campaign_status.py` instead.
>
> **Restart 1** (~cycle 486, gaps.md §21): `SemanticStore::dump()` leaked a full HNSW index dump
> every sleep cycle forever (461 orphaned pairs, 11G) because `load_warm()`'s reload path
> permanently disables `hnsw_rs`'s in-place overwrite; fixed by clearing stale dumps before
> writing, which also restores the plain-name path `load_warm()` needs to find anything on
> restart. Cleaned up the leak (`memory/warm/` 11G → 192M), resumed `SESSION_START=50 SESSIONS=82`,
> metrics/logs continuous across the restart (486→487, no gap).
>
> **Restart 2** (~cycle 542, gaps.md §22/ADR 0037): user asked whether `onnx=off` had a runtime
> cost — pulling real numbers from `metrics.jsonl` showed `perception_total_ms` climbing steadily
> (9.1s→16.1s as regions grew 1,986→6,043 units), already over half the cycle's wall-clock, and
> past ADR 0023's documented CPU/GPU crossover (~1,230 units/region). Implemented auto-enable
> (`should_use_ort_inference`, unit-tested) instead of requiring the old `--onnx-models-dir` flag.
> This surfaced two pre-existing, never-before-exercised bugs in `scripts/export_perception_graph.py`
> (stale hardcoded `LANG_HIDDEN=896` since ADR 0036 made it 1024, and hardcoded `N_UNITS`/`K_ACTIVE`
> that could never survive region growth) — fixed by deriving all dims from the actual exported
> weight shapes (`infer_dims()`) instead of a second hand-maintained copy. Verified against a copy
> of the live campaign's real grown weights (6,397 units/region) before touching the live one:
> MIGraphX loaded successfully, `perception_total_ms` 7,735 vs. the CPU baseline's ~16,051 at
> comparable scale — **~2.1× faster**, matching ADR 0023's predicted figure. Resumed
> `SESSION_START=56 SESSIONS=76`; log confirmed `ORT binding session active` on the real process.
> User also proposed eliminating the Python export step entirely (native Rust ONNX graph
> construction) — sound long-term direction, scoped as gaps.md §23, deliberately not started (no
> mature Rust ONNX-graph-builder crate exists; real rewrite effort, not a fast-follow).
>
> Full backstory: `ai-docs/gaps.md` §16/§19/§20/§21/§22/§23, ADR 0037.

## Historical (superseded by the in-progress campaign above, kept for context)

> **Last updated:** 2026-07-11
> **Branch:** `main`. Committed (through 2026-07-09): change-gated video encoder (`9bba42b`),
> SVD null-result (`b410ad7`), train_campaign.sh bugfix + replay-ratio A/B (`d5ce2ab`), N=4
> replay-ratio default (`f0fa9c6`), items queued (`709442b`), DER++ auto-λ implementation
> (`f2fd185`, gaps.md §3), DER++ auto-λ A/B reject (`72d1503`), EfficientAT-mn10 implementation +
> A/B result (gaps.md §9), M2D-CLAP scoped and not pursued (gaps.md §8, docs-only).
> **Not yet committed (2026-07-11 session):** `scripts/download_podcasts.sh` (new — pulls diverse
> podcast episodes into `clips/podcasts/`, 56 episodes/5.6GB across 10 shows), growth-pilot 2
> results (expanded-corpus characterization, `memory/growth_pilot_expanded/` — scratch, untracked),
> and a documentation audit/correction pass across `architecture.md`, `gaps.md`,
> `growth-mechanisms.md`, `roadmap.md`, `README.md`, and this file (stale open-questions,
> priority-table numbering drift, a lost deferred-items trigger table, missing Phase 14 section).
> **Status: gaps.md §16 (growth-scaling) CHARACTERIZED — three 75-cycle pilots (10-clip loop, then
> 145-clip diverse corpus, then a post-fix rerun of the same 145-clip corpus) all show gate-limited
> growth cadence, no memorization-plateau risk.**
> **Also done (2026-07-12, not yet committed): three architectural fixes found during pre-campaign
> review, gaps.md §17/§18/§19 — K_ACTIVE now scales with region growth, optimizer reset scoped to
> the grown region only (not the whole model), and growth batch size is now novelty-proportional
> instead of a flat constant. Validated with a third growth pilot (`memory/growth_pilot_v2/`,
> untracked) — mechanism confirmed working (k_active tracked region size correctly; batches varied
> 17–77 with cadence unchanged from pilot 1/2). But that pilot revealed the average batch (35.6/
> event) is well below the old flat 64, because most triggers in this corpus land at moderate, not
> severe, novelty — so **the full 10x campaign is now re-estimated at ~9.3–11.5h (was ~3.9h),
> ~2.4-3x longer than the pre-§19 number.** Full writeup: `ai-docs/gaps.md` §16 pilot 3, §19.**

## Just completed (2026-07-11) — expanded-corpus growth pilot + podcast corpus + doc audit

Continuation of the growth-scaling pilot below: user wanted the memorization-plateau caveat closed
before committing to the full 10x campaign, which meant a more diverse corpus than pilot 1's
10-clip loop. Added `scripts/download_podcasts.sh` (new, modeled on `download_h3h3.sh`) — pulls
episodes from 10 distinct podcast shows (Lex Fridman, Joe Rogan, Huberman Lab, Tim Ferriss, Diary
of a CEO, Armchair Expert, Conan O'Brien, This American Life, The Daily, Freakonomics Radio),
10–90 min duration filter, `N_PER`/`M_SEARCH` configurable. Ran twice (1 episode/show, then 10x
more): landed 56 episodes / 5.6GB in `clips/podcasts/` (Lex Fridman/Huberman Lab/Tim Ferriss
under-represented — their typical episode length exceeds the 90-min filter ceiling on most search
candidates, not a download failure).

Ran pilot 2: identical protocol to pilot 1 (`SESSIONS=15 CYCLES=5 CAPACITY=8192`, 75 cycles) against
the combined 145-clip corpus (h3h3 + shorts + podcasts), isolated `memory/growth_pilot_expanded/`.
**Result: no plateau — cadence and quality trend indistinguishable from pilot 1** (16 growth
events, same gate-limited cadence, `per_region_mse` improved ~0.0135→~0.0039 first-10-vs-last-10
cycles, final region sizes uniformly `[2048]×5`). Confirms growth is gate-limited (bounded by the
3-cycle cooldown), not corpus/signal-limited — the memorization-plateau risk flagged after pilot 1
did not materialize. Full numbers: `ai-docs/gaps.md` §16, `ai-docs/research/growth-mechanisms.md`
"Empirical validation" section (both written up this session).

Then did a full documentation audit (user request: "recheck all docs and make sure they are
correct") — delegated a read-only staleness sweep to a subagent, applied its findings directly:
- `architecture.md`: "Growth mechanism" and "Compression strategy" open questions were stale (both
  long since resolved — ADR 0024 §2c and INT8+PQ respectively); struck through with references.
  "Sleep-cycle time budget" narrowed (ADR 0024's S1–S4 controllers resolved most of what it asked).
  Implementation-state paragraph updated with everything since 2026-07-05.
- `gaps.md`: priority table's `#` column had drifted from the actual `##` section numbers since
  around item 7 (table said "❌7 DER++" but that's actually §3; "10 SigLIP2" is actually §7; etc.) —
  renumbered to match, added missing §10/§12 rows, added new §16 row for the growth pilots.
- `.claude/session-state.md` (this file): the "Deferred backlog" section had decayed into a
  circular pointer to `CLAUDE.md`, which itself points back here — the actual trigger-condition
  table had been silently pruned in an earlier edit. Restored from git history (commit `03eb9b5`).
  Flagged L/M (LoRA fine-tuning, MLP projection) as unverified against Qwen3 — they were marked
  done for Qwen2.5, which was fully removed by ADR 0036; not re-checked here.
- `roadmap.md`: had no Phase 14 section despite `architecture.md`/`gaps.md` referencing "Phase 14
  (GPU unification)" as complete. Added it (ADR 0023, GPU unification via ORT/MIGraphX).
- `README.md`: "Living archives" ADR callout list was missing ADR 0024 (large, multi-phase) and
  ADR 0036 (Qwen3 adoption) — added both.

Nothing here required code changes — pure documentation correction plus the two new artifacts
(podcast corpus, pilot 2 metrics). Not yet committed.

## Just completed (2026-07-09, later) — growth-rate pilot campaign (10x growable-core scaling)

User asked to scale somnus's own growable core (region k-WTA units + GRU + binding, currently
~5,400 units total, ~480,000× smaller than the human brain / roughly fruit-fly order of
magnitude) 10x, toward honeybee-order scale (~54,000 units) rather than mouse (~71M neurons,
a ~400x jump — corrected an earlier miscalibration in-conversation). Explicitly *not* the
frozen-encoder scale-up (Qwen3/Whisper/DINOv3/CLAP) — those are "easy, do later."

Growth mechanism (cascade-correlation-style, `grow_region_seeded`, ADR 0024 §2c
`within_session_gist_rate` gate, 3-cycle cooldown, `GROWTH_BATCH=64`) was already fully
implemented and live — no new code needed. Ran a 75-cycle characterization pilot
(`SESSIONS=15 CYCLES=5 CAPACITY=8192`, h3h3/shorts corpus, isolated `memory/growth_pilot/`,
not the main `memory/`) purely to measure real growth cadence, since nobody had run enough
cycles to know it empirically.

**Result:** 16 growth events in 75 cycles (~1 event per 4.7 cycles, close to the 3-cycle
cooldown floor — growth is essentially gate-limited, not signal-limited). Final region sizes
`[2048, 1984, 1984, 1984, 1984]` (region 0 grown 2×, others ~1.94× from the n=1024 baseline).
No plateau observed over the full pilot — one quiet session (s9) was immediately followed by
resumed growth at s10c1. `per_region_mse` *improved* alongside growth (e.g. region 0:
0.127→0.007–0.017), no quality regression from growing. Avg 21.8s/cycle wall-clock (dominated
by short pressure-triggered ~120-episode cycles after the first buffer-fill cycle); 1.7GB disk
for the whole pilot (816GB free on the volume — no capacity concern).

**Extrapolation to 10x (~9,660 units/region, ~135 growth events needed):** at the observed
rate, ~633 cycles ≈ **~3.9 hours wall-clock** for a full campaign — far more tractable than
the pessimistic first estimate (hundreds of cycles was right, but wall-clock per cycle is much
cheaper than the growth-mechanisms.md research note's aspirational cascade-correlation
estimate, since the actual implementation is a weight-tensor resize, not a candidate-training
loop). Caveats not yet tested at this scale: (1) whether the 10-clip looping corpus causes a
memorization plateau past 75 cycles (no sign of it yet, but untested); (2) whether per-cycle
compute cost grows non-trivially as region matrices get ~10× larger (linear in n_units, likely
modest but unverified).

**Not yet done:** the actual full-scale campaign to reach 10x. Discussed with user but not
launched — next step is user-driven (scope: same corpus repeated vs. expanded corpus to
pressure-test the memorization-plateau risk before committing ~4h of unattended compute).
`memory/growth_pilot/` is scratch pilot output, not committed/tracked.

## Just completed (2026-07-09) — M2D-CLAP (gaps.md item 8) — ⏸️ NOT PURSUED, no code written

User asked to try a deferred item; picked M2D-CLAP (item 8) — the more credible audio-encoder
upgrade candidate vs. the already-rejected mn10. Before implementing, ran feasibility research
(no code): weights are available with no gate, but (a) no ONNX export precedent exists anywhere
for M2D or M2D-CLAP — real risk of hitting a dead end at export, driven by dynamic
positional-embedding interpolation and an `nnAudio` STFT-based mel frontend (the same STFT-op
fragility class mn10 was chosen specifically to avoid), and (b) the license is NTT's custom
non-commercial "Evaluation" agreement, not MIT/Apache like everything else in the tree.

Weighed against that risk: the paper's +3 mAP AudioSet-tagging claim is a modest number on a
different task than somnus's actual metric (`per_region_mse` through somnus's own pipeline), and
item 9's A/B is a concrete precedent in this exact slot that paper-benchmark quality and
somnus-pipeline quality diverge (mn10 had no quality claim at all and still regressed MSE +155.6%
while a benchmark-adjacent recall metric moved the other way). User's call: **not worth the
implementation/export/license risk for the expected effect size. Documented in `gaps.md` §8,
no code written, nothing to revert.**

## Just completed (2026-07-09) — EfficientAT mn10 (gaps.md item 9) — ❌ REJECTED

Resumed the A/B campaign left mid-run at the end of the previous session (implementation was
already complete and smoke-tested). Ran clean (`rm -rf memory/ab_efficientat_mn10` first, then
`bash scripts/ab_efficientat_mn10.sh`, 5 sessions × 2 cycles per arm, fresh memory both arms,
h3h3/shorts corpus).

**Result: FAIL.** Verdict is based on the mean MSE of the four regions with identical
architecture in both arms (speech/visual/symbolic — region-1/env excluded, see script header
for why CLAP vs. mn10 env-MSE isn't a valid cross-arm comparison): control (CLAP-HTSAT) avg
0.011966 vs. treatment (mn10) avg 0.030588, **+155.6%, worse**, regressed in 9/10 cycles, no
reversals. `warm_recall_l2`/`gist_recall_rate` moved in mn10's favor but don't override a 2.5×
MSE regression on the shared representation — likely a scale-distortion artifact rather than a
genuine improvement.

**Decision:** reject: default env encoder stays CLAP-HTSAT. Implementation kept as an opt-in,
off-by-default Cargo feature (`mn10_env`) rather than deleted — it's clean and parity-tested,
and may be worth revisiting for a different reason (e.g. deployment-size constraints) later.
Full writeup: `ai-docs/gaps.md` §9. Build/clippy/test pass clean both with and without the
feature; committed together with the implementation in one shot (was never committed before
since the A/B hadn't concluded).

## Just completed (2026-07-12) — three growth-mechanism architectural fixes (gaps.md §17/§18/§19)

Found and fixed in order, while reviewing the growth mechanism ahead of the full 10x campaign.
All implemented, tested, and documented this session; **not yet committed**.

- **`gaps.md` §17 — K_ACTIVE scaling with region growth.** `K_ACTIVE=20` was a hardcoded const
  never adjusted by `grow_region`/`grow_region_seeded`/`prune_region` — sparsity ratio would have
  fallen ~10x (2%→0.2%) over the full 10x campaign. Fixed: `KWtaLayer` gained a per-layer
  `k_active` field, new `region::scaled_k(n_units)` preserves the baseline ratio linearly, wired
  into all three growth/prune call sites and all k-WTA selection code paths. Also fixed four
  ORT-bootstrap placeholder call sites and `binding_only` that hardcoded `N_UNITS`/`K_ACTIVE`
  instead of the layer's live size (latent bugs for an already-grown region after snapshot
  reload). New tests: `scaled_k_preserves_baseline_ratio`, `kwta_respects_grown_k_active`.
- **`gaps.md` §18 — scope optimizer reset to the grown region only.** Every growth/prune event
  was calling `self.optimizer = AdamWConfig::new().init(...)`, wiping Adam momentum for the
  *entire* model, not just the touched region. Investigated burn-optim's source
  (`OptimizerAdaptor`, `HashMap<ParamId, AdaptorRecord>`) before changing anything: state is
  already keyed by `ParamId`, and the resized tensors already get fresh `ParamId::new()` — so the
  full reset was pure waste. Fix: deleted the reset line from all three call sites; nothing else
  needed to change. Confirmed unrelated to (and doesn't regress) the separate
  `refresh_autodiff_state` mechanism, which resets the optimizer for a different, legitimate
  reason (NREM graph-accumulation slowdown) and is untouched.
- **`gaps.md` §19 — novelty-proportional growth batch size** (user's proposal, judged sound).
  Flat `GROWTH_BATCH=64` replaced with `GROWTH_BATCH_MIN=16`/`GROWTH_BATCH_MAX=128` and three
  pure functions (`gist_novelty`, `mse_novelty`, `growth_batch_size`) that scale the batch by how
  far the trigger signal missed its threshold — a region deep in recall failure now grows more
  per event than one that barely tripped the gate, instead of every triggered region getting the
  same fixed lump regardless of severity. New tests: `growth_batch_size_spans_min_to_max`,
  `growth_batch_size_clamps_out_of_range_novelty`, `gist_novelty_zero_at_threshold_...`,
  `mse_novelty_zero_at_threshold_...`, `worse_recall_failure_grows_a_bigger_batch`.

All three: build/clippy/fmt clean, full `somnus`/`somnus-perception` test suite passes (including
`phase5_smoke.rs`'s `grow_then_bind_produces_finite_episode`, `grow_then_replay_finite`,
`prune_region_reduces_grown_units`). Full writeups: `ai-docs/gaps.md` §17–19.

## Just completed (2026-07-12, later) — growth pilot 3/"v2": validated §17/§18/§19 live, recalibrated 10x campaign estimate

Reran the exact pilot 2 protocol (`SESSIONS=15 CYCLES=5 CAPACITY=8192`, same 145-clip h3h3+shorts+
podcasts corpus) against the post-fix build, fresh isolated `memory/growth_pilot_v2/` (untracked).
75 cycles, 39m49s wall-clock, 5 benign MIGraphX-exit-race segfaults (same known issue, auto-
recovered by `train_campaign.sh` as always).

**Mechanism confirmed working correctly:** 14 growth events, same 4–7 cycle gate-limited cadence
as pilot 1/2; `per_region_mse` improved 0.0136→0.0048 (comparable to pilot 2's 0.0135→0.0039);
`k_active` tracked region growth as designed (final `region_sizes=[1523]×5`, `scaled_k(1523)=30`
vs. baseline `scaled_k(1024)=20`, ratio preserved); batch sizes varied sensibly with novelty
(17–77 across observed novelty 0.01–0.54).

**But: this surfaced an important recalibration.** Mean batch across the 14 events was 35.6 —
nearly half the old flat 64 — because most triggers in this corpus land at moderate, not severe,
novelty. Total capacity added over 75 cycles (499 units/region) was noticeably less than the old
flat-batch mechanism would have produced for the same 14 events (896 units/region). Extrapolating
this pilot's observed mean batch (35.6/event) and cadence (1/5.36 cycles) to the 10x target
(~9,660 units/region) now projects **~242 growth events, ~1,298 cycles, ~9.3–11.5h wall-clock** —
roughly **2.4-3x longer than the pre-§19 estimate of ~3.9h/~633 cycles.** This isn't a regression;
it's §19 correctly making growth proportional to actual severity, in a setup where severity is
usually moderate. Full numbers: `ai-docs/gaps.md` §16 (pilot 3) and §19 (validation note).

**Not yet committed:** all of §17/§18/§19's code changes plus this pilot's docs updates.

**Next step is user-driven:** commit these changes, then either accept the longer (~9.3–11.5h)
campaign time and launch it, or revisit §19's `GROWTH_BATCH_MAX`/curve shape first if ~3-4h is a
hard time budget.

## Historical: EfficientAT mn10 implementation notes (2026-07-08, now committed and A/B'd — see above)

**What was implemented:**
- `crates/somnus-encoding/src/preprocess/mel.rs`: new `Mn10Mel` — EfficientAT's `AugmentMelSTFT`
  mel frontend (pre-emphasis, centered/padded Hann window, Kaldi-style mel filterbank —
  genuinely different triangle construction from the existing HTK/Slaney code, not a
  parameter variant, see `kaldi_mel_filterbank_f64`'s doc comment) ported to Rust and
  parity-tested against the PyTorch reference (`crates/somnus-encoding/tests/mel_parity.rs`,
  `mn10_zeros_parity`/`mn10_sine440_parity`, both pass at 1e-4 tolerance). Fixture files in
  `crates/somnus-encoding/tests/fixtures/mn10_*` are new, untracked, need `git add`.
- `scripts/export_efficientat.py`: exports the MobileNetV3 backbone only (no mel frontend —
  deliberately, baking `AugmentMelSTFT` in works numerically but reintroduces an STFT op,
  defeating mn10's whole MIGraphX-compat rationale; see script docstring). Verified
  numerically against PyTorch (rel L2 ~9e-7) and confirmed zero STFT/DFT ops in the graph.
  Needs its own venv (`scripts/.venv-efficientat/`, gitignored, torch 2.13/torchvision 0.28 —
  incompatible with `scripts/.venv`'s pinned torch 2.11.0, so it's isolated rather than
  risking the shared venv). Exported model already on disk:
  `models/efficientat_mn10_encoder{,_fp16}.onnx` (gitignored, not committed, regenerable via
  `scripts/.venv-efficientat/bin/python scripts/export_efficientat.py &&
  scripts/.venv-efficientat/bin/python scripts/convert_fp16.py`).
- New Cargo feature `mn10_env` (in `somnus-encoding`, `somnus-perception`, forwarded by `somnus`):
  swaps `REGION_DIMS[1]` 512→960, `AudioEncoder`'s env session/mel/sample-rate (CLAP 48kHz
  vs mn10 32kHz), and all the cross-modal-concat offset math in
  `crates/somnus-perception/src/lib.rs` (now derived from `ENV_DIM`/`CM_OFF_*` consts instead of
  hardcoded 512/896/1280 literals). Builds, tests, and clippy pass clean **both** with and
  without the feature. End-to-end smoke-tested manually (single session, small capacity,
  `--features mn10_env` binary) — ran a full sleep cycle with `env embedding dim=960`, no
  crashes, log confirms correct dim.
- `scripts/train_campaign.sh` gained an additive `SKIP_BUILD` env var (default `0`, no effect
  on any other script) so the A/B script can build each arm itself (with/without
  `--features mn10_env`) without the campaign script's own unconditional `cargo build`
  clobbering it.
- `scripts/ab_efficientat_mn10.sh`: the A/B harness. **Important asymmetry vs. every prior
  A/B in this repo:** region-1 (env) MSE is NOT a valid comparison between arms — CLAP and
  mn10 have different target distributions (contrastive/unit-norm vs. raw pooled-conv
  log-mel-scale), so `per_region_mse[1]` moving is a scale artifact, not a quality signal.
  The verdict is computed from the mean of the OTHER four regions' MSE (indices 0,2,3,4 —
  identical architecture both arms, genuinely comparable) plus `warm_recall_l2` /
  `gist_recall_rate`. Region-1 MSE is still printed for reference only.

## Just completed (2026-07-08)

Ran the A/B that gaps.md §3 flagged as its own next step: `--der-auto-lambda` (EWMA
1/variance weighting) vs. default unweighted (λ=1) sum in `fused_replay_wake_step_batch`. New
script `scripts/ab_der_auto_lambda.sh` (mirrors `ab_replay_ratio.sh`: 5 sessions × 2 cycles, fresh
memory per arm, h3h3/shorts corpus, control then treatment). **Result: FAIL.** `mean_output_mse`
control avg 0.0487 vs. treatment avg 0.0847 (**+73.8%**, worse), regressed in all 10/10 cycles
(range +16% to +937%), no reversals — decisive, not noise. Likely cause: EWMA variance estimate
too small/noisy early, causing large `1/variance` weight spikes that destabilize training rather
than damp a noisy term. Decision: do not adopt; `--der-auto-lambda` remains opt-in, defaults to
`false`, no code changes needed. Full writeup: `ai-docs/gaps.md` §3.

Note for next session: while the A/B ran, a first attempt (backgrounded via `nohup ... &`) didn't
actually die when its launching shell exited as expected — it kept running and collided with a
second, cleanly-restarted attempt writing to the same `memory/ab_der_auto_lambda/control` path
before the stale one was caught and killed. The result above is from the clean single-process run
only (verified no PID collision before trusting the numbers). Lesson: prefer the Bash tool's own
`run_in_background` over manual `nohup ... &`, and always verify only one process is writing to a
given output path before trusting metrics from a backgrounded training run.

## Just completed (2026-07-06)

Both action items from `ai-docs/research/brain-scale-comparison.md`'s "Summary judgment" are done
(`9bba42b` change-gated video encoder, `b410ad7` SVD null result). High replay ratio (item 2)
validated and adopted at N=4 (`d5ce2ab`, `f0fa9c6`) — see `ai-docs/gaps.md` §2 for the N=1/2/4
comparison table.

**gaps.md §3, DER++ auto-λ Phase 3 — implemented and committed (`f2fd185`).** Stale-output
blocking (Phase 2) was already shipped (commit 65ceaf7); this adds the auto-λ half.
`fused_replay_wake_step_batch` (`crates/somnus-perception/src/lib.rs`) previously combined
`sae_loss`+`wake_loss` as an unweighted sum. New opt-in behavior (`--der-auto-lambda`, default
`false`): tracks a running EWMA mean/mean-square of `wake_loss` (α=0.05, seeded from the first
sample to avoid a cold-start zero artifact), derives its variance, and weights `wake_loss` by
`1/variance` (clamped `[0.1, 10.0]`) — homoscedastic uncertainty weighting instead of a hand-tuned
constant. New unit test `der_auto_lambda_stays_finite_and_loss_decreases`
(`crates/somnus-perception/tests/phase5_smoke.rs`) confirms the weighted path stays finite and loss
still decreases over repeated passes. Build/clippy/test all pass; manually smoke-tested via CLI
(`--der-auto-lambda`, no crash).

## Queued for next session

- **gaps.md §9 — EfficientAT-mn10, resume the A/B.** Implementation is done (see "In
  progress" section above for full detail) — this is a resume, not a fresh start. Exact
  steps: `rm -rf memory/ab_efficientat_mn10` then `bash scripts/ab_efficientat_mn10.sh`,
  then document + commit. Was going to be "Low effort" per the original gaps.md estimate;
  turned out Medium (cold-start region-dim change, new Rust mel port, isolated venv) — same
  pattern as §3 before it. Effort actually spent is sunk; only the A/B run + writeup remain.

## Current state

- **DINOv3-ViT-S**: adopted and active (drop-in 384-dim replacement for DINOv2-S). Validated via
  A/B: visual-region MSE -95.6%, `warm_recall_l2` +61.5%, no confound in other regions.
  Details: `ai-docs/a_b_dinov3_findings.md`, `ai-docs/gaps.md` §6.
- **Qwen3-0.6B**: adopted as the sole action-decoder LM (ADR 0036); Qwen2.5 support fully removed
  (`qwen3`/`qwen3_hidden` Cargo features deleted, `LM_HIDDEN`/`LANG_HIDDEN`/`LANG_DIM`/
  `EOS_TOKEN_ID` hardcoded to Qwen3's values, `--action-model-prefix` defaults to `"qwen3"`).
  Confirmed via a speech-dense A/B (~22% lower `mean_proj_loss`, consistent, no reversals) plus a
  direct `generate()` comparison (coherent English vs. Qwen2.5's incoherent token salad).
  Details: `ai-docs/a_b_qwen3_findings.md`, `ai-docs/gaps.md` §5, ADR 0036 (supersedes ADR 0035).
- **Whisper ASR panic**: an intermittent `decode_greedy` ORT panic ("GetElementType is not
  implemented") is mitigated (not root-caused) — `catch_unwind` around the decode call in
  `crates/somnus/src/main.rs` stops it from silently killing the ASR thread for the rest of the
  session. Unverified whether this fully recovers or just downgrades to a per-cycle WARN if the
  underlying ORT `Session` is left poisoned. Caveat detail: `ai-docs/gaps.md` §5.
- All docs in sync: `ai-docs/architecture.md` + `.svg` (cross-modal dim fixed at 1280+1024=2304),
  `ai-docs/gaps.md`, ADR 0036. Build/clippy/tests pass
  (`cargo build/clippy/test --release -p somnus -p somnus-perception`).
- Housekeeping since: removed accidentally-committed model/build cruft from git (models_control/,
  models_lora_ab_control/, stray logs) and hardened `.gitignore`; corrected a stale "queued" line
  in `gaps.md` (candle deps were already removed from the workspace).

**Next step is user-driven, not queued.** Candidates already scoped in `ai-docs/gaps.md`'s
priority table: high replay ratio A/B (item 2, trivial), DER++ auto-λ (item 3 phase 3), or any
deferred encoder/LLM item. A longer single-arm Qwen3 soak test was offered and explicitly declined
by the user 2026-07-05 (take 3 already confirms the adoption; Qwen2.5 no longer exists in the tree
to re-A/B against).

## Key files

- `ai-docs/gaps.md` — full backlog with effort/impact/status table (read this first for "what's next")
- `ai-docs/a_b_dinov3_findings.md`, `ai-docs/a_b_qwen3_findings.md` — A/B evidence trails
- `scripts/ab_dinov3.sh`, `scripts/ab_qwen3.sh`, `scripts/ab_qwen3_longclips.sh` — A/B harnesses (isolated `--memory-dir` per arm, `--metrics-file` NDJSON)
- `memory/dinov3_ab_{control,treatment}/`, `memory/qwen{25,3}_ab_*/` — raw A/B metrics

## Deferred backlog (do not raise unprompted)

CLAUDE.md's "Deferred architectural items" points here for trigger conditions. That table was lost
in an earlier prune (silently collapsed to a pointer with nothing behind it) — restored 2026-07-11
from git history (commit `03eb9b5`, 2026-06-14 ADR scan). `gaps.md`'s priority table covers
everything not listed here.

| # | Source | Item | Trigger |
|---|--------|------|---------|
| A | ADR 0006 | Cold archive O(1) promotion index | `promote()` called AND cold > 500k records |
| C | ADR 0006 | Per-region sparse activation storage | Replay analysis shows episode vectors insufficient for diagnostics; no sparse-code consumer currently exists |
| E | ADR 0009 | Separate optimizer for output heads (independent LR schedule) | Conflicting gradient directions between output heads and k-WTA observed |
| F | ADR 0010 | Post-k-WTA prior injection | Pre-k-WTA injection shows poor `prior_cosine` uptake |
| I | ADR 0018 | Knowledge distillation compression | Grown regions exceed 2× baseline |
| J | ADR 0003 | Whisper-tiny → Whisper-base / WavLM-base (audio quality) | — (Tier 3, revisit opportunistically) |
| K | ADR 0003 | DINOv2-S → SigLIP-B (language-aligned visual embeddings, Phase 7+) | — (superseded in spirit by DINOv3-ViT-S adoption, gaps.md §6 — SigLIP2-ViT-B/16 tracked separately as gaps.md §7) |
| L | ADR 0012 | LoRA fine-tuning of Qwen2.5 for episode-vector prefix | — (Tier 3) — **caveat:** this was marked done 2026-06-25 for Qwen2.5, then Qwen2.5 was fully removed from the tree (ADR 0036, 2026-07-05). Not re-verified whether an equivalent LoRA path exists/is warranted for Qwen3 — check `crates/somnus/src/sleep.rs`/`main.rs` LoRA-related code before assuming this is either done or not-yet-started. |
| M | ADR 0012 | MLP projection (640→5×896) instead of linear | — (Tier 3) — **caveat:** same as L, marked done 2026-06-21 against Qwen2.5's 896-dim output; not re-verified against Qwen3's dimensions post-ADR-0036. |
| N | ADR 0014 | Visual/audio region grounding (captions, env sound labels) | — (Tier 3) |

Original table also had items B, D, G, H (language last-token pooling, full differentiable SAE,
learned per-region gate, activation-frequency pruning) — those are not in CLAUDE.md's current
deferred list, meaning they were resolved/superseded/dropped at some point after the 2026-06-14
scan. Not re-investigated here; if one becomes relevant, check `gaps.md` and ADR history before
assuming it's still open.
