# Session state — somnus

> **Last updated:** 2026-07-09
> **Branch:** `main`. Committed: change-gated video encoder (`9bba42b`), SVD null-result
> (`b410ad7`), train_campaign.sh bugfix + replay-ratio A/B (`d5ce2ab`), N=4 replay-ratio default
> (`f0fa9c6`), items queued (`709442b`), DER++ auto-λ implementation (`f2fd185`, gaps.md §3),
> DER++ auto-λ A/B reject (`72d1503`). EfficientAT-mn10 implementation + A/B result committed
> this session (gaps.md §9) — see below.
> **Status: gaps.md §3 and §9 both resolved (both REJECTED). Queue empty — see "Queued for
> next session" below.**

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

## Queued for next session

Nothing queued. Both A/B items on deck (`gaps.md` §3 DER++ auto-λ, §9 EfficientAT-mn10) are
now resolved (both rejected, kept opt-in-only). Remaining `gaps.md` priority-table items are
all either ✅ done or explicitly Deferred (M2D-CLAP #9, SigLIP2 #10, LISTA #11, Gemma 3 #12,
usearch #13) — none are queued as an immediate next step. Next step is user-driven; see
`ai-docs/gaps.md`'s priority table for deferred candidates if the user wants to pick one up.

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

Items A, C, E, F, I, J, K, L, M, N — see `CLAUDE.md` "Deferred architectural items" for the list
and trigger conditions. `gaps.md`'s priority table covers everything else.
