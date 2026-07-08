# Session state — somnus

> **Last updated:** 2026-07-08
> **Branch:** `main`. Committed: change-gated video encoder (`9bba42b`), SVD null-result
> (`b410ad7`), train_campaign.sh bugfix + replay-ratio A/B (`d5ce2ab`), N=4 replay-ratio default
> (`f0fa9c6`), items queued (`709442b`), DER++ auto-λ implementation (`f2fd185`, gaps.md §3).
> **Status: gaps.md §3 (DER++ auto-λ) A/B'd and REJECTED — FAIL, +73.8% mean_output_mse, 10/10
> cycles regressed, no reversals. Stays off by default, no code revert needed. gaps.md §9
> (EfficientAT-mn10) is next up, still queued/untouched.**

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

- **gaps.md §9 — EfficientAT-mn10 ONNX export test** (still queued, untouched). Low-risk
  MobileNetV3 audio-env encoder alternative (~5–20M params, pure CNN, no attention/ScatterElements
  ops — should export to ONNX trivially). Full spec: `ai-docs/gaps.md` §9. Effort: Low (export +
  A/B vs current CLAP-HTSAT). Verify actual penultimate-feature dim (doc estimates ~960,
  unconfirmed) before wiring up the region-input projection change.
- Read the §3 auto-λ A/B result (above) once it completes; update gaps.md/commit either way
  (adopt-as-default, keep opt-in, or revert) before starting §9.

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
