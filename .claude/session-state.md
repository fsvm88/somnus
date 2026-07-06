# Session state — somnus

> **Last updated:** 2026-07-06
> **Branch:** `main`. Change-gated video encoder work is committed (`9bba42b`); SVD check work
> committed (`b410ad7`); high-replay-ratio A/B work (this section) not yet committed.
> **Status: brain-scale-comparison survey fully closed out; gaps.md item 2 (replay ratio) validated.**

## Just completed (2026-07-06)

Both action items from `ai-docs/research/brain-scale-comparison.md`'s "Summary judgment" are done
(see prior entries in git log for details: `9bba42b` change-gated video encoder, `b410ad7` SVD
null result).

**gaps.md item 2, high replay ratio A/B — validated, not yet committed.** Found and fixed a real
bug first: `scripts/train_campaign.sh` (shared by every A/B script) hardcoded
`--nrem-replay-epochs 4` unconditionally, so the existing stale run (2026-06-27, now renamed
`memory/ab_replay_ratio_stale_20260627_buggy_n4_both_arms/`) actually compared N=4 vs N=4 —
meaningless. Fixed by parameterizing `REPLAY_EPOCHS` (env var, default 1) and having
`ab_replay_ratio.sh` set it explicitly per arm. Reran fresh (also needed since the old data
predates DINOv3/Qwen3 adoption): **-70.9% mean_output_mse** (control 0.2379 → treatment 0.0693),
consistent across every cycle with valid data, -83% to -88% per-cycle where directly comparable,
all 5 regions improve monotonically with no reversals. Full writeup: `ai-docs/gaps.md` §2.
**Open question, not decided:** whether to flip the CLI default from N=1 to N=4 — real tradeoff
(4× NREM wall-clock per sleep cycle), not a free win, so flagged rather than silently changed.
Files not yet committed: `scripts/train_campaign.sh`, `scripts/ab_replay_ratio.sh`, `gaps.md`,
this file. `memory/ab_replay_ratio/{control,treatment}/` holds the fresh run (gitignored).

## Queued for next session

Empty. Open question above (flip the default?) is the only unresolved thread — needs a user
decision, not more investigation. Otherwise fall back to `ai-docs/gaps.md`'s priority table:
items 7–13 (DER++ auto-λ, EfficientAT/M2D-CLAP/SigLIP2 encoder swaps, LISTA, Gemma 3, usearch
migration) are all explicitly deferred, not queued.

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
