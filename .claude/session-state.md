# Session state — somnus

> **Last updated:** 2026-07-06
> **Branch:** `main`. Change-gated video encoder work is committed (`9bba42b`); SVD check work
> (this section) not yet committed — see "Just completed" below for the file list.
> **Status: both action items from the brain-scale-comparison survey are done. No queue.**

## Just completed (2026-07-06)

Both action items from `ai-docs/research/brain-scale-comparison.md`'s "Summary judgment" are done:

1. **Change-gated video encoder inference** (committed `9bba42b`). `VideoFrame::mean_abs_diff`
   (`crates/somnus-sensory/src/frame.rs`) skips `enc.encode()` in the video encoder task
   (`crates/somnus/src/main.rs`) when a stride-selected frame's byte-level diff from the last
   *encoded* frame is below `--video-change-threshold` (default 0.0 = disabled, opt-in only).
   `--video-change-max-skip` (default 30) forces a periodic re-encode so a fully static scene
   doesn't stop emitting embeddings. Audio (Whisper/CLAP) left ungated (different, less contained
   problem). Details: `ai-docs/gaps.md` §13. **Not yet done:** an A/B measuring actual wake-path
   CPU savings/skip rate on a real capture session — deployed and tested, unmeasured in practice.

2. **Post-training SVD check on region/binding weights** — **null result, not committed yet**.
   New standalone bin `cargo run --release -p somnus-perception --bin svd_rank_check --
   <memory_dir> <out_dir>` loads a trained checkpoint and dumps its 21 own region/binding weight
   matrices (k-WTA encoders, cross-modal proj, binding region projections, attention Q/K/V, GRU
   gates, temporal proj — not the frozen encoders/LM) as raw f32 + manifest.
   `scripts/svd_rank_check.py` (numpy) computes effective rank per matrix. Checked against
   `memory/qwen3_ab_treatment/` (most-trained checkpoint on disk): all 21 matrices are
   near-full-rank (mean r95/full_rank = 0.772) — **low-rank factorization is not free, not
   pursued.** Full numbers and caveats: `ai-docs/gaps.md` §14. Files not yet committed:
   `crates/somnus-perception/src/bin/svd_rank_check.rs`, `crates/somnus-perception/Cargo.toml` (new
   `[[bin]]` entry), `scripts/svd_rank_check.py`, plus this file/`gaps.md`/the research note.

## Queued for next session

Empty. Both flagged-actionable items from the brain-scale survey are resolved (one shipped, one
a documented null result). Everything else in that doc (MoE routing, weight sharing, spiking/
event-driven hardware, adaptive computation time, local attention, multi-timescale processing)
stays speculative/deferred until region count or binding complexity grows past current design.
Otherwise fall back to `ai-docs/gaps.md`'s priority table for what's next (e.g. item 2, high
replay ratio A/B — trivial effort, still queued).

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
