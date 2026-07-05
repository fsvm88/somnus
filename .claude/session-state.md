# Session state — somnus

> **Last updated:** 2026-07-05 (Qwen3 A/B redo take 2: Qwen3 wins on proj_loss, consistently)
> **Branch:** `main`
> **Status:** DINOv3 done. Qwen3 pipeline fully fixed (3 real bugs found+fixed). Take 1 of the
> real A/B (standard 4-clip corpus) was inconclusive — too little speech. Take 2, on a
> speech-dense `clips/long/` lecture/podcast corpus, gave a real, fair comparison: Qwen3 shows
> a consistent ~22% lower `mean_proj_loss` than Qwen2.5 by the end of a 5-session campaign, no
> reversals. Recommend adopting Qwen3-0.6B for the action-decoder LM, pending a longer
> confirmation run or a direct generation-quality test. Full results:
> `ai-docs/a_b_qwen3_findings.md`. Nothing queued; next step is user-driven (confirm-and-adopt,
> or move to a different backlog item).

---

## ✅ DONE: Qwen3 redo — pipeline fixed, A/B inconclusive (2026-07-05)

Continuing from the "queued redo" plan below. Completed this session, in order:

1. **Restored a genuine Qwen2.5 baseline.** `models/qwen25_lm.onnx` was confirmed
   byte-identical to `qwen3_lm_base.onnx` (sha256 match on both graph and `.data`) —
   deleted and re-exported via `scripts/export_models.py::export_qwen25()`. Verified via
   ONNX graph inspection: `inputs_embeds`/896-dim in, not `input_ids`/1024-dim.

2. **Gave Qwen3 a complete GPU-path export.** Rewrote `export_qwen3()` in
   `scripts/export_models.py` to mirror `export_qwen25()` exactly: `qwen3_gain.onnx`,
   `qwen3_text_hidden.onnx`, `qwen3_lm.onnx` (new — dynamic-seq CPU generation path,
   didn't exist before), `qwen3_lm_fixed.onnx`. Also fixed a second real bug found in
   passing: the old `export_qwen3()` loaded Qwen3 weights into `Qwen2ForCausalLM`
   (wrong architecture — Qwen3 has QK-norm layers Qwen2 doesn't), not `Qwen3ForCausalLM`.
   Reused the existing `qwen3_embed_table.safetensors`/tokenizer (embedding weights load
   correctly regardless of wrapper class, only the transformer body was affected).

3. **Parameterized model-file prefix.** `ActionDecoder::new` (`crates/somnus/src/action.rs`)
   took a new `model_prefix: &str` arg; all `qwen25_*` literals became
   `format!("{model_prefix}_*")`. Added `--action-model-prefix` CLI flag (main.rs),
   default value cfg-gated to match the build (see next item). `LangEmbedder::new` also
   parameterized the same way — it previously **auto-detected** qwen3 files over qwen25
   ("prefer qwen3 if present"), which was itself a live footgun once both file sets
   coexist: it would silently pick the wrong file set for whichever binary was running.

4. **Two real release binaries, not a runtime toggle.** Added a `qwen3` Cargo feature
   to `somnus` (propagates `somnus-perception/qwen3_hidden`). `action::LM_HIDDEN` and
   `main::LANG_DIM` are now `#[cfg]`-gated 896/1024 instead of a single hardcoded 1024
   constant. `EOS_TOKEN_ID` is also cfg-gated (Qwen2.5 base stops on `<|endoftext|>`
   =151643, Qwen3 base on `<|im_end|>`=151645 — verified against each tokenizer).
   Built `target/release/somnus-qwen25-control` (default features) and
   `target/release/somnus-qwen3-treatment` (`--features qwen3`).
   **Found and fixed a third real bug via smoke test**, not anticipated by the original
   plan: `somnus-perception/src/lib.rs` had ~6 more hardcoded `1024`/`896` literals for the
   language region (`PerceptionModel::new`'s `language_layer`/`language_decoder`/
   `language_head` sizes, the cross-modal concat offset tables in two places, an ORT
   zero-fallback vector, an ONNX-export weight-shape tuple, and a settle-loop encoder
   region table) — these didn't all agree with each other or with `REGION_DIMS`, so the
   qwen25-control binary panicked immediately on first language ingest
   ("Dimensions are incompatible for matrix multiplication") until all were unified
   under the new `somnus_perception::LANG_HIDDEN` constant. Re-verified: both binaries now
   run a full smoke session (`--action-distill-only`) cleanly, and
   `cargo test -p somnus -p somnus-perception` passes both with and without `--features qwen3`.

   **Lesson for later A/Bs**: `grep` for literal `1024`/`896` across a crate before
   trusting a single named constant governs a dimension — the June 27 "prepare for
   Qwen3" commit changed some call sites and missed others, and the ones it missed had
   stale/wrong values in both directions.

5. **Cold-start A/B ran to completion — result inconclusive.** Old (invalid) run's memory
   dirs moved aside to `memory/qwen25_ab_control_invalid_20260628` /
   `memory/qwen3_ab_treatment_invalid_20260628`. Rewrote `scripts/ab_qwen3.sh` to call the
   two dedicated binaries with `--action-model-prefix {qwen25,qwen3}` directly — no
   file-swapping, so the arms can't cross-contaminate. Ran 10 sessions × 5 cycles/arm
   cold-started (up from the original's 2 cycles/session).
   **Verdict: the pipeline is now correct, but this corpus can't answer the question.**
   Language-region MSE is statistically indistinguishable between arms (0.0011 vs 0.0010),
   and `mean_proj_loss` has almost no samples (0/50 control, 1/50 treatment) — the 4 stock
   clips (`talking_head`/`urban`/`animation`/`nature`) have too little speech for
   ASR/lang-ingest to fire more than ~once per 5-cycle session, regardless of which LM is
   behind it. The large `mean_output_mse`/`warm_recall_l2` gaps that did show up (both favor
   treatment) are noise from independently-seeded non-LM regions (speech/env/visual/
   symbolic all show similar-sized gaps despite having no LM dependency), not a Qwen3
   effect. Full breakdown + recommended next corpus: `ai-docs/a_b_qwen3_findings.md`.
   Also found (not fixed, out of scope): an intermittent panic in
   `WhisperDecoder::decode_greedy` (ORT "GetElementType is not implemented" escalating past
   its usual caught-WARN path) — unrelated to this work, flagged for separate triage.

   **Next step is user-driven, not queued**: either point the harness at `clips/long/`
   (real speech, per `project_long_clips` memory) for a corpus with enough language signal,
   or test generation quality directly (drop `--action-distill-only`, compare `generate()`
   output between the two binaries). Both binaries and both full model-file sets are ready;
   nothing is blocked on infrastructure anymore.

6. **Take 2: speech-dense `clips/long/` corpus — real signal, Qwen3 wins.** New harness
   `scripts/ab_qwen3_longclips.sh`: same two binaries, 6 lecture/podcast clips (Stanford
   CS229, Bengio/LeCun podcasts, Hossenfelder physics, a consciousness lecture, an
   algorithms lecture — ~1.7 GB), 5 sessions × 4 cycles/arm. A smoke test on one clip
   confirmed the corpus swap worked: 1 cycle → `asr_count=20`, `lang_ingest_count=16`,
   `decoder_trainable=10` (vs. take 1's ~1 event per whole session).
   Full run: both arms got **identical exposure** (73 `lang_ingest` events, ~40
   `decoder_trainable` events each — confirms this is a fair comparison, not one arm
   getting more signal). `mean_proj_loss` had 12/20 non-null samples per arm this time.
   **Result: treatment (Qwen3) sits below control (Qwen2.5) at every single comparable
   `mean_proj_loss` sample across the whole 5-session campaign, no reversals** — final
   values 0.001205 (Qwen3) vs 0.001551 (Qwen2.5), a ~22% gap. `per_region_mse[language]`
   is directionally consistent (0.0010 vs 0.0011). `warm_recall_l2` came back close this
   time (0.48 vs 0.47), supporting take 1's read that its wild gap there was seed noise.
   **Recommendation: worth adopting Qwen3-0.6B for the action-decoder LM**, pending either
   a longer confirmation run or a direct generation-quality test (drop
   `--action-distill-only`, compare `generate()` output) since proj_loss is a surrogate
   for grounding quality, not generation quality itself.
   The Whisper ASR panic (see above) recurred more often on this denser-speech corpus
   (~1/5 sessions control, ~3/5 treatment) — still no data loss, still unrelated to this
   work, but the higher hit-rate makes it more worth fixing before a much longer campaign.
   Full writeup: `ai-docs/a_b_qwen3_findings.md` ("Redo take 2").

---

## ✅ COMPLETED: DINOv3-ViT-S A/B (2026-07-05)

The 2026-06-28 "DINOv3 is GPU-only / MIGraphX can't compile its attention graph" conclusion was
**wrong** — found and fixed this session. Real cause: `torch.onnx.export` wrote DINOv3 with
external weight data, and MIGraphX resolves that path relative to the process's CWD, not the
model file's directory, so it only worked when launched from exactly `models/`. Fixed by
re-packing the export as a single self-contained `.onnx` (`scripts/export_models.py`). Also fixed
a real, separate pooling bug: DINOv3 has 4 register tokens DINOv2 doesn't
(`[CLS, register×4, patch×196]` vs DINOv2's `[CLS, patch×256]`); `crates/somnus-encoding/src/video.rs`
now tracks prefix/total tokens per model instead of hardcoding "skip 1, pool 256."

Ran the real DINOv3-vs-DINOv2 A/B (`scripts/ab_dinov3.sh`, rewritten to isolate `--memory-dir` per
arm and use `--metrics-file` NDJSON, same pattern validated for Qwen3). Result: **clear win**.

- Visual-region reconstruction MSE: 2.183 → 0.097 (**-95.6%**)
- warm_recall_l2 (memory retrieval quality): 0.2994 → 0.4835 (**+61.5%**)
- Speech/env/language regions unaffected (expected no-confound pattern)
- No crashes across 10 sessions

Full writeup: `ai-docs/a_b_dinov3_findings.md`. `ai-docs/gaps.md` Item 6 marked deployed/validated.
DINOv3 is already the active encoder (`VideoEncoder::new` prefers it when present in `models/`,
which it now is) — **no rollout action needed, this A/B validates what's already running.**

---

## 🚧 QUEUED: Qwen3 — original A/B was invalid, needs a proper redo (not started)

**The 2026-06-28 "-9.6%" finding (`ai-docs/a_b_qwen3_findings.md`) does not mean what it says.**
Discovered 2026-07-05 while investigating unrelated ort-smoke test output: the language-embedding
pipeline has been silently corrupted since commit `6335b42` (2026-06-27, "prepare for
Qwen3-0.6B"), which bumped `LM_HIDDEN` in `crates/somnus/src/action.rs` from 896 → 1024 as a global
constant. `embed_flat()` (`action.rs:1207`) slices `self.embed_table` at stride `LM_HIDDEN`, but
`self.embed_table` is *always* loaded from `qwen25_embed_table.safetensors`
(`action.rs:219`, hardcoded), which is Qwen2.5's real 896-wide table. 1024-stride reads into an
896-wide table are misaligned for every token — silent, not a crash (vocab is large enough that
this rarely goes out of bounds). **Both arms of the original A/B were fed corrupted embeddings**;
neither result reflects the real architectures. On top of that, Qwen3 never got matching GPU-path
exports (`qwen3_gain.onnx` / `qwen3_text_hidden.onnx` / `qwen3_lm_fixed.onnx` don't exist — only a
bare `qwen3_lm_base.onnx`), so it was piggybacking on `ab_qwen3.sh`'s file-swap trick the whole
time. That swap's "restore" step also silently failed: `models/qwen25_lm.onnx` on disk right now
**is** the Qwen3 model (confirmed: byte-identical size to `qwen3_lm_base.onnx`/`.data`, declares an
`input_ids`/1024-dim interface, not Qwen2.5's `inputs_embeds`/896-dim one). This currently breaks
real (non-`--action-distill-only`) generation on `main`.

Neither the original Qwen3 A/B nor this session's DINOv3 A/B visibly hit errors from this, because
both used `--action-distill-only` (skips `generate()`, the one path that calls the broken
`lm_session` directly) and `--lang-use-lm` prefers the separate, untouched, still-correct
`qwen25_text_hidden.onnx` for embedding text. **The DINOv3 finding is unaffected** — visual-region
MSE and warm_recall_l2 gains are orthogonal to this bug. The Qwen3 comparison is the only casualty.

**Decision (2026-07-05): worth a real, valid test** — this isn't "Qwen3 was tried and lost," it's
"Qwen3 was never actually tested." Queued fix + redo plan, not started, roughly in order:

1. **Restore integrity first:** re-export a genuine Qwen2.5 `qwen25_lm.onnx` (currently mislabeled
   Qwen3 weights) so there's a trustworthy baseline again.
2. **Give Qwen3 a complete export**, matching Qwen2.5's GPU-path completeness: `qwen3_gain.onnx`,
   `qwen3_text_hidden.onnx`, `qwen3_lm_fixed.onnx` (mirror `export_qwen25()` in
   `scripts/export_models.py`, parameterized for Qwen3's hidden_size=1024). Reuse existing
   `qwen3_embed_table.safetensors` / `qwen3_tokenizer.json` if still valid, else regenerate.
3. **Stop hardcoding `qwen25_*` filenames** in `ActionDecoder::new` (`action.rs:217-220`) —
   parameterize the model-file prefix (CLI flag or constructor arg) so control/treatment runs
   can never cross-contaminate each other's files the way the file-swap trick did.
4. **Two separate release builds, not a runtime toggle.** `LM_HIDDEN` (896 vs 1024) cascades into
   `REGION_DIMS`/`CROSS_MODAL_DIM`/projection-layer shapes in both `somnus` and `somnus-perception` —
   this is a compile-time architecture choice and a cold-start for the projection/SAE weights
   either way (already true per the original commit message), not something to hot-swap. Build one
   binary at `LM_HIDDEN=896` against real Qwen2.5 files (control), one at `1024` against real
   Qwen3 files (treatment).
5. **Cold-start both and let them actually converge** before comparing — the invalidated run only
   used 2 NREM cycles per session, likely too short to fairly judge a bigger model. Plan for a
   longer run than the original A/B.

Also still on the table once this is done: **Option D, Qwen2.5-0.5B-Instruct** — cheap,
same 896-dim architecture as the (correctly restored) Qwen2.5 baseline, no cold-start needed. Can
run this independently of the Qwen3 redo above, on whichever build has real Qwen2.5 weights.

Not pursuing right now: deep-diving Qwen3 via LoRA before the above is fixed (Option A from the
2026-06-28 notes) — no point tuning on top of a broken embedding pipeline.

---

## Key files

- `ai-docs/gaps.md` — full backlog with effort/impact/status table
- `ai-docs/a_b_dinov3_findings.md` — DINOv3 A/B results (this session)
- `ai-docs/a_b_qwen3_findings.md` — Qwen3 A/B results (2026-06-28, corrected 2026-07-05 to remove the false DINOv3 claim)
- `scripts/ab_dinov3.sh`, `scripts/ab_qwen3.sh` — A/B harnesses, both use isolated `--memory-dir` per arm + `--metrics-file` NDJSON
- `memory/dinov3_ab_{control,treatment}/`, `memory/qwen{25,3}_ab_*/` — raw A/B metrics
