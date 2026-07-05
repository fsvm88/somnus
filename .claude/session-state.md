# Session state — somnus

> **Last updated:** 2026-07-05 (Qwen3 adopted as sole LM, Qwen2.5 dropped; gen-quality confirmed;
> Whisper ASR panic mitigated)
> **Branch:** `main`
> **Status:** DINOv3 done (validated, already active). Qwen3 A/B done in three takes (take 1
> inconclusive, take 2 speech-dense corpus favored Qwen3 on proj_loss, take 3 direct
> generation-quality test confirmed it — coherent English vs. Qwen2.5's incoherent token salad).
> **Adopted Qwen3-0.6B as the only action-decoder LM (ADR 0036) — Qwen2.5 support fully removed**
> (`qwen3`/`qwen3_hidden` Cargo features deleted, `LM_HIDDEN`/`LANG_HIDDEN`/`LANG_DIM`/
> `EOS_TOKEN_ID` hardcoded to Qwen3's values, `--action-model-prefix` defaults to `"qwen3"`).
> Also fixed a latent bug (hardcoded `qwen25_*` filenames in the `--language` startup check,
> inconsistent with every other model-loading call site) and mitigated the intermittent Whisper
> `decode_greedy` ORT panic (`catch_unwind` around the decode call — stops it from silently
> killing the whole ASR thread; unverified whether it fully recovers or just downgrades to a
> per-cycle WARN, see caveat in `ai-docs/gaps.md` §5). All docs in sync: `ai-docs/architecture.md`
> + `.svg` (cross-modal dim 1280+1024=2304, was a 2176/2304 toggle), `ai-docs/gaps.md` §5,
> ADR 0035 (superseded) + new ADR 0036. Full results in `ai-docs/a_b_qwen3_findings.md`
> ("Take 3" section is the newest).
>
> **Working tree has uncommitted changes from this session** — not committed yet (user hasn't
> asked). Touches: `crates/somnus/{Cargo.toml,src/action.rs,src/main.rs}`,
> `crates/somnus-perception/{Cargo.toml,src/lib.rs}`, `ai-docs/{architecture.md,architecture.svg,
> gaps.md,a_b_qwen3_findings.md,decisions/0035-*.md}`, new `ai-docs/decisions/0036-*.md`. Build/
> clippy/tests all pass (`cargo build/clippy/test --release -p somnus -p somnus-perception`), smoke
> session verified clean.
>
> **Queue this session, in order — all resolved:** 1. adopt Qwen3 + drop Qwen2.5 ✅ done.
> 2. longer confirmation campaign — **skipped, user-confirmed** (premise changed: take 3 already
> confirms the decision, Qwen2.5 no longer exists in the tree to re-A/B against). 3. direct
> generation-quality test ✅ done (take 3, this session). 4. fix Whisper `decode_greedy` panic
> ✅ mitigated (not root-caused — see caveat above). **Queue is empty. Nothing pending.**

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

## ✅ DONE: the above redo plan — fully executed, then graduated to adoption (2026-07-05)

The five-step redo plan that used to live in this section (restore Qwen2.5 integrity, give Qwen3 a
complete export, parameterize the model-file prefix, build two separate release binaries, cold-start
and compare) is exactly what the "DONE: Qwen3 redo" section above records, in takes 1 and 2. It's
kept here only as a one-line pointer since the detailed plan is now historical: see
`ai-docs/a_b_qwen3_findings.md` and ADR 0035 for what actually happened.

**This session (continuing from take 2) went one step further and closed the loop:**

1. **Adopted Qwen3-0.6B as the sole action-decoder LM, dropped Qwen2.5 (ADR 0036).** Removed the
   `qwen3` (somnus) / `qwen3_hidden` (somnus-perception) Cargo features entirely — CLAUDE.md's "don't use
   feature flags when you can just change the code" plus no backward-compat requirement made this
   the right call once the A/B question was answered, rather than flipping a default and keeping
   Qwen2.5 as a dead option. `LM_HIDDEN`/`LANG_HIDDEN`/`LANG_DIM`/`EOS_TOKEN_ID` are now hardcoded
   to Qwen3's values (1024, 151645). `--action-model-prefix` defaults unconditionally to `"qwen3"`.
   Found and fixed one latent bug while de-cfg'ing: the `--language` startup file-existence check in
   `main.rs` had hardcoded `qwen25_*` filenames regardless of `--action-model-prefix`, unlike every
   other model-loading call site — now derives from `args.action_model_prefix` like the rest.
   Left the two cumulative cross-modal offset tables in `somnus-perception/src/lib.rs` untouched (the
   `896` there is `384+512`, a coincidental collision with the old LM hidden dim, not a reference to
   it) — this was exactly the kind of literal a careless find-replace would have broken.
   `ai-docs/architecture.md`/`.svg` updated (cross-modal dim 1280+1024=2304, was a 2176/2304
   feature-gated toggle). Verified: `cargo build/clippy/test --release -p somnus -p somnus-perception`
   all clean, plus a live smoke session (`--action-distill-only`, ~50s) and a `generate()` smoke
   test — no dimension-mismatch panics either way.

2. **Direct generation-quality confirmation ("take 3"), same day.** Since both Cargo features were
   gone from the working tree, used a `git worktree` at the last commit that still had both
   (`c86cfdf`) to build one `somnus-qwen25-control` and one `somnus-qwen3-treatment` binary — cleaner than
   comparing working-tree-Qwen3 against a different commit's Qwen2.5, since the LM choice stays the
   only variable. Ran both against `clips/talking_head.mp4` (`--file-loop`, needed because
   non-looping `--file` mode decodes faster than realtime and the process exits before episodes
   finish forming — a real gotcha worth remembering for future ad hoc comparisons), each pointed at
   its own take-2 trained memory dir (`qwen25_ab_control_longclips` / `qwen3_ab_treatment_longclips`,
   copied to scratch first so originals are untouched), same three prompts via delayed stdin.
   **Result: Qwen3 generates coherent, grammatical, on-topic English; Qwen2.5 generates incoherent
   mixed-script token garbage — consistently, across every prompt and repeat run.** Confirms take
   2's `mean_proj_loss` gap reflects real, user-visible generation quality, not just a lower
   surrogate loss. Full transcript-level detail: `ai-docs/a_b_qwen3_findings.md` ("Take 3").
   Cleanup done: worktree removed, temp binaries/memory dirs deleted.

3. **Mitigated the intermittent Whisper `decode_greedy` ORT panic** flagged (not fixed) in take 2.
   The ASR background thread in `main.rs` now wraps `decode_greedy` in `std::panic::catch_unwind`,
   so a panic no longer silently kills the whole ASR thread for the rest of the session — it logs a
   WARN and the loop continues. **Unverified caveat:** if the underlying ORT panic leaves the
   `Session` itself poisoned, every subsequent call on that session could panic too, in which case
   this degrades "ASR dies silently forever" to "ASR logs a WARN and drops every cycle" rather than
   full recovery — only reproducing the intermittent panic would tell, and that wasn't attempted.
   The root cause of the panic itself is still undiagnosed.

**Item 2 (longer confirmation campaign): explicitly skipped, user-confirmed 2026-07-05.** Its
premise changed over the course of this session — take 3 already satisfies "confirm
before/after adoption," and Qwen2.5 no longer exists in the tree, so a longer *A/B* would have
meant rebuilding a dropped arm via worktree just to re-confirm an already-settled, twice-confirmed
result. Asked the user directly (skip / single-arm Qwen3 soak test / rebuild-and-rerun A/B) — they
chose skip. Not queued; no soak test pending unless raised again later.

---

## Key files

- `ai-docs/gaps.md` — full backlog with effort/impact/status table
- `ai-docs/a_b_dinov3_findings.md` — DINOv3 A/B results (this session)
- `ai-docs/a_b_qwen3_findings.md` — Qwen3 A/B results (2026-06-28, corrected 2026-07-05 to remove the false DINOv3 claim)
- `scripts/ab_dinov3.sh`, `scripts/ab_qwen3.sh` — A/B harnesses, both use isolated `--memory-dir` per arm + `--metrics-file` NDJSON
- `memory/dinov3_ab_{control,treatment}/`, `memory/qwen{25,3}_ab_*/` — raw A/B metrics
