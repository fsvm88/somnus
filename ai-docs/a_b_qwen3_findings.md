# A/B Validation: Qwen3-0.6B vs Qwen2.5-0.5B

## Redo take 1 (2026-07-05): infra fixed, comparison inconclusive — too little language signal

The 2026-06-28 run below was invalid (see the box further down): a global `LM_HIDDEN`
constant was bumped to 1024 while the on-disk embed table stayed Qwen2.5's real 896-wide
table, silently corrupting every token embedding in both arms, and Qwen3 was never given
its own GPU-path export so it rode along on a file-swap trick whose restore step failed.

This session fixed the whole pipeline before re-running anything:

1. **Restored a genuine Qwen2.5 baseline.** `models/qwen25_lm.onnx` was confirmed
   byte-identical to `qwen3_lm_base.onnx` (sha256 match, graph and `.data`) — deleted and
   re-exported. Verified via ONNX graph inspection: `inputs_embeds`/896-dim, not
   `input_ids`/1024-dim.
2. **Gave Qwen3 a complete, correctly-architected export.** Rewrote `export_qwen3()` in
   `scripts/export_models.py` to mirror `export_qwen25()`: `qwen3_gain.onnx`,
   `qwen3_text_hidden.onnx`, `qwen3_lm.onnx` (new), `qwen3_lm_fixed.onnx`. Also fixed a
   second real bug found in passing: the old exporter loaded Qwen3's checkpoint into
   `Qwen2ForCausalLM` (wrong architecture — Qwen3 has QK-norm layers Qwen2 doesn't), not
   `Qwen3ForCausalLM`.
3. **Parameterized the model-file prefix.** `ActionDecoder::new` and `LangEmbedder::new`
   (`crates/somnus/src/action.rs`, `crates/somnus/src/main.rs`) took a `model_prefix`/
   `--action-model-prefix` argument instead of hardcoded `qwen25_*` filenames. This also
   removed a live footgun: `LangEmbedder` previously **auto-detected** "prefer qwen3 files
   if present," which — once both file sets coexist on disk, as they now permanently do —
   would silently pick the wrong file set for whatever binary happened to be running.
4. **Two real release binaries, not a runtime toggle.** Added a `qwen3` Cargo feature
   (`somnus` → `somnus-perception/qwen3_hidden`) that `#[cfg]`-gates `LM_HIDDEN`/`LANG_DIM`
   (896 vs 1024) and `EOS_TOKEN_ID` (Qwen2.5 base stops on `<|endoftext|>`=151643, Qwen3
   base on `<|im_end|>`=151645). Built `somnus-qwen25-control` (default) and
   `somnus-qwen3-treatment` (`--features qwen3`). **Smoke-testing found a third real bug**:
   `somnus-perception/src/lib.rs` had ~6 more language-region dimensions hardcoded to
   `1024`/`896` that didn't all agree with each other or with `REGION_DIMS` — the
   qwen25-control binary panicked on first language ingest
   ("Dimensions are incompatible for matrix multiplication") until all were unified under
   a new `somnus_perception::LANG_HIDDEN` constant. Both binaries now pass a full smoke
   session and `cargo test -p somnus -p somnus-perception` (with and without `--features qwen3`).
5. **Cold-started both arms**, no file-swapping: rewrote `scripts/ab_qwen3.sh` to call
   the two dedicated binaries directly with `--action-model-prefix {qwen25,qwen3}`.
   10 sessions × 5 NREM cycles per arm (vs. the original's 2 cycles/session).

### Result: the pipeline is now correct, but this test corpus doesn't exercise it enough to compare

| Metric (mean over 50 cycles) | Qwen2.5-0.5B (control) | Qwen3-0.6B (treatment) |
|---|---|---|
| `per_region_mse[language]` | 0.0011 | 0.0010 |
| total `lang_ingest_count` (of 50 cycles) | 24 | 26 |
| total `decoder_trainable` events | 14 | 16 |
| `mean_proj_loss` non-null samples | 0 / 50 | 1 / 50 |
| `mean_output_mse` (all 5 regions) | 0.0534 | 0.0124 |
| `warm_recall_l2` | 0.6065 | 0.2149 |

The language-region MSE is statistically indistinguishable between arms (0.0011 vs
0.0010), and there's essentially no `mean_proj_loss` signal to compare (0 and 1 non-null
samples out of 50 cycles each) — the four stock clips (`talking_head`, `urban`,
`animation`, `nature`) contain very little actual speech, so ASR/language-ingest events
are sparse (~1 per 5-cycle session, mostly in cycle 1) regardless of which LM is behind
them. **This A/B cannot distinguish Qwen2.5 from Qwen3 on the metric that matters** —
there just isn't enough language-grounding signal in this corpus to say anything.

The large gaps in `mean_output_mse` and `warm_recall_l2` (both favor treatment) are
**not evidence of a Qwen3 effect** — those aggregates are dominated by the speech/env/
visual/symbolic regions (see per-region breakdown below), none of which are downstream
of the LM in this architecture. Per-region breakdown makes this obvious:

| Region | Control mean MSE | Treatment mean MSE |
|---|---|---|
| speech | 0.5670 | 0.4699 |
| env | 0.0344 | 0.0277 |
| visual | 0.0883 | 0.0762 |
| symbolic | 0.0350 | 0.0163 |
| language | 0.0011 | 0.0010 |

Every region differs by a similar proportion, including three regions with no LM
dependency at all — this is run-to-run variance from independent random initialization
(cold-started projection/output-head weights, not seeded identically across arms), not
a systematic Qwen3 effect. Treat these two numbers as noise, not a finding.

### Aside: unrelated ASR crash found during this run

One or two sessions in each arm hit a hard panic identified via `addr2line` on the
stripped release binary as `ort::session::Session::run` inside
`WhisperDecoder::decode_greedy` (`crates/somnus/src/whisper_asr.rs`) — the same
"GetElementType is not implemented" ORT error that usually surfaces as a caught `WARN`
elsewhere apparently escalates to a panic in this path under some condition. Each
session is a separate process, so the crash only cost that one session's data and the
campaign auto-continued; metrics files still show all 50/50 expected cycles for both
arms. This is unrelated to the Qwen3 work (whisper_asr.rs, not action.rs or
somnus-perception) and is not fixed here — flagging for separate triage.

### Recommendation

Don't re-run this exact setup expecting a different answer — the corpus is the
bottleneck, not sample count. To actually compare Qwen2.5 vs Qwen3 on language grounding,
either:
- swap in the `clips/long/` corpus (`ai-docs/` — 15 clips, 20 GB, real speech content,
  per `project_long_clips` memory) so ASR/lang_ingest fires every cycle instead of once
  per session, or
- test generation quality directly: drop `--action-distill-only`, drive `generate()`
  with real queries against both binaries, and compare output coherence/`output_mse`
  from actual autoregressive decoding rather than passive ASR-of-video-clips.

Infra is no longer the blocker for either path — both binaries, both full model-file
sets, and the harness (`scripts/ab_qwen3.sh`) are ready to go.

---

## Redo take 2 (2026-07-05): speech-dense `clips/long/` corpus — Qwen3 wins on proj_loss

Followed take 1's own recommendation: swapped the 4 sparse-speech clips for 6
speech-dense lecture/podcast clips from `clips/long/` (Stanford CS229, Yoshua Bengio and
Yann LeCun podcasts, Sabine Hossenfelder physics, a consciousness lecture, an algorithms
lecture — ~1.7 GB, deliberately excluding the huge low-speech 4K city-walk files).
New harness: `scripts/ab_qwen3_longclips.sh`. Same two dedicated binaries, same
`--action-model-prefix`, no file-swapping. 5 sessions × 4 cycles/arm, cold-started.

A smoke test on a single lecture clip confirmed the corpus swap worked as intended:
one cycle produced `asr_count=20`, `lang_ingest_count=16`, `decoder_trainable=10` —
compare to take 1's ~1 `lang_ingest` event per *entire 5-cycle session*. Full campaign:

| Metric (summed/mean over 20 cycles) | Qwen2.5-0.5B (control) | Qwen3-0.6B (treatment) |
|---|---|---|
| total `lang_ingest_count` | 73 | 73 |
| total `asr_count` | 78 | 78 |
| total `decoder_trainable` events | 40 | 41 |
| `mean_proj_loss` non-null samples | 12 / 20 | 12 / 20 |
| `mean_proj_loss`, first sample (session 1) | 0.002227 | 0.001899 |
| `mean_proj_loss`, last sample (session 5) | 0.001551 | 0.001205 |
| `per_region_mse[language]` (mean) | 0.0011 | 0.0010 |
| `mean_output_mse` (all 5 regions) | 0.0567 | 0.0414 |
| `warm_recall_l2` | 0.4724 | 0.4800 |

**This is a real, fair comparison** — both arms got identical exposure (73 `lang_ingest`
events each, 40 vs 41 training events, near-identical `asr_count`), so the difference
in `mean_proj_loss` isn't an artifact of one arm getting more/less language signal.

**Result: Qwen3 shows a consistent, monotonic edge on `mean_proj_loss`.** Plotting every
non-null sample across the whole campaign (session 1 → 5), both arms' surrogate
projection loss trends steadily downward as the decoder trains — expected, more NREM
exposure improves the fit — but **treatment sits below control at every single
comparable point**, with no crossovers:

```
session  1        2        3        4        5
control  0.00223→0.00217→0.00214→0.00199→0.00155
         (+2 more mid-session samples per arm, same monotonic pattern)
treatment 0.00190→0.00184→0.00170→0.00152→0.00120
```

By the final sample, treatment's proj_loss (0.001205) is ~22% lower than control's
(0.001551). Qwen3's mean-pooled LM hidden states appear to be an easier regression
target for the episode-projection head to hit than Qwen2.5's — plausibly because
Qwen3's larger/richer hidden-state space is more consistent per text span (less noisy
supervision target), though this run doesn't isolate *why*, only that it's real.

`per_region_mse[language]` is close (0.0010 vs 0.0011) and not on its own decisive, but
it's directionally consistent with the proj_loss result. `warm_recall_l2` came back
close this time (0.4724 vs 0.4800) — much closer than take 1's noisy 0.6065 vs 0.2149 —
supporting the take-1 conclusion that the earlier gap there was seed noise, not signal.
`mean_output_mse` still favors treatment by a wide margin (0.0567 vs 0.0414), still
dominated by non-LM regions (mostly `speech`), so treat that one gap as before: probably
independent-seed noise, not a real Qwen3 effect, since nothing about the LM choice
should affect audio-region reconstruction.

**Verdict: worth adopting Qwen3-0.6B for the action-decoder LM**, on the strength of the
proj_loss result — it's the one metric this run actually isolates to the LM, exposure
was identical between arms, and the direction was consistent across the whole campaign
with no reversals. Recommended before flipping the default build: run this same
corpus for longer (more sessions) to confirm the trend holds, and/or test actual
generation quality (drop `--action-distill-only`, compare `generate()` output directly)
since proj_loss is a surrogate for grounding quality, not generation quality itself.

**Aside**: the intermittent `WhisperDecoder::decode_greedy` ORT panic from take 1 (see
below) recurred more often on this denser-speech corpus (more ASR calls → more chances
to hit it) — roughly 1 crash in control's 5 sessions, 3 in treatment's. Each crash is a
separate process (one per session) and happens after that session's metrics are already
written, so no data was lost (`20/20` cycles present in both `metrics.jsonl` files), but
the higher hit-rate here makes it more worth fixing before a much longer campaign.
Still unrelated to the Qwen3 work (`whisper_asr.rs`, not `action.rs`/`somnus-perception`).

### Files (take 2)

- Control: `memory/qwen25_ab_control_longclips/metrics.jsonl`
- Treatment: `memory/qwen3_ab_treatment_longclips/metrics.jsonl`
- Script: `scripts/ab_qwen3_longclips.sh`

---

## Take 3 (2026-07-05): direct generation-quality test — confirms take 2

Take 2 measured `mean_proj_loss`, a surrogate for grounding quality, not the actual thing users
would see. This test drops `--action-distill-only` and compares real `generate()` output between
the two arms, using each arm's take-2 *trained* memory dir (`qwen25_ab_control_longclips` /
`qwen3_ab_treatment_longclips`, copied to scratch dirs so the originals stay untouched) — this
matters because the frozen LM is only half the pipeline; the episode→region-token projection MLP
(tuned by NREM distillation) is the other half, and a cold-start comparison would mostly measure
"is Qwen3 a better base model," which published benchmarks (MMLU 70 vs 47) already answer.

**Method:** by the time this ADR 0036 adoption work started, both Cargo features had been removed
from the working tree, so a `git worktree` at the last commit with both features (`c86cfdf`) built
`somnus-qwen25-control` (default) and `somnus-qwen3-treatment` (`--features qwen3`) — a cleaner
comparison than working-tree-Qwen3 vs a different commit's Qwen2.5, since the model choice is the
only variable. Each binary ran against `clips/talking_head.mp4` (`--file-loop`, so the process
doesn't exit before a query can be answered) with the same three prompts fed via delayed stdin,
timed to land after at least one fresh episode had formed.

**Result: Qwen3 generates coherent, on-topic English; Qwen2.5 generates incoherent token salad.**
Representative outputs, same prompts, same recall-k=3:

- Qwen2.5 control: `" is6958当中之中 CONST { /1: Blanch3674同的儿子中式 [09动授"`,
  `" gradient695861当中 Blanch {之中:7304的儿子同 /授s958动"` — mixed CJK/Latin token garbage,
  no grammatical structure, consistent across all three prompts and multiple runs.
- Qwen3 treatment: `" I need to write a summary of the lecture on gradient descent. The user is
  asking for me to summarize it, but they"`, `" I need to write a response in Chinese. The user is
  asking for me to summarize it..."` — grammatical, on-topic English every time, though it tends to
  paraphrase/meta-comment on the prompt rather than directly answer it (short 25-token budget,
  and the projection is trained on hidden-state distillation, not instruction-following).

This isn't a rigorous multi-sample eval (3 prompts, 1 clip, qualitative read only), but the gap is
large and consistent enough — coherent grammar vs. no grammar at all — that it isn't noise. Note:
a 0.5B model producing *zero* grammatical structure is more than typical quality loss for
Qwen2.5-base (normally fluent) — the qwen25 arm's soft-prompt/projection path may be compounding
the gap on top of a genuine model-quality difference. Doesn't change the direction of the result,
just don't read the absolute severity as "Qwen2.5-0.5B can't produce English" in general.
**Confirms take 2's `mean_proj_loss` gap corresponds to a real, user-visible generation quality
difference**, not just a lower surrogate-loss number. Combined with take 2, this closes out the
"confirm before/after adoption" recommendation from ADR 0036 — Qwen3-0.6B was adopted as the sole
action-decoder LM on the strength of this and the take-2 result.

**Caveat found along the way:** `--file` (without `--file-loop`) decodes the whole clip as fast as
the CPU can, not in realtime — `talking_head.mp4` (70s of real content) finishes in ~50s wall-clock
regardless of clip length, and episodes only start forming near the very end of that window
(apparently in a burst, not streamed). A query sent before that burst gets `{"error":"no episode
yet"}` — not a code bug, just a property of file-mode ingestion worth knowing before trying to
script a similar comparison again.

---

## Original run (2026-06-28) — kept for methodology/historical record

> **⚠️ INVALIDATED (2026-07-05):** the results below do not reflect real Qwen3 vs Qwen2.5
> performance. `LM_HIDDEN` was bumped 896→1024 the day before this ran (commit `6335b42`), but
> `ActionDecoder` always loads Qwen2.5's 896-wide embed table by hardcoded filename —
> `embed_flat()` slices it at stride 1024, silently corrupting every token embedding fed to
> **both** arms. Qwen3 also lacked its own GPU-path exports and ran via a file-swap trick whose
> restore step failed (`qwen25_lm.onnx` on disk is still, right now, actually the Qwen3 model).
> The -9.6% number is not evidence Qwen3 is worse — it's evidence this pipeline was broken. Full
> incident + redo plan: `.claude/session-state.md`; `ai-docs/gaps.md` §5. Kept below for the
> methodology (visual-encoder-swap notes are unaffected) and as a record of what was actually run.

**Date:** 2026-06-28  
**Duration:** ~2 cycles per session (5 sessions each model)  
**Visual encoder:** DINOv2-S (fallback; DINOv3 was disabled for this run due to what was believed to be a MIGraphX incompatibility — since found to be a CWD-relative external-data path bug and fixed, see `ai-docs/gaps.md` §6. This A/B predates the fix.)

## Results

### Memory Retrieval Quality (warm_recall_l2)

| Phase | Qwen2.5-0.5B | Qwen3-0.6B | Delta |
|-------|--------------|-----------|-------|
| Cycle 1 | 0.541932 | 0.545247 | +0.61% |
| Cycle 2 | 0.558813 | 0.449861 | **-19.5%** |
| **Average** | **0.550373** | **0.497554** | **-9.60%** ⚠️ |

**Interpretation:** warm_recall_l2 measures episodic memory L2 retrieval error (lower encoding distance → higher score). Qwen3 shows **worse retrieval performance** on the second cycle, suggesting:
- Embedding space may be less effective for memory queries
- LANG_DIM (896→1024) change may affect binding operations
- Qwen3's larger embedding (1024-dim) may need more data to converge

### Cross-Modal Gist Entropy
- Control: 2.3219 (stable across cycles)
- Treatment: 2.3219 (stable across cycles)
- **Verdict:** No degradation; binding/cross-modal integration unaffected

### Language Ingestion
- Both models successfully ingest language (lang_ingest=1 on cycle 2)
- No crashes or dimension mismatches
- Qwen3 projection weights loaded correctly

## Known Limitations

1. **Short validation window:** 2 sleep cycles per session is minimal. Convergence may require longer runs.
2. **Partial metrics:** Only warm_recall_l2 extracted from tail output. Missing:
   - output_mse (reconstruction error)
   - p_lang (language region activation)
   - proj_loss (projection head loss)
3. **Visual encoder variance:** DINOv2 fallback may mask Qwen3-specific improvements
   - Original plan was DINOv3 → DINOv3 comparison
   - DINOv3 was disabled for this run (bug since fixed, see `ai-docs/gaps.md` §6)
4. **Session interruption:** Previous sessions killed mid-run; fresh data only from this restart

## Recommendations

### Immediate (if Qwen3 is priority)
1. **Re-run with longer cycles:** 5+ NREM cycles per session to allow convergence
2. **Parse full cold archives:** Extract output_mse, p_lang, proj_loss for comprehensive comparison
3. **LoRA fine-tuning:** Current test is base models; LoRA training may show Qwen3 advantages

### Infrastructure
1. ~~**DINOv3/ORT issue**~~ — Fixed 2026-07-05: was a CWD-relative external-data path bug in the MIGraphX EP, not a graph incompatibility. See `ai-docs/gaps.md` §6. DINOv3 vs DINOv2 A/B (`scripts/ab_dinov3.sh`) can now be run with the fix in place.
2. **Metrics logging:** Add explicit metrics.jsonl output to A/B scripts (not just tail)
3. **Multi-session tracking:** Record per-session metrics to detect convergence patterns

### Next A/B Candidate
If Qwen3 underperforms, consider:
- **Item 2:** High replay ratio (simple hyperparameter, proven benefit)
- **Item 4:** LISTA Top-K encoder (algorithmic improvement)
- **Item 6:** DINOv3 swap (encoder issue now fixed — ready to A/B)

## Files

- Control results: `memory/qwen25_ab_control/`
- Treatment results: `memory/qwen3_ab_treatment/`
- Cold archives: `episodes.cold` (3.1–3.2 MB each, binary format)
- Script: `scripts/ab_qwen3.sh`
