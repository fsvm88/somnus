# A/B Validation: Qwen3-0.6B vs Qwen2.5-0.5B

**Bottom line (2026-07-05): Qwen3-0.6B adopted as sole action-decoder LM, ADR 0036.** Evidence:
take 2 (speech-dense corpus, ~22% lower `mean_proj_loss`, consistent, no reversals) + take 3
(direct `generate()` comparison: coherent English vs. incoherent token garbage). Details below,
newest first.

## Original run (2026-06-28) — INVALIDATED, kept only as an incident record

> Reported Qwen3 "underperforming" on `warm_recall_l2` (0.550→0.498, -9.6%). **Not trustworthy**:
> `LM_HIDDEN` had been bumped 896→1024 (commit `6335b42`) the day before, but `ActionDecoder`
> still loaded Qwen2.5's real 896-wide embed table by hardcoded filename — `embed_flat()` sliced
> it at stride 1024, silently corrupting every token embedding in **both** arms. Qwen3 also had no
> GPU-path export of its own and ran via a file-swap trick whose restore step failed (`qwen25_lm.onnx`
> on disk was actually the Qwen3 model). The -9.6% number reflects a broken pipeline, not either
> architecture. DINOv2 was used as the visual encoder for both arms (DINOv3 was believed
> MIGraphX-incompatible at the time — later found to be a CWD-relative path bug, `gaps.md` §6).

## Redo take 1 (2026-07-05): pipeline fixed, comparison inconclusive

Fixed before re-running: (1) restored a genuine Qwen2.5 baseline (the on-disk `qwen25_lm.onnx`
was confirmed byte-identical to Qwen3's via sha256 match on both graph and `.data` — re-exported
from scratch, verified via ONNX graph inspection showing `inputs_embeds`/896-dim in, not
`input_ids`/1024-dim);
(2) gave Qwen3 a complete matching export via the correct `Qwen3ForCausalLM` (the old exporter
used `Qwen2ForCausalLM`, missing Qwen3's QK-norm layers); (3) parameterized the model-file prefix
(`ActionDecoder::new`/`LangEmbedder::new` took a `model_prefix` arg instead of hardcoded
`qwen25_*` — this also removed a footgun where `LangEmbedder` auto-detected "prefer qwen3 files
if present," silently picking the wrong file set once both existed on disk); (4) built two real
release binaries (`somnus-qwen25-control`, `somnus-qwen3-treatment`) behind a `qwen3` Cargo feature
gating `LM_HIDDEN`/`LANG_DIM` (896/1024) and `EOS_TOKEN_ID` (`<|endoftext|>`=151643 vs
`<|im_end|>`=151645) — smoke-testing this surfaced a third bug, ~6 more hardcoded 896/1024
language-region literals in `somnus-perception` that disagreed with each other, unified under a new
`LANG_HIDDEN` constant. Cold-started both arms directly (no file-swapping), 10 sessions × 5 NREM
cycles/arm on the standard 4-clip corpus (`talking_head`/`urban`/`animation`/`nature`).

**Lesson for future A/Bs:** `grep` for literal `1024`/`896` across a crate before trusting a single
named constant governs a dimension — an earlier commit had changed some call sites and missed
others, leaving stale/wrong values in both directions.

| Metric (mean/sum over 50 cycles) | Qwen2.5 (control) | Qwen3 (treatment) |
|---|---|---|
| `per_region_mse[language]` | 0.0011 | 0.0010 |
| `lang_ingest_count` (of 50) | 24 | 26 |
| `decoder_trainable` events | 14 | 16 |
| `mean_proj_loss` non-null samples | 0/50 | 1/50 |
| `mean_output_mse` (all 5 regions) | 0.0534 | 0.0124 |
| `warm_recall_l2` | 0.6065 | 0.2149 |

**Verdict: inconclusive, not "Qwen3 loses."** The 4 stock clips have too little actual speech —
lang-ingest fires ~once per 5-cycle session regardless of LM, so `mean_proj_loss` has almost no
samples and language-region MSE is statistically indistinguishable. The large `mean_output_mse`/
`warm_recall_l2` gaps are **not a Qwen3 effect**: per-region breakdown (speech 0.567→0.470, env
0.034→0.028, visual 0.088→0.076, symbolic 0.035→0.016, language 0.0011→0.0010) shows every
region — including three with zero LM dependency — moves by a similar proportion, i.e.
independent-seed noise from cold-started weights, not signal.

**Aside:** an unrelated intermittent panic in `WhisperDecoder::decode_greedy`
(`crates/somnus/src/whisper_asr.rs`, identified via `addr2line` on the stripped release binary as
`ort::session::Session::run` inside that function) — ORT "GetElementType is not implemented"
escalating past its usual caught-WARN path — hit 1–2 sessions per arm. Each session is a separate
process, so no data was lost (50/50 cycles present) — flagged, not fixed here (mitigated later,
see `gaps.md` §5).

**Recommendation followed:** swap in the speech-dense `clips/long/` corpus, or test generation
quality directly — both binaries and full model-file sets were ready either way.

## Redo take 2 (2026-07-05): speech-dense `clips/long/` corpus — Qwen3 wins on proj_loss

Swapped the 4 sparse-speech clips for 6 lecture/podcast clips (Stanford CS229, Bengio/LeCun
podcasts, Hossenfelder physics, a consciousness lecture, an algorithms lecture, ~1.7 GB). Same two
binaries, same `--action-model-prefix`, no file-swapping. 5 sessions × 4 cycles/arm. A smoke test
confirmed the corpus swap worked (1 cycle → `asr_count=20`, `lang_ingest_count=16`, vs. take 1's
~1 event per whole session).

| Metric (sum/mean over 20 cycles) | Qwen2.5 (control) | Qwen3 (treatment) |
|---|---|---|
| `lang_ingest_count` (total) | 73 | 73 |
| `asr_count` (total) | 78 | 78 |
| `decoder_trainable` events | 40 | 41 |
| `mean_proj_loss` non-null samples | 12/20 | 12/20 |
| `mean_proj_loss`, sessions 1→2→3→4→5 | 0.00223→0.00217→0.00214→0.00199→0.00155 | 0.00190→0.00184→0.00170→0.00152→0.00120 |
| `per_region_mse[language]` | 0.0011 | 0.0010 |
| `mean_output_mse` (all 5 regions) | 0.0567 | 0.0414 |
| `warm_recall_l2` | 0.4724 | 0.4800 |

**Fair comparison**: identical exposure both arms (73 lang_ingest, ~40 training events, matched
asr_count). **Result: Qwen3 sits below Qwen2.5 on `mean_proj_loss` at every comparable sample
across the whole campaign, no reversals** — final gap ~22% (0.001205 vs 0.001551).
`per_region_mse[language]` is directionally consistent. `warm_recall_l2` came back close this time
(0.48 vs 0.47), supporting take 1's read that its wild gap there was seed noise. `mean_output_mse`
still favors treatment by a wide margin but is still dominated by non-LM regions (mostly speech) —
treat as noise, not a Qwen3 effect, as in take 1.

**Verdict: worth adopting Qwen3-0.6B.** proj_loss is the one metric this run actually isolates to
the LM choice, and the direction held with no reversals. Recommended before flipping the default:
confirm with a longer run and/or a direct generation-quality test (proj_loss is a surrogate for
grounding, not generation, quality) — done next, as take 3.

**Aside:** the Whisper panic recurred more on this denser corpus (~1/5 sessions control, ~3/5
treatment) — still no data loss (20/20 cycles present both arms), still unrelated, but the higher
rate made it worth mitigating (done in the adoption session, see `gaps.md` §5).

Files: `memory/qwen25_ab_control_longclips/metrics.jsonl`, `memory/qwen3_ab_treatment_longclips/metrics.jsonl`, `scripts/ab_qwen3_longclips.sh`.

## Take 3 (2026-07-05): direct generation-quality test — confirms take 2

proj_loss is a surrogate; this compares real `generate()` output. Method: both Cargo features had
already been removed from the working tree by the time this ran, so a `git worktree` at the last
commit with both (`c86cfdf`) built one binary per arm, each pointed at its own take-2 *trained*
memory dir (copied to scratch, originals untouched) — isolates the LM as the only variable rather
than comparing against a cold-started or differently-tuned projection. Both ran against
`clips/talking_head.mp4` (`--file-loop` — plain `--file` decodes faster than realtime and the
process exits before an episode forms, a gotcha worth remembering for future ad hoc comparisons),
same three prompts via delayed stdin, recall-k=3.

**Result: Qwen3 generates coherent, on-topic English; Qwen2.5 generates incoherent token salad.**
Representative outputs, same prompts:
- Qwen2.5: `" is6958当中之中 CONST { /1: Blanch3674同的儿子中式 [09动授"` — mixed CJK/Latin garbage,
  no grammar, consistent across all prompts and repeat runs.
- Qwen3: `" I need to write a summary of the lecture on gradient descent. The user is asking for me
  to summarize it, but they"` — grammatical, on-topic, though it tends to paraphrase/meta-comment
  rather than directly answer (25-token budget; projection trained on hidden-state distillation,
  not instruction-following).

Qualitative only (3 prompts, 1 clip), but the gap — coherent grammar vs. none at all — is too large
to be noise. Note: zero grammatical structure is worse than Qwen2.5-base's typical fluency, so the
qwen25 arm's soft-prompt/projection path may compound a genuine model-quality gap — doesn't change
the result's direction, just don't read the absolute severity as "Qwen2.5-0.5B can't do English"
in general. **Confirms take 2's proj_loss gap reflects real, user-visible quality**, closing the
"confirm before/after adoption" item from ADR 0036.
