---
status: accepted
date: 2026-05-17
deferred:
  - id: visual-audio-grounding
    what: Add grounding signal for visual and audio regions (e.g., image captions, environmental sound labels) to complement language grounding
    when: When ActionDecoder generation quality for language-grounded episodes is stable and the richer grounding signal is worth the annotation cost
  - id: lm-last-token-pooling
    what: Replace mean-pool of embed table with Qwen forward pass + last-token context for LangEmbedder (richer semantic representation)
    when: When language region prior_cosine plateaus and richer input representation is the identified bottleneck
---

# ADR 0014 — Phase 7: Auto-Grounding via Language Text Targets

**Status:** Implemented
**Date:** 2026-05-17

> **Decision:** Whenever ASR or stdin produces an episode, store that text as text_target on the episode for use as LM distillation target during NREM projection training.
> **Keywords:** auto-grounding, text_target, ASR, Whisper, grounding, LM training target, NREM projection, ingest_language

## Context

The ActionDecoder's (ADR 0012) projection (linear 640→5×LM_HIDDEN) maps episode vectors into Qwen2.5 token-context space, trained during NREM whenever an `EpisodeTrace` has `text_target` set. Wired in Phase 5/6, but `text_target` was only ever set by a manually-typed stdin query — the projection got zero training examples in every calibration run so far (`decoder_trainable_records=0` always), leaving generation incoherent (random projection weights). Phase 6's Whisper ASR (every 30s speech chunk transcribed and ingested into region 4) creates a natural (episode, text) pairing source that can ground the projection without manual annotation.

## Decision

Whenever a language ingest (ASR or stdin) produces an episode, automatically set that text as `text_target` on the episode just pushed to the episodic buffer. Implementation: in `ingest_language_line()`, immediately after `record_episode()`, tokenize via `ActionDecoder::tokenize()` and call `episodic.set_last_text_target(ids)` — only when an episode was produced and an ActionDecoder is present.

**Why this target is right:** placed on the **language-ingest episode** (from `ingest_language(&emb)`) — its 640-dim vector already includes the language region's k-WTA activation (a 128-dim slot derived from the 896-dim Qwen token embedding), so the projection learns to predict the original token sequence from a vector that already encodes a compressed version of it — well-conditioned. Targeting the speech-ingest episode instead (pushed a few ms earlier, before the language embedding computes) would place the target on a vector with no language component yet — harder task.

**Training signal sources** (both active when `--action --language` are set): ASR path — Whisper decodes 30s chunk → `lang_tx.blocking_send(text)` → `ingest_language_line()` sets target. Stdin path — user line routes to both `lang_rx` and `query_tx`, language ingest sets the target, the subsequent generation response trains against the same text. The `query_tx` manual-query path still sets target independently, so manual queries remain an additional grounding signal.

**Expected dynamics:** on a talking-head clip with ~90 ASR chunks/3276-episode cycle, ~90 episodes/cycle (~2.7%) get a text_target — `decoder_trainable_records` should go non-zero and `mean_proj_loss` should trend downward across sessions. The projection learns episode-vector→spoken-word-sequence; once trained, querying with a visually-grounded episode should produce output related to the co-occurring speech.

## Alternatives considered

- **Manual query annotation only** (the original design). Not scalable, creates a bootstrapping deadlock (random projection → useless generation → no motivation to query → projection stays random). Rejected.
- **Target the speech-ingest episode instead.** Produced before the language embedding computes — training the projection to predict text from a vector lacking the language component. Harder, noisier. Rejected.
- **Separate offline supervised fine-tuning dataset.** Contradicts the continuous-learning commitment, needs external labeling. Rejected.

## Consequences

`decoder_trainable_records` non-zero in all sessions with speech + `--action --language` (≈90/cycle for a 70s looped clip at CPU speed). `mean_proj_loss` appears in `SleepStats`/NDJSON, expected high initially (~cross-entropy of ~11 bits for a random projection over 151,936 tokens) and decreasing over sessions. Generation quality on speech-containing clips should improve from random tokens toward speech-related output over time. Grounding is partial — only teaches the projection to decode the language region; visual/audio still have no supervised text signal (richer grounding, e.g. image captions, deferred to Phase 8+).
