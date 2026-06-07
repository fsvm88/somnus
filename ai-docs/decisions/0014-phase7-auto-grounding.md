# ADR 0014 — Phase 7: Auto-Grounding via Language Text Targets

**Status:** Implemented
**Date:** 2026-05-17

> **Decision:** Whenever ASR or stdin produces an episode, store that text as text_target on the episode for use as LM distillation target during NREM projection training.  
> **Keywords:** auto-grounding, text_target, ASR, Whisper, grounding, LM training target, NREM projection, ingest_language

---

## Context

The ActionDecoder (ADR 0012) has a projection layer (linear 640→5×LM_HIDDEN) that maps episode
vectors into Qwen2.5 token-context space for autoregressive generation. That projection is trained
during NREM when an `EpisodeTrace` has a `text_target` field set. The mechanism was wired in Phase
5/6, but `text_target` was only ever set when the user manually typed a query at stdin — meaning the
projection received zero training examples in all calibration runs so far (`decoder_trainable_records
= 0` in every session). Generation remained incoherent because the projection weights were random.

Phase 6 added Whisper ASR: every 30-second speech chunk is now transcribed and the text is ingested
into the language region (region 4). This creates a natural source of (episode, text) pairs that
can ground the projection without any manual annotation.

---

## Decision

Whenever a language ingest (from ASR or stdin) produces an episode, automatically set that text as
the `text_target` on the episode that was just pushed to the episodic buffer.

Implementation: in `ingest_language_line()` in `main.rs`, immediately after `record_episode()`,
tokenize the ingested text with `ActionDecoder::tokenize()` and call
`episodic.set_last_text_target(ids)`. This executes only when an episode was produced (`out.episode
.is_some()`) and an ActionDecoder is present.

### Why this is the right target

The text_target is placed on the **language-ingest episode** — the episode that was produced by
`ingest_language(&emb)`. This episode's 640-dim vector includes the language region's k-WTA
activation (a 128-dim slot derived from the 896-dim Qwen token embedding of the text). The
projection is therefore trained to predict the original token sequence from an episode vector that
already encodes a compressed version of it — a well-conditioned learning problem.

The alternative of targeting the speech-ingest episode (which was pushed a few milliseconds earlier,
before the language embedding was computed) would place the text_target on a vector that has no
language component yet, making the task harder for the projection.

### Sources of training signal

Both ASR transcriptions and explicit stdin lines now ground episodes automatically when
`--action --language` are both set:

- **ASR path:** Whisper decodes a 30-second chunk → `lang_tx.blocking_send(text)` → main loop
  `ingest_language_line()` → text_target set.
- **Stdin path:** user types a line → routed to both `lang_rx` and `query_tx` → language ingest
  sets text_target; the subsequent generation response is trained against the same text.

The `query_tx` path (manual query → generation) still sets text_target independently as before, so
manual queries continue to work as an additional grounding signal.

### Expected training dynamics

On a talking-head clip with ~90 ASR chunks per 3276-episode wake cycle, roughly 90 episodes per
cycle (~2.7%) will have a text_target. `decoder_trainable_records` should now be non-zero and
`mean_proj_loss` should appear in metrics and trend downward across sessions.

The projection learns: episode vector → spoken word sequence. After sufficient training, querying
the ActionDecoder with a visually-grounded episode should produce output related to the speech that
co-occurred with that visual context.

---

## Alternatives considered

**Manual query annotation only:** The original design. Requires the user to type while watching the
video to create training examples. Not scalable and creates a bootstrapping deadlock (random
projection → useless generation → no motivation to query → projection stays random). Rejected.

**Set text_target on the speech-ingest episode instead of the language-ingest episode:** The
speech-ingest episode is produced a few milliseconds before the language embedding is computed (ASR
runs synchronously in the audio encoder task). Placing text_target there means training a projection
to predict text from an episode vector that lacks the language component. Harder problem, noisier
gradient. Rejected.

**Separate supervised fine-tuning dataset:** Collect (clip, transcript) pairs offline and fine-tune
the projection outside the online learning loop. Contradicts the continuous-learning commitment and
requires external labeling. Rejected.

---

## Consequences

- `decoder_trainable_records` will be non-zero in all sessions that include speech and use
  `--action --language`. The value equals the number of language-ingest episodes per wake cycle
  (typically ~90 for a 70-second clip looped at CPU speed).
- `mean_proj_loss` will appear in `SleepStats` and metrics NDJSON. Expected to be high initially
  (~cross-entropy of ~11 bits for random projection over 151,936-token vocabulary) and decrease
  over sessions.
- After sufficient training, ActionDecoder generation quality on speech-containing clips should
  improve from random tokens toward speech-related output.
- The grounding is partial: it teaches the projection to decode the language region. Visual and
  audio regions still provide no supervised text signal. Richer grounding (e.g., image captioning
  labels) would require additional annotation sources — deferred to Phase 8+.
