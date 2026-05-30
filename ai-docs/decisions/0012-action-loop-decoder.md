# ADR 0012 — Action Loop: Region-Token Decoder + Gain Modulation

**Date:** 2026-05-12  
**Status:** Accepted

## Context

Through Phase 5 and its extensions the system has a complete wake→encode→perceive→bind→store→sleep→consolidate cycle, but it produces no external output. It is a perceptual sink. The open architectural question in `architecture.md` ("Action loop — does the system have one in early phases? Effector: internal state only, or external?") is answered here.

Two candidate designs were considered in isolation:

- **Option 2 (internal effector):** use the episode vector's predicted regional decomposition to modulate per-region gain on the next perceptual step — attention-as-action. Pure internal feedback loop.
- **Option 3 (external effector):** a learned projection from episode vectors into a frozen LLM's prompt space, enabling text generation on query. External output, requires a training signal.

Option 2 alone was rejected: internal gain modulation is functionally close to what transformer self-attention already achieves; it does not make the system observable from outside. Option 3 alone is the right target, but its internal attention structure (over region tokens) is a natural gain signal — disposing of it would be wasteful. This ADR implements both as a unified mechanism.

## Decision

Add an **`ActionDecoder`** component consisting of:

1. A **learned projection** from episode vector space into the frozen LLM's embedding space, structured as 4 region tokens.
2. A **frozen small LLM** (Qwen2.5-0.5B, default) that processes the region tokens and, when queried, generates text.
3. A **gain modulation** path: the LLM's self-attention weights over the 4 region tokens — computed on every episode, with no generation — feed back as per-region gain multipliers into the next episode's k-WTA inputs.

### Model choice

**Default: Qwen2.5-0.5B** (Apache 2.0).
- Hidden dimension: 896
- VRAM at FP16: ~1.0 GB; at INT8: ~500 MB
- Exported to ONNX (`models/qwen2.5-0.5b.onnx`) via `scripts/export_models.py`, loaded via ORT — same pattern as Whisper/CLAP/DINOv2

Alternative: **TinyLlama-1.1B** (hidden dim 2048, ~2.2 GB FP16) if generation quality is insufficient; only requires changing the projection output dimension. No other code changes.

### Projection

```
episode_vector [512]
  ↓  linear(512, 4 × 896)   ← learned; ~1.8 M params
  ↓  reshape → region_tokens [4, 896]
```

Region token layout (fixed order):

| Index | Region        |
|-------|---------------|
| 0     | audio-speech  |
| 1     | audio-env     |
| 2     | visual        |
| 3     | symbolic      |

The projection is a single `linear_no_bias(EPISODE_DIM, 4 * LM_HIDDEN)`. No per-region sub-projections; the model learns to route internally.

### Gain modulation (every episode, no generation)

After an episode vector is produced (post-settling, ADR 0011):

1. Run the projection → `region_tokens [4, 896]`.
2. Run one LLM forward pass over the 4 tokens (no autoregressive generation; a single non-causal or causal pass with the 4-token sequence). Extract the self-attention weights of the **last layer, last head** at position 3 (symbolic token) attending to positions 0–3.
3. Softmax over the 4 weights → `attn [4]` summing to 1.0.
4. Scale to gain multipliers: `gain_i = clip(attn_i × 4.0, 0.5, 2.0)`. Clipping prevents runaway amplification.
5. Store `gain [4]` on `PerceptionProcessor`. On the **next** episode's `ingest_*` call, multiply each region's encoder embedding by its gain before the k-WTA forward pass.

Gain multipliers are reset to `[1.0, 1.0, 1.0, 1.0]` at each sleep entry so a stale prior does not persist across the wake/sleep boundary.

The cognitive interpretation: "how much the language system cares about each perceptual region shapes what the perceptual system emphasises next." This parallels linguistic priming of visual attention and is architecturally consistent with the top-down prior path (ADR 0010).

### Text generation (on query trigger)

During wake, the main loop watches for a query on **stdin** (one line = one query). On receipt:

1. Tokenize the query string.
2. Embed the query tokens using the LLM's embedding table (frozen).
3. Prepend `region_tokens` as a soft prefix: `[region_tokens | query_token_embeddings]`.
4. Autoregressive decode (greedy or top-p, configurable) up to `--action-max-tokens` (default 128).
5. Write the response to stdout as a JSON line: `{"ts": <unix_ms>, "query": "...", "response": "..."}`.

Query handling is non-blocking: a dedicated `tokio::task` reads stdin; queries are queued in an `mpsc` channel; the main loop drains the channel on each tick and runs generation synchronously (blocking the episode loop for the duration). At Qwen2.5-0.5B and 128 tokens on CPU this is expected to be 2–8 seconds — acceptable for the current non-real-time research loop.

### Training signal

**Text bypass is the decoder's training signal.** When text is injected via `--text-bypass`, the input string is tokenized and stored as `text_target: Option<Vec<u32>>` on the corresponding `EpisodeRecord`. Episodes without text bypass carry `None`.

During NREM replay, for each `EpisodeRecord` where `text_target.is_some()`:

1. Run `replay_step` as normal (SAE + output-head reconstruction).
2. Additionally: run the projection on the episode's stored vector → `region_tokens`, then run LLM forward with `[region_tokens | text_target_embeddings[:-1]]` → logits, cross-entropy vs. `text_target[1:]`. Backprop through the projection only (LLM frozen). AdamW step for the projection parameters (separate parameter group from the cognitive layer so NREM neuromodulation — ADR 0008 cosine schedule — does not touch LR for the projection; projection uses a fixed LR of 1e-4).

Cold-start: with no text-bypass history the projection is never trained and generation is incoherent. This is the correct failure mode — it reflects how much the system has been taught. A startup INFO log reports the count of `text_target`-bearing records in the current episodic buffer so the training density is visible.

### New parameters

| CLI arg | Default | Meaning |
|---|---|---|
| `--action-max-tokens` | `128` | Max tokens to generate per query. |
| `--action-top-p` | `1.0` | Top-p sampling (1.0 = greedy). |
| `--action-gain` | `true` | Enable gain modulation from LLM attention. |
| `--action-lm-dtype` | `int8` | ORT dtype for LLM (fp16 / int8). |

`--action-gain false` disables the per-episode LLM forward pass entirely; gain stays `[1.0, 1.0, 1.0, 1.0]`. Useful when no snapshot exists and the projection is random (random attention weights would inject noise into gain).

## Consequences

**VRAM.** Qwen2.5-0.5B at INT8 adds ~500 MB. Total estimated VRAM with existing components: ~2–3 GB. Well within the 14 GB budget.

**Latency per episode.** One LLM forward (4 tokens, no generation) adds ~5–15 ms on CPU. Acceptable for the current research loop; profile before tightening wake latency.

**EpisodeRecord schema change.** `text_target: Option<Vec<u32>>` added. Old cold-archive records are compatible (field treated as None). Snapshot restore unaffected.

**VarMap.** One new key: `action_decoder.projection.weight` (shape `[4 * LM_HIDDEN, EPISODE_DIM]`). Old snapshots missing this key get fresh-initialized projection (logged at INFO); generation is incoherent until trained.

**Training density.** The projection trains only on episodes with text bypass. Sessions without any text bypass produce no decoder training signal. The binary logs `decoder_trainable_records` at each NREM entry; if this is consistently 0, the decoder is not learning.

**Gain modulation is always one episode behind.** Gain for episode N is computed from episode N−1's attention weights. This is structurally identical to the inter-episode prior path (ADR 0010) and carries the same lag; it is acceptable.

**Gain and prior interaction.** Both gain (this ADR) and priors (ADR 0010) modify the k-WTA input for the next episode, but at different points: gain scales the raw encoder embedding before k-WTA; priors blend into the input after gain is applied. The order is `gain → prior blend → k-WTA`. With both enabled and `--prior-beta > 0`, the effective input is:

```
input_i = gain_i × ((1 − β) × raw_i + β × prior_i)
```

This is well-defined; gain and prior are complementary — gain is a scalar multiplier, prior is a vector offset in encoder space.

**Circular target risk.** The projection trains against text-bypass targets. If text bypass is used heavily and always with the same phrasing, the decoder will overfit to the user's vocabulary rather than the sensory content. Vary text-bypass phrasing to cover different aspects of what the system is perceiving.

## Amendment — distillation target for projection training (2026-05-30)

The projection training target was changed from **static embed-table mean** to **LM contextualised hidden states**.

Previously: for a language-tagged trace, project episode → region tokens → MSE against mean(embed_table[text_target_ids]).
Now: text_target tokens are embedded, run through the full Qwen transformer (qwen25_lm.onnx, `last_hidden_state` output), mean-pooled over token positions, and L2-normalised. The projection then trains toward this contextualised target. The key difference: the static embed-table mean is a bag-of-embeddings (inter-token relationships discarded); the LM hidden state captures how Qwen contextualises the entire text sequence. Neither requires gradient through the frozen LM — the LM runs via ONNX/ORT in inference mode; only the projection (Burn/AdamW) is updated.

`qwen25_lm.onnx` already exported `last_hidden_state` (Qwen25LMWrapper in export_models.py emits both `last_hidden_state` and `logits`); the Rust code was simply not reading it. No new model export needed.

LM hidden states are precomputed once per NREM cycle (before the replay loop) so the ORT session cost is not on the hot NREM path. The `projection_step_batch` method (old embed-table target) is retained but marked `#[allow(dead_code)]`.

## Alternatives considered

- **Separate gain and decoder (options 2 and 3 independently):** doubling the architectural surface for no benefit — the LLM forward pass needed for generation already produces the attention weights needed for gain. Rejected.
- **MLP projection instead of linear:** adds expressivity but a linear projection is sufficient to learn a subspace mapping from episode space to LM hidden space, consistent with the `linear_no_bias` convention throughout the cognitive layer. Revisit if generation quality plateaus.
- **Train LLM with LoRA instead of frozen:** would allow the LLM to adapt to the episode-vector prefix distribution. Rejected for Phase 6: LoRA + ORT is non-trivial to wire, and the projection alone is the right first experiment. Revisit in Phase 7+ if projection saturates.
- **Audio output (TTS) as effector:** more perceptually rich but requires a vocoder, introducing a large new dependency. Text output is the correct first effector; audio out can be layered on top once text generation works.
- **Attention over all layers / all heads for gain:** using last-layer last-head attention is a heuristic. A learned aggregation (weighted sum over heads/layers) would be more principled but adds parameters and complexity. Start simple; switch if gain modulation has no measurable effect on perceptual quality.
