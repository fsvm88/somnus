---
status: accepted
date: 2026-05-12
deferred:
  - id: lora-lm-adaptation
    what: Fine-tune Qwen2.5 with LoRA so the LM adapts to the episode-vector prefix distribution, not just the projection
    when: When linear projection saturates (mean_proj_loss stops decreasing over 10+ sessions); requires LoRA + ORT pipeline
  - id: mlp-projection
    what: Replace linear projection (640→5×896) with a small MLP for more expressive episode→LM-space mapping
    when: If generation quality plateaus with the linear projection after sufficient training examples
  - id: gain-aggregation
    what: Replace per-position hidden-state L2-norm gain with a learned aggregation across layers and heads
    when: If gain modulation shows no measurable effect on perceptual quality after multi-session calibration
---

# ADR 0012 — Action Loop: Region-Token Decoder + Gain Modulation

**Date:** 2026-05-12
**Status:** Accepted

> **Decision:** ActionDecoder = linear projection (EPISODE_DIM=640 → 5×896) + frozen Qwen2.5-0.5B; attention weights drive per-region gain modulation each episode; GainWorker thread handles Qwen gain ORT; NREM trains projection via LM hidden-state distillation.
> **Keywords:** ActionDecoder, Qwen2.5, qwen25_gain, GainWorker, gain modulation, region_gain, projection, LM distillation, NREM projection, 640→5×896

## Context

Through Phase 5 the system runs a complete wake→encode→perceive→bind→store→sleep→consolidate cycle but produces no external output — a perceptual sink. `architecture.md`'s "Action loop" open question is answered here. Two candidates: **Option 2** (internal effector — episode vector's regional decomposition modulates next-step per-region gain, attention-as-action) and **Option 3** (external effector — learned projection from episodes into a frozen LLM's prompt space, text generation on query). Option 2 alone was rejected — internal gain modulation is functionally close to what self-attention already does and doesn't make the system observable externally. Option 3 alone is the right target, but its internal region-token attention is a natural gain signal that would be wasteful to discard — this ADR implements both as one mechanism.

## Decision

**`ActionDecoder`:** (1) learned projection from episode-vector space into the frozen LLM's embedding space, structured as 5 region tokens; (2) a frozen small LLM (Qwen2.5-0.5B default) processing region tokens, generating text on query; (3) gain modulation — the LLM's hidden-state L2 norms over the 5 region tokens (computed every episode, no generation) feed back as per-region gain multipliers into the next episode's k-WTA inputs.

**Model choice:** Qwen2.5-0.5B (Apache 2.0), hidden dim 896, ~1.0GB FP16/~500MB INT8, exported to ONNX and loaded via ORT (same pattern as Whisper/CLAP/DINOv2). Alternative TinyLlama-1.1B (hidden dim 2048, ~2.2GB FP16) if generation quality is insufficient — only needs a projection-output-dim change.

**Projection:** `episode_vector[640] → linear(640, 5×896) → reshape → region_tokens[5,896]` (~2.87M params, `linear_no_bias`, no per-region sub-projections — the model routes internally). Fixed region-token order: 0=audio-speech, 1=audio-env, 2=visual, 3=symbolic, 4=language.

**Gain modulation (every episode, no generation):** projection → region_tokens → one LLM forward (gain session, transformer-only, static [1,5,896], no LM head) → per-position hidden-state L2 norms `norms[5]` → softmax(temp=10) → `attn[5]` → `gain_i = clip(attn_i×5.0, 0.5, 2.0)` (clip prevents runaway amplification) → stored on `PerceptionProcessor`, multiplies each region's encoder embedding on the *next* episode's `ingest_*` call, before k-WTA. Gains reset to `[1.0;5]` at each sleep entry (no stale prior across the wake/sleep boundary). Note: original design used last-layer/last-head attention at position 3; implementation uses per-position hidden-state L2 norms instead — simpler, more stable, doesn't depend on one head's pattern. Interpretation: "how much the language system cares about each perceptual region shapes what perception emphasises next" — parallels linguistic priming of visual attention, consistent with the top-down prior path (ADR 0010).

**Text generation (on query):** stdin, one line = one query. Tokenize → embed via the LLM's frozen table → prepend `region_tokens` as a soft prefix `[region_tokens | query_embeddings]` → autoregressive decode (greedy/top-p, `--action-max-tokens` default 128) → stdout JSON `{"ts","query","response"}`. Non-blocking stdin read (dedicated `tokio::task`, `mpsc` queue), main loop drains and generates synchronously (blocks the episode loop) — at Qwen2.5-0.5B/128 tokens/CPU, ~2-8s, acceptable for the current non-real-time research loop.

**Training signal — text bypass.** Injected text is tokenized, stored as `text_target: Option<Vec<u32>>` on the `EpisodeRecord` (None if no text). Auto-grounding (ADR 0014) makes every language ingest produce a target without manual queries. During NREM, for each trace with `text_target.is_some()`: run `replay_step` normally (SAE+head reconstruction), then (see 2026-05-30 amendment) precompute LM hidden-state distillation targets via `qwen25_lm.onnx`, mean-pool, run the projection on the stored episode vector, minimize MSE against the target, AdamW step for the projection only (fixed lr=1e-4, separate from the cognitive layer). Cold-start: no text-bypass history → projection never trains → generation is incoherent — correct failure mode, reflects how much the system has been taught. Startup INFO log reports `text_target`-bearing record count for training-density visibility.

**New CLI params:** `--action-max-tokens` (128), `--action-top-p` (1.0=greedy), `--action-gain` (true), `--action-lm-dtype` (int8, or fp16). `--action-gain false` disables the per-episode LLM forward pass entirely (gain stays `[1,1,1,1,1]`) — useful when no snapshot exists and the projection is random (random attention would inject noise into gain).

## Consequences

- **VRAM:** Qwen2.5-0.5B INT8 adds ~500MB; total ~2-3GB, well within 14GB.
- **Latency:** one LLM forward (4 tokens, no generation) adds ~5-15ms CPU/episode — acceptable, profile before tightening wake latency.
- **`EpisodeRecord` schema:** `text_target: Option<Vec<u32>>` added, old cold-archive records compatible (treated as None), snapshot restore unaffected.
- **VarMap:** one new key, `action_decoder.projection.weight` (`[4480,640]`). Missing in old snapshots → fresh-init (logged INFO), incoherent generation until trained.
- **Training density:** projection trains only on text-bypass episodes; sessions without any produce no signal. `decoder_trainable_records` logged at each NREM entry — consistently 0 means the decoder isn't learning.
- **Gain lag:** gain for episode N comes from episode N-1's attention — structurally identical to the inter-episode prior path (ADR 0010), same acceptable lag.
- **Gain+prior interaction:** both modify the next episode's k-WTA input at different points — `input_i = gain_i × ((1-β)×raw_i + β×prior_i)`, gain a scalar multiplier applied first, prior a vector offset blended after. Well-defined, complementary.
- **Circular target risk:** heavy, repetitively-phrased text bypass would overfit the decoder to vocabulary rather than sensory content — vary phrasing to cover different perceptual aspects.

## Amendment — distillation target for projection training (2026-05-30)

Changed the projection training target from a static embed-table mean to **LM contextualised hidden states**. Previously: project→region tokens→MSE against `mean(embed_table[text_target_ids])` (a bag-of-embeddings, discards inter-token relationships). Now: `text_target` tokens run through the full Qwen transformer (`qwen25_lm.onnx`, `last_hidden_state`), mean-pooled + L2-normalised, and the projection trains toward *that* — captures how Qwen contextualises the whole sequence. No gradient through the frozen LM (ONNX/ORT inference only); only the projection (Burn/AdamW) updates. `qwen25_lm.onnx` already exported `last_hidden_state` (`Qwen25LMWrapper` emits both it and `logits`) — the Rust code simply wasn't reading it, no new export needed. LM hidden states precomputed once per NREM cycle (before the replay loop), keeping ORT cost off the hot path. Old `projection_step_batch` (embed-table target) retained, marked `#[allow(dead_code)]`.

## Alternatives considered

- **Separate gain and decoder (options 2/3 independently).** Rejected — doubles architectural surface for nothing, the generation-path LLM forward already produces gain's attention weights.
- **MLP instead of linear projection.** A linear projection is sufficient for the episode→LM-hidden subspace mapping, consistent with the `linear_no_bias` convention throughout. Revisit if generation quality plateaus.
- **LoRA-train the LLM instead of frozen.** Would let the LLM adapt to the episode-vector prefix distribution — rejected for Phase 6 (LoRA+ORT non-trivial to wire, projection-alone is the right first experiment); revisit Phase 7+ if projection saturates.
- **Audio output (TTS) as effector.** More perceptually rich but needs a vocoder — a large new dependency. Text is the correct first effector, audio can layer on later.
- **Full layer/head attention aggregation for gain.** Last-layer-last-head is a heuristic; a learned weighted aggregation would be more principled but adds complexity — start simple, switch if gain shows no measurable perceptual-quality effect.
