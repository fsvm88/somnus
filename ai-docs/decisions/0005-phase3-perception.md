---
status: accepted
date: 2026-05-10
deferred:
  - id: kwta-to-sae
    what: Upgrade k-WTA to full differentiable SAE (coding loss + reconstruction loss jointly) once sufficient replay history exists
    when: When NREM replay quality is stable and output_mse has converged (Phase 8+)
  - id: grown-units-binding
    what: Expose grown units (above N_UNITS=1024 baseline) to the binding area — currently clamped by to_dense_n(N_UNITS)
    when: When growth stabilises and the binding area redesign justifies the episode vector dimension change; see ADR 0008
---

# 0005. Phase 3: somnus-perception crate — sparse coding, k-WTA, binding area, SAE decoder

- **Status:** accepted
- **Date:** 2026-05-10

> **Decision:** 5-region k-WTA (N_UNITS=1024, K_ACTIVE=20) with duty-cycle boosting + SAE decoder per region; 4-head cross-attention binding → 640-dim episode vector; somnus-perception crate.
> **Keywords:** k-WTA, N_UNITS, K_ACTIVE, SparseActivation, SAE, duty cycle, somnus-perception, EPISODE_DIM, cross-attention, binding area

## Context

Phase 2 delivers three frozen-encoder streams: audio-speech (Whisper-tiny, 384), audio-env (CLAP-HTSAT, 512), visual (DINOv2-S/14, 384). Phase 3 maps these into the cognitive layer: sparse activations per region, a unified episode vector from binding, and a reconstruction-error signal for sleep-time growth (Phase 5) and replay prioritization. Roadmap exit criterion: sparse activations per frame, binding outputs a fixed-size episode vector, reconstruction error observable.

## Decision

**New crate `somnus-perception`**, `candle-core`+`candle-nn` (0.10.x, CPU for Phase 3, CUDA flag deferred) — cognitive-layer weights live here so Phase 5 can train them in place.

**Regions** (four, all starting simultaneously, all n=1024):

| Region | Input source | Input dim | Output dim |
|---|---|---|---|
| audio-speech | Whisper-tiny | 384 | 1024 |
| audio-env | CLAP | 512 | 1024 |
| visual | DINOv2 | 384 | 1024 |
| symbolic | Whisper-tiny (audio-speech output) | 384 | 1024 |

Symbolic receives audio-speech in Phase 3 (words heard are the primary symbolic input before text bypass, Phase 6+) — shares the encoder output with audio-speech but has independent k-WTA weights, learns different codes.

**Region size n=1024, k=20:** k²/n≈0.4 expected overlap between random concepts — negligible interference; power-of-2 for CUDA alignment; starting small is intentional so cascade-correlation growth (Phase 5) is visible as an event, not pre-empted. Target duty cycle k/n=1.95%, boosting keeps all units alive.

**k-WTA layer:** `z=Wx+b`, keep top-k=20, zero the rest. Applied on CPU-extracted data (candle tensor→Vec<f32>→partial sort→`SparseActivation`); weight matrix stays in candle for Phase 5 gradient support. **Boosting:** per-unit duty cycle as an EMA (α=1/1000) — if below target, pre-activation boosted by `BOOST_STRENGTH×(target-duty_cycle[i])`, preventing dead units without explicit reinit.

**`SparseActivation{region, indices:Vec<u16>, values:Vec<f32>, n:usize}`** — indices sorted ascending, `n` always 1024. 20×(2+4)=120 bytes/activation, negligible at any scale.

**Binding area:** concatenates four 1024-dim dense-reconstructed activations (4096 total), linear projection to 512-dim episode vector (matches the replay-strategy note's VRAM budget: 1KB/episode FP16 × 12M ≈ 12GB). Plain linear in Phase 3; cross-attention upgrade is Phase 6+. Produces an `EpisodeVector` only once all four regions have at least one activation — `PerceptionProcessor` buffers the latest per-region activation until then (first episode appears after ~10s, when the first env embedding arrives; visual/speech update at their own rates before that).

**SAE decoder:** one thin reverse-linear (1024→input_dim) per region, randomly initialised in Phase 3, reconstruction error logged as a metric (trained in Phase 5 NREM replay, already wired so Phase 5 needs no structural change). `err = MSE(x, W_dec @ sparse_to_dense(act))`.

**`PerceptionProcessor`** (public): owns all four k-WTA layers, four SAE decoders, binding area. `ingest_speech`/`ingest_env`/`ingest_visual(values) -> PerceptionResult<PerceptionOutput>` — each updates the relevant region(s) (speech also updates symbolic), computes reconstruction error, optionally produces an `EpisodeVector` once all four regions are populated.

**Wiring in `somnus`:** called inline in the main async loop, not `spawn_blocking` — CPU k-WTA at n=1024 takes ~50µs, well under async scheduling granularity. Errors and episode vectors logged via `tracing::info!` and written to the NDJSON sink alongside embeddings.

## Alternatives considered

- **SAE instead of k-WTA for the encoder.** More principled (differentiable coding loss) but needs replay data that doesn't exist until Phase 5 — k-WTA with random weights still gives sparse codes and a valid (if untrained) reconstruction-error signal; upgradeable to SAE in Phase 4-5.
- **`ndarray` instead of `candle`.** Simpler, no autograd overhead — rejected since Phase 5 needs candle autograd; better to pay the setup cost once than refactor all layer types later.
- **Skip the SAE decoder in Phase 3.** Cheap to add (one linear/region), makes the growth-trigger signal observable from day one — deferring would need structural changes in Phase 5.
- **Binding-area cross-attention now.** Architecturally correct but deferred to Phase 6+ per roadmap — concat+linear gives a valid fixed-size episode vector immediately.

## Consequences

- `candle-core`/`candle-nn = "0.10"` added to workspace deps (CPU only, `cuda` feature deferred); `somnus-perception` added to the workspace, `somnus` gains the dependency.
- Resolves `architecture.md`'s "Sparse-coding scheme" open question: k-WTA with boosting, n=1024, k=20, target duty cycle 1.95%.
- Resolves "Initial regional set": all four Phase-3 regions, symbolic fed from audio-speech until text bypass arrives.
- Binding-area output dim (512) is now a committed Phase 3↔Phase 4 interface point (episodic buffer row size).
