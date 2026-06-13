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

Phase 2 delivers three frozen-encoder embedding streams:

| Stream | Encoder | Dim |
|--------|---------|-----|
| audio-speech | Whisper-tiny | 384 |
| audio-env | CLAP-HTSAT-unfused | 512 |
| visual | DINOv2-S/14 | 384 |

Phase 3 maps these into the cognitive layer: sparse activations per region,
a unified episode vector from the binding area, and a reconstruction-error
signal that will drive sleep-time growth (Phase 5) and replay prioritization.

The roadmap's Phase 3 exit criterion: sparse activations produced per frame;
binding area outputs a fixed-size episode vector; reconstruction error is
observable as a metric.

## Decision

### New crate: `crates/somnus-perception`

Uses `candle-core` + `candle-nn` (0.10.x, CPU for Phase 3; CUDA flag when
sleep-time training becomes the bottleneck). The cognitive-layer weights live
here so Phase 5 can train them in-place without a refactor.

### Regions

Four regions, all starting simultaneously. All have n=1024 units.

| Region | Input source | Input dim | Output dim |
|--------|-------------|-----------|------------|
| audio-speech | Whisper-tiny | 384 | 1024 |
| audio-env | CLAP | 512 | 1024 |
| visual | DINOv2 | 384 | 1024 |
| symbolic | Whisper-tiny (audio-speech output) | 384 | 1024 |

Symbolic receives audio-speech in Phase 3 because words heard *are* the
primary symbolic input before text bypass is wired (Phase 6+). The two
regions (audio-speech and symbolic) share the same encoder output but have
independent k-WTA weight matrices and learn different codes.

### Region size: n=1024, k=20

k²/n = 400/1024 ≈ 0.4 expected unit overlap between any two random concepts —
negligible interference. Power-of-2 dimension for CUDA alignment. Starting
small is intentional: cascade-correlation growth (Phase 5) should be visible
as an event, not pre-empted by over-allocation.

Target duty cycle = k/n = 1.95%. Boosting (see below) keeps all units alive.

### k-WTA layer

Linear layer W (n_in × 1024) + bias, followed by hard top-k selection.

- Forward: z = Wx + b; keep the k=20 highest activations; zero the rest.
- k-WTA is applied on CPU-extracted data (candle tensor → Vec<f32> → partial
  sort → SparseActivation). The weight matrix lives in candle for Phase 5
  gradient support.
- **Boosting**: per-unit duty cycle tracked as an exponential moving average
  (α = 1/1000). If duty_cycle[i] < target, unit i's pre-activation is boosted
  by `BOOST_STRENGTH × (target − duty_cycle[i])`. Prevents dead units without
  requiring explicit re-initialisation.

### SparseActivation

```
SparseActivation { region: RegionId, indices: Vec<u16>, values: Vec<f32>, n: usize }
```

Indices are sorted ascending. `n` is always 1024 (for `to_dense()` without
knowing the region). Storage: 20 × (2 + 4) = 120 bytes per activation —
negligible at any scale.

### Binding area

Concatenates four 1024-dim dense-reconstructed activations (4096 total) and
applies a linear projection to a 512-dim episode vector. 512-dim matches the
replay-strategy note's VRAM budget estimate (1 KB/episode at FP16 × 12M
episodes ≈ 12 GB).

Phase 3 uses a plain linear projection. Cross-attention upgrade is Phase 6+.

The binding area produces an `EpisodeVector` only when all four regions have
at least one activation. Until then, `PerceptionProcessor` buffers the latest
sparse activation per region. This means the first episode vector appears after
the first env embedding arrives (~10 s), but visual and speech activations
continue updating at their own rates before then.

### SAE decoder

One thin reverse-linear (1024 → input_dim) per region. In Phase 3 the decoder
is randomly initialised; reconstruction error is logged as a metric. Training
the decoder (and with it, the encoder weights) happens in Phase 5 NREM-like
sleep via experience replay. The error is already wired so Phase 5 can read it
without structural changes.

Reconstruction error = mean-squared error between the encoder input and the
decoder's reconstruction from the sparse code:

```
err = MSE(x, W_dec @ sparse_to_dense(act))
```

### PerceptionProcessor (public, lives in somnus-perception)

Owns all four k-WTA layers, four SAE decoders, and the binding area. Exposes:

- `ingest_speech(values: &[f32]) -> PerceptionResult<PerceptionOutput>`
- `ingest_env(values: &[f32]) -> PerceptionResult<PerceptionOutput>`
- `ingest_visual(values: &[f32]) -> PerceptionResult<PerceptionOutput>`

Each call updates the relevant region(s) (speech also updates symbolic),
computes reconstruction error, and optionally produces an `EpisodeVector` if
all four regions are populated.

### Wiring in `crates/somnus`

`PerceptionProcessor` is called inline in the `main` async loop (not in a
separate `spawn_blocking`): at n=1024, CPU k-WTA takes ~50 µs — well under
async task scheduling granularity. Reconstruction errors and episode vectors
are logged via `tracing::info!`. Episode vectors are written to the NDJSON
sink alongside embeddings.

## Alternatives considered

**SAE instead of k-WTA for the encoder.** More principled; gives a
differentiable coding loss. Deferred because SAE training requires replay data
that doesn't exist until Phase 5. k-WTA in forward-pass mode with random
weights still produces sparse codes and a valid reconstruction-error signal
(just untrained). k-WTA can be upgraded to a full SAE in Phase 4–5.

**ndarray instead of candle.** Simpler build; no autograd overhead. Rejected
because Phase 5 needs candle autograd for the cognitive weight updates. Better
to pay the candle setup cost once now than refactor all the layer types in
Phase 5.

**Skip SAE decoder in Phase 3.** The decoder is cheap to add (one linear per
region) and makes the growth-trigger signal observable as a metric from day
one. Deferring it would require structural changes in Phase 5.

**Binding area cross-attention.** Architecturally correct; deferred to Phase 6+
per roadmap. Concat + linear gives a valid fixed-size episode vector now.

## Consequences

- `candle-core = "0.10"` and `candle-nn = "0.10"` added to workspace deps
  (CPU only; `cuda` feature deferred).
- `crates/somnus-perception` added to workspace.
- `crates/somnus` gains a `somnus-perception` dependency.
- Open question from `architecture.md` "Sparse-coding scheme" is resolved:
  **k-WTA with boosting, n=1024, k=20, target duty cycle 1.95%**.
- Open question "Initial regional set" is resolved: **all four regions from
  Phase 3**, with symbolic fed from audio-speech until text bypass arrives.
- Binding area output dim (512) is now a committed interface point between
  Phase 3 and Phase 4 (episodic buffer row size).
