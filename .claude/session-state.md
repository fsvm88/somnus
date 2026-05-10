# Session state — somnus

> **Last updated:** 2026-05-10
> **Last commit on main:** (pending commit) — *Phase 2 smoke test: output improvements + NDJSON embedding sink*
> **Branch:** `main` (tracking `origin/main` on Codeberg, SHA-256 repo)

Canonical record of where the conversation arc on somnus stands. Read end-to-end when resuming a session.

## Where we are

The **vision discussion is complete and folded into docs.** All six defining commitments are stable:

1. **Continuous learning** — no fixed train/inference split; persistence across sessions; owns catastrophic forgetting.
2. **Sleep and wake** — two-stage cycle (NREM-like for replay/homeostasis/compression; REM-like for generative replay/self-probe/growth); behavioral triggers (exhaustion, wake-perf) with schedule fallback; adaptive duration with cap; salience-gated interruptibility.
3. **Multimodal sensory input** — audio + video as primary; text bypass as a debug/research channel.
4. **Brain-inspired architecture** — regional specialization (audio-speech, audio-env, visual, symbolic) + integrative/binding area; bidirectional region↔integrative paths for cross-modal top-down priors.
5. **Bottom-up growth** — starts as a "baby brain"; expands via cascade-correlation-style unit addition when behavioral saturation (gist-level recall failure) triggers during REM-like sleep.
6. **Grounded concepts** — every concept anchored to perceptual exemplars; continuous embeddings now, with discretization (VQ-style) deferred to Phase 3+.

Operating constraints: single consumer machine, ~14 GB GPU VRAM (16 GB minus desktop overhead) + ~56 GB system RAM (of 64 GB) + disk for cold long-term memory.

The high-level **architecture** in `ai-docs/architecture.md` is stable end-to-end:

- Frozen pretrained encoders (visual + dual-purpose audio + text-bypass) emit **multi-source** continuous representations.
- Regional concept areas (sparsely coded continuous embeddings) per modality and concept domain.
- Integrative / binding area composes regional outputs into unified episodes; **bidirectional** with the regions for cross-modal top-down priors (McGurk-style disambiguation).
- Three-tier memory hierarchy:
  - **Hot** (VRAM) — episodic memory, recent unified episodes, source of replay.
  - **Warm** (system RAM) — semantic memory, consolidated, sparsely coded, growable, region-sharded.
  - **Cold** (disk) — long-term archive, compressed, sharded by region × concept.
- **Recollection layer** — cross-tier index pulling cold memories back into warm/hot when cued.
- **Sleep cycle** runs replay + self-probe + growth + compression across the two stages.

## What's been done in the repo

- Project skeleton: `ai-docs/` (vision, principles, glossary, architecture, tech-stack, roadmap), `ai-docs/research/` and `ai-docs/decisions/` with conventions, `CLAUDE.md` for behavioral instructions, `.gitignore`.
- Git initialized as SHA-256 repo on `main`, pushed to Codeberg.
- `.claude/mind.mv2` (memvid project memory) is tracked; lock files are not.
- Vision and architecture sharpened across multiple commits through a structured Q1/Q2/Q3 + sub-question conversation.
- Full engineering-tradeoffs research pass (Option B — engineering estimates). Nine new notes + two supplement sections. All notes indexed in `ai-docs/research/README.md`.
  - `encoders.md` — Whisper/CLAP/HuBERT/DINOv2/SigLIP; VRAM, latency, pairing recommendations.
  - `ml-runtimes.md` — ort/candle/burn/tch-rs; split runtime (ort for frozen inference, candle for cognitive training).
  - `vector-retrieval.md` — flat/HNSW/IVF-PQ/Hopfield; memory formula; budget by scale.
  - `sparse-coding.md` — k-WTA/top-k/SAE; dead-unit problem; interaction with growth.
  - `memory-compression.md` — FP16/INT8/INT4/PQ; warm→cold pipeline; time estimates.
  - `growth-mechanisms.md` — cascade-correlation; ~10–40 min GPU per unit added; dominates sleep budget.
  - `replay-strategies.md` — uniform/PER/reservoir/generative; episodic buffer sizing (12M episodes in VRAM).
  - `multimodal-binding.md` — concatenation→gated fusion→cross-attention upgrade path.
  - `predictive-coding.md` — Friston FEP; partial adoption (SAE reconstruction error = prediction error).
  - `continuous-learning.md` — supplemented with hardware budget section.
  - `sleep-and-consolidation.md` — supplemented with sleep time/memory budget estimates.
- Phase 1 and Phase 2 architectural design. Four ADRs:
  - `ai-docs/decisions/0001-rust-workspace-layout.md` — crate structure: `crates/somnus-sensory` (lib) + `crates/somnus` (binary) for Phase 1; reserved names for Phase 2–5 crates; no premature traits.
  - `ai-docs/decisions/0002-phase1-sensory-types.md` — **fully spec'd and audit-complete.**
  - `ai-docs/decisions/0003-phase2-encoder-selection.md` — **accepted.** Whisper-tiny encoder-only (384-dim speech), CLAP-HTSAT-unfused (512-dim env), DINOv2-S/14 (384-dim visual). HF revisions pinned. ONNX artifacts in `.gitignore`'d `models/`, exported via `scripts/export_models.py`.
  - `ai-docs/decisions/0004-phase2-encoding-types.md` — **accepted (corrected).** `ort` 2.0.0-rc.12 + `load-dynamic` against system ORT 1.24.4. Exact mel-spectrogram preprocessing formulas verified by parity tests. CLAP correction: `rand_trunc` mode, Slaney scale, fmin=50 Hz (not HTK/fusion as originally stated). ADR updated with implementation notes (LOGSTEP, f64 pipeline, float-space resize).
- **Phase 2 encoding crate implemented and fully tested.** `crates/somnus-encoding`: `WhisperMel`, `ClapMel`, `frame_to_tensor` (DINOv2 image), `AudioEncoder`, `VideoEncoder`. 7 parity fixture tests pass within 1e-4 absolute vs. HuggingFace Python reference. `scripts/gen_mel_fixtures.py` generates fixtures. ADR 0004 updated with corrected CLAP parameters and implementation notes.
- **Phase 2 wiring complete.** `crates/somnus/src/main.rs` wired end-to-end: ORT initialised from system library at startup; `--models-dir` CLI arg (default `models/`) with up-front file existence checks; `AudioProcessor` (lazy resampler init via `SincResampler` from `somnus-encoding::preprocess::resample`, 480k-sample accumulation for both 16 kHz and 48 kHz, `encode_speech` + `encode_env`); `VideoEncoder` task per frame; embedding shape + latency logged via `tracing::info!`. All 31 workspace tests pass.
- **Phase 1 sensory pipeline implemented and fully tested.** Crate layout: `crates/somnus-sensory` (library) + `crates/somnus` (binary). All modules pass `cargo fmt`, `cargo clippy -D warnings`, `cargo test`. 22 tests total (13 unit, 9 integration).
  - `frame.rs` — `Timestamp`, `AudioFrame` (frame_count, duration_secs), `PixelFormat`, `VideoFrame` (bytes_per_pixel).
  - `error.rs` — `SensorError` (11 variants via thiserror), `SensorResult`.
  - `metrics.rs` — `SensorMetrics` (AtomicU64), `MetricsSnapshot` (drop_rate).
  - `audio.rs` — `AudioCapture`: cpal live mic capture, 10-format `make_cb!` dispatch, drop-oldest backpressure.
  - `video.rs` — `VideoCapture`: nokhwa pull-model webcam; `unsafe impl Send` (single dedicated thread, documented safety).
  - `media_file.rs` — `MediaFile`/`MediaFileStreams`: single decode thread, ffmpeg-next 8, nested-scope stream borrows, channel-layout fallback, `scaling::Context` built inside thread (it is `!Send`), flush sequences, `send_drop_oldest` helper.
  - `main.rs` — clap CLI (`--file`, `--mic`, `--webcam`), tokio async runtime, `spawn_blocking` bridges, `select!` loop with 10s metrics tick.
  - `tests/media_file_it.rs` — 9 integration tests: error path, video geometry, frame count, pixel invariants, audio metadata, sample finiteness, total duration, and metrics accounting for both streams. Each test synthesizes its own MP4 via `ffmpeg` CLI.
  - **ADR deviations (not in ADR 0002):** `unsafe impl Send for VideoCapture` (nokhwa Camera is !Send); `scaling::Context` moved into decode thread (ffmpeg-next scaler is !Send, unlike the resampler).

## Where to pick up

**Phase 3 design: memory / concept areas.**

Phase 2 is fully implemented and smoke-tested (2026-05-10). The `somnus` binary:
- Accepts `--file`, `--mic`, `--webcam`, `--models-dir`, `--embeddings-out`
- Runs Whisper-tiny (384-dim, 30 s window), CLAP (512-dim, 10 s window), DINOv2-S/14 (384-dim, per frame)
- Throttles video logs to 1 summary/s (`RUST_LOG=debug` for per-frame detail)
- Writes NDJSON to `--embeddings-out` (flushed every 10 s on metrics tick and on shutdown)

Smoke test on a synthesized 60 s MP4 produced: 104 video embeddings, 3 env embeddings, 1 speech embedding; `embeddings.ndjson` is 108 lines.

**Next: Phase 3 — memory / concept areas.**

Design work needed:
- ADR 0005: episodic buffer (hot VRAM) — structure, capacity, eviction policy
- ADR 0006 (or merged): sparse coding scheme for concept areas (k-WTA vs top-k vs learned penalty)
- Crate layout: `crates/somnus-memory` once boundaries are clear
- Hook embeddings from Phase 2 into the episodic buffer

## Open architectural questions

The full list lives at the bottom of `ai-docs/architecture.md`. Salient unresolved items:

- Initial regional set (audio-speech + symbolic for Phase 1, or all four from the start?)
- Encoder specifics (single multi-task audio encoder vs. two specialized; specific model choices for visual)
- Growth mechanism details (granularity, sparse-code integration)
- Recollection-layer implementation (vector ANN, Hopfield, hybrid)
- Compression strategy (quantization, pruning, distillation, layered)
- Sparse-coding scheme (k-WTA, top-k, learned penalty)
- Sleep-cycle time budget (NREM/REM split)
- Action loop (whether the system has one in early phases)
- Scheduling (async / dedicated threads / event loop)
- Tensor representation (single shared type vs. per-subsystem)

## How to resume

When the user says "resume SOMNUS prompt" (or close variation):

1. Read `CLAUDE.md` (already auto-loaded) for behavior.
2. Read this file end-to-end.
3. Skim `ai-docs/vision.md` and `ai-docs/architecture.md` to refresh the integrated picture.
4. Skim research notes in `ai-docs/research/` if the user's prompt references them.
5. Check `git log --oneline -10` for recent commit messages — they summarize each substantive change.
6. Greet briefly with "where we left off", then either ask for direction or proceed with the queued next step.

## When to update this file

At a logical break point (after a substantive set of edits has been committed), refresh:

- "Last updated" date
- "Last commit on main" hash and subject
- "What's been done in the repo"
- "Where to pick up"

Keep the rest stable unless the project picture itself shifts.
