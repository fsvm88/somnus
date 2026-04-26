# Session state — somnus

> **Last updated:** 2026-04-26
> **Last commit on main:** `1c388d1` — *Add engineering-tradeoffs research pass across all planned components*
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

## Where to pick up

1. **Phase 1 planning.** The research foundation is solid. Scope the first implementation: Rust workspace, crate split, encoder selection ADR, initial audio pipeline.
2. **Option A research pass.** Refine Option B estimates with cited benchmark numbers. Priority: encoder latency on real hardware, HNSW query latency with `usearch`.
3. **ADRs.** Multiple open questions in each research note are ADR-ready. First 2–3 ADRs could lock in encoder choice and runtime split before coding.

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
