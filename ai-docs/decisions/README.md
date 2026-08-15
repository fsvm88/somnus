# decisions/

Architecture Decision Records (ADRs). One file per decision.

## Filename

`NNNN-short-title.md`, where `NNNN` is a zero-padded sequential number starting at `0001`.

## Template

```markdown
# NNNN. Short title

- **Status:** proposed | accepted | superseded by NNNN | deprecated
- **Date:** YYYY-MM-DD

## Context

What forced this decision? Constraints, observations, or open questions that led here.

## Decision

What we decided. State it directly, in one or two sentences.

## Alternatives considered

Each alternative gets a sentence or two: what it was, why it was rejected.

## Consequences

What changes as a result? What new constraints or follow-ups does this create?

## What would invalidate this

Concrete signals that would force us to revisit. If none can be named, the decision is probably too vague.
```

## When to write an ADR

A non-obvious choice future-us will ask "why?" about; consequences touching more than one subsystem; overriding a `principles.md`/`tech-stack.md` default; promoting an experiment into the main system.

## When *not* to write an ADR

Routine implementation details; choices fully determined by `principles.md`/`tech-stack.md`; throwaway code in `experiments/`.

## Index

| # | Title | Status |
|---|-------|--------|
| [0001](0001-rust-workspace-layout.md) | Rust workspace layout | accepted |
| [0002](0002-phase1-sensory-types.md) | Phase 1 sensory pipeline: concrete types and interfaces | accepted |
| [0003](0003-phase2-encoder-selection.md) | Phase 2 encoder selection: Whisper-tiny + CLAP + DINOv2-S/14 | accepted |
| [0004](0004-phase2-encoding-types.md) | Phase 2 encoding types and ORT integration | accepted |
| [0005](0005-phase3-perception.md) | Phase 3 perception: k-WTA sparse coding + binding | accepted |
| [0006](0006-phase4-memory-hierarchy.md) | Phase 4 memory hierarchy: EpisodicBuffer + SemanticStore + ColdArchive | accepted |
| [0007](0007-phase5-sleep-cycle.md) | Phase 5 sleep cycle: NREM replay + REM growth + snapshot | accepted |
| [0008](0008-cross-attention-binding.md) | Cross-attention binding area | accepted |
| [0009](0009-output-reconstruction-channel.md) | Output / reconstruction channel | accepted |
| [0010](0010-bidirectional-top-down-priors.md) | Bidirectional top-down priors | accepted |
| [0011](0011-predictive-coding-settling-loop.md) | Predictive-coding settling loop | accepted |
| [0012](0012-action-loop-decoder.md) | Action loop: region-token decoder + gain modulation | accepted |
| [0013](0013-phase6-language-region.md) | Phase 6 language region: Qwen2.5 embeddings + ASR | accepted |
| [0014](0014-phase7-auto-grounding.md) | Phase 7 auto-grounding: ASR text targets for projection | accepted |
| [0015](0015-phase8-generation-evaluation.md) | Phase 8 generation evaluation: cosine alignment loss | accepted |
| [0016](0016-phase12a-eval-harness.md) | Phase 12a metrics eval harness | accepted |
| [0017](0017-phase12b-hnsw-persistence.md) | Phase 12b HNSW warm-index persistence | accepted |
| [0018](0018-phase12c-regional-distillation.md) | Phase 12c regional unit pruning | accepted |
| [0019](0019-gap-fixes-synthetic-replay-salience-gate-nrem-lr.md) | Gap fixes: synthetic replay, salience-gated wake reconstruct, salience → NREM LR | accepted |
| [0020](0020-systematic-forgetting-cold-truncation.md) | Systematic forgetting: bounded cold archive | accepted |
| [0021](0021-cross-modal-gist-entropy.md) | Cross-modal gist entropy: ingest_source and concept probe | accepted |
| [0022](0022-instrumentation-metrics-expansion.md) | Instrumentation and metrics expansion: salience, blend, modality, NREM coverage | accepted |
| [0023](0023-gpu-unification.md) | GPU unification: ORT/MIGraphX for perception inference | accepted; Phase 0+1 implemented |
| [0024](0024-neuromodulatory-self-regulation.md) | Neuromodulatory self-regulation: derived signals replace fixed thresholds | accepted |
| [0025](0025-sae-l1-coding-loss.md) | SAE L1 coding loss | closed — not adopted |
| [0026](0026-pruning-ab-distribution-shift.md) | A/B: magnitude pruning safety under distribution shift | closed — FAIL, pruning disabled by default |
| [0027](0027-per-region-prior-beta.md) | Per-region prior beta (Item G) | closed — FAIL (gist regression) |
| [0028](0028-memory-recall-in-generation.md) | Memory recall in generation (R1) | accepted |
| [0029](0029-multi-turn-conversation.md) | Multi-turn conversation (R2) | accepted |
| [0030](0030-r3-introspection-text-history.md) | R3 Phase 1: introspection via recent text history | accepted |
| [0031](0031-gpu-generation-fixed-shape-lm.md) | GPU generation via fixed-shape LM (qwen25_lm_fixed.onnx) | accepted |
| [0032](0032-mlp-projection.md) | MLP projection (640→1024→5×896, ADR 0012 item M) | accepted |
| [0033](0033-lora-qwen-finetuning.md) | LoRA fine-tuning of Qwen2.5-0.5B on episode prefixes (ADR 0012 item L) | validated and deployed |
| [0034](0034-kv-cache-generation.md) | KV-cache split prefill/decode generation (generation speedup) | implemented |
| [0035](0035-qwen3-hidden-size-compile-time.md) | LM hidden size (Qwen2.5 vs Qwen3) is a compile-time choice, not runtime | implemented |
| [0036](0036-adopt-qwen3-drop-qwen25.md) | Adopt Qwen3-0.6B as the action-decoder LM; drop Qwen2.5 | implemented |
| [0037](0037-auto-onnx-perception-inference.md) | Auto-enable ORT/MIGraphX perception inference past the CPU/GPU crossover | accepted |
| [0038](0038-sparse-encoder-scoring.md) | Sparse (hierarchical) k-WTA encoder scoring for the next scale order | proposed |
| [0039](0039-unified-metrics-instrumentation.md) | Unified two-tier metrics instrumentation (`somnus-metrics`) | implemented |
| [0040](0040-sparse-gather-binding-projection.md) | Sparse-gather binding projection (ONNX/MIGraphX) | implemented, deployed live |
| [0041](0041-live-episode-binding-training.md) | Live-episode binding training in NREM (close the §33 never-trained-binding gap) | implemented; not adopted — 2 failed pilots (gaps.md §35/§36) |
