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

- A non-obvious choice that future-us will ask "why?" about.
- A choice with consequences that touch more than one subsystem.
- A choice that overrides a default from `principles.md` or `tech-stack.md`.
- The promotion of an experiment into the main system.

## When *not* to write an ADR

- Routine implementation details.
- Choices fully determined by `principles.md` or `tech-stack.md`.
- Throwaway code in `experiments/`.

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
