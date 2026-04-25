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
