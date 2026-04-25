# ai-docs

Project knowledge for **somnus**. Conversation context is ephemeral; this folder is the project's long-term memory.

For *behavioral* expectations on the assistant, see `../CLAUDE.md`. This folder is for project content.

## Foundation

- [`vision.md`](vision.md) — what we're building and why.
- [`principles.md`](principles.md) — how we approach the work.
- [`glossary.md`](glossary.md) — terminology, single source of truth for terms.
- [`architecture.md`](architecture.md) — high-level system structure (a sketch; will evolve).
- [`tech-stack.md`](tech-stack.md) — Rust crates, ML/audio/vision ecosystem notes, language choices.
- [`roadmap.md`](roadmap.md) — coarse phasing.

## Living archives

- [`research/`](research/) — research notes, by topic. Each file cites primary sources.
- [`decisions/`](decisions/) — ADRs, numbered sequentially.

## Conventions

- New terms → define in `glossary.md` first, then link.
- Non-obvious decisions → ADR in `decisions/`.
- Research findings → topic file in `research/`, with citations.
- Foundation docs describe what *is* or what was just decided. Speculation goes in `research/`, clearly marked.
- One topic, one file. Avoid scattering related material across many docs.
