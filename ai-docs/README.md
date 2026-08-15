# ai-docs

> **Contents:** Index of all project documentation. Start here to find any doc.  
> **When to read:** First stop in any new conversation; use as a map to deeper files.

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
- [`gaps.md`](gaps.md) — open, undecided backlog items. [`gaps-resolved.md`](gaps-resolved.md) — the same numbered items once decided (accepted/deployed or rejected/not-pursued), moved verbatim so `gaps.md` stays open-only.
  - 0024: Neuromodulatory self-regulation — accepted; replaces most fixed CLI thresholds
    (salience gate, settle-rate, wakefulness pressure, prior-beta, growth controller) with
    derived EWMA signals. Large, multi-phase, amended 2026-06-14.
  - 0025: SAE L1 coding loss — closed; λ=0.1 destabilised cold-start MSE by 27%; item D answered.
  - 0036: Adopt Qwen3-0.6B, drop Qwen2.5 — accepted 2026-07-05; supersedes 0035.

## Conventions

- New terms → define in `glossary.md` first, then link.
- Non-obvious decisions → ADR in `decisions/`.
- Research findings → topic file in `research/`, with citations.
- Foundation docs describe what *is* or what was just decided. Speculation goes in `research/`, clearly marked.
- One topic, one file. Avoid scattering related material across many docs.
