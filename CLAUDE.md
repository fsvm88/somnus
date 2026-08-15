# somnus — instructions

This is **somnus**: a long-running research project toward an AI system that continuously learns, sleeps, ingests audio and video, and is architecturally inspired by the human brain. Rust is the implementation language by choice.

This file describes **how to behave** on this project. Project content — vision, architecture, terminology, research, decisions — lives in `ai-docs/`. Read `ai-docs/README.md` first to navigate.

## Mindset

- **Research, not product.** Optimize for learning velocity, reversibility, and clarity. Robustness, polish, generality, and backward compatibility are non-goals at this stage. A small ugly working demo beats a clean hypothetical design.
- **Brain-inspired, not brain-faithful.** Biology is a *source of ideas* — we borrow algorithmic principles (predictive coding, replay, sparse coding, neuromodulation, oscillatory gating, etc.) when they earn their keep. We are *not* simulating neurons. Every biological appeal in a design must also have an engineering justification; if the engineering doesn't stand on its own, the biology is decoration.
- **Rust is a constraint we accepted, not a religion.** When the ecosystem has a gap, prefer in this order: (1) wrap an existing crate, (2) FFI to a mature C/CUDA/Python lib, (3) build a minimal in-tree shim, (4) write from scratch only when the surface is small and well-understood. Don't hand-roll a tensor library or an audio codec.
- **Small experiments first.** New ideas land as standalone binaries or in `experiments/`. They graduate into the main system only after they've earned it, via an explicit ADR.

## Documentation discipline

Everything durable belongs in `ai-docs/`. Treat docs as the project's long-term memory; conversation context is not.

- Non-obvious decisions get a short ADR in `ai-docs/decisions/NNNN-title.md` (sequentially numbered). Capture *why*, alternatives considered, and what would invalidate the decision.
- Research that informs design lands in `ai-docs/research/<topic>.md`, with citations (paper title, authors, year, link, one-line takeaway).
- Update `ai-docs/architecture.md` when system structure changes. It is the canonical mental model.
- Update `ai-docs/architecture.svg` whenever the architecture or its key parameters change (encoders, region count, EPISODE_DIM, GRU_HIDDEN, D_PROJ, N_UNITS, K_ACTIVE, memory tier sizing, new subsystems, removed/closed items). The SVG is the visual companion to `architecture.md`; both must stay in sync. The SVG lives in git alongside the docs.
- Define new terms in `ai-docs/glossary.md` once and link from elsewhere.
- Keep `ai-docs/README.md` as the working index of what's where.
- **Don't write speculative architecture into the foundation docs.** Speculation goes in `research/`, clearly marked. Foundation docs describe what *is* or what was just decided.
- **Don't delete a doc that looks stale** without checking — staleness can mean we lost track of a research thread, not that the thread is dead.
- Honesty about state: if a subsystem is a stub, say so. If a result is from a toy dataset, say so. Aspirational docs are a research liability.

## Code conventions

- `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test` before declaring work done.
- New deps: name 2–3 alternatives, pick one, justify in an ADR or `tech-stack.md` if non-obvious. Pin versions.
- Workspace layout: a top-level Cargo workspace; split crates along architectural boundaries (`sensory`, `memory`, `cognition`, etc.) only once those boundaries are real. Don't pre-create empty crates.
- Tests scale with stakes: integration tests at crate boundaries; `proptest` for stateful logic (memory, learning rules); `criterion` benches only when there is a target to hit.
- Public APIs of shared crates get rustdoc. Internal modules: only when *why* is non-obvious.
- Seeds fixed in experiments. Configs and dataset versions recorded. We don't yet need MLflow-grade infra, but we should always be able to answer "how did you get this number?"

## Metrics discipline (standing policy as of 2026-07-15)

Every ADR, architectural improvement/fix, or algorithm-switch proposal must account for metrics before it's considered complete — no operation touching data ships unmeasured. Use `crates/somnus-metrics`'s existing two-tier design (`ai-docs/decisions/0039-unified-metrics-instrumentation.md`) as the demarcation line, don't invent a parallel system:

- **Core tier** (`timed_core!`/`counted_core!`, always-on) = normal performance/health monitoring. What you'd want firing on every production campaign run to know the system is behaving: wall-clock of every operation-granularity phase, resource sampling (RSS, disk), degradation/fallback counters, failure counters. This is the tier a new ADR's implementation should land in by default.
- **Fine tier** (`timed!`/`counted!`, behind `fine-metrics`, default off) = investigation-grade evaluation. Granular sub-breakdowns, A/B-comparison-specific counters, anything only worth the (small but nonzero) always-on cost when actively debugging a regression or validating a specific improvement claim.

When proposing or implementing a change: name which existing metrics already cover the affected code path, name what new Core/Fine metrics the change needs (new operations, new failure modes, new tunable-value counters), and don't declare the change done until those are wired and verified live (a short pilot run showing the metric fire with a plausible value, not just a clean build). If a change doesn't touch any measurable data-handling operation, say so explicitly rather than silently skipping this step.

**A/B runs always run with full debugging metrics, and deviations always get recorded** (standing instruction, 2026-08-15): every A/B pilot builds and runs both arms with `fine-metrics` enabled (the Fine tier exists precisely for A/B comparison), and the write-up records observed deviations — arm-vs-arm divergence on more than the headline metric, plus any deviation from expected/baseline behavior (unexpected counter firings, metric regressions, health anomalies), not just whether the hypothesis won.

## Working with the user

- Exploratory prompts ("how should we…", "what about X") get a recommendation + the main trade-off in a few sentences, not a finished plan. Wait for the go-ahead before implementing.
- Before adding a new external dependency or starting a new top-level subsystem, name what's coming and check.
- When biology and engineering pragmatism conflict, default to engineering pragmatism and note the divergence in the relevant doc.
- Updates and end-of-turn summaries stay short. Diffs and docs speak for themselves.
- This is a one-developer research repo. There is no team to coordinate with, no external users, no SLAs.

## Session resume

If the user says **"resume SOMNUS prompt"** (or a close variation: "resume", "resume SOMNUS", "where were we"), read `.claude/session-state.md` end-to-end and continue from where the previous session left off. That file is the canonical record of the conversation arc and what's queued next. After reading, state briefly where things stand and either ask for direction or proceed with the queued next step.

When wrapping a session at a logical break point (a substantive set of edits has been committed), update `.claude/session-state.md` so the next resume is accurate.

## File-system landmarks

- `CLAUDE.md` — this file.
- `ai-docs/` — project knowledge. Start at `ai-docs/README.md`.
- `.claude/session-state.md` — last-known session state; read on "resume SOMNUS prompt".
- `experiments/` — throwaway prototypes; expect frequent churn. No CI gates, no API stability.
- `crates/` — main system code, once it exists.

## Deferred architectural items (as of 2026-06-17)

All remaining ADR backlog items are **deferred indefinitely** — do not propose or implement them unless the user explicitly raises one or a concrete trigger condition is met. The items and their triggers are tracked in `.claude/session-state.md`. Do not suggest revisiting them unprompted.

Items deferred: A (cold archive promotion index), C (sparse activation storage), E (separate output-head optimizer), F (post-k-WTA prior injection), I (knowledge distillation compression), J (Whisper-base), K (SigLIP-B), L (LoRA fine-tuning — done for Qwen2.5 pre-ADR-0036, orphaned by the Qwen3 swap; `scripts/train_lora_qwen3.py` exists but was never run, production `qwen3_lm.onnx` is not LoRA-tuned — see gaps.md §37), N (visual/audio grounding).

Item M (MLP projection) is NOT deferred — ADR 0032 implemented it and it has been the default production path since 2026-06-21 (`--linear-proj` defaults false; confirmed dimension-generic and Qwen3-compatible via gaps.md §37's audit). Do not relist it here.
