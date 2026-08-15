# ADR 0030 — R3 Phase 1: Introspection via recent text history

**Status:** Accepted
**Date:** 2026-06-19
**Builds on:** ADR 0028 (R1 memory recall), ADR 0029 (R2 multi-turn)

## Context

The system ingests audio/text and builds episodic memory but had no path to describe recent perceptions in language — "what did you just hear?" couldn't be answered since generation had no access to a textual log of recent perceptions. R3 has two phases: Phase 1 (this ADR) uses the hot-tier text targets already on `EpisodeTrace` as the introspection source; Phase 2 (later, harder) would decode episode vectors through output heads to describe consolidated semantic memory — deferred until reconstruction quality is sufficient.

## What `text_target` contains

Every hot-buffer `EpisodeTrace` can carry `text_target: Option<Vec<u32>>`, set by: (1) auto-grounding (ADR 0014/0019) — every language ingest (ASR or stdin) tokenises and attaches raw text to the current trace; (2) query text — a stdin query is tokenised and attached before generation. The hot buffer therefore implicitly logs everything recently read/heard, with no extra infrastructure.

## Decision

**Detection:** a query is introspective if its lowercase form substring-matches any of: "what did you hear", "what have you heard", "what did you see", "what have you seen", "what have you been", "what have we been", "what do you remember", "what do you know", "what have you learned", "what was that", "recent", "recap". Simple substring match — no ML classifier needed at this scale.

**Context injection:** for an introspective query, retrieve the 8 most recent `text_target` entries (oldest-first), decode to UTF-8, prepend before calling `generate()`:
```
[Recent perceived text:
1. <oldest recent text>
...
N. <most recent text>]
<original query>
```
Stays entirely within the existing `generate()` interface — no new parameters or LM context-layout changes.

**Scope:** wake-local (hot buffer clears at sleep entry, same as R2 conversation history — introspection bounded to the current wake session). No explicit token budget — the 8 entries inject as text tokens via the existing `query_embeds` path; in practice recent-text context is short (1-3 sentences/entry) and fits well within Qwen's 2k context. `effective_query` (with prefix) goes to `generate()`; `query` (original) goes into the JSON `"query"` field so callers see what the user typed.

## Alternatives considered

- **Always inject text history regardless of query type.** Rejected — extra tokens on every query add latency with no benefit for non-introspective queries and could confuse the LM with irrelevant history.
- **Separate text-log ring buffer** independent of `EpisodeTrace`. Rejected — `text_target` on the hot buffer already *is* the log; duplicating creates divergence risk with no gain.
- **Phase 2 first** (semantic-memory narration via output-head decoding). Rejected — needs reconstruction quality not yet available; Phase 1 works now for free.

## What would invalidate this

- Output-head reconstruction quality improving enough for interpretable episode-vector decoding → revisit Phase 2.
- Introspective queries hitting Qwen's token limit → add an explicit history-prefix budget (like R2's `CONV_HISTORY_BUDGET`).
- The keyword list producing too many false positives → replace with a tiny classifier or a configurable `--introspection-keywords` flag.
