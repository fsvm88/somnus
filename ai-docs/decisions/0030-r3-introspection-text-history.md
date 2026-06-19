# ADR 0030 — R3 Phase 1: Introspection via recent text history

**Status:** Accepted  
**Date:** 2026-06-19  
**Builds on:** ADR 0028 (R1 memory recall), ADR 0029 (R2 multi-turn)

---

## Context

The system ingests audio and text and builds episodic memory, but had no path to
describe what it recently perceived in language. Queries like "what did you just
hear?" or "what have we been talking about?" could not be answered because the
generation path had no access to a textual log of recent perceptions.

R3 addresses this in two phases. Phase 1 (this ADR) uses the hot-tier text targets
that already exist on `EpisodeTrace` as the introspection source. Phase 2 (later,
harder) would decode episode vectors through output heads to describe consolidated
semantic memory — deferred until reconstruction quality is sufficient.

---

## What `text_target` contains

Every `EpisodeTrace` in the hot buffer (`EpisodicBuffer`) can carry
`text_target: Option<Vec<u32>>` — token IDs set by two paths:

1. **Auto-grounding (ADR 0014/0019):** every language ingest (ASR transcript or
   stdin text line) tokenises the raw text and attaches it to the current trace.
2. **Query text:** when a user submits a stdin query, the query is tokenised and
   attached to the current trace before generation.

This means the hot buffer implicitly holds a log of everything the system recently
read or heard, with no additional infrastructure required.

---

## Decision

### Detection

A query is classified as **introspective** if its lowercase form contains any of:

```
"what did you hear", "what have you heard",
"what did you see",  "what have you seen",
"what have you been", "what have we been",
"what do you remember", "what do you know",
"what have you learned", "what was that",
"recent", "recap"
```

Simple substring match. No ML classifier needed at this scale.

### Context injection

When the query is introspective, retrieve the 8 most recent `text_target` entries
from the hot buffer (oldest-first), decode them back to UTF-8, and prepend them to
the query string before calling `generate()`:

```
[Recent perceived text:
1. <oldest recent text>
2. ...
N. <most recent text>
]
<original query>
```

This keeps the mechanism entirely within the existing `generate()` interface —
no new parameters, no structural changes to the LM context layout.

### Scope

- **Wake-local:** the hot buffer is cleared at sleep entry (same as conversation
  history in R2), so introspection is bounded to the current wake session.
- **Token budget:** not explicitly budgeted — the 8 recent entries are injected as
  text tokens via the existing `query_embeds` path, which grows the sequence. In
  practice the recent-text context is short (1–3 sentences per entry) and fits
  within Qwen's 2k token context well below the LM limit.
- **Original query preserved in JSON output:** `effective_query` (with the prefix)
  goes into `generate()`; `query` (original) goes into the JSON `"query"` field so
  callers can see what the user typed.

---

## Alternatives considered

**Always inject text history regardless of query type.** Rejected: extra tokens on
every query adds latency with no benefit for non-introspective queries, and could
confuse the LM when the history is irrelevant to the question asked.

**Separate text-log ring buffer** (independent of `EpisodeTrace`). Rejected:
`text_target` on the hot buffer already IS the log; duplicating it would create
divergence risk with no gain.

**Phase 2 first** (semantic memory narration via output head decoding). Rejected:
requires reconstruction quality we don't yet have. Phase 1 works now and costs
nothing.

---

## What would invalidate this

- Output head reconstruction quality improves enough that episode vectors can be
  decoded to interpretable text → revisit Phase 2.
- Introspective queries start hitting the Qwen token limit → add an explicit token
  budget for the history prefix (similar to `CONV_HISTORY_BUDGET` in R2).
- Keyword list produces too many false positives → replace with a tiny classifier
  or a configurable `--introspection-keywords` flag.
