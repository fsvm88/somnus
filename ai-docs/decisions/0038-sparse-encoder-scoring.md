# 0038. Sparse (hierarchical) k-WTA encoder scoring for the next scale order

- **Status:** proposed — direction decided, implementation deliberately not started (gated on
  prerequisites below). As of 2026-07-15, prerequisites 1 and 2 both resolved negative at the
  scale currently in flight (see below) — **deferred, not closed**, pending the explicit trigger
  recorded under prerequisite 2.
- **Date:** 2026-07-14

## Context

The 10x growth campaign (gaps.md §16/§20) is growing each region's `KWtaLayer`
(`crates/somnus-perception/src/region.rs`) from `N_UNITS=1024` toward ~9,660 units, honeybee-order
scale (`research/brain-scale-comparison.md`). While investigating what would genuinely help scale
further — mouse-order is another ~1,300x in unit count — rather than just running more GPU cycles,
gaps.md §25 found and fixed a real 11.4× speedup in the *binding* projection by exploiting k-WTA's
output sparsity (gathering only the k active rows instead of a dense `n_units`-length matmul).

That fix doesn't extend to the k-WTA layer itself. `KWtaLayer::forward` has two costs:

1. **Scoring** (`self.linear.forward`, `region.rs:117`): `z = Wx`, a dense
   `[d_input] → [n_units]` matmul, `O(d_input × n_units)`.
2. **Selection** (`select_top_k`, `region.rs:151`): `O(n_units)` partition, cheap by comparison.

Sparsity is a property of the *output* (k of n_units survive selection) — it says nothing about
the *input* projection, since every unit's score must be computed before top-k can know which
ones win. `scaled_k` (`region.rs:20`) keeps `k/n` ≈ 2% constant as regions grow, which controls
storage and every *downstream* consumer's FLOPs (as §25 exploited), but the scoring matmul itself
stays `O(d_input × n_units)` regardless of `k`. A 10x region-size increase is a 10x scoring-cost
increase under the current all-to-all connectivity — linear, not the sub-linear scaling biological
cortical columns achieve via local receptive fields (a given input reaches a small, structured
fan-out of candidate units, not the entire area).

Full survey and reasoning: `research/sparse-coding.md` §"Encoder-side scaling" (written 2026-07-14,
same session).

## Decision

**Direction:** if/when this is implemented, use **hierarchical / coarse-to-fine WTA** — a cheap
coarse scorer over super-units (groups of size `block_size`, each represented by a small summary
projection) selects a handful of promising blocks, then full-resolution scoring runs only within
those blocks. This reduces the dense matmul from `O(d_input × n_units)` to
`O(d_input × n_blocks) + O(d_input × block_size × n_selected_blocks)`.

Chosen over the other two surveyed mechanisms:
- Preserves an *exact* top-k guarantee within the selected blocks (unlike sampling), so it doesn't
  silently break the growth mechanism's assumption that a new unit's correlation with residual
  error is actually measured, not stochastically missed.
- Simpler failure mode than learned routing (MoE-style gating): a bad coarse scorer degrades
  gracefully toward "more blocks selected, less speedup," not toward silently starving whole
  regions of candidate units the way a bad router could.
- This codebase already has ANN-index machinery (HNSW, `crates/somnus-memory` warm/cold stores) that
  may be reusable or at least directly informative for the coarse stage, rather than needing a
  bespoke gating network built from scratch.

**Implementation is explicitly NOT started.** This ADR records the chosen direction so a future
session doesn't re-litigate the three-way comparison, not a commitment to build it now.

## Prerequisites before implementation (2 resolved, 1 still open)

1. **Quantify the actual cost. RESOLVED 2026-07-15 — see `gaps.md` §28.** The exported ONNX
   encoder graph (`scripts/export_perception_graph.py`) has the identical `O(d_input × n_units)`
   dense matmul as the `burn` path — no sparsity or gating applied before scoring. But the live
   wake-path cost, measured from the production campaign's `OrtRunEncoder` Core metric at
   ~9,400-9,493 units/region, is small in absolute terms (~1.28ms/call mean) and is not even the
   dominant ORT cost — the fixed-size `OrtRunBinding` graph costs 2-3× more per call at this scale.
   This is evidence for (not proof of) the "ORT/GPU already absorbs the wasted FLOPs" invalidation
   condition below, at *current* scale specifically — does not resolve prerequisite 2.
2. **Confirm this matters at the scale actually being targeted. RESOLVED 2026-07-15 — see
   `gaps.md` §29.** It doesn't, yet. `gaps.md` §16 states the growth campaign's own goal
   explicitly as honeybee-order (~9,660 units/region), *not* mouse-order — and there is no
   documented roadmap item, planned follow-on campaign, or stated ambition anywhere in `ai-docs/`
   to go past honeybee-order. Combined with prerequisite 1's finding (scoring cost is small and
   non-dominant at this scale), hierarchical WTA is not worth building for the campaign this
   project is actually running. **This is a "doesn't matter yet" finding, not "wrong direction" —
   see the updated "what would invalidate this" section: the correct status is deferred with an
   explicit trigger, the same pattern CLAUDE.md uses for items A-N, not "closed."** The trigger:
   the growth campaign's target scale moves past honeybee-order, or the encoder/binding cost
   balance shifts back toward the encoder mattering.
3. **Numerical validation plan.** Hierarchical routing changes which units are even candidates for
   a given input — needs validation that `per_region_mse` still improves the way full competition
   does in the §16 pilots (2/3), and that the growth mechanism (`grow_region_seeded`) and duty-cycle
   boosting (`BOOST_STRENGTH`/`DUTY_CYCLE_ALPHA`) still behave correctly when competition is
   block-local instead of global.

## Alternatives considered

1. **Local / blocked connectivity via learned routing (MoE-style gating).** Closest biological
   analogue (cortical minicolumns), most flexible, but a bad router reintroduces the dead-unit
   problem this codebase already fought once, and routing quality itself becomes a thing that can
   silently degrade — harder to validate than a fixed coarse-to-fine scorer. Not chosen as the
   primary direction, but not ruled out either if coarse-to-fine's simpler router proves
   insufficiently selective in practice.
2. **Approximate top-k via sampling/sketching.** Cheapest to implement, weakest guarantee — units
   outside a sampled subset can never win on a given input even if they'd have scored highest.
   Rejected as the primary direction because it breaks the exact-top-k assumption the growth
   mechanism's residual-correlation logic depends on; could be revisited as a cheaper fallback if
   hierarchical WTA's coarse stage turns out too expensive itself.
3. **Do nothing; keep O(input_dim × n_units) and rely on ORT/MIGraphX for constant-factor
   speedups.** Valid short-term choice (this is what's happening right now) — ORT already handles
   the current 10x campaign's dense cost adequately. Rejected as the *long-term* answer because
   constant-factor GPU speedups don't change the linear-in-n_units scaling law; they only push out
   when it starts to dominate wall-clock.

## Consequences

- No code changes from this ADR. It exists so the direction (hierarchical/coarse-to-fine, not MoE
  routing or sampling) doesn't need re-deriving next time this is picked up.
- Establishes a dependency: whoever implements this should first produce the scoring-cost
  benchmark (prerequisite 1) and re-check this ADR's "what would invalidate this" section before
  writing code, since the chosen direction was picked from a first-pass survey, not a
  benchmark-informed comparison.
- If adopted, interacts with three existing mechanisms and needs re-validation against each:
  `grow_region_seeded` (growth), duty-cycle boosting (`region.rs`), and the binding area's
  assumption that any of a region's `n_units` can be part of the active set for a given input.

## What would invalidate this

- **Confirmed 2026-07-15 (prerequisites 1 and 2):** the scoring matmul is not currently a
  bottleneck — small in absolute terms (~1.28ms/call), not even the dominant ORT cost at
  honeybee-order scale, which is also the project's actual current target (no documented plan to
  push toward mouse-order). This does **not** mean the direction was wrong or that mouse-order
  ambitions have been abandoned — see the prerequisite 2 write-up above and `gaps.md` §29 for why
  this stays `proposed`/deferred rather than "closed — not needed." Only mark this "closed" if a
  future session establishes affirmatively that honeybee-order is a decided ceiling (not merely
  that no next campaign has been scoped yet) — that's a judgment call for whoever is running the
  project at the time, not something derivable from doc silence alone.
- The live wake-path benchmark shows ORT/MIGraphX already handles the exported encoder graph's
  scoring cost well enough (sub-linear in practice due to GPU parallelism across `n_units`) that a
  software-side hierarchical scheme adds complexity without a measurable win — GPU matmul
  throughput can absorb a lot of "wasted" FLOPs that a CPU implementation could not.
- A simpler mechanism (e.g. static, fixed-at-init block assignment with no learned or dynamic
  routing at all) turns out to capture most of the benefit — hierarchical WTA's "coarse scorer"
  stage might be over-engineering if blocks can be assigned once at growth time and never
  re-routed.
- The research direction moves away from region-scaling entirely (same transitive condition ADR
  0023/0037 already carry).
