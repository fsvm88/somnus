# ADR 0042 — BatchTopK SAE

**Date:** 2026-06-27
**Status:** Accepted — deployed, default on

## Context

The SAE k-WTA layer (`KWtaLayer`) enforces sparsity by keeping the top `K_ACTIVE` activations per sample and zeroing the rest, trained on pure L2 reconstruction loss. ADR 0025 tried adding an L1 coding-loss term on top of that per-sample top-k mask (λ=0.1) to close remaining MSE headroom once prior cosines saturated (item B, LM-contextualized language embeddings) — it made reconstruction 27% worse and was closed. That result said nothing about the k-WTA selection rule itself, only about penalizing magnitudes on top of it.

Bussmann et al. (arXiv:2412.06410, Dec 2024) propose BatchTopK: instead of a hard per-sample top-k, compute one quantile threshold τ over the whole batch such that the *average* active-feature count across the batch equals k — individual samples may end up with k±δ active units. This is a different mechanism from L1 (no penalty term, no new loss, still pure L2 reconstruction), so it isn't a retry of ADR 0025's rejected approach; it changes *which* activations get selected, not what's penalized once they are.

## Decision

Adopt BatchTopK as the default k-WTA strategy for NREM training. `KWtaLayer::forward_tracked_batch` gained a `batch_topk: bool` parameter (`crates/somnus-perception/src/region.rs`): when true, duty-cycle-boosted activation values are snapshotted, flattened across the whole batch, and a single τ is found via `select_nth_unstable` (selecting the `b·k`-th largest value) and applied uniformly; when false, the existing per-sample top-k path runs unchanged. Inference (non-batch, single-sample forward) is untouched either way — BatchTopK only applies to the batched NREM training path.

Landed in two commits: `81b64c7` (feat: BatchTopK SAE alternative k-WTA, opt-in, ADR candidate) added the mechanism behind `--batch-topk` defaulting to `false`; `0eca914` (feat: make BatchTopK the default k-WTA strategy) flipped both the `PerceptionProcessor` field default and the CLI flag's `default_value` to `true` after the A/B passed. Verified directly against current source: `crates/somnus-perception/src/lib.rs:376` (`batch_topk: true` in `PerceptionProcessor::new`) and `crates/somnus/src/lib.rs:329-330` (`--batch-topk` flag, `default_value = "true"`). `--batch-topk=false` remains available for comparison/debugging.

**Discrepancy note:** `engineering-log.md` §1 states "`--batch-topk` flag, default false," which is stale — it describes the state after the first commit (`81b64c7`) and predates the default-flip commit (`0eca914`). Current code (checked 2026-08-16) confirms the default is `true`, matching the log's own "default in all NREM training" claim in the same section. This ADR records the current, verified state; §1's "default false" line should be read as historical (pre-flip), not current.

## Alternatives considered

- **Per-sample top-k with L1 coding loss (ADR 0025).** Rejected on its own terms (−27% MSE) and structurally different — doesn't bear on whether the selection rule itself is a good one.
- **Keep per-sample top-k as default, ship BatchTopK opt-in only.** Rejected once the A/B passed — CLAUDE.md's stance (small experiments graduate via explicit decision once earned) applies once results are in, no reason to leave a validated improvement gated behind a flag nobody remembers to pass (cf. ADR 0037's same argument for auto-enabling ORT).
- **Per-sample top-k unchanged (no experiment).** Rejected before the A/B — the paper's improvement claim was cheap to test given `forward_tracked_batch` already batches training, and item 1 was carried in gaps.md as an open backlog item.

## A/B Validation

Commit `0eca914`'s message records the result: **PASS, +3.6% MSE** (within the pre-stated tolerance), per-region prior cosines stable. `engineering-log.md` §1 cites commit `25d7496` as the A/B commit; that hash does not resolve in the current repo history (likely rewritten/squashed since) — the surviving record of the same result is `0eca914`'s commit message, which is corroborating rather than an additional independent measurement.

## Consequences

- NREM training's duty-cycle update semantics become batch-order-independent under BatchTopK (τ is computed once over the flattened batch rather than sample-by-sample), a structural difference from the per-sample path that stays available via the flag.
- Individual samples in a batch may have more or fewer than exactly `K_ACTIVE` units active (k±δ around the batch mean) — a deliberate relaxation, not a bug, per the source paper.
- Inference-time sparsity guarantee (`K_ACTIVE` exactly, per sample) is unaffected; BatchTopK only touches the NREM training forward path.

## Metrics

No new metric was needed. The A/B was evaluated on the existing `mean_output_mse` (Core tier, NREM SAE reconstruction loss) — the same metric ADR 0025's L1 experiment used — plus per-region prior cosines, both already wired per ADR 0039's instrumentation. This change alters which values feed the existing k-WTA/duty-cycle path, not what's measured; no new operation, failure mode, or tunable-value counter was introduced.

## What would invalidate this

- A larger-scale or longer-horizon re-run showing the +3.6% MSE regression compounds rather than staying bounded, given the A/B's own small-dataset variance caveat (commit message: "high variance in small dataset experiments should not gate adoption... on larger scales").
- Evidence that batch-order-dependent duty-cycle semantics interact badly with a specific replay ordering or sampling strategy not exercised by the original A/B.
- The source paper's claim not replicating at this system's region/unit scale as regions grow well past the sizes tested here.

**Source:** `engineering-log.md` §1.
