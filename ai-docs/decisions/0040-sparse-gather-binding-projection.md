# 0040. Sparse-gather binding projection (ONNX/MIGraphX)

- **Status:** implemented (Python graph + Rust caller + numerical parity + synthetic benchmark);
  live production A/B against the growth campaign's real `OrtRunBinding` metric, and deployment to
  the campaign itself, deliberately not yet done — see "Not yet done" below.
- **Date:** 2026-08-12

## Context

`gaps.md` §25 found and fixed a real 11.4× speedup in the *burn/CPU* binding projection
(`BindingArea::forward`) by exploiting k-WTA output sparsity: gathering only the k active rows of
each region's projection weight instead of a dense `n_units`-length matmul on a mostly-zero vector.
That fix was correctly deployed but turned out **dormant** on the production campaign, because ORT
inference (ADR 0037) had already crossed over and `try_bind`'s ORT branch never reaches the burn
`BindingArea::forward` code path at all — the ONNX-exported binding graph
(`scripts/export_perception_graph.py::build_binding_graph`) still ran the equivalent dense
`sp_i [1, n_units] @ W_region_i [n_units, D_PROJ]` matmul, one per region, unaffected by §25's fix.

`gaps.md` §28/§29 (ADR 0038's prerequisites 1/2) characterized `OrtRunBinding` as the "fixed-size,
region-count-independent GRU+attention step" and used its live cost (2-3× the encoder's) as
evidence the encoder-side scoring cost (ADR 0038's actual subject) isn't the bottleneck. That
framing was incomplete: the GRU/attention nodes genuinely are region-size-independent, but the
graph as a whole is not, because of the dense region-projection matmul described above. A
benchmark run at the end of the previous session (`scripts/bench_binding_projection.py`) isolated
the two: at campaign scale (n_units=9,660), the dense projection was **75-76% of the full binding
graph's per-call cost**, confirming this is a real, large, and previously mischaracterized cost —
not the marginal one an earlier FLOP-based estimate assumed (the projection is memory-bandwidth-
bound: every call reads the full `W_region_i`, ~4.9MB × 5 regions, regardless of input sparsity —
exactly the profile a gather-based rewrite should cut close to proportionally).

## Decision

Port §25's already-proven sparse-gather technique to the ONNX/MIGraphX binding graph: replace each
region's dense `sp_i @ W_region_i` with `Gather(W_region_i, active_idx_i, axis=0)` (reads only the
k active rows) followed by a `[1,k] @ [k,D_PROJ]` weighted-sum matmul — numerically identical to
the dense form, since every non-gathered row contributes exactly zero to the dense matmul's sum.

**Contract change:** the binding graph's `sp_0..sp_4 [1, n_units]` dense inputs become
`active_idx_0..4 [k_active_i]` (int64) + `active_val_0..4 [1, k_active_i]` (float) pairs, with
`k_active_i` baked into the graph's static shape at export time (same re-export-on-growth
requirement that already existed for `n_units`, generalized to also cover `k_active`). Callers must
supply exactly `k_active_i` entries; `SparseActivation::to_exact_k` (new,
`crates/somnus-perception/src/region.rs`) pads short inputs with `(index 0, value 0.0)` — a Gather-then-
zero-weighted contribution that is always exactly nil, regardless of which row index 0 happens to
be — or truncates long ones, covering every caller uniformly with no branching. `k_active` per
region travels via `weights_manifest.json` (written by
`PerceptionProcessor::export_weights_for_onnx` straight from each `KWtaLayer`'s live `k_active`
field, not recomputed from `n_units` in either Rust or Python — see "Alternatives considered").
`OnnxPerceptionSession::run_binding` (`crates/somnus-perception/src/onnx_session.rs`) now takes
`&[SparseActivation; 5]` directly instead of `&[Vec<f32>; 5]` dense vectors, applying `to_exact_k`
internally; callers no longer construct dense vectors or look up `n_units` themselves.

## Alternatives considered

- **Re-derive `k_active` from `n_units` in Python** (`k = scaled_k(n_units)`, mirroring the
  existing Rust formula): rejected. `k_active` is *currently* always exactly `scaled_k(n_units)`
  per `KWtaLayer`'s own invariant, but re-deriving a Rust-owned value in Python instead of reading
  it from the export is exactly the bug class `gaps.md` §17/§22/§24 hit three times already (a
  second hardcoded/recomputed copy of a Rust ground-truth value silently drifts). Reading
  `k_active` straight from the manifest costs nothing extra once already writing the file, and
  can't drift even if a future code path ever sets `k_active` to something other than
  `scaled_k(n_units)`.
- **Dense-form graph unchanged, rely on GPU throughput to absorb the wasted FLOPs**: this was
  effectively the status quo assumption behind §28/§29's characterization. Rejected once the
  benchmark showed the dense projection dominant (not marginal) at campaign scale — see Context.
- **Elementwise `Mul` + `ReduceSum` instead of `[1,k] @ [k,D_PROJ]` MatMul**: the first working
  design (validated via a standalone probe addition to `experiments/migraphx-op-probe/probe.py`,
  which passed). Replaced with the matmul form because it needs no reshape of `proj_i` for the
  existing downstream `Concat(axis=0)` — same numerical result, fewer graph nodes. This substitution
  is also what surfaced the `Reshape` bug described below, since the matmul form initially still
  used a `Reshape` on `active_val_i` before that was removed too.

## Consequences

- The binding graph now needs re-export (`rebuild_ort_session`) whenever `k_active` changes,
  exactly as it already did for `n_units` — growth changes both together via `scaled_k`, so this
  doesn't introduce a new re-export trigger, only generalizes the existing one.
- `weights_manifest.json`'s top-level shape changed from a bare tensor-entry array to
  `{"tensors": [...], "k_active": [...]}`. Nothing else reads this file (checked: only the Rust
  writer and the Python loader) — no other consumers to update.
- The pre-rewrite dense-projection graph builder (previously `build_binding_graph`) is preserved as
  `build_binding_graph_dense` in `scripts/bench_binding_projection.py`, kept solely to give the
  benchmark a "full" baseline arm to compare against — not used by the production export path.
- `crates/somnus-perception/tests/ort_grown_region.rs`'s grown-region regression test was rewritten:
  the old version parsed ORT's dense-shape-mismatch error message to discover the graph's expected
  `n_units` (that mechanism doesn't exist under the new fixed-`k` contract, since there's no dense
  shape to mismatch against). Replaced with a test that proves the actual new guarantee — a
  deliberately mismatched-length `SparseActivation` still round-trips correctly through
  `run_binding` — which is a stronger regression check for this contract than the error-message
  approach ever was.

## Verification

- **Numerical parity: exact.** `scripts/export_perception_graph.py --validate-numpy
  --validate-migraphx` (baseline scale, CPU + MIGraphX EP) and an ad hoc campaign-scale check
  (n=9,660, k=190, CPU + MIGraphX EP) both diff against the dense `numpy_binding` reference at
  ~1e-8 — `numpy_binding` was deliberately left in its original dense form as the reference (not
  also rewritten), so this proves equivalence to the pre-rewrite computation, not just
  self-consistency of the new code.
- **MIGraphX Gather-with-runtime-indices: clean.** `experiments/migraphx-op-probe/probe.py` gained
  a probe (`Gather` fed by a CPU-computed int64 index graph input, not an initializer) at campaign
  scale — PASS, max abs diff 1.49e-07.
- **A real MIGraphX compiler bug found and worked around, unrelated to the Gather question the
  probe checked:** the natural design reshaped `active_val_i`'s `[k]` input to `[1,k]` via an ONNX
  `Reshape` node before the matmul. That combination crashed MIGraphX's `simplify_reshapes` graph
  pass (`cannot create std::vector larger than max_size()`) — a MIGraphX-internal bug, not an ORT
  or somnus bug. Fixed by declaring `active_val_i` as `[1,k]` directly at the graph-input level,
  eliminating the `Reshape` node entirely; no functional change. This is the concrete reason the
  full-graph numerical-parity check (not just the isolated op probe) is load-bearing — an op can be
  individually clean on MIGraphX and a specific node combination involving it still isn't.
- **Measured win: real and large.** Added a 3rd arm ("gather") to the already-committed
  `scripts/bench_binding_projection.py`, at campaign scale (n_units=9,660, k_active=189) on the
  real `MIGraphXExecutionProvider`, two reruns: full (dense) 0.856/0.853 ms/call → gather
  0.393/0.350 ms/call, a **54-59% reduction** (≈2.2-2.4× speedup) on `OrtRunBinding`'s per-call
  cost, stable across reruns.
- **Build/test/lint:** `cargo build/test/clippy -p somnus-perception` clean (whole-workspace build is
  currently blocked by an unrelated pre-existing `ffmpeg-next` version mismatch, not touched by
  this change). `cargo fmt` clean. 3 new unit tests for `SparseActivation::to_exact_k` (pad/
  truncate/pass-through). Both `ort_grown_region.rs` integration tests correctly skip (not fail,
  not silently pass) against the live campaign's still-pre-rewrite on-disk ONNX artifact.

## Metrics

Covered entirely by the existing `OrtRunBinding` Core metric (`somnus-metrics`,
`crates/somnus-perception/src/onnx_session.rs`) — no new metric needed for the timing side, per the
project's metrics-discipline policy: this is a before/after A/B on a call path that was already
instrumented, not a new operation.

**One new failure mode considered and deliberately not instrumented separately:** an out-of-range
`Gather` index is undefined behavior in the ONNX spec and might not fail loudly the way the old
dense shape-mismatch did. Evaluated and judged not to need a new degradation counter: every index
`SparseActivation` can produce comes from a `KWtaLayer`'s own top-k selection over exactly
`n_units` candidates (invariant covered by the existing `kwta_indices_sorted_and_in_range` test),
and the one synthetic index this change introduces — the `to_exact_k` padding index, always `0` —
is valid for any `n_units ≥ 1`. There is no code path that can hand `run_binding` an
out-of-range index today; adding a counter for a scenario the existing invariants already prevent
would be validating something that can't happen, which the project's own conventions call out to
avoid. If a future change weakens that invariant, this should be revisited. Any other failure
inside the ORT call (including a hypothetical MIGraphX-side Gather issue) still surfaces through
`run_binding`'s existing `anyhow::Result<Vec<f32>>` and is already caught by all 3 call sites'
existing `OrtBindingFallback` counter + fallback-to-burn logic — unchanged by this ADR.

## Not yet done

- **Live A/B against the real `OrtRunBinding` Core metric**, in an isolated pilot memory directory
  (the established precedent for this kind of validation run in this repo) rather than the
  synthetic benchmark above. Not run this session.
- **Deployment to the live growth campaign** (98.3% to its 10x target as of the last shutdown,
  currently stopped). This mutates a long-running production run's on-disk ONNX artifacts
  (`rebuild_ort_session` regenerates `weights_manifest.json` in the new format) — deliberately not
  done without explicit confirmation, since it's exactly the kind of hard-to-casually-reverse
  action this project's working conventions ask to check before taking.

## What would invalidate this

- The measured 54-59% win doesn't hold up under a live `OrtRunBinding` A/B on real campaign data
  (the synthetic benchmark uses random weights and a fixed access pattern; real k-WTA index
  distributions or MIGraphX's runtime behavior under sustained production load could differ).
- A future MIGraphX/ROCm update changes `Gather`-with-runtime-indices behavior on gfx1201 (the
  probe result is a point-in-time observation of this specific software stack, not a guarantee).
- The growth campaign's `k_active` invariant (`k_active == scaled_k(n_units)` always) is ever
  broken by a future change — the manifest-based design tolerates this (still reads the true live
  value), but the "why we didn't add a new failure-mode counter" reasoning under Metrics would need
  revisiting.
