# 0040. Sparse-gather binding projection (ONNX/MIGraphX)

- **Status:** implemented and deployed live to the production growth campaign; confirmed via the real `OrtRunBinding` metric (see "Live deployment and verification").
- **Date:** 2026-08-12

## Context

gaps.md §25 found and fixed an 11.4× speedup in the *burn/CPU* binding projection (`BindingArea::forward`) by exploiting k-WTA output sparsity — gathering only the k active rows of each region's projection weight instead of a dense `n_units`-length matmul on a mostly-zero vector. That fix turned out **dormant** on the production campaign: ORT inference (ADR 0037) had already crossed over, and `try_bind`'s ORT branch never reaches burn's `BindingArea::forward` — the ONNX binding graph (`export_perception_graph.py::build_binding_graph`) still ran the equivalent dense `sp_i[1,n_units]@W_region_i[n_units,D_PROJ]` matmul per region, unaffected.

gaps.md §28/§29 (ADR 0038's prerequisites 1/2) had characterized `OrtRunBinding` as "fixed-size, region-count-independent GRU+attention" and used its cost (2-3× the encoder's) as evidence the encoder's scoring cost isn't the bottleneck. Incomplete: the GRU/attention nodes are region-size-independent, but the graph as a whole isn't, because of the dense region-projection matmul. A benchmark (`bench_binding_projection.py`) isolated the two: at campaign scale (n_units=9,660), the dense projection was **75-76% of the full binding graph's per-call cost** — real and large, not the marginal cost an earlier FLOP-based estimate assumed (memory-bandwidth-bound: every call reads the full `W_region_i`, ~4.9MB×5 regions, regardless of input sparsity — exactly what a gather rewrite should cut close to proportionally).

## Decision

Port §25's sparse-gather technique to the ONNX/MIGraphX binding graph: replace each region's dense `sp_i@W_region_i` with `Gather(W_region_i, active_idx_i, axis=0)` (reads only the k active rows) + a `[1,k]@[k,D_PROJ]` weighted-sum matmul — numerically identical, since every non-gathered row contributes exactly zero to the dense sum.

**Contract change:** binding-graph inputs go from dense `sp_0..sp_4[1,n_units]` to `active_idx_0..4[k_active_i]` (int64) + `active_val_0..4[1,k_active_i]` (float) pairs, `k_active_i` baked into the graph's static shape at export (generalizing the existing re-export-on-growth requirement for `n_units` to also cover `k_active`). `SparseActivation::to_exact_k` (new, `region.rs`) pads short inputs with `(index 0, value 0.0)` — always exactly nil regardless of what row 0 is — or truncates long ones, covering every caller with no branching. `k_active` per region travels via `weights_manifest.json` (written from each `KWtaLayer`'s live field, never recomputed from `n_units` — see Alternatives). `OnnxPerceptionSession::run_binding` now takes `&[SparseActivation;5]` directly instead of `&[Vec<f32>;5]`, applying `to_exact_k` internally.

## Alternatives considered

- **Re-derive `k_active` from `n_units` in Python** (`scaled_k(n_units)`, mirroring the Rust formula). Rejected — `k_active` is *currently* always `scaled_k(n_units)` by `KWtaLayer`'s invariant, but re-deriving a Rust-owned value in Python instead of reading it from the export is the exact bug class gaps.md §17/§22/§24 hit three times (a second hardcoded/recomputed copy silently drifts). Reading from the manifest costs nothing extra and can't drift even if a future path ever decouples `k_active` from `scaled_k(n_units)`.
- **Leave the dense graph unchanged, rely on GPU throughput.** Effectively the status quo assumption behind §28/§29 — rejected once the benchmark showed the dense projection dominant, not marginal.
- **Elementwise `Mul`+`ReduceSum` instead of `[1,k]@[k,D_PROJ]` MatMul.** The first working design (probe-validated). Replaced with the matmul form — needs no reshape of `proj_i` for the downstream `Concat(axis=0)`, same result, fewer nodes. This substitution is what surfaced the `Reshape` bug below (the matmul form initially still reshaped `active_val_i` before that was removed too).

## Consequences

- The binding graph now needs re-export whenever `k_active` changes, exactly as it already did for `n_units` — growth changes both together via `scaled_k`, so this generalizes an existing trigger rather than adding a new one.
- `weights_manifest.json`'s shape changed from a bare tensor-entry array to `{"tensors":[...],"k_active":[...]}` — only the Rust writer and Python loader read this file, no other consumers to update.
- The pre-rewrite dense builder is preserved as `build_binding_graph_dense` in the bench script, kept only as the benchmark's "full" baseline arm — not on the production export path.
- `ort_grown_region.rs`'s grown-region test rewritten: the old version parsed ORT's dense-shape-mismatch error to discover expected `n_units` (no such mismatch exists under the fixed-k contract). Replaced with a test proving the real new guarantee — a deliberately mismatched-length `SparseActivation` still round-trips correctly through `run_binding` — a stronger regression check than the old error-message approach.

## Verification

- **Numerical parity: exact.** `--validate-numpy --validate-migraphx` (baseline) and an ad hoc campaign-scale check (n=9,660, k=190, CPU+MIGraphX) both diff against the deliberately-unrewritten dense `numpy_binding` reference at ~1e-8 — proves equivalence to the pre-rewrite computation, not just internal self-consistency.
- **MIGraphX Gather-with-runtime-indices: clean.** New probe (`migraphx-op-probe/probe.py`), `Gather` fed by a CPU-computed int64 index graph input (not an initializer), campaign scale — PASS, max abs diff 1.49e-07.
- **A real MIGraphX compiler bug found and worked around**, unrelated to the Gather question the probe checked: reshaping `active_val_i`'s `[k]` input to `[1,k]` via an ONNX `Reshape` node crashed MIGraphX's `simplify_reshapes` pass (`cannot create std::vector larger than max_size()` — a MIGraphX-internal bug). Fixed by declaring `active_val_i` as `[1,k]` directly, eliminating the Reshape node. This is the concrete reason a full-graph parity check (not just the isolated op probe) is load-bearing — an op can be individually clean and a specific node combination involving it still isn't.
- **Measured win: real and large.** 3rd arm ("gather") added to `bench_binding_projection.py`, campaign scale (n_units=9,660, k_active=189), real MIGraphXExecutionProvider, two reruns: full (dense) 0.856/0.853ms/call → gather 0.393/0.350ms/call — **54-59% reduction** (≈2.2-2.4× speedup), stable across reruns.
- **Build/test/lint:** `somnus-perception` build/test/clippy/fmt clean (whole-workspace build blocked by an unrelated pre-existing ffmpeg-next mismatch). 3 new `to_exact_k` unit tests (pad/truncate/pass-through). Both `ort_grown_region.rs` integration tests correctly skip against the live campaign's still-pre-rewrite ONNX artifact.

## Metrics

Covered entirely by the existing `OrtRunBinding` Core metric — no new metric needed, per metrics discipline (before/after A/B on an already-instrumented call path, not a new operation).

**Considered and deliberately not instrumented:** an out-of-range `Gather` index (undefined behavior in the ONNX spec, might not fail as loudly as the old dense shape-mismatch). Every index a `SparseActivation` can produce comes from a `KWtaLayer`'s own top-k over exactly `n_units` candidates (covered by the existing `kwta_indices_sorted_and_in_range` test), and the one synthetic index this change introduces (`to_exact_k`'s padding index, always 0) is valid for any `n_units≥1` — no code path can hand `run_binding` an out-of-range index today. Adding a counter for something the existing invariants already prevent would validate a scenario that can't happen. Revisit if a future change weakens that invariant. Any other ORT-call failure still surfaces through `run_binding`'s existing `anyhow::Result` and the 3 call sites' existing `OrtBindingFallback` counter + fallback-to-burn logic, unchanged here.

## Live deployment and verification

User asked to deploy straight to production rather than stage through an isolated pilot. Two unrelated pre-existing environment blockers were fixed first (neither caused by this change):

- **`somnus` couldn't build:** `ffmpeg-next` 8 doesn't compile against the installed ffmpeg 9.0 (missing/renamed `AVCodec` fields, new enum variants). Fixed by bumping to `ffmpeg-next="9"` (not touching installed ffmpeg) — full workspace build/test/clippy/fmt clean afterward, including `somnus-sensory`'s real-file decode tests against ffmpeg 9.0. Committed separately (`3949b52`).
- **`models/` had silently diverged from `/data/somnus/models`:** unlike `memory`/`clips`/`data` (fully gitignored, symlinked wholesale), `models/` is git-tracked in part (`.gitignore` excludes specific artifact patterns, not the directory; `README.md`/`qwen3_tokenizer.json` are real tracked files). The 13GB of encoder weights lived only at `/data/somnus/models`. **First attempt (wrong, self-corrected same session):** replaced the whole `models/` directory with a symlink — would have silently detached the two tracked files from the working tree. **Fixed correctly:** restored `models/` as a real directory, symlinked only the individual large artifacts (each matching an existing `.gitignore` glob). Filesystem-only, not a commit — flagged since the first attempt would have been a real mistake had it been committed.

**Deployed via a short 2-cycle verification leg** (`SESSION_START=79 SESSIONS=1 CYCLES=2`, same params as the campaign's existing `params.txt`) rather than a fresh isolated pilot, since the campaign's own session budget was already exhausted at 98.3% growth and the user chose a short verification-only extension.

**Result: clean, zero fallback, real measured win.** Both cycles rebuilt the Gather-based binding graph without incident (zero `ORT binding failed`/`ORT encoder failed` across the run):

| | pre-rewrite (session 78, dense) | post-rewrite (session 79, gather) |
|---|---|---|
| cycle 9/1 | 2.151ms/call (n=136) | 1.298ms/call (n=230) |
| cycle 10/2 | 2.828ms/call (n=167) | 1.169ms/call (n=121) |

A **40-59% reduction** on real production traffic (vs. the synthetic benchmark's 54-59% — close but not identical, expected given real k-WTA index distributions and concurrent GPU load differ from the synthetic fixed pattern). Cycle 2 also caught a live growth event mid-run (9,493→9,549 units/region) with a clean re-export/rebuild — incidental confirmation the `k_active`-in-manifest coupling holds across a real growth event, not just static benchmarks. Campaign state after: session 79, `[9549]×5`, 98.9% to target, clean exit, zero panics/segfaults.

## What would invalidate this

- The 54-59% win not holding under a live `OrtRunBinding` A/B on real campaign data (the synthetic benchmark uses random weights and a fixed access pattern).
- A future MIGraphX/ROCm update changing `Gather`-with-runtime-indices behavior on gfx1201 (the probe result is a point-in-time observation of this software stack).
- The `k_active==scaled_k(n_units)` invariant ever being broken — the manifest-based design tolerates this (still reads the true live value), but the "why no new failure-mode counter" reasoning under Metrics would need revisiting.
