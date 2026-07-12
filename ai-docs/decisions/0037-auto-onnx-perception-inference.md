# 0037. Auto-enable ORT/MIGraphX perception inference past the CPU/GPU crossover

- **Status:** accepted; implemented (2026-07-12)
- **Date:** 2026-07-12

## Context

ADR 0023 added an opt-in `--onnx-models-dir` flag to run perception inference (k-WTA, symbolic
cross-modal projection, binding GRU/attention) through ORT+MIGraphX instead of burn/CPU. Its own
scaling analysis (`research/gpu-unification-design.md` §2) found the CPU/GPU crossover at
N_UNITS≈1,230 per region (batch-1) — below that CPU wins, above it GPU wins by an increasing
margin (2.2× at 4K units batch-1, 6.5× with micro-batching). Nobody had actually run past that
threshold in practice: the flag requires manually remembering to pass it, and the §16 growth
pilots (and the full 10x campaign this ADR was written during) all ran with it off.

Once the full 10x campaign (gaps.md §16/§20) grew regions past ~5,000 units, `perception_total_ms`
was measured directly from the campaign's own `metrics.jsonl`: 9.1s at 1,986 units, 14.9s at
5,258, 16.1s at 6,043 — climbing steadily and already over half the ~31s/cycle wall-clock, with
`perc_kwta_us` and `perc_symbolic_us` (exactly the ops ADR 0023 flagged as GPU-accelerable)
dominating the growth. The opt-in flag was costing real, measured, worsening wall-clock for no
reason other than nobody had turned it on.

## Decision

Auto-enable ORT/MIGraphX perception inference once any region's size crosses a threshold
(`--onnx-crossover-units`, default 1,230 — ADR 0023's empirical crossover), instead of requiring
`--onnx-models-dir` to be passed explicitly. `--onnx-models-dir` still exists to override *where*
the graphs are written (default `<memory-dir>/onnx_models`), but no longer controls *whether* ORT
is used. Growth only ever increases region size (pruning is explicit and user-invoked, not
automatic), so the switch is one-directional: once ORT activates it stays active for the rest of
the process, no downgrade logic needed. A new `--force-cpu-inference` flag opts out entirely, for
testing, reproducing pre-ADR-0037 runs (all three §16 pilots), or machines without a working
Python3+onnx+onnxruntime environment or MIGraphX GPU.

Implementation: `should_use_ort_inference(region_sizes, threshold)` (pure, unit-tested,
`crates/somnus/src/main.rs`) is checked once at startup (covering a resumed, already-grown snapshot)
and after every sleep cycle (covering growth that happens mid-run). A sticky `onnx_failed` flag
prevents retrying every cycle if the environment is broken (missing Python deps, MIGraphX
unavailable) — the failure is logged once and the process stays on CPU for its lifetime rather
than re-attempting a doomed subprocess call every ~30s.

## Bugs found and fixed as a prerequisite

Enabling this surfaced two pre-existing bugs in `scripts/export_perception_graph.py` that had
never been exercised in practice, because nobody had run `--onnx-models-dir` past the crossover
before:

1. **Stale hardcoded dimensions.** The script hardcoded `N_UNITS=1024`, `K_ACTIVE=20`, and
   `REGION_DIMS=[384,512,384,384,896]` — the `896` (language dim) went stale when ADR 0036 adopted
   Qwen3 (`LANG_HIDDEN` 896→1024) and nobody updated this second copy. This alone crashed the
   first smoke test with an ONNX `MatMul` shape-inference error. More fundamentally, a hardcoded
   `N_UNITS` can never be correct once regions grow — and growth can be non-uniform across regions
   (gaps.md §16 pilot 1 observed `[2048, 1984, 1984, 1984, 1984]`), so even "just fix the 896"
   wouldn't have been sufficient. Fixed by deriving `region_dims`/`n_units`/`cross_modal_dim`
   directly from the actual weight tensor shapes Rust exports (`infer_dims()`) — the only source
   that cannot drift from what the Rust model is actually running, since it *is* the Rust model's
   state, not a second manually-maintained copy of it.
2. **`make_random_weights` (the `--validate-numpy` self-test's synthetic weight generator) never
   created `W_cross_bias`**, even though the graph construction and numpy reference both expect
   it — likely missed whenever `cross_modal_proj` gained a bias term. `--validate-numpy` has
   apparently been silently broken since. Fixed by adding it.

Both fixes verified: `--validate-numpy` passes (`ep=6.64e-09` vs. the numpy reference), and a real
export against a copy of the live campaign's actual grown weights (6,397 units/region, well past
the crossover) built and loaded successfully through MIGraphX with no shape errors. A full sleep
cycle on that copy measured `perception_total_ms=7,735` at 6,397 units/region — compared to the
live (still-CPU) campaign's `perception_total_ms=16,051` at a comparable 6,043 units/region, a
**~2.1× speedup**, matching ADR 0023's predicted crossover-scaling figure (2.2× at 4K units
batch-1) almost exactly at a slightly larger scale.

## Alternatives considered

1. **Replace the Python export step with native Rust ONNX graph construction** — proposed
   mid-implementation as the "real" long-term fix (eliminates the Python3/onnx/onnxruntime runtime
   dependency entirely, and the dynamic-shape problem is solved for free since Rust already knows
   every shape exactly). Correct direction, but no mature "build ONNX graphs from Rust" crate
   exists (only raw protobuf-message bindings, e.g. `onnx-protobuf`/`onnx-ir` — no builder
   convenience layer), so this means hand-constructing `ModelProto`/`GraphProto`/`NodeProto`
   messages replicating the full graph (Concat, MatMul, Add, TopK/duty-cycle masking, the GRU
   cell, multi-head attention, and the documented MIGraphX ScatterElements two-graph-split
   workaround) correctly enough for ORT+MIGraphX to accept. Genuinely open-ended effort with real
   correctness risk — deliberately deferred to a dedicated future session rather than folded into
   this change. Tracked in `gaps.md`.
2. **Fix only the stale `896`, leave `N_UNITS`/`K_ACTIVE` hardcoded** — would have unblocked this
   specific campaign's current region size but broken again at the next growth event past whatever
   was hardcoded. Deriving everything from actual weight shapes costs almost nothing extra once
   already touching the file, and can't drift again by construction.
3. **Keep the flag opt-in, just document the crossover in `gaps.md`** — leaves the real, measured,
   worsening cost in place for anyone who doesn't remember to pass the flag; the whole point of
   ADR 0023's crossover analysis was to inform exactly this kind of automatic decision.

## Consequences

- Every `somnus` invocation whose regions cross 1,230 units now requires a working
  Python3+onnx+onnxruntime environment (previously only true for explicit `--onnx-models-dir`
  users) unless `--force-cpu-inference` is passed. Worth calling out as a new implicit runtime
  dependency for anyone deploying `somnus` elsewhere.
- `scripts/export_perception_graph.py` is now the load-bearing, always-eventually-exercised path
  for any campaign that grows past baseline, not a rarely-touched opt-in tool — its correctness
  matters more than it used to, reinforcing the case for eventually retiring it in favor of native
  Rust construction (alternative 1).
- All growth pilots (§16, three of them) and the first ~490 cycles of the full 10x campaign (§20)
  ran without this — not directly comparable to post-restart timing/growth-rate numbers.

## What would invalidate this

- MIGraphX produces incorrect results for GRU/TopK/attention at large region sizes and there's no
  workaround (ADR 0023 already flagged this risk generally; not yet observed).
- The crossover threshold turns out wrong at the sizes this campaign is actually reaching — the
  1,230 figure came from ADR 0023's analysis at up to ~4× baseline, not the ~9.4× (9,660 units)
  this campaign targets; re-measure `perception_total_ms` post-restart to confirm ORT still wins
  at that scale rather than assuming linear extrapolation holds.
- The research direction moves away from region-scaling entirely (ADR 0023's own invalidation
  condition still applies transitively).
