# 0037. Auto-enable ORT/MIGraphX perception inference past the CPU/GPU crossover

- **Status:** accepted and confirmed. Auto-switch mechanism implemented and correct; its target was broken by two engineering-log.md §24 bugs (found 2026-07-12/13), both now fixed and verified live — real speedup confirmed at ~3-4× over the pre-crossover CPU baseline. See updates below.
- **Date:** 2026-07-12

> **⚠️ Update (2026-07-12, later):** the "~2.1×/2.4× measured speedup" claims below are almost certainly invalid. engineering-log.md §24 found the sparse→dense conversion feeding the ORT graph (`to_dense_n(N_UNITS)`) hardcodes the *baseline* `N_UNITS` instead of each region's grown size, causing every ORT call to fail dimension checks and silently fall back to burn/CPU for the entire time ORT was "active" on the campaign. The verification below never actually observed a successful accelerated call; the auto-switch mechanism itself (this ADR's actual subject) is sound and unaffected, but its payoff is unconfirmed until §24 is fixed and re-measured with zero `ORT encoder/binding failed` occurrences as the pass condition.
>
> **⚠️ Update (2026-07-13):** §24's binding-side bug fixed — verified against the real leftover campaign ONNX graph (n_units=6886), which now round-trips through `run_binding` correctly (regression test `ort_grown_region.rs`, fails pre-fix/passes post-fix). Resuming the campaign to verify live surfaced a *second*, independent bug in the same path: the encoder's `language_enc` input dim was hardcoded at 896 (stale since ADR 0036 raised LANG_HIDDEN to 1024) — same drift class, different call site. Fixed the same way, same fail-pre/pass-post regression test.
>
> **✅ Update (2026-07-13, later): fully confirmed live.** Campaign resumed at session 60 with both fixes deployed. Sessions 60-61: **zero** `ORT encoder failed`/`ORT binding failed` across ~8,000 log lines. `perception_total_ms` mean 2,158ms (session 60), 1,457ms (session 61) — vs. session 57's 250,099ms fully-broken baseline and the pre-crossover ~6,900-7,200ms baseline (sessions 50-52). **Real confirmed speedup: ~3-4× over CPU baseline**, genuinely via ORT/MIGraphX, not silent fallback. Supersedes and retracts the original "~2.1×/2.4×" below (measured entirely on the broken path, never a real accelerated call). Mechanism and payoff both now confirmed; no further action pending.

## Context

ADR 0023 added an opt-in `--onnx-models-dir` flag to run perception inference (k-WTA, symbolic cross-modal projection, binding GRU/attention) through ORT+MIGraphX instead of burn/CPU. Its scaling analysis found the CPU/GPU crossover at N_UNITS≈1,230/region (batch-1) — below wins CPU, above wins GPU by an increasing margin (2.2× at 4K units batch-1, 6.5× with micro-batching). Nobody had actually run past that threshold: the flag needs manually remembering to pass, and every §16 growth pilot (and the full 10x campaign this ADR was written during) ran with it off.

Once the full 10x campaign (engineering-log.md §16/§20) grew regions past ~5,000 units, `perception_total_ms` was measured directly from `metrics.jsonl`: 9.1s at 1,986 units, 14.9s at 5,258, 16.1s at 6,043 — climbing steadily, already over half the ~31s/cycle wall-clock, with `perc_kwta_us`/`perc_symbolic_us` (exactly the GPU-accelerable ops) dominating. The opt-in flag was costing real, measured, worsening wall-clock for no reason but nobody having turned it on.

## Decision

Auto-enable ORT/MIGraphX once any region crosses `--onnx-crossover-units` (default 1,230, ADR 0023's empirical figure), instead of requiring `--onnx-models-dir` to be passed. That flag still overrides *where* graphs are written, no longer controls *whether* ORT is used. Growth only ever increases region size (pruning is explicit/user-invoked), so the switch is one-directional — once active, stays active for the process's life, no downgrade logic. New `--force-cpu-inference` opts out entirely (testing, reproducing pre-ADR-0037 runs, machines without a working Python3+onnx+onnxruntime env or MIGraphX GPU).

Implementation: `should_use_ort_inference(region_sizes, threshold)` (pure, unit-tested) checked once at startup (covers a resumed already-grown snapshot) and after every sleep cycle (covers mid-run growth). A sticky `onnx_failed` flag prevents retrying every cycle on a broken environment — logged once, stays on CPU for the process's life rather than re-attempting a doomed subprocess call every ~30s.

## Bugs found and fixed as a prerequisite

Enabling this surfaced two pre-existing `export_perception_graph.py` bugs, never exercised since nobody had run `--onnx-models-dir` past the crossover:

1. **Stale hardcoded dimensions.** Hardcoded `N_UNITS=1024`, `K_ACTIVE=20`, `REGION_DIMS=[384,512,384,384,896]` — the 896 (language) went stale when ADR 0036 adopted Qwen3 (LANG_HIDDEN 896→1024). Crashed the first smoke test with an ONNX MatMul shape-inference error. More fundamentally, a hardcoded `N_UNITS` can never survive growth (non-uniform across regions — pilot 1 observed `[2048,1984,1984,1984,1984]`), so fixing just the 896 wouldn't have been enough. Fixed by deriving `region_dims`/`n_units`/`cross_modal_dim` from the actual exported weight tensor shapes (`infer_dims()`) — the only source that can't drift from Rust's real state, since it *is* that state.
2. **`make_random_weights`** (the `--validate-numpy` self-test's synthetic generator) never created `W_cross_bias`, though both graph construction and the numpy reference expect it — likely missed when `cross_modal_proj` gained a bias term; `--validate-numpy` had apparently been silently broken since. Fixed by adding it.

Both verified: `--validate-numpy` passes (ep=6.64e-09 vs. numpy), and a real export against a copy of the live campaign's grown weights (6,397 units/region, well past crossover) built and loaded through MIGraphX with no shape errors. A full sleep cycle on that copy: `perception_total_ms=7,735` at 6,397 units/region vs. the live CPU campaign's 16,051 at comparable 6,043 — a **~2.1× speedup**, matching ADR 0023's predicted crossover-scaling figure (2.2× at 4K units batch-1) almost exactly.

## Alternatives considered

1. **Native Rust ONNX graph construction** (proposed mid-implementation as the "real" fix) — eliminates the Python runtime dependency and solves dynamic shapes for free. No mature "build ONNX graphs from Rust" crate exists (only raw protobuf bindings, no builder layer), so this means hand-constructing `ModelProto`/`GraphProto`/`NodeProto` correctly enough for ORT+MIGraphX — genuinely open-ended, real correctness risk. Deliberately deferred (tracked in gaps.md).
2. **Fix only the stale 896, leave N_UNITS/K_ACTIVE hardcoded.** Would unblock this campaign's current size but break again at the next growth event. Deriving everything from actual weight shapes costs almost nothing extra once already touching the file and can't drift again.
3. **Keep the flag opt-in, document the crossover in gaps.md.** Leaves the real measured worsening cost in place for anyone who forgets the flag — defeats the point of ADR 0023's crossover analysis.

## Consequences

- Every `somnus` invocation whose regions cross 1,230 units now needs a working Python3+onnx+onnxruntime environment (previously only true for explicit `--onnx-models-dir` users), unless `--force-cpu-inference`.
- `export_perception_graph.py` is now the load-bearing, eventually-always-exercised path for any campaign past baseline, not a rarely-touched opt-in tool — reinforces the case for eventually retiring it for native Rust construction.
- All three §16 growth pilots and the first ~490 cycles of the full 10x campaign (§20) ran without this — not directly comparable to post-restart timing/growth-rate numbers.

## What would invalidate this

- MIGraphX produces incorrect results for GRU/TopK/attention at large region sizes with no workaround (ADR 0023 flagged this risk generally, not yet observed).
- The crossover threshold turns out wrong at the sizes this campaign actually reaches — 1,230 came from ADR 0023's analysis up to ~4× baseline, not the ~9.4× (9,660 units) this campaign targets; re-measure `perception_total_ms` rather than assume linear extrapolation holds.
- The research direction moves away from region-scaling entirely (ADR 0023's own invalidation condition applies transitively).
