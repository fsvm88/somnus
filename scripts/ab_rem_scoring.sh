#!/usr/bin/env bash
# A/B for gaps.md R1 (+R4/R8, numerics-neutral): REM scoring passes (`measure_mse`,
# `collect_high_residual_inputs`) no longer advance k-WTA duty cycles.
#
#   control   = commit aded522 (before R1/R4/R8): scoring calls KWtaLayer::forward -> duty cycles advance
#   treatment = HEAD of this tree: scoring uses forward_no_dc, batched
#
# Everything else identical: same snapshot, same SEED, same clips, no extra flags (production
# path, ORT/GPU as in a normal campaign). Both arms built with --features fine-metrics (standing
# A/B policy) and pinned to per-arm binaries via SOMNUS_BIN, so rebuilding target/ mid-run is safe.
#
# Resource note (measured on the earlier freeze-heads pilot): ~18 GiB RSS per arm, GPU shared.
# PARALLEL=1 runs both arms at once (~36 GiB RSS); default is sequential.
#
# Usage:
#   bash scripts/ab_rem_scoring.sh                    # build + copy + run (sequential)
#   PARALLEL=1 CYCLES=10 bash scripts/ab_rem_scoring.sh
#   SKIP_BUILD=1 SKIP_COPY=1 bash scripts/ab_rem_scoring.sh   # re-run/analyze only
#   PREP_ONLY=1 bash scripts/ab_rem_scoring.sh                 # build + copy, no run
#   ANALYZE_ONLY=1 bash scripts/ab_rem_scoring.sh              # comparison only
set -euo pipefail
export SEED="${SEED:-42}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CYCLES="${CYCLES:-10}"
PILOT_ROOT="${PILOT_ROOT:-memory/ab_rem_scoring}"
SRC_MEMORY="${SRC_MEMORY:-/data/somnus/memory}"
CONTROL_REV="${CONTROL_REV:-aded522}"
CONTROL_WT="${CONTROL_WT:-$PILOT_ROOT/control-worktree}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_COPY="${SKIP_COPY:-0}"
PARALLEL="${PARALLEL:-0}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
TAG="[ab-rem-scoring]"

CLIPS="$(grep '^CLIPS=' "$SRC_MEMORY/campaign_growth10x/params.txt" | tail -1 | cut -d= -f2-)"
mkdir -p "$PILOT_ROOT/bin"

if [ "$ANALYZE_ONLY" != "1" ]; then
if [ "$SKIP_BUILD" != "1" ]; then
    echo "$TAG building treatment (HEAD, fine-metrics)…"
    cargo build --release -p somnus --features fine-metrics 2>&1 | tail -2
    cp target/release/somnus "$PILOT_ROOT/bin/somnus-treatment"
    echo "$TAG building control ($CONTROL_REV, fine-metrics) in $CONTROL_WT…"
    [ -d "$CONTROL_WT" ] || git worktree add --detach "$CONTROL_WT" "$CONTROL_REV"
    # Share the main target dir's dependency cache is NOT possible across paths; use its own.
    (cd "$CONTROL_WT" && cargo build --release -p somnus --features fine-metrics 2>&1 | tail -2)
    cp "$CONTROL_WT/target/release/somnus" "$PILOT_ROOT/bin/somnus-control"
fi
test -x "$PILOT_ROOT/bin/somnus-control" && test -x "$PILOT_ROOT/bin/somnus-treatment"

if [ "$SKIP_COPY" != "1" ]; then
    for arm in control treatment; do
        rm -rf "$PILOT_ROOT/$arm"
        echo "$TAG copying trained state → $PILOT_ROOT/$arm …"
        mkdir -p "$PILOT_ROOT/$arm"
        for d in snapshot warm episodes.cold onnx_models; do
            cp -a --reflink=auto "$SRC_MEMORY/$d" "$PILOT_ROOT/$arm/$d"
        done
    done
fi

if [ "${PREP_ONLY:-0}" = "1" ]; then
    echo "$TAG prep done (binaries in $PILOT_ROOT/bin, arm state copied); run without PREP_ONLY and with SKIP_BUILD=1 SKIP_COPY=1."
    exit 0
fi

run_arm() {
    local arm="$1"
    echo "$TAG ── $arm ──"
    SOMNUS_BIN="$REPO_ROOT/$PILOT_ROOT/bin/somnus-$arm" \
    SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 CLIPS="$CLIPS" \
        MEMORY_DIR="$PILOT_ROOT/$arm" CAMPAIGN_DIR="$PILOT_ROOT/$arm/campaign" \
        SKIP_BUILD=1 bash scripts/train_campaign.sh
}
if [ "$PARALLEL" = "1" ]; then
    run_arm control > "$PILOT_ROOT/control.out" 2>&1 &
    pc=$!
    run_arm treatment > "$PILOT_ROOT/treatment.out" 2>&1 &
    pt=$!
    wait $pc && wait $pt
else
    run_arm control
    run_arm treatment
fi
fi

echo ""
echo "$TAG both arms complete — comparison:"
python3 - "$PILOT_ROOT" <<'PYEOF'
import json, statistics, struct, sys
root = sys.argv[1]

def load(arm):
    with open(f"{root}/{arm}/campaign/metrics.jsonl") as f:
        return [r for r in map(json.loads, f) if "cycle" in r and "session" in r]

ctrl, trt = load("control"), load("treatment")
if not ctrl or not trt:
    sys.exit("missing metrics for at least one arm")

def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(v) if v else None

def fm(rows, name):  # (count, mean ms) of a fine/core timer across cycles
    c = sum((r.get("fine_metrics") or {}).get(name, {}).get("count", 0) for r in rows)
    ns = sum((r.get("fine_metrics") or {}).get(name, {}).get("total_ns", 0) for r in rows)
    return c, (ns / c / 1e6 if c else 0.0), ns / 1e9

print(f"\n{'metric':34} {'control':>13} {'treatment':>13} {'delta':>9}")
print("-" * 72)
for k in ["mean_output_mse", "mean_output_mse_wake", "mean_real_inference_mse",
          "mean_synthetic_output_mse", "mean_loss", "mean_proj_loss", "gist_recall_rate",
          "within_session_gist_rate", "warm_recall_l2", "nrem_wall_ms", "rem_wall_ms",
          "sal_threshold", "settle_iters_sum"]:
    c, t = mean(ctrl, k), mean(trt, k)
    if c is None or t is None:
        continue
    print(f"{k:34} {c:13.6f} {t:13.6f} {(t-c)/max(abs(c),1e-9)*100:+8.1f}%")
for i in range(len(ctrl[-1].get("per_region_mse") or [])):
    c = statistics.mean(r["per_region_mse"][i] for r in ctrl if r.get("per_region_mse"))
    t = statistics.mean(r["per_region_mse"][i] for r in trt if r.get("per_region_mse"))
    print(f"{'per_region_mse.'+str(i):34} {c:13.6f} {t:13.6f} {(t-c)/max(abs(c),1e-9)*100:+8.1f}%")

print("\n-- R4 timers (total seconds across all cycles; calls) --")
for name in ["measure_mse", "collect_high_residual_inputs", "kwta_score_matmul", "kwta_select_topk", "sleep_rem"]:
    cc, cm, cs = fm(ctrl, name); tc, tm, ts = fm(trt, name)
    print(f"{name:30} control {cs:8.1f}s ({cc} calls)  treatment {ts:8.1f}s ({tc} calls)")

print("\n-- growth / health --")
for arm, rows in (("control", ctrl), ("treatment", trt)):
    grew = sum(1 for r in rows if r.get("grew"))
    nan = sum((r.get("fine_metrics") or {}).get("training_nan_detected", {}).get("count", 0) for r in rows)
    sizes = struct.unpack("<5Q", open(f"{root}/{arm}/snapshot/region_sizes.bin", "rb").read())
    print(f"{arm}: cycles={len(rows)} grew_events={grew} regions_grown={[r.get('regions_grown') for r in rows if r.get('regions_grown')]} "
          f"training_nan={nan} final_region_sizes={list(sizes)}")

print("\n-- per-cycle growth flag / mean_output_mse (divergence is the R1 signal) --")
for a, b in zip(ctrl, trt):
    print(f"cycle {a['cycle']:>2}: grew {a.get('grew')!s:5}/{b.get('grew')!s:5}  "
          f"mse {a.get('mean_output_mse',0):.5f}/{b.get('mean_output_mse',0):.5f}")

def flat(r):
    o = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("session", "cycle", "ts")}
    for k, v in (r.get("fine_metrics") or {}).items():
        if isinstance(v, dict) and "count" in v:
            o[f"fm.{k}.count"] = v["count"]
    return o
ac, at = {}, {}
for rows, agg in ((ctrl, ac), (trt, at)):
    for r in rows:
        for k, v in flat(r).items():
            agg.setdefault(k, []).append(v)
print("\n-- deviations (|delta| > 30%, or one-arm-only) --")
dev = False
for k in sorted(set(ac) | set(at)):
    c = statistics.mean(ac[k]) if k in ac else None
    t = statistics.mean(at[k]) if k in at else None
    if c is None or t is None:
        print(f"  ONE-ARM ONLY: {k}: control={c} treatment={t}"); dev = True; continue
    if abs(c) < 1e-9:
        if abs(t) > 1e-6: print(f"  {k}: control~0 -> treatment={t:.6g}"); dev = True
    elif abs((t - c) / c) > 0.30:
        print(f"  {k}: {c:.6g} -> {t:.6g} ({(t-c)/c*100:+.0f}%)"); dev = True
if not dev:
    print("  none")
print("\nNOTE: no duty-cycle statistic exists in metrics.jsonl; R1's direct effect is only visible "
      "through the downstream metrics above.")
PYEOF
echo ""
echo "$TAG done. Artifacts: $PILOT_ROOT/{control,treatment}/campaign/metrics.jsonl"
