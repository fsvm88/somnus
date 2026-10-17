#!/usr/bin/env bash
# ADR 0048 validation pilot: --binding-wake-lag-cycles sweep, 4 arms, one variable per arm.
#   control   : no --binding-replay (baseline, as in every prior §33 pilot)
#   lag1      : --binding-replay --binding-wake-lag-cycles 1  (must reproduce §35/§47's magnitude)
#   lag10     : --binding-replay --binding-wake-lag-cycles 10
#   lag30     : --binding-replay --binding-wake-lag-cycles 30
# Same binary, snapshot, SEED, clips for all arms; fine-metrics on (standing A/B policy).
# Resources: ~18 GiB RSS per arm; ROUND_SIZE arms run concurrently (default 2 -> ~36 GiB).
#
# Usage:
#   CYCLES=40 bash scripts/ab_binding_wake_lag.sh
#   SKIP_BUILD=1 SKIP_COPY=1 bash scripts/ab_binding_wake_lag.sh
#   ANALYZE_ONLY=1 bash scripts/ab_binding_wake_lag.sh
set -euo pipefail
export SEED="${SEED:-42}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CYCLES="${CYCLES:-40}"
ROUND_SIZE="${ROUND_SIZE:-2}"
PILOT_ROOT="${PILOT_ROOT:-memory/ab_binding_wake_lag}"
SRC_MEMORY="${SRC_MEMORY:-/data/somnus/memory}"
SKIP_BUILD="${SKIP_BUILD:-0}"; SKIP_COPY="${SKIP_COPY:-0}"; ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
TAG="[ab-binding-wake-lag]"
ARMS=(control lag1 lag10 lag30)
declare -A ARGS=(
    [control]=""
    [lag1]="--binding-replay --binding-wake-lag-cycles 1"
    [lag10]="--binding-replay --binding-wake-lag-cycles 10"
    [lag30]="--binding-replay --binding-wake-lag-cycles 30"
)
CLIPS="$(grep '^CLIPS=' "$SRC_MEMORY/campaign_growth10x/params.txt" | tail -1 | cut -d= -f2-)"
mkdir -p "$PILOT_ROOT/bin"

if [ "$ANALYZE_ONLY" != "1" ]; then
if [ "$SKIP_BUILD" != "1" ]; then
    echo "$TAG building (release, fine-metrics)…"
    cargo build --release -p somnus --features fine-metrics 2>&1 | tail -2
    cp target/release/somnus "$PILOT_ROOT/bin/somnus"
fi
if [ "$SKIP_COPY" != "1" ]; then
    for arm in "${ARMS[@]}"; do
        rm -rf "$PILOT_ROOT/$arm"; mkdir -p "$PILOT_ROOT/$arm"
        for d in snapshot warm episodes.cold onnx_models; do
            cp -a --reflink=auto "$SRC_MEMORY/$d" "$PILOT_ROOT/$arm/$d"
        done
    done
fi
run_arm() {
    local arm="$1"
    SOMNUS_BIN="$REPO_ROOT/$PILOT_ROOT/bin/somnus" SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 CLIPS="$CLIPS" \
        MEMORY_DIR="$PILOT_ROOT/$arm" CAMPAIGN_DIR="$PILOT_ROOT/$arm/campaign" \
        EXTRA_ARGS="${ARGS[$arm]}" SKIP_BUILD=1 bash scripts/train_campaign.sh
}
i=0
while [ $i -lt ${#ARMS[@]} ]; do
    pids=()
    for arm in "${ARMS[@]:$i:$ROUND_SIZE}"; do
        echo "$TAG starting $arm"
        run_arm "$arm" > "$PILOT_ROOT/$arm.out" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    i=$((i + ROUND_SIZE))
done
fi

python3 - "$PILOT_ROOT" "${ARMS[@]}" <<'PYEOF' | tee "$PILOT_ROOT/analysis.txt"
import json, statistics as st, struct, sys
root, arms = sys.argv[1], sys.argv[2:]
def load(a):
    with open(f"{root}/{a}/campaign/metrics.jsonl") as f:
        return [r for r in map(json.loads, f) if "cycle" in r and "session" in r]
D = {a: load(a) for a in arms}
n = min(len(v) for v in D.values())
print("cycles per arm:", {a: len(v) for a, v in D.items()})
bins = [(1, 10), (11, 25), (26, n)] if n >= 26 else [(1, n)]
def mean(rows, k, lo, hi):
    v = [r[k] for r in rows if lo <= r["cycle"] <= hi and isinstance(r.get(k), (int, float))]
    return st.mean(v) if v else float("nan")
for k in ["mean_output_mse", "mean_output_mse_wake", "gist_recall_rate", "within_session_gist_rate",
          "warm_recall_l2", "sal_threshold", "settle_gate_dynamic", "settle_iters_sum", "mean_loss"]:
    print(f"\n{k}  (cycle bins {bins})")
    base = {b: mean(D["lag1"], k, *b) for b in bins}
    for a in arms:
        vals = [mean(D[a], k, *b) for b in bins]
        d = [f"{(v-base[b])/max(abs(base[b]),1e-9)*100:+6.1f}%" for v, b in zip(vals, bins)]
        print(f"  {a:8}", "  ".join(f"{v:10.5f}" for v in vals), "  vs lag1:", " ".join(d))
print("\n-- binding-wake sync / divergence (fine+core metrics) --")
for a in arms:
    keys = {}
    for r in D[a]:
        for k, v in (r.get("fine_metrics") or {}).items():
            if "binding_wake" in k and isinstance(v, dict):
                c = keys.setdefault(k, [0, 0]); c[0] += v.get("count", 0); c[1] += v.get("total_ns", 0)
    print(f"  {a:8}", keys or "no binding_wake metrics")
print("\n-- growth / health --")
for a in arms:
    rows = D[a]
    nan = sum((r.get("fine_metrics") or {}).get("training_nan_detected", {}).get("count", 0) for r in rows)
    sizes = struct.unpack("<5Q", open(f"{root}/{a}/snapshot/region_sizes.bin", "rb").read())
    print(f"  {a:8} grew_cycles={[r['cycle'] for r in rows if r.get('grew')]} nan={nan} sizes={list(sizes)}")
print("\n-- per-region mean_mse by bin --")
for a in arms:
    for b in bins:
        rows = [r for r in D[a] if b[0] <= r["cycle"] <= b[1] and r.get("per_region_mse")]
        if rows:
            print(f"  {a:8} {b}", [f"{st.mean(r['per_region_mse'][i] for r in rows):.5f}" for i in range(5)])
PYEOF
