#!/usr/bin/env bash
# gaps.md §50 experiment: encoder-column norm of grown units (--grown-unit-init-scale).
#   control   = 0.01 (historical default; grown units start ~50x weaker than the original ~0.55)
#   treatment = GROWN_SCALE (default 0.5, about the original units' norm)
# Same binary/snapshot/SEED, fine-metrics, arms in parallel (~36 GiB RSS). Reads the per-region
# "kwta duty-cycle report" log lines for slot share / activation size, plus the usual MSE metrics.
# Usage: CYCLES=20 GROWN_SCALE=0.5 bash scripts/ab_grown_init_scale.sh   (SKIP_BUILD/SKIP_COPY/ANALYZE_ONLY)
set -euo pipefail
export SEED="${SEED:-42}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CYCLES="${CYCLES:-20}"; GROWN_SCALE="${GROWN_SCALE:-0.5}"
PILOT_ROOT="${PILOT_ROOT:-memory/ab_grown_init_scale}"
SRC_MEMORY="${SRC_MEMORY:-/data/somnus/memory}"
SKIP_BUILD="${SKIP_BUILD:-0}"; SKIP_COPY="${SKIP_COPY:-0}"; ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
CLIPS="$(grep '^CLIPS=' "$SRC_MEMORY/campaign_growth10x/params.txt" | tail -1 | cut -d= -f2-)"
mkdir -p "$PILOT_ROOT/bin"
if [ "$ANALYZE_ONLY" != "1" ]; then
    if [ "$SKIP_BUILD" != "1" ]; then
        cargo build --release -p somnus --features fine-metrics 2>&1 | tail -2
        cp target/release/somnus "$PILOT_ROOT/bin/somnus"
    fi
    if [ "$SKIP_COPY" != "1" ]; then
        for arm in control treatment; do
            rm -rf "$PILOT_ROOT/$arm"; mkdir -p "$PILOT_ROOT/$arm"
            for d in snapshot warm episodes.cold onnx_models; do cp -a --reflink=auto "$SRC_MEMORY/$d" "$PILOT_ROOT/$arm/$d"; done
        done
    fi
    for arm in control treatment; do
        extra="--grown-unit-init-scale 0.01"; [ "$arm" = "treatment" ] && extra="--grown-unit-init-scale $GROWN_SCALE"
        ( SOMNUS_BIN="$REPO_ROOT/$PILOT_ROOT/bin/somnus" SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 CLIPS="$CLIPS" \
            MEMORY_DIR="$PILOT_ROOT/$arm" CAMPAIGN_DIR="$PILOT_ROOT/$arm/campaign" EXTRA_ARGS="$extra" \
            SKIP_BUILD=1 bash scripts/train_campaign.sh > "$PILOT_ROOT/$arm.out" 2>&1 ) &
    done
    wait
fi
python3 - "$PILOT_ROOT" <<'PYEOF' | tee "$PILOT_ROOT/analysis.txt"
import json, re, statistics as st, struct, sys, glob
root = sys.argv[1]
ansi = re.compile(r"\x1b\[[0-9;]*m")
def load(a):
    with open(f"{root}/{a}/campaign/metrics.jsonl") as f:
        return [r for r in map(json.loads, f) if "cycle" in r and "session" in r]
D = {a: load(a) for a in ("control", "treatment")}
n = min(len(v) for v in D.values())
bins = [(1, n // 2), (n // 2 + 1, n)]
def mean(rows, k, lo, hi):
    v = [r[k] for r in rows if lo <= r["cycle"] <= hi and isinstance(r.get(k), (int, float))]
    return st.mean(v) if v else float("nan")
print(f"cycles {n}; bins {bins}")
print(f"{'metric':28} {'ctrl b1':>10} {'ctrl b2':>10} {'trt b1':>10} {'trt b2':>10}  b2 delta")
for k in ["mean_output_mse", "mean_output_mse_wake", "mean_real_inference_mse", "gist_recall_rate",
          "within_session_gist_rate", "warm_recall_l2", "sal_threshold", "settle_gate_dynamic",
          "settle_iters_sum", "mean_loss", "mean_proj_loss", "rem_wall_ms", "nrem_wall_ms"]:
    c = [mean(D["control"], k, *b) for b in bins]; t = [mean(D["treatment"], k, *b) for b in bins]
    d = (t[1] - c[1]) / max(abs(c[1]), 1e-9) * 100
    print(f"{k:28} {c[0]:10.5f} {c[1]:10.5f} {t[0]:10.5f} {t[1]:10.5f}  {d:+7.1f}%")
for i in range(5):
    c = [st.mean(r["per_region_mse"][i] for r in D["control"] if b[0] <= r["cycle"] <= b[1] and r.get("per_region_mse")) for b in bins]
    t = [st.mean(r["per_region_mse"][i] for r in D["treatment"] if b[0] <= r["cycle"] <= b[1] and r.get("per_region_mse")) for b in bins]
    print(f"{'per_region_mse.'+str(i):28} {c[0]:10.5f} {c[1]:10.5f} {t[0]:10.5f} {t[1]:10.5f}  {(t[1]-c[1])/max(abs(c[1]),1e-9)*100:+7.1f}%")
print("\n-- growth --")
for a, rows in D.items():
    sizes = struct.unpack("<5Q", open(f"{root}/{a}/snapshot/region_sizes.bin", "rb").read())
    print(f"{a:10} grew_cycles={[r['cycle'] for r in rows if r.get('grew')]} final_sizes={list(sizes)}")
print("\n-- k-WTA utilisation (last cycle's duty-cycle report: grown slot share / mean|z| base vs grown / dc max) --")
for a in D:
    rep = {}
    for f in glob.glob(f"{root}/{a}/campaign/logs/session_*.log"):
        for l in open(f, errors="replace"):
            if "kwta duty-cycle report" in l:
                d = dict(re.findall(r"(\w+)=(\[.*?\]|\S+)", ansi.sub("", l)))
                rep[int(d["region"])] = d
    for r in sorted(rep):
        d = rep[r]
        print(f"{a:10} region{r}: grown_share={float(d['grown_slot_share']):.2f} |z| base={float(d['mean_abs_active_base']):.4f} "
              f"grown={float(d['mean_abs_active_grown']):.4f} sum_dc base={float(d['sum_dc_base']):.1f} grown={float(d['sum_dc_grown']):.1f} "
              f"top1={d['top5'][:30]}")
print("\n-- duty metrics per cycle (first/mid/last): dev_mean, max --")
for a, rows in D.items():
    g = lambda k: [r['fine_metrics'].get(k, {}).get('count', 0) / 1e6 for r in rows]
    dv, mx = g('kwta_duty_dev_mean'), g('kwta_duty_max')
    pick = lambda v: [round(v[i], 3) for i in (0, len(v)//2, -1)]
    print(f"{a:10} dev_mean {pick(dv)} max {pick(mx)}")
PYEOF
