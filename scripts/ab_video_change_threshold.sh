#!/usr/bin/env bash
# A/B: change-gated video encoder real wake-path savings (gaps.md §41).
#
# Control:   trained production state, no flag (--video-change-threshold 0.0, disabled)
# Treatment: same state + --video-change-threshold (skips enc.encode() below the
#            per-frame mean_abs_diff threshold)
#
# Both arms start from a copy of the trained campaign state (snapshot + warm + cold
# + onnx), same rationale as ab_binding_replay.sh: fresh memory has no grown units
# and doesn't reflect the actual deployed configuration's downstream sensitivity.
#
# Standing A/B instruction: both arms built and run with fine-metrics; the
# comparison below records deviations, not just the headline verdict.
#
# Arms run SEQUENTIALLY so wall-clock savings are measured without CPU/GPU
# contention from the sibling arm.
#
# Usage:
#   bash scripts/ab_video_change_threshold.sh
#   THRESHOLD=0.1 CYCLES=5 bash scripts/ab_video_change_threshold.sh
#   SKIP_BUILD=1 SKIP_COPY=1 bash scripts/ab_video_change_threshold.sh   # re-run/analyze only
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CYCLES="${CYCLES:-10}"
THRESHOLD="${THRESHOLD:-0.05}"
PILOT_ROOT="${PILOT_ROOT:-memory/ab_video_change_threshold}"
SRC_MEMORY="${SRC_MEMORY:-/data/somnus/memory}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_COPY="${SKIP_COPY:-0}"

# Same clip corpus as the growth campaign's final leg.
CLIPS="$(grep '^CLIPS=' "$SRC_MEMORY/campaign_growth10x/params.txt" | tail -1 | cut -d= -f2-)"

# ── Build once, fine-metrics on (standing A/B instruction) ───────────────────
if [ "$SKIP_BUILD" != "1" ]; then
    echo "[ab-video-change-threshold] building somnus (release, fine-metrics)…"
    cargo build --release -p somnus --features fine-metrics 2>&1 | tail -2
fi

# ── Copy the trained production state for each arm ───────────────────────────
if [ "$SKIP_COPY" != "1" ]; then
    rm -rf "$PILOT_ROOT"
    for arm in control treatment; do
        echo "[ab-video-change-threshold] copying trained state → $PILOT_ROOT/$arm …"
        mkdir -p "$PILOT_ROOT/$arm"
        cp -a --reflink=auto "$SRC_MEMORY/snapshot" "$PILOT_ROOT/$arm/snapshot"
        cp -a --reflink=auto "$SRC_MEMORY/warm" "$PILOT_ROOT/$arm/warm"
        cp -a --reflink=auto "$SRC_MEMORY/episodes.cold" "$PILOT_ROOT/$arm/episodes.cold"
        cp -a --reflink=auto "$SRC_MEMORY/onnx_models" "$PILOT_ROOT/$arm/onnx_models"
    done
fi

# ── Control arm ───────────────────────────────────────────────────────────────
echo ""
echo "[ab-video-change-threshold] ── Control: trained state, no --video-change-threshold ──"
SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 \
    CLIPS="$CLIPS" \
    MEMORY_DIR="$PILOT_ROOT/control" \
    CAMPAIGN_DIR="$PILOT_ROOT/control/campaign" \
    SKIP_BUILD=1 \
    bash scripts/train_campaign.sh

# ── Treatment arm ─────────────────────────────────────────────────────────────
echo ""
echo "[ab-video-change-threshold] ── Treatment: trained state + --video-change-threshold $THRESHOLD ──"
SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 \
    CLIPS="$CLIPS" \
    MEMORY_DIR="$PILOT_ROOT/treatment" \
    CAMPAIGN_DIR="$PILOT_ROOT/treatment/campaign" \
    EXTRA_ARGS="--video-change-threshold $THRESHOLD" \
    SKIP_BUILD=1 \
    bash scripts/train_campaign.sh

# ── Comparison: skip rate + headline quality metrics + deviation scan ────────
echo ""
echo "[ab-video-change-threshold] both arms complete — comparison:"
python3 - "$PILOT_ROOT" <<'PYEOF'
import json, statistics, struct, sys

root = sys.argv[1]

def load(arm):
    rows = []
    with open(f"{root}/{arm}/campaign/metrics.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if "cycle" in r and "session" in r:
                rows.append(r)
    return rows

ctrl = load("control")
trt = load("treatment")
if not ctrl or not trt:
    print("missing metrics for at least one arm — aborting comparison")
    sys.exit(1)

def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else None

HEADLINE = [
    "mean_output_mse", "mean_output_mse_wake", "mean_loss", "mean_proj_loss",
    "gist_recall_rate", "within_session_gist_rate", "warm_recall_l2",
    "nrem_wall_ms", "rem_wall_ms",
]
print(f"\n{'metric':34} {'control':>13} {'treatment':>13} {'delta':>9}")
print("-" * 72)
for k in HEADLINE:
    c, t = mean(ctrl, k), mean(trt, k)
    if c is None or t is None:
        continue
    d = (t - c) / max(abs(c), 1e-9) * 100
    print(f"{k:34} {c:13.6f} {t:13.6f} {d:+8.1f}%")

regs = ctrl[-1].get("per_region_mse") or []
for i, _ in enumerate(regs):
    c = statistics.mean([r["per_region_mse"][i] for r in ctrl
                         if r.get("per_region_mse") and i < len(r["per_region_mse"])])
    t = statistics.mean([r["per_region_mse"][i] for r in trt
                         if r.get("per_region_mse") and i < len(r["per_region_mse"])])
    d = (t - c) / max(abs(c), 1e-9) * 100
    print(f"{('per_region_mse.' + str(i)):34} {c:13.6f} {t:13.6f} {d:+8.1f}%")

# ── Video change-gate skip rate (Core tier, always on, but still embedded under
# the per-cycle "fine_metrics" snapshot alongside true Fine-tier counters — the
# registry doesn't distinguish tiers at storage, only at macro-compile-time) ──
def counter_sum(rows, name):
    return sum((r.get("fine_metrics") or {}).get(name, {}).get("count", 0) for r in rows)

print("\n-- video change-gate skip rate (the actual §41 question) --")
for arm, rows in (("control", ctrl), ("treatment", trt)):
    enc = counter_sum(rows, "video_frames_encoded")
    skip = counter_sum(rows, "video_frames_skipped")
    total = enc + skip
    rate = (skip / total * 100) if total else 0.0
    print(f"{arm}: encoded={enc} skipped={skip} total={total} skip_rate={rate:.1f}%")

# ── Deviation scan: every shared numeric channel, plus one-arm-only channels ──
def flat(r):
    out = {}
    for k, v in r.items():
        if isinstance(v, (int, float)) and k not in ("session", "cycle"):
            out[k] = v
    for k, v in (r.get("fine_metrics") or {}).items():
        if isinstance(v, dict) and "count" in v:
            out[f"fm.{k}.count"] = v.get("count", 0)
    return out

agg_c, agg_t = {}, {}
for r in ctrl:
    for k, v in flat(r).items():
        agg_c.setdefault(k, []).append(v)
for r in trt:
    for k, v in flat(r).items():
        agg_t.setdefault(k, []).append(v)

EXPECTED_TREATMENT_ONLY = ()
print("\n-- deviations (|delta| > 30%, or one-arm-only) --")
any_dev = False
for k in sorted(set(agg_c) | set(agg_t)):
    c = statistics.mean(agg_c[k]) if k in agg_c else None
    t = statistics.mean(agg_t[k]) if k in agg_t else None
    if c is None or t is None:
        if k not in EXPECTED_TREATMENT_ONLY:
            print(f"  ONE-ARM ONLY: {k}: control={c} treatment={t}")
            any_dev = True
        continue
    if abs(c) < 1e-9:
        if abs(t) > 1e-6:
            print(f"  {k}: control~0 -> treatment={t:.6g}")
            any_dev = True
        continue
    d = (t - c) / abs(c) * 100
    if abs(d) > 30:
        print(f"  {k}: {c:.6g} -> {t:.6g} ({d:+.0f}%)")
        any_dev = True
if not any_dev:
    print("  none beyond video_frames_encoded/skipped (expected to diverge by design)")

# ── Health / growth ──
print("\n-- health / growth --")
for arm, rows in (("control", ctrl), ("treatment", trt)):
    grew = sum(1 for r in rows if r.get("grew"))
    nan = sum((r.get("fine_metrics") or {}).get("training_nan_detected", {}).get("count", 0)
              for r in rows)
    sizes = struct.unpack("<5Q", open(f"{root}/{arm}/snapshot/region_sizes.bin", "rb").read())
    print(f"{arm}: cycles={len(rows)} grew_events={grew} training_nan={nan} "
          f"final_region_sizes={list(sizes)}")
PYEOF

echo ""
echo "[ab-video-change-threshold] done. Artifacts:"
echo "  Control:   $PILOT_ROOT/control/campaign/metrics.jsonl"
echo "  Treatment: $PILOT_ROOT/treatment/campaign/metrics.jsonl"
