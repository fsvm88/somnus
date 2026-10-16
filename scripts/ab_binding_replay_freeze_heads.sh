#!/usr/bin/env bash
# A/B: ADR 0041 live-episode binding training (gaps.md §33) — candidate (d),
# freeze the output heads during binding_replay_step_batch's optimizer step.
#
# Motivation (2026-08-16, after both variants of candidate (a) failed with the
# same dosage-invariant mean_output_mse regression, engineering-log.md §44/§45):
# the five output heads are shared, unguarded state. fused_replay_wake_step_batch
# trains them on STORED episodes (the source of the production mean_output_mse
# metric); binding_replay_step_batch trains the SAME weights on LIVE episodes
# from the currently-training binder, in the same NREM batch, through the same
# optimizer. Training one set of weights toward two different input
# distributions is a plausible direct cause of the regression, independent of
# update magnitude — which would explain why scaling the update (candidate a,
# both variants) never helped.
#
# --binding-replay-freeze-heads removes the gradient of the head's own weights
# after backward() but before optimizer.step() (crates/somnus-perception/src/lib.rs)
# — gradient still flows THROUGH the heads into the live episode tensor and on
# into the binding weights (so binding still trains), but the heads' own
# weights are updated only by the fused path. Pinned by a dedicated unit test
# (binding_replay_freeze_heads_trains_binding_not_heads) before this pilot.
#
# Control:   trained production state, no flag (byte-identical NREM)
# Treatment: same state + --binding-replay --binding-replay-freeze-heads
#            (step count and LR scale stay at their defaults — isolates the
#            freeze-heads variable alone, one change per pilot per project
#            discipline)
#
# Both arms start from a copy of the trained campaign state (snapshot + warm +
# cold + onnx), same rationale as ab_binding_replay.sh.
#
# Standing A/B instruction: both arms built and run with fine-metrics; the
# comparison below records deviations, not just the headline verdict.
#
# Usage:
#   bash scripts/ab_binding_replay_freeze_heads.sh
#   CYCLES=10 bash scripts/ab_binding_replay_freeze_heads.sh
#   SKIP_BUILD=1 SKIP_COPY=1 bash scripts/ab_binding_replay_freeze_heads.sh   # re-run/analyze only
set -euo pipefail
export SEED="${SEED:-42}"   # fixed burn RNG seed for both arms (standing reproducibility policy)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CYCLES="${CYCLES:-10}"
PILOT_ROOT="${PILOT_ROOT:-memory/ab_binding_replay_freeze_heads}"
SRC_MEMORY="${SRC_MEMORY:-/data/somnus/memory}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_COPY="${SKIP_COPY:-0}"

CLIPS="$(grep '^CLIPS=' "$SRC_MEMORY/campaign_growth10x/params.txt" | tail -1 | cut -d= -f2-)"

if [ "$SKIP_BUILD" != "1" ]; then
    echo "[ab-binding-replay-freeze-heads] building somnus (release, fine-metrics)…"
    cargo build --release -p somnus --features fine-metrics 2>&1 | tail -2
fi

if [ "$SKIP_COPY" != "1" ]; then
    rm -rf "$PILOT_ROOT"
    for arm in control treatment; do
        echo "[ab-binding-replay-freeze-heads] copying trained state → $PILOT_ROOT/$arm …"
        mkdir -p "$PILOT_ROOT/$arm"
        cp -a --reflink=auto "$SRC_MEMORY/snapshot" "$PILOT_ROOT/$arm/snapshot"
        cp -a --reflink=auto "$SRC_MEMORY/warm" "$PILOT_ROOT/$arm/warm"
        cp -a --reflink=auto "$SRC_MEMORY/episodes.cold" "$PILOT_ROOT/$arm/episodes.cold"
        cp -a --reflink=auto "$SRC_MEMORY/onnx_models" "$PILOT_ROOT/$arm/onnx_models"
    done
fi

echo ""
echo "[ab-binding-replay-freeze-heads] ── Control: trained state, no --binding-replay ──"
SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 \
    CLIPS="$CLIPS" \
    MEMORY_DIR="$PILOT_ROOT/control" \
    CAMPAIGN_DIR="$PILOT_ROOT/control/campaign" \
    SKIP_BUILD=1 \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-binding-replay-freeze-heads] ── Treatment: --binding-replay --binding-replay-freeze-heads ──"
SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 \
    CLIPS="$CLIPS" \
    MEMORY_DIR="$PILOT_ROOT/treatment" \
    CAMPAIGN_DIR="$PILOT_ROOT/treatment/campaign" \
    EXTRA_ARGS="--binding-replay --binding-replay-freeze-heads" \
    SKIP_BUILD=1 \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-binding-replay-freeze-heads] both arms complete — comparison:"
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

# ── ADR 0041 metrics, this pilot's own question: does mean_output_mse recover
# once the heads are excluded from binding-replay's optimizer step? ──
print("\n-- ADR 0041 metrics --")
br = [r.get("fine_metrics", {}).get("nrem_binding_replay") for r in trt]
br = [x for x in br if x and x.get("count")]
if br:
    cnt = sum(x["count"] for x in br)
    ms = sum(x["total_ns"] for x in br) / cnt / 1e6
    print(f"nrem_binding_replay (treatment): {cnt} steps, mean {ms:.2f} ms/step")
else:
    print("nrem_binding_replay: NOT FIRED in treatment — flag not taking effect?")
leak = [r["cycle"] for r in ctrl
        if (r.get("fine_metrics") or {}).get("nrem_binding_replay", {}).get("count")]
print(f"control-arm nrem_binding_replay firings (must be none): {leak or 'none'}")

# ── Deviation scan ──
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

EXPECTED_TREATMENT_ONLY = ("fm.nrem_binding_replay.count", "fm.binding_replay_loss.count")
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
    print("  none beyond the expected treatment-only binding-replay metrics")

# ── Health / growth — §36's salience/settle cascade signal, still worth
# watching even though this candidate targets a different mechanism ──
print("\n-- health / growth (watch for §36's cascade: sal_threshold / settle_gate_dynamic / settle_iters_sum swings) --")
for arm, rows in (("control", ctrl), ("treatment", trt)):
    grew = sum(1 for r in rows if r.get("grew"))
    nan = sum((r.get("fine_metrics") or {}).get("training_nan_detected", {}).get("count", 0)
              for r in rows)
    sizes = struct.unpack("<5Q", open(f"{root}/{arm}/snapshot/region_sizes.bin", "rb").read())
    print(f"{arm}: cycles={len(rows)} grew_events={grew} training_nan={nan} "
          f"final_region_sizes={list(sizes)}")
    for k in ("sal_threshold", "settle_gate_dynamic", "settle_iters_sum", "prior_beta_effective"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            print(f"  {k}: first={vals[0]:.6f} last={vals[-1]:.6f}")
PYEOF

echo ""
echo "[ab-binding-replay-freeze-heads] done. Artifacts:"
echo "  Control:   $PILOT_ROOT/control/campaign/metrics.jsonl"
echo "  Treatment: $PILOT_ROOT/treatment/campaign/metrics.jsonl"
