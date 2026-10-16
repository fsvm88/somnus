#!/usr/bin/env bash
# A/B: ADR 0041 live-episode binding training (gaps.md §33) — fourth pilot, candidate
# (a) re-run after §35 (unbounded, FAIL — episode drift), §36 (--binding-replay-steps 1,
# FAIL — worse instability via a salience/settle cascade), and §44 (loss-scale variant of
# this same candidate, invalidated — see below).
#
# Candidate (a): reduce step *magnitude* instead of step *count*, on the theory that a
# full-strength gradient update through the live binding path is what's driving the
# same-cycle MSE blowup, not how many times it fires per cycle.
#
# §44 correction (2026-08-16): the original version of this pilot scaled the *loss*
# tensor before backward(). `binding_replay_step_batch` steps an AdamW optimizer, which
# normalizes each parameter update by its running second-moment estimate — so scaling the
# loss cancels out almost entirely before it reaches the parameters. §44's result
# (aggregate mean_output_mse +440.6%) was statistically the same as §36's unscaled
# aggregate (+445%), consistent with the loss-scale pilot having been a no-op, not
# evidence that dosage doesn't matter. `--binding-replay-lr-scale` now scales the
# *learning rate* passed to optimizer.step() directly, which actually changes the
# update magnitude — this is the real dosage test candidate (a) was meant to be.
#
# Control:   trained production state, no flag (byte-identical NREM)
# Treatment: same state + --binding-replay --binding-replay-lr-scale <SCALE>
#            (step count stays at the default cap of 1, per §36's finding that
#            capping alone isn't the answer — this isolates the scale variable)
#
# Both arms start from a copy of the trained campaign state (snapshot + warm +
# cold + onnx), same rationale as ab_binding_replay.sh.
#
# Standing A/B instruction: both arms built and run with fine-metrics; the
# comparison below records deviations, not just the headline verdict.
#
# Usage:
#   bash scripts/ab_binding_replay_lr_scale.sh
#   SCALE=0.1 CYCLES=10 bash scripts/ab_binding_replay_lr_scale.sh
#   SKIP_BUILD=1 SKIP_COPY=1 bash scripts/ab_binding_replay_lr_scale.sh   # re-run/analyze only
set -euo pipefail
export SEED="${SEED:-42}"   # fixed burn RNG seed for both arms (standing reproducibility policy)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CYCLES="${CYCLES:-10}"
SCALE="${SCALE:-0.1}"
PILOT_ROOT="${PILOT_ROOT:-memory/ab_binding_replay_lr_scale}"
SRC_MEMORY="${SRC_MEMORY:-/data/somnus/memory}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_COPY="${SKIP_COPY:-0}"

CLIPS="$(grep '^CLIPS=' "$SRC_MEMORY/campaign_growth10x/params.txt" | tail -1 | cut -d= -f2-)"

if [ "$SKIP_BUILD" != "1" ]; then
    echo "[ab-binding-replay-lr-scale] building somnus (release, fine-metrics)…"
    cargo build --release -p somnus --features fine-metrics 2>&1 | tail -2
fi

if [ "$SKIP_COPY" != "1" ]; then
    rm -rf "$PILOT_ROOT"
    for arm in control treatment; do
        echo "[ab-binding-replay-lr-scale] copying trained state → $PILOT_ROOT/$arm …"
        mkdir -p "$PILOT_ROOT/$arm"
        cp -a --reflink=auto "$SRC_MEMORY/snapshot" "$PILOT_ROOT/$arm/snapshot"
        cp -a --reflink=auto "$SRC_MEMORY/warm" "$PILOT_ROOT/$arm/warm"
        cp -a --reflink=auto "$SRC_MEMORY/episodes.cold" "$PILOT_ROOT/$arm/episodes.cold"
        cp -a --reflink=auto "$SRC_MEMORY/onnx_models" "$PILOT_ROOT/$arm/onnx_models"
    done
fi

echo ""
echo "[ab-binding-replay-lr-scale] ── Control: trained state, no --binding-replay ──"
SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 \
    CLIPS="$CLIPS" \
    MEMORY_DIR="$PILOT_ROOT/control" \
    CAMPAIGN_DIR="$PILOT_ROOT/control/campaign" \
    SKIP_BUILD=1 \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-binding-replay-lr-scale] ── Treatment: --binding-replay --binding-replay-lr-scale $SCALE ──"
SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 \
    CLIPS="$CLIPS" \
    MEMORY_DIR="$PILOT_ROOT/treatment" \
    CAMPAIGN_DIR="$PILOT_ROOT/treatment/campaign" \
    EXTRA_ARGS="--binding-replay --binding-replay-lr-scale $SCALE" \
    SKIP_BUILD=1 \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-binding-replay-lr-scale] both arms complete — comparison:"
python3 - "$PILOT_ROOT" "$SCALE" <<'PYEOF'
import json, statistics, struct, sys

root = sys.argv[1]
scale = sys.argv[2]

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

# ── ADR 0041 metrics, this pilot's own question: same-cycle MSE blowup (§36's
# failure mode) and whether it's reduced at this lr scale ──
print("\n-- ADR 0041 metrics --")
br = [r.get("fine_metrics", {}).get("nrem_binding_replay") for r in trt]
br = [x for x in br if x and x.get("count")]
if br:
    cnt = sum(x["count"] for x in br)
    ms = sum(x["total_ns"] for x in br) / cnt / 1e6
    print(f"nrem_binding_replay (treatment): {cnt} steps, mean {ms:.2f} ms/step")
else:
    print("nrem_binding_replay: NOT FIRED in treatment — flag not taking effect?")
per_cycle = []
for r in trt:
    fm = r.get("fine_metrics") or {}
    x = fm.get("binding_replay_loss")
    steps = (fm.get("nrem_binding_replay") or {}).get("count", 0)
    if x and x.get("count") and steps:
        per_cycle.append((r["cycle"], x["count"] / steps / 1e6))
if per_cycle:
    print(f"binding_replay_loss per-cycle mean (treatment, unscaled — lr_scale={scale}):")
    for cyc, v in per_cycle:
        print(f"  cycle {cyc}: {v:.6f}")
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

# ── Health / growth — §36's salience/settle cascade signal ──
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
echo "[ab-binding-replay-lr-scale] done. Artifacts:"
echo "  Control:   $PILOT_ROOT/control/campaign/metrics.jsonl"
echo "  Treatment: $PILOT_ROOT/treatment/campaign/metrics.jsonl"
