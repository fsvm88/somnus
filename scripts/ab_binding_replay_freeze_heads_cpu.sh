#!/usr/bin/env bash
# Confirming diagnostic for gaps.md §33's leading hypothesis (engineering-log.md
# §46, 2026-08-16): the fifth pilot (candidate d, freeze-heads) found the
# mean_output_mse regression moves even with NO gradient path from
# binding-replay into the heads, and traced the likely real channel to
# rebuild_ort_session firing after every sleep cycle (crates/somnus/src/lib.rs
# :2451-2469): binding weights drift -> ORT re-export picks them up -> next
# wake phase forms episodes through the new binder -> those get stored ->
# later cycles train heads against these drifted stored episodes.
#
# That reading was never isolated from a simpler one: binding-weight drift
# reaching stored episodes at all, regardless of which backend (ORT or burn)
# serves wake-phase binding inference. This script tests exactly that split.
#
# Identical to ab_binding_replay_freeze_heads.sh (candidate d) except
# --force-cpu-inference is added to BOTH arms. --force-cpu-inference (existing
# flag, crates/somnus/src/lib.rs:1046) skips resolving onnx_dir entirely, so
# rebuild_ort_session never runs and binding inference stays on burn for the
# whole run -- the ORT re-export channel is structurally absent, not just
# unlikely to fire. No code changes; one variable changed vs. the §46 pilot,
# per project discipline.
#
# Reading the result against §46 (ORT active, +586.9% mean_output_mse):
#   - delta collapses toward ~0%   -> confirms an ORT-export-specific artifact
#     (precision/quantization mismatch in the exported graph), not general
#     weight-drift. Root cause becomes an ORT export bug, not (b)/(c)'s domain.
#   - delta stays in the ~445-587% band -> rules out ORT specifically; confirms
#     the mechanism is backend-independent live-weight drift into wake-phase
#     episode formation and storage. Unblocks (b)/(c) on the real mechanism.
#
# Standing A/B instruction: both arms built and run with fine-metrics; the
# comparison below records deviations, not just the headline verdict.
#
# Usage:
#   bash scripts/ab_binding_replay_freeze_heads_cpu.sh
#   CYCLES=10 bash scripts/ab_binding_replay_freeze_heads_cpu.sh
#   SKIP_BUILD=1 SKIP_COPY=1 bash scripts/ab_binding_replay_freeze_heads_cpu.sh   # re-run/analyze only
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CYCLES="${CYCLES:-10}"
PILOT_ROOT="${PILOT_ROOT:-memory/ab_binding_replay_freeze_heads_cpu}"
SRC_MEMORY="${SRC_MEMORY:-/data/somnus/memory}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_COPY="${SKIP_COPY:-0}"

CLIPS="$(grep '^CLIPS=' "$SRC_MEMORY/campaign_growth10x/params.txt" | tail -1 | cut -d= -f2-)"

if [ "$SKIP_BUILD" != "1" ]; then
    echo "[ab-binding-replay-freeze-heads-cpu] building somnus (release, fine-metrics)…"
    cargo build --release -p somnus --features fine-metrics 2>&1 | tail -2
fi

if [ "$SKIP_COPY" != "1" ]; then
    rm -rf "$PILOT_ROOT"
    for arm in control treatment; do
        echo "[ab-binding-replay-freeze-heads-cpu] copying trained state → $PILOT_ROOT/$arm …"
        mkdir -p "$PILOT_ROOT/$arm"
        cp -a --reflink=auto "$SRC_MEMORY/snapshot" "$PILOT_ROOT/$arm/snapshot"
        cp -a --reflink=auto "$SRC_MEMORY/warm" "$PILOT_ROOT/$arm/warm"
        cp -a --reflink=auto "$SRC_MEMORY/episodes.cold" "$PILOT_ROOT/$arm/episodes.cold"
        cp -a --reflink=auto "$SRC_MEMORY/onnx_models" "$PILOT_ROOT/$arm/onnx_models"
    done
fi

echo ""
echo "[ab-binding-replay-freeze-heads-cpu] ── Control: --force-cpu-inference, no --binding-replay ──"
SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 \
    CLIPS="$CLIPS" \
    MEMORY_DIR="$PILOT_ROOT/control" \
    CAMPAIGN_DIR="$PILOT_ROOT/control/campaign" \
    EXTRA_ARGS="--force-cpu-inference" \
    SKIP_BUILD=1 \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-binding-replay-freeze-heads-cpu] ── Treatment: --force-cpu-inference --binding-replay --binding-replay-freeze-heads ──"
SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 \
    CLIPS="$CLIPS" \
    MEMORY_DIR="$PILOT_ROOT/treatment" \
    CAMPAIGN_DIR="$PILOT_ROOT/treatment/campaign" \
    EXTRA_ARGS="--force-cpu-inference --binding-replay --binding-replay-freeze-heads" \
    SKIP_BUILD=1 \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-binding-replay-freeze-heads-cpu] both arms complete — comparison:"
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

# ── This pilot's own question: with ORT structurally out of the loop
# (--force-cpu-inference on both arms), does mean_output_mse's regression
# collapse relative to §46's ORT-active +586.9%, or stay in the same band? ──
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

ort_c = sum((r.get("fine_metrics") or {}).get("ort_binding_forward", {}).get("count", 0) for r in ctrl)
ort_t = sum((r.get("fine_metrics") or {}).get("ort_binding_forward", {}).get("count", 0) for r in trt)
print(f"ort_binding_forward firings (must be 0 in both arms under --force-cpu-inference): control={ort_c} treatment={ort_t}")

c_mse = mean(ctrl, "mean_output_mse")
t_mse = mean(trt, "mean_output_mse")
if c_mse is not None and t_mse is not None:
    d = (t_mse - c_mse) / max(abs(c_mse), 1e-9) * 100
    print(f"\nmean_output_mse delta under forced-CPU inference: {d:+.1f}% (§46 ORT-active reference: +586.9%)")
    if abs(d) < 100:
        print("-> delta collapsed relative to §46: consistent with an ORT-export-specific artifact.")
    else:
        print("-> delta stayed in the same order of magnitude as §46: consistent with backend-independent weight-drift-into-stored-episodes.")

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
echo "[ab-binding-replay-freeze-heads-cpu] done. Artifacts:"
echo "  Control:   $PILOT_ROOT/control/campaign/metrics.jsonl"
echo "  Treatment: $PILOT_ROOT/treatment/campaign/metrics.jsonl"
