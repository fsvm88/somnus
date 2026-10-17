#!/usr/bin/env bash
# A/B for gaps.md §49 R5: video encoder on plain CPU EP (--video-encoder-cpu) vs MIGraphX, live pipeline.
# Isolated numbers disagree across conditions (earlier 23 vs 54 ms; 2026-10-17 on an idle GPU: MIGraphX
# 6.1 ms vs CPU 27.1 ms), so this measures what matters: encode_video per call and wake throughput
# with every other stage running. Same binary/snapshot/SEED for both arms, fine-metrics on, SEQUENTIAL
# arms (concurrent arms would contend for the very resources being measured).
# Usage: CYCLES=6 bash scripts/ab_video_encoder_cpu.sh   (SKIP_BUILD=1 SKIP_COPY=1 / ANALYZE_ONLY=1 supported)
set -euo pipefail
export SEED="${SEED:-42}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CYCLES="${CYCLES:-6}"
PILOT_ROOT="${PILOT_ROOT:-memory/ab_video_encoder_cpu}"
SRC_MEMORY="${SRC_MEMORY:-/data/somnus/memory}"
SKIP_BUILD="${SKIP_BUILD:-0}"; SKIP_COPY="${SKIP_COPY:-0}"; ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
TAG="[ab-video-encoder-cpu]"
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
            for d in snapshot warm episodes.cold onnx_models; do
                cp -a --reflink=auto "$SRC_MEMORY/$d" "$PILOT_ROOT/$arm/$d"
            done
        done
    fi
    for arm in control treatment; do
        extra=""; [ "$arm" = "treatment" ] && extra="--video-encoder-cpu"
        echo "$TAG running $arm ($extra)"
        SOMNUS_BIN="$REPO_ROOT/$PILOT_ROOT/bin/somnus" SESSIONS=1 CYCLES="$CYCLES" CAPACITY=8192 CLIPS="$CLIPS" \
            MEMORY_DIR="$PILOT_ROOT/$arm" CAMPAIGN_DIR="$PILOT_ROOT/$arm/campaign" EXTRA_ARGS="$extra" \
            SKIP_BUILD=1 bash scripts/train_campaign.sh > "$PILOT_ROOT/$arm.out" 2>&1
    done
fi
python3 - "$PILOT_ROOT" <<'PYEOF' | tee "$PILOT_ROOT/analysis.txt"
import json, statistics as st, sys
root = sys.argv[1]
def load(a):
    with open(f"{root}/{a}/campaign/metrics.jsonl") as f:
        return [r for r in map(json.loads, f) if "cycle" in r and "session" in r]
D = {a: load(a) for a in ("control", "treatment")}
def tot(rows, name):
    c = sum((r.get("fine_metrics") or {}).get(name, {}).get("count", 0) for r in rows)
    ns = sum((r.get("fine_metrics") or {}).get(name, {}).get("total_ns", 0) for r in rows)
    return c, (ns / c / 1e6 if c else float("nan"))
print(f"{'metric':28} {'control':>14} {'treatment':>14}")
for name in ("encode_video", "encode_env", "encode_speech", "asr_decode", "main_perception_ingest", "ingest_bind"):
    (cc, cm), (tc, tm) = tot(D["control"], name), tot(D["treatment"], name)
    print(f"{name:28} {cm:9.1f} ms x{cc:<5} {tm:9.1f} ms x{tc:<5}")
for k in ("wake_episodes", "perception_total_ms", "video_encode_total_ms", "mean_output_mse_wake", "gist_recall_rate", "sal_threshold"):
    vals = {a: [r[k] for r in D[a] if isinstance(r.get(k), (int, float))] for a in D}
    print(f"{k:28} {st.mean(vals['control']) if vals['control'] else float('nan'):14.4f} {st.mean(vals['treatment']) if vals['treatment'] else float('nan'):14.4f}")
PYEOF
