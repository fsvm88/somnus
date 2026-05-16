#!/usr/bin/env bash
# Novel-content calibration experiment.
# For each clip, restores baseline memory, runs the binary, captures prior_cosine
# and settle delta statistics. Results printed to stdout; raw logs saved per clip.
set -euo pipefail

BINARY=./target/release/somnus
BASELINE=memory_baseline_backup
MEMORY=memory
MODELS=models
CLIPS=("test.mp4:familiar" "clips/talking_head.mp4:talking_head" "clips/nature.mp4:nature" "clips/animation.mp4:animation" "clips/urban.mp4:urban")
LOG_DIR=experiment_logs
mkdir -p "$LOG_DIR"

summarise_log() {
    local tag=$1
    local log=$2
    # Strip ANSI escape codes once for all parsing below
    local clean
    clean=$(sed 's/\x1b\[[0-9;]*m//g' "$log")

    echo "=== $tag ==="

    # prior_cosine per region
    for region in speech env visual symbolic; do
        vals=$(echo "$clean" | grep "prior_cosine" | grep "region=\"$region\"" | grep -oP 'prior_cosine=\K[0-9.eE+-]+' || true)
        if [ -n "$vals" ]; then
            n=$(echo "$vals" | wc -l)
            mean=$(echo "$vals" | awk '{s+=$1;n++} END{printf "%.4f", s/n}')
            min=$(echo "$vals" | awk 'BEGIN{m=1e9}{if($1<m)m=$1}END{printf "%.4f",m}')
            max=$(echo "$vals" | awk 'BEGIN{m=-1e9}{if($1>m)m=$1}END{printf "%.4f",m}')
            echo "  $region: n=$n mean=$mean min=$min max=$max"
        else
            echo "  $region: (no samples)"
        fi
    done

    # settle delta: count how many episodes converge at each iter (delta < SETTLE_EPS=1.0)
    total_eps=$(echo "$clean" | grep "prior_cosine" | grep 'region="visual"' | wc -l || echo 0)
    echo "  settle convergence (out of ~$total_eps visual episodes):"
    for iter in 0 1 2; do
        ct=$(echo "$clean" | grep "settle delta" | grep "iter=$iter" | grep -oP 'delta=\K[0-9.eE+-]+' | awk '$1<1.0{c++} END{printf "%d", c+0}')
        echo "    converged at iter=$iter: $ct"
    done
    not_conv=$(echo "$clean" | grep "settle delta" | grep "iter=2" | grep -oP 'delta=\K[0-9.eE+-]+' | awk '$1>=1.0{c++} END{printf "%d", c+0}')
    echo "    not converged after 3 steps: $not_conv"
    echo ""
}

for entry in "${CLIPS[@]}"; do
    clip="${entry%%:*}"
    tag="${entry##*:}"

    echo "[$(date +%H:%M:%S)] Running: $tag ($clip)"

    # Restore baseline memory
    rm -rf "$MEMORY"
    cp -r "$BASELINE" "$MEMORY"

    log="$LOG_DIR/${tag}.log"

    # Run binary; file-source exits naturally on EOF
    RUST_LOG=info "$BINARY" \
        --file "$clip" \
        --models-dir "$MODELS" \
        --prior-beta 0.1 \
        --settle-steps 3 \
        --settle-beta 0.5 \
        2>"$log" || true

    echo "[$(date +%H:%M:%S)] Done: $tag"
done

echo ""
echo "========================= RESULTS ========================="
for entry in "${CLIPS[@]}"; do
    tag="${entry##*:}"
    summarise_log "$tag" "$LOG_DIR/${tag}.log"
done
