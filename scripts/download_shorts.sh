#!/usr/bin/env bash
# Download ~200 short diverse clips (10–180 s) for SOMNUS training.
# Targets 20 content categories for maximum sensory + semantic diversity.
# Uses yt-dlp ytsearch with a duration filter; stores in clips/shorts/.
#
# Usage:  scripts/download_shorts.sh [OUT_DIR]
#   OUT_DIR defaults to clips/shorts/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-clips/shorts}"
mkdir -p "$OUT"

# Per category: search top M_SEARCH candidates, keep up to N_PER that pass
# the duration filter (10–180 s).
N_PER=10
M_SEARCH=40

dl_cat() {
    local label="$1" query="$2"
    echo ""
    echo "==> [$label]"
    yt-dlp \
        -f 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]/best' \
        --merge-output-format mp4 \
        --no-playlist \
        --match-filter "duration > 10 & duration < 180" \
        --max-downloads "$N_PER" \
        --output "${OUT}/%(title).60s_[%(id)s].%(ext)s" \
        --restrict-filenames \
        --no-overwrites \
        --ignore-errors \
        "ytsearch${M_SEARCH}:${query}" \
    || echo "  [WARN] $label partial/failed, continuing"
}

# ── 20 categories ─────────────────────────────────────────────────────────────
# Cooking & food
dl_cat "cooking" \
    "quick cooking recipe tutorial hack kitchen"

# Animals & wildlife
dl_cat "animals" \
    "amazing animal wildlife nature behavior clip"

# Sports & fitness
dl_cat "sports" \
    "sports highlight athletic feat fitness exercise"

# Science experiments
dl_cat "science" \
    "science experiment demonstration physics chemistry"

# Comedy & sketches
dl_cat "comedy" \
    "funny comedy skit humor stand-up clip"

# Travel & cityscapes
dl_cat "travel" \
    "travel destination city street scene tour"

# Music performance
dl_cat "music" \
    "live music instrument performance concert clip"

# Technology & gadgets
dl_cat "technology" \
    "technology gadget demo product review unboxing"

# Art & illustration
dl_cat "art" \
    "speed art drawing painting illustration timelapse"

# Quick education
dl_cat "education" \
    "educational explainer fact science history quick"

# History & culture
dl_cat "history" \
    "history culture tradition fact explainer clip"

# Gaming
dl_cat "gaming" \
    "gaming highlight gameplay clip esports moment"

# DIY & crafts
dl_cat "diy" \
    "DIY craft tutorial make build how-to project"

# Health & wellness
dl_cat "health" \
    "health wellness tip nutrition exercise mindfulness"

# Space & astronomy
dl_cat "space" \
    "space astronomy NASA planet star universe clip"

# Language & linguistics
dl_cat "language" \
    "language word etymology meaning linguistics fact"

# Weather & nature phenomena
dl_cat "weather" \
    "storm weather lightning tornado nature phenomenon"

# Architecture & design
dl_cat "architecture" \
    "architecture building design structure interior"

# Cars & motorsports
dl_cat "cars" \
    "car race motorsport drift supercar driving clip"

# Dance & choreography
dl_cat "dance" \
    "dance choreography performance hip-hop ballet"

# ── Verify and prune anything that slipped through the filter ─────────────────
echo ""
echo "==> Verifying downloaded clips…"
KEPT=0
DROPPED=0
for f in "${OUT}"/*.mp4; do
    [ -f "$f" ] || continue
    dur=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null | cut -d. -f1)
    if [ -z "$dur" ] || [ "$dur" -lt 10 ]; then
        echo "  DROP (${dur}s): $(basename "$f")"
        rm -f "$f"
        DROPPED=$((DROPPED + 1))
    else
        KEPT=$((KEPT + 1))
    fi
done

echo ""
echo "==> Done: ${KEPT} clips kept, ${DROPPED} dropped."
echo "==> Output: ${OUT}/"
ls -lh "${OUT}/" | tail -10
