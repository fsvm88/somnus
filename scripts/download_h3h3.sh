#!/usr/bin/env bash
# Download a handful of H3 Podcast episodes for SOMNUS language/ASR training.
# Talking-head format → rich speech input for the language region.
#
# Usage:  scripts/download_h3h3.sh [OUT_DIR]
#   OUT_DIR defaults to clips/h3h3/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-clips/h3h3}"
mkdir -p "$OUT"

dl() {
    local label="$1" query="$2"
    echo ""
    echo "==> $label"
    yt-dlp \
        -f 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]' \
        --merge-output-format mp4 \
        --no-playlist \
        --max-downloads 1 \
        --output "${OUT}/%(title).70s_[%(id)s].%(ext)s" \
        --restrict-filenames \
        --no-overwrites \
        "$query" || echo "  [WARN] $label failed, continuing"
}

dl "H3 Podcast ep 1" "ytsearch1:H3 Podcast full episode 2023 h3h3productions"
dl "H3 Podcast ep 2" "ytsearch1:H3 Podcast full episode 2022 h3h3productions ethan"
dl "H3 Podcast ep 3" "ytsearch1:H3 Podcast full episode 2024 h3h3productions hila"
dl "H3 Podcast ep 4" "ytsearch1:H3 Podcast celebrity guest interview full 2023"
dl "H3 Podcast ep 5" "ytsearch1:H3 Podcast full episode comedy talk show 2023"

echo ""
echo "==> Verifying clips…"
for f in "${OUT}"/*.mp4; do
    [ -f "$f" ] || continue
    dur=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null | cut -d. -f1)
    min=$((dur / 60))
    echo "  ${min} min: $(basename "$f")"
done

echo ""
echo "==> Done. Output: ${OUT}/"
