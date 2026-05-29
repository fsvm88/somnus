#!/usr/bin/env bash
# Download ~15 diverse long clips for SOMNUS training using yt-dlp ytsearch.
# Targets lectures, podcasts, nature/urban walks — mix of ASR-rich and ambient.
# Uses ytsearch so yt-dlp resolves URLs internally; avoids stale video-ID rot.
#
# Usage:  scripts/download_long_clips.sh [OUT_DIR]
#   OUT_DIR defaults to clips/long/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-clips/long}"
mkdir -p "$OUT"

# Download best ≤720p mp4. No duration pre-filter (broken without JS runtime).
# We prune short files at the end.
dl() {
    local label="$1" query="$2"
    echo ""
    echo "==> $label"
    yt-dlp \
        -f 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]' \
        --merge-output-format mp4 \
        --no-playlist \
        --max-downloads 1 \
        --output "${OUT}/%(title).70s [%(id)s].%(ext)s" \
        --restrict-filenames \
        --no-overwrites \
        "$query" || echo "  [WARN] $label failed, continuing"
}

# ── Lectures (dense speech, diverse vocabulary) ───────────────────────────────
dl "MIT algorithms lecture" \
    "https://www.youtube.com/watch?v=HtSuA80QTyo"

dl "Lex Fridman + Yann LeCun full podcast" \
    "ytsearch1:Lex Fridman Yann LeCun podcast full interview 2022"

dl "Lex Fridman + Yoshua Bengio full podcast" \
    "ytsearch1:Lex Fridman Yoshua Bengio podcast full interview"

dl "Stanford ML lecture Andrew Ng" \
    "ytsearch1:Stanford machine learning lecture Andrew Ng full"

dl "Sabine Hossenfelder physics lecture long" \
    "ytsearch1:Sabine Hossenfelder physics lecture full talk 2023"

dl "Philosophy lecture consciousness 1h" \
    "ytsearch1:philosophy consciousness lecture full university 2022"

dl "History lecture ancient Rome 1h" \
    "ytsearch1:ancient Rome history lecture university full 2023"

dl "Biology lecture cell biology 1h" \
    "ytsearch1:cell biology lecture university full 2022"

# ── Nature / documentary (narration + ambient visuals) ────────────────────────
dl "Nature documentary narrated 1h" \
    "ytsearch1:nature documentary narrated full episode wildlife 2023 1 hour"

dl "Underwater ocean documentary 1h" \
    "ytsearch1:underwater ocean documentary full episode narrated 2022"

dl "Space astronomy documentary 1h" \
    "ytsearch1:space astronomy documentary full narrated 2023 1 hour"

# ── Urban walk (visual motion, ambient audio) ─────────────────────────────────
dl "Tokyo street walk 1h 4K" \
    "ytsearch1:Tokyo Japan street walk 4K 1 hour 2023"

dl "New York City walk 1h 4K" \
    "ytsearch1:New York City walking tour 4K 1 hour 2023"

dl "Paris street walk 1h 4K" \
    "ytsearch1:Paris France walking tour 4K 1 hour 2023"

dl "London walk 1h 4K" \
    "ytsearch1:London walking tour 4K 1 hour 2023"

# ── Prune anything under 25 minutes ──────────────────────────────────────────
echo ""
echo "==> Pruning clips shorter than 25 min..."
for f in "${OUT}"/*.mp4; do
    [ -f "$f" ] || continue
    dur=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null | cut -d. -f1)
    if [ -z "$dur" ] || [ "$dur" -lt 1500 ]; then
        echo "  DROP (${dur}s): $(basename "$f")"
        rm -f "$f"
    else
        min=$((dur / 60))
        echo "  KEEP (${min} min): $(basename "$f")"
    fi
done

echo ""
echo "==> Final clips in ${OUT}/:"
ls -lh "${OUT}/"
