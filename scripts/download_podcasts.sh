#!/usr/bin/env bash
# Download a diverse set of podcast episodes (different shows/hosts) for SOMNUS
# language/ASR training. Talking-head format → rich speech input for the
# language region, deliberately spread across shows so the corpus isn't
# dominated by one set of voices/topics (see clips/h3h3/ for that).
#
# Usage:  scripts/download_podcasts.sh [OUT_DIR]
#   OUT_DIR defaults to clips/podcasts/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-clips/podcasts}"
mkdir -p "$OUT"

# Per show: search top M_SEARCH candidates, keep up to N_PER that pass the
# duration filter (10 min – MAX_DURATION_S) and aren't already on disk.
N_PER="${N_PER:-10}"
M_SEARCH="${M_SEARCH:-40}"
# 14400s = 4h. Raised from 5400s (90min) so long-form shows (Lex Fridman,
# Huberman Lab, Tim Ferriss) whose typical episode runs 2-4h aren't silently
# filtered out of every search result.
MAX_DURATION_S="${MAX_DURATION_S:-14400}"

dl() {
    local label="$1" query="$2"
    echo ""
    echo "==> $label"
    yt-dlp \
        -f 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]' \
        --merge-output-format mp4 \
        --no-playlist \
        --match-filter "duration > 600 & duration < ${MAX_DURATION_S}" \
        --max-downloads "$N_PER" \
        --output "${OUT}/%(title).70s_[%(id)s].%(ext)s" \
        --restrict-filenames \
        --no-overwrites \
        --ignore-errors \
        "ytsearch${M_SEARCH}:${query}" || echo "  [WARN] $label: yt-dlp reported non-zero exit (may be a benign --max-downloads stop; verified below)"
}

# ── Up to N_PER episodes per show, spread across topics/formats for voice
# diversity. --no-overwrites means already-downloaded episodes are skipped,
# so re-running this script only tops up each show toward N_PER.
dl "Lex Fridman Podcast"        "Lex Fridman Podcast full episode 2023"
dl "Joe Rogan Experience"       "Joe Rogan Experience full episode clip 2023"
dl "Huberman Lab"               "Huberman Lab podcast full episode 2023"
dl "Tim Ferriss Show"           "Tim Ferriss Show full episode interview 2023"
dl "Diary of a CEO"             "Diary of a CEO full episode interview 2023"
dl "Armchair Expert"            "Armchair Expert full episode Dax Shepard 2023"
dl "Conan O'Brien Needs a Friend" "Conan O'Brien Needs a Friend full episode 2023"
dl "This American Life"         "This American Life full episode video 2023"
dl "The Daily"                  "The Daily podcast video episode 2023"
dl "Freakonomics Radio"         "Freakonomics Radio video episode 2023"

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
