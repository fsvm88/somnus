#!/usr/bin/env bash
# Download auto-generated English subtitles for all clips in clips/long/,
# then convert the VTT files to clean plain text in data/subtitles/.
#
# Requires: yt-dlp  (pip install yt-dlp)
#
# Usage:
#   scripts/download_subtitles.sh
#   scripts/download_subtitles.sh clips/long/specific_clip_[ID].mp4 ...
#
# Output: data/subtitles/<ID>.txt  — one cleaned sentence per line
# These files can be fed directly to scripts/inject_text.py.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/subtitles

# Accept explicit clip paths as args, or default to all long clips.
if [ $# -gt 0 ]; then
    CLIPS=("$@")
else
    CLIPS=(clips/long/*.mp4)
fi

for clip in "${CLIPS[@]}"; do
    # Extract YouTube ID from filename: title_[ID].mp4
    ID=$(basename "$clip" | grep -oP '\[[A-Za-z0-9_-]+\]' | tail -1 | tr -d '[]')
    if [ -z "$ID" ]; then
        echo "warn: cannot extract YouTube ID from '$clip'" >&2
        continue
    fi

    OUTFILE="data/subtitles/${ID}.txt"
    if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
        echo "skip: $ID ($(wc -l < "$OUTFILE") lines already)"
        continue
    fi

    echo "downloading: $ID  ($clip)"
    TMPDIR_SUB=$(mktemp -d)
    trap 'rm -rf "$TMPDIR_SUB"' EXIT

    yt-dlp \
        --write-auto-sub \
        --sub-langs "en.*" \
        --sub-format vtt \
        --skip-download \
        --output "$TMPDIR_SUB/%(id)s" \
        "https://www.youtube.com/watch?v=${ID}" 2>&1 | grep -v "^\[debug\]" || {
        echo "warn: yt-dlp failed for $ID" >&2
        rm -rf "$TMPDIR_SUB"
        trap - EXIT
        continue
    }

    VTT=$(ls "$TMPDIR_SUB"/*.vtt 2>/dev/null | head -1)
    if [ -z "$VTT" ]; then
        echo "warn: no VTT file produced for $ID" >&2
        rm -rf "$TMPDIR_SUB"
        trap - EXIT
        continue
    fi

    # Convert VTT to plain text: strip headers, timestamps, tags, deduplicate.
    python3 - "$VTT" "$OUTFILE" <<'PYEOF'
import sys, re

vtt_path, out_path = sys.argv[1], sys.argv[2]
lines = open(vtt_path, encoding='utf-8', errors='ignore').readlines()

seen, out = set(), []
for line in lines:
    line = line.strip()
    # Skip blank, WEBVTT header, cue ID numbers, and timestamp lines
    if (not line
            or line.startswith('WEBVTT')
            or line.startswith('NOTE')
            or re.match(r'^\d+$', line)
            or re.match(r'^\d{2}:\d{2}', line)
            or '-->' in line):
        continue
    # Strip VTT inline tags and HTML entities
    line = re.sub(r'<[^>]+>', '', line)
    line = re.sub(r'&\w+;', ' ', line).strip()
    if not line or line in seen:
        continue
    seen.add(line)
    out.append(line)

with open(out_path, 'w') as f:
    f.write('\n'.join(out) + '\n')
print(f'  → {out_path} ({len(out)} lines)')
PYEOF

    rm -rf "$TMPDIR_SUB"
    trap - EXIT
done

echo ""
echo "Done. Subtitle files in data/subtitles/:"
ls -lh data/subtitles/*.txt 2>/dev/null || echo "  (none)"
