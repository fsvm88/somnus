#!/usr/bin/env bash
# Transcribe video/audio clips to plain text using openai-whisper (CPU).
# Output: data/subtitles/<stem>.txt  — one file per clip.
# Already-transcribed clips are skipped.
#
# Usage:
#   scripts/transcribe_clips.sh                         # all clips/long/*.mp4
#   scripts/transcribe_clips.sh clips/long/Sabine*.mp4  # specific clips
#
# Requirements:
#   pip install openai-whisper
#
# Model choice (MODEL env var, default: base):
#   tiny  — ~15× real-time on CPU, lower accuracy
#   base  — ~7× real-time on CPU, good accuracy      ← default
#   small — ~3× real-time on CPU, better accuracy
#
# Time estimates at base on a modern CPU:
#   37-min clip  → ~5 min   |   165-min clip → ~24 min
#   Total 15 clips (~1600 min audio) → ~3-4 hours

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-base}"
OUT_DIR="data/subtitles"
mkdir -p "$OUT_DIR"

# Check whisper is available
if ! python3 -c "import whisper" 2>/dev/null; then
    echo "error: openai-whisper not installed. Run: pip install openai-whisper" >&2
    exit 1
fi

if [ $# -gt 0 ]; then
    CLIPS=("$@")
else
    CLIPS=(clips/long/*.mp4)
fi

TOTAL=${#CLIPS[@]}
DONE=0
SKIPPED=0

echo "[transcribe] model=$MODEL  output=$OUT_DIR  clips=$TOTAL"
echo ""

for clip in "${CLIPS[@]}"; do
    STEM=$(basename "$clip" .mp4)
    OUTFILE="$OUT_DIR/${STEM}.txt"

    if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
        echo "[skip] $STEM"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    DURATION=$(python3 -c "
import subprocess, json, sys
r = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                   '-of','json','$clip'], capture_output=True, text=True)
d = json.loads(r.stdout).get('format',{}).get('duration','?')
try:
    m,s = divmod(float(d), 60)
    print(f'{int(m)}m{int(s)}s')
except:
    print(d)
" 2>/dev/null || echo "?")

    echo "[transcribe] $STEM  ($DURATION)"
    START=$(date +%s)

    python3 - "$clip" "$MODEL" "$OUT_DIR" <<'PYEOF'
import sys, whisper, os

clip, model_name, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
stem = os.path.splitext(os.path.basename(clip))[0]
out_path = os.path.join(out_dir, stem + '.txt')

model = whisper.load_model(model_name)
result = model.transcribe(clip, language='en', fp16=False, verbose=False)

# Write one segment per line (natural sentence-like chunks from Whisper)
lines = []
for seg in result['segments']:
    text = seg['text'].strip()
    if text:
        lines.append(text)

with open(out_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f'  → {out_path} ({len(lines)} segments, {len(" ".join(lines).split())} words)')
PYEOF

    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "  completed in ${ELAPSED}s"
    DONE=$((DONE + 1))
done

echo ""
echo "[transcribe] done: $DONE transcribed, $SKIPPED skipped"
echo ""
ls -lh "$OUT_DIR"/*.txt 2>/dev/null | awk '{print "  "$5, $NF}' || true
