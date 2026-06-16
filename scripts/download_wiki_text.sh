#!/usr/bin/env bash
# Download Simple English Wikipedia dump and extract to plain text.
# Output: data/wiki_text/ — a tree of .txt files ready for TEXT_CORPUS.
#
# Usage:  scripts/download_wiki_text.sh [OUT_DIR]
#   OUT_DIR defaults to data/wiki_text/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-data/wiki_text}"
DUMP_DIR="data/wiki_dump"
DUMP_FILE="${DUMP_DIR}/simplewiki-latest-pages-articles.xml.bz2"
DUMP_URL="https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"

mkdir -p "$DUMP_DIR" "$OUT"

# ── Download ──────────────────────────────────────────────────────────────────
if [ -f "$DUMP_FILE" ]; then
    echo "==> Dump already present, skipping download."
else
    echo "==> Downloading Simple English Wikipedia dump (~250 MB)…"
    curl -L --progress-bar -o "$DUMP_FILE" "$DUMP_URL"
fi

# ── Extract to plain text ─────────────────────────────────────────────────────
echo ""
echo "==> Extracting to plain text (this takes a few minutes)…"
python3 "$REPO_ROOT/scripts/extract_wiki.py" "$DUMP_FILE" "$OUT"

TOTAL=$(find "$OUT" -name "*.txt" | wc -l)
SIZE=$(du -sh "$OUT" | cut -f1)
echo ""
echo "==> Done: ${TOTAL} files, ${SIZE} in ${OUT}/"
echo "    Use with: TEXT_CORPUS=\"${OUT}/*.txt\""
