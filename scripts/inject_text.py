#!/usr/bin/env python3
"""
Stream text corpus into somnus's language region via stdin.

Each output line becomes one language ingest event. Chunks are formed by
grouping consecutive sentences up to --chunk-chars characters, approximating
the length distribution of ASR transcriptions (~100–250 chars).

Usage:
  python scripts/inject_text.py data/subtitles/*.txt | somnus --language ...
  python scripts/inject_text.py corpus.txt --rate 5 --shuffle

The default rate (5 chunks/sec) gives ~25% language episode density when the
system is also ingesting video at normal GPU speed (~20 video eps/sec).
Tune upward if lang_cos is not rising, downward if visual prior_cosine degrades.
"""

import argparse
import re
import sys
import time
import random
from pathlib import Path


def sentences(text: str):
    """Split text into sentences, skipping blank/junk lines."""
    for raw in re.split(r'(?<=[.!?])\s+|\n{2,}', text):
        s = raw.strip()
        # Drop lines that are just numbers, timestamps, or very short
        if not s or len(s) < 10 or re.match(r'^[\d:,\.\-\s]+$', s):
            continue
        # Strip residual VTT/SRT tags
        s = re.sub(r'<[^>]+>', '', s).strip()
        if s:
            yield s


def make_chunks(text: str, max_chars: int, min_chars: int) -> list[str]:
    chunks, buf, buf_len = [], [], 0
    for sent in sentences(text):
        if buf_len + len(sent) > max_chars and buf:
            chunk = ' '.join(buf)
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            buf, buf_len = [], 0
        buf.append(sent)
        buf_len += len(sent) + 1
    if buf:
        chunk = ' '.join(buf)
        if len(chunk) >= min_chars:
            chunks.append(chunk)
    return chunks


def load_corpus(paths: list[Path], max_chars: int, min_chars: int) -> list[str]:
    all_chunks = []
    for path in paths:
        try:
            text = path.read_text(errors='ignore')
            chunks = make_chunks(text, max_chars, min_chars)
            all_chunks.extend(chunks)
            print(f'# loaded {len(chunks)} chunks from {path}', file=sys.stderr)
        except Exception as e:
            print(f'# warn: {path}: {e}', file=sys.stderr)
    return all_chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('files', nargs='+', type=Path, help='text corpus files')
    ap.add_argument('--rate', type=float, default=5.0,
                    help='chunks per second (default: 5)')
    ap.add_argument('--chunk-chars', type=int, default=200,
                    help='max chars per chunk (default: 200)')
    ap.add_argument('--min-chars', type=int, default=20,
                    help='min chars to emit a chunk (default: 20)')
    ap.add_argument('--shuffle', action='store_true',
                    help='shuffle chunks on each pass through the corpus')
    args = ap.parse_args()

    interval = 1.0 / max(args.rate, 0.001)

    chunks = load_corpus(args.files, args.chunk_chars, args.min_chars)
    if not chunks:
        print('# error: no usable chunks found', file=sys.stderr)
        sys.exit(1)

    print(f'# corpus: {len(chunks)} chunks, rate={args.rate}/s, interval={interval:.3f}s',
          file=sys.stderr)

    pass_num = 0
    while True:
        pass_num += 1
        order = list(range(len(chunks)))
        if args.shuffle:
            random.shuffle(order)
        print(f'# pass {pass_num} ({len(chunks)} chunks)', file=sys.stderr)
        for i in order:
            print(chunks[i], flush=True)
            time.sleep(interval)


if __name__ == '__main__':
    main()
