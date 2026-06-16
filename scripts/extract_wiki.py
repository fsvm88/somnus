#!/usr/bin/env python3
"""
Extract plain text from a Wikipedia XML bz2 dump.
Streams the file without loading it fully into memory.

Usage:
    python3 scripts/extract_wiki.py simplewiki.xml.bz2 data/wiki_text/
"""

import bz2
import re
import sys
import os
from pathlib import Path
from xml.etree.ElementTree import iterparse

CHUNK_BYTES = 1 << 20  # 1 MB per output file


def strip_wikitext(text: str) -> str:
    # Remove <ref>...</ref> and self-closing <ref ... />
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove all other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove {{...}} templates (nested up to 4 levels)
    for _ in range(4):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    # Remove {| ... |} wikitables
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # Remove any leftover wikitable row/header/cell markup lines (|, |-, !, ||)
    text = re.sub(r'^\s*[|!].*$', '', text, flags=re.MULTILINE)
    # [[File:...]] and [[Image:...]] — drop entirely
    text = re.sub(r'\[\[(?:File|Image|Media):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    # [[link|display]] → display text; [[link]] → link text
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # External links [url text] → text; bare [url] → drop
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    # Remove wikitext formatting (bold/italic)
    text = re.sub(r"'{2,}", '', text)
    # Remove section headers (== Heading ==)
    text = re.sub(r'=+[^=\n]+=+', '', text)
    # Remove category / interlanguage links
    text = re.sub(r'\[\[(?:Category|[a-z\-]{2,10}):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract(dump_path: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_idx = 0
    buf_parts = []
    buf_size = 0

    def flush():
        nonlocal file_idx, buf_parts, buf_size
        if not buf_parts:
            return
        out_path = out_dir / f"wiki_{file_idx:05d}.txt"
        out_path.write_text('\n\n'.join(buf_parts), encoding='utf-8')
        file_idx += 1
        buf_parts = []
        buf_size = 0

    ns_tag = ''
    title = ''
    in_ns0 = False
    articles = 0

    print(f"Extracting {dump_path} → {out_dir}/", flush=True)

    with bz2.open(dump_path, 'rb') as f:
        for event, elem in iterparse(f, events=('start', 'end')):
            # Strip namespace prefix from tag
            tag = elem.tag.split('}', 1)[-1] if '}' in elem.tag else elem.tag

            if event == 'start':
                if tag == 'page':
                    title = ''
                    in_ns0 = False
                continue

            # end events
            if tag == 'title':
                title = elem.text or ''
            elif tag == 'ns':
                in_ns0 = (elem.text == '0')
            elif tag == 'text' and in_ns0:
                raw = elem.text or ''
                # Skip redirects and stubs
                if raw.lower().startswith('#redirect') or len(raw) < 200:
                    elem.clear()
                    continue
                plain = strip_wikitext(raw)
                if len(plain) < 100:
                    elem.clear()
                    continue
                chunk = f"# {title}\n\n{plain}"
                buf_parts.append(chunk)
                buf_size += len(chunk)
                articles += 1
                if buf_size >= CHUNK_BYTES:
                    flush()
                if articles % 5000 == 0:
                    print(f"  {articles} articles…", flush=True)
            elif tag == 'page':
                elem.clear()

    flush()
    print(f"Done: {articles} articles → {file_idx} files in {out_dir}/")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dump.xml.bz2> <out_dir>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
