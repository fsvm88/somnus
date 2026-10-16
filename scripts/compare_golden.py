#!/usr/bin/env python3
"""Compare two golden_dump JSON outputs (burn 0.21 reference vs 0.22) within a tolerance.

Usage: scripts/compare_golden.py reference.json candidate.json [--rtol 1e-3] [--atol 1e-5]
Both come from `cargo run -p somnus-perception --release --example golden_dump -- <weights.safetensors>`.
"""
import json
import sys


def walk(a, b, path, rtol, atol, bad):
    if isinstance(a, dict):
        for k in a:
            walk(a[k], b.get(k) if isinstance(b, dict) else None, f"{path}.{k}", rtol, atol, bad)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            bad.append((path, "length", len(a), None if b is None else len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", rtol, atol, bad)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not isinstance(b, (int, float)) or abs(a - b) > atol + rtol * abs(a):
            bad.append((path, a, b))
    elif a != b:
        bad.append((path, a, b))


def main():
    ref, cand = (json.load(open(p)) for p in sys.argv[1:3])
    rtol = float(sys.argv[sys.argv.index("--rtol") + 1]) if "--rtol" in sys.argv else 1e-3
    atol = float(sys.argv[sys.argv.index("--atol") + 1]) if "--atol" in sys.argv else 1e-5
    bad = []
    walk(ref, cand, "", rtol, atol, bad)
    for b in bad[:40]:
        print("DIFF", *b)
    print(f"{len(bad)} differences (rtol={rtol}, atol={atol})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
