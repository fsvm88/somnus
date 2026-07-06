#!/usr/bin/env python3
"""Effective-rank check on somnus's own region/binding weight matrices.

Loads the raw f32 export from `svd_rank_check` (somnus-perception bin), runs SVD on
each weight matrix, and reports the rank needed to capture 90/95/99% of the
matrix's energy (sum of squared singular values). A low effective rank relative
to min(shape) means a LoRA-style low-rank factorization would reconstruct the
matrix almost exactly at a fraction of the parameter count.

Usage: scripts/.venv/bin/python3 scripts/svd_rank_check.py <export_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np


def effective_rank(shape, s, thresholds=(0.90, 0.95, 0.99)):
    energy = s**2
    total = energy.sum()
    if total <= 0:
        return {f"r{int(t * 100)}": 0 for t in thresholds}
    cum = np.cumsum(energy) / total
    out = {}
    for t in thresholds:
        idx = int(np.searchsorted(cum, t) + 1)
        out[f"r{int(t * 100)}"] = min(idx, len(s))
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    export_dir = Path(sys.argv[1])
    manifest = json.loads((export_dir / "weights_manifest.json").read_text())

    rows = []
    for entry in manifest:
        shape = entry["shape"]
        if len(shape) != 2:
            continue  # skip 1-D biases; nothing to factor
        n, m = shape
        data = np.fromfile(export_dir / entry["file"], dtype="<f4")
        assert data.size == n * m, f"{entry['name']}: expected {n * m} floats, got {data.size}"
        w = data.reshape(n, m)
        s = np.linalg.svd(w, compute_uv=False)
        full_rank = min(n, m)
        ranks = effective_rank(shape, s)
        lora_params_at_r95 = ranks["r95"] * (n + m)
        dense_params = n * m
        rows.append(
            {
                "name": entry["name"],
                "shape": f"{n}x{m}",
                "full_rank": full_rank,
                **ranks,
                "compression_at_r95": f"{dense_params / max(lora_params_at_r95, 1):.2f}x",
            }
        )

    name_w = max(len(r["name"]) for r in rows)
    shape_w = max(len(r["shape"]) for r in rows)
    header = f"{'name':<{name_w}}  {'shape':<{shape_w}}  {'full':>5}  {'r90':>5}  {'r95':>5}  {'r99':>5}  compression@r95"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<{name_w}}  {r['shape']:<{shape_w}}  {r['full_rank']:>5}  "
            f"{r['r90']:>5}  {r['r95']:>5}  {r['r99']:>5}  {r['compression_at_r95']}"
        )

    avg_ratio_frac = np.mean([r["r95"] / r["full_rank"] for r in rows])
    print(f"\nMean r95/full_rank across {len(rows)} matrices: {avg_ratio_frac:.3f}")
    print(
        "(closer to 0 => fast singular-value decay, low-rank factorization is near-free; "
        "closer to 1 => matrix is effectively full-rank, factorization would lose accuracy)"
    )


if __name__ == "__main__":
    main()
