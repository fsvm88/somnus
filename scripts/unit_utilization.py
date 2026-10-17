#!/usr/bin/env python3
"""Per-region k-WTA encoder column-norm distribution from a weights.safetensors snapshot.

Units are columns of `<region>_layer.linear.weight` ([n_in, n_units]); the first N_UNITS (1024) are the
original units, the rest were added by region growth (initialised at norm 0.01, see
`grow_region`/`grow_region_seeded`). Used for gaps.md §50 (grown units stay at init norm).

Usage: python3 scripts/unit_utilization.py <snapshot>/weights.safetensors   (needs numpy only)
"""
import json
import struct
import sys

import numpy as np

N_BASE = 1024
raw = open(sys.argv[1], "rb").read()
n = struct.unpack("<Q", raw[:8])[0]
hdr = json.loads(raw[8 : 8 + n])
base = 8 + n
for key in sorted(k for k in hdr if k.endswith("_layer.linear.weight")):
    s, e = hdr[key]["data_offsets"]
    w = np.frombuffer(raw[base + s : base + e], dtype=np.float32).reshape(hdr[key]["shape"])
    nrm = np.linalg.norm(w, axis=0)
    b, g = nrm[:N_BASE], nrm[N_BASE:]
    line = f"{key.split('.')[0]:16} units={w.shape[1]:6} base med={np.median(b):.3f} min={b.min():.3f}"
    if len(g):
        line += (f" | grown n={len(g)} min={g.min():.4f} p50={np.median(g):.4f} "
                 f"p99={np.percentile(g, 99):.4f} max={g.max():.4f} "
                 f"frac>0.1={(g > 0.1).mean():.4f}")
    print(line)
