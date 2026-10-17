#!/usr/bin/env python3
"""Rescale the encoder columns of grown k-WTA units in a weights.safetensors, in place.

Units >= N_BASE (1024) of every `<region>_layer.linear.weight` ([n_in, n_units], unit = column) are
rescaled to L2 norm TARGET, keeping direction (columns with ~zero norm are left alone). Everything
else (decoders, heads, binding, base units) is untouched; only the affected tensors' bytes change.
Experiment tool for gaps.md §50 (grown units sit at their 0.01 init norm vs ~0.55 for the originals).

Usage: python3 scripts/rescale_grown_units.py <snapshot>/weights.safetensors [target_norm=0.5]
Needs numpy only. Always run on a COPY of a snapshot (it overwrites the file).
"""
import json
import struct
import sys

import numpy as np

N_BASE = 1024
path = sys.argv[1]
target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
raw = bytearray(open(path, "rb").read())
n = struct.unpack("<Q", raw[:8])[0]
hdr = json.loads(bytes(raw[8 : 8 + n]))
base = 8 + n
for key in sorted(k for k in hdr if k.endswith("_layer.linear.weight")):
    meta = hdr[key]
    assert meta["dtype"] == "F32", (key, meta["dtype"])
    s, e = meta["data_offsets"]
    w = np.frombuffer(bytes(raw[base + s : base + e]), dtype=np.float32).reshape(meta["shape"]).copy()
    nrm = np.linalg.norm(w[:, N_BASE:], axis=0)
    scale = np.where(nrm > 1e-6, target / np.maximum(nrm, 1e-12), 1.0).astype(np.float32)
    w[:, N_BASE:] *= scale
    raw[base + s : base + e] = w.tobytes()
    new = np.linalg.norm(w[:, N_BASE:], axis=0)
    print(f"{key}: {w.shape[1] - N_BASE} grown cols, norm p50 {np.median(nrm):.4f} -> {np.median(new):.4f}")
open(path, "wb").write(raw)
