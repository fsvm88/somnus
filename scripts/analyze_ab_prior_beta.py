#!/usr/bin/env python3
"""Analyze per-region prior-beta A/B (ADR 0027)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CTRL = ROOT / "memory/ab_prior_beta/control/campaign/metrics.jsonl"
TRMT = ROOT / "memory/ab_prior_beta/treatment/campaign/metrics.jsonl"

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def fv(v, fmt=""):
    return format(v, fmt) if v is not None else "  ---  "

def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

ctrl = load(CTRL)
trmt = load(TRMT)

print(f"Loaded: control={len(ctrl)} cycles, treatment={len(trmt)} cycles\n")

REGIONS = ["speech", "env", "visual", "symbolic", "language"]

def print_arm(name, rows):
    print(f"=== {name} ===")
    for r in rows:
        sess = r.get("session", "?")
        cycle = r.get("cycle", "?")
        mse = r.get("output_mse")
        wrl = r.get("warm_recall_l2")
        gist = r.get("gist_recall_rate")
        pc = r.get("prior_cosine_mean", [None]*5)
        betas = r.get("prior_betas_effective", [None]*5)
        beta_str = ",".join(fv(b, ".3f") for b in (betas or [None]*5))
        pc_str = " ".join(fv(c, ".3f") for c in (pc or [None]*5))
        print(
            f"  s{sess}c{cycle}  mse={fv(mse,'.4f')}  warm_recall={fv(wrl,'.4f')}"
            f"  gist={fv(gist,'.3f')}  betas=[{beta_str}]  pc=[{pc_str}]"
        )
    print()

print_arm("CONTROL", ctrl)
print_arm("TREATMENT", trmt)

# Threshold check: s3-5 averages
def arm_avg(rows, key, s_range=(3,5)):
    vals = [r.get(key) for r in rows
            if r.get("session", 0) >= s_range[0] and r.get("session", 0) <= s_range[1]]
    return avg(vals)

def arm_pc_avg(rows, region_idx, s_range=(3,5)):
    vals = []
    for r in rows:
        if r.get("session", 0) < s_range[0] or r.get("session", 0) > s_range[1]:
            continue
        pc = r.get("prior_cosine_mean") or []
        if region_idx < len(pc) and pc[region_idx] is not None:
            vals.append(pc[region_idx])
    return avg(vals)

print("=== THRESHOLD CHECK (s3-5 avg) ===")
print(f"{'Metric':<30} {'Control':>10} {'Treatment':>10} {'Δ':>10} {'Threshold':>12} {'Verdict':>8}")
print("-" * 85)

metrics = [
    ("output_mse", "output_mse", "≤ ctrl×1.10"),
    ("warm_recall_l2", "warm_recall_l2", "±15%"),
    ("gist_recall_rate", "gist_recall_rate", "±0.05"),
]
for label, key, threshold in metrics:
    c = arm_avg(ctrl, key)
    t = arm_avg(trmt, key)
    if c is not None and t is not None:
        delta = (t - c) / c * 100 if key != "gist_recall_rate" else (t - c)
        if key == "output_mse":
            verdict = "PASS" if t <= c * 1.10 else "FAIL"
            delta_str = f"{delta:+.1f}%"
        elif key == "warm_recall_l2":
            verdict = "PASS" if abs(delta) <= 15.0 else "FAIL"
            delta_str = f"{delta:+.1f}%"
        else:  # gist
            verdict = "PASS" if abs(delta) <= 0.05 else "FAIL"
            delta_str = f"{delta:+.3f}"
    else:
        verdict = "---"
        delta_str = "---"
    print(f"  {label:<28} {fv(c,'.4f'):>10} {fv(t,'.4f'):>10} {delta_str:>10} {threshold:>12} {verdict:>8}")

print()
for i, region in enumerate(REGIONS):
    c = arm_pc_avg(ctrl, i)
    t = arm_pc_avg(trmt, i)
    if c is not None and t is not None:
        delta = t - c
        verdict = "PASS" if abs(delta) <= 0.05 else "FAIL"
        delta_str = f"{delta:+.3f}"
    else:
        verdict = "---"
        delta_str = "---"
    print(f"  prior cosine — {region:<14} {fv(c,'.4f'):>10} {fv(t,'.4f'):>10} {delta_str:>10} {'±0.05':>12} {verdict:>8}")
