#!/usr/bin/env python3
"""
Eval harness for somnus sleep-cycle metrics.

Usage:
    python scripts/eval_metrics.py memory/metrics_*.jsonl
    python scripts/eval_metrics.py --out-dir /tmp/plots memory/metrics.jsonl

Reads one or more metrics_*.jsonl files (one JSON object per line) and writes
four PNG files:
  1. language_trajectory.png  — prior_cosine per region across all cycles
  2. growth_timeline.png      — cycles where growth fired, annotated
  3. per_region_mse.png       — per-region output_mse across cycles
  4. sleep_triggers.png       — sleep trigger and nrem exit reason distributions
"""

import argparse
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REGION_NAMES = ["visual", "env", "speech", "symbolic", "language"]
REGION_COLORS = ["#4c8ef5", "#e06c75", "#98c379", "#c678dd", "#e5c07b"]


def load_records(paths):
    records = []
    for path in paths:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  warn: {path}:{lineno}: {e}", file=sys.stderr)
    return records


def get(rec, key, default=None):
    return rec.get(key, default)


def get_list(rec, key, idx, default=None):
    val = rec.get(key)
    if val is None or not isinstance(val, list) or idx >= len(val):
        return default
    return val[idx]


def make_x_labels(records):
    """Build x-axis tick labels.

    When any record has a `session` field: "S{n}C{m}".
    Otherwise fall back to plain "C{n}" (single-session or pre-campaign data).
    """
    has_session = any(get(r, "session") is not None for r in records)
    labels = []
    for i, r in enumerate(records):
        cycle = get(r, "cycle")
        session = get(r, "session")
        if has_session:
            s = session if session is not None else "?"
            c = cycle if cycle is not None else i + 1
            labels.append(f"S{s}C{c}")
        else:
            labels.append(f"C{cycle}" if cycle is not None else str(i + 1))
    return labels


def session_boundaries(records):
    """Return x-indices where a new session starts (i > 0 and session changes)."""
    boundaries = []
    prev_session = None
    for i, r in enumerate(records):
        s = get(r, "session")
        if s is not None and s != prev_session and i > 0:
            boundaries.append(i)
        if s is not None:
            prev_session = s
    return boundaries


def draw_session_lines(ax, records):
    """Draw vertical dashed lines at session boundaries."""
    for xi in session_boundaries(records):
        ax.axvline(xi - 0.5, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)


def plot_language_trajectory(records, out_path, x_labels):
    fig, ax = plt.subplots(figsize=(max(8, len(records) * 0.5), 5))
    x = np.arange(len(records))

    draw_session_lines(ax, records)
    for i, (name, color) in enumerate(zip(REGION_NAMES, REGION_COLORS)):
        ys = [get_list(r, "prior_cosine_mean", i) for r in records]
        valid = [(xi, y) for xi, y in zip(x, ys) if y is not None]
        if not valid:
            continue
        xs_v, ys_v = zip(*valid)
        lw = 2.5 if i == 4 else 1.2
        ls = "-" if i == 4 else "--"
        ax.plot(xs_v, ys_v, color=color, linewidth=lw, linestyle=ls,
                marker="o", markersize=4, label=name)

    ax.set_title("Prior cosine trajectory per region")
    ax.set_ylabel("prior_cosine_mean")
    ax.set_xlabel("sleep cycle")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_growth_timeline(records, out_path, x_labels):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(8, len(records) * 0.5), 6),
                                          gridspec_kw={"height_ratios": [1, 3]})
    x = np.arange(len(records))

    draw_session_lines(ax_top, records)
    draw_session_lines(ax_bot, records)

    # Top: grew flag
    grew = [1 if get(r, "grew") else 0 for r in records]
    ax_top.bar(x, grew, color="#e06c75", width=0.6)
    ax_top.set_ylabel("grew", fontsize=8)
    ax_top.set_yticks([0, 1])
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([""] * len(x))
    ax_top.set_ylim(0, 1.4)
    ax_top.set_title("Growth events timeline")
    ax_top.grid(True, axis="y", alpha=0.3)

    # Bottom: region sizes per region
    for i, (name, color) in enumerate(zip(REGION_NAMES, REGION_COLORS)):
        ys = [get_list(r, "region_sizes", i) for r in records]
        valid = [(xi, y) for xi, y in zip(x, ys) if y is not None]
        if not valid:
            continue
        xs_v, ys_v = zip(*valid)
        ax_bot.plot(xs_v, ys_v, color=color, linewidth=1.5, marker="o",
                    markersize=4, label=name)

    # Annotate growth events
    for xi, r in enumerate(records):
        if get(r, "grew"):
            regions_grown = get(r, "regions_grown", [])
            sizes = get(r, "region_sizes", [])
            note = f"+{[REGION_NAMES[g] if g < len(REGION_NAMES) else g for g in regions_grown]}"
            ax_bot.annotate(note, xy=(xi, max(s for s in sizes if s) if sizes else 1024),
                            xytext=(0, 8), textcoords="offset points",
                            fontsize=6, ha="center", color="#e06c75",
                            arrowprops=dict(arrowstyle="-", color="#e06c75", lw=0.5))

    ax_bot.set_ylabel("region_sizes (units)")
    ax_bot.set_xlabel("sleep cycle")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax_bot.legend(loc="upper left", fontsize=8)
    ax_bot.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_region_mse(records, out_path, x_labels):
    fig, ax = plt.subplots(figsize=(max(8, len(records) * 0.5), 5))
    x = np.arange(len(records))

    draw_session_lines(ax, records)
    for i, (name, color) in enumerate(zip(REGION_NAMES, REGION_COLORS)):
        ys = [get_list(r, "per_region_mse", i) for r in records]
        valid = [(xi, y) for xi, y in zip(x, ys) if y is not None]
        if not valid:
            continue
        xs_v, ys_v = zip(*valid)
        ax.plot(xs_v, ys_v, color=color, linewidth=1.5, marker="o",
                markersize=4, label=name)

    ax.set_title("Per-region output MSE (NREM replay)")
    ax.set_ylabel("per_region_mse")
    ax.set_xlabel("sleep cycle")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_sleep_triggers(records, out_path):
    trigger_counts = Counter()
    exit_counts = Counter()
    for r in records:
        trig = get(r, "sleep_trigger")
        if trig:
            trigger_counts[trig] += 1
        else:
            # older files: infer from was_interrupted field
            trigger_counts["(unknown)"] += 1

        ex = get(r, "nrem_exit_reason")
        if ex:
            exit_counts[ex] += 1
        elif get(r, "was_interrupted"):
            exit_counts["interrupted"] += 1
        else:
            exit_counts["traces_exhausted"] += 1

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))

    def _bar(ax, counts, title, colors):
        if not counts:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            return
        labels = list(counts.keys())
        vals = [counts[k] for k in labels]
        c = [colors.get(k, "#aaaaaa") for k in labels]
        bars = ax.bar(labels, vals, color=c, width=0.5)
        ax.bar_label(bars, padding=2, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    trigger_colors = {
        "high_water": "#4c8ef5",
        "schedule": "#98c379",
        "perf_degradation": "#e06c75",
        "(unknown)": "#aaaaaa",
    }
    exit_colors = {
        "traces_exhausted": "#4c8ef5",
        "loss_target_met": "#98c379",
        "time_budget": "#e5c07b",
        "salience": "#c678dd",
        "interrupted": "#e06c75",
    }
    _bar(ax_l, trigger_counts, "Sleep trigger distribution", trigger_colors)
    _bar(ax_r, exit_counts, "NREM exit reason distribution", exit_colors)

    fig.suptitle("Sleep trigger and NREM exit reasons", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_timing_breakdown(records, out_path, x_labels):
    """Stacked bar chart showing sleep-cycle phase timing per cycle.

    Tolerates records that predate the timing fields (shows them as zero).
    Stacks: NREM replay, distillation precompute, REM work, spill, snapshot.
    An "other" bar absorbs unmeasured overhead so bars don't imply 100% coverage.
    """
    TIMING_FIELDS = [
        ("nrem_wall_ms",          "NREM replay",          "#4c8ef5"),
        ("distill_precompute_ms", "distill precompute",   "#e5c07b"),
        ("rem_wall_ms",           "REM (probe+grow+synth)","#98c379"),
        ("spill_wall_ms",         "spill hot→cold/warm",  "#c678dd"),
        ("snapshot_wall_ms",      "snapshot writes",      "#e06c75"),
    ]
    has_any = any(
        any(get(r, f) is not None for f, _, _ in TIMING_FIELDS)
        for r in records
    )
    if not has_any:
        # All records predate timing instrumentation — skip rather than crash.
        print(f"  skip: no timing fields in records — {out_path} not written")
        return

    x = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(max(8, len(records) * 0.5), 5))
    draw_session_lines(ax, records)

    bottoms = np.zeros(len(records))
    for field, label, color in TIMING_FIELDS:
        vals = np.array([float(get(r, field) or 0) for r in records])
        ax.bar(x, vals / 1000.0, bottom=bottoms / 1000.0,
               color=color, label=label, width=0.7)
        bottoms += vals

    ax.set_title("Sleep-cycle phase timing breakdown")
    ax.set_ylabel("wall time (s)")
    ax.set_xlabel("sleep cycle")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", metavar="JSONL",
                        help="metrics_*.jsonl files to analyse")
    parser.add_argument("--out-dir", metavar="DIR",
                        help="directory for output PNGs (default: same as first input)")
    args = parser.parse_args()

    missing = [p for p in args.files if not os.path.exists(p)]
    if missing:
        print(f"error: file(s) not found: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(args.files)} file(s)…")
    records = load_records(args.files)
    if not records:
        print("error: no valid records found", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(records)} cycles loaded")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.files[0]))
    os.makedirs(out_dir, exist_ok=True)

    x_labels = make_x_labels(records)

    print("Generating charts…")
    plot_language_trajectory(records,
                             os.path.join(out_dir, "language_trajectory.png"),
                             x_labels)
    plot_growth_timeline(records,
                         os.path.join(out_dir, "growth_timeline.png"),
                         x_labels)
    plot_per_region_mse(records,
                        os.path.join(out_dir, "per_region_mse.png"),
                        x_labels)
    plot_sleep_triggers(records,
                        os.path.join(out_dir, "sleep_triggers.png"))
    plot_timing_breakdown(records,
                          os.path.join(out_dir, "timing_breakdown.png"),
                          x_labels)
    print("Done.")


if __name__ == "__main__":
    main()
