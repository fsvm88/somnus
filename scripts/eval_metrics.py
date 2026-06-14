#!/usr/bin/env python3
"""
Eval harness for somnus sleep-cycle metrics.

Usage:
    python scripts/eval_metrics.py memory/metrics_*.jsonl
    python scripts/eval_metrics.py --out-dir /tmp/plots memory/metrics.jsonl

Reads one or more metrics_*.jsonl files (one JSON object per line) and writes
ten PNG files:
  1. language_trajectory.png  — prior_cosine per region across all cycles
  2. growth_timeline.png      — cycles where growth fired, annotated
  3. per_region_mse.png       — per-region output_mse across cycles
  4. sleep_triggers.png       — sleep trigger and nrem exit reason distributions
  5. timing_breakdown.png     — sleep-cycle phase wall times (stacked bars)
  6. wake_timing.png          — wake-phase encode timing totals and per-episode
  7. memory_growth.png        — warm/cold store sizes and spill cost over time
  8. recall_quality.png       — salience trajectory, blend rate, reconstruction rate
  9. settle_stats.png         — settle delta vs SETTLE_EPS, trigger fraction, mean iters
 10. modality_balance.png     — episode source breakdown and LM dedup efficiency
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


def plot_wake_timing(records, out_path, x_labels):
    """Wake-phase encode and gain timing per cycle.

    Two sub-plots:
      Top: stacked bar of total ms spent per wake cycle on each encode path.
      Bottom: per-episode averages (total_ms / wake_episodes), revealing whether
              per-frame cost is changing rather than just total volume.

    Skips silently if no wake-timing fields exist (pre-instrumentation records).
    """
    WAKE_FIELDS = [
        ("speech_encode_total_ms", "speech encode (Whisper)", "#4c8ef5"),
        ("env_encode_total_ms",    "env encode (CLAP)",       "#98c379"),
        ("video_encode_total_ms",  "video encode (DINOv2)",   "#c678dd"),
        ("gain_update_total_ms",   "gain update (Qwen)",      "#e5c07b"),
    ]
    has_any = any(
        any(get(r, f) is not None for f, _, _ in WAKE_FIELDS)
        for r in records
    )
    if not has_any:
        print(f"  skip: no wake timing fields in records — {out_path} not written")
        return

    x = np.arange(len(records))
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(8, len(records) * 0.5), 7),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    draw_session_lines(ax_top, records)
    draw_session_lines(ax_bot, records)

    # Top: stacked totals (seconds)
    bottoms = np.zeros(len(records))
    for field, label, color in WAKE_FIELDS:
        vals = np.array([float(get(r, field) or 0) for r in records])
        ax_top.bar(x, vals / 1000.0, bottom=bottoms / 1000.0,
                   color=color, label=label, width=0.7)
        bottoms += vals

    ax_top.set_title("Wake-phase encode & gain timing per cycle")
    ax_top.set_ylabel("total wall time (s)")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([""] * len(x))
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(True, axis="y", alpha=0.3)

    # Bottom: per-episode averages (ms per episode)
    episodes = np.array([float(get(r, "wake_episodes") or 1) for r in records])
    episodes = np.maximum(episodes, 1)  # avoid divide-by-zero on empty wake cycles
    for field, label, color in WAKE_FIELDS:
        vals = np.array([float(get(r, field) or 0) for r in records])
        ax_bot.plot(x, vals / episodes, color=color, linewidth=1.5,
                    marker="o", markersize=4, label=label)

    ax_bot.set_title("Per-episode averages")
    ax_bot.set_ylabel("ms / episode")
    ax_bot.set_xlabel("sleep cycle")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_memory_growth(records, out_path, x_labels):
    """warm/cold store sizes and spill wall time per cycle."""
    has_mem = any(get(r, "warm_len") is not None for r in records)
    if not has_mem:
        print(f"  skip: no warm_len field — {out_path} not written")
        return

    x = np.arange(len(records))
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(8, len(records) * 0.5), 7),
                                          gridspec_kw={"height_ratios": [3, 2]})
    draw_session_lines(ax_top, records)
    draw_session_lines(ax_bot, records)

    warm = [get(r, "warm_len") for r in records]
    cold = [get(r, "cold_len") for r in records]
    for ys, label, color in [(warm, "warm (HNSW)", "#4c8ef5"), (cold, "cold (archive)", "#e06c75")]:
        valid = [(xi, y) for xi, y in zip(x, ys) if y is not None]
        if valid:
            xs_v, ys_v = zip(*valid)
            ax_top.plot(xs_v, ys_v, color=color, linewidth=1.8, marker="o", markersize=4, label=label)

    ax_top.set_title("Memory store sizes over time")
    ax_top.set_ylabel("records")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([""] * len(x))
    ax_top.legend(loc="upper left", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    spill = [get(r, "spill_wall_ms") for r in records]
    valid_s = [(xi, y) for xi, y in zip(x, spill) if y is not None]
    if valid_s:
        xs_v, ys_v = zip(*valid_s)
        ax_bot.bar(xs_v, [y / 1000.0 for y in ys_v], color="#c678dd", width=0.7, label="spill_wall_ms")
    ax_bot.set_title("Spill wall time (includes HNSW rebuild when warm > max)")
    ax_bot.set_ylabel("wall time (s)")
    ax_bot.set_xlabel("sleep cycle")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_recall_quality(records, out_path, x_labels):
    """Salience trajectory with threshold reference lines, blend rate, and reconstruction rate."""
    has_sal = any(get(r, "mean_salience_wake") is not None for r in records)
    if not has_sal:
        print(f"  skip: no salience fields — {out_path} not written")
        return

    BLEND_THRESH = 0.5
    RECONSTRUCT_THRESH = 0.8

    x = np.arange(len(records))
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(8, len(records) * 0.5), 7),
                                          gridspec_kw={"height_ratios": [3, 2]})
    draw_session_lines(ax_top, records)
    draw_session_lines(ax_bot, records)

    mean_sal = [get(r, "mean_salience_wake") for r in records]
    min_sal  = [get(r, "min_salience_wake")  for r in records]
    for ys, label, color, lw in [
        (mean_sal, "mean_salience_wake", "#4c8ef5", 2.0),
        (min_sal,  "min_salience_wake",  "#e5c07b", 1.2),
    ]:
        valid = [(xi, y) for xi, y in zip(x, ys) if y is not None]
        if valid:
            xs_v, ys_v = zip(*valid)
            ax_top.plot(xs_v, ys_v, color=color, linewidth=lw, marker="o", markersize=4, label=label)

    ax_top.axhline(BLEND_THRESH,       color="#e06c75", linewidth=1.0, linestyle="--",
                   label=f"blend threshold ({BLEND_THRESH})", alpha=0.8)
    ax_top.axhline(RECONSTRUCT_THRESH, color="#c678dd", linewidth=1.0, linestyle=":",
                   label=f"reconstruct threshold ({RECONSTRUCT_THRESH})", alpha=0.8)
    ax_top.set_title("Warm-recall salience (L2 distance to nearest memory)")
    ax_top.set_ylabel("L2 distance")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([""] * len(x))
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    episodes = np.array([float(get(r, "wake_episodes") or 1) for r in records])
    episodes = np.maximum(episodes, 1)
    high_sal_frac = np.array([float(get(r, "high_salience_episodes") or 0) for r in records]) / episodes
    blend_frac    = np.array([float(get(r, "blend_count")             or 0) for r in records]) / episodes
    recon_frac    = np.array([float(get(r, "wake_reconstruct_fired")  or 0) for r in records]) / episodes

    ax_bot.plot(x, high_sal_frac, color="#4c8ef5", linewidth=1.5, marker="o", markersize=4,
                label="high_salience fraction")
    ax_bot.plot(x, blend_frac,    color="#e06c75", linewidth=1.5, marker="s", markersize=4,
                label="blend fraction")
    ax_bot.plot(x, recon_frac,    color="#98c379", linewidth=1.5, marker="^", markersize=4,
                label="reconstruct_fired fraction")
    ax_bot.set_title("Recall engagement rates (fraction of wake episodes)")
    ax_bot.set_ylabel("fraction")
    ax_bot.set_ylim(0, 1.05)
    ax_bot.set_xlabel("sleep cycle")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax_bot.legend(loc="lower right", fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_settle_stats(records, out_path, x_labels):
    """Settle delta vs SETTLE_EPS, trigger fraction, and mean iterations per cycle."""
    has_settle = any(get(r, "settle_delta_mean") is not None for r in records)
    if not has_settle:
        print(f"  skip: no settle_delta_mean field — {out_path} not written")
        return

    SETTLE_EPS = 4.0

    x = np.arange(len(records))
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(8, len(records) * 0.5), 7),
                                          gridspec_kw={"height_ratios": [2, 2]})
    draw_session_lines(ax_top, records)
    draw_session_lines(ax_bot, records)

    deltas = [get(r, "settle_delta_mean") for r in records]
    valid_d = [(xi, y) for xi, y in zip(x, deltas) if y is not None]
    if valid_d:
        xs_v, ys_v = zip(*valid_d)
        ax_top.plot(xs_v, ys_v, color="#4c8ef5", linewidth=2.0, marker="o", markersize=4,
                    label="settle_delta_mean")
    ax_top.axhline(SETTLE_EPS, color="#e06c75", linewidth=1.0, linestyle="--",
                   label=f"SETTLE_EPS ({SETTLE_EPS})", alpha=0.8)
    ax_top.set_title("Settling convergence: mean L2 delta per settled episode")
    ax_top.set_ylabel("L2 delta")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([""] * len(x))
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    episodes = np.array([float(get(r, "wake_episodes") or 1) for r in records])
    episodes = np.maximum(episodes, 1)
    trigger_frac = np.array([float(get(r, "settle_trigger_count") or 0) for r in records]) / episodes

    trigger_counts = np.array([float(get(r, "settle_trigger_count") or 0) for r in records])
    iters_sum = np.array([float(get(r, "settle_iters_sum") or 0) for r in records])
    mean_iters = np.where(trigger_counts > 0, iters_sum / trigger_counts, 0.0)

    ax_bot.plot(x, trigger_frac, color="#98c379", linewidth=1.5, marker="o", markersize=4,
                label="trigger fraction (gate fires / wake_episodes)")
    ax_bot2 = ax_bot.twinx()
    ax_bot2.plot(x, mean_iters, color="#c678dd", linewidth=1.5, marker="s", markersize=4,
                 linestyle="--", label="mean iters per triggered episode")
    ax_bot2.set_ylabel("mean iters", color="#c678dd")
    ax_bot2.tick_params(axis="y", labelcolor="#c678dd")

    ax_bot.set_title("Settle gate trigger rate and iteration count")
    ax_bot.set_ylabel("fraction")
    ax_bot.set_ylim(0, 1.05)
    ax_bot.set_xlabel("sleep cycle")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    lines1, labels1 = ax_bot.get_legend_handles_labels()
    lines2, labels2 = ax_bot2.get_legend_handles_labels()
    ax_bot.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_modality_balance(records, out_path, x_labels):
    """Episode source stacked bars and LM dedup efficiency per cycle."""
    has_modal = any(get(r, "episodes_mixed") is not None for r in records)
    if not has_modal:
        print(f"  skip: no modality fields — {out_path} not written")
        return

    MODAL_FIELDS = [
        ("episodes_speech",   "speech",   "#4c8ef5"),
        ("episodes_env",      "env",      "#e06c75"),
        ("episodes_visual",   "visual",   "#98c379"),
        ("episodes_language", "language", "#e5c07b"),
        ("episodes_mixed",    "mixed",    "#c678dd"),
    ]

    x = np.arange(len(records))
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(8, len(records) * 0.5), 7),
                                          gridspec_kw={"height_ratios": [3, 2]})
    draw_session_lines(ax_top, records)
    draw_session_lines(ax_bot, records)

    bottoms = np.zeros(len(records))
    for field, label, color in MODAL_FIELDS:
        vals = np.array([float(get(r, field) or 0) for r in records])
        ax_top.bar(x, vals, bottom=bottoms, color=color, label=label, width=0.7)
        bottoms += vals

    ax_top.set_title("Episode source breakdown per cycle")
    ax_top.set_ylabel("episode count")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([""] * len(x))
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(True, axis="y", alpha=0.3)

    # LM dedup efficiency: unique / computed ratio and distill time
    computed = np.array([float(get(r, "lm_hidden_computed") or 0) for r in records])
    unique   = np.array([float(get(r, "lm_hidden_unique")   or 0) for r in records])
    dedup_ratio = np.where(computed > 0, unique / computed, np.nan)

    ax_bot.plot(x, dedup_ratio, color="#4c8ef5", linewidth=1.8, marker="o", markersize=4,
                label="LM unique/computed (dedup efficiency)")
    ax_bot.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_bot.set_ylim(0, 1.1)
    ax_bot.set_ylabel("ratio (lower = more dedup)")

    ax_bot2 = ax_bot.twinx()
    distill = [get(r, "distill_precompute_ms") for r in records]
    valid_t = [(xi, y) for xi, y in zip(x, distill) if y is not None]
    if valid_t:
        xs_v, ys_v = zip(*valid_t)
        ax_bot2.plot(xs_v, [y / 1000.0 for y in ys_v], color="#e5c07b", linewidth=1.2,
                     marker="s", markersize=3, linestyle="--", label="distill_precompute (s)")
    ax_bot2.set_ylabel("distill time (s)", color="#e5c07b")
    ax_bot2.tick_params(axis="y", labelcolor="#e5c07b")

    ax_bot.set_title("LM dedup efficiency and distillation precompute time")
    ax_bot.set_xlabel("sleep cycle")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    lines1, labels1 = ax_bot.get_legend_handles_labels()
    lines2, labels2 = ax_bot2.get_legend_handles_labels()
    ax_bot.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_gist_quality(records, out_path, x_labels):
    """Gist recall rate (cross-session vs within-session) and growth trigger signal."""
    has_gist = any(get(r, "gist_recall_rate") is not None for r in records)
    if not has_gist:
        print(f"  skip: no gist_recall_rate field — {out_path} not written")
        return

    x = np.arange(len(records))
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(8, len(records) * 0.5), 7),
                                          gridspec_kw={"height_ratios": [3, 1]})
    draw_session_lines(ax_top, records)
    draw_session_lines(ax_bot, records)

    gist = [get(r, "gist_recall_rate") for r in records]
    ws_gist = [get(r, "within_session_gist_rate") for r in records]

    for ys, label, color, ls in [
        (gist,    "gist_recall_rate (cross-session)",   "#e06c75", "-"),
        (ws_gist, "within_session_gist_rate (clean)",   "#98c379", "--"),
    ]:
        valid = [(xi, y) for xi, y in zip(x, ys) if y is not None]
        if valid:
            xs_v, ys_v = zip(*valid)
            ax_top.plot(xs_v, ys_v, color=color, linewidth=1.8, marker="o",
                        markersize=4, linestyle=ls, label=label)

    ax_top.axhline(0.85, color="#e5c07b", linewidth=1.0, linestyle=":",
                   label="GIST_GROWTH_THRESHOLD (0.85)", alpha=0.8)
    ax_top.set_title("Gist recall rate: cross-session vs within-session (ADR 0024)")
    ax_top.set_ylabel("gist recall rate")
    ax_top.set_ylim(0, 1.05)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([""] * len(x))
    ax_top.legend(loc="lower right", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    grew = [1 if get(r, "grew") else 0 for r in records]
    ax_bot.bar(x, grew, color="#e06c75", width=0.6, label="growth event")
    ax_bot.set_ylabel("grew")
    ax_bot.set_yticks([0, 1])
    ax_bot.set_ylim(0, 1.4)
    ax_bot.set_xlabel("sleep cycle")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.grid(True, axis="y", alpha=0.3)

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
    plot_wake_timing(records,
                     os.path.join(out_dir, "wake_timing.png"),
                     x_labels)
    plot_memory_growth(records,
                       os.path.join(out_dir, "memory_growth.png"),
                       x_labels)
    plot_recall_quality(records,
                        os.path.join(out_dir, "recall_quality.png"),
                        x_labels)
    plot_settle_stats(records,
                      os.path.join(out_dir, "settle_stats.png"),
                      x_labels)
    plot_modality_balance(records,
                          os.path.join(out_dir, "modality_balance.png"),
                          x_labels)
    plot_gist_quality(records,
                      os.path.join(out_dir, "gist_quality.png"),
                      x_labels)
    print("Done.")


if __name__ == "__main__":
    main()
