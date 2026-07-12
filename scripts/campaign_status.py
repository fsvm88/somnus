#!/usr/bin/env python3
"""
One-shot status report for a train_campaign.sh run: progress, region growth,
quality trend, disk usage, and basic health checks — everything you'd
otherwise reconstruct by hand from metrics.jsonl + logs/ + du.

Usage:
    scripts/campaign_status.py                        # newest memory/campaign_* dir
    scripts/campaign_status.py memory/campaign_growth10x
    scripts/campaign_status.py --target-units 9660 --window 10

Reads only params.txt + metrics.jsonl + logs/*.log under the campaign dir,
plus `du`/`pgrep` for disk/process state. No non-stdlib dependencies.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

# Index order is RegionId::as_index (crates/somnus-perception/src/region.rs), NOT
# eval_metrics.py's REGION_NAMES (which has visual/speech swapped at 0/2 — see
# that file's fix commit for detail). Keep this in sync if RegionId ever changes.
REGION_NAMES = ["speech", "env", "visual", "symbolic", "language"]

# Pilot-3 baseline (ai-docs/gaps.md §16 "pilot 3 / v2", §19 recalibration note) — the
# post-§17/§18/§19 characterization run this campaign's schedule is judged against.
EXPECTED_MEAN_BATCH = 35.6          # units/growth-event
EXPECTED_CADENCE_CYCLES = 5.36      # cycles/growth-event
EXPECTED_RATE_RANGE = (25.9, 31.9)  # s/cycle, observed range across pilots 2-3
EXPECTED_TOTAL_HOURS_RANGE = (9.3, 11.5)  # full 10x-campaign estimate

_CODES = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "cyan": "\033[36m", "magenta": "\033[35m",
    "label": "\033[94m",  # light blue
    "value": "\033[92m",  # light green
}


def assess_range(value, low, high, lower_is_better=True):
    """green = beating the pilot-3 range, red = missing it, yellow = within it."""
    if lower_is_better:
        if value < low:
            return "green"
        if value > high:
            return "red"
        return "yellow"
    if value > high:
        return "green"
    if value < low:
        return "red"
    return "yellow"


def assess_point(value, expected, tolerance=0.10, lower_is_better=False):
    """Same idea as assess_range but against a single point estimate ± tolerance."""
    if expected == 0:
        return "yellow"
    ratio = value / expected
    if lower_is_better:
        if ratio <= 1 - tolerance:
            return "green"
        if ratio >= 1 + tolerance:
            return "red"
        return "yellow"
    if ratio >= 1 + tolerance:
        return "green"
    if ratio <= 1 - tolerance:
        return "red"
    return "yellow"


class Color:
    """Minimal ANSI wrapper — no-ops when stdout isn't a tty, NO_COLOR is set,
    or --no-color is passed, so piping to a file/log stays plain text."""

    def __init__(self, enabled):
        self.enabled = enabled

    def __call__(self, code, text):
        if not self.enabled:
            return str(text)
        return f"{_CODES[code]}{text}{_CODES['reset']}"

    def header(self, text):
        return self("bold", self("cyan", text))

    def n(self, value):
        """A plain numeric/value token — light green."""
        return self("value", value)

    def kv(self, key, value, sep="="):
        """key=value or 'key: value', key in light blue, value in light green."""
        return f"{self('label', key)}{sep}{self.n(value)}"


def find_campaign_dir(explicit):
    if explicit:
        return explicit
    candidates = sorted(
        glob.glob("memory/campaign_*"), key=os.path.getmtime, reverse=True
    )
    if not candidates:
        sys.exit("no memory/campaign_* directory found; pass one explicitly")
    return candidates[0]


def read_params(campaign_dir):
    path = os.path.join(campaign_dir, "params.txt")
    params = {}
    if not os.path.exists(path):
        return params
    for line in open(path):
        line = line.strip()
        if "=" in line:
            k, _, v = line.partition("=")
            params[k] = v
    return params


def read_metrics(campaign_dir):
    path = os.path.join(campaign_dir, "metrics.jsonl")
    if not os.path.exists(path):
        sys.exit(f"no metrics.jsonl in {campaign_dir}")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn trailing line mid-write
    if not records:
        sys.exit(f"metrics.jsonl in {campaign_dir} is empty")
    return records


def gv(d, key, default=float("nan")):
    """dict.get that also treats an explicit JSON null as missing, not just an absent key
    (some metrics fields — e.g. within_session_gist_rate before the first gist probe — are
    present but null early in a session; plain .get(key, default) doesn't catch that)."""
    v = d.get(key)
    return default if v is None else v


def fmt_hms(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def du(path):
    if not os.path.exists(path):
        return None
    out = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return out.stdout.split()[0]


def process_status(campaign_dir):
    out = subprocess.run(
        ["pgrep", "-af", "train_campaign.sh"], capture_output=True, text=True
    )
    wrapper_running = bool(out.stdout.strip())
    somnus_out = subprocess.run(
        ["pgrep", "-af", "target/release/somnus"], capture_output=True, text=True
    )
    somnus_line = somnus_out.stdout.strip().splitlines()
    somnus_running = bool(somnus_line)
    session = None
    if somnus_line:
        m = re.search(r"--session (\d+)", somnus_line[0])
        if m:
            session = int(m.group(1))
    return wrapper_running, somnus_running, session


def latest_session_log(campaign_dir):
    logs = glob.glob(os.path.join(campaign_dir, "logs", "session_*.log"))
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def onnx_status(campaign_dir):
    """Best-effort ORT-active status (ADR 0037), scanned from the newest session log.
    Each train_campaign.sh session is a fresh somnus process that re-checks the crossover
    at its own startup, so the newest log alone reflects current state — no need to
    scan history. Pre-ADR-0037 logs (before 2026-07-12) won't have these markers."""
    newest = latest_session_log(campaign_dir)
    if not newest:
        return "unknown"
    active = failed = seen_check = False
    with open(newest, errors="replace") as f:
        for line in f:
            if "ORT binding session active" in line:
                active = True
            elif "ORT session init failed" in line or "ORT session rebuild failed" in line:
                failed = True
            elif "ORT crossover" in line or "crossover" in line and "region" in line:
                seen_check = True
    if active:
        return "active"
    if failed:
        return "cpu (init failed, check log)"
    if seen_check:
        return "cpu (below crossover)"
    return "cpu (pre-ADR-0037 run)"


def scan_panics(campaign_dir):
    hits = []
    for path in sorted(
        glob.glob(os.path.join(campaign_dir, "logs", "session_*.log"))
    ):
        with open(path, errors="replace") as f:
            for line in f:
                if "panicked at" in line:
                    hits.append((os.path.basename(path), line.strip()))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign_dir", nargs="?", default=None)
    ap.add_argument("--target-units", type=int, default=9660,
                     help="per-region unit target to project growth ETA against (default: 9660, the 10x-scale honeybee-order target from gaps.md §16)")
    ap.add_argument("--window", type=int, default=10,
                     help="cycle window for early-vs-late quality trend comparison (default: 10)")
    ap.add_argument("--rate-window", type=int, default=50,
                     help="cycle window used to estimate current s/cycle rate (default: 50, most recent)")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = ap.parse_args()

    c = Color(sys.stdout.isatty() and not args.no_color and not os.environ.get("NO_COLOR"))

    campaign_dir = find_campaign_dir(args.campaign_dir)
    params = read_params(campaign_dir)
    records = read_metrics(campaign_dir)
    n = len(records)
    first, last = records[0], records[-1]

    target_cycles = None
    if "SESSIONS" in params and "CYCLES" in params:
        # params.txt is appended-to (not overwritten) across train_campaign.sh restarts, so
        # after a resume with a fresh SESSION_START/SESSIONS this is the *last* leg's block —
        # SESSIONS*CYCLES alone would understate the original whole-campaign target. Using
        # (SESSION_START + SESSIONS - 1) * CYCLES instead gives the intended final cycle
        # count regardless of how many times the campaign has been stopped and resumed.
        session_start = int(params.get("SESSION_START", 1))
        session_end = session_start + int(params["SESSIONS"]) - 1
        target_cycles = session_end * int(params["CYCLES"])

    elapsed = last["ts"] - first["ts"]
    rate_slice = records[-args.rate_window:]
    if len(rate_slice) > 1:
        rate = (rate_slice[-1]["ts"] - rate_slice[0]["ts"]) / (len(rate_slice) - 1)
    else:
        rate = elapsed / max(n - 1, 1)

    wrapper_running, somnus_running, live_session = process_status(campaign_dir)

    print(c("bold", f"=== somnus campaign status: {campaign_dir} ==="))
    if params:
        parts = [
            c.kv("SESSIONS", params.get("SESSIONS")),
            c.kv("CYCLES", params.get("CYCLES")),
            c.kv("CAPACITY", params.get("CAPACITY")),
            c.kv("REPLAY_EPOCHS", params.get("REPLAY_EPOCHS")),
            c.kv("onnx", onnx_status(campaign_dir)),
        ]
        print(c("label", "params: ") + " ".join(parts))
    if wrapper_running:
        label = f"RUNNING (somnus session {live_session})" if somnus_running else "RUNNING (between sessions)"
        status = c("green", label)
    elif somnus_running is False:
        status = c("yellow", "STOPPED")
    else:
        status = c("red", "STOPPED (but a somnus process is still alive — check for orphan)")
    print(c("label", "process: ") + status)

    print()
    print(c.header("-- progress --"))
    cycles_line = c.kv("cycles logged", n, sep=": ")
    if target_cycles:
        cycles_line += f" / {c.n(target_cycles)} target ({c.n(f'{100*n/target_cycles:.1f}%')})"
    print(cycles_line)
    print(c("label", "span: ") + c.n(f"{fmt_ts(first['ts'])} -> {fmt_ts(last['ts'])}")
          + f"  (elapsed {c.n(fmt_hms(elapsed))})")
    rate_color = assess_range(rate, *EXPECTED_RATE_RANGE, lower_is_better=True)
    print(c("label", "rate: ") + c(rate_color, f"{rate:.1f}") + " s/cycle "
          + f"(last {c.n(len(rate_slice))} cycles, pilot-3 was "
          + f"{EXPECTED_RATE_RANGE[0]}-{EXPECTED_RATE_RANGE[1]}s)")
    if target_cycles and n < target_cycles:
        remaining_cycles = target_cycles - n
        print(c("label", "ETA to session target: ") + c("yellow", fmt_hms(remaining_cycles * rate))
              + f" ({c.n(remaining_cycles)} cycles remaining)")
    elif target_cycles:
        print(c("green", "session target reached."))

    cycles_per_session = int(params["CYCLES"]) if "CYCLES" in params else None
    cur_session = last.get("session")
    cur_cycle = last.get("cycle")
    if cycles_per_session and cur_session is not None and cur_cycle is not None:
        if not wrapper_running:
            print(c("label", "current session: ")
                  + f"{c.n(cur_session)} finished at cycle {c.n(cur_cycle)}/{c.n(cycles_per_session)} (campaign not running)")
        elif cur_cycle >= cycles_per_session:
            # Last logged cycle closed out its session; the running somnus process (if any)
            # is already into the next one, still warming up (no cycle logged yet).
            nxt = live_session if live_session else cur_session + 1
            print(c("label", "current session: ") + f"{c.n(nxt)} just started, no cycle logged yet")
        else:
            remaining_in_session = cycles_per_session - cur_cycle
            print(c("label", "current session: ")
                  + f"{c.n(cur_session)}, cycle {c.n(cur_cycle)}/{c.n(cycles_per_session)} "
                  + f"({c.n(f'{100*cur_cycle/cycles_per_session:.0f}%')}) — "
                  + c("label", "ETA to end of session: ")
                  + c("yellow", fmt_hms(remaining_in_session * rate)))

    print()
    print(c.header("-- growth --"))
    sizes = last["region_sizes"]
    baseline = records[0]["region_sizes"]
    # Row color = actual growth-so-far vs. what pilot-3's rate would predict by this many
    # cycles — "better/in line/worse than expected", not a raw fraction of the 10x target
    # (which says nothing about pace).
    expected_units_per_cycle = EXPECTED_MEAN_BATCH / EXPECTED_CADENCE_CYCLES
    expected_growth_so_far = expected_units_per_cycle * n
    print(c("dim", f"{'region':<10}{'size':>8}{'delta':>10}{'%-of-target':>14}"))
    for i, name in enumerate(REGION_NAMES):
        pct = 100 * sizes[i] / args.target_units
        actual_growth = sizes[i] - baseline[i]
        row_color = assess_point(actual_growth, expected_growth_so_far, tolerance=0.15, lower_is_better=False)
        row = f"{name:<10}{sizes[i]:>8}{sizes[i]-baseline[i]:>+10}{pct:>13.1f}%"
        print(c(row_color, row))

    growth_events = [r for r in records if r.get("grew")]
    if growth_events:
        # batch size per event, inferred from the region-size delta at the growth cycle
        # (only meaningful when all regions grow together, which is how grow_region_seeded
        # currently fires — flag if that ever stops being true).
        deltas = []
        prev_sizes = baseline
        for r in records:
            if r.get("grew"):
                d = r["region_sizes"][0] - prev_sizes[0]
                if d > 0:
                    deltas.append(d)
            prev_sizes = r["region_sizes"]
        mean_batch = sum(deltas) / len(deltas) if deltas else 0
        cadence = n / len(growth_events)
        batch_color = assess_point(mean_batch, EXPECTED_MEAN_BATCH, tolerance=0.10, lower_is_better=False)
        cadence_color = assess_point(cadence, EXPECTED_CADENCE_CYCLES, tolerance=0.10, lower_is_better=True)
        print(c("label", "growth events: ") + c.n(len(growth_events))
              + c("label", "   mean batch: ") + c(batch_color, f"{mean_batch:.1f}") + " units/event"
              + c("label", "   cadence: ") + f"1/{c(cadence_color, f'{cadence:.2f}')} cycles"
              + c("dim", f"  (pilot-3: {EXPECTED_MEAN_BATCH}/event, 1/{EXPECTED_CADENCE_CYCLES})"))
        remaining = max(args.target_units - sizes[0], 0)
        if remaining > 0 and mean_batch > 0:
            remaining_events = remaining / mean_batch
            remaining_cycles = remaining_events * cadence
            print(c("label", f"ETA to target ({args.target_units}/region): ")
                  + c("yellow", fmt_hms(remaining_cycles * rate))
                  + f"  (~{c.n(f'{remaining_events:.0f}')} more events, ~{c.n(f'{remaining_cycles:.0f}')} cycles)")
            projected_total_h = (elapsed + remaining_cycles * rate) / 3600
            schedule_color = assess_range(projected_total_h, *EXPECTED_TOTAL_HOURS_RANGE, lower_is_better=True)
            print(c("label", "projected total campaign time: ")
                  + c(schedule_color, f"{projected_total_h:.1f}h")
                  + c("dim", f"  (pilot-3 estimate: {EXPECTED_TOTAL_HOURS_RANGE[0]}-{EXPECTED_TOTAL_HOURS_RANGE[1]}h)"))
        else:
            print(c("green", "target reached."))
    else:
        print(c("dim", "no growth events yet."))

    pruned = sum(r.get("total_units_pruned", 0) for r in records)
    if pruned:
        print(c("label", "total units pruned (cumulative field, last value): ")
              + c.n(last.get("total_units_pruned", 0)))

    print()
    print(c.header("-- quality --"))
    early = records[:args.window]
    late = records[-args.window:]
    early_mse = sum(r["mean_output_mse"] for r in early) / len(early)
    late_mse = sum(r["mean_output_mse"] for r in late) / len(late)
    delta_pct = 100 * (late_mse - early_mse) / early_mse if early_mse else 0
    delta_color = "green" if delta_pct < -1 else "red" if delta_pct > 1 else "yellow"
    print(c("label", "mean_output_mse: ")
          + c.kv(f"first{len(early)}", f"{early_mse:.4f}", sep="=") + "  "
          + c.kv(f"last{len(late)}", f"{late_mse:.4f}", sep="=") + "  "
          + f"({c(delta_color, f'{delta_pct:+.1f}%')})")
    print(c("label", "per_region_mse (latest): ") + "  ".join(
        c.kv(name, f"{last['per_region_mse'][i]:.4f}") for i, name in enumerate(REGION_NAMES)
    ))
    print("   ".join([
        c.kv("gist_recall_rate", f"{gv(last, 'gist_recall_rate'):.3f}", sep=": "),
        c.kv("warm_recall_l2", f"{gv(last, 'warm_recall_l2'):.3f}", sep=": "),
        c.kv("within_session_gist_rate", f"{gv(last, 'within_session_gist_rate'):.3f}", sep=": "),
    ]))

    print()
    print(c.header("-- latest cycle timing --"))
    print("  ".join([
        c.kv("perception", f"{last.get('perception_total_ms', 0)}ms"),
        c.kv("nrem", f"{last.get('nrem_wall_ms', 0)}ms"),
        c.kv("rem", f"{last.get('rem_wall_ms', 0)}ms"),
        c.kv("snapshot", f"{last.get('snapshot_wall_ms', 0)}ms"),
        c.kv("spill", f"{last.get('spill_wall_ms', 0)}ms"),
    ]))
    print("  ".join([
        c.kv("sleep_trigger", last.get("sleep_trigger")),
        c.kv("nrem_exit_reason", last.get("nrem_exit_reason")),
        c.kv("traces_replayed", last.get("traces_replayed")),
    ]))

    print()
    print(c.header("-- disk usage --"))
    rows = [
        ("memory/warm", du("memory/warm")),
        ("memory/snapshot", du("memory/snapshot")),
        ("memory/episodes.cold", du("memory/episodes.cold")),
        (f"{campaign_dir} (metrics+logs)", du(campaign_dir)),
    ]
    for label, size in rows:
        if size:
            print(c("label", f"{label:<32}") + c.n(f"{size:>8}"))
    total = du("memory")
    if total:
        print(c("bold", c("label", f"{'memory/ (total)':<32}")) + c("bold", c.n(f"{total:>8}")))
    dfree = subprocess.run(["df", "-h", "memory"], capture_output=True, text=True)
    if dfree.returncode == 0 and len(dfree.stdout.splitlines()) > 1:
        cols = dfree.stdout.splitlines()[1].split()
        if len(cols) >= 4:
            print(c("label", "filesystem free: ") + c.n(cols[3]))

    print()
    print(c.header("-- health --"))
    logs = glob.glob(os.path.join(campaign_dir, "logs", "session_*.log"))
    print(c("label", "session logs: ") + c.n(len(logs)))
    newest = latest_session_log(campaign_dir)
    if newest:
        age = time.time() - os.path.getmtime(newest)
        is_stale = wrapper_running and age > 300
        stale_flag = c("red", "  [STALE — process may be stuck]") if is_stale else ""
        print(c("label", "newest log: ") + f"{os.path.basename(newest)} "
              + f"(last write {c.n(fmt_hms(age))} ago){stale_flag}")
    panics = scan_panics(campaign_dir)
    if panics:
        print(c("red", f"panics found: {len(panics)} — first: {panics[0][0]}: {panics[0][1][:120]}"))
    else:
        print(c("green", "panics found: 0"))
    print(c("dim", "note: SIGSEGV (exit 139, known MIGraphX destructor race) doesn't appear in per-session"))
    print(c("dim", "      logs — it's reported by the wrapper's own stdout, which this script can't see"))
    print(c("dim", "      after the launching shell exits. Check the task/nohup output for a live count."))


if __name__ == "__main__":
    main()
