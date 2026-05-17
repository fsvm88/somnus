#!/usr/bin/env python3
"""somnus status reporter.

Usage:
    python scripts/status.py [--memory-dir memory] [--metrics metrics.jsonl] [--pid-file somnus.pid] [-n 10]

Prints a formatted summary of:
  - Process (PID, uptime, CPU) if pid file is present or somnus is found via ps
  - Cold archive stats (episode count, file size)
  - Snapshot (next_id, region sizes)
  - Last N sleep cycles from the metrics NDJSON file
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

REGION_NAMES = ["AudioSpeech", "AudioEnv", "Visual", "Symbolic", "Language"]
COLD_MAGIC   = b"SMNCOLD1"
COLD_DIM     = 640
RECORD_BYTES = 8 + 8 + 4 + COLD_DIM  # 660 bytes


def fmt_size(n_bytes: int) -> str:
    if n_bytes >= 1 << 30:
        return f"{n_bytes / (1 << 30):.1f} GiB"
    if n_bytes >= 1 << 20:
        return f"{n_bytes / (1 << 20):.1f} MiB"
    if n_bytes >= 1 << 10:
        return f"{n_bytes / (1 << 10):.1f} KiB"
    return f"{n_bytes} B"


def fmt_dur(secs: float) -> str:
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ── process info ──────────────────────────────────────────────────────────────

def find_pid(pid_file: Path | None) -> int | None:
    if pid_file and pid_file.exists():
        try:
            return int(pid_file.read_text().strip())
        except (ValueError, OSError):
            pass
    # Fall back to pgrep
    try:
        out = subprocess.check_output(["pgrep", "-x", "somnus"], text=True).strip()
        pids = [int(p) for p in out.splitlines() if p.strip()]
        return pids[0] if pids else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def process_info(pid: int | None) -> str:
    if pid is None:
        return "  process: not found\n"
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return f"  process: PID {pid} not running\n"
    lines = []
    lines.append(f"  PID:    {pid}")
    # Uptime from /proc/<pid>/stat field 22 (starttime in jiffies since boot)
    try:
        stat = (proc / "stat").read_text()
        # stat format: pid (comm) state ppid ... starttime ...
        # field 22 (0-indexed=21) is starttime
        # strip the comm which may contain spaces: find last ')' then split
        after_comm = stat[stat.rfind(")") + 2:]
        fields = after_comm.split()
        starttime_jiffies = int(fields[19])  # field 22 relative to after_comm split
        hz = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", 100))
        uptime_secs_str = Path("/proc/uptime").read_text().split()[0]
        uptime_secs = float(uptime_secs_str)
        start_secs_since_boot = starttime_jiffies / hz
        age = uptime_secs - start_secs_since_boot
        lines.append(f"  uptime: {fmt_dur(age)}")
    except Exception:
        pass
    # CPU% from /proc/<pid>/stat
    try:
        stat_vals = stat[stat.rfind(")") + 2:].split()
        utime = int(stat_vals[11])
        stime = int(stat_vals[12])
        total_cpu_jiffies = utime + stime
        lines.append(f"  CPU:    {total_cpu_jiffies / hz:.0f}s CPU time")
    except Exception:
        pass
    # RSS from /proc/<pid>/status
    try:
        for line in (proc / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                lines.append(f"  RSS:    {fmt_size(kb * 1024)}")
                break
    except Exception:
        pass
    return "\n".join(lines) + "\n"


# ── cold archive ──────────────────────────────────────────────────────────────

def cold_info(memory_dir: Path) -> str:
    path = memory_dir / "episodes.cold"
    if not path.exists():
        return "  cold archive: not found\n"
    size = path.stat().st_size
    # Verify magic and compute record count
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
        if magic != COLD_MAGIC:
            return f"  cold archive: bad magic ({magic!r})\n"
        n_records = (size - 8) // RECORD_BYTES
        return f"  cold archive: {n_records:,} episodes  {fmt_size(size)}\n"
    except OSError as e:
        return f"  cold archive: {e}\n"


# ── snapshot ──────────────────────────────────────────────────────────────────

def snapshot_info(memory_dir: Path) -> str:
    snap = memory_dir / "snapshot"
    if not snap.exists():
        return "  snapshot: not found\n"
    lines = []
    next_id_path = snap / "next_id.bin"
    if next_id_path.exists():
        data = next_id_path.read_bytes()
        next_id = struct.unpack_from("<Q", data)[0]
        lines.append(f"  next_id:      {next_id:,}")
    sizes_path = snap / "region_sizes.bin"
    if sizes_path.exists():
        data = sizes_path.read_bytes()
        n = len(data) // 8
        sizes = struct.unpack_from(f"<{n}Q", data)
        parts = "  ".join(f"{REGION_NAMES[i] if i < len(REGION_NAMES) else i}={s}" for i, s in enumerate(sizes))
        lines.append(f"  region sizes: {parts}")
    weights_path = snap / "weights.safetensors"
    if weights_path.exists():
        sz = weights_path.stat().st_size
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(weights_path.stat().st_mtime))
        lines.append(f"  weights:      {fmt_size(sz)}  saved {mtime}")
    return "\n".join(lines) + "\n" if lines else "  snapshot: empty\n"


# ── metrics ───────────────────────────────────────────────────────────────────

def metrics_tail(metrics_path: Path, n: int) -> str:
    if not metrics_path.exists():
        return "  metrics file: not found\n"
    lines = []
    try:
        with open(metrics_path) as f:
            all_lines = [l.rstrip() for l in f if l.strip()]
    except OSError as e:
        return f"  metrics file: {e}\n"
    tail = all_lines[-n:] if len(all_lines) >= n else all_lines
    if not tail:
        return "  metrics file: empty\n"
    out = []
    for raw in tail:
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        cycle  = d.get("cycle", "?")
        ts     = d.get("ts", 0)
        ts_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "?"
        replayed = d.get("traces_replayed", 0)
        loss   = d.get("mean_loss", float("nan"))
        mse    = d.get("mean_output_mse", float("nan"))
        sizes  = d.get("region_sizes", [])
        grew   = "grown" if d.get("grew") else "stable"
        wake_ep = d.get("wake_episodes", 0)
        asr    = d.get("asr_count", 0)
        lang   = d.get("lang_ingest_count", 0)
        mse_wake = d.get("mean_output_mse_wake")
        mse_wake_str = f"{mse_wake:.3f}" if mse_wake is not None else "-"
        pc     = d.get("prior_cosine_mean", [None] * 5)
        pc_str = "  ".join(
            f"{REGION_NAMES[i][0]}={v:.3f}" if v is not None else f"{REGION_NAMES[i][0]}=-"
            for i, v in enumerate(pc[:5])
        )
        sizes_str = "/".join(str(s) for s in sizes) if sizes else "-"
        out.append(
            f"  [{cycle:>4}] {ts_str}  "
            f"replayed={replayed:>5}  loss={loss:.4f}  mse={mse:.3f}  "
            f"mse_wake={mse_wake_str}  wake_ep={wake_ep:>4}  "
            f"asr={asr}  lang={lang}  sizes={sizes_str}  {grew}"
        )
        if any(v is not None for v in pc):
            out.append(f"           prior_cos: {pc_str}")
    return "\n".join(out) + "\n" if out else "  (no valid records)\n"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="somnus status reporter")
    ap.add_argument("--memory-dir", default="memory", type=Path)
    ap.add_argument("--metrics",    default=None,     type=Path)
    ap.add_argument("--pid-file",   default=None,     type=Path)
    ap.add_argument("-n",           default=10,       type=int, help="last N sleep cycles to show")
    args = ap.parse_args()

    sep = "─" * 60

    print(sep)
    print("somnus status")
    print(sep)

    pid = find_pid(args.pid_file)
    print("\n[process]")
    print(process_info(pid), end="")

    print("[memory]")
    print(cold_info(args.memory_dir), end="")
    print(snapshot_info(args.memory_dir), end="")

    if args.metrics:
        metrics_path = args.metrics
    else:
        default_metrics = Path("memory/metrics.jsonl")
        metrics_path = default_metrics if default_metrics.exists() else None

    if metrics_path:
        print(f"\n[last {args.n} sleep cycles — {metrics_path}]")
        print(metrics_tail(metrics_path, args.n), end="")
    else:
        print("\n[metrics] no metrics file (pass --metrics <path> or use --metrics-file memory/metrics.jsonl)")

    print(sep)


if __name__ == "__main__":
    main()
