#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Live CLI dashboard for srt-slurm jobs.

Usage:
  srtctl monitor                        # Active + recently completed jobs
  srtctl monitor --all                  # Also include older jobs from outputs/
  srtctl monitor --outputs PATH         # Override outputs directory
  srtctl monitor --interval 10          # 10s refresh (default: 5)
  srtctl monitor --once                 # Print once and exit

Keybindings (live mode):
  c   Toggle between last concurrency and all concurrencies in Metrics column
  q   Quit
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from itertools import accumulate
from pathlib import Path

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import select as _select
    import termios
    import tty

    _HAS_TTY = True
except ImportError:
    _HAS_TTY = False

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _get_term_height() -> int:
    """/dev/tty works over SSH where stdin/stdout may not be a real PTY."""
    try:
        import fcntl
        import struct

        with open("/dev/tty") as tty:
            rows, _ = struct.unpack("hh", fcntl.ioctl(tty, termios.TIOCGWINSZ, b"\x00\x00\x00\x00"))
        if 10 <= rows <= 500:
            return rows
    except Exception:
        pass
    return 60


# ─── Stage detection ──────────────────────────────────────────────────────────

_IN_PROGRESS_PATTERNS: list[tuple[str, str, str, str]] = [
    (r"Server is healthy - starting benchmark|Running \S+ benchmark", "benchmarking", "Benchmarking", "bright_green"),
    (r"Polling http://|Model is not ready|Waiting for server health", "awaiting_workers", "Awaiting Workers", "cyan"),
    (r"Starting frontend", "frontend", "Starting Frontend", "cyan"),
    (r"Starting backend workers", "workers", "Starting Workers", "yellow"),
    (r"etcd is ready|NATS is ready", "head_ready", "Head Ready", "yellow"),
    (r"Starting infrastructure services", "infra", "Starting Infra", "dim yellow"),
    (r"Sweep Orchestrator$", "starting", "Starting", "dim"),
]

_WORKER_RE = re.compile(
    r"waiting for (-?\d+) prefills? and (-?\d+) decodes?\. Have (\d+) prefills? and (\d+) decodes?",
    re.IGNORECASE,
)
_WORKER_READY_RE = re.compile(
    r"Model is ready\. Have (\d+) prefills? and (\d+) decodes?",
    re.IGNORECASE,
)

_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

_BENCH_PATTERNS: list[tuple[str, str]] = [
    (r"output.{0,20}throughput[:\s=]+(\d[\d,]*\.?\d*)\s*tok", "throughput_toks"),
    (r"request.{0,10}throughput[:\s=]+(\d[\d,]*\.?\d*)\s*req", "request_throughput"),
    (r"ttft.*?mean[:\s=]+(\d[\d,]*\.?\d*)\s*ms", "ttft_mean_ms"),
    (r"tpot.*?mean[:\s=]+(\d[\d,]*\.?\d*)\s*ms", "tpot_mean_ms"),
]

_ACTIVE_STATES = {"RUNNING", "PENDING", "COMPLETING"}
_TERMINAL_STAGES = {"completed", "finalizing", "failed", "killed", "timeout"}
_IN_PROGRESS_STAGES = {"benchmarking", "awaiting_workers", "frontend", "workers", "head_ready", "infra", "starting"}
_PANEL_NAMES = ["sweep", "worker", "bench"]

# Fixed width for the concurrency field so rows align across concurrencies.
_CONC_FIELD = 14

# Pre-compiled for _colorize_log (called on every render frame).
_LOG_ERR_RE = re.compile(r"✗|Sweep failed|Benchmark failed|Error during sweep:")
_LOG_OK_RE = re.compile(r"✓|completed successfully", re.IGNORECASE)


# ─── Mutable dashboard state ──────────────────────────────────────────────────


class _State:
    def __init__(self) -> None:
        self.show_all_concurrencies: bool = False
        self.show_all_jobs: bool = False
        self.seen_job_ids: set[str] = set()
        self.scroll_offset: int = 0
        self.selected_idx: int = 0
        self.detail_job_id: str | None = None
        self.detail_sweep_lines: list[str] = []
        self.detail_worker_files: list[Path] = []
        self.detail_worker_idx: int = 0
        self.detail_worker_lines: list[str] = []
        self.detail_auto_refresh: bool = False
        self.detail_panel_idx: int = 0  # 0=sweep 1=worker 2=bench
        self.detail_panel_active: bool = False
        self.detail_bench_sections: list[tuple[int | None, list[str]]] = []
        self.detail_bench_section_idx: int = 0
        self.delete_confirm_job_id: str | None = None
        self.cancel_confirm_job_id: str | None = None


# ─── Data gathering ────────────────────────────────────────────────────────────


def _squeue_jobs() -> dict[str, dict]:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    try:
        result = subprocess.run(
            ["squeue", "-u", user, "--format=%i|%j|%T|%M|%N|%P", "--noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    jobs: dict[str, dict] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 6:
            continue
        job_id, name, state, elapsed, nodelist, partition = (p.strip() for p in parts[:6])
        jobs[job_id] = {"name": name, "state": state, "elapsed": elapsed, "nodelist": nodelist, "partition": partition}
    return jobs


def _read_all_lines(path: Path) -> list[str]:
    try:
        with open(path, "rb") as fh:
            data = fh.read().decode("utf-8", errors="replace")
        return [ln.rstrip() for ln in data.splitlines()]
    except OSError:
        return []


def _tail(path: Path, n: int = 150) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - n * 120))
            data = fh.read().decode("utf-8", errors="replace")
        return [ln.rstrip() for ln in data.splitlines()[-n:]]
    except OSError:
        return []


def _detect_stage(log_lines: list[str]) -> tuple[str, str, str]:
    text = "\n".join(log_lines)
    has_cleanup = bool(re.search(r"\[INFO\] Cleanup$", text, re.MULTILINE))
    has_done = bool(re.search(r"Benchmark completed successfully", text))
    has_fail = bool(re.search(r"Benchmark failed with exit code|Error during sweep:|Sweep failed \(exit code", text))

    if has_cleanup:
        if has_done:
            return "completed", "Completed", "bright_green"
        if has_fail:
            return "failed", "Failed", "red"
        return "finalizing", "Finalizing", "dim cyan"
    if has_done:
        return "completed", "Completed", "bright_green"
    if has_fail:
        return "failed", "Failed", "red"

    for line in reversed(log_lines):
        for pattern, stage_id, label, color in _IN_PROGRESS_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                return stage_id, label, color

    return "unknown", "No Log Yet", "dim"


def _worker_progress(log_lines: list[str]) -> str:
    for line in reversed(log_lines):
        m = _WORKER_READY_RE.search(line)
        if m:
            p, d = int(m.group(1)), int(m.group(2))
            return f"{p}/{p}P  {d}/{d}D"
        m = _WORKER_RE.search(line)
        if m:
            have_p, have_d = int(m.group(3)), int(m.group(4))
            want_p = have_p + max(0, int(m.group(1)))
            want_d = have_d + max(0, int(m.group(2)))
            return f"{have_p}/{want_p}P  {have_d}/{want_d}D"
    return ""


def _rollup_runs(log_dir: Path) -> list[dict] | None:
    rollup = log_dir / "benchmark-rollup.json"
    try:
        data = json.loads(rollup.read_text())
        runs = [r for r in (data.get("runs") or []) if r]
        return runs or None
    except Exception:
        return None


def _concurrency_from_path(p: Path) -> int:
    m = re.search(r"results_concurrency_(\d+)", p.name)
    return int(m.group(1)) if m else 0


_RESULT_NUM_RE = r'"{key}"\s*:\s*([\d.eE+\-]+)'


def _read_result_fast(path: Path) -> dict:
    """Read only summary fields from a (potentially huge) sa-bench result JSON.

    sa-bench embeds per-request arrays that can reach 100s of MB.  Summary
    fields we need are split: throughput fields sit in the first ~512 bytes,
    latency fields are written at the end.  Reading head+tail costs <5 KB
    instead of 100-300 MB per file.
    """

    def _get(text: str, key: str) -> float | None:
        m = re.search(_RESULT_NUM_RE.format(key=key), text)
        return float(m.group(1)) if m else None

    with open(path, "rb") as f:
        head = f.read(512).decode("utf-8", errors="replace")
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        tail = f.read().decode("utf-8", errors="replace")

    return {
        "concurrency": _get(head, "max_concurrency"),
        "throughput_toks": _get(head, "output_throughput"),
        "request_throughput": _get(head, "request_throughput"),
        "ttft_mean_ms": _get(tail, "mean_ttft_ms"),
        "tpot_mean_ms": _get(tail, "mean_tpot_ms"),
    }


_BENCH_SECTION_RE = re.compile(
    r"Maximum\s+request\s+concurr[a-z]*\s*[=:]\s*(\d+)",
    re.IGNORECASE,
)
# Matches the "---...---" separator bench.sh echoes after each real run.
# Pure-dash lines only — benchmark_serving.py uses "---text---" headers with text, not this.
_BENCH_SEP_RE = re.compile(r"^-{5,}$")


def _split_bench_sections(lines: list[str]) -> list[tuple[int, list[str]]]:
    """Split benchmark.out into per-concurrency sections, keeping the real run.

    bench.sh runs each concurrency twice: warmup first, real benchmark second.
    benchmark_serving.py prints "Maximum request concurrency: N" at the start of
    each run, so every concurrency appears at least twice as a section marker.
    We keep only the last occurrence (the real run) and discard all prior ones
    (warmup).

    Each kept section is trimmed at the pure-dash separator that bench.sh echoes
    after the real run, preventing warmup output from the next concurrency from
    bleeding in.

    Falls back to a single section with all lines when no markers are found.
    """
    all_sections: list[tuple[int | None, list[str]]] = []
    current_conc: int | None = None
    current: list[str] = []

    for line in lines:
        m = _BENCH_SECTION_RE.search(line)
        if m:
            conc = int(m.group(1))
            if current or current_conc is not None:
                all_sections.append((current_conc, current))
            current_conc = conc
            current = [line]
        else:
            current.append(line)

    if current or current_conc is not None:
        all_sections.append((current_conc, current))

    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    for i, (conc, _) in enumerate(all_sections):
        if conc is None:
            continue
        if conc not in first_seen:
            first_seen[conc] = i
        last_seen[conc] = i

    if not first_seen:
        return [(None, lines)]  # type: ignore[return-value]

    # first_seen is insertion-ordered (Python 3.7+), already in first-appearance order.
    def _trim(sec_lines: list[str]) -> list[str]:
        for i, ln in enumerate(sec_lines):
            if _BENCH_SEP_RE.match(ln):
                return sec_lines[: i + 1]
        return sec_lines

    return [(c, _trim(all_sections[last_seen[c]][1])) for c in first_seen]


def _partial_runs(log_dir: Path) -> list[dict] | None:
    """Read individual sa-bench result files written per concurrency during a live run."""
    result_files = list(log_dir.glob("sa-bench_*/results_*.json"))
    if not result_files:
        return None
    result_files.sort(key=_concurrency_from_path)
    runs = []
    for f in result_files:
        try:
            runs.append(_read_result_fast(f))
        except Exception:
            continue
    return runs or None


def _live_metrics(log_dir: Path) -> dict | None:
    lines = _tail(log_dir / "benchmark.out", 100)
    if not lines:
        return None
    metrics: dict[str, float] = {}
    for line in reversed(lines):
        for pattern, key in _BENCH_PATTERNS:
            if key not in metrics:
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    with contextlib.suppress(ValueError):
                        metrics[key] = float(m.group(1).replace(",", ""))
        if len(metrics) >= 3:
            break
    return metrics or None


def _mtime_age_str(path: Path) -> str:
    try:
        s = int(time.time() - path.stat().st_mtime)
        if s < 60:
            return f"{s}s ago"
        if s < 3600:
            return f"{s // 60}m ago"
        return f"{s // 3600}h ago"
    except OSError:
        return ""


def _gather_job_info(job_id: str, outputs_dir: Path, sq: dict | None) -> dict:
    job_dir = outputs_dir / job_id
    logs_dir = job_dir / "logs"
    sweep_log = logs_dir / f"sweep_{job_id}.log"

    info: dict = {
        "job_id": job_id,
        "name": (sq or {}).get("name", job_id),
        "slurm_state": (sq or {}).get("state", "ENDED"),
        "elapsed": (sq or {}).get("elapsed", "—"),
        "stage_id": "unknown",
        "stage_label": "—",
        "stage_color": "dim",
        "worker_progress": "",
        "runs": None,
        "live_metrics": None,
        "gpu_info": "",
        "bench_config": "",
        "log_age": "",
    }

    try:
        meta = json.loads((job_dir / f"{job_id}.json").read_text())
        if meta.get("job_name"):
            info["name"] = meta["job_name"]
        res = meta.get("resources", {})
        gpu = res.get("gpu_type", "")
        p, d, gpn = res.get("prefill_nodes", 0), res.get("decode_nodes", 0), res.get("gpus_per_node", 0)
        prec = meta.get("model", {}).get("precision", "")
        info["gpu_info"] = f"{gpu}  {p}P/{d}D×{gpn}  {prec}".strip()
        bench = meta.get("benchmark", {})
        isl, osl, btype = bench.get("isl"), bench.get("osl"), bench.get("type", "")
        info["bench_config"] = f"{btype}  {isl}→{osl}" if (isl and osl) else btype
    except Exception:
        pass

    if not job_dir.exists():
        info["stage_label"] = "No Output Dir"
        return info

    if info["slurm_state"] == "TIMEOUT":
        info.update(stage_id="timeout", stage_label="Timed Out", stage_color="red")

    if sweep_log.exists():
        log_lines = _tail(sweep_log)
        info["log_age"] = _mtime_age_str(sweep_log)

        if info["stage_id"] != "timeout":
            stage_id, label, color = _detect_stage(log_lines)
            info.update(stage_id=stage_id, stage_label=label, stage_color=color)

            if info["slurm_state"] == "ENDED" and stage_id in _IN_PROGRESS_STAGES:
                info.update(stage_id="killed", stage_label="Killed", stage_color="red")

        info["worker_progress"] = _worker_progress(log_lines)

        if info["stage_id"] in ("benchmarking", "completed", "finalizing", "killed", "failed"):
            info["runs"] = _rollup_runs(logs_dir) or _partial_runs(logs_dir)
            if not info["runs"]:
                info["live_metrics"] = _live_metrics(logs_dir)
    else:
        info["stage_label"] = "Initializing"
        info["stage_color"] = "yellow"

    return info


def _job_sort_key(j: dict) -> tuple:
    slurm_order = {"RUNNING": 0, "PENDING": 1, "COMPLETING": 2}.get(j["slurm_state"], 3)
    done_order = {"completed": 0, "finalizing": 1, "failed": 2}.get(j["stage_id"], 3)
    return (slurm_order, done_order, -int(j["job_id"]))


def _gather_all(outputs_dir: Path, include_all: bool, seen_job_ids: set[str]) -> list[dict]:
    sq_jobs = _squeue_jobs()
    seen_job_ids.update(sq_jobs.keys())
    job_ids = set(seen_job_ids)

    if include_all and outputs_dir.is_dir():
        for d in outputs_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                job_ids.add(d.name)

    jobs = [_gather_job_info(jid, outputs_dir, sq_jobs.get(jid)) for jid in job_ids]

    for j in jobs:
        if j["job_id"] in seen_job_ids and j["stage_id"] in _TERMINAL_STAGES:
            j["worker_progress"] = ""

    jobs.sort(key=_job_sort_key)
    return jobs


# ─── Rendering ────────────────────────────────────────────────────────────────

_STATE_COLORS = {
    "RUNNING": "green",
    "PENDING": "yellow",
    "COMPLETING": "cyan",
    "FAILED": "red",
    "CANCELLED": "dim red",
    "TIMEOUT": "red",
    "ENDED": "dim",
}


def _format_run(r: dict) -> str:
    parts: list[str] = []
    if (c := r.get("concurrency")) is not None:
        parts.append(f"c={int(c):<{_CONC_FIELD - 2}}")
    if t := r.get("throughput_toks"):
        parts.append(f"{t:>8,.0f} tok/s")
    if ttft := r.get("ttft_mean_ms"):
        parts.append(f"TTFT {ttft:>5.0f}ms")
    if tpot := r.get("tpot_mean_ms"):
        parts.append(f"TPOT {tpot:>4.1f}ms")
    return "    ".join(parts)


def _running_line(live: dict | None) -> Text:
    t = Text()
    if live:
        parts: list[str] = []
        if v := live.get("throughput_toks"):
            parts.append(f"{v:>8,.0f} tok/s")
        if v := live.get("ttft_mean_ms"):
            parts.append(f"TTFT {v:>5.0f}ms")
        if parts:
            t.append("    ".join(parts) + "    ", style="dim")
    t.append("running…", style="dim")
    return t


def _metrics_multiline(j: dict, show_all: bool) -> bool:
    """True when the metrics cell spans multiple lines (drives separator and height estimate)."""
    runs = j.get("runs") or []
    benchmarking = j.get("stage_id") == "benchmarking"
    actual_count = len(runs)
    return show_all and (actual_count > 1 or (benchmarking and bool(runs)))


def _metrics_cell(j: dict, show_all: bool, clip_rows: int | None = None) -> str | Text:
    runs: list[dict] | None = j["runs"]
    live: dict | None = j["live_metrics"]
    benchmarking = j["stage_id"] == "benchmarking"

    if runs:
        actual = sorted(runs, key=lambda r: int(r.get("concurrency") or 0))
        if show_all:
            display = actual[-clip_rows:] if clip_rows and clip_rows < len(actual) else actual
        else:
            display = [actual[-1]]
        t = Text("\n".join(_format_run(r) for r in display))
        if benchmarking:
            t.append("\n")
            t.append_text(_running_line(live))
        return t

    if live:
        parts: list[str] = []
        if v := live.get("throughput_toks"):
            parts.append(f"{v:,.0f} tok/s")
        if v := live.get("ttft_mean_ms"):
            parts.append(f"TTFT {v:.0f}ms")
        if v := live.get("tpot_mean_ms"):
            parts.append(f"TPOT {v:.1f}ms")
        return "  ".join(parts) if parts else "-"

    if benchmarking:
        return Text("running…", style="dim")

    return "-"


def _pad_to_height(text: Text, height: int) -> None:
    """Pad text with blank lines to fill the allocated panel height, preventing stale artifacts."""
    n = text.plain.count("\n")
    if n < height:
        text.append("\n" * (height - n))


def _colorize_log(lines: list[str], empty_msg: str = "(empty)", max_width: int = 0) -> Text:
    if not lines:
        return Text(empty_msg, style="dim")
    t = Text()
    for line in lines:
        line = _ANSI_RE.sub("", line)
        if max_width and len(line) > max_width:
            line = line[: max_width - 1] + "…"
        if _LOG_ERR_RE.search(line):
            t.append(line + "\n", style="bold red")
        elif "[ERROR]" in line:
            t.append(line + "\n", style="red")
        elif "[WARNING]" in line:
            t.append(line + "\n", style="yellow")
        elif _LOG_OK_RE.search(line):
            t.append(line + "\n", style="green")
        elif "[INFO]" in line:
            t.append(line + "\n", style="dim")
        else:
            t.append(line + "\n")
    return t


def _close_detail(state: _State) -> None:
    state.detail_job_id = None
    state.detail_sweep_lines = []
    state.detail_worker_files = []
    state.detail_worker_idx = 0
    state.detail_worker_lines = []
    state.detail_bench_sections = []
    state.detail_bench_section_idx = 0
    state.detail_panel_active = False


def _refresh_detail_lines(state: _State, jid: str, logs_dir: Path) -> None:
    state.detail_sweep_lines = _tail(logs_dir / f"sweep_{jid}.log", 100)
    all_bench = _read_all_lines(logs_dir / "benchmark.out")
    sections = _split_bench_sections(all_bench)
    state.detail_bench_sections = sections
    state.detail_bench_section_idx = min(state.detail_bench_section_idx, max(0, len(sections) - 1))
    if state.detail_worker_files:
        state.detail_worker_lines = _tail(state.detail_worker_files[state.detail_worker_idx], 100)


def _detail_log_path(
    jid: str, logs_dir: Path, worker_files: list[Path], worker_idx: int, panel_idx: int
) -> Path | None:
    paths: list[Path | None] = [
        logs_dir / f"sweep_{jid}.log",
        worker_files[worker_idx] if worker_files else None,
        logs_dir / "benchmark.out",
    ]
    return paths[panel_idx]


def _detail_view(job_id: str, jobs: list[dict], state: _State, term_height: int = 50, term_cols: int = 200) -> Layout:
    """Two-column detail layout: sweep (left) | worker/benchmark stacked (right)."""
    job_info = next((j for j in jobs if j["job_id"] == job_id), None)
    name = job_info["name"] if job_info else job_id
    stage_label = job_info["stage_label"] if job_info else "?"
    stage_color = job_info["stage_color"] if job_info else "dim"

    job_title = (
        f"[bold cyan]{job_id}[/bold cyan] [dim cyan]{name}[/dim cyan] [{stage_color}]{stage_label}[/{stage_color}]"
    )

    content_h = term_height - 1
    half_h = content_h // 2
    sweep_max = max(1, content_h - 2)
    worker_max = max(1, half_h - 2)
    bench_max = max(1, content_h - half_h - 2)
    panel_w = max(40, term_cols // 2 - 6)

    _open_hint = r"  [bold white]\[↵][/bold white]"

    sweep_focused = state.detail_panel_active and state.detail_panel_idx == 0
    sweep_text = _colorize_log(state.detail_sweep_lines[-sweep_max:], "No sweep log.", max_width=panel_w)
    _pad_to_height(sweep_text, sweep_max)
    sweep_panel = Panel(
        sweep_text,
        title=f"{job_title} · [bold]sweep[/bold]" + (_open_hint if sweep_focused else ""),
        border_style="white" if sweep_focused else "cyan",
    )

    worker_focused = state.detail_panel_active and state.detail_panel_idx == 1
    if state.detail_worker_files:
        wname = state.detail_worker_files[state.detail_worker_idx].stem
        n_w = len(state.detail_worker_files)
        worker_title = f"[bold]worker[/bold] [steel_blue]{wname} [{state.detail_worker_idx + 1}/{n_w}][/steel_blue]"
        worker_text = _colorize_log(state.detail_worker_lines[-worker_max:], "(empty)", max_width=panel_w)
    else:
        worker_title = "[bold]workers[/bold] [steel_blue](none found)[/steel_blue]"
        worker_text = Text("No worker logs found.", style="dim")
    _pad_to_height(worker_text, worker_max)
    worker_panel = Panel(
        worker_text,
        title=worker_title + (_open_hint if worker_focused else ""),
        border_style="white" if worker_focused else "blue",
    )

    bench_focused = state.detail_panel_active and state.detail_panel_idx == 2
    sections = state.detail_bench_sections
    if sections:
        conc, sec_lines = sections[state.detail_bench_section_idx]
        bench_content = _colorize_log(sec_lines[-bench_max:], "(empty)", max_width=panel_w)
        conc_label = f"c={conc}" if conc is not None else "log"
        bench_nav = f"[dim steel_blue] [{state.detail_bench_section_idx + 1}/{len(sections)}][/dim steel_blue]"
        bench_title_str = f"[bold]benchmark[/bold] · [cyan]{conc_label}[/cyan]{bench_nav}"
    else:
        bench_content = Text("No benchmark output.", style="dim")
        bench_title_str = "[bold]benchmark[/bold]"
    _pad_to_height(bench_content, bench_max)
    bench_panel = Panel(
        bench_content,
        title=bench_title_str + (_open_hint if bench_focused else ""),
        border_style="white" if bench_focused else "green",
    )

    right = Layout(name="right")
    right.split_column(
        Layout(worker_panel, name="worker", ratio=1),
        Layout(bench_panel, name="bench", ratio=1),
    )
    cols = Layout()
    cols.split_row(Layout(sweep_panel, name="sweep", ratio=1), right)
    return cols


def _build_table(jobs: list[dict], show_all: bool, selected_rel: int = -1, last_job_clip: int | None = None) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
        show_edge=False,
        expand=True,
    )

    table.add_column("Job ID", width=11, no_wrap=True)
    table.add_column("Name", min_width=24, max_width=44, no_wrap=True)
    table.add_column("Slurm", width=10, no_wrap=True)
    table.add_column("Stage", width=22, no_wrap=True)
    table.add_column("Workers", width=13, no_wrap=True)
    table.add_column("Time", width=8, no_wrap=True)
    table.add_column("Config", width=26, no_wrap=True)
    table.add_column("Metrics", ratio=1, no_wrap=True)

    for i, j in enumerate(jobs):
        slurm_state = j["slurm_state"]
        state_color = _STATE_COLORS.get(slurm_state, "white")

        stage_txt = Text(j["stage_label"], style=j["stage_color"])
        if j["stage_id"] in ("completed", "failed", "finalizing") and j["log_age"]:
            stage_txt.append(f"  ({j['log_age']})", style="dim")

        cfg_parts = [p for p in (j["gpu_info"], j["bench_config"]) if p]

        job_id_cell = Text()
        if i == selected_rel:
            job_id_cell.append("▶ ", style="bold cyan")
            job_id_cell.append(j["job_id"], style="bold white")
        else:
            job_id_cell.append("  ", style="dim")
            job_id_cell.append(j["job_id"], style="bold white")

        table.add_row(
            job_id_cell,
            j["name"][:49],
            Text(slurm_state, style=state_color),
            stage_txt,
            Text(j["worker_progress"], style="dim cyan"),
            j["elapsed"],
            "\n".join(cfg_parts),
            _metrics_cell(j, show_all, clip_rows=last_job_clip if i == 0 else None),
            end_section=_metrics_multiline(j, show_all),
        )

    return table


_PANEL_OVERHEAD = 4  # panel top border + table header + table separator + panel bottom border
# The keybinding bar is pinned via Layout (always 1 line); term_height - 1 goes to the panel.


def _job_row_height(j: dict, show_all: bool) -> int:
    cfg_lines = len([p for p in (j.get("gpu_info", ""), j.get("bench_config", "")) if p]) or 1
    runs = j.get("runs")
    benchmarking = j.get("stage_id") == "benchmarking"
    actual_count = len(runs) if runs else 0
    if show_all and actual_count >= 1:
        metrics_lines = actual_count + (1 if benchmarking else 0)
    elif benchmarking and runs:
        metrics_lines = 2
    else:
        metrics_lines = 1
    return max(cfg_lines, metrics_lines) + (1 if _metrics_multiline(j, show_all) else 0)


def _compute_viewport(
    jobs: list[dict], selected_idx: int, scroll_offset: int, available_lines: int, show_all: bool
) -> tuple[int, int, int, int]:
    """Cursor-following scroll viewport.

    Returns (scroll_offset, start_idx, selected_idx, n_below).
    """
    n = len(jobs)
    if n == 0:
        return 0, 0, 0, 0

    selected_idx = max(0, min(selected_idx, n - 1))
    heights = [_job_row_height(j, show_all) for j in jobs]
    prefix = list(accumulate(heights))

    sel_top = prefix[selected_idx - 1] if selected_idx > 0 else 0
    sel_bot = prefix[selected_idx]
    if sel_top < scroll_offset:
        scroll_offset = sel_top
    elif sel_bot > scroll_offset + available_lines:
        # If taller than viewport, pin its top; otherwise scroll up to show bottom.
        scroll_offset = sel_top if sel_bot - sel_top >= available_lines else sel_bot - available_lines

    max_scroll = max(0, prefix[-1] - available_lines)
    scroll_offset = max(0, min(scroll_offset, max_scroll))

    start_idx = min(bisect.bisect_right(prefix, scroll_offset), n - 1)
    base = prefix[start_idx - 1] if start_idx > 0 else 0
    end_idx = bisect.bisect_left(prefix, base + available_lines, start_idx)
    n_below = max(0, n - end_idx - 1)

    return scroll_offset, start_idx, selected_idx, n_below


def _render(
    jobs: list[dict] | None,
    outputs_dir: Path,
    interval: float,
    state: _State,
    loading: bool = False,
    spin_idx: int = 0,
    term_height: int = 50,
    term_cols: int = 200,
) -> tuple[Layout, int, int]:
    def _kb(bar: Text, key: str, label: str, active: bool = False) -> None:
        bar.append(f" {key} ", style="bold black on cyan" if active else "bold black on white")
        bar.append(f" {label}  ", style="bright_white" if active else "dim")

    jobs = jobs or []

    if state.detail_job_id is not None:
        cols = _detail_view(state.detail_job_id, jobs, state, term_height=term_height, term_cols=term_cols)
        wfiles = state.detail_worker_files
        bsections = state.detail_bench_sections
        bar = Text(" ")
        _kb(
            bar,
            "↑↓",
            f"panel [{_PANEL_NAMES[state.detail_panel_idx]}]" if state.detail_panel_active else "select panel",
        )
        if state.detail_panel_active:
            if state.detail_panel_idx == 1 and wfiles:
                _kb(bar, "←→", f"worker [{state.detail_worker_idx + 1}/{len(wfiles)}]")
            elif state.detail_panel_idx == 2 and bsections:
                _kb(bar, "←→", f"concurrency [{state.detail_bench_section_idx + 1}/{len(bsections)}]")
        _kb(bar, "↵", "open log")
        _kb(bar, "r", "refresh", active=state.detail_auto_refresh)
        _kb(bar, "ESC", "back")
        _kb(bar, "q", "quit")
        layout = Layout()
        layout.split_column(
            Layout(cols, name="content", ratio=1),
            Layout(bar, name="bar", size=1),
        )
        return layout, state.scroll_offset, state.selected_idx

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_below = 0
    start_idx = 0
    scroll_offset = state.scroll_offset
    selected_idx = state.selected_idx

    if loading and not jobs:
        spin_char = _SPINNER[spin_idx % len(_SPINNER)]
        subtitle_parts = [f"{spin_char} loading", str(outputs_dir), f"every {interval}s"]
        content: Table | Text = Text(f"{spin_char} Loading…", style="dim")
    else:
        available = max(1, term_height - 1 - _PANEL_OVERHEAD)
        scroll_offset, start_idx, selected_idx, n_below = _compute_viewport(
            jobs, state.selected_idx, state.scroll_offset, available, state.show_all_concurrencies
        )

        state_counts = Counter(j["slurm_state"] for j in jobs)
        n_running = state_counts["RUNNING"]
        n_pending = state_counts["PENDING"]
        n_done = sum(v for k, v in state_counts.items() if k not in _ACTIVE_STATES)

        counts = []
        if n_running:
            counts.append(f"{n_running} running")
        if n_pending:
            counts.append(f"{n_pending} pending")
        if n_done:
            counts.append(f"{n_done} done")

        spin_char = _SPINNER[spin_idx % len(_SPINNER)]
        refresh_indicator = f"{spin_char} refreshing" if loading else f"every {interval}s"
        subtitle_parts = (counts or []) + [str(outputs_dir), refresh_indicator]
        display_jobs = jobs[start_idx:]
        selected_rel = selected_idx - start_idx

        first_job_clip = None
        if display_jobs and state.show_all_concurrencies and selected_rel != 0:
            first_j = display_jobs[0]
            actual_count = len(first_j.get("runs") or [])
            if actual_count > 1:
                H1 = _job_row_height(first_j, state.show_all_concurrencies)
                remaining = available - H1
                lines_for_first = H1
                S = 0
                for j in display_jobs[1:]:
                    h = _job_row_height(j, state.show_all_concurrencies)
                    prev_S = S
                    S += h
                    if prev_S < remaining < S:
                        lines_for_first = max(1, H1 - (S - remaining))
                        break
                    if remaining <= S:
                        break
                sep = 1 if _metrics_multiline(first_j, state.show_all_concurrencies) else 0
                clip = lines_for_first - sep
                if 0 < clip < actual_count:
                    first_job_clip = clip

        content = (
            _build_table(
                display_jobs, state.show_all_concurrencies, selected_rel=selected_rel, last_job_clip=first_job_clip
            )
            if display_jobs
            else Text("No jobs found.", style="dim")
        )

    panel = Panel(
        content,
        title=f"[bold cyan]srt-slurm[/bold cyan]  [dim]{now}[/dim]",
        subtitle="[dim]" + "  ·  ".join(subtitle_parts) + "[/dim]",
        border_style="cyan",
    )

    bar = Text(" ")
    if state.cancel_confirm_job_id is not None:
        bar.append(f"  Shutdown {state.cancel_confirm_job_id}? ", style="bold yellow")
        bar.append(" y ", style="bold black on yellow")
        bar.append(" yes  ", style="dim")
        bar.append(" n ", style="bold black on white")
        bar.append(" no  ", style="dim")
    elif state.delete_confirm_job_id is not None:
        bar.append(f"  Delete {state.delete_confirm_job_id}? ", style="bold red")
        bar.append(" y ", style="bold black on red")
        bar.append(" yes  ", style="dim")
        bar.append(" n ", style="bold black on white")
        bar.append(" no  ", style="dim")
    else:
        _kb(bar, "↑↓", "navigate")
        _kb(bar, "↵", "open")
        _kb(bar, "y", "yaml")
        _kb(bar, "d", "delete/shutdown")
        _kb(
            bar,
            "c",
            "all concurrencies" if state.show_all_concurrencies else "last concurrency",
            active=state.show_all_concurrencies,
        )
        _kb(bar, "a", "all jobs" if state.show_all_jobs else "active jobs", active=state.show_all_jobs)
        _kb(bar, "q", "quit")

    n_above = start_idx
    if n_above > 0 or n_below > 0:
        bar.append("   ")
        if n_above:
            bar.append(f"↑ {n_above}  ", style="dim")
        if n_below:
            bar.append(f"↓ {n_below}", style="dim")

    layout = Layout()
    layout.split_column(
        Layout(panel, name="panel", ratio=1),
        Layout(bar, name="bar", size=1),
    )
    return layout, scroll_offset, selected_idx


# ─── Main ─────────────────────────────────────────────────────────────────────


def _find_outputs_dir() -> Path:
    cwd = Path.cwd()
    if (cwd / "outputs").is_dir():
        return cwd / "outputs"
    parent = cwd.parent
    if (parent / "outputs").is_dir():
        return parent / "outputs"
    return cwd / "outputs"


def _session_path() -> Path:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    return Path(f"/tmp/srt-dash-{user}.json")


def _save_session(session_file: Path, key: str, outputs_dir: Path, job_ids: set[str]) -> None:
    try:
        all_sessions: dict = {}
        if session_file.exists():
            with contextlib.suppress(Exception):
                all_sessions = json.loads(session_file.read_text())
        all_sessions[key] = {
            "outputs_dir": str(outputs_dir.resolve()),
            "job_ids": sorted(job_ids),
        }
        session_file.write_text(json.dumps(all_sessions))
    except Exception:
        pass


def _load_session(session_file: Path, key: str) -> tuple[set[str], Path | None]:
    try:
        if session_file.exists():
            entry = json.loads(session_file.read_text()).get(key, {})
            if entry:
                return set(entry.get("job_ids", [])), Path(entry["outputs_dir"])
    except Exception:
        pass
    return set(), None


def _execute(args: argparse.Namespace) -> None:
    session_file = _session_path()

    if args.clear_sessions:
        if session_file.exists():
            session_file.unlink()
            print(f"Cleared session cache: {session_file}")
        else:
            print(f"No session cache found at {session_file}")
        return

    outputs_dir = args.outputs or _find_outputs_dir()
    try:
        term_cols = os.get_terminal_size().columns
    except OSError:
        term_cols = shutil.get_terminal_size(fallback=(200, 50)).columns

    state = _State()
    state.show_all_jobs = args.all
    key = ""
    if not args.once:
        if args.resume:
            key = args.resume
            job_ids, saved_dir = _load_session(session_file, key)
            if saved_dir and args.outputs is None:
                outputs_dir = saved_dir
            state.seen_job_ids.update(job_ids)
        else:
            key = uuid.uuid4().hex[:12]

    if args.once:
        console = Console(width=max(term_cols, 160))
        jobs = _gather_all(outputs_dir, state.show_all_jobs, state.seen_job_ids)
        layout, _, _ = _render(
            jobs, outputs_dir, args.interval, state, term_height=_get_term_height(), term_cols=term_cols
        )
        console.print(layout)
        return

    # force_terminal=True ensures Rich renders escape codes even if stdout
    # detection is ambiguous (tmux, remote sessions, etc.)
    console = Console(width=max(term_cols, 160), force_terminal=True, force_jupyter=False)

    _cache_lock = threading.Lock()
    _cached_jobs: list[dict] | None = None
    _is_loading = True
    _fetch_trigger = threading.Event()
    _stop = threading.Event()

    def _fetch_loop() -> None:
        nonlocal _cached_jobs, _is_loading
        while not _stop.is_set():
            try:
                jobs = _gather_all(outputs_dir, state.show_all_jobs, state.seen_job_ids)
                with _cache_lock:
                    _cached_jobs = jobs
                    _is_loading = False
            except Exception:
                with _cache_lock:
                    _is_loading = False  # preserve stale _cached_jobs on error
            _fetch_trigger.clear()
            _fetch_trigger.wait(timeout=args.interval)

    fetch_thread = threading.Thread(target=_fetch_loop, daemon=True)
    fetch_thread.start()

    # Set up terminal for single-keypress reading in the main thread.
    # Doing this here (not in a daemon thread) guarantees the finally block
    # always restores echo, even on Ctrl-C or normal exit.
    use_tty = _HAS_TTY and sys.stdin.isatty()
    fd = sys.stdin.fileno() if use_tty else -1
    old_term = termios.tcgetattr(fd) if use_tty else None

    try:
        if use_tty:
            tty.setcbreak(fd)

        with Live(console=console, refresh_per_second=4, screen=True) as live:

            def _launch_vim(path: Path) -> None:
                live.stop()
                if use_tty and old_term is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
                subprocess.run(["vim", str(path)])
                if use_tty:
                    tty.setcbreak(fd)
                live.start()

            spin_idx = 0
            last_render = 0.0
            last_detail_refresh = 0.0
            term_height = _get_term_height()
            last_height_check = time.monotonic()
            quit_requested = False

            while not quit_requested:
                if use_tty and _select.select([sys.stdin], [], [], 0)[0]:
                    # os.read bypasses Python buffering; 6 bytes covers single chars, 3-byte arrows.
                    raw = os.read(fd, 6)

                    if state.detail_job_id is not None:
                        if raw[:3] in (b"\x1b[A", b"\x1b[B"):
                            if not state.detail_panel_active:
                                state.detail_panel_active = True
                                state.detail_panel_idx = 0
                            else:
                                delta = -1 if raw[:3] == b"\x1b[A" else 1
                                state.detail_panel_idx = (state.detail_panel_idx + delta) % 3
                        elif raw[:1] == b"\t":
                            if not state.detail_panel_active:
                                state.detail_panel_active = True
                                state.detail_panel_idx = 0
                            else:
                                state.detail_panel_idx = (state.detail_panel_idx + 1) % 3
                        elif raw[:3] in (b"\x1b[C", b"\x1b[D"):
                            if state.detail_panel_active:
                                if state.detail_panel_idx == 1 and state.detail_worker_files:
                                    n_w = len(state.detail_worker_files)
                                    delta = 1 if raw[:3] == b"\x1b[C" else -1
                                    state.detail_worker_idx = (state.detail_worker_idx + delta) % n_w
                                    state.detail_worker_lines = _tail(
                                        state.detail_worker_files[state.detail_worker_idx], 100
                                    )
                                elif state.detail_panel_idx == 2 and state.detail_bench_sections:
                                    delta = 1 if raw[:3] == b"\x1b[C" else -1
                                    n_s = len(state.detail_bench_sections)
                                    state.detail_bench_section_idx = (state.detail_bench_section_idx + delta) % n_s
                        elif raw[:1] == b"\x1b":
                            _close_detail(state)
                        elif raw[:1] in (b"\r", b"\n"):
                            jid = state.detail_job_id
                            logs_dir = outputs_dir / jid / "logs"
                            _vpath = _detail_log_path(
                                jid,
                                logs_dir,
                                state.detail_worker_files,
                                state.detail_worker_idx,
                                state.detail_panel_idx,
                            )
                            if _vpath is not None:
                                _launch_vim(_vpath)
                        elif raw[:1] not in (b"\x1b", b""):
                            ch = raw[:1].decode("utf-8", errors="replace")
                            if ch.lower() == "r":
                                state.detail_auto_refresh = not state.detail_auto_refresh
                                last_detail_refresh = 0.0
                            elif ch in ("q", "Q", "\x03"):
                                quit_requested = True
                                continue

                    else:
                        if state.cancel_confirm_job_id is not None:
                            if raw[:1] in (b"y", b"Y"):
                                jid = state.cancel_confirm_job_id
                                with contextlib.suppress(Exception):
                                    subprocess.run(["scancel", jid], timeout=10, capture_output=True)
                                state.cancel_confirm_job_id = None
                                _fetch_trigger.set()
                            elif raw[:1] in (b"n", b"N") or raw[:1] == b"\x1b":
                                state.cancel_confirm_job_id = None
                        elif state.delete_confirm_job_id is not None:
                            if raw[:1] in (b"y", b"Y"):
                                jid = state.delete_confirm_job_id
                                shutil.rmtree(outputs_dir / jid, ignore_errors=True)
                                state.seen_job_ids.discard(jid)
                                state.delete_confirm_job_id = None
                                state.selected_idx = max(0, state.selected_idx - 1)
                                _fetch_trigger.set()
                            elif raw[:1] in (b"n", b"N") or raw[:1] == b"\x1b":
                                state.delete_confirm_job_id = None
                        elif raw[:3] in (b"\x1b[A", b"\x1b[B"):
                            with _cache_lock:
                                n_jobs = len(_cached_jobs or [])
                            if raw[:3] == b"\x1b[A":
                                state.selected_idx = max(0, state.selected_idx - 1)
                            else:
                                state.selected_idx = min(n_jobs - 1, state.selected_idx + 1)
                        elif raw[:1] in (b"\r", b"\n"):
                            with _cache_lock:
                                jobs_snap = _cached_jobs or []
                            if 0 <= state.selected_idx < len(jobs_snap):
                                jid = jobs_snap[state.selected_idx]["job_id"]
                                state.detail_job_id = jid
                                state.detail_panel_idx = 0
                                logs_dir = outputs_dir / jid / "logs"
                                worker_files = sorted(
                                    (f for f in logs_dir.glob("*.out") if f.name != "benchmark.out"),
                                    key=lambda p: p.name,
                                )
                                state.detail_worker_files = worker_files
                                state.detail_worker_idx = 0
                                _refresh_detail_lines(state, jid, logs_dir)
                        elif raw[:1] not in (b"\x1b", b""):
                            ch = raw[:1].decode("utf-8", errors="replace")
                            if ch.lower() == "c":
                                state.show_all_concurrencies = not state.show_all_concurrencies
                            elif ch.lower() == "a":
                                state.show_all_jobs = not state.show_all_jobs
                                state.scroll_offset = 0
                                state.selected_idx = 0
                                _fetch_trigger.set()
                            elif ch.lower() == "y":
                                with _cache_lock:
                                    jobs_snap = _cached_jobs or []
                                if 0 <= state.selected_idx < len(jobs_snap):
                                    jid = jobs_snap[state.selected_idx]["job_id"]
                                    yaml_path = outputs_dir / jid / "config.yaml"
                                    if yaml_path.exists():
                                        _launch_vim(yaml_path)
                            elif ch.lower() == "d":
                                with _cache_lock:
                                    jobs_snap = _cached_jobs or []
                                if 0 <= state.selected_idx < len(jobs_snap):
                                    job = jobs_snap[state.selected_idx]
                                    if job["slurm_state"] in _ACTIVE_STATES:
                                        state.cancel_confirm_job_id = job["job_id"]
                                    else:
                                        state.delete_confirm_job_id = job["job_id"]
                            elif ch in ("q", "Q", "\x03"):
                                quit_requested = True
                                continue

                    last_render = 0.0  # force immediate re-render so input feels instant

                now = time.monotonic()
                if now >= last_height_check + 2.0:
                    term_height = _get_term_height()
                    last_height_check = now
                if (
                    state.detail_auto_refresh
                    and state.detail_job_id is not None
                    and now >= last_detail_refresh + args.interval
                ):
                    jid = state.detail_job_id
                    logs_dir = outputs_dir / jid / "logs"
                    _refresh_detail_lines(state, jid, logs_dir)
                    last_detail_refresh = now
                    last_render = 0.0
                if now >= last_render + 0.25:
                    with _cache_lock:
                        snap_jobs = _cached_jobs
                        snap_loading = _is_loading
                    try:
                        renderable, clamped_scroll, clamped_sel = _render(
                            snap_jobs,
                            outputs_dir,
                            args.interval,
                            state,
                            loading=snap_loading,
                            spin_idx=spin_idx,
                            term_height=term_height,
                            term_cols=term_cols,
                        )
                        state.scroll_offset = clamped_scroll
                        state.selected_idx = clamped_sel
                        live.update(renderable)
                    except Exception as exc:
                        live.update(Text(f"Error: {exc}", style="red"))
                    spin_idx += 1
                    last_render = now

                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        if key:
            _save_session(session_file, key, outputs_dir, state.seen_job_ids)
        if old_term is not None:
            with contextlib.suppress(Exception):
                termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        if key:
            console.print(
                f"[dim]To resume this session, use[/dim] [bold cyan]srtctl monitor --resume {key}[/bold cyan]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live CLI dashboard for srt-slurm jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--outputs", "-o", type=Path, default=None, help="Path to outputs/ directory")
    parser.add_argument("--interval", "-i", type=float, default=5.0, help="Refresh interval in seconds (default: 5)")
    parser.add_argument("--all", "-a", action="store_true", help="Include older jobs from outputs/ on startup")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--resume", type=str, default=None, metavar="KEY", help="Resume a previous session by key")
    parser.add_argument(
        "--clear-sessions", action="store_true", help="Delete all saved --resume session state and exit"
    )
    _execute(parser.parse_args())


if __name__ == "__main__":
    main()
