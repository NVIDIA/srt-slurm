#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate benchmark-rollup.json from MLPerf LoadGen logs.

The harness writes one ``mlperf_log_summary.txt`` per run under
``<log_dir>/mlperf/<scenario>/<performance|accuracy>/``. This rollup turns each
into one record so downstream consumers don't have to parse LoadGen's
human-readable summary themselves.

The section parser is deliberately generic (``key : value`` inside
``===``-delimited sections) and the whole section is carried through verbatim.
LoadGen renames and adds metrics between submission rounds, so anything that
hardcoded the full metric list would quietly start dropping fields on the next
round. Only the handful of fields ``srtctl monitor`` renders are normalized on
top, and each is looked up by pattern rather than exact name.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

SEPARATOR_RE = re.compile(r"^=+\s*$")
KEY_VALUE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[^:]+?)\s*:\s*(?P<value>.*?)\s*$")

# Section title -> key in the emitted record.
SECTION_KEYS = {
    "MLPerf Results Summary": "summary",
    "Additional Stats": "additional_stats",
    "Test Parameters Used": "test_parameters",
}

TOKENS_PER_SECOND_RE = re.compile(r"tokens per second", re.IGNORECASE)
SAMPLES_PER_SECOND_RE = re.compile(r"samples per second", re.IGNORECASE)
TTFT_P99_RE = re.compile(r"99\.00 percentile first token latency", re.IGNORECASE)
TPOT_P99_RE = re.compile(r"99\.00 percentile time to output token", re.IGNORECASE)


def _coerce(value: str) -> Any:
    """Numbers become numbers; everything else stays the string LoadGen wrote."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_summary(text: str) -> dict[str, dict[str, Any]]:
    """Parse an ``mlperf_log_summary.txt`` into its sections.

    Indented key/value lines directly under ``Result is`` are LoadGen's
    validity constraints ("Min duration satisfied : Yes"); they are collected
    separately from the headline metrics so a caller can tell "the run is
    invalid" from "the run was slow".
    """
    sections: dict[str, dict[str, Any]] = {key: {} for key in SECTION_KEYS.values()}
    sections["constraints"] = {}

    lines = text.splitlines()
    current: str | None = None
    index = 0
    while index < len(lines):
        # A section header is exactly three lines: separator, title, separator.
        if (
            SEPARATOR_RE.match(lines[index])
            and index + 2 < len(lines)
            and SEPARATOR_RE.match(lines[index + 2])
        ):
            current = SECTION_KEYS.get(lines[index + 1].strip())
            index += 3
            continue

        line = lines[index]
        index += 1
        if current is None:
            continue
        match = KEY_VALUE_RE.match(line)
        if not match or not match.group("key").strip():
            continue
        key = match.group("key").strip()
        value = _coerce(match.group("value"))
        if current == "summary" and match.group("indent"):
            sections["constraints"][key] = value
        else:
            sections[current][key] = value
    return sections


def _find(section: dict[str, Any], pattern: re.Pattern[str]) -> float | None:
    """First numeric value whose key matches ``pattern``.

    Keys are scanned in the order LoadGen printed them, which is why the
    Server scenario needs ``_completed_first`` on top: it prints "Scheduled
    samples per second" before "Completed samples per second", and only the
    completed figure describes what the SUT actually did.
    """
    for key, value in section.items():
        if pattern.search(key) and isinstance(value, (int, float)):
            return float(value)
    return None


def _completed_first(section: dict[str, Any], pattern: re.Pattern[str]) -> float | None:
    for key, value in section.items():
        if pattern.search(key) and "completed" in key.lower() and isinstance(value, (int, float)):
            return float(value)
    return _find(section, pattern)


def _ns_to_ms(value: float | None) -> float | None:
    return value / 1e6 if value is not None else None


def build_run(summary_path: Path, results_dir: Path) -> dict[str, Any]:
    sections = parse_summary(summary_path.read_text())
    summary = sections["summary"]
    stats = sections["additional_stats"]
    run_dir = summary_path.parent

    run: dict[str, Any] = {
        "path": str(run_dir.relative_to(results_dir)),
        "scenario": summary.get("Scenario"),
        "loadgen_mode": summary.get("Mode"),
        "result": summary.get("Result is"),
        # LoadGen prints no "Result is" line in AccuracyOnly mode, so absence
        # is not failure — only an explicit non-VALID verdict is.
        "valid": None if "Result is" not in summary else summary.get("Result is") == "VALID",
        "throughput_toks": _completed_first(summary, TOKENS_PER_SECOND_RE),
        "samples_per_second": _completed_first(summary, SAMPLES_PER_SECOND_RE),
        "ttft_p99_ms": _ns_to_ms(_find(stats, TTFT_P99_RE)),
        "tpot_p99_ms": _ns_to_ms(_find(stats, TPOT_P99_RE)),
        "target_qps": sections["test_parameters"].get("target_qps"),
        "constraints": sections["constraints"],
        "summary": summary,
        "additional_stats": stats,
        "test_parameters": sections["test_parameters"],
    }

    # Written by eval_mlperf_accuracy.py when MLPERF_EVAL_ACCURACY=1; an
    # accuracy run without scoring still produces a valid record.
    accuracy_path = run_dir / "accuracy.json"
    if accuracy_path.is_file():
        try:
            run["accuracy"] = json.loads(accuracy_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping unreadable {accuracy_path.name}: {exc}", file=sys.stderr)

    return run


def main(log_dir: str) -> int:
    results_dir = Path(log_dir) / "mlperf"
    if not results_dir.is_dir():
        print(f"no mlperf results dir at {results_dir}", file=sys.stderr)
        return 0

    runs = []
    for summary_path in sorted(results_dir.glob("*/*/mlperf_log_summary.txt")):
        try:
            runs.append(build_run(summary_path, results_dir))
        except OSError as exc:
            print(f"skipping unreadable {summary_path}: {exc}", file=sys.stderr)

    if not runs:
        print(f"no mlperf_log_summary.txt found under {results_dir}", file=sys.stderr)
        return 0

    out = Path(log_dir) / "benchmark-rollup.json"
    out.write_text(json.dumps({"benchmark_type": "mlperf", "runs": runs}, indent=1))
    print(f"wrote {out} ({len(runs)} run(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
