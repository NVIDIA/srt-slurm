#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate lm-eval scores against minimum thresholds."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def load_thresholds(path: str) -> dict[str, float]:
    """Load a {task_name: min_score} threshold config."""
    with open(path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate eval scores")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.85,
        help="Fallback minimum score when no threshold config matches.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Path to thresholds JSON config.",
    )
    parser.add_argument(
        "--metric-prefix",
        default="exact_match,",
        help="Only check metrics whose name starts with this prefix.",
    )
    parser.add_argument(
        "--results-glob",
        default="results*.json",
        help="Glob pattern for result files.",
    )
    args = parser.parse_args()

    thresholds: dict[str, float] = {}
    thresholds_path = args.thresholds
    if thresholds_path is None:
        default_path = Path(__file__).parent / "thresholds.json"
        if default_path.exists():
            thresholds_path = str(default_path)
    if thresholds_path:
        try:
            thresholds = load_thresholds(thresholds_path)
            print(f"Loaded thresholds from {thresholds_path}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARN: could not load thresholds from {thresholds_path}: {e}", file=sys.stderr)

    failed = False
    checked = 0

    for result_file in sorted(glob.glob(args.results_glob)):
        with open(result_file) as f:
            data = json.load(f)
        for task, metrics in data.get("results", {}).items():
            min_score = thresholds.get(task, args.min_score)
            for name, value in metrics.items():
                if not name.startswith(args.metric_prefix) or "stderr" in name:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                checked += 1
                if value < min_score:
                    print(f"FAIL: {task} {name} = {value:.4f} (< {min_score})", file=sys.stderr)
                    failed = True
                else:
                    print(f"PASS: {task} {name} = {value:.4f} (>= {min_score})")

    if checked == 0:
        print(f"WARN: no metrics matched prefix {args.metric_prefix!r}", file=sys.stderr)

    return 1 if failed or checked == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
