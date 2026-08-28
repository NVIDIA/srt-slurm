# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small standalone CLI for re-processing an existing Dynamo run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from srtctl.ruter.normalize import default_output_dir, normalize_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a completed Dynamo router benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="Write normalized router and worker JSONL")
    init_parser.add_argument("root", type=Path, nargs="?", default=Path("."), help="srt-slurm run directory")
    init_parser.add_argument("--output", type=Path, help="Output directory (defaults to logs/.ruter)")
    args = parser.parse_args()

    if args.command == "init":
        root = args.root.resolve()
        report = normalize_run(root, output_dir=args.output or default_output_dir(root))
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
