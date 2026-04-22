# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry for the local mock sweep.

    python -m srtctl.cli.mock_worker \\
        --config cfg.yaml --output-dir ./out/42042 \\
        --job-id 42042 [--tick-s 0.4] [--phase-pause-s 0.25] [--nodelist n1 n2]

Runs the full SweepOrchestrator against patched infrastructure (see
`srtctl.mock`). Intended to be spawned by tests and by higher-level harnesses
that want a plausible sequence of srt-slurm artifacts without touching a real
cluster.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from srtctl.mock import MockOptions, run_mock_sweep

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a mock srtctl sweep (no real slurm involvement).",
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Per-job output directory.")
    parser.add_argument("--job-id", required=True, help="Fake SLURM job id.")
    parser.add_argument(
        "--tick-s",
        type=float,
        default=0.4,
        help="Wall time each fake srun child pretends to run.",
    )
    parser.add_argument(
        "--phase-pause-s",
        type=float,
        default=0.25,
        help="Pause between orchestrator phases (held by fake child durations).",
    )
    parser.add_argument(
        "--nodelist",
        nargs="+",
        default=["mock-node-01"],
        help="Fake SLURM nodelist.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    options = MockOptions(
        child_duration_s=args.tick_s,
        phase_pause_s=args.phase_pause_s,
        nodelist=tuple(args.nodelist),
    )
    exit_code = run_mock_sweep(
        config_path=args.config,
        output_dir=args.output_dir,
        job_id=args.job_id,
        options=options,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
