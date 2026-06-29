# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the benchmark phase against an already-started srtctl deployment."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import load_config
from srtctl.core.processes import ProcessRegistry, setup_signal_handlers
from srtctl.core.runtime import RuntimeContext
from srtctl.core.slurm import get_slurm_job_id
from srtctl.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark phase for an already-started deployment")
    parser.add_argument("config", type=Path, help="Path to the original benchmark YAML configuration")
    args = parser.parse_args()

    setup_logging()

    try:
        if not args.config.exists():
            logger.error("Config file not found: %s", args.config)
            sys.exit(1)

        config = load_config(args.config)
        job_id = get_slurm_job_id()
        if not job_id:
            logger.error("Not running in SLURM (SLURM_JOB_ID not set)")
            sys.exit(1)

        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        registry = ProcessRegistry(job_id=job_id)
        stop_event = threading.Event()
        setup_signal_handlers(stop_event, registry)

        exit_code = orchestrator.run_benchmark(registry, stop_event, reporter=None)
        orchestrator.run_postprocess(exit_code, reporter=None)
        sys.exit(exit_code)
    except Exception as exc:
        logger.exception("Fatal benchmark error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
