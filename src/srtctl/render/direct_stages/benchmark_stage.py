# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load-generation stage for direct execution."""

from __future__ import annotations

import os
import time
from typing import Any


class BenchmarkStageMixin:
    """Run the benchmark while continuously checking serving liveness."""

    plan: dict[str, Any]

    def log(self, message: str) -> None:
        raise NotImplementedError

    def _die(self, message: str) -> None:
        raise NotImplementedError

    def _launch_shell(self, label: str, log_name: str, command: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _assert_services_alive(self) -> None:
        raise NotImplementedError

    def _run_benchmark(self) -> None:
        self.log("Starting benchmark")
        benchmark = self._launch_shell(
            "benchmark", "aiperf.log", str(self.plan["benchmark_command"]), env=dict(os.environ)
        )
        while benchmark.process.poll() is None:
            self._assert_services_alive()
            time.sleep(1)
        if benchmark.process.returncode != 0:
            self._die(f"Benchmark exited with status {benchmark.process.returncode}; inspect {benchmark.log_path}")
        self.log("Benchmark completed successfully")
