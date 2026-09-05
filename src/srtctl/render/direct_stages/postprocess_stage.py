# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ruter post-processing stage for direct execution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


class PostProcessStageMixin:
    """Normalize direct-run artifacts into the portable ruter bundle."""

    plan: dict[str, Any]
    source_dir: Path
    output_dir: Path
    log_dir: Path
    ruter_python: Path

    def log(self, message: str) -> None:
        raise NotImplementedError

    def _run_logged(self, args: list[str], **kwargs: Any) -> None:
        raise NotImplementedError

    def _normalize_ruter(self) -> None:
        if not self.plan["ruter_enabled"]:
            return
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(self.source_dir / "src") if not existing else f"{self.source_dir / 'src'}:{existing}"
        )
        try:
            self._run_logged(
                [
                    str(self.ruter_python),
                    "-m",
                    "srtctl.ruter",
                    "init",
                    str(self.output_dir),
                    "--output",
                    str(self.log_dir / ".ruter"),
                ],
                log_name="ruter.log",
                env=environment,
            )
        except (OSError, subprocess.CalledProcessError):
            self.log("WARNING: ruter normalization failed; inspect " + str(self.log_dir / "ruter.log"))
