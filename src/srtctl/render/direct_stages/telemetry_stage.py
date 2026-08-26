# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tachometer collection and compaction stage for direct execution."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common import ManagedProcess


class TelemetryStageMixin:
    """Own Tachometer collection for the direct benchmark run."""

    plan: dict[str, Any]
    source_dir: Path
    output_dir: Path
    artifact_dir: Path
    tachometer: ManagedProcess | None
    tachometer_local_dir: Path | None

    def _die(self, message: str) -> None:
        raise NotImplementedError

    def _launch(self, label: str, log_name: str, args: list[str], **kwargs: Any) -> ManagedProcess:
        raise NotImplementedError

    def _run_logged(self, args: list[str], **kwargs: Any) -> None:
        raise NotImplementedError

    def _start_tachometer(self) -> None:
        if not self.plan["tachometer_enabled"]:
            return
        configured = str(self.source_dir / "bin" / "tachometer-scraper")
        if not os.access(configured, os.X_OK):
            self._die(f"Tachometer scraper is not executable: {configured}")
        storage = self.artifact_dir / "tachometer" / "raw" / "scrape"
        local_dir = self.artifact_dir / "tachometer" / "local"
        storage.parent.mkdir(parents=True, exist_ok=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        config = str(self.plan["tachometer_config"]).replace("${TACHOMETER_STORAGE}", str(storage))
        config_path = self.output_dir / "tachometer.toml"
        config_path.write_text(config, encoding="utf-8")
        args = [configured, "--config", str(config_path), "--local-dir", str(local_dir)]
        if int(self.plan["tachometer_sync_interval_secs"]) > 0:
            args.extend(["--sync-interval", str(self.plan["tachometer_sync_interval_secs"])])
        environment = dict(os.environ)
        if int(self.plan["tachometer_compaction_threads"]) > 0:
            environment["POLARS_MAX_THREADS"] = str(self.plan["tachometer_compaction_threads"])
        self.tachometer = self._launch("tachometer", "tachometer.log", args, env=environment)
        self.tachometer_local_dir = local_dir
        time.sleep(2)
        if self.tachometer.process.poll() is not None:
            self._die(f"Tachometer exited at startup; inspect {self.tachometer.log_path}")

    def _compact_tachometer(self) -> None:
        if self.tachometer_local_dir is None:
            return
        if (
            not any(self.tachometer_local_dir.glob("*.parquet"))
            and not (self.tachometer_local_dir / "current.arrow").is_file()
        ):
            return
        environment = dict(os.environ)
        if int(self.plan["tachometer_compaction_threads"]) > 0:
            environment["POLARS_MAX_THREADS"] = str(self.plan["tachometer_compaction_threads"])
        self._run_logged(
            [
                str(self.source_dir / "bin" / "tachometer-scraper"),
                "compact",
                str(self.tachometer_local_dir),
                "--output",
                str(self.artifact_dir / "tachometer" / "final"),
            ],
            log_name="tachometer.log",
            env=environment,
        )
