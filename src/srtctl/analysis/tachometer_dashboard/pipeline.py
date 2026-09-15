# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort post-run UI creation from one closed raw Tachometer capture."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)
HTML_FILENAME = "dashboard.html"
STATUS_FILENAME = "dashboard-status.json"
BUILD_LOG_FILENAME = "dashboard-build.log"
BUILD_TIMEOUT_SECONDS = 1800


def _write_status(log_dir: Path, status: dict[str, Any]) -> None:
    """Publish status separately from HTML; rendering success is not capture completeness."""
    path = log_dir / STATUS_FILENAME
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def _build(config: SrtConfig, runtime: RuntimeContext, status: dict[str, Any]) -> Path | None:
    log_dir = runtime.log_dir
    tachometer = config.observability.tachometer
    # Explicitly choose the writer's local capture. raw/scrape/ is its upload
    # mirror, not another independent set of measurements to concatenate.
    source = log_dir / tachometer.storage_subdir / "local"
    output = log_dir / HTML_FILENAME
    status.update(source=str(source), output=str(output), previous_artifact=output.is_file())
    if not config.observability.tachometer_enabled:
        status.update(state="disabled", reason="Tachometer capture is disabled")
        return None
    if not source.is_dir() or not any(p.suffix in {".parquet", ".arrow"} for p in source.iterdir() if p.is_file()):
        status.update(state="missing", reason="No raw Tachometer Parquet/Arrow capture is available")
        return None

    status["state"] = "building"
    _write_status(log_dir, status)
    # A subprocess bounds runtime and isolates reader/renderer failures from
    # benchmark results. Its output stays private until the complete HTML and
    # catalog have been checked; an older HTML is never mistaken for success.
    with tempfile.TemporaryDirectory(prefix=".dashboard-build-", dir=log_dir) as temporary:
        staging = Path(temporary)
        staged_html = staging / HTML_FILENAME
        staged_data = staged_html.with_suffix(".data")
        command = [
            sys.executable,
            "-m",
            "srtctl.analysis.tachometer_dashboard",
            str(source),
            "--out",
            str(staged_html),
            "--title",
            f"{config.name} · {runtime.job_id}",
        ]
        with (log_dir / BUILD_LOG_FILENAME).open("w") as build_log:
            result = subprocess.run(
                command,
                stdout=build_log,
                stderr=subprocess.STDOUT,
                timeout=BUILD_TIMEOUT_SECONDS,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Dashboard builder exited {result.returncode}; see {BUILD_LOG_FILENAME}")
        if not staged_html.is_file() or staged_html.stat().st_size == 0:
            raise RuntimeError("Dashboard builder did not produce a nonempty HTML artifact")
        catalog = json.loads((staged_data / "catalog.json").read_text())
        if not catalog.get("row_count") or not catalog.get("metrics") or not catalog.get("source_files"):
            raise RuntimeError("Dashboard builder did not produce a populated raw metric catalog")
        # Keep an old sidecar only until the new build succeeds. HTML is
        # self-contained and its single replace is the publication boundary.
        data_dir = output.with_suffix(".data")
        previous_data = staging / "previous.data"
        if data_dir.exists():
            data_dir.replace(previous_data)
        try:
            staged_data.replace(data_dir)
            staged_html.replace(output)
        except OSError:
            if previous_data.exists():
                if data_dir.exists():
                    data_dir.replace(staging / "failed.data")
                previous_data.replace(data_dir)
            raise
        status.update(
            state="ready",
            previous_artifact=False,
            row_count=catalog["row_count"],
            metric_families=len(catalog["metrics"]),
            source_files=catalog["source_files"],
            excluded_source_files=catalog.get("excluded_source_files", []),
            source_warnings=catalog.get("source_warnings", []),
            note="UI creation succeeded; this does not certify capture completeness or every metric's arithmetic.",
        )
    return output


def try_build(config: SrtConfig, runtime: RuntimeContext) -> Path | None:
    """Build the raw run UI and record ready/disabled/missing/failed status.

    Measurements come only from the configured Tachometer local capture. The
    recipe supplies a display title and source location, never chart values.
    No failure falls back to legacy log/client/processed-JSONL ingestion.
    """
    status: dict[str, Any] = {"schema_version": 1, "state": "failed", "source_policy": "raw-tachometer-only"}
    output = None
    try:
        output = _build(config, runtime, status)
    except Exception as error:  # noqa: BLE001 - UI creation cannot fail the benchmark
        status.update(state="failed", reason=str(error))
        logger.warning("Raw dashboard build failed: %s", error)
    try:
        _write_status(runtime.log_dir, status)
    except OSError as error:
        logger.warning("Could not write dashboard status: %s", error)
    if output is not None:
        logger.info("Raw dashboard: %s", output)
    elif status["state"] != "failed":
        logger.warning("Raw dashboard %s: %s", status["state"], status.get("reason"))
    return output
