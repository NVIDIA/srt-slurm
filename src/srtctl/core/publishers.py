# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, destination-neutral dispatch to installed result publishers.

Publishers own artifact selection, credentials, remote retries and deduplication.
An accepted submission is deliberately distinct from a published result.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from srtctl.core.schema import ResultPublisherConfig

logger = logging.getLogger(__name__)
PROTOCOL_VERSION = 1
MAX_RESPONSE_BYTES = 64 * 1024


def _response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or type(value.get("protocol_version")) is not int:
        raise ValueError("Missing publisher protocol version")
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported publisher protocol version")
    if value.get("status") not in ("accepted", "published", "skipped"):
        raise ValueError("Invalid publisher status")
    links = value.get("links", {})
    if not isinstance(links, dict):
        raise TypeError("Invalid publisher links")
    for name, url in links.items():
        if not isinstance(name, str) or not isinstance(url, str):
            raise TypeError("Invalid publisher link")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.username or parts.password:
            raise ValueError("Publisher links must be HTTP(S) URLs without credentials")
    # Only documented fields are persisted; arbitrary output can contain secrets.
    return {"protocol_version": PROTOCOL_VERSION, "status": value["status"], "links": links}


def _invoke(config: ResultPublisherConfig, request: dict[str, Any]) -> dict[str, Any]:
    # Spool output to disk rather than accepting an unbounded capture in memory.
    # Stderr and rejected output are never copied into benchmark logs/archives.
    with tempfile.TemporaryFile() as output:
        with subprocess.Popen(
            config.command,
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.DEVNULL,
            cwd=request["run_dir"],
            start_new_session=True,
        ) as process:
            try:
                process.communicate(json.dumps(request).encode(), timeout=config.timeout_seconds)
            except subprocess.TimeoutExpired:
                # Kill descendants as well: a shell wrapper must not leave an upload
                # running after the caller has recorded an ambiguous timeout.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, config.command)
        if output.tell() > MAX_RESPONSE_BYTES:
            raise ValueError("Publisher response too large")
        output.seek(0)
        return _response(json.load(output))


def publish_results(
    publishers: list[ResultPublisherConfig],
    *,
    log_dir: Path,
    job_id: str,
    benchmark_type: str,
    run_exit_code: int,
) -> None:
    """Invoke each publisher once; failures never change the benchmark outcome.

    Called after local postprocessing, including on failed runs. A zero run exit
    code is not proof of valid benchmark artifacts (e.g. serve-only mode).
    Publishers must validate the client-specific completion evidence themselves.
    """
    if not publishers:
        return
    try:
        resolved_log_dir = log_dir.resolve()
    except (OSError, RuntimeError) as error:
        logger.warning("Result publication unavailable (%s)", type(error).__name__)
        return
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "run_dir": str(resolved_log_dir.parent),
        "log_dir": str(resolved_log_dir),
        "job_id": job_id,
        "benchmark_type": benchmark_type,
        "run_exit_code": run_exit_code,
    }
    for config in publishers:
        receipt: dict[str, Any] = {"request": request}
        try:
            receipt["response"] = _invoke(config, request)
            receipt["state"] = receipt["response"]["status"]
            logger.info("Result publisher %s: %s", config.name, receipt["state"])
        except Exception as error:  # noqa: BLE001 - reporting cannot fail a run
            receipt.update(state="unknown", error_type=type(error).__name__)
            # Exception messages may contain command arguments or service secrets.
            logger.warning("Result publisher %s failed (%s); no automatic retry", config.name, type(error).__name__)
        try:
            directory = log_dir / "publishers"
            directory.mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as output:
                temporary = Path(output.name)
                json.dump(receipt, output, indent=2)
                output.write("\n")
            temporary.replace(directory / f"{config.name}.json")
        except OSError:
            logger.warning("Could not save result publisher %s receipt", config.name)
