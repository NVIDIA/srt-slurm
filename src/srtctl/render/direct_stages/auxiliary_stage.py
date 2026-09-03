# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic auxiliary/sidecar service stage for direct execution.

Mirrors ``TelemetryStageMixin``'s shape (build config/args, ``self._launch``,
sleep-then-poll early-exit check) but for user-declared ``auxiliary_services``
entries instead of the hard-coded Tachometer process. See
``docs/auxiliary-services.md``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common import ManagedProcess


class AuxiliaryServiceStageMixin:
    """Launch user-declared sidecar services alongside the direct run."""

    plan: dict[str, Any]
    output_dir: Path
    log_dir: Path
    auxiliary_services: dict[str, ManagedProcess]

    def _die(self, message: str) -> None:
        raise NotImplementedError

    def log(self, message: str) -> None:
        raise NotImplementedError

    def _launch(self, label: str, log_name: str, args: list[str], **kwargs: Any) -> ManagedProcess:
        raise NotImplementedError

    def _run_logged(self, args: list[str], **kwargs: Any) -> None:
        raise NotImplementedError

    def _build_auxiliary_service_source(self, service: dict[str, Any]) -> Path | None:
        """Clone ``service['source']`` once, returning the directory to build/launch from."""
        source = service.get("source")
        if not source:
            return None
        name = service["name"]
        checkout_root = self.output_dir / "auxiliary_services" / name / "src"
        if not checkout_root.exists():
            checkout_root.parent.mkdir(parents=True, exist_ok=True)
            self.log(f"Cloning auxiliary service {name} source: {source['git']}@{source['rev']}")
            # env/timeout/http.version=HTTP/1.1: same guard as the real-SLURM-path
            # clone (cli/mixins/auxiliary_stage.py) against intermittent git
            # smart-HTTP/HTTP2 stalls seen on some clusters (NVIDIA/InferenceMAX#271).
            self._run_logged(
                [
                    "env", "GIT_TERMINAL_PROMPT=0", "timeout", "120s",
                    "git", "-c", "http.version=HTTP/1.1", "clone", "--filter=blob:none",
                    str(source["git"]), str(checkout_root),
                ],
                log_name=f"{name}.build.log",
            )
            self._run_logged(
                [
                    "env", "GIT_TERMINAL_PROMPT=0", "timeout", "120s",
                    "git", "-c", "http.version=HTTP/1.1", "-C", str(checkout_root),
                    "fetch", "origin", str(source["rev"]),
                ],
                log_name=f"{name}.build.log",
            )
            self._run_logged(
                [
                    "env", "GIT_TERMINAL_PROMPT=0", "timeout", "120s",
                    "git", "-c", "http.version=HTTP/1.1", "-C", str(checkout_root),
                    "checkout", "FETCH_HEAD",
                ],
                log_name=f"{name}.build.log",
            )
        work_dir = checkout_root
        if source.get("path"):
            work_dir = checkout_root / str(source["path"])
        return work_dir

    def _start_auxiliary_services(self) -> None:
        services: list[dict[str, Any]] = list(self.plan.get("auxiliary_services") or [])
        if not services:
            return
        etcd_endpoints = f"http://127.0.0.1:{self.plan['etcd_client_port']}"
        nats_server = f"nats://127.0.0.1:{self.plan['nats_port']}"

        for service in services:
            name = service["name"]
            work_dir = self._build_auxiliary_service_source(service)
            build_command = service.get("build_command")
            if work_dir is not None and build_command:
                self.log(f"Building auxiliary service {name}: {' '.join(build_command)}")
                self._run_logged(build_command, log_name=f"{name}.build.log", cwd=work_dir)

            environment = dict(os.environ)
            if service.get("inherit_discovery_env", True):
                environment["ETCD_ENDPOINTS"] = etcd_endpoints
                environment["NATS_SERVER"] = nats_server
            environment.update(service.get("env") or {})

            command = list(service["command"])
            self.log(f"Starting auxiliary service {name}: {' '.join(command)}")
            managed = self._launch(name, f"{name}.log", command, env=environment)
            self.auxiliary_services[name] = managed
            time.sleep(2)
            if managed.process.poll() is not None:
                self._die(f"Auxiliary service {name} exited at startup; inspect {managed.log_path}")
