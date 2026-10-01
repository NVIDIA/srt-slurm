# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: ray``: a Ray cluster across the service's nodes, one raylet per node.

The first instance (the first node the placement selects) is the head: ``ray
start --head`` with the GCS on ``options.port`` and the dashboard on
``options.dashboard_port``, both bound to the node's fabric IP. Every other
instance joins it with ``ray start --address``. Both run with ``--block`` so
the raylet lives exactly as long as its srun step and stops with it on
teardown.

Two facts shape the launch:

- Whatever Ray spawns later (actors, the driver of a ``ray job submit``) runs
  inside these steps' containers. The service therefore carries the job's
  container, mounts and environment; the client that submits work only needs
  the network.
- Per-node readiness differs by role. The head is ready when its dashboard
  answers ``/api/version``; a worker only logs ``Ray runtime started``. The
  fleet is ready when the head's ``/nodes?view=summary`` lists every member
  ALIVE, which is what the ``wait_fleet_ready`` hook checks before the job
  moves on.

Placement ``workers`` (the default) is every worker node; ``head`` gives a
single-node cluster; ``all`` is every allocated node. GPUs default to the
node's count and are exported as ``CUDA_VISIBLE_DEVICES`` so the raylet and
``--num-gpus`` agree on GRES partitions.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import requests
from marshmallow import ValidationError

from srtctl.core.slurm import get_hostname_ip
from srtctl.ports import RAY_DASHBOARD_PORT, RAY_GCS_PORT
from srtctl.services.config import HttpProbe, LogProbe, ServiceReadinessConfig
from srtctl.services.registry import ServiceKind, ServiceLaunchContext, register_service

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.services.config import ServiceConfig

logger = logging.getLogger(__name__)

RAY_READINESS_TIMEOUT = 600  # the first `ray start` in a fresh container imports torch and friends
RAY_STARTED_PATTERN = r"Ray runtime started"
RAY_PLACEMENTS = ("head", "workers", "all")


def _int_option(service: ServiceConfig, key: str, default: int) -> int:
    value = service.options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"services[{service.name}].options.{key} must be a positive integer; got {value!r}")
    return value


@register_service("ray")
class RayService(ServiceKind):
    """One Ray cluster: head on the first instance, workers joining it, readiness per role plus a fleet gate."""

    builds_command = True
    default_start = "before_workers"
    default_critical = True
    default_placement = "workers"
    default_readiness_timeout = RAY_READINESS_TIMEOUT
    option_keys = ("port", "dashboard_port", "num_gpus")

    # -- helpers ---------------------------------------------------------------------

    @staticmethod
    def gcs_port(service: ServiceConfig) -> int:
        return _int_option(service, "port", RAY_GCS_PORT)

    @staticmethod
    def dashboard_port(service: ServiceConfig) -> int:
        return _int_option(service, "dashboard_port", RAY_DASHBOARD_PORT)

    @staticmethod
    def num_gpus(service: ServiceConfig, runtime: RuntimeContext) -> int | None:
        """GPUs per raylet: ``options.num_gpus``, else the node's count; None only in the dry-run preview."""
        if "num_gpus" in service.options:
            return _int_option(service, "num_gpus", 1)
        gpus_per_node = getattr(runtime, "gpus_per_node", None)
        return int(gpus_per_node) if gpus_per_node is not None else None

    @staticmethod
    def head_node(service: ServiceConfig, runtime: RuntimeContext) -> str:
        """The node instance 0 lands on, mirroring the stage's placement order."""
        pool = service.effective_pool
        if pool is not None:
            return runtime.nodes.pools[pool][0]
        if service.effective_placement == "workers":
            return runtime.nodes.worker[0]
        return runtime.nodes.head

    @classmethod
    def head_ip(cls, service: ServiceConfig, runtime: RuntimeContext) -> str:
        return get_hostname_ip(cls.head_node(service, runtime), runtime.network_interface)

    @staticmethod
    def is_head(ctx: ServiceLaunchContext) -> bool:
        return ctx.index == 0

    @classmethod
    def dashboard_url(cls, service: ServiceConfig, runtime: RuntimeContext) -> str:
        return f"http://{cls.head_ip(service, runtime)}:{cls.dashboard_port(service)}"

    # -- kind hooks ------------------------------------------------------------------

    def validate(self, service: ServiceConfig, config: SrtConfig) -> None:
        if service.effective_placement not in RAY_PLACEMENTS:
            raise ValidationError(
                f"services[{service.name}] (type ray) must be placed on {', '.join(RAY_PLACEMENTS)}; "
                f"got {service.effective_placement!r}. Ray owns whole nodes, not a role's GPU slice."
            )
        if service.command is not None:
            raise ValidationError(
                f"services[{service.name}] (type ray) builds its own `ray start` commands; use args/options "
                "instead of command"
            )
        self.gcs_port(service)
        self.dashboard_port(service)
        if "num_gpus" in service.options:
            _int_option(service, "num_gpus", 1)

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        port = self.gcs_port(service)
        num_gpus = self.num_gpus(service, ctx.runtime)
        common = [
            f"--node-ip-address={ctx.node_ip}",
            f"--num-gpus={num_gpus if num_gpus is not None else '<gpus_per_node>'}",
            "--disable-usage-stats",
            "--block",
            *service.args,
        ]
        if self.is_head(ctx):
            return [
                "ray",
                "start",
                "--head",
                f"--port={port}",
                "--dashboard-host=0.0.0.0",
                f"--dashboard-port={self.dashboard_port(service)}",
                *common,
            ]
        # The stage launches the head first and waits for its dashboard before any
        # worker starts, so the GCS is up here; the tcp wait only covers the window
        # between "dashboard answers" and "GCS accepts registrations". exec makes the
        # raylet the step's process, so SIGTERM on teardown reaches it directly.
        head = f"{self.head_ip(service, ctx.runtime)}:{port}"
        join = " ".join(["ray", "start", f"--address={head}", *common])
        wait = (
            f"for i in $(seq 1 60); do (exec 3<>/dev/tcp/{self.head_ip(service, ctx.runtime)}/{port}) "
            f'2>/dev/null && break; echo "ray worker: waiting for head {head} ($i/60)"; sleep 5; done'
        )
        return ["bash", "-c", f"{wait}; exec {join}"]

    def default_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        num_gpus = self.num_gpus(service, ctx.runtime)
        devices = ",".join(str(i) for i in range(num_gpus)) if num_gpus is not None else "<0..gpus_per_node-1>"
        return {
            # The raylet's device view must match --num-gpus; on GRES partitions a
            # step without this can see fewer devices than the node has.
            "CUDA_VISIBLE_DEVICES": devices,
            # Ray's memory monitor kills workers under host-memory pressure, which
            # a trainer offloading to host RAM triggers on purpose.
            "RAY_memory_monitor_refresh_ms": "0",
        }

    def readiness(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> ServiceReadinessConfig | None:
        timeout = self.default_readiness_timeout
        if self.is_head(ctx):
            probe = HttpProbe(port=self.dashboard_port(service), path="/api/version", status=200)
            return ServiceReadinessConfig(http=probe, timeout_seconds=timeout, interval_seconds=5)
        return ServiceReadinessConfig(
            log=LogProbe(pattern=RAY_STARTED_PATTERN), timeout_seconds=timeout, interval_seconds=5
        )

    def wait_fleet_ready(self, service: ServiceConfig, runtime: RuntimeContext, procs: list[ManagedProcess]) -> None:
        expected = len(procs)
        if expected <= 1:
            return
        url = f"{self.dashboard_url(service, runtime)}/nodes?view=summary"
        deadline = time.monotonic() + self.default_readiness_timeout
        alive = 0
        logger.info("Waiting for the Ray cluster to report %d alive nodes (%s)", expected, url)
        while True:
            alive = self.alive_nodes(url)
            if alive >= expected:
                logger.info("Ray cluster ready: %d/%d nodes alive", alive, expected)
                return
            dead = [proc for proc in procs if not proc.is_running]
            if dead:
                raise RuntimeError(
                    f"services[{service.name}]: ray step on {dead[0].node} exited (code {dead[0].exit_code}) before "
                    f"the cluster formed ({alive}/{expected} alive); see {dead[0].log_file}"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"services[{service.name}]: only {alive}/{expected} Ray nodes alive after "
                    f"{self.default_readiness_timeout}s; see the service_{service.name}_*.out logs"
                )
            time.sleep(5)

    @staticmethod
    def alive_nodes(url: str) -> int:
        """Count ALIVE raylets in the dashboard's node summary; 0 when it does not answer yet."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            summary = response.json().get("data", {}).get("summary", [])
        except (requests.exceptions.RequestException, ValueError):
            return 0
        return sum(1 for node in summary if (node.get("raylet") or {}).get("state") == "ALIVE")
