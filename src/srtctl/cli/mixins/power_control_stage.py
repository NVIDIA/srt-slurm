# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Power control stage mixin for SweepOrchestrator: EPD node_agents and appliers per pool.

``power_control`` (docs/config-reference.md#power_control) runs NVIDIA's EPD power
controller over the job's workers without touching the engines. Once the workers are
healthy and before any load, the stage:

1. groups the workers' GPUs into pools from the role -> node -> GPU-index map.
   ``Process.gpu_indices`` is ``Endpoint.gpus_on_node(node_rank)``: the exact GPUs a worker
   holds on each of its nodes, also for MPI workers packed across partial nodes, so a node
   shared by two workers of one pool is one group and idle GPUs stay out. A node whose
   GPUs would belong to two pools is refused;
2. launches, per pool and node, one ``dynamo.power.node_agent`` srun step in the worker
   image (``--overlap``, the recipe's srun options, the recipe ``environment:`` with the
   workers' ``{node}``/``{node_id}`` templating, the EPD package and optionally Dynamo's
   ``dynamo/power`` bind-mounted read-only over the image's site-packages). A group that
   covers the whole node sees every GPU; a group that is only part of the node (a TEP6
   tail) gets ``NVIDIA_VISIBLE_DEVICES`` = its indices, because EPD's NVML monitor senses
   every GPU it can see. ``--gpu-indices`` are host NVML indices (the node_agent verifies that
   it sees exactly as many GPUs as the group has), ``--gpus-per-node`` the node's physical GPU count
   and ``--pool-gpu-count`` the pool's exact GPU total (per-GPU FoM normalisation).
   ``--allocation`` (``power_control.allocation``) selects the law NUPD splits the pool's
   budget with, and ``--node-buffers`` lists every group's UBPC buffer as ``<node>=<buffer>``
   (its roles' value, or the pool's explicit one), so one shared controller holds prefill
   nodes at 2.0 and decode nodes at 1.1 (EPD's ``NUPD_ALLOCATION`` / ``EPD_NODE_BUFFERS``). Its
   ``--metrics-url`` list names the frontend listeners (``frontend=<url>``) and the pool
   ranks that run a Dynamo system server (``worker=<url>``: every rank of a per-node launch,
   only the MPI leader of a TRT-LLM step), each node's IP resolved once. ``min_tgp_w`` is
   never passed to the node_agent (EPD's cap law has no floor); the applier enforces it as
   ``--min-w-per-gpu``. Rank 0, the pool's first node, also runs the EPD
   controller, whose ``DecisionFilePublisher`` writes
   ``<LOG_DIR>/epd/actuation/power_decision.json`` (``DECISION_FILE_RELPATH``), so the
   node_agent must hand ``--log-dir`` to EPD as ``LOG_DIR`` unchanged. Its SIGTERM budget
   follows the cadence (``node_agent_terminate_timeout``): EPD's stop joins an in-flight
   cycle for ``interval_s`` + 5 s before it writes ``run_summary.json``;
3. launches one ``dynamo.power.applier`` per node on the bare host (no container). It polls
   exactly that decision file and sets this node's caps per GPU index with
   ``sudo nvidia-smi -pl <W> -i <N>`` or ``dcgmi``, falls back to the uniform budget on a
   stale file, and restores the hardware default limits when it exits.

``validate_power_control_layout`` resolves the pools at job start, before host setup and
the workers, so a layout the launch would refuse costs seconds rather than a model load.
Both steps are registered as managed processes (``power_control.critical``). Teardown
(``stop_power_control``, on the orchestrator's cleanup path) stops the node_agents first
(EPD writes its run summary), then the appliers (which restore the defaults), logs an error
for a pool whose decision file never appeared (EPD never ran as the controller, so the run
was a static-cap run at the uniform budget), then runs a final container-less per-node
reset (``sudo -n``; the exit status covers every GPU) so no cap outlives the allocation.

Every step launches through ``start_srun_process`` with a ``step_name``; the mock
orchestrator patches this module's ``start_srun_process`` and ``get_hostname_ip``.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from srtctl.cli.mixins.worker_stage import render_node_environment
from srtctl.core.ip_utils import url_host
from srtctl.core.processes import ManagedProcess, ProcessRegistry, list_step_ids
from srtctl.core.schema import PowerControlConfig, PowerPoolSpec, installs_dynamo
from srtctl.core.slurm import CONTAINER_REMAP_ROOT_EXPORT, get_hostname_ip, start_srun_process
from srtctl.frontends import FRONTEND_NONE, get_frontend

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

#: Directory below the run's log dir; each pool's subdirectory is that pool's EPD ``LOG_DIR``.
POWER_CONTROL_DIRNAME = "power_control"
#: EPD's file actuator writes ``{"per_node_gpu_power_limit_w": {<node>: <W>}}`` here, below the pool's LOG_DIR.
DECISION_FILE_RELPATH = Path("epd") / "actuation" / "power_decision.json"
NODE_AGENT_MODULE = "dynamo.power.node_agent"
APPLIER_MODULE = "dynamo.power.applier"
# The applier restores the default limits on SIGTERM; also the floor of the node_agent's budget.
POWER_TERMINATE_TIMEOUT_SECONDS = 30.0
# Added to interval_s for the node_agent's SIGTERM budget: EPD's pull.stop joins an in-flight
# cycle for interval_s + 5 s, then does the final NVML read-back and writes run_summary.json.
NODE_AGENT_STOP_MARGIN_SECONDS = 15.0
# Wall-clock budget of one container-less reset step per node, after the appliers are gone.
POWER_RESET_TIMEOUT_SECONDS = 120


def node_agent_terminate_timeout(interval_s: float) -> float:
    """SIGTERM budget of a node_agent step: the cadence plus a margin, never below the appliers' 30 s.

    With a 30 s cadence EPD may need 35 s to finish a cycle that was in flight when SIGTERM
    landed (an HTTP FoM scrape timing out, slow snapshot I/O) before ``core.stop`` writes
    ``run_summary.json``; a flat 30 s budget has the registry abort the step and lose it.
    """
    return max(POWER_TERMINATE_TIMEOUT_SECONDS, float(interval_s) + NODE_AGENT_STOP_MARGIN_SECONDS)


@dataclass(frozen=True)
class GpuGroup:
    """The GPUs on one node that belong to one pool: EPD's unit of control.

    One node_agent senses exactly these GPU indices and the applier caps exactly them,
    so idle GPUs on a partially used node neither drag the node's measured draw down
    nor get a cap. ``ubpc_buffer`` is this group's own burst allowance (the pool's
    explicit value, else the one its roles imply): the group's ``--node-buffers`` entry.
    """

    pool: str
    node: str
    gpu_indices: frozenset[int]
    roles: tuple[str, ...]
    ubpc_buffer: float
    het_group: int | None
    processes: tuple[Process, ...]

    @property
    def gpu_list(self) -> str:
        """``--gpu-indices`` value: the sorted indices, comma-separated."""
        return ",".join(str(index) for index in sorted(self.gpu_indices))


@dataclass(frozen=True)
class PowerPool:
    """One EPD controller: a pool spec resolved onto the GPU groups of the allocation.

    The first group is the controller's node (rank 0). ``ubpc_buffer`` is the spec's own
    value or the largest its roles imply (``--ubpc-buffer``); each group carries its own
    (``node_buffers``).
    """

    spec: PowerPoolSpec
    groups: tuple[GpuGroup, ...]
    ubpc_buffer: float

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def nodes(self) -> tuple[str, ...]:
        """The pool's roster, in allocation order: ``--nodelist`` and ``EPD_NODELIST``."""
        return tuple(group.node for group in self.groups)

    @property
    def node_buffers(self) -> str:
        """``--node-buffers`` value: ``<node>=<buffer>`` per group in roster order (EPD's ``EPD_NODE_BUFFERS``)."""
        return ",".join(f"{group.node}={group.ubpc_buffer:g}" for group in self.groups)

    @property
    def gpu_count(self) -> int:
        """GPUs in use, not nodes times GPUs per node: what the budget counts."""
        return sum(len(group.gpu_indices) for group in self.groups)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(role for group in self.groups for role in group.roles))

    @property
    def processes(self) -> tuple[Process, ...]:
        return tuple(process for group in self.groups for process in group.processes)


def compute_power_pools(power: PowerControlConfig, processes: Sequence[Process]) -> list[PowerPool]:
    """Resolve the recipe's pools onto the workers' GPU groups, one group per (pool, node).

    A process contributes its ``gpu_indices`` (``Endpoint.gpus_on_node`` for its rank) to
    the group of its node; two workers of one pool on one node (TEP6 packing) make one
    group. Each group carries the UBPC buffer of the roles it runs (a pool's explicit
    ``ubpc_buffer`` wins) for ``--node-buffers``. Raises ``ValueError`` when a node would
    belong to two pools, when an explicit node pool names a node that runs no worker, or
    when a pool ends up with no GPU.
    """
    specs = power.resolve_pools([process.endpoint_mode for process in processes])
    owner: dict[str, str] = {}
    pools: list[PowerPool] = []
    for spec in specs:
        if spec.roles:
            members = [process for process in processes if process.endpoint_mode in spec.roles]
        else:
            members = [process for process in processes if process.node in spec.nodes]
            idle = [node for node in spec.nodes if node not in {process.node for process in members}]
            if idle:
                raise ValueError(f"power_control.pools[{spec.name}].nodes: {', '.join(idle)} run no worker")
        if not members:
            raise ValueError(f"power_control pool {spec.name!r} contains no worker GPU")
        by_node: dict[str, list[Process]] = {}
        for process in members:
            by_node.setdefault(process.node, []).append(process)
        groups: list[GpuGroup] = []
        for node, node_processes in by_node.items():
            if node in owner:
                raise ValueError(
                    f"node {node} would belong to power pools {owner[node]!r} and {spec.name!r}; a node is one EPD "
                    "unit of control and belongs to exactly one pool (use pools: shared, or draw the pool boundary "
                    "on node lines)"
                )
            owner[node] = spec.name
            roles = tuple(dict.fromkeys(process.endpoint_mode for process in node_processes))
            groups.append(
                GpuGroup(
                    pool=spec.name,
                    node=node,
                    gpu_indices=frozenset().union(*(process.gpu_indices for process in node_processes)),
                    roles=roles,
                    ubpc_buffer=spec.ubpc_buffer if spec.ubpc_buffer is not None else power.ubpc_buffer_for(roles),
                    het_group=node_processes[0].het_group,
                    processes=tuple(node_processes),
                )
            )
        pool_roles = tuple(dict.fromkeys(role for group in groups for role in group.roles))
        buffer = spec.ubpc_buffer if spec.ubpc_buffer is not None else power.ubpc_buffer_for(pool_roles)
        pools.append(PowerPool(spec=spec, groups=tuple(groups), ubpc_buffer=buffer))
    return pools


def build_node_agent_command(
    power: PowerControlConfig,
    pool: PowerPool,
    group: GpuGroup,
    rank: int,
    metrics_urls: Sequence[str],
    log_dir: Path,
    gpus_per_node: int,
) -> list[str]:
    """The ``dynamo.power.node_agent`` argv of ``group`` in ``pool`` (``log_dir`` as the container sees it)."""
    if not metrics_urls:
        raise ValueError(f"power pool {pool.name!r} has no metrics URL: no frontend and no worker serves metrics")
    command = [
        "python3",
        "-m",
        NODE_AGENT_MODULE,
        "--pool-name",
        pool.name,
        "--rank",
        str(rank),
        "--nodelist",
        ",".join(pool.nodes),
        "--log-dir",
        str(log_dir),
        "--gpu-indices",
        group.gpu_list,
        "--mode",
        power.mode,
        "--budget-w-per-gpu",
        str(pool.spec.budget_w_per_gpu),
        "--fom",
        pool.spec.fom,
        "--interval-s",
        str(power.interval_s),
        "--perf-threshold",
        str(power.perf_threshold),
        "--ubpc-buffer",
        str(pool.ubpc_buffer),
    ]
    # min_tgp_w goes to the applier (--min-w-per-gpu): the node_agent refuses --min-tgp-w
    # because EPD's cap law has no floor.
    if power.max_tgp_w is not None:
        command.extend(["--max-tgp-w", str(power.max_tgp_w)])
    command.extend(
        [
            "--warmup-policy",
            power.warmup_policy,
            "--allocation",
            power.allocation,
            "--node-buffers",
            pool.node_buffers,
            "--metrics-url",
            ",".join(metrics_urls),
            "--gpus-per-node",
            str(gpus_per_node),
            "--pool-gpu-count",
            str(pool.gpu_count),
        ]
    )
    return command


def node_agent_environment(pool: PowerPool, log_dir: Path, gpus_per_node: int) -> dict[str, str]:
    """EPD's own identity, log and actuator variables, mirroring the node_agent flags.

    EPD reads its roster from ``EPD_NODELIST``/``EPD_NUM_NODES`` (several controllers can
    share one allocation that way), its output root from ``LOG_DIR`` (``EPD_LOG_DIR`` for
    the pull driver), and never touches a GPU with ``EPD_ACTUATOR=file``.
    ``EPD_RUN_TRACE``/``EPD_SUMMARY_STDOUT`` keep ``run_trace.csv`` and the stdout banner.
    """
    return {
        "LOG_DIR": str(log_dir),
        "EPD_LOG_DIR": str(log_dir),
        "EPD_ACTUATOR": "file",
        "EPD_NODELIST": ",".join(pool.nodes),
        "EPD_NUM_NODES": str(len(pool.nodes)),
        "GPUS_PER_NODE": str(gpus_per_node),
        "EPD_RUN_TRACE": "1",
        "EPD_SUMMARY_STDOUT": "1",
    }


def build_applier_command(
    power: PowerControlConfig,
    pool: PowerPool,
    group: GpuGroup,
    decision_file: Path,
    trace_csv: Path,
    python: str = sys.executable,
) -> list[str]:
    """The ``dynamo.power.applier`` argv for ``group``, run on the bare host with the orchestrator's interpreter."""
    command = [
        python,
        "-m",
        APPLIER_MODULE,
        "--decision-file",
        str(decision_file),
        "--node",
        group.node,
        "--gpu-indices",
        group.gpu_list,
        "--mode",
        power.applier,
        "--budget-w-per-gpu",
        str(pool.spec.budget_w_per_gpu),
        "--stale-after-s",
        str(power.stale_after_s),
        "--poll-s",
        str(power.poll_s),
        "--restore-default-on-exit",
        "--trace-csv",
        str(trace_csv),
    ]
    if power.min_tgp_w is not None:
        # The per-GPU floor lives in the applier: EPD's NUPD cap law has none.
        command.extend(["--min-w-per-gpu", str(power.min_tgp_w)])
    return command


def _sudo_smi_reset(indices: list[int]) -> str:
    # The exact nvidia-smi argument order hecate's sudoers rule allows: nvidia-smi -pl <W> -i <N>.
    # `sudo -n` fails at once where the rule is missing instead of waiting on a password prompt
    # until the step's timeout; `rc` carries every GPU's outcome, not only the last one's.
    return (
        f"rc=0; for i in {' '.join(str(index) for index in indices)}; do "
        'd=$(nvidia-smi -i "$i" --query-gpu=power.default_limit --format=csv,noheader,nounits) && '
        'sudo -n nvidia-smi -pl "${d%%.*}" -i "$i" || rc=1; done; exit $rc'
    )


def _dcgmi_reset(indices: list[int]) -> str:
    # DCGM caps persist as hostengine target config: set the default on a throw-away group of these
    # GPUs. The group is deleted whatever `dcgmi config` returned; the exit status is that call's.
    gpu_list = ",".join(str(index) for index in indices)
    return (
        f"d=$(nvidia-smi -i {indices[0]} --query-gpu=power.default_limit --format=csv,noheader,nounits) || exit 1; "
        f"g=$(dcgmi group -c srtctl_power_reset -a {gpu_list} | grep -o '[0-9]*$') || exit 1; "
        '[ -n "$g" ] || exit 1; rc=0; dcgmi config -g "$g" --set -P "${d%%.*}" || rc=1; '
        'dcgmi group -d "$g"; exit $rc'
    )


# power_control.applier -> the bash that restores the hardware default limit of each GPU index.
_RESET_SCRIPTS: dict[str, Callable[[list[int]], str]] = {"sudo-smi": _sudo_smi_reset, "dcgmi": _dcgmi_reset}


def power_reset_script(applier: str, gpu_indices: frozenset[int]) -> str:
    """Bash that restores each GPU's hardware default power limit through the applier's privileged path.

    ``nvidia-smi --query-gpu=power.default_limit`` (read-only, unprivileged) supplies the
    default; the applier mode decides how it is written back.
    """
    return _RESET_SCRIPTS[applier](sorted(gpu_indices))


class PowerControlStageMixin:
    """Mixin for the EPD power-control stage.

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
        self.backend_processes: list[Process]
        self._build_worker_preamble(): WorkerStageMixin (the node_agent container installs Dynamo like a worker)
        self._compute_frontend_topology(): FrontendStageMixin (the frontend /metrics URL)
    """

    config: SrtConfig
    runtime: RuntimeContext
    _power_control_pools: tuple[PowerPool, ...] = ()
    _power_node_agents: tuple[ManagedProcess, ...] = ()
    _power_appliers: tuple[ManagedProcess, ...] = ()

    @property
    def backend_processes(self) -> list[Process]:
        """Backend worker processes (provided by the orchestrator)."""
        raise NotImplementedError

    def _build_worker_preamble(self) -> str | None:
        """Setup script and Dynamo install preamble (WorkerStageMixin)."""
        raise NotImplementedError

    def _compute_frontend_topology(self) -> Any:
        """Frontend topology helper (FrontendStageMixin)."""
        raise NotImplementedError

    def power_pools(self) -> list[PowerPool]:
        """The pools of this job, from the recipe block and the workers' GPU map."""
        return compute_power_pools(self.config.power_control, self.backend_processes)

    def validate_power_control_layout(self) -> list[PowerPool]:
        """Resolve the pools from the recipe and the GPU map at job start; a no-op unless enabled.

        ``SweepOrchestrator.run`` calls this before host setup and the workers, so a layout
        the launch would refuse (a node in two pools, an explicit node pool naming an idle
        node, a pool with no GPU) fails in seconds instead of after a model load that takes
        tens of minutes on a large model. The recipe loader already refuses a job without a
        metrics source (``frontend.type: none``); ``start_power_control`` resolves the pools
        again as the second gate.
        """
        if not self.config.power_control.enabled:
            return []
        pools = self.power_pools()
        for pool in pools:
            logger.info(
                "Power pool %s resolves to nodes=%s gpus=%d budget=%d W/GPU fom=%s",
                pool.name,
                ",".join(pool.nodes),
                pool.gpu_count,
                pool.spec.budget_w_per_gpu,
                pool.spec.fom,
            )
        return pools

    def _pool_dir(self, pool: PowerPool) -> Path:
        """The pool's EPD ``LOG_DIR`` on the host (shared filesystem)."""
        return self.runtime.log_dir / POWER_CONTROL_DIRNAME / pool.name

    def _container_pool_dir(self, pool: PowerPool) -> Path:
        """The same directory as the node_agent container sees it."""
        return self.runtime.container_log_dir / POWER_CONTROL_DIRNAME / pool.name

    def _node_ip(self, node: str, node_ips: dict[str, str]) -> str:
        """``get_hostname_ip`` once per node (one srun per call under Slurm), memoised in ``node_ips``."""
        if node not in node_ips:
            node_ips[node] = get_hostname_ip(node, self.runtime.network_interface)
        return node_ips[node]

    def _frontend_metrics_urls(self, node_ips: dict[str, str] | None = None) -> list[str]:
        """``/metrics`` of every frontend process (or direct endpoint); empty for a services-only job."""
        if self.config.frontend.type == FRONTEND_NONE:
            return []
        node_ips = {} if node_ips is None else node_ips
        frontend = get_frontend(self.config.frontend.type)
        topology = self._compute_frontend_topology()
        nodes = frontend.direct_endpoint_nodes(self.backend_processes) or list(topology.frontend_nodes)
        port = frontend.frontend_metrics_port(self.config.frontend.args) or topology.frontend_port
        return [f"http://{url_host(self._node_ip(node, node_ips))}:{port}{frontend.metrics_path}" for node in nodes]

    def _rank_serves_system_metrics(self, process: Process) -> bool:
        """Whether this rank runs a Dynamo system server: every rank of a per-node launch, the leader of an MPI step.

        A TRT-LLM worker is one MPI step across its nodes and the Dynamo main (the system
        port with ``/metrics``) runs on rank 0 only; its follower nodes would answer every
        scrape with connection refused.
        """
        if process.is_leader:
            return True
        return not self.config.backend_for_role(process.endpoint_mode).get_srun_config().launch_per_endpoint

    def _pool_worker_metrics_urls(self, pool: PowerPool, node_ips: dict[str, str] | None = None) -> list[str]:
        """The system-metrics URL of every rank in ``pool`` that serves one (the frontend's call), one IP lookup per node."""
        if self.config.frontend.type == FRONTEND_NONE:
            return []
        node_ips = {} if node_ips is None else node_ips
        frontend = get_frontend(self.config.frontend.type)
        urls: list[str] = []
        for process in pool.processes:
            if not self._rank_serves_system_metrics(process):
                continue
            port = frontend.worker_metrics_port(process, self.runtime)
            backend = self.config.backend_for_role(process.endpoint_mode)
            path = frontend.worker_metrics_path(backend, process.endpoint_mode)
            if port is None or path is None:
                continue
            urls.append(f"http://{url_host(self._node_ip(process.node, node_ips))}:{port}{path}")
        return list(dict.fromkeys(urls))

    def node_agent_mounts(self) -> dict[Path, Path]:
        """The job's mounts plus the read-only EPD overlay (and Dynamo's ``dynamo/power`` when given)."""
        power = self.config.power_control
        if power.epd_source is None:
            raise ValueError("power_control.epd_source is required to launch node_agents")
        site_packages = power.site_packages.rstrip("/")
        mounts = dict(self.runtime.container_mounts)
        # `:ro` rides on the container side of the mount spec, exactly as a recipe's extra_mount does.
        mounts[Path(os.path.expandvars(power.epd_source)).expanduser()] = Path(f"{site_packages}/epd:ro")
        if power.dynamo_power_source is not None:
            source = Path(os.path.expandvars(power.dynamo_power_source)).expanduser()
            mounts[source] = Path(f"{site_packages}/dynamo/power:ro")
        return mounts

    def _node_agent_image(self, group: GpuGroup) -> str:
        """The worker image of the group's (first) role, so the node_agent imports the same Dynamo."""
        role = group.roles[0]
        if role in self.config.role_containers:
            return self.config.worker_container_for_role(role)
        return str(self.runtime.container_image)

    def _start_node_agent(self, pool: PowerPool, group: GpuGroup, rank: int, metrics_urls: list[str]) -> ManagedProcess:
        power = self.config.power_control
        container_log_dir = self._container_pool_dir(pool)
        command = build_node_agent_command(
            power, pool, group, rank, metrics_urls, container_log_dir, self.runtime.gpus_per_node
        )
        # The recipe environment with the workers' {node}/{node_id} templating, then EPD's own variables.
        env = {
            **render_node_environment(
                self.runtime.environment, node=group.node, node_id=self.runtime.nodes.worker.index(group.node)
            ),
            **node_agent_environment(pool, container_log_dir, self.runtime.gpus_per_node),
        }
        if len(group.gpu_indices) < self.runtime.gpus_per_node:
            # EPD's NVML monitor senses every GPU it can see (no index or UUID filter), so a
            # group that is only part of the node (a TEP6 tail) gets a container that sees
            # exactly those GPUs; the node_agent checks that it sees exactly the group's GPU count.
            env["NVIDIA_VISIBLE_DEVICES"] = group.gpu_list
        step_name = f"power_node_agent_{pool.name}_{group.node}"
        log_file = self._pool_dir(pool) / f"node_agent_{group.node}.out"
        logger.info(
            "Starting power node_agent for pool %s on %s (rank %d, GPUs %s): %s",
            pool.name,
            group.node,
            rank,
            group.gpu_list,
            shlex.join(command),
        )
        popen = start_srun_process(
            command=command,
            nodelist=[group.node],
            output=str(log_file),
            container_image=self._node_agent_image(group),
            container_mounts=self.node_agent_mounts(),
            env_to_set=env,
            bash_preamble=self._build_worker_preamble(),
            srun_options=self.runtime.srun_options,
            srun_export_env=CONTAINER_REMAP_ROOT_EXPORT if installs_dynamo(self.config) else None,
            het_group=group.het_group,
            step_name=step_name,
        )
        return ManagedProcess(
            name=step_name,
            popen=popen,
            log_file=log_file,
            node=group.node,
            critical=power.critical,
            # SIGTERM through the step: EPD's stop finishes an in-flight cycle (interval_s + 5 s), does
            # its final readback and writes run_summary.json, so the budget follows the cadence.
            terminate_timeout=node_agent_terminate_timeout(power.interval_s),
            step_name=step_name,
            shutdown_tier=0,
        )

    def _start_applier(self, pool: PowerPool, group: GpuGroup) -> ManagedProcess:
        power = self.config.power_control
        pool_dir = self._pool_dir(pool)
        command = build_applier_command(
            power,
            pool,
            group,
            decision_file=pool_dir / DECISION_FILE_RELPATH,
            trace_csv=pool_dir / f"applier_{group.node}.csv",
        )
        env: dict[str, str] = {}
        source_root = power.dynamo_power_source_root
        if source_root is not None:
            inherited = os.environ.get("PYTHONPATH")
            env["PYTHONPATH"] = f"{source_root}:{inherited}" if inherited else source_root
        step_name = f"power_applier_{group.node}"
        log_file = pool_dir / f"applier_{group.node}.out"
        logger.info(
            "Starting power applier on %s (pool %s, GPUs %s): %s",
            group.node,
            pool.name,
            group.gpu_list,
            shlex.join(command),
        )
        popen = start_srun_process(
            command=command,
            nodelist=[group.node],
            output=str(log_file),
            container_image=None,  # bare host: in-container nvidia-smi -pl is refused and dcgmi lives on the host
            env_to_set=env or None,
            srun_options=self.runtime.srun_options,
            het_group=group.het_group,
            step_name=step_name,
        )
        return ManagedProcess(
            name=step_name,
            popen=popen,
            log_file=log_file,
            node=group.node,
            critical=power.critical,
            # SIGTERM through the step: the applier restores the hardware default limits before exiting.
            terminate_timeout=POWER_TERMINATE_TIMEOUT_SECONDS,
            step_name=step_name,
            # A tier behind the node_agents (and the workers), so any cleanup path lifts the caps last.
            shutdown_tier=1,
        )

    def start_power_control(self, registry: ProcessRegistry) -> list[ManagedProcess]:
        """Launch the appliers and node_agents of every pool; a no-op unless ``power_control.enabled``.

        Called once the workers are healthy and before the benchmark starts. Every step
        is registered the moment it exists, so a failure mid-launch still leaves the
        partial launch reachable by the registry's cleanup.
        """
        power = self.config.power_control
        if not power.enabled:
            return []
        pools = self.power_pools()
        node_ips: dict[str, str] = {}  # one IP lookup (one srun) per node across the frontend and every pool
        frontend_urls = self._frontend_metrics_urls(node_ips)
        node_agents: list[ManagedProcess] = []
        appliers: list[ManagedProcess] = []
        self._power_control_pools = tuple(pools)
        for pool in pools:
            pool_dir = self._pool_dir(pool)
            (pool_dir / DECISION_FILE_RELPATH.parent).mkdir(parents=True, exist_ok=True)
            # Tagged so a FoM scope (frontend / pool_workers) selects the right endpoints.
            metrics_urls = [
                *(f"frontend={url}" for url in frontend_urls),
                *(f"worker={url}" for url in self._pool_worker_metrics_urls(pool, node_ips)),
            ]
            logger.info(
                "Power pool %s: mode=%s allocation=%s nodes=%s gpus=%d budget=%d W/GPU fom=%s ubpc_buffer=%s "
                "node_buffers=%s log_dir=%s",
                pool.name,
                power.mode,
                power.allocation,
                ",".join(pool.nodes),
                pool.gpu_count,
                pool.spec.budget_w_per_gpu,
                pool.spec.fom,
                pool.ubpc_buffer,
                pool.node_buffers,
                pool_dir,
            )
            # Appliers first: they idle until the controller's first decision lands.
            for group in pool.groups:
                applier = self._start_applier(pool, group)
                registry.add_process(applier)
                appliers.append(applier)
                self._power_appliers = tuple(appliers)
            for rank, group in enumerate(pool.groups):
                node_agent = self._start_node_agent(pool, group, rank, metrics_urls)
                registry.add_process(node_agent)
                node_agents.append(node_agent)
                self._power_node_agents = tuple(node_agents)
        logger.info(
            "Power control started: %d pool(s), %d node_agent(s), %d applier(s)",
            len(pools),
            len(node_agents),
            len(appliers),
        )
        return [*appliers, *node_agents]

    @staticmethod
    def _stop_power_steps(processes: Sequence[ManagedProcess], step_ids: dict[str, str] | None, kind: str) -> None:
        """SIGTERM a group of steps together, then wait for each (the registry's group-shutdown shape).

        The steps are demoted from ``critical`` first: ``stop_power_control`` runs from the
        orchestrator's ``finally`` block before ``stop_event`` is set, so the process monitor is
        still polling ``check_failures`` while these steps exit with 143 on our SIGTERM. Without
        the demotion a fast exit (a snapshot-only node_agent) is logged as "Critical process
        failure detected!" and the monitor starts a second, concurrent ``registry.cleanup``
        (seen on hecate job 774072; the sweep still completed). An exit we asked for is not a
        failure.
        """
        running = [process for process in processes if process.is_running]
        for process in running:
            process.critical = False
            process.request_stop(step_ids)
        for process in running:
            process.await_stop()
            if process.exit_code in (None, -9):
                logger.warning("power %s %s did not exit after SIGTERM and was killed", kind, process.name)
            else:
                logger.info("power %s %s stopped", kind, process.name)

    def stop_power_control(self) -> None:
        """Lift the caps on the way out: node_agents, then appliers, then a per-node reset. Never raises.

        Runs from the orchestrator's cleanup path before the registry stops the workers,
        so the GPUs are back at their hardware default limits while the engines are still
        up. The order matters: a node_agent stopped after its applier would leave a decision
        file that nothing applies, and an applier stopped before the final reset is the
        one that restores the defaults in the common case; the reset is the safety net.
        """
        if not self._power_control_pools:
            return
        steps = [*self._power_node_agents, *self._power_appliers]
        try:
            step_ids = list_step_ids(self.runtime.job_id) if any(step.is_running for step in steps) else None
            self._stop_power_steps(self._power_node_agents, step_ids, "node_agent")
            self._stop_power_steps(self._power_appliers, step_ids, "applier")
        except (OSError, subprocess.SubprocessError):
            logger.exception("Stopping the power-control steps failed; running the per-node power-limit reset anyway")
        self._report_missing_decision_files()
        self._reset_power_limits()

    def _report_missing_decision_files(self) -> None:
        """Error for a pool whose controller never published a decision: a static-cap run, not an EPD run.

        EPD publishes its uniform starting caps to the decision file when it starts (cycle 0,
        rank 0), so a pool directory without it after a run means the node_agent never started
        EPD as the controller, or handed EPD a different ``LOG_DIR`` than the one the appliers
        poll; the appliers then held the uniform budget throughout and, as nothing exited,
        ``critical`` never fired.
        """
        for pool in self._power_control_pools:
            decision_file = self._pool_dir(pool) / DECISION_FILE_RELPATH
            if decision_file.exists():
                continue
            logger.error(
                "Power pool %s: EPD never wrote %s, so the appliers held the uniform budget (%d W/GPU) for the whole "
                "run: a static-cap run, not an EPD run. Check node_agent_%s.out (rank 0): the controller must start "
                "EPD with --log-dir handed over as LOG_DIR unchanged",
                pool.name,
                decision_file,
                pool.spec.budget_w_per_gpu,
                pool.nodes[0],
            )

    def _reset_power_limits(self) -> None:
        """Restore every controlled GPU's hardware default limit from the bare host, one srun per node."""
        power = self.config.power_control
        started: list[tuple[str, subprocess.Popen, Path]] = []
        for pool in self._power_control_pools:
            for group in pool.groups:
                log_file = self._pool_dir(pool) / f"power_reset_{group.node}.out"
                try:
                    popen = start_srun_process(
                        command=["bash", "-c", power_reset_script(power.applier, group.gpu_indices)],
                        nodelist=[group.node],
                        output=str(log_file),
                        container_image=None,  # bare host, like the applier
                        srun_options=self.runtime.srun_options,
                        het_group=group.het_group,
                        step_name=f"power_reset_{group.node}",
                    )
                except (OSError, subprocess.SubprocessError):
                    logger.exception("Could not launch the power-limit reset on %s", group.node)
                    continue
                started.append((group.node, popen, log_file))
        for node, popen, log_file in started:
            try:
                returncode = popen.wait(timeout=POWER_RESET_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                popen.kill()
                popen.wait()
                logger.error(
                    "Power-limit reset timed out after %ds on %s (see %s)", POWER_RESET_TIMEOUT_SECONDS, node, log_file
                )
                continue
            if returncode != 0:
                logger.error(
                    "Power-limit reset exited %d on %s (see %s); the node may still be capped",
                    returncode,
                    node,
                    log_file,
                )
            else:
                logger.info("Power limits restored to the hardware defaults on %s", node)
