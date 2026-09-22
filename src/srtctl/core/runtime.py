# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime context and node configuration.

This module provides the single source of truth for all runtime values,
replacing scattered bash variables and Jinja templating with typed Python.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.power.contract import CONTAINER_LOG_DIR
from srtctl.ports import FRONTEND_PUBLIC_PORT

from .config import get_srtslurm_setting
from .slurm import get_hostname_ip, get_slurm_het_nodelists, get_slurm_nodelist

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from srtctl.core.schema import DynamoConfig, SrtConfig


@dataclass(frozen=True)
class Nodes:
    """Node allocation for head, benchmark, infra, and worker nodes.

    Attributes:
        head: Head node hostname (runs nginx, frontends)
        bench: Benchmark node hostname (runs the benchmark client)
        infra: Infrastructure node hostname (runs NATS, etcd). Same as head unless
               etcd_nats_dedicated_node is enabled.
        worker: Tuple of the engine worker node hostnames (prefill + decode + agg)
        pools: Nodes owned by services (``services[].nodes``), by service name, in
             declaration order. Carved after the engine worker nodes. Empty for
             recipes without node-owning services.
        het: True when the job was submitted as a SLURM heterogeneous job. In
             this mode worker srun calls need ``--het-group=<group>`` so SLURM
             routes them to the right component.
        prefill_group: Worker nodes that belong to het component 0 (prefill +
             optionally the dedicated infra node). Empty tuple when het=False.
        decode_group: Worker nodes that belong to het component 1 (decode).
             Empty tuple when het=False.
    """

    head: str
    bench: str
    infra: str
    worker: tuple[str, ...]
    het: bool = False
    prefill_group: tuple[str, ...] = ()
    decode_group: tuple[str, ...] = ()
    pools: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def compute(self) -> tuple[str, ...]:
        """Every node that runs work: the engine worker nodes, then each pool, in allocation order."""
        seen: dict[str, None] = dict.fromkeys(self.worker)
        for nodes in self.pools.values():
            seen.update(dict.fromkeys(nodes))
        return tuple(seen)

    def het_group_for(self, node: str) -> int | None:
        """Return the het component (0 or 1) a node belongs to, or None.

        Returns None for non-het jobs so callers can pass the result directly
        to ``start_srun_process(het_group=...)`` as a no-op fallback.
        """
        if not self.het:
            return None
        if node in self.prefill_group:
            return 0
        if node in self.decode_group:
            return 1
        # Head and infra share group 0 under het (infra is folded into the
        # prefill component, head sits on the prefill side).
        if node == self.infra or node == self.head:
            return 0
        return None

    @classmethod
    def from_slurm(
        cls,
        frontend_dedicated_node: bool = False,
        client_dedicated_node: bool = False,
        etcd_nats_dedicated_node: bool = False,
        colocate_dedicated_nodes: bool = True,
        engine_nodes: int | None = None,
        pools: Sequence[tuple[str, int]] = (),
    ) -> "Nodes":
        """Create Nodes from SLURM environment.

        Args:
            frontend_dedicated_node: If True, reserve a node exclusively for the
                                     frontend/orchestrator; it is excluded from
                                     the worker pool.
            client_dedicated_node: If True, reserve a node exclusively for the
                                   benchmark client; it is excluded from the
                                   worker pool. Reserved from the tail of the
                                   nodelist (never the first node), since SLURM
                                   runs the do_sweep batch script unsandboxed
                                   on the first node and co-locating the
                                   benchmark client there would undermine the
                                   isolation this flag exists to provide.
            etcd_nats_dedicated_node: If True, reserve a node exclusively for
                                      etcd/nats.
            colocate_dedicated_nodes: Governs how the dedicated-node flags above
                                      combine when more than one is set. If True
                                      (default), every requested role (infra,
                                      frontend, client) shares a single reserved
                                      node. If False, each requested role gets
                                      its own reserved node. A role that is not
                                      requested keeps its normal default
                                      placement (frontend/client fall back to
                                      colocating with whichever node ends up
                                      being head; infra falls back to head).
            engine_nodes: How many of the non-reserved nodes the engine roles
                          own. Required when ``pools`` is given; None keeps the
                          legacy behavior where every non-reserved node is a
                          worker node.
            pools: ``(service name, node count)`` pairs for services that own
                   nodes, carved after the engine worker nodes in this order.
        """
        dedicated_roles = [
            role
            for role, wanted in (
                ("infra", etcd_nats_dedicated_node),
                ("frontend", frontend_dedicated_node),
                ("client", client_dedicated_node),
            )
            if wanted
        ]

        het_lists = get_slurm_het_nodelists()
        if het_lists is not None:
            if frontend_dedicated_node or client_dedicated_node:
                raise ValueError(
                    "frontend_dedicated_node/client_dedicated_node are not supported for heterogeneous SLURM jobs"
                )
            if pools:
                raise ValueError("services[].nodes (pools) are not supported for heterogeneous SLURM jobs")
            return cls._from_het_slurm(het_lists, etcd_nats_dedicated_node)

        nodelist = get_slurm_nodelist()
        if not nodelist:
            raise RuntimeError("SLURM_NODELIST not set - are we running in SLURM?")

        if not dedicated_roles:
            head = bench = infra = nodelist[0]
            worker, carved = cls._carve_pools(tuple(nodelist), engine_nodes, pools)
            return cls(head=head, bench=bench, infra=infra, worker=worker, pools=carved)

        num_reserved = 1 if colocate_dedicated_nodes else len(dedicated_roles)
        if len(nodelist) <= num_reserved:
            raise ValueError(
                f"dedicated node(s) for {'+'.join(dedicated_roles)} require at least {num_reserved + 1} nodes"
            )

        # SLURM runs the batch script (the do_sweep orchestrator) on the first
        # node of the allocation, unsandboxed. A dedicated *client* node exists
        # to isolate benchmark measurements from noisy neighbors, so it must
        # never land on that first node — reserve it from the tail instead.
        # Non-client roles (infra, frontend) keep the original front-of-list
        # reservation for backward compatibility.
        has_client = "client" in dedicated_roles
        if colocate_dedicated_nodes:
            if has_client:
                shared = nodelist[-1]
                worker = tuple(nodelist[:-1])
            else:
                shared = nodelist[0]
                worker = tuple(nodelist[1:])
            reserved = {role: shared for role in dedicated_roles}
        else:
            front_roles = [role for role in dedicated_roles if role != "client"]
            reserved = dict(zip(front_roles, nodelist, strict=False))
            if has_client:
                reserved["client"] = nodelist[-1]
                worker = tuple(nodelist[len(front_roles) : -1])
            else:
                worker = tuple(nodelist[len(front_roles) :])

        worker, carved = cls._carve_pools(tuple(worker), engine_nodes, pools)
        first_compute = (worker or tuple(n for nodes in carved.values() for n in nodes))[0]
        head = reserved.get("frontend", first_compute)
        bench = reserved.get("client", head)
        infra = reserved.get("infra", head)

        return cls(head=head, bench=bench, infra=infra, worker=worker, pools=carved)

    @staticmethod
    def planned_role_indices(
        total_nodes: int,
        *,
        frontend_dedicated_node: bool = False,
        client_dedicated_node: bool = False,
        etcd_nats_dedicated_node: bool = False,
        colocate_dedicated_nodes: bool = True,
    ) -> tuple[int, int]:
        """Where ``from_slurm`` will put the head (frontend) and the benchmark client.

        Positions in the allocation's nodelist, before the job exists: the same carving
        rules as :meth:`from_slurm`, applied to indices instead of hostnames, so a
        launcher that submits a rendered script can tell its own client where the
        endpoint is. Pools are not modelled (the head is the first engine node either
        way). Returns ``(head_index, client_index)``.
        """
        dedicated_roles = [
            role
            for role, wanted in (
                ("infra", etcd_nats_dedicated_node),
                ("frontend", frontend_dedicated_node),
                ("client", client_dedicated_node),
            )
            if wanted
        ]
        if not dedicated_roles:
            return 0, 0
        num_reserved = 1 if colocate_dedicated_nodes else len(dedicated_roles)
        if total_nodes <= num_reserved:
            raise ValueError(
                f"dedicated node(s) for {'+'.join(dedicated_roles)} require at least {num_reserved + 1} nodes"
            )
        last = total_nodes - 1
        has_client = "client" in dedicated_roles
        if colocate_dedicated_nodes:
            shared = last if has_client else 0
            reserved = {role: shared for role in dedicated_roles}
            first_worker = 0 if has_client else 1
        else:
            front_roles = [role for role in dedicated_roles if role != "client"]
            reserved = dict(zip(front_roles, range(len(front_roles)), strict=False))
            if has_client:
                reserved["client"] = last
            first_worker = len(front_roles)
        head = reserved.get("frontend", first_worker)
        return head, reserved.get("client", head)

    @staticmethod
    def _carve_pools(
        remaining: tuple[str, ...], engine_nodes: int | None, pools: Sequence[tuple[str, int]]
    ) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
        """Split the non-reserved nodes into the engine worker nodes and the service pools.

        Legacy recipes (no pools) keep every node as a worker node. With pools,
        the engine roles take the first ``engine_nodes`` nodes and each pool the
        next ``count`` in declaration order; the allocation must be large enough.
        """
        if not pools:
            return remaining, {}
        if engine_nodes is None:
            raise ValueError("engine_nodes is required when pools are declared")
        needed = engine_nodes + sum(count for _, count in pools)
        if len(remaining) < needed:
            raise ValueError(
                f"allocation has {len(remaining)} non-reserved node(s) but the recipe needs {needed}: "
                f"{engine_nodes} for engine roles plus pools " + ", ".join(f"{n}={c}" for n, c in pools)
            )
        worker = remaining[:engine_nodes]
        carved: dict[str, tuple[str, ...]] = {}
        cursor = engine_nodes
        for name, count in pools:
            carved[name] = remaining[cursor : cursor + count]
            cursor += count
        return worker, carved

    @classmethod
    def _from_het_slurm(
        cls,
        het_lists: list[list[str]],
        etcd_nats_dedicated_node: bool,
    ) -> "Nodes":
        """Carve a Nodes from a SLURM heterogeneous-job allocation.

        Group 0 holds prefill (and the dedicated infra node when configured);
        group 1 holds decode. Head/bench live on group 0.
        """
        if len(het_lists) != 2:
            raise ValueError(
                f"het_jobs expects exactly 2 components (prefill, decode); SLURM_HET_SIZE reported {len(het_lists)}"
            )
        group0, group1 = het_lists
        if not group0 or not group1:
            raise RuntimeError("Empty SLURM_JOB_NODELIST_HET_GROUP_* — are we inside a het job?")

        if etcd_nats_dedicated_node:
            if len(group0) < 2:
                raise ValueError("etcd_nats_dedicated_node requires >= 2 nodes in het group 0")
            infra = group0[0]
            head = group0[1]
            prefill_group = tuple(group0[1:])
        else:
            infra = group0[0]
            head = group0[0]
            prefill_group = tuple(group0)
        bench = head
        decode_group = tuple(group1)
        worker = prefill_group + decode_group
        return cls(
            head=head,
            bench=bench,
            infra=infra,
            worker=worker,
            het=True,
            prefill_group=prefill_group,
            decode_group=decode_group,
        )


@dataclass(frozen=True)
class RuntimeContext:
    """Runtime context with all computed values.

    This is the single source of truth for all runtime values and paths.
    All paths are absolute Path objects. Created via from_config() classmethod.
    """

    # Runtime identifiers
    job_id: str
    run_name: str

    # Node topology
    nodes: Nodes
    head_node_ip: str
    infra_node_ip: str

    # Computed paths (all absolute)
    log_dir: Path
    model_path: Path  # For HF models (hf:prefix), this is the HF model ID as a Path
    container_image: Path

    # Resource configuration
    gpus_per_node: int
    network_interface: str | None

    # Fields with defaults must come after required fields
    # HuggingFace model support - True if model.path was "hf:model/name"
    is_hf_model: bool = False
    gpu_type: str | None = None
    visible_devices_env: str = "CUDA_VISIBLE_DEVICES"

    # Container mounts: host_path -> container_path
    container_mounts: dict[Path, Path] = field(default_factory=dict)

    # Additional srun options
    srun_options: dict[str, str] = field(default_factory=dict)

    # Environment variables
    environment: dict[str, str] = field(default_factory=dict)

    # Frontend port (for benchmark endpoint)
    frontend_port: int = FRONTEND_PUBLIC_PORT

    # Optional lustre->node-local model staging (see model.stage_dir)
    stage_dir: str | None = None
    staged_model_path: Path | None = None
    # Request plane for dynamo workers
    request_plane: str = "tcp"
    # Full Dynamo configuration for native sidecar launch settings.
    dynamo: "DynamoConfig | None" = None

    @property
    def container_log_dir(self) -> Path:
        """``log_dir`` as processes inside the container see it.

        ``from_config`` mounts the run's log directory at ``CONTAINER_LOG_DIR``;
        this follows that mount so a remapped log mount needs no other change.
        Every path handed to a containerized process (config dumps, profiler
        output, fingerprints, benchmark artifacts) is built from this, never
        from the host ``log_dir``, which is not visible in the container on
        every cluster.
        """
        return self.container_mounts.get(self.log_dir, Path(CONTAINER_LOG_DIR))

    @classmethod
    def from_config(
        cls,
        config: "SrtConfig",
        job_id: str,
        log_dir_base: Path | None = None,
    ) -> "RuntimeContext":
        """Create RuntimeContext from config and job_id.

        All path computation happens here, once at startup.

        Args:
            config: Validated SrtConfig (frozen dataclass)
            job_id: SLURM job ID
            log_dir_base: Base directory for logs (default: ./outputs)
        """
        # Get nodes from SLURM
        pools = [(svc.name, svc.nodes) for svc in config.pool_services if svc.nodes is not None]
        nodes = Nodes.from_slurm(
            frontend_dedicated_node=config.frontend.dedicated_node,
            client_dedicated_node=config.benchmark.client_dedicated_node,
            etcd_nats_dedicated_node=config.infra.etcd_nats_dedicated_node,
            colocate_dedicated_nodes=config.benchmark.colocate_with_frontend,
            engine_nodes=config.engine_node_count if pools else None,
            pools=pools,
        )

        # Compute run_name
        run_name = f"{config.name}_{job_id}"

        # Resolve node IPs on the cluster-selected fabric. Some systems expose
        # a public default route and a separate private control/data plane; the
        # latter is what containers on peer Slurm nodes can reliably reach.
        network_interface = get_srtslurm_setting("network_interface", "eth0")
        head_node_ip = get_hostname_ip(nodes.head, network_interface)
        infra_node_ip = get_hostname_ip(nodes.infra, network_interface)

        # Compute log directory using FormattablePath or default logic
        # Check for SRTCTL_OUTPUT_DIR from sbatch script first (ensures consistency)
        output_dir_env = os.environ.get("SRTCTL_OUTPUT_DIR")
        if output_dir_env:
            log_dir = Path(output_dir_env) / "logs"
        elif log_dir_base is None:
            log_dir_base = Path.cwd() / "outputs"
            log_dir = log_dir_base / job_id / "logs"
        else:
            log_dir = log_dir_base / job_id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Resolve model path (expand env vars)
        # Support HuggingFace model IDs with "hf:" prefix (e.g., "hf:facebook/opt-125m")
        model_path_str = os.path.expandvars(config.model.path)
        is_hf_model = model_path_str.startswith("hf:")

        if is_hf_model:
            # HuggingFace model ID - store as Path for compatibility, skip validation
            hf_model_id = model_path_str[3:]  # Remove "hf:" prefix
            model_path = Path(hf_model_id)
        else:
            # Local path - validate exists
            model_path = Path(model_path_str).resolve()
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            if not model_path.is_dir():
                raise ValueError(f"Model path is not a directory: {model_path}")

        # Resolve container image (expand env vars)
        # container_image can be either:
        # 1. A path to a container file (e.g., /containers/sglang.sqsh) - validate it exists
        # 2. An image name (e.g., nvcr.io/nvidia/pytorch:23.12) - don't validate
        container_image_str = os.path.expandvars(config.model.container)

        # If it looks like a file path (starts with / or ./), validate it exists
        # Image names are typically registry paths without leading / or ./
        if container_image_str.startswith(("/", "./")):
            container_image = Path(container_image_str).resolve()
            if not container_image.exists():
                raise FileNotFoundError(f"Container image path does not exist: {container_image}")
            if not container_image.is_file():
                raise ValueError(f"Container image path is not a file: {container_image}")
        else:
            # Image name (e.g., nvcr.io/nvidia/pytorch:23.12) - keep as string, convert to Path for type compatibility
            container_image = Path(container_image_str)

        # Build container mounts
        container_mounts: dict[Path, Path] = {
            log_dir: Path(CONTAINER_LOG_DIR),
        }
        # Only mount local model paths - HF models are downloaded at runtime
        if not is_hf_model:
            container_mounts[model_path] = Path("/model")

        # Optional: stage the model to node-local storage before workers start.
        # Mount the node-local ROOT (parent of stage_dir; it pre-exists on nodes)
        # so the staged copy is visible in-container at its real path; workers read
        # <stage_dir>/<model_name> instead of the /model mount. See _stage_model().
        stage_dir: str | None = None
        staged_model_path: Path | None = None
        if not is_hf_model and config.model.stage_dir:
            stage_dir = os.path.expandvars(config.model.stage_dir)
            staged_model_path = Path(stage_dir) / model_path.name
            container_mounts[Path(stage_dir).parent] = Path(stage_dir).parent

        # Add configs directory (NATS, etcd binaries) from source root
        # SRTCTL_SOURCE_DIR is set by the sbatch script
        source_dir = os.environ.get("SRTCTL_SOURCE_DIR")
        if source_dir:
            configs_dir = Path(source_dir) / "configs"
            if configs_dir.exists():
                container_mounts[configs_dir.resolve()] = Path("/configs")

            # Repo-root benchmarks/: launchers and clients that are not core (RL frameworks
            # under benchmarks/rl/). Recipes run them as custom benchmark commands by their
            # container path, /benchmarks/<folder>/launch.sh.
            benchmarks_dir = Path(source_dir) / "benchmarks"
            if benchmarks_dir.exists():
                container_mounts[benchmarks_dir.resolve()] = Path("/benchmarks")

            wheelhouse_dir = Path(source_dir) / "wheelhouse" / "dynamo"
            if wheelhouse_dir.exists():
                container_mounts[wheelhouse_dir.resolve()] = Path("/srtctl-wheels")

        runtime_scripts_dir = Path(__file__).resolve().parent.parent / "runtime_scripts"
        if runtime_scripts_dir.exists():
            container_mounts[runtime_scripts_dir.resolve()] = Path("/srtctl-runtime")

        # Mount srtctl benchmark scripts
        from srtctl.benchmarks.base import SCRIPTS_DIR

        if SCRIPTS_DIR.exists():
            container_mounts[SCRIPTS_DIR.resolve()] = Path("/srtctl-benchmarks")

        # Add cluster-level mounts from srtslurm.yaml
        cluster_mounts = get_srtslurm_setting("default_mounts")
        if cluster_mounts:
            for host_path, container_path in cluster_mounts.items():
                expanded_host = os.path.expandvars(host_path)
                container_mounts[Path(expanded_host).resolve()] = Path(container_path)

        # Add extra mounts from config. Sources are resolved, so two entries can collapse
        # into one (on clusters where e.g. /lustre is a symlink onto /scratch) and a later
        # entry silently replaces an earlier one's container path. Say so instead.
        if config.extra_mount:
            for mount_spec in config.extra_mount:
                host_path, container_path = mount_spec.split(":", 1)
                expanded_host = os.path.expandvars(host_path)
                resolved_host = Path(expanded_host).expanduser().resolve()
                previous = container_mounts.get(resolved_host)
                if previous is not None and previous != Path(container_path):
                    logger.warning(
                        "extra_mount %r resolves to %s, already mounted at %s; the container will see it at %s only",
                        mount_spec,
                        resolved_host,
                        previous,
                        container_path,
                    )
                container_mounts[resolved_host] = Path(container_path)

        # Mount InferenceX workspace if available (for lm-eval support).
        # Skip exists() check: the orchestrator runs on the SLURM head node
        # where the GH Actions workspace path may not be directly accessible,
        # but it IS accessible from compute nodes via shared filesystem.
        infmax_ws = os.environ.get("INFMAX_WORKSPACE")
        if infmax_ws:
            container_mounts[Path(infmax_ws)] = Path("/infmax-workspace")

        # Add FormattablePath mounts from config.container_mounts
        # These need to be expanded with the runtime context, so we create a
        # temporary context first and then update
        environment = config.dynamo.get_wheel_environment()
        environment.update(config.environment)

        visible_devices_env = get_srtslurm_setting("visible_devices_env", "CUDA_VISIBLE_DEVICES")
        temp_context = cls(
            job_id=job_id,
            run_name=run_name,
            nodes=nodes,
            head_node_ip=head_node_ip,
            infra_node_ip=infra_node_ip,
            log_dir=log_dir,
            model_path=model_path,
            container_image=container_image,
            gpus_per_node=config.resources.gpus_per_node,
            gpu_type=config.resources.gpu_type,
            network_interface=network_interface,
            visible_devices_env=visible_devices_env,
            container_mounts={},
            srun_options=dict(config.srun_options),
            environment=environment,
            is_hf_model=is_hf_model,
            request_plane=config.dynamo.request_plane,
            dynamo=config.dynamo,
        )

        # Expand FormattablePath mounts
        for host_template, container_template in config.container_mounts.items():
            host_path = host_template.get_path(temp_context, ensure_exists=False)
            container_path = container_template.get_path(temp_context, make_absolute=False, ensure_exists=False)
            container_mounts[host_path] = container_path

        return cls(
            job_id=job_id,
            run_name=run_name,
            nodes=nodes,
            head_node_ip=head_node_ip,
            infra_node_ip=infra_node_ip,
            log_dir=log_dir,
            model_path=model_path,
            container_image=container_image,
            gpus_per_node=config.resources.gpus_per_node,
            gpu_type=config.resources.gpu_type,
            network_interface=network_interface,
            visible_devices_env=visible_devices_env,
            container_mounts=container_mounts,
            srun_options=dict(config.srun_options),
            environment=environment,
            is_hf_model=is_hf_model,
            stage_dir=stage_dir,
            staged_model_path=staged_model_path,
            request_plane=config.dynamo.request_plane,
            dynamo=config.dynamo,
        )

    @property
    def worker_model_arg(self) -> str:
        """Model path passed to the serving worker: the staged node-local path
        when model staging is enabled, else the "/model" mount (or the HF id)."""
        if self.is_hf_model:
            return str(self.model_path)
        if self.staged_model_path is not None:
            return str(self.staged_model_path)
        return "/model"

    def format_string(self, template: str, **extra_kwargs) -> str:
        """Format a template string with runtime values.

        Available placeholders:
            {job_id}, {run_name}, {head_node_ip}, {log_dir},
            {model_path}, {container_image}, plus any extra_kwargs.
        """
        format_dict = {
            "job_id": self.job_id,
            "run_name": self.run_name,
            "head_node_ip": self.head_node_ip,
            "log_dir": str(self.log_dir),
            "model_path": str(self.model_path),
            "container_image": str(self.container_image),
            "gpus_per_node": self.gpus_per_node,
        }
        format_dict.update(extra_kwargs)

        try:
            formatted = template.format(**format_dict)
        except KeyError as e:
            missing_key = str(e).strip("'\"")
            available_keys = sorted(set(format_dict.keys()))
            raise KeyError(
                f"Missing placeholder '{missing_key}' in template. Available placeholders: {', '.join(available_keys)}."
            ) from e
        return os.path.expandvars(formatted)
