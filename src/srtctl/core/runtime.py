# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime context and node configuration.

This module provides the single source of truth for all runtime values,
replacing scattered bash variables and Jinja templating with typed Python.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config import get_srtslurm_setting
from .slurm import get_hostname_ip, get_slurm_nodelist

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig


@dataclass(frozen=True)
class Nodes:
    """Node allocation for head, benchmark, infra, and worker nodes.

    Attributes:
        head: Head node hostname (runs nginx, frontends)
        bench: Benchmark node hostname (runs the benchmark client)
        infra: Infrastructure node hostname (runs NATS, etcd). Same as head unless
               etcd_nats_dedicated_node is enabled.
        worker: Tuple of all worker node hostnames (prefill + decode)
    """

    head: str
    bench: str
    infra: str
    worker: tuple[str, ...]

    @classmethod
    def from_slurm(
        cls,
        benchmark_on_separate_node: bool = False,
        etcd_nats_dedicated_node: bool = False,
    ) -> "Nodes":
        """Create Nodes from SLURM environment.

        Args:
            benchmark_on_separate_node: If True, first node is benchmark-only,
                                        second is head, rest are workers.
            etcd_nats_dedicated_node: If True, dedicate first node for etcd/nats,
                                      second node is head, rest are workers.
        """
        nodelist = get_slurm_nodelist()
        if not nodelist:
            raise RuntimeError("SLURM_NODELIST not set - are we running in SLURM?")

        if etcd_nats_dedicated_node:
            if len(nodelist) < 2:
                raise ValueError("etcd_nats_dedicated_node requires at least 2 nodes")
            infra = nodelist[0]
            head = nodelist[1]
            bench = head
            worker = tuple(nodelist[1:])
        elif benchmark_on_separate_node:
            if len(nodelist) < 2:
                raise ValueError("benchmark_on_separate_node requires at least 2 nodes")
            bench = nodelist[0]
            head = nodelist[1]
            infra = head
            worker = tuple(nodelist[1:])
        else:
            head = nodelist[0]
            bench = head
            infra = head
            worker = tuple(nodelist[:])

        return cls(head=head, bench=bench, infra=infra, worker=worker)


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

    # Container mounts: host_path -> container_path
    container_mounts: dict[Path, Path] = field(default_factory=dict)

    # Additional srun options
    srun_options: dict[str, str] = field(default_factory=dict)

    # Environment variables
    environment: dict[str, str] = field(default_factory=dict)

    # Frontend port (for benchmark endpoint)
    frontend_port: int = 8000

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
        nodes = Nodes.from_slurm(
            benchmark_on_separate_node=False,
            etcd_nats_dedicated_node=config.infra.etcd_nats_dedicated_node,
        )

        # Compute run_name
        run_name = f"{config.name}_{job_id}"

        # Resolve node IPs
        head_node_ip = get_hostname_ip(nodes.head)
        infra_node_ip = get_hostname_ip(nodes.infra)

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
        if container_image_str.startswith("/") or container_image_str.startswith("./"):
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
            log_dir: Path("/logs"),
        }
        # Only mount local model paths - HF models are downloaded at runtime
        if not is_hf_model:
            container_mounts[model_path] = Path("/model")

        # Add configs directory (NATS, etcd binaries) from source root
        # SRTCTL_SOURCE_DIR is set by the sbatch script
        source_dir = os.environ.get("SRTCTL_SOURCE_DIR")
        if source_dir:
            configs_dir = Path(source_dir) / "configs"
            if configs_dir.exists():
                container_mounts[configs_dir.resolve()] = Path("/configs")

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

        # Add extra mounts from config
        if config.extra_mount:
            for mount_spec in config.extra_mount:
                host_path, container_path = mount_spec.split(":", 1)
                container_mounts[Path(host_path).resolve()] = Path(container_path)

        # Optional external eval-harness mount (used by the lm-eval runner when
        # present). Skip exists() here: the orchestrator runs on the SLURM head
        # node where this path may live only on the compute-node shared FS.
        eval_ws = os.environ.get("LM_EVAL_WORKSPACE")
        if eval_ws:
            container_mounts[Path(eval_ws)] = Path("/lm-eval-workspace")

        # Add FormattablePath mounts from config.container_mounts
        # These need to be expanded with the runtime context, so we create a
        # temporary context first and then update
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
            network_interface=get_srtslurm_setting("network_interface", "eth0"),
            container_mounts={},
            srun_options=dict(config.srun_options),
            environment=dict(config.environment),
            is_hf_model=is_hf_model,
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
            network_interface=get_srtslurm_setting("network_interface", "eth0"),
            container_mounts=container_mounts,
            srun_options=dict(config.srun_options),
            environment=dict(config.environment),
            is_hf_model=is_hf_model,
        )

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
