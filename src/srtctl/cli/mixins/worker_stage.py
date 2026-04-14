# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Worker stage mixin for SweepOrchestrator.

Handles starting backend worker processes (prefill/decode/agg).
"""

import logging
import shlex
from collections import defaultdict
from typing import TYPE_CHECKING

from srtctl.backends.base import BackendProtocol
from srtctl.core.processes import ManagedProcess, NamedProcesses
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint, Process

logger = logging.getLogger(__name__)


class WorkerStageMixin:
    """Mixin for worker process startup stage.

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
        self.backend: BackendProtocol
        self.backend_processes: list[Process]
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    @property
    def backend(self) -> BackendProtocol:
        """Access the backend config (implements BackendProtocol)."""
        return self.config.backend

    @property
    def backend_processes(self) -> list["Process"]:
        """Compute physical process topology from endpoints (cached)."""
        raise NotImplementedError

    @property
    def endpoints(self) -> list["Endpoint"]:
        """Endpoint allocation topology."""
        raise NotImplementedError

    def _build_worker_preamble(self) -> str | None:
        """Build bash preamble for worker processes.

        Runs (in order):
        1. Custom setup script from /configs/ (if config.setup_script set)
        2. Dynamo installation (if frontend type is dynamo)
        """
        parts = []

        # 1. Custom setup script (runs first)
        if self.config.setup_script:
            script_path = f"/configs/{self.config.setup_script}"
            parts.append(
                f"echo 'Running setup script: {script_path}' && "
                f"if [ -f '{script_path}' ]; then bash '{script_path}'; else echo 'WARNING: {script_path} not found'; fi"
            )

        # 2. Dynamo installation (required for dynamo.sglang when using dynamo frontend)
        # Skip if dynamo.install is False (container already has dynamo installed)
        if self.config.frontend.type == "dynamo" and self.config.dynamo.install:
            parts.append(self.config.dynamo.get_install_commands())

        if not parts:
            return None

        return " && ".join(parts)

    def start_worker(self, process: "Process", endpoint_processes: list["Process"]) -> ManagedProcess:
        """Start a single worker process (one srun per node, used by SGLang)."""
        mode = process.endpoint_mode
        index = process.endpoint_index

        logger.info("Starting %s worker %d on %s", mode, index, process.node)

        # Log and config files
        worker_log = self.runtime.log_dir / f"{process.node}_{mode}_w{index}.out"
        config_dump = self.runtime.log_dir / f"{process.node}_config.json"

        # Profiling setup
        profiling = self.config.profiling
        nsys_prefix = None
        if profiling.enabled:
            (self.runtime.log_dir / "profiles" / mode).mkdir(parents=True, exist_ok=True)
        if profiling.is_nsys:
            nsys_output = f"/logs/profiles/{mode}/{process.node}_{mode}_w{index}_profile"
            nsys_prefix = profiling.get_nsys_prefix(nsys_output, frontend_type=self.config.frontend.type)

        # Build command using backend's method
        cmd = self.backend.build_worker_command(
            process=process,
            endpoint_processes=endpoint_processes,
            runtime=self.runtime,
            frontend_type=self.config.frontend.type,
            nsys_prefix=nsys_prefix,
            dump_config_path=config_dump,
        )

        # Environment variables
        env_to_set = {
            "HEAD_NODE_IP": self.runtime.head_node_ip,
            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.infra}:2379",
            "NATS_SERVER": f"nats://{self.runtime.nodes.infra}:4222",
            "DYN_SYSTEM_PORT": str(process.sys_port),
            "DYN_REQUEST_PLANE": "nats",
        }

        # Add mode-specific environment variables from backend
        # Support simple {node} and {node_id} templating
        # Unknown placeholders are left unchanged (no error thrown)
        node_id = self.runtime.nodes.worker.index(process.node)
        template_vars = {"node": process.node, "node_id": node_id}

        class SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"  # Leave unknown placeholders unchanged

        for key, value in self.backend.get_environment_for_mode(mode).items():
            formatted_value = value.format_map(SafeDict(template_vars))
            env_to_set[key] = formatted_value

        # Add config environment variables with same templating support
        for key, value in self.runtime.environment.items():
            formatted_value = value.format_map(SafeDict(template_vars))
            env_to_set[key] = formatted_value

        # Add profiling environment variables
        if profiling.enabled:
            profile_dir = str(self.runtime.log_dir / "profiles")
            env_to_set.update(profiling.get_env_vars(mode, profile_dir))

        # Set CUDA_VISIBLE_DEVICES if not using all GPUs
        if len(process.gpu_indices) < self.runtime.gpus_per_node:
            env_to_set["CUDA_VISIBLE_DEVICES"] = process.cuda_visible_devices

        # Add backend-specific process environment variables (e.g., unique ports)
        env_to_set.update(self.backend.get_process_environment(process))

        # Log env vars in the format: VAR=value VAR2=value2
        env_str = " ".join(f"{k}={v}" for k, v in sorted(env_to_set.items()))
        logger.info("Env: %s", env_str)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", worker_log)
        if profiling.enabled:
            logger.info("Profiling: %s mode", profiling.type)

        # Build bash preamble (setup script + dynamo install)
        bash_preamble = self._build_worker_preamble()

        proc = start_srun_process(
            command=cmd,
            nodelist=[process.node],
            output=str(worker_log),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
            bash_preamble=bash_preamble,
            srun_options=self.runtime.srun_options,
        )

        return ManagedProcess(
            name=f"{mode}_{index}_{process.node}",
            popen=proc,
            log_file=worker_log,
            node=process.node,
            critical=True,
        )

    def start_endpoint_worker(self, endpoint_processes: list["Process"]) -> ManagedProcess:
        """Start a worker using MPI-style launching (one srun per endpoint, used by TRTLLM).

        This launches a single srun command that spans all nodes in the endpoint,
        with ntasks = total GPUs across all nodes.
        """
        # Use the leader process for metadata
        leader = endpoint_processes[0]
        mode = leader.endpoint_mode
        index = leader.endpoint_index

        # Collect all unique nodes for this endpoint
        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        num_nodes = len(endpoint_nodes)
        total_gpus = num_nodes * len(leader.gpu_indices)

        logger.info(
            "Starting %s worker %d on %d nodes (%s) with %d total GPUs (MPI mode)",
            mode,
            index,
            num_nodes,
            ",".join(endpoint_nodes),
            total_gpus,
        )

        # Log and config files (use leader node in name)
        worker_log = self.runtime.log_dir / f"{leader.node}_{mode}_w{index}.out"
        config_dump = self.runtime.log_dir / f"{leader.node}_config.json"

        # Profiling setup
        profiling = self.config.profiling
        nsys_prefix = None
        if profiling.enabled:
            (self.runtime.log_dir / "profiles" / mode).mkdir(parents=True, exist_ok=True)
        if profiling.is_nsys:
            nsys_output = f"/logs/profiles/{mode}/{leader.node}_{mode}_w{index}_profile"
            nsys_prefix = profiling.get_nsys_prefix(nsys_output, frontend_type=self.config.frontend.type)

        # Build command using backend's method
        cmd = self.backend.build_worker_command(
            process=leader,
            endpoint_processes=endpoint_processes,
            runtime=self.runtime,
            frontend_type=self.config.frontend.type,
            nsys_prefix=nsys_prefix,
            dump_config_path=config_dump,
        )

        # Environment variables
        env_to_set = {
            "HEAD_NODE_IP": self.runtime.head_node_ip,
            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.infra}:2379",
            "NATS_SERVER": f"nats://{self.runtime.nodes.infra}:4222",
            "DYN_SYSTEM_PORT": str(leader.sys_port),
        }

        # Add mode-specific environment variables from backend
        env_to_set.update(self.backend.get_environment_for_mode(mode))

        # Add config environment variables
        env_to_set.update(self.runtime.environment)

        # Add profiling environment variables
        if profiling.enabled:
            profile_dir = str(self.runtime.log_dir / "profiles")
            env_to_set.update(profiling.get_env_vars(mode, profile_dir))

        # Set CUDA_VISIBLE_DEVICES if not using all GPUs on the node
        if len(leader.gpu_indices) < self.runtime.gpus_per_node:
            env_to_set["CUDA_VISIBLE_DEVICES"] = leader.cuda_visible_devices

        # Log env vars in the format: VAR=value VAR2=value2
        env_str = " ".join(f"{k}={v}" for k, v in sorted(env_to_set.items()))
        logger.info("Env: %s", env_str)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", worker_log)
        if profiling.enabled:
            logger.info("Profiling: %s mode", profiling.type)

        # Build bash preamble (setup script + dynamo install)
        bash_preamble = self._build_worker_preamble()

        # Get srun config from backend
        srun_config = self.backend.get_srun_config()

        proc = start_srun_process(
            command=cmd,
            nodes=num_nodes,
            ntasks=total_gpus,
            nodelist=endpoint_nodes,
            output=str(worker_log),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
            bash_preamble=bash_preamble,
            mpi=srun_config.mpi,
            oversubscribe=srun_config.oversubscribe,
            cpu_bind=srun_config.cpu_bind,
        )

        return ManagedProcess(
            name=f"{mode}_{index}_{leader.node}",
            popen=proc,
            log_file=worker_log,
            node=leader.node,
            critical=True,
        )

    def start_all_workers(self) -> NamedProcesses:
        """Start all backend workers."""
        logger.info("Starting backend workers")

        # Check if backend uses MPI-style per-endpoint launching
        srun_config = self.backend.get_srun_config()
        launch_per_endpoint = srun_config.launch_per_endpoint

        grouped: dict[tuple, list[Process]] = defaultdict(list)
        for process in self.backend_processes:
            key = (process.endpoint_mode, process.endpoint_index)
            grouped[key].append(process)

        result: NamedProcesses = {}

        if launch_per_endpoint:
            # MPI-style: one srun per endpoint (TRTLLM)
            for _endpoint_key, endpoint_processes in grouped.items():
                managed = self.start_endpoint_worker(endpoint_processes)
                result[managed.name] = managed
        else:
            # Per-process: one srun per node (SGLang)
            for _endpoint_key, endpoint_processes in grouped.items():
                for process in endpoint_processes:
                    managed = self.start_worker(process, endpoint_processes)
                    result[managed.name] = managed

        logger.info("Started %d worker processes", len(result))
        return result
