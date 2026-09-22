# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base types and protocols for backend configurations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol

from srtctl.ports import DYN_SYSTEM_PORT_BASE

if TYPE_CHECKING:
    from pathlib import Path

    from srtctl.backends.sglang import MooncakeKVStoreConfig
    from srtctl.backends.vllm import VLLMFailoverConfig, VLLMMooncakeKVStoreConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process


class BackendType(str, Enum):
    """Supported backend types."""

    SGLANG = "sglang"
    TRTLLM = "trtllm"
    VLLM = "vllm"
    MOCKER = "mocker"


@dataclass
class SrunConfig:
    """Configuration for srun process launching.

    Attributes:
        mpi: MPI type (e.g., "pmix" for TRTLLM). None for non-MPI backends.
        oversubscribe: Use --oversubscribe flag (for MPI jobs).
        launch_per_endpoint: If True, launch one srun per endpoint (all nodes together).
                            If False, launch one srun per process (per node).
        cpu_bind: CPU binding mode (e.g., "verbose,none" for TRTLLM). None to omit.
        sequential_node_start: With launch_per_endpoint, how many endpoints that share a
                               leader node start at once, each batch gated on readiness.
                               0 starts every endpoint in parallel.
    """

    mpi: str | None = None
    oversubscribe: bool = False
    launch_per_endpoint: bool = False
    cpu_bind: str | None = None
    sequential_node_start: int = 0


class BackendProtocol(Protocol):
    """Protocol that all backend configurations must implement.

    This allows frozen dataclasses to act as backends by implementing these methods.
    Each backend is responsible for:
    1. Allocating logical endpoints (serving units)
    2. Converting endpoints to physical processes
    3. Building commands to start those processes
    """

    @property
    def type(self) -> str:
        """Backend type identifier."""
        ...

    @property
    def mooncake_kv_store(self) -> "MooncakeKVStoreConfig | VLLMMooncakeKVStoreConfig | None":
        """The recipe's Mooncake KV store block, or None when the engine has none.

        Set, it implies the mooncake-master service and the MOONCAKE_* worker
        environment from get_mooncake_worker_env.
        """
        ...

    @property
    def failover(self) -> "VLLMFailoverConfig | None":
        """Shadow engine recovery, or None when the engine has none.

        Set, it implies the gms service and the per-worker environment from
        get_failover_environment.
        """
        ...

    @property
    def prefill_environment(self) -> dict[str, str]:
        """Environment the recipe declares for prefill workers (roles.prefill.env), before engine defaults."""
        ...

    @property
    def decode_environment(self) -> dict[str, str]:
        """Environment the recipe declares for decode workers (roles.decode.env), before engine defaults."""
        ...

    @property
    def aggregated_environment(self) -> dict[str, str]:
        """Environment the recipe declares for aggregated workers (roles.agg.env), before engine defaults."""
        ...

    def get_srun_config(self) -> SrunConfig:
        """Get srun configuration for this backend.

        Returns SrunConfig with MPI settings and launch strategy.
        """
        ...

    def get_config_for_mode(self, mode: str) -> dict[str, Any]:
        """Get config dict for a worker mode (prefill/decode/agg)."""
        ...

    def get_environment_for_mode(self, mode: str) -> dict[str, str]:
        """Get environment variables for a worker mode."""
        ...

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
        spread_workers: bool = False,
    ) -> list["Endpoint"]:
        """Allocate logical endpoints based on resource requirements."""
        ...

    def endpoints_to_processes(
        self,
        endpoints: list["Endpoint"],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: Optional["NodePortAllocator"] = None,
        frontend_type: str = "dynamo",
        dynamo_sidecar: bool = False,
    ) -> list["Process"]:
        """Convert logical endpoints to physical processes."""
        ...

    def build_worker_command(
        self,
        process: "Process",
        endpoint_processes: list["Process"],
        runtime: "RuntimeContext",
        frontend_type: str = "dynamo",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Optional["Path"] = None,
        profiling: "ProfilingConfig | None" = None,
    ) -> list[str]:
        """Build command to start a worker process."""
        ...

    def get_process_environment(self, process: "Process") -> dict[str, str]:
        """Get process-specific environment variables.

        Unlike get_environment_for_mode() which returns static env vars per mode,
        this method returns dynamic env vars that depend on the specific process
        (e.g., unique ports allocated to each worker).

        Args:
            process: The process to get environment for.

        Returns:
            Dict of environment variable names to values.
        """
        ...

    def get_mooncake_worker_env(self, infra_node_ip: str, local_hostname: str) -> dict[str, str]:
        """MOONCAKE_* environment for a worker; empty when mooncake_kv_store is None."""
        ...

    def get_failover_environment(self, process: "Process", job_id: str) -> dict[str, str]:
        """Shadow engine recovery environment for a worker; empty when failover is None."""
        ...

    def should_set_visible_devices(self) -> bool:
        """Whether the worker stage pins each process to its GPUs with the cluster's device mask.

        The variable is the cluster's ``visible_devices_env`` (CUDA_VISIBLE_DEVICES,
        ROCR_VISIBLE_DEVICES). True for engines that read the environment; an
        engine that takes its devices on the command line answers False.
        """
        ...

    def get_served_model_name(self, default: str) -> str:
        """Get served model name from backend config, or return default."""
        ...
