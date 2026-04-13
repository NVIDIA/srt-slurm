# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base types and protocols for backend configurations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Endpoint, Process


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
    """

    mpi: str | None = None
    oversubscribe: bool = False
    launch_per_endpoint: bool = False
    cpu_bind: str | None = None


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
    ) -> list["Endpoint"]:
        """Allocate logical endpoints based on resource requirements."""
        ...

    def endpoints_to_processes(
        self,
        endpoints: list["Endpoint"],
        base_sys_port: int = 8081,
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

    def get_served_model_name(self, default: str) -> str:
        """Get served model name from backend config, or return default."""
        ...
