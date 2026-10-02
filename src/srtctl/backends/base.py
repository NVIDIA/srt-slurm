# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base classes and shared behavior for backend configurations.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeAlias

from marshmallow import ValidationError, fields

from srtctl.ports import DYN_SYSTEM_PORT_BASE

if TYPE_CHECKING:
    from pathlib import Path

    from srtctl.backends.sglang import MooncakeKVStoreConfig
    from srtctl.backends.vllm import VLLMFailoverConfig, VLLMMooncakeKVStoreConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process


WorkerMode: TypeAlias = Literal["prefill", "decode", "agg"]


class BackendType(str, Enum):
    """Supported backend types."""

    SGLANG = "sglang"
    TRTLLM = "trtllm"
    VLLM = "vllm"
    MOCKER = "mocker"
    ATOM = "atom"
    TILERT = "tilert"


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
        kill_on_bad_exit: Pass ``--kill-on-bad-exit=1`` on every endpoint step, so one task
                          exiting non-zero ends the whole step (and srun exits) instead of
                          leaving the other ranks up with no engine behind them.
    """

    mpi: str | None = None
    oversubscribe: bool = False
    launch_per_endpoint: bool = False
    cpu_bind: str | None = None
    sequential_node_start: int = 0
    kill_on_bad_exit: bool = False


class RoleSettings(ABC):
    """What an engine reads from one role of the recipe (``roles.<role>``).

    ``srtctl.core.schema.RoleConfig`` inherits this data contract. Keeping the
    contract here lets the schema import engines without a cycle. Concrete
    dataclasses supply the fields and defaults.
    """

    env: Mapping[str, str]
    args: Mapping[str, Any]
    extra_args: Sequence[str]
    kv_events: bool | Mapping[str, Any] | None


class BoundRolesField(fields.Field):
    """Marshmallow field for an engine's ``roles``: bound by SrtConfig, never read from a recipe, never dumped."""

    def _deserialize(self, value: Any, attr: str | None, data: Mapping[str, Any] | None, **kwargs: Any) -> Any:
        raise ValidationError("roles are declared at the recipe's top level (roles.<role>), not on the engine")

    def _serialize(self, value: Any, attr: str | None, obj: Any, **kwargs: Any) -> None:
        return None


def role_for_mode(roles: Mapping[str, RoleSettings], mode: str) -> RoleSettings | None:
    """The role a worker mode runs under (``aggregated`` and ``agg`` are both the agg role)."""
    return roles.get("agg" if mode == "aggregated" else mode)


def role_args(roles: Mapping[str, RoleSettings], mode: str) -> dict[str, Any]:
    """``roles.<role>.args`` for a worker mode, as a fresh dict; empty when the role is absent."""
    role = role_for_mode(roles, mode)
    return dict(role.args) if role is not None else {}


def role_env(roles: Mapping[str, RoleSettings], mode: str) -> dict[str, str]:
    """``roles.<role>.env`` for a worker mode, as a fresh dict; empty when the role is absent."""
    role = role_for_mode(roles, mode)
    return dict(role.env) if role is not None else {}


def role_kv_events(roles: Mapping[str, RoleSettings], mode: str, defaults: Mapping[str, Any]) -> dict[str, Any] | None:
    """``roles.<role>.kv_events`` for a worker mode over ``defaults``; None when the role publishes none."""
    role = role_for_mode(roles, mode)
    kv_events = role.kv_events if role is not None else None
    if not kv_events:
        return None
    if kv_events is True:
        return dict(defaults)
    return {**defaults, **kv_events}


class Backend(ABC):
    """Abstract base class for backend configurations.

    Concrete frozen dataclasses inherit optional defaults and implement the launch hooks.
    Each backend is responsible for:
    1. Allocating logical endpoints (serving units)
    2. Converting endpoints to physical processes
    3. Building commands to start those processes
    """

    @property
    @abstractmethod
    def type(self) -> str:
        """Backend type identifier."""
        raise NotImplementedError

    @property
    def mooncake_kv_store(self) -> "MooncakeKVStoreConfig | VLLMMooncakeKVStoreConfig | None":
        """The recipe's Mooncake KV store block, or None when the engine has none.

        Set, it implies the mooncake-master service and the MOONCAKE_* worker
        environment from get_mooncake_worker_env.
        """
        return None

    @property
    def failover(self) -> "VLLMFailoverConfig | None":
        """Shadow engine recovery, or None when the engine has none.

        Set, it implies the gms service and the per-worker environment from
        get_failover_environment.
        """
        return None

    # Bound recipe roles; concrete dataclasses own the field and its serialization.
    roles: Mapping[str, RoleSettings]

    def get_srun_config(self) -> SrunConfig:
        """Get srun configuration for this backend.

        Returns SrunConfig with MPI settings and launch strategy.
        """
        return SrunConfig()

    def fatal_log_patterns(self, mode: WorkerMode) -> tuple[str, ...]:
        """Regular expressions that, printed in a worker's log, mean the engine is gone.

        The process monitor fails a critical worker whose srun step is still
        running when a new log line matches one of these (see
        ``ManagedProcess.fatal_log_patterns``). Engines whose step exits with the
        engine answer ``()``; an engine behind a launcher that keeps the step
        alive names the lines the launcher prints once the engine has died.
        """
        return ()

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        """The role's engine arguments (``roles.<role>.args``) for a worker mode (prefill/decode/agg)."""
        return role_args(self.roles, mode)

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        """The role's environment (``roles.<role>.env``) for a worker mode, before engine defaults."""
        return role_env(self.roles, mode)

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def endpoints_to_processes(
        self,
        endpoints: list["Endpoint"],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: Optional["NodePortAllocator"] = None,
        frontend_type: str = "dynamo",
        dynamo_sidecar: bool = False,
    ) -> list["Process"]:
        """Convert logical endpoints to physical processes."""
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

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
        return {}

    def get_mooncake_worker_env(self, infra_node_ip: str, local_hostname: str) -> dict[str, str]:
        """MOONCAKE_* environment for a worker; empty when mooncake_kv_store is None."""
        return {}

    def get_failover_environment(self, process: "Process", job_id: str) -> dict[str, str]:
        """Shadow engine recovery environment for a worker; empty when failover is None."""
        return {}

    def should_set_visible_devices(self) -> bool:
        """Whether the worker stage pins each process to its GPUs with the cluster's device mask.

        The variable is the cluster's ``visible_devices_env`` (CUDA_VISIBLE_DEVICES,
        ROCR_VISIBLE_DEVICES). True for engines that read the environment; an
        engine that takes its devices on the command line answers False.
        """
        return True

    def get_served_model_name(self, default: str) -> str:
        """Get served model name from backend config, or return default."""
        return default

    def is_grpc_mode(self, mode: str) -> bool:
        """Whether the mode's workers serve gRPC instead of HTTP.

        A static router reads it to advertise ``grpc://`` worker URLs. Defaults
        to ``False``; override for an engine with a gRPC server.
        """
        return False
