# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core modules for srtctl.

This package contains:
- config: Configuration loading and validation
- schema: Frozen dataclass schemas (SrtConfig, etc.)
- formatting: FormattablePath and FormattableString for deferred expansion
- runtime: RuntimeContext for computed paths and values
- topology: Endpoint and Process dataclasses for worker allocation
- processes: Process lifecycle management
- slurm: SLURM utilities (srun, nodelist, IP resolution)
- health: HTTP health check and port waiting utilities
- ip_utils: IP address resolution utilities
"""

# Re-export backend configs
from srtctl.backends import (
    BackendConfig,
    BackendProtocol,
    BackendType,
    SGLangProtocol,
    SGLangServerConfig,
)

from .config import (
    find_cluster_config_path,
    get_srtslurm_setting,
    load_config,
)
from .formatting import FormattablePath, FormattableString
from .health import (
    WorkerHealthResult,
    check_dynamo_health,
    check_sglang_router_health,
    check_static_router_health,
    wait_for_etcd,
    wait_for_health,
    wait_for_http_endpoints,
    wait_for_model,
    wait_for_port,
)
from .ip_utils import get_local_ip, get_node_ip
from .processes import (
    ManagedProcess,
    NamedProcesses,
    ProcessRegistry,
    setup_signal_handlers,
    start_process_monitor,
)
from .runtime import Nodes, RuntimeContext
from .schema import (
    DEFAULT_AI_ANALYSIS_PROMPT,
    AIAnalysisConfig,
    BenchmarkConfig,
    ClusterConfig,
    FrontendConfig,
    HealthCheckConfig,
    ModelConfig,
    OutputConfig,
    ProfilingConfig,
    ProfilingPhaseConfig,
    ResourceConfig,
    SlurmConfig,
    SrtConfig,
)
from .slurm import (
    get_container_mounts_str,
    get_hostname_ip,
    get_node_ips,
    get_slurm_job_id,
    get_slurm_nodelist,
    run_command,
    start_srun_process,
)
from .topology import (
    Endpoint,
    NodePortAllocator,
    Process,
    allocate_endpoints,
    endpoints_to_processes,
)

__all__ = [
    "DEFAULT_AI_ANALYSIS_PROMPT",
    "AIAnalysisConfig",
    "BackendConfig",
    "BackendProtocol",
    "BackendType",
    "BenchmarkConfig",
    "ClusterConfig",
    # Topology (worker allocation)
    "Endpoint",
    # Formatting
    "FormattablePath",
    "FormattableString",
    "FrontendConfig",
    "HealthCheckConfig",
    # Process management
    "ManagedProcess",
    "ModelConfig",
    "NamedProcesses",
    "NodePortAllocator",
    # Runtime
    "Nodes",
    "OutputConfig",
    "Process",
    "ProcessRegistry",
    "ProfilingConfig",
    "ProfilingPhaseConfig",
    "ResourceConfig",
    "RuntimeContext",
    # Backend configs (re-exported from backends)
    "SGLangProtocol",
    "SGLangServerConfig",
    "SlurmConfig",
    # Schema types (frozen dataclasses)
    "SrtConfig",
    "WorkerHealthResult",
    "allocate_endpoints",
    "check_dynamo_health",
    "check_sglang_router_health",
    "check_static_router_health",
    "endpoints_to_processes",
    "find_cluster_config_path",
    "get_container_mounts_str",
    "get_hostname_ip",
    "get_local_ip",
    # IP utilities
    "get_node_ip",
    "get_node_ips",
    # SLURM utilities
    "get_slurm_job_id",
    "get_slurm_nodelist",
    "get_srtslurm_setting",
    # Config loading
    "load_config",
    "run_command",
    "setup_signal_handlers",
    "start_process_monitor",
    "start_srun_process",
    "wait_for_etcd",
    "wait_for_health",
    "wait_for_http_endpoints",
    "wait_for_model",
    # Health checks
    "wait_for_port",
]
