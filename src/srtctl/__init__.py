# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
srtctl - Benchmark submission framework for distributed serving workloads.

This package provides Python-first orchestration for LLM inference benchmarks
on SLURM clusters, supporting multiple backends:
- SGLang (with prefill/decode disaggregation)
- vLLM (placeholder)
- TensorRT-LLM (placeholder)

Key modules:
- core.config: Configuration loading and validation
- core.schema: Frozen dataclass definitions (SrtConfig, etc.)
- core.runtime: RuntimeContext for computed paths and values
- core.topology: Endpoint and Process dataclasses for worker allocation
- core.processes: Process lifecycle management
- core.slurm: SLURM utilities (srun, nodelist, IP resolution)
- core.health: HTTP health check and port waiting utilities
- core.ip_utils: IP address resolution utilities
- backends: Backend-specific configuration dataclasses
- cli.submit: Job submission interface
- cli.do_sweep: Main orchestration script
- logging_utils: Logging configuration

Usage:
    # Submit with orchestrator (Python-controlled)
    srtctl apply -f config.yaml
"""

__version__ = "0.3.0"

# Logging utilities (should be first)
# Backend configs
from .backends import (
    BackendConfig,
    BackendProtocol,
    BackendType,
    SGLangProtocol,
)

# Core modules
from .core.config import get_srtslurm_setting, load_config
from .core.formatting import FormattablePath, FormattableString
from .core.processes import (
    ManagedProcess,
    NamedProcesses,
    ProcessRegistry,
)
from .core.runtime import Nodes, RuntimeContext
from .core.schema import SrtConfig
from .core.slurm import get_hostname_ip, get_slurm_job_id
from .core.topology import Endpoint, Process, allocate_endpoints, endpoints_to_processes
from .logging_utils import setup_logging

__all__ = [
    # Version
    "__version__",
    # Logging
    "setup_logging",
    # Config
    "load_config",
    "get_srtslurm_setting",
    "SrtConfig",
    # Formatting
    "FormattablePath",
    "FormattableString",
    # Runtime
    "Nodes",
    "RuntimeContext",
    "get_slurm_job_id",
    "get_hostname_ip",
    # Endpoints
    "Endpoint",
    "Process",
    "allocate_endpoints",
    "endpoints_to_processes",
    # Process management
    "ManagedProcess",
    "NamedProcesses",
    "ProcessRegistry",
    # Backends
    "BackendProtocol",
    "BackendConfig",
    "BackendType",
    "SGLangProtocol",
]
