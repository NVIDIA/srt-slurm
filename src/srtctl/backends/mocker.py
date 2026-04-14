# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Mocker backend configuration.

Implements BackendProtocol for the dynamo.mocker scheduler simulator.
Used for smoke-testing the full srt-slurm pipeline (SLURM, mounts,
tokenizer, discovery, frontend, benchmark) without loading model weights.

The mocker validates model paths, reads tokenizer config, registers with
etcd/NATS discovery, simulates scheduling and KV cache management, and
generates random tokens at configurable simulated latency.
"""

import builtins
from collections.abc import Sequence
from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
)

from marshmallow import Schema
from marshmallow_dataclass import dataclass
from srtctl.backends.base import BackendProtocol

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Endpoint, Process

# Type alias for worker modes
WorkerMode = Literal["prefill", "decode", "agg"]


@dataclass(frozen=True)
class MockerServerConfig:
    """Mocker CLI configuration per mode (prefill/decode/aggregated).

    Each mode can have its own configuration dict that gets converted
    to CLI flags when starting the mocker. Use for per-mode overrides
    of mocker-specific parameters.
    """

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    aggregated: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class MockerProtocol(BackendProtocol):
    """Dynamo Mocker protocol - implements BackendProtocol.

    This frozen dataclass both holds configuration AND implements the
    BackendProtocol methods for process allocation and launching.

    The mocker is a drop-in replacement for real inference backends
    (sglang, vllm, trtllm) that simulates scheduling without loading
    model weights. It validates model paths, loads tokenizer config,
    and registers with etcd/NATS discovery identically to real workers.

    Example YAML:
        backend:
          type: mocker
          speedup_ratio: 100
          engine_type: vllm

    Or with per-mode overrides:
        backend:
          type: mocker
          speedup_ratio: 100
          engine_type: sglang
          mocker_config:
            prefill:
              max-num-seqs: 512
            decode:
              max-num-seqs: 128
    """

    type: Literal["mocker"] = "mocker"

    # Simulation parameters
    engine_type: str = "vllm"
    speedup_ratio: float = 100.0
    decode_speedup_ratio: float = 1.0
    num_gpu_blocks_override: int = 16384
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int | None = None
    data_parallel_size: int = 1
    num_workers: int = 1
    startup_time: float | None = None
    kv_transfer_bandwidth: float | None = None
    kv_cache_dtype: str | None = None
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    preemption_mode: str | None = None

    # Environment variables per mode
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)

    # Per-mode CLI overrides
    mocker_config: MockerServerConfig | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    # =========================================================================
    # BackendProtocol Implementation
    # =========================================================================

    def get_srun_config(self) -> "SrunConfig":
        """Mocker uses per-process launching (one srun per node)."""
        from srtctl.backends.base import SrunConfig

        return SrunConfig(mpi=None, oversubscribe=False, launch_per_endpoint=False)

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        """Get merged config dict for a worker mode."""
        if not self.mocker_config:
            return {}

        if mode == "prefill":
            return dict(self.mocker_config.prefill or {})
        elif mode == "decode":
            return dict(self.mocker_config.decode or {})
        elif mode == "agg":
            return dict(self.mocker_config.aggregated or {})
        return {}

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        """Get environment variables for a worker mode."""
        if mode == "prefill":
            return dict(self.prefill_environment)
        elif mode == "decode":
            return dict(self.decode_environment)
        elif mode == "agg":
            return dict(self.aggregated_environment)
        return {}

    def get_process_environment(self, process: "Process") -> dict[str, str]:
        """Get process-specific environment variables.

        The mocker does not need per-process env vars (no NIXL ports, etc.).
        """
        return {}

    def get_served_model_name(self, default: str) -> str:
        """Get served model name — mocker uses default (model path basename)."""
        return default

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
        """Allocate endpoints to nodes."""
        from srtctl.core.topology import allocate_endpoints

        return allocate_endpoints(
            num_prefill=num_prefill,
            num_decode=num_decode,
            num_agg=num_agg,
            gpus_per_prefill=gpus_per_prefill,
            gpus_per_decode=gpus_per_decode,
            gpus_per_agg=gpus_per_agg,
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
        )

    def endpoints_to_processes(
        self,
        endpoints: list["Endpoint"],
        base_sys_port: int = 8081,
    ) -> list["Process"]:
        """Convert endpoints to processes."""
        from srtctl.core.topology import endpoints_to_processes

        return endpoints_to_processes(endpoints, base_sys_port=base_sys_port)

    def build_worker_command(
        self,
        process: "Process",
        endpoint_processes: list["Process"],
        runtime: "RuntimeContext",
        frontend_type: str = "dynamo",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Path | None = None,
    ) -> list[str]:
        """Build the command to start a mocker worker process.

        Args:
            process: The process to start
            endpoint_processes: All processes for this endpoint (for multi-node)
            runtime: Runtime context with paths and settings
            frontend_type: Frontend type (mocker always uses dynamo discovery)
            nsys_prefix: Optional nsys profiling command prefix
            dump_config_path: Unused (mocker has no config dump)
        """
        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)

        # Determine model path: HF model ID or container mount path
        model_arg = str(runtime.model_path) if runtime.is_hf_model else "/model"

        # Start with nsys prefix if provided
        cmd: list[str] = list(nsys_prefix) if nsys_prefix else []

        cmd.extend(
            [
                "python3",
                "-m",
                "dynamo.mocker",
                "--model-path",
                model_arg,
            ]
        )

        # Disaggregation mode for prefill/decode workers
        if mode != "agg":
            cmd.extend(["--disaggregation-mode", mode])

        # Bootstrap port for prefill workers (disaggregated serving rendezvous)
        if mode == "prefill" and process.bootstrap_port is not None:
            cmd.extend(["--bootstrap-ports", str(process.bootstrap_port)])

        # Core simulation parameters (always emitted)
        cmd.extend(["--engine-type", self.engine_type])
        cmd.extend(["--speedup-ratio", str(self.speedup_ratio)])
        cmd.extend(["--data-parallel-size", str(self.data_parallel_size)])
        cmd.extend(["--num-gpu-blocks-override", str(self.num_gpu_blocks_override)])
        cmd.extend(["--max-num-seqs", str(self.max_num_seqs)])
        cmd.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])

        # Optional parameters (only emitted when non-default)
        if self.decode_speedup_ratio != 1.0:
            cmd.extend(["--decode-speedup-ratio", str(self.decode_speedup_ratio)])
        if self.block_size is not None:
            cmd.extend(["--block-size", str(self.block_size)])
        if self.num_workers > 1:
            cmd.extend(["--num-workers", str(self.num_workers)])
        if self.startup_time is not None:
            cmd.extend(["--startup-time", str(self.startup_time)])
        if self.kv_transfer_bandwidth is not None:
            cmd.extend(["--kv-transfer-bandwidth", str(self.kv_transfer_bandwidth)])
        if self.kv_cache_dtype is not None:
            cmd.extend(["--kv-cache-dtype", self.kv_cache_dtype])
        if not self.enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        if not self.enable_chunked_prefill:
            cmd.append("--no-enable-chunked-prefill")
        if self.preemption_mode is not None:
            cmd.extend(["--preemption-mode", self.preemption_mode])

        # Per-mode config overrides from mocker_config
        cmd.extend(_config_to_cli_args(config))

        return cmd


def _config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Convert config dict to CLI arguments."""
    args: list[str] = []
    for key, value in sorted(config.items()):
        flag_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag_name}")
        elif isinstance(value, list):
            args.append(f"--{flag_name}")
            args.extend(str(v) for v in value)
        elif value is not None:
            args.extend([f"--{flag_name}", str(value)])
    return args
