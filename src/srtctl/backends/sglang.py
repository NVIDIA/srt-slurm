# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang backend configuration.

Implements BackendProtocol for SGLang inference serving with prefill/decode disaggregation.
"""

import builtins
import json
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

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Endpoint, Process

# Type alias for worker modes
WorkerMode = Literal["prefill", "decode", "agg"]


@dataclass(frozen=True)
class SGLangServerConfig:
    """SGLang server CLI configuration per mode (prefill/decode/aggregated).

    Each mode can have its own configuration dict that gets converted
    to CLI flags when starting the worker.
    """

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    aggregated: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class SGLangProtocol:
    """SGLang protocol - implements BackendProtocol.

    This frozen dataclass both holds configuration AND implements the
    BackendProtocol methods for process allocation and launching.

    Example YAML:
        backend:
          type: sglang
          prefill_environment:
            CUDA_LAUNCH_BLOCKING: "1"
          sglang_config:
            prefill:
              mem-fraction-static: 0.8
              chunked-prefill-size: 8192
            decode:
              mem-fraction-static: 0.9
    """

    type: Literal["sglang"] = "sglang"
    gpu_type: str | None = None

    # Environment variables per mode
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)

    # SGLang server CLI config per mode
    sglang_config: SGLangServerConfig | None = None

    # KV events config - enables --kv-events-config with auto-allocated ports
    # Per-mode: {"prefill": true, "decode": {"publisher": "zmq", "topic": "custom"}}
    # Or global: true (enables for prefill+decode with defaults)
    kv_events_config: bool | dict[str, Any] | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    # =========================================================================
    # BackendProtocol Implementation
    # =========================================================================

    def get_srun_config(self) -> "SrunConfig":
        """SGLang uses per-process launching (one srun per node)."""
        from srtctl.backends.base import SrunConfig

        return SrunConfig(mpi=None, oversubscribe=False, launch_per_endpoint=False)

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        """Get merged config dict for a worker mode."""
        if not self.sglang_config:
            return {}

        if mode == "prefill":
            return dict(self.sglang_config.prefill or {})
        elif mode == "decode":
            return dict(self.sglang_config.decode or {})
        elif mode == "agg":
            return dict(self.sglang_config.aggregated or {})
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

        SGLang handles kv-events via CLI args (--kv-events-config), so no
        additional process-specific env vars are needed here.
        """
        return {}

    def is_grpc_mode(self, mode: WorkerMode) -> bool:
        """Check if gRPC mode is enabled for a worker mode."""
        config = self.get_config_for_mode(mode)
        return config.get("grpc-mode", False)

    def get_served_model_name(self, default: str) -> str:
        """Get served model name from SGLang config, or return default."""
        if self.sglang_config:
            for cfg in [self.sglang_config.prefill, self.sglang_config.aggregated, self.sglang_config.decode]:
                if cfg:
                    name = cfg.get("served-model-name") or cfg.get("served_model_name")
                    if name:
                        return name
        return default

    def get_kv_events_config_for_mode(self, mode: WorkerMode) -> dict[str, str] | None:
        """Get kv-events config for a worker mode.

        Returns None if disabled, or dict with publisher/topic if enabled.
        """
        if not self.kv_events_config:
            return None

        # Global bool: enable for prefill+decode with defaults
        if self.kv_events_config is True:
            if mode in ("prefill", "decode"):
                return {"publisher": "zmq", "topic": "kv-events"}
            return None

        # Per-mode config dict
        if isinstance(self.kv_events_config, dict):
            # Normalize mode key: use "aggregated" for aggregated mode
            mode_cfg = self.kv_events_config.get("aggregated") if mode == "agg" else self.kv_events_config.get(mode)

            if mode_cfg is None:
                return None
            if mode_cfg is True:
                return {"publisher": "zmq", "topic": "kv-events"}
            if isinstance(mode_cfg, dict):
                # Merge with defaults
                return {"publisher": "zmq", "topic": "kv-events", **mode_cfg}

        return None

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
        """Build the command to start an SGLang worker process.

        Args:
            process: The process to start
            endpoint_processes: All processes for this endpoint (for multi-node)
            runtime: Runtime context with paths and settings
            frontend_type: Frontend type - "sglang" uses sglang.launch_server, "dynamo" uses dynamo.sglang
            nsys_prefix: Optional nsys profiling command prefix
            dump_config_path: Path to dump config JSON
        """
        from srtctl.core.slurm import get_hostname_ip

        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)

        # Pop keys that are handled explicitly to avoid duplicate flags from _config_to_cli_args
        config.pop("model-path", None)
        config.pop("model_path", None)
        config.pop("served-model-name", None)
        config.pop("served_model_name", None)

        # Determine if multi-node
        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        is_multi_node = len(endpoint_nodes) > 1

        # Get leader IP for distributed init
        leader_ip = get_hostname_ip(endpoint_nodes[0])
        dist_init_port = 29500

        # Choose Python module based on frontend type
        use_sglang = frontend_type == "sglang"
        python_module = "sglang.launch_server" if use_sglang else "dynamo.sglang"

        # Get served model name from config
        served_model_name = self.get_served_model_name(runtime.model_path.name)

        # Determine model path: HF model ID or container mount path
        # For HF models (hf:prefix), model_path contains the HF model ID (e.g., "facebook/opt-125m")
        # For local models, model is mounted to /model in the container
        model_arg = str(runtime.model_path) if runtime.is_hf_model else "/model"

        # Start with nsys prefix if provided
        cmd: list[str] = list(nsys_prefix) if nsys_prefix else []

        cmd.extend(
            [
                "python3",
                "-m",
                python_module,
                "--model-path",
                model_arg,
                "--served-model-name",
                served_model_name,
                "--host",
                "0.0.0.0",
            ]
        )

        # Always pass --port when using sglang.launch_server or dynamo.sglang
        cmd.extend(["--port", str(process.http_port)])

        # Add disaggregation mode for prefill/decode workers (both dynamo and sglang frontend)
        if mode != "agg":
            cmd.extend(["--disaggregation-mode", mode])
            # Bootstrap port only needed for sglang frontend (dynamo handles internally)
            if frontend_type == "sglang" and mode == "prefill" and process.bootstrap_port is not None:
                cmd.extend(["--disaggregation-bootstrap-port", str(process.bootstrap_port)])

        # Add multi-node coordination flags
        if is_multi_node:
            node_rank = endpoint_nodes.index(process.node)
            cmd.extend(
                [
                    "--dist-init-addr",
                    f"{leader_ip}:{dist_init_port}",
                    "--nnodes",
                    str(len(endpoint_nodes)),
                    "--node-rank",
                    str(node_rank),
                ]
            )

        # Add config dump path (not when using sglang frontend)
        if dump_config_path and frontend_type != "sglang":
            cmd.extend(["--dump-config-to", str(dump_config_path)])

        # Add kv-events-config if enabled for this mode and we have an allocated port
        kv_cfg = self.get_kv_events_config_for_mode(mode)
        if kv_cfg and process.kv_events_port is not None:
            # Add the endpoint with the allocated port
            kv_cfg["endpoint"] = f"tcp://*:{process.kv_events_port}"
            cmd.extend(["--kv-events-config", json.dumps(kv_cfg)])

        # Add all config flags
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
