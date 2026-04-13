# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import builtins
import uuid
from collections.abc import Sequence
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import yaml
from marshmallow import Schema
from marshmallow_dataclass import dataclass

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Endpoint, Process

# Type alias for worker modes
WorkerMode = Literal["prefill", "decode", "agg"]


@dataclass(frozen=True)
class TRTLLMServerConfig:
    """SGLang server CLI configuration per mode (prefill/decode/aggregated).

    Each mode can have its own configuration dict that gets converted
    to CLI flags when starting the worker.
    """

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    aggregated: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class TRTLLMProtocol:
    """TRTLLM protocol - implements BackendProtocol.

    This frozen dataclass both holds configuration AND implements the
    BackendProtocol methods for process allocation and launching.

    Example YAML:
        backend:
          type: trtllm
          prefill_environment:
            CUDA_LAUNCH_BLOCKING: "1"
          trtllm_config:
            prefill:
              mem-fraction-static: 0.8
              chunked-prefill-size: 8192
            decode:
              mem-fraction-static: 0.9
    """

    type: Literal["trtllm"] = "trtllm"

    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)

    trtllm_config: TRTLLMServerConfig | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    # =========================================================================
    # BackendProtocol Implementation
    # =========================================================================

    def get_srun_config(self) -> "SrunConfig":
        """TRTLLM uses MPI-style launching (one srun per endpoint with all nodes)."""
        from srtctl.backends.base import SrunConfig

        return SrunConfig(
            mpi="pmix",
            oversubscribe=True,
            launch_per_endpoint=True,
            cpu_bind="verbose,none",
        )

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        if not self.trtllm_config:
            return {}

        if mode == "prefill":
            return dict(self.trtllm_config.prefill or {})
        elif mode == "decode":
            return dict(self.trtllm_config.decode or {})
        elif mode == "agg":
            return dict(self.trtllm_config.aggregated or {})
        return {}

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        eplb_prefix = f"moe_shared_{uuid.uuid4().hex}"

        env_by_mode: dict[WorkerMode, dict[str, str]] = {
            "prefill": self.prefill_environment,
            "decode": self.decode_environment,
            "agg": self.aggregated_environment,
        }
        base_env = env_by_mode.get(mode)
        if base_env is None:
            return {}
        return {**base_env, "TRTLLM_EPLB_SHM_NAME": eplb_prefix}

    def get_process_environment(self, process: "Process") -> dict[str, str]:
        """Get process-specific environment variables.

        TRTLLM doesn't currently require process-specific env vars.
        """
        return {}

    def get_served_model_name(self, default: str) -> str:
        """Get served model name from TRTLLM config, or return default."""
        # TRTLLM doesn't have served-model-name in config, just use default
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
        """Build the command to start a TRTLLM worker process."""

        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)

        # Write config to host path (log_dir)
        config_filename = f"trtllm_config_{mode}.yaml"
        host_config_path = runtime.log_dir / config_filename
        host_config_path.write_text(yaml.safe_dump(config))

        # Use container paths for the command (log_dir is mounted to /logs)
        container_config_path = Path("/logs") / config_filename

        # Determine model path: HF model ID or container mount path
        # For HF models (hf:prefix), model_path contains the HF model ID (e.g., "facebook/opt-125m")
        # For local models, model is mounted to /model in the container
        model_arg = str(runtime.model_path) if runtime.is_hf_model else "/model"

        cmd = [
            "trtllm-llmapi-launch",
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model-path",
            model_arg,
            "--served-model-name",
            runtime.model_path.name,
        ]

        # Only add disaggregation mode for prefill/decode, not for agg
        if mode != "agg":
            cmd.extend(["--disaggregation-mode", mode])

        cmd.extend(
            [
                "--extra-engine-args",
                str(container_config_path),
                "--request-plane",
                "nats",
            ]
        )

        return cmd
