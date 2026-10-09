# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import builtins
import uuid
from collections.abc import Sequence
from dataclasses import field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import yaml
from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.backends.sidecar import build_sidecar_launch_command, get_dynamo_sidecar_config, sidecar_grpc_port
from srtctl.ports import DYN_SYSTEM_PORT_BASE, TRTLLM_DIST_INIT_PORTS

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import DynamoConfig, ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

# Type alias for worker modes
WorkerMode = Literal["prefill", "decode", "agg"]

# Dynamo constructs LLM directly, bypassing serve.py's pool provisioning.
# Reuse tekit 6b43a830f3's context manager inside the launcher rank-zero task;
# externally launched ranks read mooncake.json from TRTLLM_MOONCAKE_RUN_DIR.
_DYNAMO_MOONCAKE_ENTRYPOINT = """import runpy
import sys
import yaml
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import maybe_provision_pool

with open(sys.argv.pop(1)) as handle:
    config = yaml.safe_load(handle)
connector = KvCacheConnectorConfig(**config["kv_connector_config"])
with maybe_provision_pool(connector):
    runpy.run_module("dynamo.trtllm", run_name="__main__", alter_sys=True)
"""


@dataclass(frozen=True)
class TRTLLMMooncakeKVStoreConfig:
    """Pool master settings for TRT-LLM's ``mooncake_store`` connector.

    The master address file is generated in the shared ``/logs`` mount. The
    connector reads it via ``file:///logs/mooncake_master.addr``.
    """

    container: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    master_extra_args: list[str] = field(default_factory=list)
    eviction_ratio: float = 0.05
    master_timeout_s: int = 60
    store_role: Literal["both", "producer", "consumer"] = "both"

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if self.master_timeout_s <= 0:
            raise ValueError("mooncake_kv_store.master_timeout_s must be positive")
        if not 0 < self.eviction_ratio < 1:
            raise ValueError("mooncake_kv_store.eviction_ratio must be between 0 and 1")


# Log lines that mean the engine behind a TRT-LLM worker step is gone while the
# step itself may stay up. ``trtllm-llmapi-launch`` runs the engine as a child of
# the rank-0 task and prints ``Rank<N> Task exit code: <code>`` when that child
# exits; the follower ranks block in ``MPICommExecutor`` with no timeout, so
# neither srun nor the process registry hears about the death otherwise.
# ``Failed to initialize executor`` is TRT-LLM's own terminal start-up line.
# Deliberately absent: ``Traceback`` (Dynamo logs a "response stream is closed"
# traceback for every request the client cancels at EOS) and ``MPI_Abort``
# (printed on ordinary teardown). A bare-word marker here would fail healthy runs.
TRTLLM_FATAL_LOG_PATTERNS: tuple[str, ...] = (
    r"^Rank\d+ Task exit code: (?!0$)\d+$",
    r"Failed to initialize executor",
)


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

    # Extra `trtllm-serve` CLI flags per mode, appended verbatim to the worker
    # command (frontend.type: trtllm_serve only -- dynamo.trtllm takes a
    # different CLI).
    #
    # `trtllm_config` already covers everything that belongs in the engine YAML,
    # which is nearly everything: trtllm-serve merges that file into LlmArgs. But
    # a few of its options configure the OpenAI SERVER layer rather than the
    # engine and have no LlmArgs field, so no YAML key can reach them. The one
    # that matters in practice is `--tool_parser` (a click.Choice consumed
    # directly by the server constructor); note that its sibling
    # `--reasoning_parser` IS forwarded into get_llm_args() and so remains
    # settable from `trtllm_config`.
    #
    #     backend:
    #       type: trtllm
    #       prefill_extra_args: ["--tool_parser", "glm47"]
    #       decode_extra_args:  ["--tool_parser", "glm47"]
    prefill_extra_args: list[str] = field(default_factory=list)
    decode_extra_args: list[str] = field(default_factory=list)
    aggregated_extra_args: list[str] = field(default_factory=list)

    trtllm_config: TRTLLMServerConfig | None = None

    # Mooncake pool master; a declared mooncake-master service maps here in v2.
    mooncake_kv_store: TRTLLMMooncakeKVStoreConfig | None = None

    # The name clients must use in a request's "model" field.
    # Defaults to the checkpoint directory name.
    #
    #     engine:
    #       type: trtllm
    #       served_model_name: "deepseek-ai/deepseek-r1"
    #
    # Set it when the client cannot be told which name to ask for. agentperf
    # takes the name as a flag, so it never needs this; the MLPerf harness has
    # it fixed in the benchmark definition, so the server must match or every
    # request 404s.
    #
    # Top-level rather than a trtllm_config key because trtllm_config is dumped
    # straight into the engine's YAML file, and this is a launcher flag the
    # engine does not recognise.
    served_model_name: str | None = None

    # Publish TRT-LLM engine metrics without enabling KV-cache events.
    # Requires a Dynamo build supporting --publish-metrics; set False to omit
    # the flag for older builds. Native trtllm-serve and sidecars are unaffected.
    # Iteration statistics stay off regardless: srtctl bakes
    # enable_iter_perf_stats: false into every engine section unless the recipe
    # or observability sets it (TRTLLM_ENGINE_DEFAULTS), so this flag costs the
    # per-request perf metrics only.
    publish_metrics: bool = True

    # None means unspecified: metrics default on, events off (observability
    # promotes this to True). Explicit False is a master opt-out of BOTH
    # publication flags, even when publish_metrics is True. Preserve None in
    # schema round-trips so an omitted value never becomes an explicit opt-out.
    publish_events_and_metrics: bool | None = None

    # Controls batched startup of workers that share the same node.
    # 0 = start all workers in parallel (no constraint).
    # 1 = fully sequential: one worker at a time, each must be ready before the next.
    # N > 1 = start N workers simultaneously per batch, wait for all to be ready, then next batch.
    # For trtllm_serve: readiness is an HTTP 200 on the worker's http_port.
    # For dynamo.trtllm: readiness is a TCP connection on the worker's sys_port.
    sequential_node_start: int = 0

    # Worker memory policy. None (default) uses `numactl -m 0,1` only for
    # gb200/gb300/vrnvl72 prefill and decode workers (case-sensitive GPU type).
    # True uses nodes 0,1 for any GPU type or mode; False leaves the policy
    # unchanged. CPU binding does not change these policies. "local" strictly
    # binds prefill/aggregated memory to the task GPU's NUMA node independently
    # of CPU binding; decode keeps the two-node policy (nodes 0,1).
    # Local mode fails startup if GPU NUMA affinity is unknown. Local memory
    # exhaustion can fail allocations; existing/shared pages are not migrated.
    numa_memory_bind: bool | Literal["local"] | None = None

    # Optional stricter NUMA CPU affinity for the worker process, in addition
    # to numa_memory_bind. A previous post-hoc `taskset -pc <cpuset> $PPID`
    # approach (see bind-b300-prefill-cpus.sh) only pins the leader PID
    # *after* launch, so secondary threads spawned by Python/UCX/MPI/TRT-LLM
    # can still land cross-socket. When true, srtctl instead:
    #   1. sets TLLM_NUMA_AWARE_WORKER_AFFINITY=0 (disables TRT-LLM's own
    #      internal NUMA thread-pinning, which fights with the OS-level mask)
    #   2. wraps the worker command (prefill/decode/agg) in `taskset -c
    #      <cpu_list>`, applied *before* exec so every spawned thread
    #      inherits the mask. The CPU list is discovered at runtime
    #      (configs/numa_cpu_bind.sh) from the physical GPU this task owns,
    #      not a static SLURM_LOCALID table — a static table assumes
    #      SLURM_LOCALID is a node-wide GPU ordinal, which breaks when two
    #      endpoints share a node (each gets its own srun step, so LOCALID
    #      restarts at 0 for both).
    # Set numa_memory_bind="local" to also bind memory to that same NUMA node.
    numa_cpu_bind: bool = False

    # Decode-only CPU binding override. None inherits numa_cpu_bind; prefill
    # and aggregated workers continue to use numa_cpu_bind.
    decode_numa_cpu_bind: bool | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    @property
    def dynamo_metrics_flags(self) -> tuple[str, ...]:
        """Effective publication flags, preserving the explicit legacy opt-out."""
        if self.publish_events_and_metrics is False:
            return ()
        flags = []
        if self.publish_metrics:
            flags.append("--publish-metrics")
        if self.publish_events_and_metrics:
            flags.append("--publish-events-and-metrics")
        return tuple(flags)

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
            sequential_node_start=self.sequential_node_start,
            # A rank exiting non-zero (or the rank-zero sidecar) must end the
            # whole endpoint step; the launcher would otherwise keep it up.
            kill_on_bad_exit=True,
        )

    def fatal_log_patterns(self, mode: WorkerMode) -> tuple[str, ...]:
        """The launcher's task-exit line and the executor's start-up failure, for every mode."""
        return TRTLLM_FATAL_LOG_PATTERNS

    @property
    def failover(self) -> None:
        """TRT-LLM has no shadow engine recovery."""
        return None

    def get_mooncake_worker_env(self, infra_node_ip: str, local_hostname: str) -> dict[str, str]:
        if self.mooncake_kv_store is None:
            return {}
        return {
            **self.mooncake_kv_store.env,
            "TRTLLM_MOONCAKE_MASTER_TIMEOUT": str(self.mooncake_kv_store.master_timeout_s),
        }

    def get_failover_environment(self, process: "Process", job_id: str) -> dict[str, str]:
        return {}

    def should_set_visible_devices(self) -> bool:
        return True

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

    def get_extra_args_for_mode(self, mode: WorkerMode) -> list[str]:
        """Extra trtllm-serve CLI flags for this mode (see the field docs)."""
        by_mode: dict[WorkerMode, list[str]] = {
            "prefill": self.prefill_extra_args,
            "decode": self.decode_extra_args,
            "agg": self.aggregated_extra_args,
        }
        return list(by_mode.get(mode) or [])

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
        env = {**base_env, "TRTLLM_EPLB_SHM_NAME": eplb_prefix}
        if mode == "prefill" and self.mooncake_kv_store is not None:
            env["TRTLLM_MOONCAKE_STORE_ROLE"] = self.mooncake_kv_store.store_role
        if self.numa_cpu_bind_for_mode(mode):
            env["TLLM_NUMA_AWARE_WORKER_AFFINITY"] = "0"
        return env

    def get_process_environment(self, process: "Process") -> dict[str, str]:
        """Get process-specific environment variables.

        TRTLLM doesn't currently require process-specific env vars.
        """
        return {}

    def get_served_model_name(self, default: str) -> str:
        """Get the configured served model name, or return default."""
        return self.served_model_name or default

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
            spread_workers=spread_workers,
            pack_multinode_workers=True,
        )

    def endpoints_to_processes(
        self,
        endpoints: list["Endpoint"],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: "NodePortAllocator | None" = None,
        frontend_type: str = "dynamo",
        dynamo_sidecar: bool = False,
    ) -> list["Process"]:
        """Convert endpoints to processes, each with its torch.distributed bootstrap port."""
        from srtctl.core.topology import endpoints_to_processes, port_allocator_for

        allocator = port_allocator_for(port_allocator, base_sys_port)
        processes = endpoints_to_processes(endpoints, port_allocator=allocator, sidecar_grpc=dynamo_sidecar)
        # MASTER_PORT for the endpoint is the leader's; every process gets one so
        # the allocation is uniform and any rank could lead.
        return [replace(p, trtllm_dist_init_port=allocator.next(TRTLLM_DIST_INIT_PORTS)) for p in processes]

    def numa_cpu_bind_for_mode(self, mode: WorkerMode) -> bool:
        """Resolve the decode CPU override, falling back to the shared policy."""
        if mode == "decode" and self.decode_numa_cpu_bind is not None:
            return self.decode_numa_cpu_bind
        return self.numa_cpu_bind

    def _wrap_with_numa_bind(self, cmd: list[str], *, bind_memory: bool, mode: WorkerMode) -> list[str]:
        """Resolve the task GPU's NUMA node for independent CPU and memory policies.

        Applies to all worker modes (prefill/decode/agg) when CPU binding or
        local memory binding is enabled. Placement depends on which physical GPU the task owns
        (resolved from CUDA_VISIBLE_DEVICES and SLURM_LOCALID) and srun sets
        SLURM_LOCALID per-task at launch time — since the same argv is
        replicated across all ranks of the endpoint's srun (MPI-style
        launch), the lookup must happen in a script at runtime rather than
        being baked into the static command list.
        """
        bind_cpu = self.numa_cpu_bind_for_mode(mode)
        if not bind_cpu and not bind_memory:
            return cmd
        memory_args = ["--bind-memory"] if bind_memory else []
        if not bind_cpu:
            memory_args.append("--no-bind-cpu")
        return ["bash", "/configs/numa_cpu_bind.sh", *memory_args, *cmd]

    def build_worker_command(
        self,
        process: "Process",
        endpoint_processes: list["Process"],
        runtime: "RuntimeContext",
        frontend_type: str = "dynamo",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Path | None = None,
        profiling: "ProfilingConfig | None" = None,
    ) -> list[str]:
        """Build the command to start a TRTLLM worker process."""

        from srtctl.frontends import get_frontend

        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)
        # The frontend owns the worker shape; nothing below compares frontend names.
        frontend = get_frontend(frontend_type)

        sidecar_config = get_dynamo_sidecar_config(runtime)
        if sidecar_config is not None:
            if frontend.worker_launch != "dynamo":
                raise ValueError("TensorRT-LLM sidecar mode requires frontend.type: dynamo")
            if mode != "agg":
                raise ValueError("TensorRT-LLM sidecar mode supports aggregated workers only")

        # Write config to host path (log_dir)
        config_filename = f"trtllm_config_{mode}.yaml"
        host_config_path = runtime.log_dir / config_filename
        host_config_path.write_text(yaml.safe_dump(config))

        # Use container paths for the command (log_dir is mounted to /logs)
        container_config_path = Path("/logs") / config_filename

        # Determine model path: HF model ID or container mount path
        # For HF models (hf:prefix), model_path contains the HF model ID (e.g., "facebook/opt-125m")
        # For local models, model is mounted to /model in the container
        model_arg = runtime.worker_model_arg

        # Temporary A/B policy: keep decode on both NUMA nodes when testing local prefill memory.
        memory_bind = self.numa_memory_bind
        if memory_bind == "local" and mode == "decode":
            memory_bind = True
        if memory_bind is None:
            use_numactl = runtime.gpu_type in ("gb200", "gb300", "vrnvl72") and mode in ("prefill", "decode")
        else:
            use_numactl = memory_bind is True
        # Only explicit local mode moves the memory policy into the CPU wrapper.
        bind_local_memory = memory_bind == "local"
        numactl_prefix = ["numactl", "-m", "0,1"] if use_numactl else []
        base_prefix = list(nsys_prefix or []) + numactl_prefix + ["trtllm-llmapi-launch"]

        if sidecar_config is not None:
            return self._build_sidecar_command(
                process=process,
                config=config,
                model_arg=model_arg,
                container_config_path=container_config_path,
                base_prefix=base_prefix,
                sidecar_config=sidecar_config,
                bind_memory=bind_local_memory,
            )

        # trtllm-serve path: launch an OpenAI-compatible trtllm-serve worker. In
        # disaggregated mode the trtllm_serve frontend fronts these via a static
        # ser.yaml (context/generation server URLs). In aggregated mode the one
        # worker is also the public frontend, so it binds runtime.frontend_port.
        # There is no Dynamo request plane and no --disaggregation-mode: a disagg
        # worker is prefill or decode purely by which list it appears in in ser.yaml.
        if frontend.worker_launch == "direct":
            http_port = runtime.frontend_port if frontend.worker_api_port(mode) == "public" else process.http_port
            cmd = base_prefix + [
                "trtllm-serve",
                model_arg,
                "--host",
                "0.0.0.0",
                "--port",
                str(http_port),
            ]
            # Parallelism also lives in the engine yaml, but pass it explicitly to match
            # the trtllm-serve CLI contract (srun --ntasks == TP*PP is set by the worker stage).
            for flag, key in (
                ("--tensor_parallel_size", "tensor_parallel_size"),
                ("--moe_expert_parallel_size", "moe_expert_parallel_size"),
                ("--pipeline_parallel_size", "pipeline_parallel_size"),
            ):
                value = config.get(key)
                if value is not None:
                    cmd.extend([flag, str(value)])
            # Engine config file. Verified against tensorrt-llm 1.3.0rc15/rc17 and the
            # ai-dynamo tensorrtllm-runtime 1.3.0-dev.1 container, which accept --config;
            # some trtllm-serve builds spell this --extra_llm_api_options.
            cmd.extend(["--config", str(container_config_path)])
            if self.served_model_name:
                cmd.extend(["--served_model_name", self.served_model_name])
            cmd.extend(self.get_extra_args_for_mode(mode))
            return self._wrap_with_numa_bind(cmd, bind_memory=bind_local_memory, mode=mode)

        # dynamo.trtllm path (default): workers register into etcd/NATS and the dynamo
        # frontend discovers them.
        entrypoint = ["python3", "-m", "dynamo.trtllm"]
        connector = config.get("kv_connector_config") or {}
        if connector.get("mooncake_store") is not None:
            entrypoint = ["python3", "-c", _DYNAMO_MOONCAKE_ENTRYPOINT, str(container_config_path)]
        cmd = (
            base_prefix
            + entrypoint
            + [
                "--model-path",
                model_arg,
                "--served-model-name",
                self.get_served_model_name(runtime.model_path.name),
            ]
        )

        # Only add disaggregation mode for prefill/decode, not for agg
        if mode != "agg":
            cmd.extend(["--disaggregation-mode", mode])

        cmd.extend(
            [
                "--extra-engine-args",
                str(container_config_path),
                "--request-plane",
                runtime.request_plane,
            ]
        )

        cmd.extend(self.dynamo_metrics_flags)

        return self._wrap_with_numa_bind(cmd, bind_memory=bind_local_memory, mode=mode)

    def _build_sidecar_command(
        self,
        *,
        process: "Process",
        config: dict[str, Any],
        model_arg: str,
        container_config_path: Path,
        base_prefix: list[str],
        sidecar_config: "DynamoConfig",
        bind_memory: bool,
    ) -> list[str]:
        """Build a lifecycle-coupled TensorRT-LLM native-gRPC and sidecar launch."""
        grpc_port = sidecar_grpc_port(process)
        engine = self._wrap_with_numa_bind(
            base_prefix
            + [
                "python3",
                "-m",
                "tensorrt_llm.commands.serve",
                model_arg,
                "--grpc",
                "--host",
                "127.0.0.1",
                "--port",
                str(grpc_port),
                "--extra_llm_api_options",
                str(container_config_path),
            ],
            bind_memory=bind_memory,
            mode=process.endpoint_mode,
        )

        sidecar = (
            [sidecar_config.sidecar_binary]
            if sidecar_config.sidecar_binary is not None
            else ["python3", "-m", "dynamo.trtllm.sidecar"]
        )
        sidecar.extend(
            [
                "--grpc-endpoint",
                f"127.0.0.1:{grpc_port}",
                "--model-path",
                model_arg,
            ]
        )
        context_length = sidecar_config.sidecar_context_length
        if context_length is None:
            context_length = config.get("max_seq_len") or config.get("max-seq-len")
        if context_length is not None:
            sidecar.extend(["--context-length", str(context_length)])
        sidecar.extend(sidecar_config.sidecar_args)

        return build_sidecar_launch_command(
            engine=engine,
            sidecar=sidecar,
            grpc_port=grpc_port,
            engine_name="TensorRT-LLM",
            startup_timeout=sidecar_config.sidecar_startup_timeout,
            rank_zero_only=True,
        )
