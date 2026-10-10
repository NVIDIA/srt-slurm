# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import builtins
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import yaml
from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.backends.base import (
    Backend,
    BoundRolesField,
    ProcessGroup,
    RoleSettings,
    WorkerMode,
    role_env,
    role_for_mode,
)
from srtctl.backends.sidecar import build_sidecar_launch_command, get_dynamo_sidecar_config, sidecar_grpc_port
from srtctl.ports import DYN_SYSTEM_PORT_BASE, TRTLLM_DIST_INIT_PORTS

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import DynamoConfig, ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process


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

# DWDP (TRT-LLM's distributed-weight data-parallel prefill, Kimi-K3 on GB300/R200):
# ``dwdp_config: {dwdp_size: N, ...}`` in a role's engine args makes N single-GPU
# ``trtllm-serve`` workers of that role one expert-sharded group. Each worker is
# its own endpoint here (``gpus: 1``), and before loading weights it joins a
# TCPStore rendezvous that TRT-LLM reads from the environment: its own
# ``TRTLLM_DWDP_RANK`` and the group's shared ``TRTLLM_DWDP_MASTER_ADDR`` /
# ``TRTLLM_DWDP_MASTER_PORT``. The rank is the worker's endpoint index (worker 0
# is the master), and the worker stage fills in the master's address from the
# rank-0 endpoint of the same role, since a backend only sees its own process.
#
# The N processes also share the model's ``trust_remote_code`` module cache and
# the CuTe DSL autotuner cache by default, and race on them (a half-written
# ``transformers_modules`` file or a cache owned by another rank fails the
# executor init). Each rank therefore gets a private ``HOME`` /
# ``HF_MODULES_CACHE`` / ``TLLM_AUTOTUNER_CACHE_PATH`` under the job's log dir; a
# recipe that sets any of those three in the role env keeps its own value.
DWDP_RANK_ENV = "TRTLLM_DWDP_RANK"
DWDP_MASTER_ADDR_ENV = "TRTLLM_DWDP_MASTER_ADDR"
DWDP_MASTER_PORT_ENV = "TRTLLM_DWDP_MASTER_PORT"
DWDP_RENDEZVOUS_TIMEOUT_ENV = "TRTLLM_DWDP_RENDEZVOUS_TIMEOUT_S"
DWDP_MASTER_PORT = 29600
DWDP_RENDEZVOUS_TIMEOUT_S = 3600
# Container path of the job's log dir (what build_worker_command writes the engine YAML under).
_CONTAINER_LOG_DIR = "/logs"


@dataclass(frozen=True)
class TRTLLMBackend(Backend):
    """TRTLLM backend configuration and launch implementation.

    This frozen dataclass both holds configuration AND implements the
    Backend methods for process allocation and launching.

    Example YAML:
        engine: trtllm
        roles:
          prefill:
            env:
              CUDA_LAUNCH_BLOCKING: "1"
            args:
              max_batch_size: 256
          decode:
            args:
              max_batch_size: 64
    """

    # Engine type discriminator.
    type: Literal["trtllm"] = "trtllm"
    # trtllm-serve serves Prometheus text at /prometheus/metrics (mounted when
    # return_perf_metrics is true); its /metrics route is JSON iteration stats.
    prometheus_metrics_path: ClassVar[str] = "/prometheus/metrics"
    nvtx_environment: ClassVar[Mapping[str, str]] = {"TLLM_LLMAPI_ENABLE_NVTX": "1", "TLLM_PROFILE_LOG_RANKS": "all"}

    # The roles this engine runs (`roles.<role>` of the recipe), bound by SrtConfig and
    # never written on `engine:`. Per-role env and args (the engine YAML) are read from
    # here, and so are `roles.<role>.extra_args`: extra `trtllm-serve` CLI flags appended
    # verbatim to the worker command (frontend.type: trtllm_serve only -- dynamo.trtllm
    # takes a different CLI).
    #
    # `args` already covers everything that belongs in the engine YAML, which is nearly
    # everything: trtllm-serve merges that file into LlmArgs. But a few of its options
    # configure the OpenAI SERVER layer rather than the engine and have no LlmArgs field,
    # so no YAML key can reach them. The one that matters in practice is `--tool_parser`
    # (a click.Choice consumed directly by the server constructor); note that its sibling
    # `--reasoning_parser` IS forwarded into get_llm_args() and so remains settable from
    # `args`.
    #
    #     roles:
    #       prefill:
    #         extra_args: ["--tool_parser", "glm47"]
    #       decode:
    #         extra_args: ["--tool_parser", "glm47"]
    roles: Mapping[str, RoleSettings] = field(default_factory=dict, metadata={"marshmallow_field": BoundRolesField()})

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
    # Top-level rather than a roles.<role>.args key because a role's args are dumped
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

    # Legacy compatibility flag for Dynamo builds without --publish-metrics.
    # True emits only --publish-events-and-metrics, regardless of publish_metrics.
    # False or None uses publish_metrics instead. Observability does not enable it.
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
    # unchanged. CPU binding does not change these policies. "local" requires
    # numa_cpu_bind=True and strictly binds memory to the task GPU's NUMA node.
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

    Schema: ClassVar[builtins.type[Schema]] = Schema
    process_exporter_groups: ClassVar[tuple[ProcessGroup, ...]] = (
        ProcessGroup("trtllm_llmapi_launch", cmdline=(r"(^|[ /])trtllm-llmapi-launch( |$)",)),
        ProcessGroup("trtllm_engine", cmdline=(r"(^| )tensorrt_llm\.llmapi\.mgmn_worker_node( |$)",)),
        ProcessGroup("dynamo_trtllm", cmdline=(r"dynamo\.trtllm",)),
        ProcessGroup("trtllm_serve", cmdline=("trtllm-serve",)),
    )

    def __post_init__(self) -> None:
        if self.numa_memory_bind == "local" and not self.numa_cpu_bind:
            raise ValueError("numa_memory_bind: local requires numa_cpu_bind: true")

    @property
    def dynamo_metrics_flags(self) -> tuple[str, ...]:
        """Select the legacy combined flag or the metrics-only flag exclusively."""
        if self.publish_events_and_metrics:
            return ("--publish-events-and-metrics",)
        return ("--publish-metrics",) if self.publish_metrics else ()

    # =========================================================================
    # Backend Implementation
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

    def get_extra_args_for_mode(self, mode: WorkerMode) -> list[str]:
        """Extra trtllm-serve CLI flags for this mode (``roles.<role>.extra_args``)."""
        role = role_for_mode(self.roles, mode)
        return list(role.extra_args) if role is not None else []

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        eplb_prefix = f"moe_shared_{uuid.uuid4().hex}"
        env = {**role_env(self.roles, mode), "TRTLLM_EPLB_SHM_NAME": eplb_prefix}
        if self.numa_cpu_bind:
            env["TLLM_NUMA_AWARE_WORKER_AFFINITY"] = "0"
        return env

    def get_served_model_name(self, default: str) -> str:
        """Get the configured served model name, or return default."""
        return self.served_model_name or default

    def dwdp_size(self, mode: WorkerMode) -> int:
        """``dwdp_config.dwdp_size`` of a role's engine args; 0 when the role is not a DWDP group."""
        dwdp = self.get_config_for_mode(mode).get("dwdp_config")
        if not isinstance(dwdp, Mapping):
            return 0
        try:
            return int(dwdp.get("dwdp_size") or 0)
        except (TypeError, ValueError):
            return 0

    def dwdp_private_dir(self, process: "Process") -> str:
        """Container path of a DWDP worker's private HOME (module and autotuner caches live under it)."""
        return f"{_CONTAINER_LOG_DIR}/worker_priv/{process.endpoint_mode}_w{process.endpoint_index}"

    def get_process_environment(self, process: "Process") -> dict[str, str]:
        """Per-worker DWDP environment (see the DWDP note at the top of this module); empty otherwise.

        The master address is not known here (another endpoint's node) -- the
        worker stage sets ``TRTLLM_DWDP_MASTER_ADDR`` from the role's rank-0
        endpoint when it sees ``TRTLLM_DWDP_RANK``.
        """
        mode = process.endpoint_mode
        if not self.dwdp_size(mode):
            return {}
        role_env = self.get_environment_for_mode(mode)
        private = self.dwdp_private_dir(process)
        env = {
            DWDP_RANK_ENV: str(process.endpoint_index),
            DWDP_MASTER_PORT_ENV: str(DWDP_MASTER_PORT),
            DWDP_RENDEZVOUS_TIMEOUT_ENV: str(DWDP_RENDEZVOUS_TIMEOUT_S),
        }
        for key, value in (
            ("HOME", private),
            ("HF_MODULES_CACHE", f"{private}/hf_modules"),
            ("TLLM_AUTOTUNER_CACHE_PATH", f"{private}/autotune_cache.json"),
        ):
            if key not in role_env:
                env[key] = value
        # The recipe may pin the port / timeout for the whole role.
        for key in (DWDP_MASTER_PORT_ENV, DWDP_RENDEZVOUS_TIMEOUT_ENV):
            if key in role_env:
                env.pop(key)
        return env

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

    def _wrap_with_numa_cpu_bind(self, cmd: list[str], *, bind_memory: bool) -> list[str]:
        """Wrap ``cmd`` in configs/numa_cpu_bind.sh, which taskset-binds per task.

        Applies to all worker modes (prefill/decode/agg) when numa_cpu_bind
        is enabled. The CPU list depends on which physical GPU the task owns
        (resolved from CUDA_VISIBLE_DEVICES and SLURM_LOCALID) and srun sets
        SLURM_LOCALID per-task at launch time — since the same argv is
        replicated across all ranks of the endpoint's srun (MPI-style
        launch), the lookup must happen in a script at runtime rather than
        being baked into the static command list.
        """
        if not self.numa_cpu_bind:
            return cmd
        memory_args = ["--bind-memory"] if bind_memory else []
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

        if self.numa_memory_bind is None:
            use_numactl = runtime.gpu_type in ("gb200", "gb300", "vrnvl72") and mode in ("prefill", "decode")
        else:
            use_numactl = self.numa_memory_bind is True
        # Only explicit local mode moves the memory policy into the CPU wrapper.
        bind_local_memory = self.numa_memory_bind == "local"
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
            return self._wrap_with_numa_cpu_bind(cmd, bind_memory=bind_local_memory)

        # dynamo.trtllm path (default): workers register into etcd/NATS and the dynamo
        # frontend discovers them.
        cmd = base_prefix + [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model-path",
            model_arg,
            "--served-model-name",
            self.get_served_model_name(runtime.model_path.name),
        ]

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

        return self._wrap_with_numa_cpu_bind(cmd, bind_memory=bind_local_memory)

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
        engine = self._wrap_with_numa_cpu_bind(
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
