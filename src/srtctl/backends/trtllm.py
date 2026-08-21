# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import builtins
import shlex
import uuid
from collections.abc import Sequence
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import yaml
from marshmallow import Schema, ValidationError
from marshmallow_dataclass import dataclass

from srtctl.ports import DYN_SYSTEM_PORT_BASE

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

# Type alias for worker modes
WorkerMode = Literal["prefill", "decode", "agg"]

# Default memory binding applied on Grace-based nodes when `cpu_binding` is not
# configured. Kept for backwards compatibility with recipes written before
# `cpu_binding` existed.
_LEGACY_MEMBIND_GPU_TYPES = ("gb200", "gb300")
_LEGACY_MEMBIND = "0,1"

# Bash array name used to select a per-task CPU list at launch time. Prefixed to
# avoid colliding with anything the container's own entrypoint may define.
_CPU_LIST_VAR = "__srt_cpu_list"


@dataclass(frozen=True)
class TRTLLMCpuBinding:
    """Pin TRTLLM worker ranks to CPUs *before* the worker process starts.

    TRTLLM has its own NUMA-aware affinity (``configure_cpu_affinity`` in
    ``tensorrt_llm/llmapi/utils.py``), but it runs from inside the already-running
    worker, so ``sched_setaffinity`` only re-pins the calling thread. Threads that
    the process created earlier keep the unconstrained mask and stay free to run on
    a remote socket. On a node with more than one CPU NUMA domain, a cross-domain
    hop is expensive enough that those stray threads show up as decode-loop bubbles.

    Applying the mask with ``taskset`` before ``exec`` fixes that: every thread the
    worker will ever create inherits it. The companion
    ``TLLM_NUMA_AWARE_WORKER_AFFINITY`` setting is required, because with that
    variable unset TRTLLM treats an externally constrained mask as a mistake and
    *removes* it again (``process.cpu_affinity(all_cpus)``).

    Example YAML for a node with 4 GPUs and 2 CPU NUMA domains, where GPUs 0-1 are
    local to CPUs 0-71 and GPUs 2-3 to CPUs 72-143::

        backend:
          type: trtllm
          cpu_binding:
            gpu_types: ["gb300"]
            cpus_per_local_gpu: ["0-71", "0-71", "72-143", "72-143"]
            membind: "0,1"

    Attributes:
        cpus: ``taskset -c`` list applied to every rank of the endpoint. Use when a
            single CPU set is correct for all ranks on the node.
        cpus_per_local_gpu: ``taskset -c`` list per node-local GPU index. Entry ``i``
            is the CPU set for the GPU with node-local index ``i``, so it describes
            the node's topology once and srtslurm maps it onto whichever GPUs an
            endpoint was actually allocated. Mutually exclusive with ``cpus``.
        gpu_types: ``resources.gpu_type`` values these CPU lists were written for.
            CPU layout is a property of the node, so a stale list copied to different
            hardware pins ranks to the wrong socket — a silent perf loss that looks
            like nothing at all. Naming the machine types turns that into a config
            error at load time (dry-run) instead. ``null`` (default) accepts any
            ``gpu_type``.
        membind: Value passed to ``numactl -m`` (memory-node binding, no CPU effect).
            ``null`` disables the ``numactl`` wrapper entirely.
        numa_aware_worker_affinity: Value exported as
            ``TLLM_NUMA_AWARE_WORKER_AFFINITY``. Defaults to ``"0"`` so TRTLLM leaves
            the externally applied mask alone; a per-mode ``*_environment`` entry
            still wins over this.
    """

    cpus: str | None = None
    cpus_per_local_gpu: list[str] | None = None
    gpu_types: list[str] | None = None
    membind: str | None = _LEGACY_MEMBIND
    numa_aware_worker_affinity: str = "0"

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if self.cpus and self.cpus_per_local_gpu:
            raise ValidationError(
                "cpu_binding.cpus and cpu_binding.cpus_per_local_gpu are mutually exclusive; "
                "use cpus for one CPU set shared by every rank, or cpus_per_local_gpu to "
                "describe the node's per-GPU NUMA topology."
            )
        if self.cpus_per_local_gpu is not None and not self.cpus_per_local_gpu:
            raise ValidationError("cpu_binding.cpus_per_local_gpu must not be empty")
        if self.cpus_per_local_gpu and any(not entry.strip() for entry in self.cpus_per_local_gpu):
            raise ValidationError("cpu_binding.cpus_per_local_gpu entries must be non-empty CPU lists")
        if self.gpu_types is not None and not self.gpu_types:
            raise ValidationError("cpu_binding.gpu_types must not be empty; omit it to apply to every GPU type")

    @property
    def pins_cpus(self) -> bool:
        """Whether this config actually constrains CPU affinity (vs. memory only)."""
        return bool(self.cpus or self.cpus_per_local_gpu)

    def applies_to(self, gpu_type: str | None) -> bool:
        """Whether this binding is in scope for a run's ``resources.gpu_type``."""
        if self.gpu_types is None:
            return True
        return gpu_type in self.gpu_types

    def cpu_list_for_gpus(self, local_gpu_indices: Sequence[int]) -> list[str] | None:
        """Resolve the per-task CPU lists for the node-local GPUs of one endpoint.

        Returns one entry per task in node-local task order (``SLURM_LOCALID``),
        or ``None`` when no per-GPU mapping is configured.
        """
        if not self.cpus_per_local_gpu:
            return None
        table = self.cpus_per_local_gpu
        return [table[index % len(table)] for index in sorted(local_gpu_indices)]


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

    # CPU/memory pinning for worker ranks. When unset, srtslurm keeps the historical
    # behavior: `numactl -m 0,1` (memory only) on gb200/gb300 prefill/decode workers,
    # and TRTLLM's own in-process NUMA-aware affinity decides CPU placement.
    cpu_binding: TRTLLMCpuBinding | None = None

    # Whether dynamo.trtllm workers pass `--publish-events-and-metrics`.
    # Enables the worker to publish KV-cache events (add/evict) + metrics, which
    # the dynamo frontend consumes for KV-cache-aware routing (router-mode: kv).
    # This may impact performance so should be disabled if exact KV aware routing
    # is not needed.
    publish_events_and_metrics: bool = False

    # Controls batched startup of workers that share the same node.
    # 0 = start all workers in parallel (no constraint).
    # 1 = fully sequential: one worker at a time, each must be ready before the next.
    # N > 1 = start N workers simultaneously per batch, wait for all to be ready, then next batch.
    # For trtllm_serve: readiness is an HTTP 200 on the worker's http_port.
    # For dynamo.trtllm: readiness is a TCP connection on the worker's sys_port.
    sequential_node_start: int = 0

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

        # Emitted first so an explicit `*_environment` entry still wins: pinning with
        # taskset is pointless unless TRTLLM is told to leave the mask alone, but the
        # recipe author keeps the final say.
        affinity_env: dict[str, str] = {}
        if self.cpu_binding is not None and self.cpu_binding.pins_cpus:
            affinity_env["TLLM_NUMA_AWARE_WORKER_AFFINITY"] = self.cpu_binding.numa_aware_worker_affinity

        return {**affinity_env, **base_env, "TRTLLM_EPLB_SHM_NAME": eplb_prefix}

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
        )

    def endpoints_to_processes(
        self,
        endpoints: list["Endpoint"],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: "NodePortAllocator | None" = None,
        frontend_type: str = "dynamo",
    ) -> list["Process"]:
        """Convert endpoints to processes."""
        from srtctl.core.topology import endpoints_to_processes

        return endpoints_to_processes(endpoints, base_sys_port=base_sys_port, port_allocator=port_allocator)

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
        model_arg = runtime.worker_model_arg

        # CPU/memory binding. `leading` (nsys) stays outermost so profiling still wraps
        # the whole worker; the taskset/numactl pair goes between it and the launcher so
        # the mask is installed by the last exec before TRTLLM's own threads exist.
        leading_prefix = list(nsys_prefix or [])
        taskset_prefix, per_task_cpus = self._cpu_pin_prefix(process)
        membind = self._membind_for(runtime, mode)
        numactl_prefix = ["numactl", "-m", membind] if membind else []
        base_prefix = leading_prefix + taskset_prefix + numactl_prefix + ["trtllm-llmapi-launch"]

        # trtllm-serve path: launch an OpenAI-compatible trtllm-serve worker. In
        # disaggregated mode the trtllm_serve frontend fronts these via a static
        # ser.yaml (context/generation server URLs). In aggregated mode the one
        # worker is also the public frontend, so it binds runtime.frontend_port.
        # There is no Dynamo request plane and no --disaggregation-mode: a disagg
        # worker is prefill or decode purely by which list it appears in in ser.yaml.
        if frontend_type == "trtllm_serve":
            http_port = runtime.frontend_port if mode == "agg" else process.http_port
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
            return _resolve_per_task_cpus(cmd, len(leading_prefix), per_task_cpus)

        # dynamo.trtllm path (default): workers register into etcd/NATS and the dynamo
        # frontend discovers them.
        cmd = base_prefix + [
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
                runtime.request_plane,
            ]
        )

        if self.publish_events_and_metrics:
            cmd.append("--publish-events-and-metrics")

        return _resolve_per_task_cpus(cmd, len(leading_prefix), per_task_cpus)

    # =========================================================================
    # CPU / memory binding helpers
    # =========================================================================

    def _membind_for(self, runtime: "RuntimeContext", mode: WorkerMode) -> str | None:
        """Value for ``numactl -m``, or None to skip the numactl wrapper."""
        if self.cpu_binding is not None:
            return self.cpu_binding.membind
        # Legacy default, preserved for recipes written before `cpu_binding` existed:
        # memory-only interleave across both Grace sockets on prefill/decode workers.
        if runtime.gpu_type in _LEGACY_MEMBIND_GPU_TYPES and mode in ("prefill", "decode"):
            return _LEGACY_MEMBIND
        return None

    def _cpu_pin_prefix(self, process: "Process") -> tuple[list[str], list[str] | None]:
        """Resolve CPU pinning for one endpoint.

        Returns ``(taskset_prefix, per_task_cpus)``. At most one is non-empty: a
        single CPU set becomes a literal ``taskset -c`` prefix, while a per-GPU
        topology map has to be selected by node-local task id at launch time and is
        returned for :func:`_resolve_per_task_cpus` to render.
        """
        if self.cpu_binding is None:
            return [], None
        if self.cpu_binding.cpus:
            return ["taskset", "-c", self.cpu_binding.cpus], None
        return [], self.cpu_binding.cpu_list_for_gpus(sorted(process.gpu_indices))


def _resolve_per_task_cpus(cmd: list[str], leading: int, per_task_cpus: list[str] | None) -> list[str]:
    """Insert a per-task ``taskset -c`` that is resolved inside the srun task.

    One srun launches every rank of a TRTLLM endpoint, so a per-GPU CPU map cannot be
    baked into the argv — the rank is only known once the task is running. Wrap the
    command in ``bash -c`` and index the CPU table by ``SLURM_LOCALID`` instead. The
    table is already ordered by this endpoint's node-local GPUs, so entry ``i`` belongs
    to node-local task ``i``. ``exec`` keeps the process tree the same depth as the
    unwrapped command.

    ``leading`` is the number of argv entries (the nsys prefix) that must stay outside
    the taskset call.
    """
    if not per_task_cpus:
        return cmd

    head, tail = cmd[:leading], cmd[leading:]
    table = " ".join(shlex.quote(entry) for entry in per_task_cpus)
    index = f"${{SLURM_LOCALID:-0}} % ${{#{_CPU_LIST_VAR}[@]}}"
    pin = f'taskset -c "${{{_CPU_LIST_VAR}[{index}]}}"'
    parts = [part for part in (shlex.join(head), pin, shlex.join(tail)) if part]
    return ["bash", "-c", f"{_CPU_LIST_VAR}=({table}); exec {' '.join(parts)}"]
