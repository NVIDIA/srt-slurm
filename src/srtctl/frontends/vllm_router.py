# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official vLLM Router frontend adapter."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.core.health import WorkerHealthResult, probe_http_ok
from srtctl.frontends.base import logical_health_expectations, register_frontend
from srtctl.frontends.static_router import RouterWorker, StaticRouterFrontend
from srtctl.ports import VLLM_DISCOVERY_PORT

if TYPE_CHECKING:
    from srtctl.core.topology import Process


def routed_process_dp_size(backend: Any, process: Process) -> int:
    """Return the number of Router-visible DP ranks behind one base URL.

    Upstream's per-node topology exposes one URL for each node-local hybrid-LB
    pool. A model-parallel replica that spans nodes instead exposes one global
    API leader, so Router must treat that URL as one worker.
    """
    global_dp_size = int(backend._get_dp_size(process.endpoint_mode) or 1)
    if global_dp_size <= 1:
        return 1

    replica_size = backend._get_model_parallel_size(process.endpoint_mode)
    local_gpu_count = len(process.gpu_indices)
    if replica_size > local_gpu_count:
        return 1
    if local_gpu_count % replica_size != 0:
        raise ValueError(
            f"vLLM Router {process.endpoint_mode} local GPU allocation {local_gpu_count} "
            f"is not divisible by TP*PP={replica_size}"
        )
    return local_gpu_count // replica_size


def node_local_data_parallel_size(backend: Any, backend_processes: list[Process]) -> int:
    """Return Router's single DP expansion factor for all advertised URLs."""
    routed_sizes = {routed_process_dp_size(backend, process) for process in backend_processes if process.http_port > 0}
    if len(routed_sizes) > 1:
        sizes = ", ".join(str(size) for size in sorted(routed_sizes))
        raise ValueError(f"vLLM Router requires one uniform node-local DP expansion factor; derived {sizes}")
    return next(iter(routed_sizes), 1)


@register_frontend("vllm-router")
class VLLMRouterFrontend(StaticRouterFrontend):
    """Route aggregate or P/D traffic to direct vLLM API servers."""

    type: ClassVar[str] = "vllm-router"
    required_backend: ClassVar[str | None] = "vllm"
    # Router expands each node-local hybrid-LB pool into its DP ranks itself.
    expands_node_local_dp: ClassVar[bool] = True
    executable: ClassVar[tuple[str, ...]] = ("vllm-router",)
    pd_flag: ClassVar[str] = "--vllm-pd-disaggregation"
    process_name: ClassVar[str] = "vllm_router"

    def validate(self, config: Any) -> None:
        """Router expands each advertised URL by one node-local DP factor, so the vLLM topology must be uniform.

        Every routable worker must be an independently addressable ``vllm serve``
        (``per_node`` DP), its GPU count must equal DP*TP*PP*PCP, and every pool
        must derive the same ``--intra-node-data-parallel-size``.
        """
        backend = config.backend
        resources = config.resources
        endpoint_gpu_counts: dict[str, int] = {
            "prefill": resources.gpus_per_prefill if resources.num_prefill else 0,
            "decode": resources.gpus_per_decode if resources.num_decode else 0,
            "agg": resources.gpus_per_agg if resources.num_agg else 0,
        }
        if backend.find_dp_modes() and backend.dp_launch_mode != "per_node":
            raise ValueError(
                "frontend.type: vllm-router with data-parallel-size requires "
                "backend.dp_launch_mode: per_node; deprecated per_gpu processes are "
                "Dynamo registrations, not independently routable vLLM API servers"
            )

        expansion_by_mode: dict[str, int] = {}
        for mode, gpu_count in endpoint_gpu_counts.items():
            if gpu_count <= 0:
                continue
            if not backend._is_dp_mode(mode):
                expansion_by_mode[mode] = 1
                continue
            try:
                configured_dp_size = backend._get_dp_size(mode)
                dp_size = int(configured_dp_size) if configured_dp_size is not None else 1
                if dp_size < 1:
                    raise ValueError(
                        f"vLLM {mode} data-parallel-size must be a positive integer; got {configured_dp_size!r}"
                    )
                replica_size = backend._get_model_parallel_size(mode)
            except (TypeError, ValueError) as exc:
                raise ValueError(str(exc)) from exc

            required_gpus = dp_size * replica_size
            if required_gpus != gpu_count:
                raise ValueError(
                    f"vLLM Router {mode} parallelism requires DP*TP*PP*PCP="
                    f"{dp_size}*{replica_size}={required_gpus} GPUs, "
                    f"but resources allocate {gpu_count} GPUs per worker"
                )

            local_gpu_count = min(gpu_count, resources.gpus_per_node)
            if replica_size > local_gpu_count:
                expansion_by_mode[mode] = 1
            else:
                expansion_by_mode[mode] = backend._get_local_dp_size(mode, local_gpu_count)

        expansions = set(expansion_by_mode.values())
        if len(expansions) > 1:
            detail = ", ".join(f"{mode}={size}" for mode, size in expansion_by_mode.items())
            raise ValueError(
                "vLLM Router has one --intra-node-data-parallel-size for all worker pools, "
                f"but the allocated topology derives different expansion factors: {detail}"
            )

        frontend_args = config.frontend.args or {}
        configured_expansion = frontend_args.get(
            "intra-node-data-parallel-size", frontend_args.get("intra_node_data_parallel_size")
        )
        derived_expansion = next(iter(expansions), 1)
        try:
            configured_expansion_value = int(configured_expansion) if configured_expansion is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"frontend.args.intra-node-data-parallel-size must be an integer; got {configured_expansion!r}"
            ) from exc
        if configured_expansion_value is not None and configured_expansion_value != derived_expansion:
            raise ValueError(
                "frontend.args.intra-node-data-parallel-size conflicts with the allocated vLLM topology: "
                f"configured {configured_expansion}, derived {derived_expansion}"
            )

        if backend.discovers_workers():
            self._validate_discovery(config)

    def _validate_discovery(self, config: Any) -> None:
        """Rules for a discovery connector (MoRI-IO): both roles on it, one router on the head node, a P/D topology.

        Workers register with the one ZMQ endpoint the router binds and are told
        the head node's address, so nginx fan-out and any other router placement
        would advertise a listener that does not exist.
        """
        backend = config.backend
        rows = {mode: backend.kv_connector_for_mode(mode) for mode in ("prefill", "decode")}
        if any(row is None or not row.discovery for row in rows.values()):
            names = ", ".join(f"{mode}={backend.connector_for_mode(mode)!r}" for mode in rows)
            raise ValueError(
                "a discovery connector must be set on both prefill and decode so the roles find each other "
                f"through the Router; got {names}"
            )
        if config.resources.num_agg:
            raise ValueError(
                "a discovery connector requires a prefill/decode topology; aggregate workers transfer no KV"
            )
        if config.frontend.enable_multiple_frontends:
            raise ValueError(
                "vLLM Router discovery uses one registration endpoint; set frontend.enable_multiple_frontends: false"
            )
        if config.frontend.orchestrator_placement != "head":
            raise ValueError(
                "vLLM Router discovery advertises the head node to every worker; "
                "set frontend.orchestrator_placement: head"
            )

    def build_router_command(self, workers: list[RouterWorker], host: str, port: int, backend: Any) -> list[str]:
        """Add the Router's discovery contract when the workers register instead of being listed.

        vllm-project/router ``RouterArgs``: ``--kv-connector`` names the transfer
        connector the P/D pair runs and ``--vllm-discovery-address`` binds the ZMQ
        endpoint workers register with; it listens on every interface of the router node.
        """
        command = super().build_router_command(workers, host, port, backend)
        if not self.discovers_workers(backend):
            return command
        insertion = command.index("--host")
        command[insertion:insertion] = [
            "--kv-connector",
            str(backend.connector_for_mode("prefill")).lower(),
            "--vllm-discovery-address",
            f"0.0.0.0:{VLLM_DISCOVERY_PORT}",
        ]
        return command

    def probe_ready(
        self, host: str, port: int, expected_prefill: int, expected_decode: int, config: Any
    ) -> WorkerHealthResult:
        """Discovery mode polls the Router's ``/health``; static mode counts its ``/workers`` registry.

        Why the two differ: in discovery mode the Router keeps registered workers
        in a separate ``ServiceRegistry`` but still serves ``/workers`` and
        ``/get_server_info`` from the static list built from ``--prefill`` and
        ``--decode`` URLs, which is empty because none were given. The only place
        the discovery registry shows is ``/health``, which answers 503 "Waiting
        for discovered workers" until one prefill and one decode have registered
        (vllm-project/router 43140bc8e2, ``VllmPDRouter::health`` through
        ``discovery_is_ready``). That is weaker than the count-based gate the
        static path gets: with several workers per role, ``/health`` turns 200
        after the first of each, and the per-worker ``/health`` gate that follows
        proves the workers are up, not that the Router knows them all. The proper
        fix is upstream, in ``src/routers/router_manager.rs``: have ``/workers``
        merge the discovery registry with the same ``worker_type`` and ``stats``
        it reports for static workers. Once that ships, delete this override and
        the count-based ``/workers`` probe covers both modes.
        """
        if self.discovers_workers(config.backend):
            return probe_http_ok(host, port, "/health", "vLLM Router reports a registered prefill and decode worker")
        return super().probe_ready(host, port, expected_prefill, expected_decode, config)

    def build_bash_preamble(self, config: Any) -> str | None:
        """Run the recipe setup script in the vLLM Router container."""
        setup_script = getattr(config, "setup_script", None)
        if not setup_script:
            return None
        script_name = shlex.quote(setup_script)
        return (
            f"setup_script={script_name} && "
            'script_path="/configs/${setup_script}" && '
            'patch_script_path="/configs/patches/${setup_script}" && '
            'echo "Running setup script: ${script_path} (fallback ${patch_script_path})" && '
            'if [ -f "${script_path}" ]; then bash "${script_path}"; '
            'elif [ -f "${patch_script_path}" ]; then bash "${patch_script_path}"; '
            'else echo "WARNING: ${script_path} or ${patch_script_path} not found"; fi'
        )

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        """Require every exact advertised base API to be accepting requests.

        Router can expand a partially ready hybrid-DP pool into the expected
        worker count before every base API has bound its port. This second gate
        closes that race before benchmark or eval traffic begins.
        """
        return [
            f"{worker.url.rstrip('/')}/health"
            for worker in self.collect_workers(backend, backend_processes, network_interface)
        ]

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
    ) -> list[str]:
        """Derive topology- and health-related Router arguments."""
        frontend_args = config.frontend.args or {}
        normalized_frontend_args = {str(key).replace("_", "-") for key in frontend_args}
        managed_args: list[str] = []

        local_dp_size = node_local_data_parallel_size(backend, backend_processes)
        configured_dp_size = frontend_args.get(
            "intra-node-data-parallel-size",
            frontend_args.get("intra_node_data_parallel_size"),
        )
        if configured_dp_size is not None and int(configured_dp_size) != local_dp_size:
            raise ValueError(
                "frontend.args.intra-node-data-parallel-size conflicts with the allocated vLLM topology: "
                f"configured {configured_dp_size}, derived {local_dp_size}"
            )
        if local_dp_size > 1 and configured_dp_size is None:
            managed_args.extend(["--intra-node-data-parallel-size", str(local_dp_size)])

        if "worker-startup-timeout-secs" not in normalized_frontend_args:
            health_check = config.health_check
            timeout_seconds = health_check.max_attempts * health_check.interval_seconds
            managed_args.extend(["--worker-startup-timeout-secs", str(timeout_seconds)])
        return managed_args

    def discovers_workers(self, backend: Any) -> bool:
        """Discovery mode when the P/D connector registers workers with the Router instead of being listed."""
        return backend.discovers_workers()

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        """Advertise vLLM's NIXL side-channel port for static P/D routing; discovered workers bring their own."""
        if self.discovers_workers(backend):
            return None
        return process.nixl_port

    def health_expectations(self, config: Any, processes: list[Process] | None) -> tuple[int, int, str]:
        """Router's /workers lists one entry per DP rank it expands each advertised URL into.

        In discovery mode readiness is the Router's ``/health``, which needs one
        registered prefill and one registered decode; the logical counts only
        describe the topology in the log.
        """
        logical_prefill, logical_decode, worker_desc = logical_health_expectations(config)
        if self.discovers_workers(config.backend):
            return logical_prefill, logical_decode, f"{worker_desc}, registering with the Router over ZMQ discovery"
        if processes is None:
            return logical_prefill, logical_decode, worker_desc
        n_prefill = sum(
            routed_process_dp_size(config.backend, process)
            for process in processes
            if process.endpoint_mode == "prefill" and process.http_port > 0
        )
        n_decode = sum(
            routed_process_dp_size(config.backend, process)
            for process in processes
            if process.endpoint_mode in {"decode", "agg"} and process.http_port > 0
        )
        return n_prefill, n_decode, f"{n_prefill}P + {n_decode}D Router workers; logical workers: {worker_desc}"

    def worker_metrics_port(self, process: Process, runtime: Any) -> int | None:
        """Every node-local hybrid-LB pool has its own API and /metrics; a positive http_port marks one."""
        return process.http_port if process.http_port > 0 else None

    def worker_endpoint_port(self, process: Process, config: Any, runtime: Any) -> int | None:
        """Router-facing pools are addressable whether or not they are the endpoint's leader rank."""
        return process.http_port if process.http_port > 0 else None
