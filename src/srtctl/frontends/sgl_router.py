# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Rust ``sgl-router`` frontend.

The upstream router accepts a static aggregate ``--worker-urls`` pool.  Its
prefill/decode discovery API is Kubernetes-only, so this frontend deliberately
does not model srt-slurm's static disaggregated topology.
"""

import logging
import shlex
import threading
from typing import TYPE_CHECKING, Any

from srtctl.core.health import WorkerHealthResult, check_sgl_router_health
from srtctl.core.slurm import get_hostname_ip, start_srun_process

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


class SGLRouterFrontend:
    """Launch the experimental Rust router against static aggregate workers."""

    @property
    def type(self) -> str:
        return "sgl-router"

    @property
    def health_endpoint(self) -> str:
        return "/metrics"

    def parse_health(
        self,
        response_json: Any,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        """Parse the router's Prometheus worker-count metric."""
        if not isinstance(response_json, str):
            return WorkerHealthResult(
                ready=False,
                message="Expected Prometheus text from sgl-router /metrics",
                prefill_expected=expected_prefill,
                decode_expected=expected_decode,
            )
        return check_sgl_router_health(response_json, expected_prefill, expected_decode)

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert frontend arguments to CLI arguments."""
        if not args:
            return []
        result: list[str] = []
        for key, value in args.items():
            if value is True:
                result.append(f"--{key.replace('_', '-')}")
            elif value is not False and value is not None:
                result.extend([f"--{key.replace('_', '-')}", str(value)])
        return result

    @staticmethod
    def _build_preamble(source: str, configured_binary: str | None) -> str:
        """Build the router once when no executable override is supplied."""
        default_binary = configured_binary or ""
        manifest_path = f"{source.rstrip('/')}/experimental/sgl-router/Cargo.toml"
        built_binary = f"{source.rstrip('/')}/experimental/sgl-router/target/release/sgl-router"
        return "\n".join(
            (
                f"SRTCTL_SGL_ROUTER_SOURCE={shlex.quote(source)}",
                f"SRTCTL_SGL_ROUTER_BINARY=${{SRTCTL_SGL_ROUTER_BINARY:-{shlex.quote(default_binary)}}}",
                'if [[ -z "${SRTCTL_SGL_ROUTER_BINARY}" ]]; then',
                '  echo "Building experimental sgl-router from ${SRTCTL_SGL_ROUTER_SOURCE}"',
                f"  cargo build --manifest-path {shlex.quote(manifest_path)} --release",
                f"  SRTCTL_SGL_ROUTER_BINARY={shlex.quote(built_binary)}",
                "fi",
                '[[ -x "${SRTCTL_SGL_ROUTER_BINARY}" ]] || { echo "sgl-router binary is not executable: ${SRTCTL_SGL_ROUTER_BINARY}" >&2; exit 1; }',
                "export SRTCTL_SGL_ROUTER_BINARY",
            )
        )

    def start_frontends(
        self,
        topology: Any,  # FrontendTopology
        runtime: "RuntimeContext",
        config: Any,  # SrtConfig
        backend: Any,  # BackendProtocol
        backend_processes: list["Process"],
        stop_event: threading.Event | None = None,  # unused: returns immediately
    ) -> list["ManagedProcess"]:
        """Start one static-worker router per selected frontend node."""
        from srtctl.core.processes import ManagedProcess

        del backend, stop_event
        router_config = config.frontend.sgl_router
        assert router_config is not None  # validated by SrtConfig

        worker_urls = [
            f"http://{get_hostname_ip(process.node)}:{process.http_port}"
            for process in backend_processes
            if process.is_leader and process.endpoint_mode == "agg"
        ]
        if not worker_urls:
            raise ValueError("sgl-router requires at least one aggregate worker leader")

        tokenizer_path = (
            runtime.worker_model_arg if runtime.is_hf_model else f"{runtime.worker_model_arg}/tokenizer.json"
        )
        router_args = [
            "--host",
            "0.0.0.0",
            "--port",
            str(topology.frontend_port),
            "--model-id",
            config.served_model_name,
            "--tokenizer-path",
            tokenizer_path,
            "--worker-urls",
            *worker_urls,
            *self.get_frontend_args_list(config.frontend.args),
        ]
        command = [
            "bash",
            "-lc",
            'exec "$SRTCTL_SGL_ROUTER_BINARY" "$@"',
            "sgl-router",
            *router_args,
        ]
        preamble = self._build_preamble(router_config.source, router_config.binary)
        env_to_set = dict(config.frontend.env or {})

        processes: list[ManagedProcess] = []
        for index, node in enumerate(topology.frontend_nodes):
            router_log = runtime.log_dir / f"{node}_sgl_router_{index}.out"
            logger.info("Starting experimental sgl-router %d on %s: %s", index, node, shlex.join(router_args))
            proc = start_srun_process(
                command=command,
                nodelist=[node],
                output=str(router_log),
                container_image=str(runtime.container_image),
                container_mounts=runtime.container_mounts,
                env_to_set=env_to_set or None,
                bash_preamble=preamble,
                het_group=runtime.nodes.het_group_for(node),
            )
            processes.append(
                ManagedProcess(
                    name=f"sgl_router_{index}",
                    popen=proc,
                    log_file=router_log,
                    node=node,
                    critical=True,
                )
            )
        return processes
