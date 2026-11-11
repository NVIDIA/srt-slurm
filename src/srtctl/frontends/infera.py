# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infera's router in front of SGLang workers that register themselves in etcd.

Each worker is ``python3 -m infera.engine.sglang``: it starts SGLang's own
server (``sglang.launch_server``) on the allocated HTTP port with the flags the
SGLang backend builds, then registers the worker's URL, role, bootstrap address,
DP size and KV-events endpoint under an etcd prefix. ``infera-router`` (the
Rust data plane, started through ``python3 -m infera.server --router-backend
rust``) watches that prefix and routes over HTTP, reading KV events straight
from SGLang's ZMQ publishers. The worker is therefore a direct worker (the
engine's own server); the frontend is dynamic because the router learns its
workers from etcd.
"""

from __future__ import annotations

import logging
import shlex
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from srtctl.core.health import WorkerHealthResult
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.frontends.base import Frontend, numactl_prefix, register_frontend
from srtctl.frontends.dynamic_frontend import DynamicFrontend
from srtctl.frontends.static_router import setup_script_preamble
from srtctl.ports import ETCD_CLIENT_PORT
from srtctl.services.config import ServiceConfig
from srtctl.services.implicit import ETCD_SERVICE_NAME, EffectiveService, infra_placement

if TYPE_CHECKING:
    from srtctl.backends.base import WorkerMode
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

# Both sides speak HTTP and ZMQ only, so no NATS service is implied.
ROUTER_MANAGED_ARGS = frozenset(
    {
        "router-backend",
        "host",
        "port",
        "discovery-backend",
        "etcd-endpoint",
        "request-transport",
        "kv-event-transport",
    }
)
_ROLE_BY_DISAGG_MODE = {"prefill": "prefill", "decode": "decode", "mixed": "decode"}
# Infera's registering launcher for each engine this frontend pairs with. Only SGLang:
# Infera also wraps vLLM (infera.engine.vllm), but a row for it would also need the vLLM
# backend to call the worker hooks and this frontend's port answers checked against it.
WORKER_LAUNCHERS = {"sglang": "infera.engine.sglang"}
# The launcher's readiness listener port; unset, it binds a fixed default.
READINESS_PORT_ENV = "INFERA_READINESS_PORT"
# The release setup_script installs (configs/install-infera.sh); the router and the
# workers must run the same one.
VERSION_ENV = "INFERA_VERSION"
# `python3 -m infera.server` routes kv-aware unless frontend.args says otherwise.
DEFAULT_ROUTER_POLICY = "kv-aware"
# kv-aware's decode balancing by in-flight input tokens, which needs no KV events; the
# router reads this env when frontend.args leaves decode-input-demand unset.
DECODE_INPUT_DEMAND_ENV = "INFERA_DECODE_INPUT_DEMAND"


def etcd_endpoint(runtime: RuntimeContext) -> str:
    """The job's etcd on the infra node; workers and the router must name the same one."""
    return f"http://{runtime.infra_node_ip}:{ETCD_CLIENT_PORT}"


def decode_input_demand(config: Any) -> str:
    """The router's decode-input-demand mode: frontend.args, else the router's env, else off."""
    env = {**(config.environment or {}), **(config.frontend.env or {})}
    return str((config.frontend.args or {}).get("decode-input-demand", env.get(DECODE_INPUT_DEMAND_ENV, "off")))


@register_frontend("infera")
class InferaFrontend(DynamicFrontend):
    """``infera-router`` discovering ``infera.engine.sglang`` workers through etcd."""

    type: ClassVar[str] = "infera"
    required_backend: ClassVar[str | None] = "sglang"
    # The worker is SGLang's own OpenAI server; the launcher only adds registration.
    worker_launch: ClassVar[Literal["dynamo", "direct"]] = "direct"

    @property
    def health_endpoint(self) -> str:
        """The router lists every registered worker with its role and status here."""
        return "/v1/workers"

    def validate(self, config: Any) -> None:
        clashes = sorted(ROUTER_MANAGED_ARGS & set(config.frontend.args or {}))
        if clashes:
            raise ValueError(
                f"frontend.args {clashes} are set by srtctl for frontend.type: infera "
                "(etcd discovery, HTTP requests, ZMQ KV events); remove them"
            )
        for service in config.services or ():
            if service.name == ETCD_SERVICE_NAME and service.enabled and service.external:
                raise ValueError(
                    "frontend.type: infera registers workers in the job's own etcd; "
                    "an external etcd service is not supported"
                )
        policy = (config.frontend.args or {}).get("router-policy", DEFAULT_ROUTER_POLICY)
        publishing = [
            role for role, backend in config.active_role_backends() if backend.get_kv_events_config_for_mode(role)
        ]
        if policy == "kv-aware" and not publishing:
            if decode_input_demand(config) in ("on", "shadow"):
                logger.warning(
                    "frontend.type: infera: no role publishes KV events, so kv-aware picks prefill workers "
                    "on load (and session affinity) alone; decode-input-demand still balances decode"
                )
            else:
                raise ValueError(
                    "frontend.type: infera routes kv-aware (infera.server's default router-policy) on KV events, "
                    "but no role publishes them, so the router would route on load alone; set kv_events: true "
                    "on the prefill (or agg) role, set frontend.args.router-policy, or turn on "
                    "decode-input-demand if kv-aware is there for decode balancing"
                )
        versions = {
            source: env[VERSION_ENV]
            for source, env in (
                ("environment", config.environment),
                ("frontend.env", config.frontend.env),
                *((f"roles.{name}.env", role.env) for name, role in (config.roles or {}).items()),
            )
            if env and VERSION_ENV in env
        }
        if len(set(versions.values())) > 1:
            raise ValueError(
                f"the Infera router and workers must run one release, but {VERSION_ENV} differs: "
                + ", ".join(f"{source}={version}" for source, version in versions.items())
            )

    def worker_metrics_path(self, backend: Any, mode: WorkerMode) -> str | None:
        """Workers are SGLang's own servers, so metrics are the engine's, not a runtime's."""
        return Frontend.worker_metrics_path(self, backend, mode)

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> int | None:
        """The leader rank's SGLang HTTP server carries /metrics; followers serve nothing."""
        if process.is_leader and process.http_port > 0:
            return process.http_port
        return None

    def worker_endpoint_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        if process.is_leader and process.http_port > 0:
            return process.http_port
        return None

    def profiling_control_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        return process.http_port if process.http_port > 0 else None

    def worker_ready_port(self, process: Process) -> int:
        return process.http_port

    def implied_services(self, config: Any) -> list[EffectiveService]:
        """etcd for worker registration; requests and KV events do not ride on NATS."""
        return [
            EffectiveService(
                ServiceConfig(name=ETCD_SERVICE_NAME, type="etcd", placement=infra_placement(config)),
                implicit=True,
                reason="frontend.type infera",
            )
        ]

    def worker_module(self, backend_type: str) -> str:
        """Infera's launcher, which starts the engine's own server and registers it in etcd."""
        try:
            return WORKER_LAUNCHERS[backend_type]
        except KeyError:
            raise ValueError(f"frontend.type: infera has no worker launcher for engine {backend_type!r}") from None

    def worker_launcher_args(self, backend: Any, process: Process, runtime: RuntimeContext) -> list[str]:
        """Registration in the job's etcd over HTTP and ZMQ.

        Every listener the launcher opens is the allocator's. A role that publishes
        KV events gets ``--kv-events-config`` with its allocated port from the
        backend, and the launcher advertises that endpoint. Infera's own KV probe
        plane, which the router does not read, stays off.
        """
        args = [
            "--discovery-backend",
            "etcd",
            "--etcd-endpoint",
            etcd_endpoint(runtime),
            "--advertise-host",
            get_hostname_ip(process.node, runtime.network_interface),
            "--request-transport",
            "http",
            "--kv-event-transport",
            "zmq",
            "--kv-events",
            "off",
        ]
        # Without --kv-events-config the launcher publishes anyway, on a port it picks itself.
        if backend.get_kv_events_config_for_mode(process.endpoint_mode) is None:
            args.append("--no-enable-kv-events")
        return args

    def worker_environment(self, process: Process, runtime: RuntimeContext) -> dict[str, str]:
        """The readiness listener the launcher opens once registered: the process's system port.

        srtctl reads readiness from the router, and the system port is otherwise
        unused without Dynamo, so co-located workers do not contend for Infera's
        default.
        """
        return {READINESS_PORT_ENV: str(process.sys_port)}

    def parse_health(self, response_json: dict, expected_prefill: int, expected_decode: int) -> WorkerHealthResult:
        """Count active registrations by role; an aggregate (``mixed``) worker counts as decode.

        An entry without ``status`` or ``disagg_mode`` does not count, so a schema
        change in the router fails readiness instead of passing it.
        """
        counts = {"prefill": 0, "decode": 0}
        for worker in response_json.get("workers") or []:
            role = _ROLE_BY_DISAGG_MODE.get(str(worker.get("disagg_mode")))
            if role is not None and worker.get("status") == "active":
                counts[role] += 1
        ready = counts["prefill"] >= expected_prefill and counts["decode"] >= expected_decode
        return WorkerHealthResult(
            ready=ready,
            message=(
                f"infera-router: {counts['prefill']}/{expected_prefill} prefill, "
                f"{counts['decode']}/{expected_decode} decode registered"
            ),
            prefill_ready=counts["prefill"],
            prefill_expected=expected_prefill,
            decode_ready=counts["decode"],
            decode_expected=expected_decode,
        )

    def build_router_command(self, port: int, runtime: RuntimeContext, config: Any) -> list[str]:
        """Infera's launcher execs the Rust data plane after resolving a Hugging Face id to a local tokenizer.

        kv-aware routing hashes prompts with the served model's tokenizer, so it
        defaults to the checkpoint the workers load.
        """
        args = dict(config.frontend.args or {})
        args.setdefault("router-tokenizer-path", runtime.worker_model_arg)
        return [
            "python3",
            "-m",
            "infera.server",
            "--router-backend",
            "rust",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--discovery-backend",
            "etcd",
            "--etcd-endpoint",
            etcd_endpoint(runtime),
            "--request-transport",
            "http",
            "--kv-event-transport",
            "zmq",
            *self.get_frontend_args_list(args),
        ]

    def start_process(self, **kwargs: Any) -> Any:
        """Launch one router process; split out for tests."""
        return start_srun_process(**kwargs)

    def start_frontends(
        self,
        topology: Any,
        runtime: RuntimeContext,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        stop_event: threading.Event | None = None,  # unused: returns immediately
    ) -> list[ManagedProcess]:
        from srtctl.core.processes import FRONTEND_TERMINATE_TIMEOUT_SECONDS, ManagedProcess

        processes: list[ManagedProcess] = []
        for idx, node in enumerate(topology.frontend_nodes):
            log_file = runtime.log_dir / f"{node}_infera_router_{idx}.out"
            cmd = numactl_prefix(config) + self.build_router_command(topology.frontend_port, runtime, config)
            logger.info("Starting infera router %d on %s: %s", idx, node, shlex.join(cmd))

            env = dict(runtime.environment)
            env.update(config.frontend.env or {})
            step_name = f"infera_router_{idx}"
            proc = self.start_process(
                command=cmd,
                nodelist=[node],
                output=str(log_file),
                container_image=config.frontend.container_image or str(runtime.container_image),
                reuse_container=runtime.reuse_container,
                container_prepare_timeout_seconds=runtime.container_prepare_timeout_seconds,
                container_mounts=runtime.container_mounts,
                env_to_set=env or None,
                bash_preamble=setup_script_preamble(config),
                het_group=runtime.nodes.het_group_for(node),
                step_name=step_name,
                srun_options=runtime.srun_options,
            )
            processes.append(
                ManagedProcess(
                    name=step_name,
                    popen=proc,
                    log_file=log_file,
                    node=node,
                    critical=True,
                    terminate_timeout=FRONTEND_TERMINATE_TIMEOUT_SECONDS,
                    step_name=step_name,
                )
            )
        return processes
