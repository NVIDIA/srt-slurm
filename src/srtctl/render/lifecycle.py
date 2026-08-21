# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a self-contained, single-node server lifecycle script."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from srtctl.backends.sglang import SGLangProtocol
from srtctl.core.power.contract import CONTAINER_LOG_DIR
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process
from srtctl.ports import (
    DYN_SYSTEM_PORT_BASE,
    ETCD_CLIENT_PORT,
    FRONTEND_PUBLIC_PORT,
    NATS_PORT,
    SGLANG_NCCL_PORT_BASE,
)

_ARTIFACT_DIR_PLACEHOLDER = "__SRTCTL_ARTIFACT_DIR__"


@dataclass(frozen=True)
class LocalProcess:
    """One direct-host process emitted into the lifecycle script."""

    label: str
    log_name: str
    command: str
    http_port: int


@dataclass(frozen=True)
class LocalLifecycleRenderContext:
    """All values needed by the direct-host Bash template."""

    name: str
    source_dir: str
    output_base: str
    model_name: str
    frontend_type: str
    frontend_port: int
    etcd_client_port: int
    etcd_peer_port: int
    nats_port: int
    worker_processes: tuple[LocalProcess, ...]
    router_command: str
    expected_prefill: int
    expected_decode: int
    health_timeout_seconds: int
    health_interval_seconds: int
    needs_dynamo_infra: bool
    global_environment: tuple[tuple[str, str], ...]
    benchmark_environment: tuple[tuple[str, str], ...]
    benchmark_command: str
    tachometer_enabled: bool
    tachometer_binary: str | None
    tachometer_config: str | None
    ruter_enabled: bool
    sgl_router_source: str | None
    sgl_router_binary: str | None


def heredoc_marker(payload: str, *, prefix: str = "SRTCTL_RUNTIME_CONFIG") -> str:
    """Return a here-doc marker that cannot collide with *payload*."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    marker = f"{prefix}_{digest}"
    while marker in payload:
        marker = f"{marker}_END"
    return marker


def _shell_command(
    args: list[str], environment: dict[str, str] | None = None, *, executable: str = "$SRTCTL_PYTHON"
) -> str:
    """Return a shell-safe command with a script-selected executable."""
    parts = []
    for key, value in sorted((environment or {}).items()):
        if not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid environment variable name for --bash: {key!r}")
        quoted_value = shlex.quote(str(value))
        # ``ARTIFACT_DIR`` is selected by the lifecycle script at runtime.  Keep
        # all other config values shell-quoted while letting this one placeholder
        # expand in the child process that owns the frontend.
        quoted_value = quoted_value.replace(_ARTIFACT_DIR_PLACEHOLDER, '"${ARTIFACT_DIR}"')
        parts.append(f"{key}={quoted_value}")
    parts.append(executable)
    parts.extend(shlex.quote(str(arg)) for arg in args)
    return " ".join(parts)


def _cli_args(values: dict[str, Any] | None) -> list[str]:
    """Convert the normal YAML CLI mapping into deterministic arguments."""
    args: list[str] = []
    for key, value in sorted((values or {}).items()):
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            args.append(flag)
            args.extend(str(item) for item in value)
        elif isinstance(value, dict):
            args.extend((flag, json.dumps(value, separators=(",", ":"))))
        else:
            args.extend((flag, str(value)))
    return args


def _local_model_path(config: SrtConfig) -> str:
    path = os.path.expandvars(config.model.path)
    if path.startswith("hf:"):
        return path.removeprefix("hf:")
    return str(Path(path).expanduser().resolve())


def _format_environment(
    values: dict[str, str],
    *,
    node: str = "127.0.0.1",
    artifact_dir: str | None = None,
) -> dict[str, str]:
    """Apply topology placeholders and direct-lifecycle runtime paths."""

    class SafeDict(dict[str, str]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    substitutions = SafeDict(node=node, node_id="0")
    if artifact_dir is not None:
        substitutions["artifact_dir"] = artifact_dir
    return {key: str(value).format_map(substitutions) for key, value in values.items()}


def _validate_local_config(config: SrtConfig) -> None:
    resources = config.resources
    if not isinstance(config.backend, SGLangProtocol):
        raise NotImplementedError("--bash currently supports backend.type: sglang only")
    if resources.total_nodes != 1:
        raise ValueError("--bash requires a single-node resource topology")
    if config.infra.etcd_nats_dedicated_node:
        raise ValueError("--bash does not support infra.etcd_nats_dedicated_node on a single host")
    if config.frontend.type not in {"dynamo", "sglang", "sgl-router"}:
        raise ValueError("--bash currently supports frontend.type: dynamo, sglang, or sgl-router")
    if config.frontend.type == "sglang" and (config.frontend.args or {}).get("policy") == "cache-aware-zmq":
        raise ValueError("--bash uses the SGLang Model Gateway; use frontend.args.policy: cache_aware")
    if config.frontend.enable_multiple_frontends:
        raise ValueError("--bash requires frontend.enable_multiple_frontends: false")
    if config.benchmark.type != "custom" or not config.benchmark.command:
        raise ValueError("--bash requires benchmark.type: custom with benchmark.command")
    if config.telemetry.enabled:
        raise ValueError("--bash does not support DCGM power telemetry")


def _direct_port(config: SrtConfig, name: str, default: int) -> int:
    """Read an optional direct-host port override from global environment."""
    value = config.environment.get(name, str(default))
    try:
        port = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer port, got {value!r}") from error
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535, got {port}")
    return port


def _build_local_processes(
    config: SrtConfig, *, etcd_client_port: int, nats_port: int
) -> tuple[list[Process], tuple[LocalProcess, ...]]:
    """Use the normal topology allocation, constrained to a single loopback host."""
    resources = config.resources
    backend = config.backend
    assert isinstance(backend, SGLangProtocol)
    endpoints = backend.allocate_endpoints(
        num_prefill=resources.num_prefill,
        num_decode=resources.num_decode,
        num_agg=resources.num_agg,
        gpus_per_prefill=resources.gpus_per_prefill,
        gpus_per_decode=resources.gpus_per_decode,
        gpus_per_agg=resources.gpus_per_agg,
        gpus_per_node=resources.gpus_per_node,
        available_nodes=("127.0.0.1",),
        spread_workers=resources.spread_workers,
    )
    if any(endpoint.num_nodes != 1 for endpoint in endpoints):
        raise ValueError("--bash cannot place a tensor-parallel worker across multiple hosts")

    processes = backend.endpoints_to_processes(endpoints, frontend_type=config.frontend.type)
    used_gpus = {gpu for process in processes for gpu in process.gpu_indices}
    if len(used_gpus) > resources.gpus_per_node:
        raise ValueError("--bash worker GPU allocations exceed resources.gpus_per_node")

    model_path = _local_model_path(config)
    served_model_name = config.served_model_name
    rendered: list[LocalProcess] = []
    for process in processes:
        mode = process.endpoint_mode
        worker_config = backend.get_config_for_mode(mode)
        for key in ("model-path", "model_path", "served-model-name", "served_model_name"):
            worker_config.pop(key, None)
        # Match the normal Slurm worker command: SGLang otherwise probes a
        # random free TCP port for its TP rendezvous, which races when direct
        # workers start concurrently on one host.
        worker_config.pop("nccl-port", None)
        worker_config.pop("nccl_port", None)
        nccl_port = SGLANG_NCCL_PORT_BASE + process.sys_port - DYN_SYSTEM_PORT_BASE
        if nccl_port > 65_535:
            raise ValueError(f"Direct-host NCCL port exceeds range: {nccl_port}")

        module = "dynamo.sglang" if config.frontend.type == "dynamo" else "sglang.launch_server"
        args = [
            "-m",
            module,
            "--model-path",
            model_path,
            "--served-model-name",
            served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(process.http_port),
            "--nccl-port",
            str(nccl_port),
        ]
        if mode != "agg":
            args.extend(("--disaggregation-mode", mode))
            if mode == "prefill" and process.bootstrap_port is not None:
                args.extend(("--disaggregation-bootstrap-port", str(process.bootstrap_port)))

        kv_events = backend.get_kv_events_config_for_mode(mode)
        if kv_events and process.kv_events_port is not None:
            kv_events = dict(kv_events)
            kv_events["endpoint"] = f"tcp://*:{process.kv_events_port}"
            args.extend(("--kv-events-config", json.dumps(kv_events, separators=(",", ":"))))
        if config.frontend.type == "dynamo":
            args.extend(("--request-plane", config.dynamo.request_plane))
        args.extend(_cli_args(worker_config))

        environment = _format_environment(backend.get_environment_for_mode(mode))
        environment.update(config.environment)
        environment["CUDA_VISIBLE_DEVICES"] = process.cuda_visible_devices
        if config.frontend.type == "dynamo":
            environment.update(
                {
                    "DYN_SYSTEM_PORT": str(process.sys_port),
                    "DYN_REQUEST_PLANE": config.dynamo.request_plane,
                    "DYN_SKIP_SGLANG_LOG_FORMATTING": "1",
                    "ETCD_ENDPOINTS": f"http://127.0.0.1:{etcd_client_port}",
                    "NATS_SERVER": f"nats://127.0.0.1:{nats_port}",
                }
            )
            if config.dynamo.event_plane:
                environment["DYN_EVENT_PLANE"] = config.dynamo.event_plane

        log_name = (
            f"worker-{process.endpoint_index}.log" if mode == "agg" else f"worker-{mode}-{process.endpoint_index}.log"
        )
        rendered.append(
            LocalProcess(
                label=f"{mode}-{process.endpoint_index}",
                log_name=log_name,
                command=_shell_command(args, environment),
                http_port=process.http_port,
            )
        )
    return processes, tuple(rendered)


def _build_router_command(config: SrtConfig, processes: list[Process], *, etcd_client_port: int, nats_port: int) -> str:
    frontend_environment = _format_environment(dict(config.frontend.env or {}), artifact_dir=_ARTIFACT_DIR_PLACEHOLDER)
    if config.frontend.type == "dynamo":
        # ``observability.enabled`` uses the container-stable ``/logs`` path
        # for SLURM. A direct-host lifecycle has no such mount, so preserve the
        # same relative trace name below this run's artifacts instead.
        trace_path = frontend_environment.get("DYN_REQUEST_TRACE_FILE_PATH")
        container_trace_prefix = f"{CONTAINER_LOG_DIR}/"
        if trace_path and trace_path.startswith(container_trace_prefix):
            frontend_environment["DYN_REQUEST_TRACE_FILE_PATH"] = (
                f"{_ARTIFACT_DIR_PLACEHOLDER}/{trace_path.removeprefix(container_trace_prefix)}"
            )
        frontend_environment.update(
            {
                "ETCD_ENDPOINTS": f"http://127.0.0.1:{etcd_client_port}",
                "NATS_SERVER": f"nats://127.0.0.1:{nats_port}",
                "DYN_REQUEST_PLANE": config.dynamo.request_plane,
                "DYN_SKIP_SGLANG_LOG_FORMATTING": "1",
            }
        )
        if config.dynamo.event_plane:
            frontend_environment["DYN_EVENT_PLANE"] = config.dynamo.event_plane
        return _shell_command(
            ["-m", "dynamo.frontend", "--http-port", str(FRONTEND_PUBLIC_PORT), *_cli_args(config.frontend.args)],
            frontend_environment,
        )

    worker_urls = [
        f"http://127.0.0.1:{process.http_port}"
        for process in sorted(processes, key=lambda item: (item.endpoint_mode, item.endpoint_index, item.node_rank))
        if process.is_leader
    ]
    if config.frontend.type == "sgl-router":
        tokenizer_path = _local_model_path(config)
        if not config.model.path.startswith("hf:"):
            tokenizer_path = f"{tokenizer_path}/tokenizer.json"
        args = [
            "--host",
            "127.0.0.1",
            "--port",
            str(FRONTEND_PUBLIC_PORT),
            "--model-id",
            config.served_model_name,
            "--tokenizer-path",
            tokenizer_path,
            "--worker-urls",
            *worker_urls,
            *_cli_args(config.frontend.args),
        ]
        return _shell_command(args, frontend_environment, executable="$SRTCTL_SGL_ROUTER")

    args = [
        "-m",
        "sglang_router.launch_router",
        "--worker-urls",
        *worker_urls,
        "--host",
        "127.0.0.1",
        "--port",
        str(FRONTEND_PUBLIC_PORT),
        *_cli_args(config.frontend.args),
    ]
    return _shell_command(args, frontend_environment)


def _build_tachometer_config(config: SrtConfig, processes: list[Process]) -> str | None:
    tachometer = config.observability.tachometer
    if not tachometer.enabled:
        return None

    def append_endpoint(name: str, url: str, metric_filter: str, metadata: dict[str, str]) -> None:
        lines.extend(
            (
                "[[endpoints]]",
                f"name = {json.dumps(name)}",
                f"url = {json.dumps(url)}",
                f"frequency = {tachometer.default_frequency}",
                f"filter = {json.dumps(metric_filter)}",
                "[endpoints.node_metadata]",
                *(f"{json.dumps(key)} = {json.dumps(value)}" for key, value in sorted(metadata.items())),
                "",
            )
        )

    lines = [
        'storage = "${TACHOMETER_STORAGE}"',
        "rows_per_parquet = 1000000",
        "save_interval_secs = 5",
        "",
    ]
    append_endpoint(
        "router", f"http://127.0.0.1:{FRONTEND_PUBLIC_PORT}/metrics", "frontend", {"router": config.frontend.type}
    )
    for process in sorted(processes, key=lambda item: (item.endpoint_mode, item.endpoint_index, item.node_rank)):
        metrics_port = process.sys_port if config.frontend.type == "dynamo" else process.http_port
        append_endpoint(
            f"worker_{process.endpoint_mode}_{process.endpoint_index}_{process.node_rank}",
            f"http://127.0.0.1:{metrics_port}/metrics",
            "backend",
            {
                "worker_role": process.endpoint_mode,
                "worker_index": str(process.endpoint_index),
                "worker_process": str(process.node_rank),
            },
        )
    return "\n".join(lines)


def build_local_lifecycle_render_context(
    config: SrtConfig,
    *,
    source_dir: Path,
    output_base: Path,
) -> LocalLifecycleRenderContext:
    """Build the direct-host lifecycle plan for ``srtctl apply --bash``."""
    _validate_local_config(config)
    assert config.benchmark.command is not None
    etcd_client_port = _direct_port(config, "SRTCTL_ETCD_PORT", ETCD_CLIENT_PORT)
    etcd_peer_port = _direct_port(config, "SRTCTL_ETCD_PEER_PORT", etcd_client_port + 1)
    nats_port = _direct_port(config, "SRTCTL_NATS_PORT", NATS_PORT)
    processes, workers = _build_local_processes(config, etcd_client_port=etcd_client_port, nats_port=nats_port)
    tachometer_config = _build_tachometer_config(config, processes)
    resources = config.resources
    expected_prefill = resources.num_prefill
    expected_decode = resources.num_agg if resources.num_agg else resources.num_decode
    health_interval = max(1, int(config.health_check.interval_seconds))
    sgl_router_config = config.frontend.sgl_router
    return LocalLifecycleRenderContext(
        name=config.name,
        source_dir=str(source_dir.resolve()),
        output_base=str(output_base.resolve()),
        model_name=config.served_model_name,
        frontend_type=config.frontend.type,
        frontend_port=FRONTEND_PUBLIC_PORT,
        etcd_client_port=etcd_client_port,
        etcd_peer_port=etcd_peer_port,
        nats_port=nats_port,
        worker_processes=workers,
        router_command=_build_router_command(config, processes, etcd_client_port=etcd_client_port, nats_port=nats_port),
        expected_prefill=expected_prefill,
        expected_decode=expected_decode,
        health_timeout_seconds=max(1, int(config.health_check.max_attempts) * health_interval),
        health_interval_seconds=health_interval,
        needs_dynamo_infra=config.frontend.type == "dynamo",
        global_environment=tuple(sorted((key, str(value)) for key, value in config.environment.items())),
        benchmark_environment=tuple(sorted((key, str(value)) for key, value in config.benchmark.env.items())),
        benchmark_command=config.benchmark.command,
        tachometer_enabled=tachometer_config is not None,
        tachometer_binary=config.observability.tachometer.binary_path if tachometer_config is not None else None,
        tachometer_config=tachometer_config,
        ruter_enabled=config.frontend.type == "dynamo" and config.observability.enabled,
        sgl_router_source=sgl_router_config.source if sgl_router_config is not None else None,
        sgl_router_binary=sgl_router_config.binary if sgl_router_config is not None else None,
    )


def render_local_lifecycle(context: LocalLifecycleRenderContext) -> str:
    """Render the self-contained local Bash execution artifact."""
    template_dir = Path(__file__).parent.parent / "templates"
    environment = Environment(loader=FileSystemLoader(str(template_dir)), keep_trailing_newline=True)
    return environment.get_template("local_lifecycle.sh.j2").render(
        context=context,
        quote=shlex.quote,
        telemetry_marker=heredoc_marker(context.tachometer_config or "", prefix="SRTCTL_TACHOMETER"),
    )
