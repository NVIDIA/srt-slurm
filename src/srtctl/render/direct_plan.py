# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile the containerized single-node Dynamo execution plan."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from srtctl.backends.sglang import SGLangProtocol
from srtctl.core.power.contract import CONTAINER_LOG_DIR
from srtctl.core.schema import SrtConfig, dynamo_cargo_patch_commands, dynamo_source_cache_key
from srtctl.core.topology import Process
from srtctl.ports import (
    DYN_SYSTEM_PORT_BASE,
    ETCD_CLIENT_PORT,
    FRONTEND_PUBLIC_PORT,
    MOONCAKE_HTTP_METADATA_PORT,
    MOONCAKE_MASTER_PORT,
    MOONCAKE_METRICS_PORT,
    NATS_PORT,
    SGLANG_NCCL_PORT_BASE,
)

_ARTIFACT_DIR_PLACEHOLDER = "__SRTCTL_ARTIFACT_DIR__"
_DIRECT_RUNTIME_ENV = frozenset({"SRTCTL_LOCAL_CONTAINER_IMAGE", "SRTCTL_SGLANG_SOURCE"})
_REMOVED_DIRECT_ENV = frozenset(
    {"SRTCTL_ETCD_BINARY", "SRTCTL_NATS_BINARY", "SRTCTL_RUTER_PYTHON", "SRTCTL_TACHOMETER"}
)


@dataclass(frozen=True)
class DirectProcess:
    """One Dynamo worker emitted into the direct execution plan."""

    label: str
    log_name: str
    command: str
    http_port: int


@dataclass(frozen=True)
class DirectPlanContext:
    """All values needed by the direct host runner and in-container plan."""

    name: str
    source_dir: str
    output_base: str
    model_name: str
    model_path: str
    local_container_image: str
    sglang_source: str
    frontend_port: int
    etcd_client_port: int
    etcd_peer_port: int
    nats_port: int
    worker_processes: tuple[DirectProcess, ...]
    router_command: str
    expected_prefill: int
    expected_decode: int
    health_timeout_seconds: int
    health_interval_seconds: int
    dynamo_source_hash: str | None
    dynamo_source_cache_key: str | None
    dynamo_cargo_patch_commands: tuple[str, ...]
    dynamo_top_of_tree: bool
    sglang_runtime_key: str
    setup_script: str | None
    mooncake_master_command: tuple[str, ...] | None
    mooncake_container: str | None
    mooncake_environment: tuple[tuple[str, str], ...]
    mooncake_master_port: int
    mooncake_metadata_port: int
    mooncake_metrics_port: int
    global_environment: tuple[tuple[str, str], ...]
    benchmark_environment: tuple[tuple[str, str], ...]
    benchmark_command: str
    tachometer_enabled: bool
    tachometer_config: str | None
    tachometer_sync_interval_secs: int
    tachometer_compaction_threads: int
    ruter_enabled: bool
    direct_plan_json: str
    direct_host_plan_json: str


def heredoc_marker(payload: str, *, prefix: str = "SRTCTL_RUNTIME_CONFIG") -> str:
    """Return a here-doc marker that cannot collide with *payload*."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    marker = f"{prefix}_{digest}"
    while marker in payload:
        marker = f"{marker}_END"
    return marker


def _shell_command(args: list[str], environment: dict[str, str] | None = None) -> str:
    """Return a shell-safe command that uses the script-selected Python."""
    parts = []
    for key, value in sorted((environment or {}).items()):
        if not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid environment variable name for --bash: {key!r}")
        quoted_value = shlex.quote(str(value))
        # ``ARTIFACT_DIR`` is selected by the direct runner at runtime.  Keep
        # all other config values shell-quoted while letting this one placeholder
        # expand in the child process that owns the frontend.
        quoted_value = quoted_value.replace(_ARTIFACT_DIR_PLACEHOLDER, '"${ARTIFACT_DIR}"')
        parts.append(f"{key}={quoted_value}")
    parts.append("$SRTCTL_PYTHON")
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


def _direct_model_path(config: SrtConfig) -> str:
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
    """Apply topology placeholders and direct-runner runtime paths."""

    class SafeDict(dict[str, str]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    substitutions = SafeDict(node=node, node_id="0")
    if artifact_dir is not None:
        substitutions["artifact_dir"] = artifact_dir
    return {key: str(value).format_map(substitutions) for key, value in values.items()}


def _validate_direct_config(config: SrtConfig) -> None:
    resources = config.resources
    if not isinstance(config.backend, SGLangProtocol):
        raise NotImplementedError("--bash currently supports backend.type: sglang only")
    if resources.total_nodes != 1:
        raise ValueError("--bash requires a single-node resource topology")
    if config.infra.etcd_nats_dedicated_node:
        raise ValueError("--bash does not support infra.etcd_nats_dedicated_node on a single host")
    if config.frontend.type != "dynamo":
        raise ValueError("--bash supports frontend.type: dynamo only")
    if config.frontend.enable_multiple_frontends:
        raise ValueError("--bash requires frontend.enable_multiple_frontends: false")
    if config.benchmark.type != "custom" or not config.benchmark.command:
        raise ValueError("--bash requires benchmark.type: custom with benchmark.command")
    if config.telemetry.enabled:
        raise ValueError("--bash does not support DCGM power telemetry")
    if config.profiling.enabled:
        raise ValueError("--bash does not support profiling; use the Slurm lifecycle")
    removed_environment = sorted(_REMOVED_DIRECT_ENV & config.environment.keys())
    if removed_environment:
        raise ValueError(f"--bash no longer supports direct runtime overrides: {', '.join(removed_environment)}")
    tachometer = config.observability.tachometer
    if tachometer.dcgm_exporter is not None or tachometer.node_exporter is not None:
        raise ValueError("--bash does not manage Tachometer exporter containers")
    if tachometer.storage_subdir != "tachometer":
        raise ValueError("--bash requires observability.tachometer.storage_subdir: tachometer")
    if tachometer.binary_path not in (None, "tachometer-scraper"):
        raise ValueError("--bash uses the bundled tachometer-scraper")
    container_image = str(config.environment.get("SRTCTL_LOCAL_CONTAINER_IMAGE", "")).strip()
    if not container_image:
        raise ValueError("--bash requires environment.SRTCTL_LOCAL_CONTAINER_IMAGE")
    sglang_source = config.environment.get("SRTCTL_SGLANG_SOURCE")
    if not sglang_source:
        raise ValueError("--bash requires environment.SRTCTL_SGLANG_SOURCE")
    if not os.path.expandvars(str(sglang_source)).startswith("/"):
        raise ValueError("SRTCTL_SGLANG_SOURCE must be an absolute path")
    if not config.dynamo.install or not (config.dynamo.hash or config.dynamo.top_of_tree):
        raise ValueError("--bash requires dynamo.hash or dynamo.top_of_tree")
    mooncake_cfg = config.backend.mooncake_kv_store
    if mooncake_cfg is not None and not mooncake_cfg.container:
        raise ValueError("--bash requires backend.mooncake_kv_store.container")


def _direct_port(config: SrtConfig, name: str, default: int) -> int:
    """Read an optional direct-run port override from global environment."""
    value = config.environment.get(name, str(default))
    try:
        port = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer port, got {value!r}") from error
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535, got {port}")
    return port


def _build_direct_processes(
    config: SrtConfig, *, etcd_client_port: int, nats_port: int
) -> tuple[list[Process], tuple[DirectProcess, ...]]:
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

    processes = backend.endpoints_to_processes(endpoints, frontend_type="dynamo")
    used_gpus = {gpu for process in processes for gpu in process.gpu_indices}
    if len(used_gpus) > resources.gpus_per_node:
        raise ValueError("--bash worker GPU allocations exceed resources.gpus_per_node")

    model_path = _direct_model_path(config)
    served_model_name = config.served_model_name
    rendered: list[DirectProcess] = []
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

        args = [
            "-m",
            "dynamo.sglang",
            "--model-path",
            model_path,
            "--served-model-name",
            served_model_name,
            "--host",
            "0.0.0.0",
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
        args.extend(("--request-plane", config.dynamo.request_plane))
        args.extend(_cli_args(worker_config))

        environment = _format_environment(backend.get_environment_for_mode(mode))
        environment.update({key: value for key, value in config.environment.items() if key not in _DIRECT_RUNTIME_ENV})
        environment["CUDA_VISIBLE_DEVICES"] = process.cuda_visible_devices
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
        if backend.mooncake_kv_store is not None:
            environment.update(backend.get_mooncake_worker_env("127.0.0.1", "127.0.0.1"))

        log_name = (
            f"worker-{process.endpoint_index}.log" if mode == "agg" else f"worker-{mode}-{process.endpoint_index}.log"
        )
        rendered.append(
            DirectProcess(
                label=f"{mode}-{process.endpoint_index}",
                log_name=log_name,
                command=_shell_command(args, environment),
                http_port=process.http_port,
            )
        )
    return processes, tuple(rendered)


def _build_router_command(config: SrtConfig, *, etcd_client_port: int, nats_port: int) -> str:
    frontend_environment = _format_environment(dict(config.frontend.env or {}), artifact_dir=_ARTIFACT_DIR_PLACEHOLDER)
    frontend_args = dict(config.frontend.args or {})
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
        ["-m", "dynamo.frontend", "--http-port", str(FRONTEND_PUBLIC_PORT), *_cli_args(frontend_args)],
        frontend_environment,
    )


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
    common_metadata = {"hostname": "127.0.0.1", "run_name": config.name, **tachometer.extra_metadata}
    append_endpoint(
        "router",
        f"http://127.0.0.1:{FRONTEND_PUBLIC_PORT}/metrics",
        "frontend",
        {"router": config.frontend.type, **common_metadata},
    )
    for process in sorted(processes, key=lambda item: (item.endpoint_mode, item.endpoint_index, item.node_rank)):
        append_endpoint(
            f"worker_{process.endpoint_mode}_{process.endpoint_index}_{process.node_rank}",
            f"http://127.0.0.1:{process.sys_port}/metrics",
            "backend",
            {
                "worker_role": process.endpoint_mode,
                "worker_index": str(process.endpoint_index),
                "worker_process": str(process.node_rank),
                **common_metadata,
            },
        )
    return "\n".join(lines)


def _build_direct_mooncake_master_command(config: SrtConfig) -> tuple[str, ...] | None:
    """Return the direct-host Mooncake master command from the shared YAML fields."""
    backend = config.backend
    assert isinstance(backend, SGLangProtocol)
    mooncake_cfg = backend.mooncake_kv_store
    if mooncake_cfg is None:
        return None
    return (
        "mooncake_master",
        f"--port={MOONCAKE_MASTER_PORT}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_port={MOONCAKE_HTTP_METADATA_PORT}",
        "--eviction_high_watermark_ratio=0.9",
        "--default_kv_lease_ttl=10000",
        "--rpc_thread_num=16",
        "--enable_metric_reporting=true",
        f"--metrics_port={MOONCAKE_METRICS_PORT}",
        *mooncake_cfg.master_extra_args,
    )


def _build_direct_plan_json(context: DirectPlanContext) -> str:
    """Serialize the direct-only execution plan consumed inside the container.

    This intentionally contains launch facts rather than the complete YAML:
    the direct runner has no srtctl imports and can therefore run in the
    serving venv created from SGLang source.
    """
    plan = {
        "name": context.name,
        "source_dir": context.source_dir,
        "output_base": context.output_base,
        "model_name": context.model_name,
        "frontend_port": context.frontend_port,
        "etcd_client_port": context.etcd_client_port,
        "etcd_peer_port": context.etcd_peer_port,
        "nats_port": context.nats_port,
        "worker_processes": [asdict(process) for process in context.worker_processes],
        "router_command": context.router_command,
        "expected_prefill": context.expected_prefill,
        "expected_decode": context.expected_decode,
        "health_timeout_seconds": context.health_timeout_seconds,
        "health_interval_seconds": context.health_interval_seconds,
        "dynamo_source_hash": context.dynamo_source_hash,
        "dynamo_source_cache_key": context.dynamo_source_cache_key,
        "dynamo_cargo_patch_commands": list(context.dynamo_cargo_patch_commands),
        "dynamo_top_of_tree": context.dynamo_top_of_tree,
        "sglang_source": context.sglang_source,
        "sglang_runtime_key": context.sglang_runtime_key,
        "setup_script": context.setup_script,
        "mooncake_master_command": list(context.mooncake_master_command or ()),
        "mooncake_master_port": context.mooncake_master_port,
        "mooncake_metadata_port": context.mooncake_metadata_port,
        "mooncake_metrics_port": context.mooncake_metrics_port,
        "global_environment": list(context.global_environment),
        "benchmark_environment": list(context.benchmark_environment),
        "benchmark_command": context.benchmark_command,
        "tachometer_enabled": context.tachometer_enabled,
        "tachometer_config": context.tachometer_config,
        "tachometer_sync_interval_secs": context.tachometer_sync_interval_secs,
        "tachometer_compaction_threads": context.tachometer_compaction_threads,
        "ruter_enabled": context.ruter_enabled,
    }
    return json.dumps(plan, sort_keys=True, separators=(",", ":"))


def _build_direct_host_plan_json(context: DirectPlanContext) -> str:
    """Serialize the host/container boundary consumed by ``direct_host_runner``.

    The host runner needs only Docker-facing facts plus the already-normalized
    container plan. Keeping this separate means the rendered Bash is a generic
    Python bootstrap rather than a second lifecycle implementation.
    """
    plan = {
        "name": context.name,
        "source_dir": context.source_dir,
        "output_base": context.output_base,
        "model_path": context.model_path,
        "local_container_image": context.local_container_image,
        "sglang_source": context.sglang_source,
        "sglang_runtime_key": context.sglang_runtime_key,
        "mooncake_master_command": list(context.mooncake_master_command or ()),
        "mooncake_container": context.mooncake_container,
        "mooncake_environment": list(context.mooncake_environment),
        "direct_plan": json.loads(context.direct_plan_json),
    }
    return json.dumps(plan, sort_keys=True, separators=(",", ":"))


def build_direct_plan_context(
    config: SrtConfig,
    *,
    source_dir: Path,
    output_base: Path,
) -> DirectPlanContext:
    """Build the direct execution plan for ``srtctl apply --bash``."""
    _validate_direct_config(config)
    assert isinstance(config.backend, SGLangProtocol)
    assert config.benchmark.command is not None
    etcd_client_port = _direct_port(config, "SRTCTL_ETCD_PORT", ETCD_CLIENT_PORT)
    etcd_peer_port = _direct_port(config, "SRTCTL_ETCD_PEER_PORT", etcd_client_port + 1)
    nats_port = _direct_port(config, "SRTCTL_NATS_PORT", NATS_PORT)
    processes, workers = _build_direct_processes(config, etcd_client_port=etcd_client_port, nats_port=nats_port)
    tachometer_config = _build_tachometer_config(config, processes)
    resources = config.resources
    expected_prefill = resources.num_prefill
    expected_decode = resources.num_agg if resources.num_agg else resources.num_decode
    health_interval = max(1, int(config.health_check.interval_seconds))
    dynamo_source_hash = config.dynamo.hash if config.dynamo.install else None
    cargo_patches = config.dynamo.cargo_patches if dynamo_source_hash else None
    dynamo_top_of_tree = config.dynamo.install and config.dynamo.top_of_tree
    model_path = _direct_model_path(config)
    local_container_image = str(config.environment["SRTCTL_LOCAL_CONTAINER_IMAGE"])
    sglang_source = os.path.expandvars(str(config.environment["SRTCTL_SGLANG_SOURCE"]))
    global_environment = {key: value for key, value in config.environment.items() if key not in _DIRECT_RUNTIME_ENV}
    runtime_identity = json.dumps(
        {
            "dynamo_source_cache_key": dynamo_source_cache_key(dynamo_source_hash, cargo_patches)
            if dynamo_source_hash
            else None,
            "dynamo_top_of_tree": dynamo_top_of_tree,
            "container_image": local_container_image,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    mooncake_cfg = config.backend.mooncake_kv_store
    context = DirectPlanContext(
        name=config.name,
        source_dir=str(source_dir.resolve()),
        output_base=str(output_base.resolve()),
        model_name=config.served_model_name,
        model_path=model_path,
        local_container_image=local_container_image,
        sglang_source=sglang_source,
        frontend_port=FRONTEND_PUBLIC_PORT,
        etcd_client_port=etcd_client_port,
        etcd_peer_port=etcd_peer_port,
        nats_port=nats_port,
        worker_processes=workers,
        router_command=_build_router_command(config, etcd_client_port=etcd_client_port, nats_port=nats_port),
        expected_prefill=expected_prefill,
        expected_decode=expected_decode,
        health_timeout_seconds=max(1, int(config.health_check.max_attempts) * health_interval),
        health_interval_seconds=health_interval,
        dynamo_source_hash=dynamo_source_hash,
        dynamo_source_cache_key=dynamo_source_cache_key(dynamo_source_hash, cargo_patches)
        if dynamo_source_hash
        else None,
        dynamo_cargo_patch_commands=dynamo_cargo_patch_commands(cargo_patches),
        dynamo_top_of_tree=dynamo_top_of_tree,
        sglang_runtime_key=hashlib.sha256(runtime_identity.encode("utf-8")).hexdigest()[:16],
        setup_script=config.setup_script,
        mooncake_master_command=_build_direct_mooncake_master_command(config),
        mooncake_container=mooncake_cfg.container if mooncake_cfg is not None else None,
        mooncake_environment=tuple(sorted(mooncake_cfg.env.items())) if mooncake_cfg is not None else (),
        mooncake_master_port=MOONCAKE_MASTER_PORT,
        mooncake_metadata_port=MOONCAKE_HTTP_METADATA_PORT,
        mooncake_metrics_port=MOONCAKE_METRICS_PORT,
        global_environment=tuple(sorted((key, str(value)) for key, value in global_environment.items())),
        benchmark_environment=tuple(sorted((key, str(value)) for key, value in config.benchmark.env.items())),
        benchmark_command=config.benchmark.command,
        tachometer_enabled=tachometer_config is not None,
        tachometer_config=tachometer_config,
        tachometer_sync_interval_secs=config.observability.tachometer.sync_interval_secs,
        tachometer_compaction_threads=config.observability.tachometer.compaction_threads,
        ruter_enabled=config.observability.enabled,
        direct_plan_json="",
        direct_host_plan_json="",
    )
    context = replace(context, direct_plan_json=_build_direct_plan_json(context))
    return replace(context, direct_host_plan_json=_build_direct_host_plan_json(context))


def render_direct_container_shim(context: DirectPlanContext) -> str:
    """Render the self-contained direct-container execution artifact."""
    template_dir = Path(__file__).parent.parent / "templates"
    environment = Environment(loader=FileSystemLoader(str(template_dir)), keep_trailing_newline=True)
    return environment.get_template("direct_container.sh.j2").render(
        context=context,
        quote=shlex.quote,
        direct_host_plan_marker=heredoc_marker(context.direct_host_plan_json, prefix="SRTCTL_DIRECT_HOST_PLAN"),
    )
