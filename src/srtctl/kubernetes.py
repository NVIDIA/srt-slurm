# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render and manage DynamoGraphDeployments from srtctl configurations."""

from __future__ import annotations

import copy
import json
import math
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from uuid import uuid4

import yaml

from srtctl.backends import MockerProtocol, SGLangProtocol, TRTLLMProtocol, VLLMProtocol
from srtctl.benchmarks import get_runner
from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner
from srtctl.core.runtime import RuntimeContext
from srtctl.core.schema import SrtConfig, TelemetryProvider, build_otel_env

_DGD_API_VERSION = "nvidia.com/v1beta1"
_DGD_KIND = "DynamoGraphDeployment"
_DYNAMO_FRONTEND_PORT = 8000
_DYNAMO_SYSTEM_PORT = 9090
_DEFAULT_BOOTSTRAP_PORT = 12345
_DEFAULT_SGLANG_KV_EVENTS_PORT = 5557
_DEFAULT_VLLM_KV_EVENTS_PORT = 20080
_DNS_LABEL = re.compile(r"[^a-z0-9-]+")
_SEMVER = re.compile(r"^(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})$")
_TERMINAL_WAITING_REASONS = {
    "CreateContainerConfigError",
    "CreateContainerError",
    "ErrImagePull",
    "ImagePullBackOff",
    "InvalidImageName",
    "RunContainerError",
}
_BENCHMARK_MOUNT_PATH = "/srtctl-benchmarks"
_BENCHMARK_OUTPUT_PATH = "/logs"
_MAX_CONFIG_MAP_BYTES = 900 * 1024
_MAX_CAPTURED_LOG_BYTES = 8 * 1024 * 1024

WorkerMode = Literal["prefill", "decode", "agg"]


def _safe_name(value: str, *, max_length: int = 63) -> str:
    normalized = _DNS_LABEL.sub("-", value.lower()).strip("-") or "deployment"
    return normalized[:max_length].rstrip("-")


def _derived_name(name: str, suffix: str) -> str:
    return f"{_safe_name(name, max_length=63 - len(suffix) - 1)}-{suffix}"


def _string_env(values: dict[str, Any]) -> list[dict[str, str]]:
    return [{"name": key, "value": str(value)} for key, value in sorted(values.items())]


def _env_from(config: SrtConfig) -> list[dict[str, dict[str, str]]]:
    result: list[dict[str, dict[str, str]]] = []
    result.extend({"secretRef": {"name": name}} for name in config.kubernetes.env_from_secrets)
    result.extend({"configMapRef": {"name": name}} for name in config.kubernetes.env_from_config_maps)
    return result


def _config_to_cli_args(values: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in sorted(values.items()):
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            args.append(flag)
            args.extend(str(item) for item in value)
        elif value is not None:
            args.extend([flag, str(value)])
    return args


def _pop_cli_keys(values: dict[str, Any], *keys: str) -> None:
    normalized = {key.replace("_", "-") for key in keys}
    for key in list(values):
        if key.replace("_", "-") in normalized:
            values.pop(key)


def _has_cli_key(values: dict[str, Any], *keys: str) -> bool:
    normalized = {key.replace("_", "-") for key in keys}
    return any(key.replace("_", "-") in normalized for key in values)


def _model_arg(config: SrtConfig) -> str:
    if config.model.path.startswith("hf:"):
        return config.model.path.removeprefix("hf:")
    return config.model.path


def _runtime_version(config: SrtConfig) -> str | None:
    explicit = config.kubernetes.runtime_version
    if explicit is not None:
        if not _SEMVER.fullmatch(explicit):
            raise ValueError("kubernetes.runtime_version must use MAJOR.MINOR.PATCH")
        return explicit
    declared = config.identity.frameworks.get("dynamo")
    if declared and _SEMVER.fullmatch(declared):
        return declared
    return None


def _validate_runtime_image(config: SrtConfig, runtime_version: str | None) -> None:
    image = config.model.container
    if image.startswith(("/", "docker://", "file://")) or image.endswith(".sqsh"):
        raise ValueError("model.container must be an OCI image reference for Kubernetes, not a SLURM container path")
    image_without_digest = image.rsplit("@", 1)[0]
    last_segment = image_without_digest.rsplit("/", 1)[-1]
    tag = last_segment.rsplit(":", 1)[1] if ":" in last_segment else None
    if runtime_version is None and (tag is None or _SEMVER.fullmatch(tag) is None):
        raise ValueError(
            "kubernetes.runtime_version is required when model.container does not have a MAJOR.MINOR.PATCH tag"
        )


def _component_shape(config: SrtConfig, mode: WorkerMode) -> tuple[int, int, int]:
    resources = config.resources
    if mode == "prefill":
        replicas = resources.num_prefill
        total_gpus = resources.gpus_per_prefill
        nodes = resources.prefill_nodes or 0
    elif mode == "decode":
        replicas = resources.num_decode
        total_gpus = resources.gpus_per_decode
        nodes = resources.decode_nodes or 0
    else:
        replicas = resources.num_agg
        total_gpus = resources.gpus_per_agg
        nodes = resources.agg_nodes or 0

    if replicas < 1:
        raise ValueError(f"resources must define at least one {mode} worker")

    node_count = 1
    if nodes > replicas and nodes % replicas != 0:
        raise ValueError(
            f"resources.{mode}_nodes={nodes} cannot be divided evenly across {replicas} Kubernetes replicas"
        )
    if nodes >= replicas and nodes % replicas == 0:
        node_count = nodes // replicas
    node_count = max(node_count, math.ceil(total_gpus / resources.gpus_per_node))
    if total_gpus % node_count != 0:
        raise ValueError(f"{mode} worker GPU count {total_gpus} is not divisible across {node_count} Kubernetes nodes")
    return replicas, total_gpus // node_count, node_count


def _mode_environment(config: SrtConfig, mode: WorkerMode, component: str) -> dict[str, str]:
    environment = config.backend.get_environment_for_mode(mode)
    if isinstance(config.backend, TRTLLMProtocol):
        environment["TRTLLM_EPLB_SHM_NAME"] = f"{_safe_name(config.kubernetes.name or config.name)}-{mode}"
    environment.update(build_otel_env(config.observability, component))
    return environment


def _sglang_command(config: SrtConfig, mode: WorkerMode, total_gpus: int) -> tuple[list[str], list[str]]:
    backend = config.backend
    assert isinstance(backend, SGLangProtocol)
    mode_config = backend.get_config_for_mode(mode)
    _pop_cli_keys(
        mode_config,
        "model-path",
        "served-model-name",
        "host",
        "disaggregation-mode",
        "disaggregation-bootstrap-port",
    )
    args = [
        "--model-path",
        _model_arg(config),
        "--served-model-name",
        config.served_model_name,
        "--host",
        "0.0.0.0",
    ]
    if not _has_cli_key(mode_config, "tp", "tp-size", "tensor-parallel-size"):
        args.extend(["--tp-size", str(total_gpus)])
    if mode != "agg":
        args.extend(
            [
                "--disaggregation-mode",
                mode,
                "--disaggregation-bootstrap-port",
                str(_DEFAULT_BOOTSTRAP_PORT),
            ]
        )
    kv_config = backend.get_kv_events_config_for_mode(mode)
    if kv_config is not None:
        kv_config["endpoint"] = f"tcp://*:{_DEFAULT_SGLANG_KV_EVENTS_PORT}"
        args.extend(["--kv-events-config", json.dumps(kv_config, separators=(",", ":"))])
    args.extend(_config_to_cli_args(mode_config))
    return ["python3", "-m", "dynamo.sglang"], args


def _vllm_command(config: SrtConfig, mode: WorkerMode, total_gpus: int) -> tuple[list[str], list[str]]:
    from srtctl.backends.vllm import _connector_to_kv_transfer_config

    backend = config.backend
    assert isinstance(backend, VLLMProtocol)
    mode_config = backend.get_config_for_mode(mode)
    _pop_cli_keys(mode_config, "model", "served-model-name", "disaggregation-mode")
    args = ["--model", _model_arg(config), "--served-model-name", config.served_model_name]
    if mode != "agg":
        args.extend(["--disaggregation-mode", mode])
    connector = mode_config.pop("connector", None)
    if connector is None:
        connector = backend.connector
    if connector and str(connector).lower() not in {"none", "null"}:
        args.extend(["--kv-transfer-config", _connector_to_kv_transfer_config(str(connector))])
    if not _has_cli_key(mode_config, "tensor-parallel-size", "tp", "tp-size", "data-parallel-size"):
        args.extend(["--tensor-parallel-size", str(total_gpus)])
    kv_config = backend.get_kv_events_config_for_mode(mode)
    if kv_config is not None:
        kv_config["endpoint"] = f"tcp://*:{_DEFAULT_VLLM_KV_EVENTS_PORT}"
        args.extend(["--kv-events-config", json.dumps(kv_config, separators=(",", ":"))])
    args.extend(_config_to_cli_args(mode_config))
    return ["python3", "-m", "dynamo.vllm"], args


def _mocker_command(config: SrtConfig, mode: WorkerMode) -> tuple[list[str], list[str]]:
    backend = config.backend
    assert isinstance(backend, MockerProtocol)
    args = ["--model-path", _model_arg(config), "--model-name", config.served_model_name]
    if mode != "agg":
        args.extend(["--disaggregation-mode", mode])
    args.extend(
        [
            "--engine-type",
            backend.engine_type,
            "--speedup-ratio",
            str(backend.speedup_ratio),
            "--data-parallel-size",
            str(backend.data_parallel_size),
            "--num-gpu-blocks-override",
            str(backend.num_gpu_blocks_override),
            "--max-num-seqs",
            str(backend.max_num_seqs),
            "--max-num-batched-tokens",
            str(backend.max_num_batched_tokens),
        ]
    )
    if backend.decode_speedup_ratio != 1.0:
        args.extend(["--decode-speedup-ratio", str(backend.decode_speedup_ratio)])
    if backend.block_size is not None:
        args.extend(["--block-size", str(backend.block_size)])
    if backend.num_workers > 1:
        args.extend(["--num-workers", str(backend.num_workers)])
    if backend.startup_time is not None:
        args.extend(["--startup-time", str(backend.startup_time)])
    if backend.kv_transfer_bandwidth is not None:
        args.extend(["--kv-transfer-bandwidth", str(backend.kv_transfer_bandwidth)])
    if backend.kv_cache_dtype is not None:
        args.extend(["--kv-cache-dtype", backend.kv_cache_dtype])
    if not backend.enable_prefix_caching:
        args.append("--no-enable-prefix-caching")
    if not backend.enable_chunked_prefill:
        args.append("--no-enable-chunked-prefill")
    if backend.preemption_mode is not None:
        args.extend(["--preemption-mode", backend.preemption_mode])
    args.extend(_config_to_cli_args(backend.get_config_for_mode(mode)))
    return ["python3", "-m", "dynamo.mocker"], args


def _trtllm_command(config: SrtConfig, mode: WorkerMode) -> tuple[list[str], list[str]]:
    backend = config.backend
    assert isinstance(backend, TRTLLMProtocol)
    args = [
        "--model-path",
        _model_arg(config),
        "--served-model-name",
        config.served_model_name,
    ]
    if mode != "agg":
        args.extend(["--disaggregation-mode", mode])
    args.extend(["--extra-engine-args", f"/etc/srtctl/trtllm-{mode}.yaml"])
    args.extend(["--request-plane", config.dynamo.request_plane])
    if backend.publish_events_and_metrics:
        args.append("--publish-events-and-metrics")
    return ["python3", "-m", "dynamo.trtllm"], args


def _worker_command(config: SrtConfig, mode: WorkerMode, total_gpus: int) -> tuple[list[str], list[str]]:
    if isinstance(config.backend, SGLangProtocol):
        return _sglang_command(config, mode, total_gpus)
    if isinstance(config.backend, VLLMProtocol):
        return _vllm_command(config, mode, total_gpus)
    if isinstance(config.backend, MockerProtocol):
        return _mocker_command(config, mode)
    if isinstance(config.backend, TRTLLMProtocol):
        return _trtllm_command(config, mode)
    raise ValueError(f"Kubernetes rendering does not support backend.type={config.backend_type!r}")


def _container_resources(config: SrtConfig, component: str, gpu_count: int) -> dict[str, dict[str, str]]:
    configured = copy.deepcopy(config.kubernetes.component_resources.get(component, {}))
    if component == "worker" and not configured:
        configured = copy.deepcopy(config.kubernetes.component_resources.get("aggregated", {}))
    requests = configured.setdefault("requests", {})
    limits = configured.setdefault("limits", {})
    if gpu_count > 0:
        limits.setdefault("nvidia.com/gpu", str(gpu_count))
        requests.setdefault("nvidia.com/gpu", str(gpu_count))
    return {key: value for key, value in configured.items() if value}


def _exporter_command(command: str | None, default: str, port: int) -> list[str]:
    value = command or default
    return shlex.split(value.replace("{port}", str(port)))


def _toml_metadata(values: dict[str, str]) -> str:
    return "\n".join(f"{json.dumps(key)} = {json.dumps(value)}" for key, value in sorted(values.items()))


def _telemetry_storage_volume(config: SrtConfig) -> dict[str, Any]:
    claim = config.kubernetes.telemetry_persistent_volume_claim
    if claim:
        return {"name": "srt-telemetry", "persistentVolumeClaim": {"claimName": claim}}
    return {"name": "srt-telemetry", "emptyDir": {}}


def _tachometer_command(config: SrtConfig, script: str) -> dict[str, Any]:
    telemetry = config.telemetry
    if telemetry.container_image is None:
        raise ValueError("Kubernetes telemetry requires telemetry.container_image")
    return {
        "name": "srt-tachometer",
        "image": telemetry.container_image,
        "imagePullPolicy": config.kubernetes.image_pull_policy,
        "command": ["/bin/bash", "-ceu"],
        "args": [script],
        "env": [
            {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
            {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
            {"name": "POLARS_MAX_THREADS", "value": str(telemetry.compaction_threads)},
        ],
        "volumeMounts": [{"name": "srt-telemetry", "mountPath": config.kubernetes.telemetry_mount_path}],
    }


def _component_tachometer_script(config: SrtConfig, component: str, main_port: int) -> str:
    telemetry = config.telemetry
    storage = f"{config.kubernetes.telemetry_mount_path}/{telemetry.storage_subdir}"
    if re.fullmatch(r"/[A-Za-z0-9._/-]+", storage) is None:
        raise ValueError("Kubernetes telemetry storage path contains unsupported shell characters")
    metadata = {
        "component": component,
        "deployment": _safe_name(config.kubernetes.name or config.name),
        **telemetry.extra_metadata,
    }
    static_metadata = _toml_metadata(metadata)
    return f'''cat > /tmp/tachometer-config.toml <<'EOF'
storage = "{storage}/__POD_NAME__"

[[endpoints]]
name = "{component}"
url = "http://127.0.0.1:{main_port}/metrics"
frequency = {telemetry.default_frequency}
filter = "{"frontend" if component == "frontend" else "backend"}"
[endpoints.node_metadata]
"pod" = "__POD_NAME__"
"node" = "__NODE_NAME__"
{static_metadata}
EOF
sed -i "s/__POD_NAME__/$POD_NAME/g; s/__NODE_NAME__/$NODE_NAME/g" /tmp/tachometer-config.toml
mkdir -p {shlex.quote(storage)}/"$POD_NAME" /tmp/tachometer-local
exec {shlex.quote(telemetry.binary_path)} --config /tmp/tachometer-config.toml --local-dir /tmp/tachometer-local{f" --sync-interval {telemetry.sync_interval_secs}" if telemetry.sync_interval_secs > 0 else ""}
'''


def _telemetry_pod_items(
    config: SrtConfig,
    *,
    component: str,
    main_port: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    telemetry = config.telemetry
    if not telemetry.enabled:
        return [], []
    if telemetry.provider != TelemetryProvider.SCRAPER:
        raise ValueError("Kubernetes rendering supports telemetry.provider: scraper")
    sidecar = _tachometer_command(config, _component_tachometer_script(config, component, main_port))
    return [sidecar], [_telemetry_storage_volume(config)]


def _node_tachometer_script(config: SrtConfig) -> str:
    telemetry = config.telemetry
    storage = f"{config.kubernetes.telemetry_mount_path}/{telemetry.storage_subdir}"
    if re.fullmatch(r"/[A-Za-z0-9._/-]+", storage) is None:
        raise ValueError("Kubernetes telemetry storage path contains unsupported shell characters")
    dcgm = telemetry.dcgm_exporter
    node = telemetry.node_exporter
    if dcgm is None or node is None:
        raise ValueError("Kubernetes telemetry requires telemetry.dcgm_exporter and telemetry.node_exporter")
    metadata = {
        "component": "node",
        "deployment": _safe_name(config.kubernetes.name or config.name),
        **telemetry.extra_metadata,
    }
    static_metadata = _toml_metadata(metadata)
    return f'''cat > /tmp/tachometer-config.toml <<'EOF'
storage = "{storage}/node-__NODE_NAME__"

[[endpoints]]
name = "dcgm"
url = "http://127.0.0.1:{dcgm.port}/metrics"
frequency = {telemetry.default_frequency}
filter = "dcgm"
[endpoints.node_metadata]
"pod" = "__POD_NAME__"
"node" = "__NODE_NAME__"
{static_metadata}

[[endpoints]]
name = "node-exporter"
url = "http://127.0.0.1:{node.port}/metrics"
frequency = {telemetry.default_frequency}
filter = "node_exporter"
[endpoints.node_metadata]
"pod" = "__POD_NAME__"
"node" = "__NODE_NAME__"
{static_metadata}
EOF
sed -i "s/__POD_NAME__/$POD_NAME/g; s/__NODE_NAME__/$NODE_NAME/g" /tmp/tachometer-config.toml
mkdir -p {shlex.quote(storage)}/"node-$NODE_NAME" /tmp/tachometer-local
exec {shlex.quote(telemetry.binary_path)} --config /tmp/tachometer-config.toml --local-dir /tmp/tachometer-local{f" --sync-interval {telemetry.sync_interval_secs}" if telemetry.sync_interval_secs > 0 else ""}
'''


def _telemetry_daemon_set(config: SrtConfig, name: str) -> dict[str, Any] | None:
    telemetry = config.telemetry
    if not telemetry.enabled:
        return None
    if telemetry.provider != TelemetryProvider.SCRAPER:
        raise ValueError("Kubernetes rendering supports telemetry.provider: scraper")
    dcgm = telemetry.dcgm_exporter
    node = telemetry.node_exporter
    if dcgm is None or node is None:
        raise ValueError("Kubernetes telemetry requires telemetry.dcgm_exporter and telemetry.node_exporter")

    labels = {
        "app.kubernetes.io/managed-by": "srtctl",
        "app.kubernetes.io/instance": name,
        "app.kubernetes.io/component": "telemetry",
    }
    pull_policy = config.kubernetes.image_pull_policy
    dcgm_container = {
        "name": "srt-dcgm-exporter",
        "image": dcgm.container_image,
        "imagePullPolicy": pull_policy,
        "command": _exporter_command(
            dcgm.command,
            "dcgm-exporter --collect-interval=100 --address :{port}",
            dcgm.port,
        ),
        "env": [
            {"name": "DCGM_EXPORTER_KUBERNETES", "value": "true"},
            {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
            {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
            {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        ],
        "ports": [{"name": "dcgm-metrics", "containerPort": dcgm.port}],
        "securityContext": {"capabilities": {"add": ["SYS_ADMIN"]}, "runAsNonRoot": False, "runAsUser": 0},
        "volumeMounts": [
            {"name": "srt-pod-resources", "mountPath": "/var/lib/kubelet/pod-resources", "readOnly": True}
        ],
    }
    node_container = {
        "name": "srt-node-exporter",
        "image": node.container_image,
        "imagePullPolicy": pull_policy,
        "command": _exporter_command(
            node.command,
            (
                "/bin/node_exporter --web.listen-address=:{port} --path.rootfs=/host "
                "--collector.disable-defaults --collector.cpu --collector.infiniband --collector.meminfo"
            ),
            node.port,
        ),
        "ports": [{"name": "node-metrics", "containerPort": node.port}],
        "volumeMounts": [{"name": "srt-host-root", "mountPath": "/host", "readOnly": True}],
    }
    tachometer_container = _tachometer_command(config, _node_tachometer_script(config))
    pod_spec: dict[str, Any] = {
        "containers": [dcgm_container, node_container, tachometer_container],
        "nodeSelector": dict(config.kubernetes.node_selector),
        "tolerations": copy.deepcopy(config.kubernetes.tolerations),
        "volumes": [
            _telemetry_storage_volume(config),
            {"name": "srt-host-root", "hostPath": {"path": "/", "type": "Directory"}},
            {
                "name": "srt-pod-resources",
                "hostPath": {"path": "/var/lib/kubelet/pod-resources", "type": "Directory"},
            },
        ],
    }
    if config.kubernetes.service_account_name:
        pod_spec["serviceAccountName"] = config.kubernetes.service_account_name
    if config.kubernetes.image_pull_secrets:
        pod_spec["imagePullSecrets"] = [{"name": value} for value in config.kubernetes.image_pull_secrets]
    return {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {
            "name": _derived_name(name, "telemetry"),
            "namespace": config.kubernetes.namespace,
            "labels": dict(labels),
        },
        "spec": {
            "selector": {"matchLabels": dict(labels)},
            "template": {"metadata": {"labels": dict(labels)}, "spec": pod_spec},
        },
    }


def _merge_volumes(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for group in groups:
        for volume in group:
            name = str(volume.get("name", ""))
            if not name:
                raise ValueError("Kubernetes volumes must have a name")
            if name in names:
                raise ValueError(f"duplicate Kubernetes volume name: {name}")
            names.add(name)
            result.append(copy.deepcopy(volume))
    return result


def _pod_spec(
    config: SrtConfig, main: dict[str, Any], sidecars: list[dict[str, Any]], volumes: list[dict[str, Any]]
) -> dict[str, Any]:
    kubernetes = config.kubernetes
    spec: dict[str, Any] = {
        "containers": [main, *sidecars],
        "nodeSelector": dict(kubernetes.node_selector),
        "tolerations": copy.deepcopy(kubernetes.tolerations),
    }
    if kubernetes.service_account_name:
        spec["serviceAccountName"] = kubernetes.service_account_name
    if kubernetes.image_pull_secrets:
        spec["imagePullSecrets"] = [{"name": name} for name in kubernetes.image_pull_secrets]
    merged_volumes = _merge_volumes(kubernetes.volumes, volumes)
    if merged_volumes:
        spec["volumes"] = merged_volumes
    return spec


def _main_container(
    config: SrtConfig,
    *,
    command: list[str],
    args: list[str],
    environment: dict[str, str],
    resources: dict[str, dict[str, str]],
    extra_volume_mounts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    container: dict[str, Any] = {
        "name": "main",
        "image": config.model.container,
        "imagePullPolicy": config.kubernetes.image_pull_policy,
        "command": command,
        "args": args,
    }
    env = _string_env(environment)
    if env:
        container["env"] = env
    env_from = _env_from(config)
    if env_from:
        container["envFrom"] = env_from
    if resources:
        container["resources"] = resources
    volume_mounts = [*copy.deepcopy(config.kubernetes.volume_mounts), *(extra_volume_mounts or [])]
    if volume_mounts:
        container["volumeMounts"] = volume_mounts
    if config.kubernetes.working_dir:
        container["workingDir"] = config.kubernetes.working_dir
    return container


def _component(
    config: SrtConfig,
    *,
    name: str,
    component_type: str,
    mode: WorkerMode | None,
    runtime_version: str | None,
    trtllm_config_map: str | None,
) -> dict[str, Any]:
    if mode is None:
        replicas = 1 + config.frontend.num_additional_frontends if config.frontend.enable_multiple_frontends else 1
        command = ["python3", "-m", "dynamo.frontend"]
        args = _config_to_cli_args(config.frontend.args or {})
        environment = {
            **(config.frontend.env or {}),
            **build_otel_env(config.observability, "frontend"),
        }
        resources = _container_resources(config, "frontend", 0)
        node_count = 1
        gpu_count = 0
        main_port = _DYNAMO_FRONTEND_PORT
    else:
        replicas, gpu_count, node_count = _component_shape(config, mode)
        total_gpus = gpu_count * node_count
        command, args = _worker_command(config, mode, total_gpus)
        environment = _mode_environment(config, mode, name)
        resources = _container_resources(config, name, 0 if isinstance(config.backend, MockerProtocol) else gpu_count)
        main_port = _DYNAMO_SYSTEM_PORT

    extra_mounts: list[dict[str, Any]] = []
    extra_volumes: list[dict[str, Any]] = []
    if trtllm_config_map is not None and mode is not None:
        extra_mounts.append({"name": "srt-trtllm-config", "mountPath": "/etc/srtctl", "readOnly": True})
        extra_volumes.append({"name": "srt-trtllm-config", "configMap": {"name": trtllm_config_map}})

    sidecars, telemetry_volumes = _telemetry_pod_items(config, component=name, main_port=main_port)
    main = _main_container(
        config,
        command=command,
        args=args,
        environment=environment,
        resources=resources,
        extra_volume_mounts=extra_mounts,
    )
    component: dict[str, Any] = {
        "name": name,
        "type": component_type,
        "replicas": replicas,
        "podTemplate": {
            "metadata": {"labels": {"srtctl.nvidia.com/component": name}},
            "spec": _pod_spec(config, main, sidecars, [*extra_volumes, *telemetry_volumes]),
        },
    }
    if mode is not None and config.resources.spread_workers:
        component["podTemplate"]["spec"]["affinity"] = {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {"matchLabels": {"srtctl.nvidia.com/component": name}},
                        "topologyKey": "kubernetes.io/hostname",
                    }
                ]
            }
        }
    if runtime_version is not None:
        component["runtimeVersionOverride"] = runtime_version
    if mode is not None and node_count > 1:
        component["multinode"] = {"nodeCount": node_count}
    return component


def _trtllm_config_map(config: SrtConfig, name: str) -> dict[str, Any] | None:
    if not isinstance(config.backend, TRTLLMProtocol):
        return None
    data = {
        f"trtllm-{mode}.yaml": yaml.safe_dump(config.backend.get_config_for_mode(mode), sort_keys=False)
        for mode in ("prefill", "decode", "agg")
    }
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": _derived_name(name, "trtllm"), "namespace": config.kubernetes.namespace},
        "data": data,
    }


def render_dgd(config: SrtConfig) -> dict[str, Any]:
    """Render one v1beta1 DynamoGraphDeployment."""
    if config.frontend.type != "dynamo":
        raise ValueError("Kubernetes DGD rendering requires frontend.type: dynamo")
    kubernetes = config.kubernetes
    name = _safe_name(kubernetes.name or config.name)
    if not name:
        raise ValueError("Kubernetes deployment name is empty")
    runtime_version = _runtime_version(config)
    _validate_runtime_image(config, runtime_version)
    config_map_name = _derived_name(name, "trtllm") if isinstance(config.backend, TRTLLMProtocol) else None

    components = [
        _component(
            config,
            name="frontend",
            component_type="frontend",
            mode=None,
            runtime_version=runtime_version,
            trtllm_config_map=None,
        )
    ]
    if config.resources.is_disaggregated:
        for mode in ("decode", "prefill"):
            components.append(
                _component(
                    config,
                    name=mode,
                    component_type=mode,
                    mode=mode,
                    runtime_version=runtime_version,
                    trtllm_config_map=config_map_name,
                )
            )
    else:
        components.append(
            _component(
                config,
                name="worker",
                component_type="worker",
                mode="agg",
                runtime_version=runtime_version,
                trtllm_config_map=config_map_name,
            )
        )

    labels = {
        "app.kubernetes.io/managed-by": "srtctl",
        "app.kubernetes.io/instance": name,
        **kubernetes.labels,
    }
    annotations = {
        "nvidia.com/dynamo-discovery-backend": "kubernetes",
        **kubernetes.annotations,
    }
    global_environment = {
        "DYN_DISCOVERY_BACKEND": "kubernetes",
        "DYN_REQUEST_PLANE": config.dynamo.request_plane,
        **config.environment,
    }
    if config.dynamo.event_plane:
        global_environment["DYN_EVENT_PLANE"] = config.dynamo.event_plane
    spec: dict[str, Any] = {
        "labels": dict(labels),
        "annotations": dict(annotations),
        "env": _string_env(global_environment),
        "components": components,
    }
    if not isinstance(config.backend, MockerProtocol):
        spec["backendFramework"] = config.backend_type
    if kubernetes.priority_class_name:
        spec["priorityClassName"] = kubernetes.priority_class_name
    return {
        "apiVersion": _DGD_API_VERSION,
        "kind": _DGD_KIND,
        "metadata": {
            "name": name,
            "namespace": kubernetes.namespace,
            "labels": dict(labels),
            "annotations": dict(annotations),
        },
        "spec": spec,
    }


def render_kubernetes_manifests(config: SrtConfig) -> list[dict[str, Any]]:
    """Render supporting resources followed by the DGD."""
    name = _safe_name(config.kubernetes.name or config.name)
    manifests: list[dict[str, Any]] = []
    trtllm_config_map = _trtllm_config_map(config, name)
    if trtllm_config_map is not None:
        manifests.append(trtllm_config_map)
    telemetry_daemon_set = _telemetry_daemon_set(config, name)
    if telemetry_daemon_set is not None:
        manifests.append(telemetry_daemon_set)
    manifests.append(render_dgd(config))
    return manifests


def dump_kubernetes_yaml(config: SrtConfig) -> str:
    return yaml.safe_dump_all(render_kubernetes_manifests(config), sort_keys=False, explicit_start=True)


def _benchmark_runtime(config: SrtConfig) -> RuntimeContext:
    """Build the narrow runtime view consumed by benchmark runners."""
    return cast(
        RuntimeContext,
        SimpleNamespace(
            frontend_port=_DYNAMO_FRONTEND_PORT,
            model_path=Path(_model_arg(config)),
            is_hf_model=config.model.path.startswith("hf:"),
            container_image=Path(config.model.container),
            container_mounts={},
        ),
    )


def _benchmark_command(config: SrtConfig, runner: BenchmarkRunner, frontend_host: str) -> list[str]:
    command = runner.build_command(config, _benchmark_runtime(config))
    local_url = f"http://localhost:{_DYNAMO_FRONTEND_PORT}"
    service_url = f"http://{frontend_host}:{_DYNAMO_FRONTEND_PORT}"
    return [
        service_url if value == local_url else f"http://{frontend_host}" if value == "http://localhost" else value
        for value in command
    ]


def _benchmark_script_config_map(
    config: SrtConfig,
    runner: BenchmarkRunner,
    name: str,
    labels: dict[str, str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if config.benchmark.type == "custom":
        return None, []
    local_script_dir = getattr(runner, "local_script_dir", None)
    if not local_script_dir:
        raise ValueError(f"benchmark.type={config.benchmark.type!r} does not expose packaged scripts")

    selected_root = Path(local_script_dir).resolve()
    scripts_root = SCRIPTS_DIR.resolve()
    try:
        selected_root.relative_to(scripts_root)
    except ValueError as error:
        raise ValueError(f"benchmark script directory is outside the srtctl package: {selected_root}") from error

    selected_files = {
        path
        for root in (selected_root, scripts_root / "lib")
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    data: dict[str, str] = {}
    items: list[dict[str, Any]] = []
    total_bytes = 0
    for index, path in enumerate(sorted(selected_files)):
        content = path.read_text(encoding="utf-8")
        total_bytes += len(content.encode())
        key = f"script-{index:04d}"
        data[key] = content
        items.append(
            {
                "key": key,
                "path": str(path.relative_to(scripts_root)),
                "mode": 0o555,
            }
        )
    if total_bytes > _MAX_CONFIG_MAP_BYTES:
        raise ValueError(
            f"benchmark scripts require {total_bytes} bytes, above the {_MAX_CONFIG_MAP_BYTES}-byte ConfigMap budget"
        )
    return (
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": config.kubernetes.namespace,
                "labels": dict(labels),
            },
            "data": data,
        },
        items,
    )


def render_kubernetes_run_manifests(
    config: SrtConfig,
    *,
    run_id: str,
    timeout_seconds: float | None = None,
    retain_finished: bool = False,
) -> list[dict[str, Any]]:
    """Render an ephemeral benchmark Job and its packaged scripts."""
    if config.benchmark.type == "manual":
        raise ValueError("k8s run requires a non-manual benchmark.type; use k8s apply for a manual deployment")
    runner = get_runner(config.benchmark.type)
    errors = runner.validate_config(config)
    if errors:
        raise ValueError("invalid benchmark configuration: " + "; ".join(errors))

    timeout = config.kubernetes.benchmark_timeout_seconds if timeout_seconds is None else timeout_seconds
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Kubernetes benchmark timeout must be finite and positive")

    deployment_name = _safe_name(config.kubernetes.name or config.name)
    safe_run_id = _safe_name(run_id, max_length=24)
    job_name = _derived_name(deployment_name, f"bench-{safe_run_id}")
    scripts_name = _derived_name(deployment_name, f"scripts-{safe_run_id}")
    frontend_host = f"{deployment_name}-frontend.{config.kubernetes.namespace}.svc.cluster.local"
    labels = {
        "app.kubernetes.io/managed-by": "srtctl",
        "app.kubernetes.io/instance": deployment_name,
        "app.kubernetes.io/component": "benchmark",
        "srtctl.nvidia.com/run-id": safe_run_id,
    }
    script_config_map, script_items = _benchmark_script_config_map(config, runner, scripts_name, labels)

    environment = {
        **config.environment,
        **config.benchmark.env,
        "SRTCTL_FRONTEND_TYPE": config.frontend.type,
        "SRT_FRONTEND_HOST": frontend_host,
        "SRT_FRONTEND_PORT": str(_DYNAMO_FRONTEND_PORT),
    }
    image = str(config.benchmark.container_image or runner.get_container_image(config, _benchmark_runtime(config)))
    container: dict[str, Any] = {
        "name": "benchmark",
        "image": image,
        "imagePullPolicy": config.kubernetes.image_pull_policy,
        "command": _benchmark_command(config, runner, frontend_host),
        "env": _string_env(environment),
        "volumeMounts": copy.deepcopy(config.kubernetes.volume_mounts),
    }
    env_from = _env_from(config)
    if env_from:
        container["envFrom"] = env_from
    if config.kubernetes.benchmark_resources:
        container["resources"] = copy.deepcopy(config.kubernetes.benchmark_resources)
    if config.kubernetes.working_dir:
        container["workingDir"] = config.kubernetes.working_dir

    volumes = copy.deepcopy(config.kubernetes.volumes)
    mount_paths = {str(item.get("mountPath", "")) for item in container["volumeMounts"]}
    volume_names = {str(item.get("name", "")) for item in volumes}
    if _BENCHMARK_OUTPUT_PATH not in mount_paths:
        if "srt-benchmark-output" in volume_names:
            raise ValueError("kubernetes.volumes already contains the reserved name srt-benchmark-output")
        output_volume: dict[str, Any] = {"name": "srt-benchmark-output"}
        if config.kubernetes.benchmark_persistent_volume_claim:
            output_volume["persistentVolumeClaim"] = {"claimName": config.kubernetes.benchmark_persistent_volume_claim}
        else:
            output_volume["emptyDir"] = {}
        volumes.append(output_volume)
        container["volumeMounts"].append({"name": "srt-benchmark-output", "mountPath": _BENCHMARK_OUTPUT_PATH})
    if script_config_map is not None:
        if "srt-benchmark-scripts" in volume_names:
            raise ValueError("kubernetes.volumes already contains the reserved name srt-benchmark-scripts")
        volumes.append(
            {
                "name": "srt-benchmark-scripts",
                "configMap": {"name": scripts_name, "items": script_items},
            }
        )
        container["volumeMounts"].append(
            {"name": "srt-benchmark-scripts", "mountPath": _BENCHMARK_MOUNT_PATH, "readOnly": True}
        )

    pod_spec: dict[str, Any] = {
        "automountServiceAccountToken": False,
        "restartPolicy": "Never",
        "containers": [container],
        "nodeSelector": dict(config.kubernetes.node_selector),
        "tolerations": copy.deepcopy(config.kubernetes.tolerations),
        "volumes": volumes,
    }
    if config.kubernetes.service_account_name:
        pod_spec["serviceAccountName"] = config.kubernetes.service_account_name
    if config.kubernetes.image_pull_secrets:
        pod_spec["imagePullSecrets"] = [{"name": value} for value in config.kubernetes.image_pull_secrets]
    job_spec: dict[str, Any] = {
        "backoffLimit": 0,
        "activeDeadlineSeconds": math.ceil(timeout),
        "template": {"metadata": {"labels": dict(labels)}, "spec": pod_spec},
    }
    if not retain_finished:
        job_spec["ttlSecondsAfterFinished"] = config.kubernetes.job_ttl_after_finished_seconds
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": config.kubernetes.namespace,
            "labels": dict(labels),
        },
        "spec": job_spec,
    }
    return [resource for resource in (script_config_map, job) if resource is not None]


def dump_kubernetes_run_yaml(
    config: SrtConfig,
    *,
    run_id: str,
    timeout_seconds: float | None = None,
    retain_finished: bool = False,
) -> str:
    return yaml.safe_dump_all(
        render_kubernetes_run_manifests(
            config,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            retain_finished=retain_finished,
        ),
        sort_keys=False,
        explicit_start=True,
    )


def _kubectl(
    args: list[str],
    *,
    executable: str,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [executable, *args],
            input=input_text,
            capture_output=True,
            text=True,
            check=check,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"kubectl executable not found: {executable}") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {detail}") from error


def _kubectl_json(
    args: list[str],
    *,
    executable: str,
    allow_failure: bool = False,
) -> dict[str, Any] | None:
    result = _kubectl(args, executable=executable, check=not allow_failure)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"kubectl {' '.join(args)} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"kubectl {' '.join(args)} did not return a JSON object")
    return value


def _deployment_identity(config: SrtConfig) -> tuple[str, str]:
    return _safe_name(config.kubernetes.name or config.name), config.kubernetes.namespace


def _dgd_exists(config: SrtConfig, *, executable: str) -> bool:
    name, namespace = _deployment_identity(config)
    result = _kubectl(
        [
            "get",
            "dynamographdeployment",
            name,
            "--namespace",
            namespace,
            "--ignore-not-found",
            "--output",
            "name",
        ],
        executable=executable,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"could not check for DynamoGraphDeployment {namespace}/{name}: {detail}")
    return bool(result.stdout.strip())


def _resource_items(value: dict[str, Any] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    return [item for item in value.get("items", []) if isinstance(item, dict)]


def _relevant_pods(config: SrtConfig, *, executable: str) -> list[dict[str, Any]]:
    name, namespace = _deployment_identity(config)
    response = _kubectl_json(
        ["get", "pods", "--namespace", namespace, "--output", "json"],
        executable=executable,
        allow_failure=True,
    )
    result = []
    for pod in _resource_items(response):
        metadata = pod.get("metadata", {})
        pod_name = str(metadata.get("name", ""))
        labels = metadata.get("labels", {})
        if labels.get("app.kubernetes.io/instance") == name or pod_name.startswith(f"{name}-"):
            result.append(pod)
    return result


def _terminal_pod_failure(pods: list[dict[str, Any]]) -> str | None:
    failures: list[tuple[int, str]] = []
    for pod in pods:
        pod_name = pod.get("metadata", {}).get("name", "<unknown>")
        statuses = [
            *pod.get("status", {}).get("initContainerStatuses", []),
            *pod.get("status", {}).get("containerStatuses", []),
        ]
        for container in statuses:
            name = container.get("name", "<unknown>")
            state = container.get("state", {})
            waiting = state.get("waiting", {})
            reason = waiting.get("reason")
            if reason in _TERMINAL_WAITING_REASONS:
                failures.append(
                    (
                        150,
                        f"pod {pod_name} container {name} is waiting: {reason}: {waiting.get('message', '')}".rstrip(),
                    )
                )
            terminated = state.get("terminated") or container.get("lastState", {}).get("terminated")
            if terminated and int(terminated.get("exitCode", 0)) != 0:
                termination_reason = str(terminated.get("reason", "unknown"))
                priority = 300 if termination_reason.casefold() == "oomkilled" else 100
                failures.append(
                    (
                        priority,
                        (
                            f"pod {pod_name} container {name} exited with {terminated.get('exitCode')}; "
                            f"reason={termination_reason}; restart count={container.get('restartCount', 0)}"
                        ),
                    )
                )
    return max(failures, key=lambda item: item[0])[1] if failures else None


def _pod_summary(pod: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    metadata = pod.get("metadata", {})
    status = pod.get("status", {})
    labels = metadata.get("labels", {})
    container_statuses = status.get("containerStatuses", [])
    waiting = {
        item.get("name", "unknown"): item.get("state", {}).get("waiting", {}).get("reason")
        for item in [*status.get("initContainerStatuses", []), *container_statuses]
        if item.get("state", {}).get("waiting", {}).get("reason")
    }
    terminated = {
        item.get("name", "unknown"): {
            "exit_code": item.get("state", {}).get("terminated", {}).get("exitCode"),
            "reason": item.get("state", {}).get("terminated", {}).get("reason"),
        }
        for item in container_statuses
        if item.get("state", {}).get("terminated")
    }
    pod_name = str(metadata.get("name", ""))
    return {
        "name": pod_name,
        "component": labels.get("srtctl.nvidia.com/component") or labels.get("app.kubernetes.io/component"),
        "phase": status.get("phase", "Unknown"),
        "ready": bool(container_statuses) and all(item.get("ready", False) for item in container_statuses),
        "restarts": sum(int(item.get("restartCount", 0)) for item in container_statuses),
        "node": pod.get("spec", {}).get("nodeName"),
        "waiting": waiting,
        "terminated": terminated,
        "metrics": metrics.get(pod_name, {}),
    }


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    metadata = job.get("metadata", {})
    status = job.get("status", {})
    state = "pending"
    if status.get("failed") or any(
        item.get("type") == "Failed" and item.get("status") == "True" for item in status.get("conditions", [])
    ):
        state = "failed"
    elif status.get("succeeded") or any(
        item.get("type") == "Complete" and item.get("status") == "True" for item in status.get("conditions", [])
    ):
        state = "succeeded"
    elif status.get("active"):
        state = "running"
    return {
        "name": metadata.get("name"),
        "run_id": metadata.get("labels", {}).get("srtctl.nvidia.com/run-id"),
        "state": state,
        "active": int(status.get("active", 0)),
        "succeeded": int(status.get("succeeded", 0)),
        "failed": int(status.get("failed", 0)),
        "started_at": status.get("startTime"),
        "completed_at": status.get("completionTime"),
    }


def get_kubernetes_status(config: SrtConfig, *, executable: str = "kubectl") -> dict[str, Any]:
    """Return normalized deployment, Job, pod, event, and metrics state."""
    name, namespace = _deployment_identity(config)
    dgd = _kubectl_json(
        ["get", "dynamographdeployment", name, "--namespace", namespace, "--output", "json"],
        executable=executable,
        allow_failure=True,
    )
    jobs = _kubectl_json(
        [
            "get",
            "jobs",
            "--namespace",
            namespace,
            "--selector",
            f"app.kubernetes.io/instance={name}",
            "--output",
            "json",
        ],
        executable=executable,
        allow_failure=True,
    )
    pods = _relevant_pods(config, executable=executable)
    metrics_response = _kubectl_json(
        ["get", "--raw", f"/apis/metrics.k8s.io/v1beta1/namespaces/{namespace}/pods"],
        executable=executable,
        allow_failure=True,
    )
    metrics = {
        str(item.get("metadata", {}).get("name", "")): {
            str(container.get("name", "")): container.get("usage", {}) for container in item.get("containers", [])
        }
        for item in _resource_items(metrics_response)
    }
    pod_names = {str(pod.get("metadata", {}).get("name", "")) for pod in pods}
    job_items = _resource_items(jobs)
    resource_names = {name, *pod_names, *(str(job.get("metadata", {}).get("name", "")) for job in job_items)}
    events_response = _kubectl_json(
        ["get", "events", "--namespace", namespace, "--output", "json"],
        executable=executable,
        allow_failure=True,
    )
    events = []
    for event in _resource_items(events_response):
        involved = event.get("involvedObject", {})
        if str(involved.get("name", "")) not in resource_names:
            continue
        events.append(
            {
                "time": event.get("eventTime")
                or event.get("lastTimestamp")
                or event.get("metadata", {}).get("creationTimestamp"),
                "type": event.get("type"),
                "reason": event.get("reason"),
                "object": f"{involved.get('kind', '')}/{involved.get('name', '')}",
                "message": event.get("message"),
                "count": event.get("count", 1),
            }
        )
    events.sort(key=lambda event: str(event.get("time") or ""))

    dgd_status = (dgd or {}).get("status", {})
    ready = any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in dgd_status.get("conditions", [])
    )
    return {
        "deployment": {
            "name": name,
            "namespace": namespace,
            "exists": dgd is not None,
            "ready": ready,
            "state": dgd_status.get("state"),
            "conditions": dgd_status.get("conditions", []),
        },
        "jobs": [
            _job_summary(job)
            for job in sorted(job_items, key=lambda item: item.get("metadata", {}).get("creationTimestamp", ""))
        ],
        "pods": [
            _pod_summary(pod, metrics)
            for pod in sorted(pods, key=lambda item: item.get("metadata", {}).get("name", ""))
        ],
        "events": events[-50:],
        "metrics_available": metrics_response is not None,
    }


def wait_for_kubernetes_job(
    config: SrtConfig,
    job_name: str,
    *,
    executable: str = "kubectl",
    timeout_seconds: float | None = None,
    on_update: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    timeout = config.kubernetes.benchmark_timeout_seconds if timeout_seconds is None else timeout_seconds
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Kubernetes benchmark timeout must be finite and positive")
    namespace = config.kubernetes.namespace
    deadline = time.monotonic() + timeout + config.kubernetes.startup_timeout_seconds
    previous_state: str | None = None
    while time.monotonic() < deadline:
        job = _kubectl_json(
            ["get", "job", job_name, "--namespace", namespace, "--output", "json"],
            executable=executable,
        )
        assert job is not None
        summary = _job_summary(job)
        state = str(summary["state"])
        if state != previous_state and on_update is not None:
            on_update(f"Benchmark Job {namespace}/{job_name}: {state}")
            previous_state = state
        if state == "succeeded":
            return job
        pod_response = _kubectl_json(
            [
                "get",
                "pods",
                "--namespace",
                namespace,
                "--selector",
                f"job-name={job_name}",
                "--output",
                "json",
            ],
            executable=executable,
        )
        failure = _terminal_pod_failure(_resource_items(pod_response))
        if state == "failed" or failure is not None:
            detail = failure or f"Job status: {job.get('status', {})}"
            raise RuntimeError(f"Kubernetes benchmark Job {namespace}/{job_name} failed: {detail}")
        serving_pods = [
            pod
            for pod in _relevant_pods(config, executable=executable)
            if pod.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component") != "benchmark"
        ]
        serving_failure = _terminal_pod_failure(serving_pods)
        if serving_failure is not None:
            raise RuntimeError(f"DynamoGraphDeployment failed while the benchmark was running: {serving_failure}")
        time.sleep(config.kubernetes.poll_interval_seconds)
    raise TimeoutError(f"Kubernetes benchmark Job {namespace}/{job_name} exceeded {timeout}s")


def stream_kubernetes_logs(
    config: SrtConfig,
    *,
    executable: str = "kubectl",
    follow: bool = False,
    component: str | None = None,
    tail: int = 200,
) -> int:
    """Stream or print logs from all deployment and benchmark containers."""
    name, namespace = _deployment_identity(config)
    selector = f"app.kubernetes.io/instance={name}"
    if component:
        label = (
            "app.kubernetes.io/component" if component in {"benchmark", "telemetry"} else "srtctl.nvidia.com/component"
        )
        selector += f",{label}={component}"
    args = [
        executable,
        "logs",
        "--namespace",
        namespace,
        "--selector",
        selector,
        "--all-containers=true",
        "--prefix=true",
        "--max-log-requests=100",
        f"--tail={tail}",
    ]
    if follow:
        args.append("--follow")
    try:
        return subprocess.run(args, check=False).returncode
    except FileNotFoundError as error:
        raise RuntimeError(f"kubectl executable not found: {executable}") from error


def _start_job_log_stream(
    config: SrtConfig,
    job_name: str,
    *,
    executable: str,
) -> subprocess.Popen[Any]:
    try:
        return subprocess.Popen(
            [
                executable,
                "logs",
                f"job/{job_name}",
                "--namespace",
                config.kubernetes.namespace,
                "--all-containers=true",
                "--prefix=true",
                "--follow",
                f"--pod-running-timeout={math.ceil(config.kubernetes.startup_timeout_seconds)}s",
            ]
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"kubectl executable not found: {executable}") from error


def _stop_log_stream(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _safe_file_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def collect_kubernetes_diagnostics(
    config: SrtConfig,
    output_dir: Path,
    *,
    executable: str = "kubectl",
) -> list[str]:
    """Capture bounded pod logs and normalized status without dumping secret-bearing pod specs."""
    destination = output_dir / "kubernetes"
    destination.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    try:
        status = get_kubernetes_status(config, executable=executable)
        (destination / "status.yaml").write_text(yaml.safe_dump(status, sort_keys=False), encoding="utf-8")
    except Exception as error:  # noqa: BLE001 - diagnostics are best effort
        warnings.append(f"could not capture Kubernetes status: {error}")

    try:
        pods = _relevant_pods(config, executable=executable)
    except Exception as error:  # noqa: BLE001 - diagnostics are best effort
        warnings.append(f"could not list Kubernetes pods: {error}")
        return warnings
    namespace = config.kubernetes.namespace
    remaining = _MAX_CAPTURED_LOG_BYTES
    logs_root = destination / "logs"
    for pod in pods:
        pod_name = str(pod.get("metadata", {}).get("name", "unknown"))
        statuses = {
            str(item.get("name", "")): item
            for item in [
                *pod.get("status", {}).get("initContainerStatuses", []),
                *pod.get("status", {}).get("containerStatuses", []),
            ]
        }
        containers = [
            *pod.get("spec", {}).get("initContainers", []),
            *pod.get("spec", {}).get("containers", []),
        ]
        for container in containers:
            if remaining <= 0:
                break
            container_name = str(container.get("name", "unknown"))
            for previous in (False, True):
                if previous and int(statuses.get(container_name, {}).get("restartCount", 0)) < 1:
                    continue
                limit = min(512 * 1024, remaining)
                args = [
                    "logs",
                    pod_name,
                    "--namespace",
                    namespace,
                    "--container",
                    container_name,
                    f"--limit-bytes={limit}",
                ]
                if previous:
                    args.append("--previous")
                result = _kubectl(args, executable=executable, check=False)
                if result.returncode != 0:
                    continue
                encoded = result.stdout.encode()
                if len(encoded) > limit:
                    encoded = encoded[-limit:]
                remaining -= len(encoded)
                logs_root.mkdir(parents=True, exist_ok=True)
                suffix = "-previous" if previous else ""
                path = logs_root / (f"{_safe_file_name(pod_name)}-{_safe_file_name(container_name)}{suffix}.log")
                path.write_bytes(encoded)
    return warnings


def collect_kubernetes_artifacts(
    config: SrtConfig,
    job_name: str,
    output_dir: Path,
    *,
    executable: str = "kubectl",
) -> list[str]:
    """Copy benchmark output and pod-local Tachometer data before cleanup."""
    warnings: list[str] = []
    namespace = config.kubernetes.namespace
    pod_response = _kubectl_json(
        [
            "get",
            "pods",
            "--namespace",
            namespace,
            "--selector",
            f"job-name={job_name}",
            "--output",
            "json",
        ],
        executable=executable,
        allow_failure=True,
    )
    job_pods = _resource_items(pod_response)
    if job_pods:
        pod_name = str(job_pods[0].get("metadata", {}).get("name", ""))
        output_dir.mkdir(parents=True, exist_ok=True)
        result = _kubectl(
            [
                "cp",
                "--namespace",
                namespace,
                "--container",
                "benchmark",
                f"{pod_name}:{_BENCHMARK_OUTPUT_PATH}/.",
                str(output_dir),
            ],
            executable=executable,
            check=False,
        )
        if result.returncode != 0:
            warnings.append(f"could not copy benchmark artifacts: {(result.stderr or result.stdout).strip()}")
    else:
        warnings.append(f"could not find a pod for benchmark Job {namespace}/{job_name}")

    if not config.telemetry.enabled:
        return warnings
    telemetry_root = output_dir / "telemetry"
    telemetry_pods = [
        pod
        for pod in _relevant_pods(config, executable=executable)
        if "srt-tachometer" in {str(item.get("name", "")) for item in pod.get("spec", {}).get("containers", [])}
    ]
    if config.kubernetes.telemetry_persistent_volume_claim:
        telemetry_pods = telemetry_pods[:1]
    for pod in telemetry_pods:
        containers = {str(item.get("name", "")) for item in pod.get("spec", {}).get("containers", [])}
        assert "srt-tachometer" in containers
        pod_name = str(pod.get("metadata", {}).get("name", ""))
        pod_destination = (
            telemetry_root
            if config.kubernetes.telemetry_persistent_volume_claim
            else telemetry_root / _safe_file_name(pod_name)
        )
        pod_destination.mkdir(parents=True, exist_ok=True)
        source = f"{config.kubernetes.telemetry_mount_path}/{config.telemetry.storage_subdir}/."
        result = _kubectl(
            [
                "cp",
                "--namespace",
                namespace,
                "--container",
                "srt-tachometer",
                f"{pod_name}:{source}",
                str(pod_destination),
            ],
            executable=executable,
            check=False,
        )
        if result.returncode != 0:
            warnings.append(f"could not copy Tachometer artifacts from {pod_name}")
    return warnings


def delete_kubernetes_run(
    config: SrtConfig,
    *,
    run_id: str,
    executable: str = "kubectl",
    timeout_seconds: float | None = None,
) -> str:
    result = _kubectl(
        ["delete", "--filename", "-", "--ignore-not-found"],
        executable=executable,
        input_text=dump_kubernetes_run_yaml(config, run_id=run_id, timeout_seconds=timeout_seconds),
    )
    return result.stdout


def run_kubernetes(
    config: SrtConfig,
    *,
    executable: str = "kubectl",
    readiness_timeout_seconds: float | None = None,
    benchmark_timeout_seconds: float | None = None,
    output_dir: Path | None = None,
    keep_resources: bool = False,
    stream_logs: bool = False,
    on_update: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Deploy, run the recipe benchmark, capture diagnostics, and clean up."""
    name, namespace = _deployment_identity(config)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    destination = output_dir or Path("outputs") / f"{name}-k8s-{run_id}"
    destination.mkdir(parents=True, exist_ok=True)
    run_yaml = dump_kubernetes_run_yaml(
        config,
        run_id=run_id,
        timeout_seconds=benchmark_timeout_seconds,
        retain_finished=keep_resources,
    )
    run_manifests = list(yaml.safe_load_all(run_yaml))
    job = next(resource for resource in run_manifests if resource.get("kind") == "Job")
    job_name = str(job["metadata"]["name"])
    log_process: subprocess.Popen[Any] | None = None
    run_applied = False
    deployment_applied = False
    deployment_preexisting = False
    primary_error: BaseException | None = None
    try:
        deployment_preexisting = _dgd_exists(config, executable=executable)
        if on_update:
            on_update(f"Applying DynamoGraphDeployment {namespace}/{name}")
        deployment_applied = True
        apply_kubernetes(
            config,
            executable=executable,
            wait=True,
            timeout_seconds=readiness_timeout_seconds,
        )
        if on_update:
            on_update(f"Frontend ready at http://{name}-frontend.{namespace}.svc.cluster.local:8000")
            on_update(f"Creating benchmark Job {namespace}/{job_name}")
        _kubectl(["apply", "--filename", "-"], executable=executable, input_text=run_yaml)
        run_applied = True
        if stream_logs:
            log_process = _start_job_log_stream(config, job_name, executable=executable)
        wait_for_kubernetes_job(
            config,
            job_name,
            executable=executable,
            timeout_seconds=benchmark_timeout_seconds,
            on_update=on_update,
        )
        _stop_log_stream(log_process)
        log_process = None
        warnings = collect_kubernetes_diagnostics(config, destination, executable=executable)
        warnings.extend(collect_kubernetes_artifacts(config, job_name, destination, executable=executable))
        for warning in warnings:
            if on_update:
                on_update(f"Warning: {warning}")
        return {
            "run_id": run_id,
            "job": job_name,
            "namespace": namespace,
            "output_dir": str(destination.resolve()),
            "warnings": warnings,
        }
    except BaseException as error:
        primary_error = error
        try:
            warnings = collect_kubernetes_diagnostics(config, destination, executable=executable)
            if run_applied:
                warnings.extend(collect_kubernetes_artifacts(config, job_name, destination, executable=executable))
            for warning in warnings:
                if on_update:
                    on_update(f"Warning: {warning}")
        except Exception as diagnostic_error:  # noqa: BLE001 - preserve the run failure
            if on_update:
                on_update(f"Warning: could not capture diagnostics: {diagnostic_error}")
        raise
    finally:
        _stop_log_stream(log_process)
        if not keep_resources:
            cleanup_failure: Exception | None = None
            if run_applied:
                try:
                    delete_kubernetes_run(
                        config,
                        run_id=run_id,
                        executable=executable,
                        timeout_seconds=benchmark_timeout_seconds,
                    )
                except Exception as cleanup_error:  # noqa: BLE001 - finish all cleanup before reporting
                    if on_update:
                        on_update(f"Warning: benchmark cleanup failed: {cleanup_error}")
                    if primary_error is None:
                        cleanup_failure = cleanup_error
            if deployment_applied and not deployment_preexisting:
                try:
                    delete_kubernetes(config, executable=executable, include_runs=False)
                except Exception as cleanup_error:  # noqa: BLE001 - finish all cleanup before reporting
                    if on_update:
                        on_update(f"Warning: deployment cleanup failed: {cleanup_error}")
                    if primary_error is None and cleanup_failure is None:
                        cleanup_failure = cleanup_error
            elif deployment_applied and deployment_preexisting and on_update:
                on_update(f"Leaving pre-existing DynamoGraphDeployment {namespace}/{name} in place")
            if cleanup_failure is not None:
                raise cleanup_failure


def wait_for_dgd(
    config: SrtConfig,
    *,
    executable: str = "kubectl",
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    name = _safe_name(config.kubernetes.name or config.name)
    namespace = config.kubernetes.namespace
    timeout = config.kubernetes.startup_timeout_seconds if timeout_seconds is None else timeout_seconds
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Kubernetes readiness timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _kubectl(
            ["get", "dynamographdeployment", name, "--namespace", namespace, "--output", "json"],
            executable=executable,
        )
        resource = json.loads(result.stdout)
        status = resource.get("status", {})
        if any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in status.get("conditions", [])
        ):
            return resource
        if str(status.get("state", "")).lower() in {"failed", "error"}:
            raise RuntimeError(f"DynamoGraphDeployment {namespace}/{name} failed: {status}")
        pod_failure = _terminal_pod_failure(_relevant_pods(config, executable=executable))
        if pod_failure is not None:
            raise RuntimeError(f"DynamoGraphDeployment {namespace}/{name} failed: {pod_failure}")
        time.sleep(config.kubernetes.poll_interval_seconds)
    raise TimeoutError(f"DynamoGraphDeployment {namespace}/{name} did not become ready within {timeout}s")


def apply_kubernetes(
    config: SrtConfig,
    *,
    executable: str = "kubectl",
    wait: bool = True,
    timeout_seconds: float | None = None,
) -> str:
    result = _kubectl(["apply", "--filename", "-"], executable=executable, input_text=dump_kubernetes_yaml(config))
    if wait:
        wait_for_dgd(config, executable=executable, timeout_seconds=timeout_seconds)
    return result.stdout


def delete_kubernetes(
    config: SrtConfig,
    *,
    executable: str = "kubectl",
    include_runs: bool = True,
) -> str:
    output = ""
    if include_runs:
        name, namespace = _deployment_identity(config)
        run_result = _kubectl(
            [
                "delete",
                "jobs,configmaps",
                "--namespace",
                namespace,
                "--selector",
                f"app.kubernetes.io/instance={name},app.kubernetes.io/component=benchmark",
                "--ignore-not-found",
            ],
            executable=executable,
        )
        output += run_result.stdout
    result = _kubectl(
        ["delete", "--filename", "-", "--ignore-not-found"],
        executable=executable,
        input_text=dump_kubernetes_yaml(config),
    )
    return output + result.stdout


def write_kubernetes_yaml(config: SrtConfig, path: Path) -> None:
    path.write_text(dump_kubernetes_yaml(config))
