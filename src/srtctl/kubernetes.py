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
from pathlib import Path
from typing import Any, Literal

import yaml

from srtctl.backends import MockerProtocol, SGLangProtocol, TRTLLMProtocol, VLLMProtocol
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


def delete_kubernetes(config: SrtConfig, *, executable: str = "kubectl") -> str:
    result = _kubectl(
        ["delete", "--filename", "-", "--ignore-not-found"],
        executable=executable,
        input_text=dump_kubernetes_yaml(config),
    )
    return result.stdout


def write_kubernetes_yaml(config: SrtConfig, path: Path) -> None:
    path.write_text(dump_kubernetes_yaml(config))
