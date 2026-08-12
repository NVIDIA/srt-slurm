# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.cli.kubernetes import dump_configs, load_kubernetes_configs
from srtctl.core.schema import SrtConfig
from srtctl.kubernetes import (
    apply_kubernetes,
    delete_kubernetes,
    dump_kubernetes_yaml,
    render_kubernetes_manifests,
    wait_for_dgd,
)


def _config(*, telemetry: bool = False, runtime_tag: str = "1.4.0") -> SrtConfig:
    values = {
        "name": "k8s-test",
        "model": {
            "path": "hf:Qwen/Qwen3-0.6B",
            "container": f"registry.example/dynamo-sglang:{runtime_tag}",
            "precision": "bf16",
        },
        "resources": {
            "gpu_type": "h100",
            "gpus_per_node": 8,
            "prefill_nodes": 2,
            "prefill_workers": 1,
            "decode_nodes": 1,
            "decode_workers": 1,
        },
        "backend": {
            "type": "sglang",
            "sglang_config": {
                "prefill": {"trust-remote-code": True},
                "decode": {"trust-remote-code": True},
            },
        },
        "frontend": {
            "type": "dynamo",
            "enable_multiple_frontends": False,
            "args": {"router-mode": "kv"},
        },
        "observability": {"enable_otel": True, "otel_endpoint": "http://otel:4317"},
        "kubernetes": {
            "namespace": "bench",
            "env_from_secrets": ["hf-token"],
            "node_selector": {"accelerator": "h100"},
        },
    }
    if telemetry:
        values["telemetry"] = {
            "enabled": True,
            "container_image": "ghcr.io/nvidia-dev/warnold-tachometer-scraper:latest",
            "binary_path": "/usr/local/bin/tachometer-scraper",
            "storage_subdir": "tachometer",
            "extra_metadata": {"scenario": "k8s-test"},
            "dcgm_exporter": {
                "container_image": "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04",
                "port": 9400,
            },
            "node_exporter": {"container_image": "quay.io/prometheus/node-exporter:v1.9.1", "port": 9100},
        }
        values["kubernetes"]["telemetry_persistent_volume_claim"] = "benchmark-telemetry"
    loaded = SrtConfig.Schema().load(values)
    assert isinstance(loaded, SrtConfig)
    return loaded


def _component(dgd: dict, name: str) -> dict:
    return next(component for component in dgd["spec"]["components"] if component["name"] == name)


def test_render_disaggregated_sglang_dgd() -> None:
    config = _config()
    manifests = render_kubernetes_manifests(config)

    assert len(manifests) == 1
    dgd = manifests[0]
    assert dgd["apiVersion"] == "nvidia.com/v1beta1"
    assert dgd["kind"] == "DynamoGraphDeployment"
    assert dgd["metadata"]["namespace"] == "bench"
    assert dgd["spec"]["backendFramework"] == "sglang"
    assert [component["type"] for component in dgd["spec"]["components"]] == [
        "frontend",
        "decode",
        "prefill",
    ]

    prefill = _component(dgd, "prefill")
    assert prefill["multinode"] == {"nodeCount": 2}
    main = prefill["podTemplate"]["spec"]["containers"][0]
    assert main["command"] == ["python3", "-m", "dynamo.sglang"]
    assert main["resources"]["limits"]["nvidia.com/gpu"] == "8"
    assert main["args"][main["args"].index("--tp-size") + 1] == "16"
    assert main["envFrom"] == [{"secretRef": {"name": "hf-token"}}]
    env = {item["name"]: item["value"] for item in main["env"]}
    assert env["OTEL_SERVICE_NAME"] == "dynamo-prefill"

    rendered = dump_kubernetes_yaml(config)
    assert "&id" not in rendered
    assert "router_zoo" not in rendered


def test_render_tachometer_sidecars_and_node_daemon_set() -> None:
    daemon_set, dgd = render_kubernetes_manifests(_config(telemetry=True))

    assert daemon_set["kind"] == "DaemonSet"
    assert daemon_set["metadata"]["name"] == "k8s-test-telemetry"
    containers = daemon_set["spec"]["template"]["spec"]["containers"]
    assert [container["name"] for container in containers] == [
        "srt-dcgm-exporter",
        "srt-node-exporter",
        "srt-tachometer",
    ]
    node_script = containers[2]["args"][0]
    assert 'filter = "dcgm"' in node_script
    assert 'filter = "node_exporter"' in node_script
    assert "node-__NODE_NAME__" in node_script
    assert "<<'EOF'" in node_script

    for component in dgd["spec"]["components"]:
        pod = component["podTemplate"]["spec"]
        assert [container["name"] for container in pod["containers"]] == ["main", "srt-tachometer"]
        tachometer = pod["containers"][1]
        filter_name = "frontend" if component["name"] == "frontend" else "backend"
        assert f'filter = "{filter_name}"' in tachometer["args"][0]
        assert "dcgm" not in tachometer["args"][0]
        assert pod["volumes"] == [
            {"name": "srt-telemetry", "persistentVolumeClaim": {"claimName": "benchmark-telemetry"}}
        ]
    assert "&id" not in dump_kubernetes_yaml(_config(telemetry=True))


def test_runtime_override_is_required_for_non_semver_image_tag() -> None:
    config = _config(runtime_tag="latest")
    with pytest.raises(ValueError, match="kubernetes.runtime_version is required"):
        render_kubernetes_manifests(config)

    values = SrtConfig.Schema().dump(config)
    values["kubernetes"]["runtime_version"] = "1.4.0"
    loaded = SrtConfig.Schema().load(values)
    assert _component(render_kubernetes_manifests(loaded)[0], "frontend")["runtimeVersionOverride"] == "1.4.0"


def test_non_uniform_multinode_shape_is_rejected() -> None:
    values = SrtConfig.Schema().dump(_config())
    values["resources"].update({"prefill_nodes": 3, "prefill_workers": 2})
    config = SrtConfig.Schema().load(values)
    with pytest.raises(ValueError, match="cannot be divided evenly"):
        render_kubernetes_manifests(config)


def test_spread_workers_adds_hostname_anti_affinity() -> None:
    values = SrtConfig.Schema().dump(_config())
    values["resources"].update({"prefill_nodes": 2, "prefill_workers": 2, "spread_workers": True})
    config = SrtConfig.Schema().load(values)
    prefill = _component(render_kubernetes_manifests(config)[0], "prefill")
    terms = prefill["podTemplate"]["spec"]["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]
    assert terms == [
        {
            "labelSelector": {"matchLabels": {"srtctl.nvidia.com/component": "prefill"}},
            "topologyKey": "kubernetes.io/hostname",
        }
    ]


@pytest.mark.parametrize(
    ("backend", "module", "supporting_kind"),
    [
        ({"type": "mocker"}, "dynamo.mocker", None),
        (
            {
                "type": "vllm",
                "connector": "nixl",
                "vllm_config": {"aggregated": {"gpu-memory-utilization": 0.9}},
            },
            "dynamo.vllm",
            None,
        ),
        (
            {"type": "trtllm", "trtllm_config": {"aggregated": {"max_num_tokens": 8192}}},
            "dynamo.trtllm",
            "ConfigMap",
        ),
    ],
)
def test_aggregated_backend_rendering(backend: dict, module: str, supporting_kind: str | None) -> None:
    values = SrtConfig.Schema().dump(_config())
    values["resources"] = {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "agg_nodes": 1,
        "agg_workers": 1,
    }
    values["backend"] = backend
    config = SrtConfig.Schema().load(values)

    manifests = render_kubernetes_manifests(config)
    dgd = manifests[-1]
    worker = _component(dgd, "worker")
    main = worker["podTemplate"]["spec"]["containers"][0]
    assert main["command"] == ["python3", "-m", module]
    assert worker["type"] == "worker"
    assert (manifests[0]["kind"] if len(manifests) > 1 else None) == supporting_kind
    if backend["type"] == "mocker":
        assert "--model-name" in main["args"]
        assert "resources" not in main
        assert "backendFramework" not in dgd["spec"]
    else:
        assert dgd["spec"]["backendFramework"] == backend["type"]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"namespace": "Not-Valid"}, "namespace"),
        ({"image_pull_policy": "Sometimes"}, "image_pull_policy"),
        ({"telemetry_mount_path": "logs"}, "telemetry_mount_path"),
        ({"poll_interval_seconds": 0}, "poll_interval_seconds"),
    ],
)
def test_kubernetes_schema_validation(values: dict, message: str) -> None:
    raw = SrtConfig.Schema().dump(_config())
    raw["kubernetes"].update(values)
    with pytest.raises(ValidationError, match=message):
        SrtConfig.Schema().load(raw)


def test_override_recipe_generation(tmp_path: Path) -> None:
    raw = SrtConfig.Schema().dump(_config())
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "base": raw,
                "override_small": {"resources": {"prefill_nodes": 1}},
                "override_large": {"resources": {"prefill_nodes": 2}},
            },
            sort_keys=False,
        )
    )

    configs = load_kubernetes_configs(recipe, selector="override_small")
    assert len(configs) == 1
    assert configs[0].resources.prefill_nodes == 1
    assert "kind: DynamoGraphDeployment" in dump_configs(configs)


def test_apply_and_delete_use_generated_manifests(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(telemetry=True)
    calls: list[tuple[list[str], str | None]] = []

    def fake_kubectl(
        args: list[str], *, executable: str, input_text: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, input_text))
        return subprocess.CompletedProcess([executable, *args], 0, "ok\n", "")

    monkeypatch.setattr("srtctl.kubernetes._kubectl", fake_kubectl)
    monkeypatch.setattr("srtctl.kubernetes.wait_for_dgd", lambda *args, **kwargs: json.loads("{}"))

    assert apply_kubernetes(config) == "ok\n"
    assert calls[0][0] == ["apply", "--filename", "-"]
    assert "kind: DaemonSet" in (calls[0][1] or "")
    assert delete_kubernetes(config) == "ok\n"
    assert calls[1][0] == ["delete", "--filename", "-", "--ignore-not-found"]
    assert calls[1][1] == calls[0][1]


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ({"conditions": [{"type": "Ready", "status": "True"}]}, None),
        ({"state": "failed", "message": "image pull failed"}, RuntimeError),
    ],
)
def test_wait_for_dgd_status(monkeypatch: pytest.MonkeyPatch, status: dict, error: type[Exception] | None) -> None:
    resource = {"status": status}

    def fake_kubectl(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["kubectl"], 0, json.dumps(resource), "")

    monkeypatch.setattr("srtctl.kubernetes._kubectl", fake_kubectl)
    if error is not None:
        with pytest.raises(error):
            wait_for_dgd(_config())
    else:
        assert wait_for_dgd(_config()) == resource
