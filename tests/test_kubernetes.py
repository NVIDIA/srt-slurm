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
    get_kubernetes_status,
    render_kubernetes_manifests,
    render_kubernetes_run_manifests,
    run_kubernetes,
    wait_for_dgd,
    wait_for_kubernetes_job,
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
        ({"benchmark_timeout_seconds": 0}, "benchmark_timeout_seconds"),
        ({"job_ttl_after_finished_seconds": -1}, "job_ttl_after_finished_seconds"),
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
    assert delete_kubernetes(config) == "ok\nok\n"
    assert calls[1][0][:2] == ["delete", "jobs,configmaps"]
    assert calls[1][1] is None
    assert calls[2][0] == ["delete", "--filename", "-", "--ignore-not-found"]
    assert calls[2][1] == calls[0][1]


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


def test_render_benchmark_job_packages_scripts_and_targets_frontend_service() -> None:
    raw = SrtConfig.Schema().dump(_config())
    raw["benchmark"] = {
        "type": "sa-bench",
        "isl": 1024,
        "osl": 128,
        "concurrencies": [8, 16],
        "container_image": "registry.example/benchmark:latest",
        "env": {"HF_TOKEN": "from-secret-preferred"},
    }
    raw["kubernetes"].update(
        {
            "benchmark_persistent_volume_claim": "benchmark-results",
            "benchmark_resources": {
                "requests": {"cpu": "4", "memory": "8Gi"},
                "limits": {"cpu": "8", "memory": "16Gi"},
            },
        }
    )
    config = SrtConfig.Schema().load(raw)

    config_map, job = render_kubernetes_run_manifests(config, run_id="run-1")

    assert config_map["kind"] == "ConfigMap"
    item_paths = {item["path"] for item in job["spec"]["template"]["spec"]["volumes"][-1]["configMap"]["items"]}
    assert "sa-bench/bench.sh" in item_paths
    assert "lib/profiling.sh" in item_paths
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "registry.example/benchmark:latest"
    assert container["command"][1] == "/srtctl-benchmarks/sa-bench/bench.sh"
    assert container["command"][2] == "http://k8s-test-frontend.bench.svc.cluster.local:8000"
    assert container["resources"]["limits"]["memory"] == "16Gi"
    assert {
        "name": "srt-benchmark-output",
        "persistentVolumeClaim": {"claimName": "benchmark-results"},
    } in job["spec"]["template"]["spec"]["volumes"]
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["activeDeadlineSeconds"] == 3600


def test_render_custom_benchmark_job_needs_no_script_config_map() -> None:
    raw = SrtConfig.Schema().dump(_config())
    raw["benchmark"] = {
        "type": "custom",
        "command": 'curl -fsS "http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}/health"',
        "container_image": "curlimages/curl:8.14.1",
    }
    config = SrtConfig.Schema().load(raw)

    manifests = render_kubernetes_run_manifests(config, run_id="smoke")

    assert len(manifests) == 1
    job = manifests[0]
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == [
        "bash",
        "-lc",
        'curl -fsS "http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}/health"',
    ]
    assert all(volume["name"] != "srt-benchmark-scripts" for volume in job["spec"]["template"]["spec"]["volumes"])
    retained_job = render_kubernetes_run_manifests(config, run_id="retained", retain_finished=True)[0]
    assert "ttlSecondsAfterFinished" not in retained_job["spec"]


def test_status_reports_jobs_pods_metrics_and_events(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_kubectl(
        args: list[str], *, executable: str, input_text: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        del input_text, check
        if args[:2] == ["get", "dynamographdeployment"]:
            payload = {"status": {"conditions": [{"type": "Ready", "status": "True"}]}}
        elif args[:2] == ["get", "jobs"]:
            payload = {
                "items": [
                    {
                        "metadata": {
                            "name": "k8s-test-bench-run-1",
                            "labels": {"srtctl.nvidia.com/run-id": "run-1"},
                        },
                        "status": {"active": 1},
                    }
                ]
            }
        elif args[:2] == ["get", "pods"]:
            payload = {
                "items": [
                    {
                        "metadata": {
                            "name": "k8s-test-frontend-0",
                            "labels": {"app.kubernetes.io/instance": "k8s-test"},
                        },
                        "spec": {"nodeName": "node-a"},
                        "status": {
                            "phase": "Running",
                            "containerStatuses": [{"name": "main", "ready": True, "restartCount": 0}],
                        },
                    }
                ]
            }
        elif args[:2] == ["get", "--raw"]:
            payload = {
                "items": [
                    {
                        "metadata": {"name": "k8s-test-frontend-0"},
                        "containers": [{"name": "main", "usage": {"cpu": "2", "memory": "1Gi"}}],
                    }
                ]
            }
        elif args[:2] == ["get", "events"]:
            payload = {
                "items": [
                    {
                        "type": "Normal",
                        "reason": "Started",
                        "message": "Started container",
                        "involvedObject": {"kind": "Pod", "name": "k8s-test-frontend-0"},
                    }
                ]
            }
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess([executable, *args], 0, json.dumps(payload), "")

    monkeypatch.setattr("srtctl.kubernetes._kubectl", fake_kubectl)

    status = get_kubernetes_status(_config())

    assert status["deployment"]["ready"] is True
    assert status["jobs"][0]["state"] == "running"
    assert status["pods"][0]["metrics"]["main"]["memory"] == "1Gi"
    assert status["events"][0]["reason"] == "Started"


def test_wait_for_job_surfaces_terminal_pod_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_kubectl(
        args: list[str], *, executable: str, input_text: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        del input_text, check
        payload: dict
        if args[:2] == ["get", "job"]:
            payload = {"status": {"active": 1}}
        else:
            payload = {
                "items": [
                    {
                        "metadata": {"name": "benchmark-pod"},
                        "status": {
                            "containerStatuses": [
                                {
                                    "name": "benchmark",
                                    "restartCount": 0,
                                    "state": {"terminated": {"exitCode": 137, "reason": "OOMKilled"}},
                                }
                            ]
                        },
                    }
                ]
            }
        return subprocess.CompletedProcess([executable, *args], 0, json.dumps(payload), "")

    monkeypatch.setattr("srtctl.kubernetes._kubectl", fake_kubectl)

    with pytest.raises(RuntimeError, match="OOMKilled"):
        wait_for_kubernetes_job(_config(), "benchmark")


def test_run_cleans_up_after_benchmark_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = SrtConfig.Schema().dump(_config())
    raw["benchmark"] = {
        "type": "custom",
        "command": "false",
        "container_image": "registry.example/benchmark:latest",
    }
    config = SrtConfig.Schema().load(raw)
    cleaned: list[str] = []

    monkeypatch.setattr("srtctl.kubernetes.apply_kubernetes", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "srtctl.kubernetes._kubectl",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    monkeypatch.setattr(
        "srtctl.kubernetes.wait_for_kubernetes_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("benchmark failed")),
    )
    monkeypatch.setattr("srtctl.kubernetes.collect_kubernetes_diagnostics", lambda *args, **kwargs: [])
    monkeypatch.setattr("srtctl.kubernetes.delete_kubernetes_run", lambda *args, **kwargs: cleaned.append("run") or "")
    monkeypatch.setattr(
        "srtctl.kubernetes.delete_kubernetes", lambda *args, **kwargs: cleaned.append("deployment") or ""
    )

    with pytest.raises(RuntimeError, match="benchmark failed"):
        run_kubernetes(config, output_dir=tmp_path, stream_logs=False)

    assert cleaned == ["run", "deployment"]


def test_run_leaves_preexisting_deployment_after_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = SrtConfig.Schema().dump(_config())
    raw["benchmark"] = {
        "type": "custom",
        "command": "false",
        "container_image": "registry.example/benchmark:latest",
    }
    config = SrtConfig.Schema().load(raw)
    cleaned: list[str] = []

    def fake_kubectl(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        stdout = "dynamographdeployment.nvidia.com/k8s-test\n" if "--ignore-not-found" in args else ""
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr("srtctl.kubernetes.apply_kubernetes", lambda *args, **kwargs: "")
    monkeypatch.setattr("srtctl.kubernetes._kubectl", fake_kubectl)
    monkeypatch.setattr(
        "srtctl.kubernetes.wait_for_kubernetes_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("benchmark failed")),
    )
    monkeypatch.setattr("srtctl.kubernetes.collect_kubernetes_diagnostics", lambda *args, **kwargs: [])
    monkeypatch.setattr("srtctl.kubernetes.collect_kubernetes_artifacts", lambda *args, **kwargs: [])
    monkeypatch.setattr("srtctl.kubernetes.delete_kubernetes_run", lambda *args, **kwargs: cleaned.append("run") or "")
    monkeypatch.setattr(
        "srtctl.kubernetes.delete_kubernetes", lambda *args, **kwargs: cleaned.append("deployment") or ""
    )

    with pytest.raises(RuntimeError, match="benchmark failed"):
        run_kubernetes(config, output_dir=tmp_path, stream_logs=False)

    assert cleaned == ["run"]


def test_run_attempts_deployment_cleanup_when_job_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = SrtConfig.Schema().dump(_config())
    raw["benchmark"] = {
        "type": "custom",
        "command": "true",
        "container_image": "registry.example/benchmark:latest",
    }
    config = SrtConfig.Schema().load(raw)
    cleaned: list[str] = []

    monkeypatch.setattr("srtctl.kubernetes.apply_kubernetes", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "srtctl.kubernetes._kubectl",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    monkeypatch.setattr("srtctl.kubernetes.wait_for_kubernetes_job", lambda *args, **kwargs: {})
    monkeypatch.setattr("srtctl.kubernetes.collect_kubernetes_diagnostics", lambda *args, **kwargs: [])
    monkeypatch.setattr("srtctl.kubernetes.collect_kubernetes_artifacts", lambda *args, **kwargs: [])

    def fail_run_cleanup(*args, **kwargs):
        cleaned.append("run")
        raise RuntimeError("run cleanup failed")

    monkeypatch.setattr("srtctl.kubernetes.delete_kubernetes_run", fail_run_cleanup)
    monkeypatch.setattr(
        "srtctl.kubernetes.delete_kubernetes", lambda *args, **kwargs: cleaned.append("deployment") or ""
    )

    with pytest.raises(RuntimeError, match="run cleanup failed"):
        run_kubernetes(config, output_dir=tmp_path, stream_logs=False)

    assert cleaned == ["run", "deployment"]
