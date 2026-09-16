"""Cluster-configured GPU visibility and exporter behavior."""

from types import SimpleNamespace

import pytest

from srtctl.backends import VLLMProtocol
from srtctl.cli.mixins.worker_stage import WorkerStageMixin
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.schema import ClusterConfig, SrtConfig
from srtctl.core.topology import Process
from srtctl.services.implicit import effective_services


@pytest.mark.parametrize(
    ("mask", "indices", "enabled", "sidecar", "expected"),
    [
        ("ROCR_VISIBLE_DEVICES", {2, 3}, True, False, {"ROCR_VISIBLE_DEVICES": "2,3"}),
        ("CUDA_VISIBLE_DEVICES", {2, 3}, True, False, {"CUDA_VISIBLE_DEVICES": "2,3"}),
        ("OTHER_DEVICE_MASK", {2, 3}, True, False, {"OTHER_DEVICE_MASK": "2,3"}),
        ("ROCR_VISIBLE_DEVICES", {2, 3}, False, False, {}),
        ("ROCR_VISIBLE_DEVICES", set(range(8)), True, False, {}),
        ("ROCR_VISIBLE_DEVICES", {2, 3}, False, True, {"ROCR_VISIBLE_DEVICES": "2,3"}),
    ],
)
def test_worker_mask_uses_cluster_setting(mask, indices, enabled, sidecar, expected):
    process = Process(
        node="node0",
        gpu_indices=frozenset(indices),
        sys_port=7500,
        http_port=8000,
        endpoint_mode="agg",
        endpoint_index=0,
    )
    mixin = WorkerStageMixin()
    mixin.config = SimpleNamespace(
        backend=VLLMProtocol(set_visible_devices=enabled),
        dynamo=SimpleNamespace(sidecar=sidecar),
    )
    mixin.runtime = SimpleNamespace(visible_devices_env=mask, gpus_per_node=8)
    assert mixin._visible_device_environment(process) == expected


def test_cluster_can_disable_gpu_exporter_without_disabling_host_metrics():
    cluster = ClusterConfig.Schema().dump(ClusterConfig.Schema().load({"default_gpu_exporter": None}))
    resolved = resolve_config_with_defaults(
        {
            "name": "test",
            "model": {"path": "/model", "container": "/image", "precision": "bf16"},
            "resources": {"gpu_type": "mi355x", "agg_nodes": 1},
        },
        cluster,
    )
    config = SrtConfig.Schema().load(resolved)
    services = {entry.service.type for entry in effective_services(config)}
    assert "dcgm-exporter" not in services
    assert {"node-exporter", "process-exporter"} <= services


@pytest.mark.parametrize("recipe_override", [False, True])
def test_exporter_resolution_honors_recipe_and_resolves_cluster_image_alias(recipe_override):
    recipe = {
        "name": "test",
        "model": {"path": "/model", "container": "/image", "precision": "bf16"},
        "resources": {"gpu_type": "mi355x", "agg_nodes": 1},
    }
    if recipe_override:
        recipe["observability"] = {
            "tachometer": {
                "dcgm_exporter": {
                    "container_image": "recipe:image",
                    "port": 9200,
                    "command": "recipe-exporter",
                }
            }
        }
    resolved = resolve_config_with_defaults(
        recipe,
        {
            "default_gpu_exporter": {"container_image": "gpu-exporter", "port": 9300, "command": "device-exporter"},
            "containers": {"gpu-exporter": "cluster:image"},
        },
    )
    config = SrtConfig.Schema().load(resolved)
    exporter = config.observability.tachometer.resolved_dcgm_exporter
    assert (exporter.container_image, exporter.port, exporter.command) == (
        ("recipe:image", 9200, "recipe-exporter") if recipe_override else ("cluster:image", 9300, "device-exporter")
    )
