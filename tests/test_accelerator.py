"""Tests for accelerator-specific runtime behavior."""

from types import SimpleNamespace

import pytest

from srtctl.backends import VLLMProtocol
from srtctl.cli.mixins.worker_stage import WorkerStageMixin
from srtctl.core.accelerator import visible_device_environment
from srtctl.core.schema import ClusterConfig
from srtctl.core.topology import Process


def test_nvidia_visible_device_environment() -> None:
    assert visible_device_environment("nvidia", "2,3") == {"CUDA_VISIBLE_DEVICES": "2,3"}


def test_amd_visible_device_environment_uses_rocr_linux_contract() -> None:
    assert visible_device_environment("amd", "2,3") == {"ROCR_VISIBLE_DEVICES": "2,3"}


def test_unknown_accelerator_vendor_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported accelerator vendor"):
        visible_device_environment("intel", "0")  # type: ignore[arg-type]


def test_amd_default_services_omit_nvidia_exporter(monkeypatch) -> None:
    """Upstream's default-on capture must not start NVIDIA software on AMD."""
    from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig
    from srtctl.services.implicit import effective_services

    monkeypatch.setattr(
        "srtctl.core.config.get_srtslurm_setting",
        lambda key, default=None: "amd" if key == "accelerator_vendor" else default,
    )
    config = SrtConfig(
        name="amd-default-exporters",
        model=ModelConfig(path="/model", container="/image", precision="bf16"),
        resources=ResourceConfig(gpu_type="mi355x", agg_nodes=1),
    )
    services = {entry.service.type for entry in effective_services(config)}
    assert "dcgm-exporter" not in services
    assert {"node-exporter", "process-exporter"} <= services


def test_cluster_config_defaults_to_nvidia() -> None:
    config = ClusterConfig.Schema().load({})
    assert config.accelerator_vendor == "nvidia"


def test_cluster_config_accepts_amd() -> None:
    config = ClusterConfig.Schema().load({"accelerator_vendor": "amd"})
    assert config.accelerator_vendor == "amd"


def test_cluster_config_rejects_unknown_accelerator() -> None:
    with pytest.raises(Exception, match="accelerator_vendor"):
        ClusterConfig.Schema().load({"accelerator_vendor": "intel"})


@pytest.mark.parametrize(
    ("vendor", "gpu_indices", "set_visible_devices", "legacy_setting", "expected"),
    [
        ("amd", {2, 3}, True, False, {"ROCR_VISIBLE_DEVICES": "2,3"}),
        ("amd", {2, 3}, False, True, {}),
        ("amd", {2, 3}, None, True, {"ROCR_VISIBLE_DEVICES": "2,3"}),
        ("amd", set(range(8)), True, False, {}),
        ("nvidia", {2, 3}, True, False, {"CUDA_VISIBLE_DEVICES": "2,3"}),
    ],
)
def test_amd_worker_device_mask_honors_new_setting_and_legacy_alias(
    vendor: str,
    gpu_indices: set[int],
    set_visible_devices: bool | None,
    legacy_setting: bool,
    expected: dict[str, str],
) -> None:
    """The backend setting must control the environment passed to an AMD worker."""
    process = Process(
        node="node0",
        gpu_indices=frozenset(gpu_indices),
        sys_port=7500,
        http_port=8000,
        endpoint_mode="agg",
        endpoint_index=0,
    )
    mixin = WorkerStageMixin()
    mixin.config = SimpleNamespace(
        backend=VLLMProtocol(
            set_visible_devices=set_visible_devices,
            set_cuda_visible_devices=legacy_setting,
        ),
        dynamo=SimpleNamespace(sidecar=False),
    )
    mixin.runtime = SimpleNamespace(accelerator_vendor=vendor, gpus_per_node=8)

    assert mixin._visible_device_environment(process) == expected
