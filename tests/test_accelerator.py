"""Tests for accelerator-specific runtime behavior."""

import base64
import re
from pathlib import Path
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
    config = ClusterConfig.Schema().load(
        {
            "accelerator_vendor": "amd",
            "gpu_sbatch_directive": "gres",
            "runtime_config_transport": "embedded",
        }
    )
    assert config.accelerator_vendor == "amd"
    assert config.gpu_sbatch_directive == "gres"
    assert config.runtime_config_transport == "embedded"


def test_cluster_config_rejects_unknown_accelerator() -> None:
    with pytest.raises(Exception, match="accelerator_vendor"):
        ClusterConfig.Schema().load({"accelerator_vendor": "intel"})


def test_cluster_config_rejects_unknown_gpu_directive() -> None:
    with pytest.raises(Exception, match="gpu_sbatch_directive"):
        ClusterConfig.Schema().load({"gpu_sbatch_directive": "rocm"})


@pytest.mark.parametrize(
    ("set_visible_devices", "legacy_setting", "expected"),
    [
        (True, False, {"ROCR_VISIBLE_DEVICES": "2,3"}),
        (False, True, {}),
        (None, True, {"ROCR_VISIBLE_DEVICES": "2,3"}),
    ],
)
def test_amd_worker_device_mask_honors_new_setting_and_legacy_alias(
    set_visible_devices: bool | None,
    legacy_setting: bool,
    expected: dict[str, str],
) -> None:
    """The backend setting must control the environment passed to an AMD worker."""
    process = Process(
        node="node0",
        gpu_indices=frozenset({2, 3}),
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
    mixin.runtime = SimpleNamespace(accelerator_vendor="amd", gpus_per_node=8)

    assert mixin._visible_device_environment(process) == expected


@pytest.mark.parametrize(
    ("directive", "expected", "unexpected"),
    [
        ("gpus-per-node", "#SBATCH --gpus-per-node=8", "#SBATCH --gres=gpu:8"),
        ("gres", "#SBATCH --gres=gpu:8", "#SBATCH --gpus-per-node=8"),
        ("none", None, "#SBATCH --gpus-per-node=8"),
    ],
)
def test_gpu_sbatch_directive_rendering(monkeypatch, directive, expected, unexpected) -> None:
    from srtctl.cli import submit
    from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

    settings = {
        "gpu_sbatch_directive": directive,
        "use_segment_sbatch_directive": False,
    }
    monkeypatch.setattr(submit, "get_srtslurm_setting", lambda key, default=None: settings.get(key, default))
    config = SrtConfig(
        name="amd-render-test",
        model=ModelConfig(path="/model", container="/container.sqsh", precision="fp16"),
        resources=ResourceConfig(gpu_type="mi300x", gpus_per_node=8, agg_nodes=1),
    )

    script = submit.generate_minimal_sbatch_script(config, Path("/tmp/amd-render-test.yaml"))

    if expected is not None:
        assert expected in script
    else:
        assert "#SBATCH --gpus-per-node=" not in script
        assert "#SBATCH --gres=gpu:" not in script
    assert unexpected not in script


def test_legacy_gpu_directive_boolean_is_preserved(monkeypatch) -> None:
    from srtctl.cli import submit
    from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

    settings = {
        "gpu_sbatch_directive": None,
        "use_gpus_per_node_directive": False,
        "use_segment_sbatch_directive": False,
    }
    monkeypatch.setattr(submit, "get_srtslurm_setting", lambda key, default=None: settings.get(key, default))
    config = SrtConfig(
        name="legacy-render-test",
        model=ModelConfig(path="/model", container="/container.sqsh", precision="fp16"),
        resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
    )

    script = submit.generate_minimal_sbatch_script(config, Path("/tmp/legacy-render-test.yaml"))

    assert "#SBATCH --gpus-per-node=" not in script
    assert "#SBATCH --gres=gpu:" not in script


def test_shared_filesystem_runtime_config_transport_is_unchanged(monkeypatch) -> None:
    from srtctl.cli import submit
    from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

    monkeypatch.setattr(submit, "get_srtslurm_setting", lambda key, default=None: default)
    config = SrtConfig(
        name="shared-output-test",
        model=ModelConfig(path="/model", container="/container.sqsh", precision="fp16"),
        resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
    )

    script = submit.generate_minimal_sbatch_script(config, Path("/tmp/not-required-for-shared.yaml"))

    assert "#SBATCH --output=" in script
    assert "/%j/logs/sweep_%j.log" in script
    assert ".srtctl-sweep-%j.log" not in script
    assert "base64 --decode" not in script


def test_embedded_transport_preserves_resolved_yaml_as_inert_data(monkeypatch, tmp_path) -> None:
    from srtctl.cli import submit
    from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

    settings = {"runtime_config_transport": "embedded"}
    monkeypatch.setattr(submit, "get_srtslurm_setting", lambda key, default=None: settings.get(key, default))
    cluster_config_path = tmp_path / "srtslurm-source.yaml"
    cluster_config_text = "cluster: mi300x-amds\ncontainers:\n  rocm: /images/rocm.sqsh\n"
    cluster_config_path.write_text(cluster_config_text)
    monkeypatch.setattr(submit, "find_cluster_config_path", lambda: cluster_config_path)
    config = SrtConfig(
        name="node-local-output-test",
        model=ModelConfig(path="/model", container="/container.sqsh", precision="fp16"),
        resources=ResourceConfig(gpu_type="mi300x", gpus_per_node=1, agg_nodes=1),
    )
    source_text = "name: source\nnote: original\n"
    runtime_text = 'name: resolved\nnote: "\'; touch /tmp/must-not-run; #"\n'
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(runtime_text)

    script = submit.generate_minimal_sbatch_script(
        config,
        config_path,
        runtime_config_filename="config_variant.yaml",
        runtime_config_text=runtime_text,
        source_config_text=source_text,
    )

    assert "#SBATCH --output=" in script
    assert "/.srtctl-sweep-%j.log" in script
    assert 'mv "${BOOTSTRAP_LOG}" "${LOG_DIR}/sweep_${SLURM_JOB_ID}.log"' in script
    assert '--ntasks-per-node=1 mkdir -p "${LOG_DIR}"' in script
    assert "#SBATCH --chdir=" in script
    assert 'export SRTSLURM_CONFIG="${OUTPUT_DIR}/srtslurm.yaml"' in script
    assert "touch /tmp/must-not-run" not in script

    embedded = re.findall(
        r'RUNTIME_CONFIG="\$\{OUTPUT_DIR\}/([^\"]+)".*?printf \'%s\' \'([^\']+)\' \| base64 --decode',
        script,
        flags=re.DOTALL,
    )
    decoded = {filename: base64.b64decode(payload).decode() for filename, payload in embedded}
    assert decoded == {
        "config.yaml": source_text,
        "config_variant.yaml": runtime_text,
        "srtslurm.yaml": cluster_config_text,
    }
