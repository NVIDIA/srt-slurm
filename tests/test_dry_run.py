# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dry-run config details display (mounts, env vars)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from srtctl.cli.submit import show_config_details
from srtctl.core.schema import SrtConfig

# Minimal valid config that all tests build on
BASE_CONFIG = {
    "name": "test-job",
    "model": {
        "path": "/models/test-model",
        "container": "test-container.sqsh",
        "precision": "fp8",
    },
    "resources": {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    },
    "benchmark": {"type": "manual"},
}


def _make_config(overrides: dict | None = None) -> SrtConfig:
    """Build an SrtConfig from BASE_CONFIG with optional overrides merged in."""
    data = {**BASE_CONFIG}
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in data and isinstance(data[key], dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        tmp_path = Path(f.name)
    return SrtConfig.from_yaml(tmp_path)


class TestDryRunMounts:
    """Test that container mounts from all sources appear in dry-run output."""

    def test_builtin_mounts_always_shown(self, capsys):
        config = _make_config()
        show_config_details(config)
        output = capsys.readouterr().out
        assert "/model" in output
        assert "/logs" in output

    def test_extra_mount_from_recipe(self, capsys):
        config = _make_config({"extra_mount": ["/data/custom:/custom", "/shared/cache:/cache"]})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "/data/custom" in output
        assert "/custom" in output
        assert "/shared/cache" in output
        assert "/cache" in output
        assert "recipe" in output

    def test_cluster_mounts_from_srtslurm_yaml(self, capsys):
        cluster_mounts = {"/shared/datasets": "/datasets", "/shared/models": "/models"}
        with patch("srtctl.cli.submit.get_srtslurm_setting", return_value=cluster_mounts):
            config = _make_config()
            show_config_details(config)
        output = capsys.readouterr().out
        assert "/shared/datasets" in output
        assert "/datasets" in output
        assert "srtslurm.yaml" in output

    def test_mounts_from_both_cluster_and_recipe(self, capsys):
        """Mounts from srtslurm.yaml AND recipe extra_mount should both appear."""
        cluster_mounts = {"/cluster/data": "/data"}

        def mock_setting(key, default=None):
            if key == "default_mounts":
                return cluster_mounts
            return default

        with patch("srtctl.cli.submit.get_srtslurm_setting", side_effect=mock_setting):
            config = _make_config({"extra_mount": ["/recipe/models:/models"]})
            show_config_details(config)
        output = capsys.readouterr().out
        assert "/cluster/data" in output
        assert "srtslurm.yaml" in output
        assert "/recipe/models" in output
        assert "recipe" in output

    def test_no_extra_mounts_only_builtins(self, capsys):
        config = _make_config()
        show_config_details(config)
        output = capsys.readouterr().out
        assert "/model" in output
        assert "recipe" not in output


class TestDryRunEnvironment:
    """Test that environment variables from all levels appear in dry-run output."""

    def test_global_environment(self, capsys):
        config = _make_config({"environment": {"NCCL_SOCKET_IFNAME": "eth0", "MY_VAR": "hello"}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "NCCL_SOCKET_IFNAME" in output
        assert "eth0" in output
        assert "MY_VAR" in output
        assert "global" in output

    def test_backend_prefill_decode_environment(self, capsys):
        config = _make_config(
            {
                "backend": {
                    "type": "sglang",
                    "prefill_environment": {
                        "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "1800",
                        "PYTHONUNBUFFERED": "1",
                    },
                    "decode_environment": {
                        "SGLANG_ENABLE_FLASHINFER_GEMM": "1",
                    },
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT" in output
        assert "1800" in output
        assert "prefill" in output
        assert "SGLANG_ENABLE_FLASHINFER_GEMM" in output
        assert "decode" in output

    def test_global_and_backend_env_together(self, capsys):
        """Global environment AND backend per-mode env should both appear."""
        config = _make_config(
            {
                "environment": {"GLOBAL_VAR": "global_val"},
                "backend": {
                    "type": "sglang",
                    "prefill_environment": {"PREFILL_VAR": "prefill_val"},
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "GLOBAL_VAR" in output
        assert "global" in output
        assert "PREFILL_VAR" in output
        assert "prefill" in output

    def test_no_environment_shows_message(self, capsys):
        config = _make_config()
        show_config_details(config)
        output = capsys.readouterr().out
        assert "No custom environment variables configured" in output

    def test_trtllm_backend_environment(self, capsys):
        config = _make_config(
            {
                "backend": {
                    "type": "trtllm",
                    "prefill_environment": {
                        "TRTLLM_ENABLE_PDL": "1",
                        "NCCL_GRAPH_MIXING_SUPPORT": "0",
                    },
                    "decode_environment": {
                        "TRTLLM_SERVER_DISABLE_GC": "1",
                    },
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "TRTLLM_ENABLE_PDL" in output
        assert "prefill" in output
        assert "TRTLLM_SERVER_DISABLE_GC" in output
        assert "decode" in output

    def test_custom_benchmark_environment(self, capsys):
        config = _make_config(
            {
                "benchmark": {
                    "type": "custom",
                    "command": "python /bench/run.py",
                    "env": {"BENCH_FOO": "bar"},
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "BENCH_FOO" in output
        assert "benchmark" in output


class TestDryRunSrunOptions:
    """Test that srun options appear in dry-run output."""

    def test_srun_options_shown(self, capsys):
        config = _make_config({"srun_options": {"export": "ALL", "cpu-bind": "none"}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "--export ALL" in output
        assert "--cpu-bind none" in output

    def test_no_srun_options_no_output(self, capsys):
        config = _make_config()
        show_config_details(config)
        output = capsys.readouterr().out
        assert "srun options" not in output


class TestDryRunExecutionExtensions:
    """Test custom benchmark and telemetry details display."""

    def test_custom_benchmark_details_shown(self, capsys):
        config = _make_config(
            {
                "benchmark": {
                    "type": "custom",
                    "command": "python /bench/run.py",
                    "container_image": "nvcr.io/nvidia/python:3.11",
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Execution Extensions" in output
        assert "container_image" in output
        assert "nvcr.io/nvidia/python:3.11" in output

    def test_telemetry_details_shown(self, capsys):
        config = _make_config(
            {
                "telemetry": {
                    "enabled": True,
                    "container_image": "telemetry:latest",
                    "dcgm_exporter": {"container_image": "dcgm:latest", "port": 9401},
                    "node_exporter": {"container_image": "node:latest", "port": 9101},
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "telemetry" in output
        assert "scraper" in output
        assert "storage_subdir" in output
