# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dry-run config details display (mounts, env vars)."""

import os
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

    def test_extra_mount_expands_env_and_user_in_dry_run(self, capsys):
        with patch.dict(os.environ, {"SRT_EXTRA_ROOT": "/expanded/extra", "HOME": "/home/tester"}):
            config = _make_config({"extra_mount": ["$SRT_EXTRA_ROOT:/extra", "~/cache:/cache"]})
            show_config_details(config)
        output = capsys.readouterr().out
        assert "/expanded/extra" in output
        assert "/home/tester/cache" in output
        assert "$SRT_EXTRA_ROOT" not in output
        assert "~/cache" not in output

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
        assert "--export=ALL" in output
        assert "--cpu-bind=none" in output

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

    def test_observability_tachometer_details_shown(self, capsys):
        config = _make_config(
            {
                "observability": {
                    "enabled": True,
                    "tachometer": {"enabled": True},
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "observability" in output
        assert "raw_metrics" in output
        assert "tachometer" in output
        assert "binary_path" in output
        assert "tachometer-scraper" in output
        assert "storage_subdir" in output

    def test_dcgm_power_telemetry_details_shown(self, capsys):
        config = _make_config(
            {
                "benchmark": {"type": "sa-bench", "isl": 8192, "osl": 1024, "concurrencies": [4]},
                "telemetry": {
                    "enabled": True,
                    "default_frequency": 1.0,
                    "storage_subdir": "power",
                    "required": True,
                    "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401},
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "dcgm-power" in output
        assert "required" in output
        assert "<log_dir>/power" in output
        assert "dcgm-exporter (port 9401)" in output

    def test_mooncake_kv_store_details_shown(self, capsys):
        """mooncake_kv_store should appear in env vars and execution extensions."""
        config = _make_config(
            {
                "backend": {
                    "type": "sglang",
                    "mooncake_kv_store": {
                        "container": "nvcr.io/nvidia/mooncake:latest",
                        "env": {
                            "MOONCAKE_PROTOCOL": "rdma",
                            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4gb",
                        },
                    },
                    "sglang_config": {
                        "prefill": {"disaggregation-transfer-backend": "mooncake"},
                        "decode": {"disaggregation-transfer-backend": "mooncake"},
                    },
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        # Env table shows mooncake-scoped env vars
        assert "mooncake" in output
        assert "MOONCAKE_PROTOCOL" in output
        assert "rdma" in output
        assert "MOONCAKE_GLOBAL_SEGMENT_SIZE" in output
        # Execution extensions shows master + container
        assert "nvcr.io/nvidia/mooncake:latest" in output
        assert "master_port" in output

    def test_mooncake_kv_store_no_container_shows_default(self, capsys):
        """mooncake_kv_store without explicit container falls back to job container label."""
        config = _make_config(
            {
                "backend": {
                    "type": "sglang",
                    "mooncake_kv_store": {"env": {"MOONCAKE_PROTOCOL": "tcp"}},
                    "sglang_config": {
                        "prefill": {"disaggregation-transfer-backend": "mooncake"},
                        "decode": {"disaggregation-transfer-backend": "mooncake"},
                    },
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "<job container>" in output
        assert "MOONCAKE_PROTOCOL" in output

    def test_vllm_mooncake_kv_store_dry_run(self, capsys):
        """vLLM mooncake_kv_store renders the shared master port (8700)."""
        from srtctl.ports import MOONCAKE_MASTER_PORT

        kv_cfg = '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
        config = _make_config(
            {
                "backend": {
                    "type": "vllm",
                    "mooncake_kv_store": {
                        "container": "inferactinc/public:mk-int-20260507",
                        "env": {"MOONCAKE_PROTOCOL": "rdma"},
                        "master_extra_args": ["--nof_eviction_high_watermark_ratio=0.9"],
                    },
                    "vllm_config": {
                        "prefill": {"kv-transfer-config": kv_cfg},
                        "decode": {"kv-transfer-config": kv_cfg},
                    },
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "mooncake" in output
        assert "MOONCAKE_PROTOCOL" in output
        # Rich truncates long values with an ellipsis; assert on the stable prefix.
        assert "inferactinc/public:mk-int-202605" in output
        # Shared with the SGLang launch — same port pair.
        assert str(MOONCAKE_MASTER_PORT) in output
        assert "master_extra_args" in output
        assert "nof_eviction" in output

    def test_vllm_mooncake_store_config_in_dry_run(self, capsys):
        """vLLM store_config + MOONCAKE_CONFIG_PATH appear in the dry-run extensions panel."""
        kv_cfg = '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
        config = _make_config(
            {
                "backend": {
                    "type": "vllm",
                    "mooncake_kv_store": {
                        "env": {"MOONCAKE_PROTOCOL": "rdma"},
                        "store_config": {
                            "metadata_server": "P2PHANDSHAKE",
                            "global_segment_size": "100GB",
                            "local_buffer_size": "4GB",
                            "protocol": "rdma",
                            "device_name": "",
                        },
                    },
                    "vllm_config": {
                        "prefill": {"kv-transfer-config": kv_cfg},
                        "decode": {"kv-transfer-config": kv_cfg},
                    },
                }
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "MOONCAKE_CONFIG_PATH" in output
        assert "/logs/mooncake_store_config.json" in output
        assert "P2PHANDSHAKE" in output
        assert "100GB" in output


class TestDryRunHetJobs:
    """Het structure panel appears only when het is enabled."""

    def test_het_panel_rendered_when_enabled(self, capsys):
        config = _make_config(
            {
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 12,
                    "decode_nodes": 10,
                    "prefill_workers": 12,
                    "decode_workers": 10,
                    "het_jobs": True,
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Heterogeneous Job" in output
        assert "prefill" in output
        assert "decode" in output

    def test_het_panel_hidden_when_disabled(self, capsys):
        """No het panel when het_jobs is unset (recipe default)."""
        config = _make_config()
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Heterogeneous Job" not in output

    def test_het_panel_shows_infra_folded_into_prefill(self, capsys):
        config = _make_config(
            {
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 12,
                    "decode_nodes": 10,
                    "prefill_workers": 12,
                    "decode_workers": 10,
                    "het_jobs": True,
                },
                "infra": {"etcd_nats_dedicated_node": True},
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Heterogeneous Job" in output
        assert "first node" in output  # infra note on the prefill row


class TestDryRunRemapRoot:
    """ENROOT_REMAP_ROOT is surfaced only when dynamo will be installed."""

    def test_remap_root_shown_for_dynamo_install(self, capsys):
        config = _make_config({"frontend": {"type": "dynamo"}, "dynamo": {"install": True}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "ENROOT_REMAP_ROOT" in output

    def test_remap_root_absent_for_sglang_frontend(self, capsys):
        config = _make_config({"frontend": {"type": "sglang"}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "ENROOT_REMAP_ROOT" not in output

    def test_remap_root_absent_when_install_false(self, capsys):
        config = _make_config({"frontend": {"type": "dynamo"}, "dynamo": {"install": False}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "ENROOT_REMAP_ROOT" not in output


class TestDryRunVllmOrchestrationWarnings:
    """Dry-run warns when recipes set topology-managed vLLM flags."""

    def test_orchestration_flags_emit_warnings(self, capsys):
        config = _make_config(
            {
                "resources": {
                    "gpu_type": "b200",
                    "gpus_per_node": 8,
                    "prefill_nodes": None,
                    "decode_nodes": None,
                    "prefill_workers": None,
                    "decode_workers": None,
                    "agg_nodes": 2,
                    "agg_workers": 1,
                },
                "frontend": {"type": "vllm", "enable_multiple_frontends": False},
                "backend": {
                    "type": "vllm",
                    "vllm_config": {
                        "aggregated": {
                            "tensor-parallel-size": 8,
                            "headless": True,
                            "master-addr": "10.9.9.9",
                            "master-port": 26300,
                        }
                    },
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "WARNING:" in output
        assert "vllm_config.aggregated.headless" in output
        assert "vllm_config.aggregated.master-addr" in output
        assert "vllm_config.aggregated.master-port" not in output

    def test_clean_recipe_has_no_orchestration_warnings(self, capsys):
        config = _make_config(
            {
                "resources": {
                    "gpu_type": "b200",
                    "gpus_per_node": 8,
                    "prefill_nodes": None,
                    "decode_nodes": None,
                    "prefill_workers": None,
                    "decode_workers": None,
                    "agg_nodes": 2,
                    "agg_workers": 1,
                },
                "frontend": {"type": "vllm", "enable_multiple_frontends": False},
                "backend": {
                    "type": "vllm",
                    "vllm_config": {"aggregated": {"tensor-parallel-size": 8}},
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "vllm_config.aggregated.headless" not in output
        assert "derives this from the job topology" not in output

    def test_dynamo_recipe_has_no_direct_vllm_orchestration_warning(self, capsys):
        config = _make_config(
            {
                "resources": {
                    "gpu_type": "b200",
                    "gpus_per_node": 8,
                    "prefill_nodes": None,
                    "decode_nodes": None,
                    "prefill_workers": None,
                    "decode_workers": None,
                    "agg_nodes": 2,
                    "agg_workers": 1,
                },
                "frontend": {"type": "dynamo"},
                "backend": {
                    "type": "vllm",
                    "vllm_config": {"aggregated": {"master-addr": "10.9.9.9"}},
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "vllm_config.aggregated.master-addr" not in output
        assert "configured value is ignored" not in output


class TestInfmaxWorkspaceMount:
    """INFMAX_WORKSPACE comes from the environment, not the recipe.

    That is precisely why dry-run has to show it: nothing in the config file mentions
    it, so a reader comparing recipe against dry-run sees what looks like a complete
    picture and is wrong. Recipes whose benchmark command lives under
    /infmax-workspace fail with exit 127 after the workers have loaded when it is
    missing -- observed on ablation arms 2751879-82, where diffing the working run's
    --container-mounts against the failed arm's showed this single missing entry.
    """

    AGENTIC = {"benchmark": {"type": "custom",
                             "command": "bash /infmax-workspace/benchmarks/multi_node/agentic_srt.sh"}}

    def test_mount_is_shown_when_the_variable_is_set(self, capsys):
        config = _make_config(self.AGENTIC)
        with patch.dict(os.environ, {"INFMAX_WORKSPACE": "/lustre/checkouts/InferenceX-abc"}):
            show_config_details(config)
        output = capsys.readouterr().out
        assert "/lustre/checkouts/InferenceX-abc" in output
        assert "/infmax-workspace" in output

    def test_absence_is_flagged_when_the_benchmark_needs_it(self, capsys):
        """The failure this prevents is expensive and silent: the table listed every
        other mount, so the missing row read as 'no such mount exists' rather than
        'not set in this environment'."""
        config = _make_config(self.AGENTIC)
        env = {k: v for k, v in os.environ.items() if k != "INFMAX_WORKSPACE"}
        with patch.dict(os.environ, env, clear=True):
            show_config_details(config)
        output = capsys.readouterr().out
        assert "MISSING" in output
        assert "127" in output, "the dry-run must name the failure mode, not just the gap"

    def test_no_warning_when_the_benchmark_does_not_use_it(self, capsys):
        """Most recipes never touch /infmax-workspace; they must not be nagged."""
        config = _make_config()
        env = {k: v for k, v in os.environ.items() if k != "INFMAX_WORKSPACE"}
        with patch.dict(os.environ, env, clear=True):
            show_config_details(config)
        output = capsys.readouterr().out
        assert "MISSING" not in output
