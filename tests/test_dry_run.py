# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dry-run config details display (mounts, env vars)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
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


def test_cluster_gpu_visibility_is_visible(tmp_path, monkeypatch, capsys):
    cluster_config = tmp_path / "srtslurm.yaml"
    cluster_config.write_text(yaml.safe_dump({"visible_devices_env": "ROCR_VISIBLE_DEVICES"}))
    monkeypatch.setenv("SRTSLURM_CONFIG", str(cluster_config))
    show_config_details(_make_config())
    assert "GPU subset visibility variable: ROCR_VISIBLE_DEVICES" in capsys.readouterr().out


class TestDryRunDynamoMetrics:
    @pytest.mark.parametrize(
        ("settings", "expected", "excluded"),
        [
            ({}, "--publish-metrics", "--publish-events-and-metrics"),
            ({"publish_events_and_metrics": None}, "--publish-metrics", "--publish-events-and-metrics"),
            ({"publish_events_and_metrics": False}, "No publication flag", "--publish-"),
            (
                {"publish_metrics": True, "publish_events_and_metrics": False},
                "No publication flag",
                "--publish-",
            ),
            ({"publish_metrics": False}, "No publication flag", "--publish-metrics"),
            (
                {"publish_metrics": False, "publish_events_and_metrics": True},
                "--publish-events-and-metrics",
                "--publish-metrics",
            ),
        ],
    )
    def test_selected_flag_is_visible(self, capsys, settings, expected, excluded):
        config = _make_config({"backend": {"type": "trtllm", **settings}, "frontend": {"type": "dynamo"}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Dynamo TRT-LLM Metrics" in output
        assert expected in output
        assert excluded not in output

    @pytest.mark.parametrize("frontend", ["trtllm_serve", "dynamo"])
    def test_unrelated_workers_have_no_dynamo_trtllm_publication_panel(self, capsys, frontend):
        backend = "trtllm" if frontend == "trtllm_serve" else "sglang"
        config = _make_config(
            {"backend": {"type": backend}, "frontend": {"type": frontend, "enable_multiple_frontends": False}}
        )
        show_config_details(config)
        assert "Dynamo TRT-LLM Metrics" not in capsys.readouterr().out

    def test_both_flags_are_visible(self, capsys):
        config = _make_config(
            {"backend": {"type": "trtllm", "publish_events_and_metrics": True}, "frontend": {"type": "dynamo"}}
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "--publish-metrics" in output
        assert "--publish-events-and-metrics" in output

    @pytest.mark.parametrize("enabled", [False, True])
    def test_explicit_combined_false_wins_over_observability(self, capsys, enabled):
        config = _make_config(
            {
                "backend": {"type": "trtllm", "publish_events_and_metrics": False},
                "frontend": {"type": "dynamo"},
                "observability": {"enabled": enabled},
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "No publication flag" in output
        assert "backend.publish_events_and_metrics: false" in output
        assert "--publish-" not in output

    def test_sidecar_has_no_dynamo_trtllm_publication_panel(self, capsys):
        config = _make_config(
            {
                "backend": {"type": "trtllm", "trtllm_config": {"aggregated": {"max_seq_len": 8192}}},
                "frontend": {"type": "dynamo"},
                "dynamo": {"sidecar": True},
                "resources": {
                    "prefill_nodes": 0,
                    "decode_nodes": 0,
                    "prefill_workers": 0,
                    "decode_workers": 0,
                    "agg_nodes": 1,
                    "agg_workers": 1,
                },
            }
        )
        show_config_details(config)
        assert "Dynamo TRT-LLM Metrics" not in capsys.readouterr().out


class TestDryRunTrtllmEngineStatistics:
    """The engine-yaml statistics keys srtctl defaults at load time are shown for
    every TRT-LLM backend, so a run that expects the iteration-level trtllm_*
    gauges can see before submitting that enable_iter_perf_stats is off."""

    def test_dynamo_shows_iteration_stats_off_per_mode(self, capsys):
        config = _make_config({"backend": {"type": "trtllm"}, "frontend": {"type": "dynamo"}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "TRT-LLM Engine Statistics" in output
        assert "prefill: enable_iter_perf_stats=false, return_perf_metrics=unset" in output
        assert "decode: enable_iter_perf_stats=false, return_perf_metrics=unset" in output

    def test_trtllm_serve_shows_both_defaults(self, capsys):
        config = _make_config(
            {"backend": {"type": "trtllm"}, "frontend": {"type": "trtllm_serve", "enable_multiple_frontends": False}}
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "TRT-LLM Engine Statistics" in output
        assert "prefill: enable_iter_perf_stats=false, return_perf_metrics=true" in output
        assert "decode: enable_iter_perf_stats=false, return_perf_metrics=true" in output

    def test_observability_shows_iteration_stats_on(self, capsys):
        config = _make_config(
            {"backend": {"type": "trtllm"}, "frontend": {"type": "dynamo"}, "observability": {"enabled": True}}
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "prefill: enable_iter_perf_stats=true, return_perf_metrics=true" in output

    def test_aggregated_layout_shows_the_agg_section(self, capsys):
        # ResourceConfig.is_disaggregated is `prefill_nodes is not None or
        # decode_nodes is not None`, so the agg layout needs the pair unset.
        config = _make_config(
            {
                "backend": {"type": "trtllm", "trtllm_config": {"aggregated": {"max_seq_len": 8192}}},
                "frontend": {"type": "dynamo"},
                "dynamo": {"sidecar": True},
                "resources": {
                    "prefill_nodes": None,
                    "decode_nodes": None,
                    "prefill_workers": None,
                    "decode_workers": None,
                    "agg_nodes": 1,
                    "agg_workers": 1,
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "TRT-LLM Engine Statistics" in output
        assert "agg: enable_iter_perf_stats=false" in output
        assert "prefill: enable_iter_perf_stats" not in output

    def test_non_trtllm_backend_has_no_engine_statistics_panel(self, capsys):
        config = _make_config({"backend": {"type": "sglang"}, "frontend": {"type": "dynamo"}})
        show_config_details(config)
        assert "TRT-LLM Engine Statistics" not in capsys.readouterr().out


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

    def test_nsys_profiling_details_shown(self, capsys):
        config = _make_config(
            {
                "profiling": {
                    "type": "nsys",
                    "nsys_trace": "cuda-sw,nvtx",
                    "trace_fork_before_exec": True,
                    "capture_range_end": "repeat:1:async",
                    "nsys_library_paths": ["/host/lib64", "/host/lib"],
                    "prefill": {"start_step": 1, "stop_step": 3, "worker_index": 0, "worker_rank": 0},
                    "decode": {"start_step": 2, "stop_step": 4, "capture_scope": "all"},
                }
            }
        )

        show_config_details(config)
        output = capsys.readouterr().out
        assert "cuda-sw,nvtx" in output
        assert "repeat:1:async" in output
        assert "/host/lib64:/host/lib" in output
        assert "all physical processes" in output
        assert "Execution Extensions" in output
        assert "profiling" in output

    def test_inline_worker_selection_translation_shown(self, capsys):
        config = _make_config(
            {
                "frontend": {
                    "type": "dynamo",
                    "worker_selection": {
                        "prefill": "max-kv-overlap",
                        "decode": "default",
                        "instances": [
                            {
                                "name": "max-kv-overlap",
                                "type": "dynamo-two-tier-cost-fn",
                            }
                        ],
                    },
                }
            }
        )

        show_config_details(config)
        output = capsys.readouterr().out
        assert "router_policy_config" in output
        assert "/logs/router_policy_config.yaml" in output
        assert "(auto)" in output
        assert "worker_selection" in output
        assert "max-kv-overlap" in output

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
        assert "profiling" not in output

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
        assert "tachometer" in output
        assert "binary_path" in output
        assert "tachometer-scraper" in output
        assert "storage_subdir" in output

    def test_tachometer_details_shown_without_explicit_block(self, capsys):
        """observability.enabled alone implies Tachometer; dry-run must say so."""
        config = _make_config({"observability": {"enabled": True}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "tachometer" in output
        assert "enabled" in output
        assert "storage_subdir" in output
        # Built-in exporters are part of the default and must be visible.
        assert "dcgm_exporter" in output
        assert "node_exporter" in output
        assert "process_exporter" in output
        assert ":9401" in output
        assert ":9101" in output
        assert ":9256" in output
        # The process exporter is host-native by default; dry-run must say so
        # (and name the binary make setup installs) rather than print an image.
        # Rich wraps the cell, so the two halves are asserted separately.
        assert "host binary" in output
        assert "configs/process-exporter" in output

    def test_dcgm_power_telemetry_details_shown(self, capsys):
        config = _make_config(
            {
                "benchmark": {"type": "sa-bench", "isl": 8192, "osl": 1024, "concurrencies": [4]},
                "telemetry": {
                    "enabled": True,
                    "collect_interval_ms": 1000,
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

    def test_cpu_power_exporter_details_shown(self, capsys):
        config = _make_config(
            {
                "benchmark": {"type": "sa-bench", "isl": 8192, "osl": 1024, "concurrencies": [4]},
                "telemetry": {
                    "enabled": True,
                    "collect_interval_ms": 1000,
                    "storage_subdir": "power",
                    "required": True,
                    "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401},
                    "cpu_power_exporter": {"port": 9405},
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "cpu_power_exporter" in output
        assert "9405" in output

    def test_cpu_power_exporter_details_hidden_when_telemetry_disabled(self, capsys):
        """A cpu_power_exporter block that will never actually launch must not be displayed."""
        config = _make_config(
            {
                "benchmark": {"type": "sa-bench", "isl": 8192, "osl": 1024, "concurrencies": [4]},
                "telemetry": {
                    "enabled": False,
                    "cpu_power_exporter": {"port": 9405},
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "cpu_power_exporter" not in output

    def test_cpu_power_host_collector_details_shown(self, capsys):
        config = _make_config(
            {
                "benchmark": {"type": "sa-bench", "isl": 8192, "osl": 1024, "concurrencies": [4]},
                "telemetry": {
                    "enabled": True,
                    "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401},
                    "cpu_power": {"enabled": True, "source": "acpi", "required": True},
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "cpu_power" in output
        assert "host collector" in output
        assert "<log_dir>/cpu_power" in output
        assert "required" in output

    def test_cpu_power_host_collector_details_hidden_when_disabled(self, capsys):
        config = _make_config(
            {
                "benchmark": {"type": "sa-bench", "isl": 8192, "osl": 1024, "concurrencies": [4]},
                "telemetry": {
                    "enabled": True,
                    "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401},
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "host collector" not in output

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

    def test_vllm_process_local_mooncake_map_in_dry_run(self, capsys):
        config = _make_config(
            {
                "backend": {
                    "type": "vllm",
                    "mooncake_kv_store": {
                        "device_names_by_gpu": [f"mlx5_{i}" for i in range(8)],
                        "store_config": {"global_segment_size": "150GB"},
                    },
                    "vllm_config": {
                        "prefill": {
                            "kv-transfer-config": '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
                        },
                        "decode": {
                            "kv-transfer-config": '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
                        },
                    },
                },
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "device_names_by_gpu" in output
        assert "mlx5_7" in output
        assert "mooncake_store_config_gpu" in output
        assert "process config" in output


class TestDryRunServices:
    """services: command, env, source, and readiness must be visible before submitting."""

    def test_service_command_env_and_readiness_shown(self, capsys):
        config = _make_config(
            {
                "services": [
                    {
                        "name": "thunderagent-router",
                        "command": ["python3", "-m", "dynamo.thunderagent_router", "--endpoint", "dyn://ns.comp.ep"],
                        "env": {"ROUTER_LOG_LEVEL": "debug"},
                        "readiness": {"port": 9100},
                    }
                ]
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Services:" in output
        assert "thunderagent-router" in output
        assert "dynamo.thunderagent_router" in output
        assert "ROUTER_LOG_LEVEL" in output
        assert "debug" in output
        assert "tcp/9100" in output
        assert "placement=head" in output

    def test_source_and_build_command_shown(self, capsys):
        config = _make_config(
            {
                "services": [
                    {
                        "name": "thunderagent-router",
                        "command": ["python3", "-m", "dynamo.thunderagent_router"],
                        "source": {"git": "https://github.com/ai-dynamo/dynamo", "rev": "refs/pull/14000/head"},
                        "build_command": ["bash", "-lc", "maturin develop --uv && pip install -e ."],
                    }
                ]
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "https://github.com/ai-dynamo/dynamo" in output
        assert "refs/pull/14000/head" in output
        assert "maturin develop" in output

    def test_mooncake_store_shows_type_defaults(self, capsys):
        config = _make_config(
            {
                "backend": {
                    "type": "sglang",
                    "mooncake_kv_store": {"container": "mooncake.sqsh"},
                    "sglang_config": {
                        "prefill": {"disaggregation-transfer-backend": "mooncake"},
                        "decode": {"disaggregation-transfer-backend": "mooncake"},
                    },
                },
                "services": [
                    {
                        "name": "store",
                        "type": "mooncake-store",
                        "placement": {"node": "workers"},
                        "env": {"MOONCAKE_GLOBAL_SEGMENT_SIZE": "100gb"},
                    }
                ],
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "mooncake.mooncake_store_service" in output
        assert "type=mooncake-store" in output
        assert "start=before_workers" in output
        assert "critical=true" in output
        assert "100gb" in output

    def test_no_services_omits_the_panel(self, capsys):
        # No discovery plane (static frontend), no tachometer: nothing declared, nothing implied.
        config = _make_config(
            {"frontend": {"type": "sglang-router"}, "observability": {"tachometer": {"enabled": False}}}
        )
        show_config_details(config)
        assert "Services:" not in capsys.readouterr().out

    def test_implicit_services_are_listed_and_marked(self, capsys):
        config = _make_config({"frontend": {"type": "dynamo"}, "dynamo": {"request_plane": "nats"}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Services:" in output
        assert "etcd" in output and "nats" in output
        assert "implied by: frontend.type dynamo" in output
        assert "implied by: dynamo.request_plane nats" in output
        assert "/configs/etcd" in output
        assert "dcgm-exporter" in output and "node-exporter" in output
        assert "implied by: observability.tachometer default exporters" in output

    def test_external_service_is_shown_as_not_launched(self, capsys):
        config = _make_config(
            {
                "frontend": {"type": "dynamo"},
                "services": [{"name": "etcd", "type": "etcd", "external": "http://etcd.shared:2379"}],
            }
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "external: http://etcd.shared:2379 (not launched)" in output
        assert "implied by: frontend.type dynamo" not in output.split("nats")[0]


class TestDryRunPostEval:
    def test_post_eval_dispatch_shown(self, capsys):
        config = _make_config(
            {"post_eval": {"passthrough_env": ["EVAL_FRAMEWORK", "EVAL_SUITE"], "command": ["bash", "run.sh"]}}
        )
        show_config_details(config)
        output = capsys.readouterr().out
        assert "Post-eval dispatch:" in output
        assert "bash run.sh" in output
        assert "EVAL_FRAMEWORK, EVAL_SUITE" in output

    def test_default_post_eval_is_silent(self, capsys):
        show_config_details(_make_config())
        assert "Post-eval dispatch" not in capsys.readouterr().out


class TestDryRunDynamoSource:
    """dynamo.source: the repo, ref, and whether it is pinned must be visible before submitting."""

    def test_git_source_shown_unpinned_and_pinned(self, capsys):
        base = {"frontend": {"type": "dynamo"}}
        config = _make_config({**base, "dynamo": {"source": {"rev": "refs/pull/14000/head"}}})
        show_config_details(config)
        output = capsys.readouterr().out
        assert "https://github.com/ai-dynamo/dynamo.git @ refs/pull/14000/head" in output
        assert "resolved from rev at submit" in output

        sha = "2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a"
        config = _make_config({**base, "dynamo": {"source": {"rev": "v1.4.2", "sha": sha}}})
        show_config_details(config)
        assert f"dynamo source sha: {sha}" in capsys.readouterr().out

    def test_pypi_source_shown(self, capsys):
        config = _make_config({"frontend": {"type": "dynamo"}, "dynamo": {"source": {"pypi": "1.4.2"}}})
        show_config_details(config)
        assert "PyPI ai-dynamo==1.4.2" in capsys.readouterr().out


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
        config = _make_config({"frontend": {"type": "sglang-router"}})
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

    AGENTIC = {
        "benchmark": {"type": "custom", "command": "bash /infmax-workspace/benchmarks/multi_node/agentic_srt.sh"}
    }

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


@pytest.mark.parametrize("nsys, expected", [({}, "enabled"), ({"enabled": False}, "disabled")])
def test_observability_nsys_details(capsys, nsys, expected):
    cfg = _make_config({"observability": {"enabled": True, "nsys": nsys}, "frontend": {"type": "dynamo"}})
    show_config_details(cfg)
    output = capsys.readouterr().out
    assert "nsys" in output and expected in output
    if expected == "enabled":
        for text in (
            "NVTX (no CUDA tracing)",
            "nsys CPU sampling",
            "process-tree (every target)",
            "Dynamo frontends",
            "measured_workload",
            "after warmup",
            "1800s",
            "DYN_ENABLE_RUST_NVTX",
        ):
            assert text in output
    else:
        assert "nsys targets" not in output


def test_explicit_profiling_explains_observability_precedence(capsys):
    cfg = _make_config(
        {
            "observability": {"enabled": True},
            "profiling": {
                "type": "nsys-time",
                "delay_secs": 1,
                "duration_secs": 2,
            },
        }
    )
    show_config_details(cfg)
    output = capsys.readouterr().out
    assert "superseded by profiling" in output
    assert "nsys targets" not in output
