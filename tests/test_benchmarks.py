# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark runners."""

import pytest

from srtctl.benchmarks import get_runner, list_benchmarks
from srtctl.benchmarks.base import SCRIPTS_DIR


class TestBenchmarkRegistry:
    """Test benchmark runner registry."""

    def test_list_benchmarks(self):
        """All expected benchmarks are registered."""
        benchmarks = list_benchmarks()
        assert "custom" in benchmarks
        assert "sa-bench" in benchmarks
        assert "sglang-bench" in benchmarks
        assert "mmlu" in benchmarks
        assert "gpqa" in benchmarks
        assert "gsm8k" in benchmarks
        assert "longbenchv2" in benchmarks
        assert "router" in benchmarks

    def test_get_runner_valid(self):
        """Can get runner for valid benchmark type."""
        runner = get_runner("sa-bench")
        assert runner.name == "SA-Bench"
        assert "sa-bench" in runner.script_path

    def test_get_runner_invalid(self):
        """Raises ValueError for unknown benchmark type."""
        with pytest.raises(ValueError, match="Unknown benchmark type"):
            get_runner("nonexistent-benchmark")


class TestSABenchRunner:
    """Test SA-Bench runner."""

    def test_validate_config_missing_isl(self):
        """Validates that isl is required."""
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench", osl=1024, concurrencies="4x8"),
        )
        errors = runner.validate_config(config)
        assert any("isl" in e for e in errors)

    def test_validate_config_valid(self):
        """Valid config passes validation."""
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench", isl=1024, osl=1024, concurrencies="4x8"),
        )
        errors = runner.validate_config(config)
        assert errors == []

    def test_validate_custom_dataset_requires_path(self):
        """Custom dataset requires dataset_path."""
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench", dataset_name="custom", concurrencies="4x8"),
        )
        errors = runner.validate_config(config)
        assert any("dataset_path" in e for e in errors)
        assert not any("isl" in e for e in errors)

    def test_validate_custom_dataset_valid(self):
        """Custom dataset with path passes validation (isl/osl not required)."""
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench", dataset_name="custom", dataset_path="/data/bench.jsonl", concurrencies="4x8"
            ),
        )
        errors = runner.validate_config(config)
        assert errors == []

    def test_build_command_custom_dataset(self):
        """build_command passes dataset_path through as container path."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.model_path = "/model"
        runtime.is_hf_model = False

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench",
                dataset_name="custom",
                dataset_path="/glm5_datasets/bench.jsonl",
                concurrencies="4x8",
            ),
        )
        cmd = runner.build_command(config, runtime)
        assert "custom" in cmd
        assert "/glm5_datasets/bench.jsonl" in cmd

    def test_build_command_default_dataset_random(self):
        """Default dataset and HTTP lifecycle preserve legacy behavior."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.model_path = "/model"
        runtime.is_hf_model = False

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench", isl=1024, osl=128, concurrencies="4x8"),
        )
        cmd = runner.build_command(config, runtime)
        assert "random" in cmd
        assert cmd[-2] == ""  # empty dataset path
        assert cmd[-1] == "false"  # per-request HTTP sessions by default

    def test_build_command_enables_http_connection_reuse(self):
        """Explicit opt-in is appended without shifting existing arguments."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        runtime = MagicMock(frontend_port=8000, model_path="/model", is_hf_model=False)
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench",
                isl=1024,
                osl=128,
                concurrencies="4x8",
                dataset_path="/data/bench.jsonl",
                reuse_http_connections=True,
            ),
        )

        cmd = runner.build_command(config, runtime)

        assert cmd[-2] == "/data/bench.jsonl"
        assert cmd[-1] == "true"

    def test_http_connection_reuse_schema_default_and_roundtrip(self):
        """The YAML field is typed and remains opt-in when omitted."""
        from srtctl.core.schema import BenchmarkConfig, SrtConfig

        assert BenchmarkConfig().reuse_http_connections is False

        raw = {
            "name": "test",
            "model": {"path": "/model", "container": "/image", "precision": "fp4"},
            "resources": {"gpu_type": "h100"},
            "benchmark": {
                "type": "sa-bench",
                "isl": 1024,
                "osl": 128,
                "concurrencies": [4, 8],
            },
        }

        default_config = SrtConfig.Schema().load(raw)
        default_dump = SrtConfig.Schema().dump(default_config)

        assert default_config.benchmark.reuse_http_connections is False
        assert default_dump["benchmark"]["reuse_http_connections"] is False

        raw["benchmark"]["reuse_http_connections"] = True
        config = SrtConfig.Schema().load(raw)
        dumped = SrtConfig.Schema().dump(config)

        assert config.benchmark.reuse_http_connections is True
        assert dumped["benchmark"]["reuse_http_connections"] is True


class TestCustomBenchmarkRunner:
    """Test custom benchmark runner."""

    @staticmethod
    def _benchmark_stage(
        frontend_type,
        processes,
        *,
        benchmark_type="custom",
        prefill_environment=None,
        aggregated_environment=None,
    ):
        from types import SimpleNamespace

        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin

        class Stage(BenchmarkStageMixin):
            @property
            def backend_processes(self):
                return processes

        stage = Stage()
        stage.config = SimpleNamespace(
            benchmark=SimpleNamespace(type=benchmark_type, aiperf_package=None),
            backend=SimpleNamespace(
                prefill_environment=prefill_environment or {},
                aggregated_environment=aggregated_environment or {},
            ),
            frontend=SimpleNamespace(type=frontend_type),
            profiling=SimpleNamespace(enabled=False),
        )
        stage.runtime = SimpleNamespace(
            environment={},
            frontend_port=8000,
            network_interface="ibp1s0",
            nodes=SimpleNamespace(head="head-node"),
        )
        return stage

    def test_validate_config_requires_command(self):
        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="custom"),
        )
        errors = runner.validate_config(config)
        assert errors == ["benchmark.command is required for benchmark.type=custom"]

    def test_build_command_uses_custom_container_and_env(self):
        from unittest.mock import MagicMock

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = CustomBenchmarkRunner()
        runtime = MagicMock()
        runtime.container_image = "/default-image"

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command="python /bench/run.py --foo bar",
                container_image="nvcr.io/nvidia/python:3.11",
                env={"FOO": "bar"},
            ),
        )

        assert runner.build_command(config, runtime) == ["bash", "-lc", "python /bench/run.py --foo bar"]
        assert runner.get_container_image(config, runtime) == "nvcr.io/nvidia/python:3.11"
        assert runner.get_environment(config, runtime) == {"FOO": "bar"}

    def test_disaggregated_worker_endpoints_use_logical_leaders(self):
        from unittest.mock import patch

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.topology import Process

        processes = [
            Process("node-a", frozenset(range(4)), 7500, 6100, "prefill", 0, node_rank=0),
            Process("node-b", frozenset(range(4)), 7501, 0, "prefill", 0, node_rank=1),
            Process("node-c", frozenset(range(4)), 7502, 6100, "prefill", 1, node_rank=0),
            Process("node-d", frozenset(range(4)), 7503, 0, "prefill", 1, node_rank=1),
            Process("node-e", frozenset(range(4)), 7504, 6100, "decode", 0, node_rank=0),
            Process("node-f", frozenset(range(4)), 7505, 0, "decode", 0, node_rank=1),
        ]
        stage = self._benchmark_stage("dynamo", processes)

        with patch(
            "srtctl.cli.mixins.benchmark_stage.get_hostname_ip",
            side_effect=lambda node, interface: f"ip-{node}",
        ):
            env = stage._get_benchmark_env(CustomBenchmarkRunner())

        assert env["SRT_PREFILL_IPS"] == "ip-node-a,ip-node-c"
        assert env["SRT_PREFILL_ENDPOINTS"] == "ip-node-a:7500,ip-node-c:7502"
        assert env["SRT_DECODE_IPS"] == "ip-node-e"
        assert env["SRT_DECODE_ENDPOINTS"] == "ip-node-e:7504"
        assert "SRT_AGG_IPS" not in env
        assert env["AIPERF_SERVER_METRICS_URLS"] == (
            "http://ip-node-a:7500/metrics,http://ip-node-c:7502/metrics,http://ip-node-e:7504/metrics"
        )

    def test_aggregated_worker_endpoint_uses_http_port_without_dynamo(self):
        from unittest.mock import patch

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.topology import Process

        processes = [
            Process("node-a", frozenset(range(4)), 7500, 6100, "agg", 0, node_rank=0),
            Process("node-b", frozenset(range(4)), 7501, 0, "agg", 0, node_rank=1),
        ]
        stage = self._benchmark_stage("sglang", processes)

        with patch(
            "srtctl.cli.mixins.benchmark_stage.get_hostname_ip",
            side_effect=lambda node, interface: f"ip-{node}",
        ):
            env = stage._get_benchmark_env(CustomBenchmarkRunner())

        assert env["SRT_AGG_IPS"] == "ip-node-a"
        assert env["SRT_AGG_ENDPOINTS"] == "ip-node-a:6100"
        assert "SRT_PREFILL_ENDPOINTS" not in env
        assert "SRT_DECODE_ENDPOINTS" not in env
        assert env["AIPERF_SERVER_METRICS_URLS"] == "http://ip-node-a:6100/metrics"

    def test_worker_endpoint_order_keeps_colocated_logical_workers_aligned(self):
        from unittest.mock import patch

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.topology import Process

        processes = [
            Process("node-a", frozenset({0, 1}), 7500, 6100, "decode", 0),
            Process("node-a", frozenset({2, 3}), 7501, 6132, "decode", 1),
        ]
        stage = self._benchmark_stage("dynamo", processes)

        with patch(
            "srtctl.cli.mixins.benchmark_stage.get_hostname_ip",
            side_effect=lambda node, interface: "10.0.0.1",
        ):
            env = stage._get_benchmark_env(CustomBenchmarkRunner())

        assert env["SRT_DECODE_IPS"] == "10.0.0.1,10.0.0.1"
        assert env["SRT_DECODE_ENDPOINTS"] == "10.0.0.1:7500,10.0.0.1:7501"

    def test_builtin_aiperf_retains_physical_process_metrics(self):
        from unittest.mock import patch

        from srtctl.benchmarks.trace_replay import TraceReplayRunner
        from srtctl.core.topology import Process

        processes = [
            Process("node-a", frozenset(range(4)), 7500, 6100, "prefill", 0, node_rank=0),
            Process("node-b", frozenset(range(4)), 7501, 0, "prefill", 0, node_rank=1),
            Process("node-c", frozenset(range(4)), 7502, 6100, "decode", 0, node_rank=0),
        ]
        stage = self._benchmark_stage("dynamo", processes, benchmark_type="trace-replay")

        with patch(
            "srtctl.cli.mixins.benchmark_stage.get_hostname_ip",
            side_effect=lambda node, interface: f"ip-{node}",
        ):
            env = stage._get_benchmark_env(TraceReplayRunner())

        assert env["AIPERF_SERVER_METRICS_URLS"] == (
            "http://ip-node-a:7500/metrics,http://ip-node-b:7501/metrics,http://ip-node-c:7502/metrics"
        )


class TestSGLangBenchRunner:
    """Test SGLang-Bench runner."""

    def test_validate_config_valid(self):
        from srtctl.benchmarks.sglang_bench import SGLangBenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SGLangBenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sglang-bench", isl=1024, osl=1024, concurrencies="4x8", req_rate="inf"),
        )
        errors = runner.validate_config(config)
        assert errors == []

    def test_validate_config_missing_fields(self):
        from srtctl.benchmarks.sglang_bench import SGLangBenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SGLangBenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sglang-bench"),
        )
        errors = runner.validate_config(config)
        assert any("benchmark.isl is required" in e for e in errors)
        assert any("benchmark.osl is required" in e for e in errors)
        assert any("benchmark.concurrencies is required" in e for e in errors)

    def test_validate_config_rejects_zero_values(self):
        from srtctl.benchmarks.sglang_bench import SGLangBenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SGLangBenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sglang-bench", isl=0, osl=1, concurrencies=[0], req_rate=0),
        )
        errors = runner.validate_config(config)
        assert any("benchmark.isl must be a positive integer" in e for e in errors)
        assert any("benchmark.concurrencies values must be positive integers" in e for e in errors)
        assert any(
            "benchmark.req_rate must be a positive integer" in e or "benchmark.req_rate must be a positive number" in e
            for e in errors
        )

    def test_build_command(self):
        from unittest.mock import MagicMock

        from srtctl.benchmarks.sglang_bench import SGLangBenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SGLangBenchRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sglang-bench", isl=1024, osl=128, concurrencies=[1, 2]),
        )

        cmd = runner.build_command(config, runtime)
        assert cmd == [
            "bash",
            "/srtctl-benchmarks/sglang-bench/bench.sh",
            "http://localhost:8000",
            "1024",
            "128",
            "1x2",
            "inf",
        ]


class TestMooncakeRouterRunner:
    """Test Mooncake Router benchmark runner."""

    def test_build_command_includes_tokenizer_path(self):
        """Build command passes tokenizer path to aiperf.

        This fixes a bug where aiperf couldn't load the tokenizer because it was
        using the served model name (e.g., "Qwen/Qwen3-32B") to find the tokenizer,
        but that's not a valid HuggingFace ID or local path. The fix passes
        --tokenizer /model explicitly since the model is mounted there.
        """
        from unittest.mock import MagicMock

        from srtctl.benchmarks.mooncake_router import MooncakeRouterRunner

        runner = MooncakeRouterRunner()

        config = MagicMock()
        config.benchmark = MagicMock()
        config.benchmark.mooncake_workload = "conversation"
        config.benchmark.ttft_threshold_ms = 2000
        config.benchmark.itl_threshold_ms = 25
        config.served_model_name = "Qwen/Qwen3-32B"

        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False  # Local model mounted at /model

        cmd = runner.build_command(config, runtime)

        # Command: bash, script, endpoint, model_name, workload, ttft, itl, tokenizer_path
        assert cmd[7] == "/model"  # tokenizer path


class TestTraceReplayRunner:
    """Test Trace Replay benchmark runner."""

    def test_in_registry(self):
        """trace-replay is registered in benchmark list."""
        benchmarks = list_benchmarks()
        assert "trace-replay" in benchmarks

    def test_get_runner(self):
        """Can get runner for trace-replay."""
        runner = get_runner("trace-replay")
        assert runner.name == "Trace-Replay-Bench"
        assert "trace-replay" in runner.script_path

    def test_validate_missing_trace_file(self):
        """Validates that trace_file is required."""
        from srtctl.benchmarks.trace_replay import TraceReplayRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = TraceReplayRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            benchmark=BenchmarkConfig(type="trace-replay", concurrencies=[4, 8]),
        )
        errors = runner.validate_config(config)
        assert any("trace_file" in e for e in errors)

    def test_validate_missing_concurrencies(self):
        """Validates that concurrencies is required."""
        from srtctl.benchmarks.trace_replay import TraceReplayRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = TraceReplayRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            benchmark=BenchmarkConfig(type="trace-replay", trace_file="/traces/dataset.jsonl"),
        )
        errors = runner.validate_config(config)
        assert any("concurrencies" in e for e in errors)

    def test_validate_valid(self):
        """Valid config passes validation."""
        from srtctl.benchmarks.trace_replay import TraceReplayRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = TraceReplayRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            benchmark=BenchmarkConfig(
                type="trace-replay",
                trace_file="/traces/dataset.jsonl",
                concurrencies=[4, 8],
            ),
        )
        errors = runner.validate_config(config)
        assert errors == []

    def test_build_command(self):
        """Build command includes all expected arguments."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.trace_replay import TraceReplayRunner

        runner = TraceReplayRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False

        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model/kimi-k25", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            benchmark=BenchmarkConfig(
                type="trace-replay",
                trace_file="/traces/dataset.jsonl",
                concurrencies=[4, 8],
                ttft_threshold_ms=3000,
                itl_threshold_ms=7,
            ),
        )

        cmd = runner.build_command(config, runtime)

        assert cmd[0] == "bash"
        assert "trace-replay" in cmd[1]
        assert cmd[2] == "http://localhost:8000"  # endpoint
        assert cmd[3] == "kimi-k25"  # model name (from path)
        assert cmd[4] == "/traces/dataset.jsonl"  # trace file
        assert cmd[5] == "4,8"  # concurrencies
        assert cmd[6] == "3000"  # ttft threshold
        assert cmd[7] == "7"  # itl threshold
        assert cmd[8] == "/model"  # tokenizer path (local model)

    def test_build_command_default_thresholds(self):
        """Build command uses default thresholds when not specified."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.trace_replay import TraceReplayRunner

        runner = TraceReplayRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False

        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model/test", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            benchmark=BenchmarkConfig(
                type="trace-replay",
                trace_file="/traces/dataset.jsonl",
                concurrencies=[1],
            ),
        )

        cmd = runner.build_command(config, runtime)
        assert cmd[6] == "2000"  # default ttft
        assert cmd[7] == "25"  # default itl

    def test_build_command_with_aiperf_args(self):
        """aiperf_args are passed through as CLI flags."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.trace_replay import TraceReplayRunner

        runner = TraceReplayRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False

        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model/kimi", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            benchmark=BenchmarkConfig(
                type="trace-replay",
                trace_file="/traces/dataset.jsonl",
                concurrencies=[4],
                aiperf_args={
                    "benchmark-duration": 600,
                    "workers-max": 200,
                    "request-timeout-seconds": 1200,
                    "profile-export-level": "raw",
                },
            ),
        )

        cmd = runner.build_command(config, runtime)

        # Positional args come first (9 of them)
        assert cmd[8] == "/model"  # tokenizer path

        # aiperf_args appended after positional args
        extra = cmd[9:]
        assert "--benchmark-duration" in extra
        assert extra[extra.index("--benchmark-duration") + 1] == "600"
        assert "--workers-max" in extra
        assert extra[extra.index("--workers-max") + 1] == "200"
        assert "--request-timeout-seconds" in extra
        assert "--profile-export-level" in extra
        assert extra[extra.index("--profile-export-level") + 1] == "raw"

    def test_build_command_aiperf_args_bool(self):
        """Boolean aiperf_args are passed as flags without values."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.trace_replay import TraceReplayRunner

        runner = TraceReplayRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False

        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model/test", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            benchmark=BenchmarkConfig(
                type="trace-replay",
                trace_file="/traces/dataset.jsonl",
                concurrencies=[1],
                aiperf_args={"export-http-trace": True, "disabled-flag": False},
            ),
        )

        cmd = runner.build_command(config, runtime)
        extra = cmd[9:]
        assert "--export-http-trace" in extra
        assert "--disabled-flag" not in extra

    def test_config_roundtrip(self):
        """Config with trace-replay loads correctly from YAML."""
        import tempfile
        from pathlib import Path

        import yaml

        from srtctl.core.schema import SrtConfig

        config_data = {
            "name": "trace-test",
            "model": {"path": "/model", "container": "/image", "precision": "fp4"},
            "resources": {"gpu_type": "gb200"},
            "benchmark": {
                "type": "trace-replay",
                "trace_file": "/traces/dataset.jsonl",
                "concurrencies": [4, 8],
                "ttft_threshold_ms": 3000,
                "itl_threshold_ms": 7,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            tmp_path = Path(f.name)

        config = SrtConfig.from_yaml(tmp_path)
        assert config.benchmark.type == "trace-replay"
        assert config.benchmark.trace_file == "/traces/dataset.jsonl"
        assert config.benchmark.concurrencies == [4, 8]
        assert config.benchmark.ttft_threshold_ms == 3000
        assert config.benchmark.itl_threshold_ms == 7


class TestLMEvalRunner:
    """Test LM-Eval runner."""

    def test_registry_includes_lm_eval(self):
        """lm-eval is in the benchmark registry."""
        assert "lm-eval" in list_benchmarks()

    def test_get_runner(self):
        """Can get lm-eval runner."""
        runner = get_runner("lm-eval")
        assert runner.name == "lm-eval"

    def test_script_path(self):
        """Script path points to lm-eval bench.sh."""
        runner = get_runner("lm-eval")
        assert "lm-eval/bench.sh" in runner.script_path

    def test_local_script_dir(self):
        """Local script dir points to lm-eval scripts."""
        runner = get_runner("lm-eval")
        assert runner.local_script_dir.endswith("lm-eval")

    def test_validate_config_always_valid(self):
        """lm-eval accepts any config."""
        from srtctl.benchmarks.lm_eval import LMEvalRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = LMEvalRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench"),
        )
        assert runner.validate_config(config) == []

    def test_build_command(self):
        """build_command returns correct bash command."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.lm_eval import LMEvalRunner

        runner = LMEvalRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000

        config = MagicMock()
        cmd = runner.build_command(config, runtime)
        assert cmd == [
            "bash",
            "/srtctl-benchmarks/lm-eval/bench.sh",
            "http://localhost:8000",
            "/infmax-workspace",
        ]


class TestGSM8KRunner:
    """Test the unified GSM8K runner (backend auto-detect)."""

    def _sglang_config(self, **benchmark_kwargs):
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        # Default backend is SGLangProtocol.
        return SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            benchmark=BenchmarkConfig(type="gsm8k", **benchmark_kwargs),
        )

    def _vllm_config(self, served_model_name="Qwen3.5-397B-A17B-NVFP4", **benchmark_kwargs):
        from srtctl.backends.vllm import VLLMProtocol, VLLMServerConfig
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        return SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            backend=VLLMProtocol(vllm_config=VLLMServerConfig(decode={"served-model-name": served_model_name})),
            benchmark=BenchmarkConfig(type="gsm8k", **benchmark_kwargs),
        )

    def test_registered(self):
        """gsm8k resolves to the unified runner."""
        runner = get_runner("gsm8k")
        assert runner.name == "GSM8K"

    def test_sglang_backend_uses_sglang_harness(self):
        """Non-vLLM backends run the sglang harness."""
        from unittest.mock import MagicMock

        runner = get_runner("gsm8k")
        runtime = MagicMock()
        runtime.frontend_port = 8000

        cmd = runner.build_command(self._sglang_config(), runtime)
        assert cmd == [
            "bash",
            "/srtctl-benchmarks/gsm8k/sglang-bench.sh",
            "http://localhost:8000",
            "1319",
            "16384",
            "512",
            "5",
            "",
            "",
            "",
        ]

    def test_vllm_backend_uses_vllm_eval(self):
        """vLLM backend runs the vendored vLLM eval and sends the served model name."""
        from unittest.mock import MagicMock

        runner = get_runner("gsm8k")
        runtime = MagicMock()
        runtime.frontend_port = 8000

        cmd = runner.build_command(self._vllm_config(), runtime)
        assert cmd == [
            "bash",
            "/srtctl-benchmarks/gsm8k/vllm-bench.sh",
            "http://localhost",
            "8000",
            "Qwen3.5-397B-A17B-NVFP4",
            "1319",
            "256",
            "5",
            "0.0",
            "1",
        ]

    def test_vllm_backend_overrides_and_repeat(self):
        """Config fields override defaults and repeat is passed through on the vLLM path."""
        from unittest.mock import MagicMock

        runner = get_runner("gsm8k")
        runtime = MagicMock()
        runtime.frontend_port = 9001

        cmd = runner.build_command(
            self._vllm_config(num_examples=100, max_tokens=2048, num_shots=8, temperature=0.6, repeat=5),
            runtime,
        )
        assert cmd == [
            "bash",
            "/srtctl-benchmarks/gsm8k/vllm-bench.sh",
            "http://localhost",
            "9001",
            "Qwen3.5-397B-A17B-NVFP4",
            "100",
            "2048",
            "8",
            "0.6",
            "5",
        ]

    def test_validate_config_rejects_nonpositive(self):
        """Non-positive numeric knobs are rejected."""
        runner = get_runner("gsm8k")
        errors = runner.validate_config(self._vllm_config(num_examples=0, repeat=-1, num_shots=-1))
        assert any("benchmark.num_examples must be > 0" in e for e in errors)
        assert any("benchmark.repeat must be > 0" in e for e in errors)
        assert any("benchmark.num_shots must be >= 0" in e for e in errors)


class TestScriptsExist:
    """Test that benchmark scripts exist."""

    def test_scripts_dir_exists(self):
        """Scripts directory exists."""
        assert SCRIPTS_DIR.exists()

    def test_sa_bench_script_exists(self):
        """SA-Bench script exists."""
        script = SCRIPTS_DIR / "sa-bench" / "bench.sh"
        assert script.exists()

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [(None, ["false", "false"]), ("false", ["false", "false"]), ("true", ["true", "true"])],
    )
    def test_sa_bench_http_reuse_flag_reaches_warmup_and_formal(self, mode, expected):
        """The optional shell argument controls both benchmark processes."""
        import subprocess

        script = SCRIPTS_DIR / "sa-bench" / "bench.sh"
        args = [
            "http://localhost:8000",
            "1",
            "1",
            "2",
            "inf",
            "/model",
            "model",
            "false",
            "1",
            "0",
            "0",
            "0.8",
            "1",
            "1",
            "",
            "false",
            "random",
            "",
        ]
        if mode is not None:
            args.append(mode)

        result = subprocess.run(
            [
                "bash",
                "-c",
                r"""
script=$1
shift
python3() {
    if [ "${1:-}" != "-u" ]; then
        return 0
    fi
    local reuse=false arg
    for arg in "$@"; do
        if [ "$arg" = "--reuse-http-connections" ]; then
            reuse=true
        fi
    done
    printf 'BENCHMARK_CALL reuse=%s\n' "$reuse"
}
curl() { return 0; }
mkdir() { return 0; }
source "$script" "$@"
""",
                "_",
                str(script),
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        calls = [
            line.removeprefix("BENCHMARK_CALL reuse=")
            for line in result.stdout.splitlines()
            if line.startswith("BENCHMARK_CALL reuse=")
        ]
        assert calls == expected

    def test_mmlu_script_exists(self):
        """MMLU script exists."""
        script = SCRIPTS_DIR / "mmlu" / "bench.sh"
        assert script.exists()

    def test_gsm8k_scripts_exist(self):
        """Both gsm8k harness wrappers and the bundled vLLM eval script exist."""
        assert (SCRIPTS_DIR / "gsm8k" / "sglang-bench.sh").exists()
        assert (SCRIPTS_DIR / "gsm8k" / "vllm-bench.sh").exists()
        assert (SCRIPTS_DIR / "gsm8k" / "gsm8k_eval.py").exists()


class TestCustomDatasetLoader:
    """Test benchmark_dataset.py custom JSONL loader."""

    def test_trtllm_format(self, tmp_path):
        """Loads TRT-LLM OpenAI-style JSONL."""
        import sys

        scripts_dir = str(SCRIPTS_DIR / "sa-bench")
        sys.path.insert(0, scripts_dir)
        try:
            from benchmark_dataset import sample_custom_requests
        finally:
            sys.path.pop(0)

        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text(
            '{"input": {"messages": [{"role": "user", "content": "Hello world"}], "max_tokens": 64}}\n'
            '{"input": {"messages": [{"role": "user", "content": "How are you?"}], "max_tokens": 128}}\n'
        )

        results = sample_custom_requests(str(dataset_file), num_requests=10)
        assert len(results) == 2
        assert all(len(r) == 4 for r in results)
        assert results[0][3] is None

    def test_flat_format(self, tmp_path):
        """Loads flat prompt/output_len JSONL."""
        import sys

        scripts_dir = str(SCRIPTS_DIR / "sa-bench")
        sys.path.insert(0, scripts_dir)
        try:
            from benchmark_dataset import sample_custom_requests
        finally:
            sys.path.pop(0)

        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text(
            '{"prompt": "Summarize this article", "expected_output_len": 256}\n'
            '{"prompt": "Translate to French", "max_tokens": 100}\n'
        )

        results = sample_custom_requests(str(dataset_file), num_requests=10)
        assert len(results) == 2
        output_lens = {r[2] for r in results}
        assert 256 in output_lens
        assert 100 in output_lens

    def test_num_requests_limit(self, tmp_path):
        """Respects num_requests cap."""
        import sys

        scripts_dir = str(SCRIPTS_DIR / "sa-bench")
        sys.path.insert(0, scripts_dir)
        try:
            from benchmark_dataset import sample_custom_requests
        finally:
            sys.path.pop(0)

        lines = [f'{{"prompt": "request {i}", "expected_output_len": 64}}\n' for i in range(50)]
        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text("".join(lines))

        results = sample_custom_requests(str(dataset_file), num_requests=5)
        assert len(results) == 5

    def test_precomputed_token_lengths(self, tmp_path):
        """Uses precomputed num_tokens when available in TRT-LLM format."""
        import sys

        scripts_dir = str(SCRIPTS_DIR / "sa-bench")
        sys.path.insert(0, scripts_dir)
        try:
            from benchmark_dataset import sample_custom_requests
        finally:
            sys.path.pop(0)

        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text(
            '{"input": {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64, "num_tokens": 42}}\n'
        )

        results = sample_custom_requests(str(dataset_file), num_requests=10)
        assert len(results) == 1
        assert results[0][1] == 42

    def test_prompt_len_estimated_when_missing(self, tmp_path):
        """Estimates prompt_len from text length when not provided."""
        import sys

        scripts_dir = str(SCRIPTS_DIR / "sa-bench")
        sys.path.insert(0, scripts_dir)
        try:
            from benchmark_dataset import sample_custom_requests
        finally:
            sys.path.pop(0)

        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text('{"prompt": "abcdefghijklmnop", "expected_output_len": 64}\n')

        results = sample_custom_requests(str(dataset_file), num_requests=10)
        assert len(results) == 1
        assert results[0][1] == 4  # len("abcdefghijklmnop") // 4

    def test_config_roundtrip_custom_dataset(self):
        """Config with custom dataset loads correctly from YAML."""
        import tempfile
        from pathlib import Path

        import yaml

        from srtctl.core.schema import SrtConfig

        config_data = {
            "name": "custom-dataset-test",
            "model": {"path": "/model", "container": "/image", "precision": "fp4"},
            "resources": {"gpu_type": "h100"},
            "benchmark": {
                "type": "sa-bench",
                "dataset_name": "custom",
                "dataset_path": "/data/my_dataset.jsonl",
                "concurrencies": [4, 8],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            tmp_path = Path(f.name)

        config = SrtConfig.from_yaml(tmp_path)
        assert config.benchmark.dataset_name == "custom"
        assert config.benchmark.dataset_path == "/data/my_dataset.jsonl"
        assert config.benchmark.concurrencies == [4, 8]


class TestRunPostEval:
    """Test SweepOrchestrator._run_post_eval method."""

    @staticmethod
    def _make_orchestrator():
        """Create a SweepOrchestrator with mocked config/runtime."""
        from pathlib import Path

        from srtctl.cli.do_sweep import SweepOrchestrator
        from srtctl.core.runtime import Nodes, RuntimeContext
        from srtctl.core.schema import (
            BenchmarkConfig,
            FrontendConfig,
            HealthCheckConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model/test-model", container="/image", precision="fp4"),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=1,
                decode_nodes=2,
                prefill_workers=1,
                decode_workers=2,
            ),
            benchmark=BenchmarkConfig(type="sa-bench", isl=1024, osl=1024, concurrencies="128x256x512"),
            health_check=HealthCheckConfig(max_attempts=3, interval_seconds=1),
            frontend=FrontendConfig(type="dynamo"),
        )
        runtime = RuntimeContext(
            job_id="12345",
            run_name="test-run",
            nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0", "node1", "node2")),
            head_node_ip="10.0.0.1",
            infra_node_ip="10.0.0.1",
            log_dir=Path("/tmp/logs"),
            model_path=Path("/model/test-model"),
            container_image=Path("/path/to/container.sqsh"),
            gpus_per_node=8,
            network_interface=None,
            container_mounts={},
            environment={},
        )
        return SweepOrchestrator(config=config, runtime=runtime)

    def test_post_benchmark_port_check_fails(self):
        """Returns 1 when port check fails in post-benchmark mode."""
        import os
        import threading
        from unittest.mock import patch

        orch = self._make_orchestrator()
        stop = threading.Event()
        with patch.dict(os.environ, {"EVAL_ONLY": "false"}, clear=False):
            with patch("srtctl.cli.do_sweep.wait_for_port", return_value=False):
                result = orch._run_post_eval(stop)
        assert result == 1

    def test_eval_only_health_check_fails(self):
        """Returns 1 when health check fails in eval-only mode."""
        import os
        import threading
        from unittest.mock import patch

        orch = self._make_orchestrator()
        stop = threading.Event()
        with patch.dict(os.environ, {"EVAL_ONLY": "true"}, clear=False):
            with patch("srtctl.core.health.wait_for_model", return_value=False):
                result = orch._run_post_eval(stop)
        assert result == 1

    def test_runner_not_available(self):
        """Returns 1 when lm-eval runner is not registered."""
        import os
        import threading
        from unittest.mock import patch

        orch = self._make_orchestrator()
        stop = threading.Event()
        with patch.dict(os.environ, {"EVAL_ONLY": "false"}, clear=False):
            with patch("srtctl.cli.do_sweep.wait_for_port", return_value=True):
                with patch("srtctl.benchmarks.get_runner", side_effect=ValueError("not found")):
                    result = orch._run_post_eval(stop)
        assert result == 1

    def test_successful_eval(self):
        """Returns 0 when eval completes successfully."""
        import os
        import threading
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()
        stop = threading.Event()

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]
        mock_proc.returncode = 0

        with patch.dict(os.environ, {"EVAL_ONLY": "false"}, clear=False):
            with patch("srtctl.cli.do_sweep.wait_for_port", return_value=True):
                with patch("srtctl.cli.do_sweep.start_srun_process", return_value=mock_proc):
                    result = orch._run_post_eval(stop)
        assert result == 0

    def test_eval_only_successful(self):
        """Returns 0 in eval-only mode when health check and eval succeed."""
        import os
        import threading
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()
        stop = threading.Event()

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]
        mock_proc.returncode = 0

        with patch.dict(os.environ, {"EVAL_ONLY": "true"}, clear=False):
            with patch("srtctl.core.health.wait_for_model", return_value=True):
                with patch("srtctl.cli.do_sweep.start_srun_process", return_value=mock_proc):
                    result = orch._run_post_eval(stop)
        assert result == 0

    def test_env_var_passthrough(self):
        """Eval env vars are passed through to srun."""
        import os
        import threading
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()
        stop = threading.Event()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0

        env_vars = {
            "EVAL_ONLY": "false",
            "RUN_EVAL": "true",
            "FRAMEWORK": "sglang",
            "PRECISION": "fp4",
            "MODEL": "test-model",
        }

        captured_kwargs = {}

        def capture_srun(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_proc

        with patch.dict(os.environ, env_vars, clear=False):
            with patch("srtctl.cli.do_sweep.wait_for_port", return_value=True):
                with patch("srtctl.cli.do_sweep.start_srun_process", side_effect=capture_srun):
                    orch._run_post_eval(stop)

        env_to_set = captured_kwargs["env_to_set"]
        assert env_to_set["RUN_EVAL"] == "true"
        assert env_to_set["FRAMEWORK"] == "sglang"
        assert env_to_set["PRECISION"] == "fp4"
        assert env_to_set["MODEL"] == "test-model"
        assert env_to_set["MODEL_NAME"] == "test-model"

    def test_eval_conc_from_env(self):
        """EVAL_CONC from env takes priority over benchmark concurrencies."""
        import os
        import threading
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()
        stop = threading.Event()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0

        captured_kwargs = {}

        def capture_srun(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_proc

        with patch.dict(os.environ, {"EVAL_ONLY": "false", "EVAL_CONC": "64"}, clear=False):
            with patch("srtctl.cli.do_sweep.wait_for_port", return_value=True):
                with patch("srtctl.cli.do_sweep.start_srun_process", side_effect=capture_srun):
                    orch._run_post_eval(stop)

        assert captured_kwargs["env_to_set"]["EVAL_CONC"] == "64"

    def test_eval_conc_fallback_to_max_concurrency(self):
        """EVAL_CONC falls back to max of benchmark concurrencies."""
        import os
        import threading
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()
        stop = threading.Event()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0

        captured_kwargs = {}

        def capture_srun(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_proc

        env = {"EVAL_ONLY": "false"}
        # Remove EVAL_CONC if present
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("EVAL_CONC", None)
            with patch("srtctl.cli.do_sweep.wait_for_port", return_value=True):
                with patch("srtctl.cli.do_sweep.start_srun_process", side_effect=capture_srun):
                    orch._run_post_eval(stop)

        # concurrencies="128x256x512", max is 512
        assert captured_kwargs["env_to_set"]["EVAL_CONC"] == "512"

    def test_stop_event_terminates_eval(self):
        """Stop event terminates the eval process."""
        import os
        import threading
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()
        stop = threading.Event()
        stop.set()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with patch.dict(os.environ, {"EVAL_ONLY": "false"}, clear=False):
            with patch("srtctl.cli.do_sweep.wait_for_port", return_value=True):
                with patch("srtctl.cli.do_sweep.start_srun_process", return_value=mock_proc):
                    result = orch._run_post_eval(stop)

        assert result == 1
        mock_proc.terminate.assert_called_once()


class TestSweepRunEvalIntegration:
    """Test eval-related branches in SweepOrchestrator.run()."""

    @staticmethod
    def _make_orchestrator():
        return TestRunPostEval._make_orchestrator()

    def test_run_eval_only_mode(self):
        """EVAL_ONLY=true skips benchmark and runs _run_post_eval."""
        import os
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()

        with patch.dict(os.environ, {"EVAL_ONLY": "true"}, clear=False):
            with patch.object(orch, "start_head_infrastructure") as mock_head:
                mock_head.return_value = MagicMock()
                with patch.object(orch, "start_all_workers", return_value={}):
                    with patch.object(orch, "start_frontend", return_value=[]):
                        with patch.object(orch, "_run_post_eval", return_value=0) as mock_eval:
                            with patch.object(orch, "run_benchmark") as mock_bench:
                                with patch.object(orch, "run_postprocess"):
                                    with patch("srtctl.cli.do_sweep.StatusReporter") as mock_reporter_cls:
                                        mock_reporter_cls.from_config.return_value = MagicMock()
                                        exit_code = orch.run()

        mock_eval.assert_called_once()
        mock_bench.assert_not_called()
        assert exit_code == 0

    def test_run_with_post_benchmark_eval(self):
        """RUN_EVAL=true runs benchmark then _run_post_eval."""
        import os
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()

        with patch.dict(os.environ, {"EVAL_ONLY": "false", "RUN_EVAL": "true"}, clear=False):
            with patch.object(orch, "start_head_infrastructure") as mock_head:
                mock_head.return_value = MagicMock()
                with patch.object(orch, "start_all_workers", return_value={}):
                    with patch.object(orch, "start_frontend", return_value=[]):
                        with patch.object(orch, "run_benchmark", return_value=0) as mock_bench:
                            with patch.object(orch, "_run_post_eval", return_value=0) as mock_eval:
                                with patch.object(orch, "run_postprocess"):
                                    with patch("srtctl.cli.do_sweep.StatusReporter") as mock_reporter_cls:
                                        mock_reporter_cls.from_config.return_value = MagicMock()
                                        exit_code = orch.run()

        mock_bench.assert_called_once()
        mock_eval.assert_called_once()
        assert exit_code == 0

    def test_run_eval_only_failure(self):
        """EVAL_ONLY=true with eval failure returns non-zero exit code."""
        import os
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()

        with patch.dict(os.environ, {"EVAL_ONLY": "true"}, clear=False):
            with patch.object(orch, "start_head_infrastructure") as mock_head:
                mock_head.return_value = MagicMock()
                with patch.object(orch, "start_all_workers", return_value={}):
                    with patch.object(orch, "start_frontend", return_value=[]):
                        with patch.object(orch, "_run_post_eval", return_value=1):
                            with patch.object(orch, "run_postprocess"):
                                with patch("srtctl.cli.do_sweep.StatusReporter") as mock_reporter_cls:
                                    mock_reporter_cls.from_config.return_value = MagicMock()
                                    exit_code = orch.run()

        assert exit_code == 1

    def test_run_post_benchmark_eval_failure_nonfatal(self):
        """RUN_EVAL=true with eval failure still returns benchmark exit code 0."""
        import os
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()

        with patch.dict(os.environ, {"EVAL_ONLY": "false", "RUN_EVAL": "true"}, clear=False):
            with patch.object(orch, "start_head_infrastructure") as mock_head:
                mock_head.return_value = MagicMock()
                with patch.object(orch, "start_all_workers", return_value={}):
                    with patch.object(orch, "start_frontend", return_value=[]):
                        with patch.object(orch, "run_benchmark", return_value=0):
                            with patch.object(orch, "_run_post_eval", return_value=1):
                                with patch.object(orch, "run_postprocess"):
                                    with patch("srtctl.cli.do_sweep.StatusReporter") as mock_reporter_cls:
                                        mock_reporter_cls.from_config.return_value = MagicMock()
                                        exit_code = orch.run()

        assert exit_code == 0

    def test_run_eval_skipped_when_benchmark_fails(self):
        """RUN_EVAL=true but benchmark fails: eval is skipped."""
        import os
        from unittest.mock import MagicMock, patch

        orch = self._make_orchestrator()

        with patch.dict(os.environ, {"EVAL_ONLY": "false", "RUN_EVAL": "true"}, clear=False):
            with patch.object(orch, "start_head_infrastructure") as mock_head:
                mock_head.return_value = MagicMock()
                with patch.object(orch, "start_all_workers", return_value={}):
                    with patch.object(orch, "start_frontend", return_value=[]):
                        with patch.object(orch, "run_benchmark", return_value=1):
                            with patch.object(orch, "_run_post_eval") as mock_eval:
                                with patch.object(orch, "run_postprocess"):
                                    with patch("srtctl.cli.do_sweep.StatusReporter") as mock_reporter_cls:
                                        mock_reporter_cls.from_config.return_value = MagicMock()
                                        exit_code = orch.run()

        mock_eval.assert_not_called()
        assert exit_code == 1
