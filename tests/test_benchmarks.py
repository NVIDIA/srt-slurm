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
        assert "aime" in benchmarks
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
        """Default dataset_name is 'random' when not specified."""
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
        assert cmd[-1] == ""  # empty dataset path


class TestCustomBenchmarkRunner:
    """Test custom benchmark runner."""

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


class TestAIMERunner:
    """Test AIME runner."""

    def test_get_runner(self):
        """Can get runner for AIME."""
        runner = get_runner("aime")
        assert runner.name == "AIME"
        assert "aime" in runner.script_path

    def test_validate_valid(self):
        from srtctl.benchmarks.aime import AIMERunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = AIMERunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="aime", aime_dataset="aime25", repeat=8),
        )
        errors = runner.validate_config(config)
        assert errors == []

    def test_validate_invalid_dataset(self):
        from srtctl.benchmarks.aime import AIMERunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = AIMERunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="aime", aime_dataset="math-500"),
        )
        errors = runner.validate_config(config)
        assert any("aime_dataset" in e for e in errors)

    def test_build_command(self):
        from unittest.mock import MagicMock

        from srtctl.benchmarks.aime import AIMERunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = AIMERunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/models/qwen", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="aime",
                aime_dataset="aime24",
                num_examples=10,
                max_tokens=4096,
                repeat=4,
                num_threads=8,
                temperature=0.6,
                top_p=0.95,
            ),
        )

        cmd = runner.build_command(config, runtime)
        assert cmd == [
            "bash",
            "/srtctl-benchmarks/aime/bench.sh",
            "http://localhost:8000",
            "qwen",
            "aime24",
            "10",
            "4096",
            "8",
            "4",
            "0.6",
            "0.95",
            "",
        ]


class TestScriptsExist:
    """Test that benchmark scripts exist."""

    def test_scripts_dir_exists(self):
        """Scripts directory exists."""
        assert SCRIPTS_DIR.exists()

    def test_sa_bench_script_exists(self):
        """SA-Bench script exists."""
        script = SCRIPTS_DIR / "sa-bench" / "bench.sh"
        assert script.exists()

    def test_mmlu_script_exists(self):
        """MMLU script exists."""
        script = SCRIPTS_DIR / "mmlu" / "bench.sh"
        assert script.exists()

    def test_aime_script_exists(self):
        """AIME script exists."""
        script = SCRIPTS_DIR / "aime" / "bench.sh"
        assert script.exists()

    def test_gsm8k_script_exists(self):
        """GSM8K script exists."""
        script = SCRIPTS_DIR / "gsm8k" / "bench.sh"
        assert script.exists()


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
