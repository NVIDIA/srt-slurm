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
        assert "sa-bench" in benchmarks
        assert "sglang-bench" in benchmarks
        assert "mmlu" in benchmarks
        assert "gpqa" in benchmarks
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
