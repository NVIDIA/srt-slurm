# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiling configuration, validation, and benchmark runner."""

import pytest

from srtctl.benchmarks import get_runner
from srtctl.benchmarks.base import SCRIPTS_DIR


class TestProfilingConfig:
    """Tests for ProfilingConfig dataclass."""

    def test_profiling_defaults(self):
        """Test profiling config defaults."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig()

        assert profiling.enabled is False
        assert profiling.is_nsys is False
        assert profiling.is_torch is False
        assert profiling.type == "none"

    def test_nsys_profiling(self):
        """Test nsys profiling configuration."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(
            type="nsys",
        )

        assert profiling.enabled is True
        assert profiling.is_nsys is True
        assert profiling.is_torch is False

        # Test nsys prefix generation
        prefix = profiling.get_nsys_prefix("/output/test")
        assert "nsys" in prefix
        assert "profile" in prefix
        assert "/output/test" in prefix

        # Dynamo frontend requires trace-fork-before-exec, sglangrouter does not.
        prefix_dynamo = profiling.get_nsys_prefix("/output/test", frontend_type="dynamo")
        assert "--trace-fork-before-exec=true" in prefix_dynamo
        prefix_router = profiling.get_nsys_prefix("/output/test", frontend_type="sglangrouter")
        assert "--trace-fork-before-exec=true" not in prefix_router

    def test_nsys_profiling_with_extra_args(self):
        """Test nsys profiling with custom nsys_args."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(
            type="nsys",
            nsys_args=["--stats=true", "--trace=osrt"],
        )

        prefix = profiling.get_nsys_prefix("/output/test")
        assert "nsys" in prefix
        assert "profile" in prefix
        assert "/output/test" in prefix
        assert "--stats=true" in prefix
        assert "--trace=osrt" in prefix
        # Extra args appear before -o output
        o_idx = prefix.index("-o")
        stats_idx = prefix.index("--stats=true")
        assert stats_idx < o_idx

    def test_nsys_trtllm_prefix_includes_extra_args(self):
        """TRTLLM nsys wrap should honor nsys_args (same ordering as default path: before -o)."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(
            type="nsys",
            nsys_args=["--stats=true"],
        )
        prefix = profiling.get_nsys_prefix("/out/rank", backend_type="trtllm")
        assert "--stats=true" in prefix
        assert prefix.index("--stats=true") < prefix.index("-o")

    def test_torch_profiling(self):
        """Test torch profiling configuration."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="torch",
            prefill=ProfilingPhaseConfig(start_step=5, stop_step=15),
            decode=ProfilingPhaseConfig(start_step=10, stop_step=20),
        )

        assert profiling.enabled is True
        assert profiling.is_torch is True
        assert profiling.is_nsys is False

        # Test env vars generation for prefill
        env = profiling.get_env_vars("prefill", "/logs/profiles")
        assert env["PROFILING_MODE"] == "prefill"
        assert env["PROFILE_TYPE"] == "torch"
        assert env["PROFILE_PREFILL_START_STEP"] == "5"
        assert env["PROFILE_PREFILL_STOP_STEP"] == "15"
        assert env["SGLANG_TORCH_PROFILER_DIR"] == "/logs/profiles/prefill"

        # Test env vars generation for decode (different steps)
        env_decode = profiling.get_env_vars("decode", "/logs/profiles")
        assert env_decode["PROFILE_DECODE_START_STEP"] == "10"
        assert env_decode["PROFILE_DECODE_STOP_STEP"] == "20"

    def test_aggregated_profiling(self):
        """Test aggregated profiling configuration."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="torch",
            aggregated=ProfilingPhaseConfig(start_step=0, stop_step=100),
        )

        env = profiling.get_env_vars("agg", "/logs/profiles")
        assert env["PROFILE_TYPE"] == "torch"
        assert env["PROFILE_AGG_START_STEP"] == "0"
        assert env["PROFILE_AGG_STOP_STEP"] == "100"


class TestProfilingValidation:
    """Tests for profiling config validation in SrtConfig."""

    def test_disagg_requires_prefill_and_decode(self):
        """Disaggregated mode requires both prefill and decode profiling configs."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Missing decode config should fail (with valid single worker config)
        with pytest.raises(ValidationError, match="both profiling.prefill and profiling.decode"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(
                    gpu_type="h100",
                    prefill_nodes=1,
                    decode_nodes=1,
                    prefill_workers=1,
                    decode_workers=1,
                ),
                profiling=ProfilingConfig(
                    type="torch",
                    prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                    # Missing decode config
                ),
            )

    def test_agg_requires_aggregated_config(self):
        """Aggregated mode requires aggregated profiling config."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Aggregated mode without aggregated profiling config should fail
        with pytest.raises(ValidationError, match="profiling.aggregated to be set"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
                profiling=ProfilingConfig(
                    type="torch",
                    # Missing aggregated config
                ),
            )

    def test_profiling_allows_multiple_workers_disagg(self):
        """Profiling in disaggregated mode supports multiple workers."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Should not raise
        SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=2,
                decode_workers=3,
            ),
            profiling=ProfilingConfig(
                type="torch",
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )

    def test_profiling_allows_multiple_workers_agg(self):
        """Profiling in aggregated mode supports multiple workers."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Should not raise
        SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                agg_nodes=2,
                agg_workers=2,
            ),
            profiling=ProfilingConfig(
                type="torch",
                aggregated=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )

    def test_valid_profiling_config_disagg(self):
        """Valid profiling config with 1P + 1D passes validation."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Should not raise
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            profiling=ProfilingConfig(
                type="torch",
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )
        assert config.profiling.enabled


class TestProfilingIntegration:
    """Integration tests for profiling + benchmarks."""

    def test_no_profiling_benchmark_runner(self):
        """There is no dedicated 'profiling' benchmark runner anymore."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_runner("profiling")

    def test_profiling_does_not_override_benchmark_type(self):
        """Profiling is orthogonal to benchmark selection."""
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # User sets benchmark.type to "sa-bench" and has profiling enabled.
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            benchmark=BenchmarkConfig(type="sa-bench"),
            profiling=ProfilingConfig(
                type="torch",
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )

        assert config.profiling.enabled is True
        runner = get_runner(config.benchmark.type)
        assert runner.name == "SA-Bench"
        assert (SCRIPTS_DIR / "sa-bench" / "bench.sh").exists()

    def test_sglang_bench_script_exists(self):
        assert (SCRIPTS_DIR / "sglang-bench" / "bench.sh").exists()

    def test_sglang_bench_runner_validate_config(self):
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = get_runner("sglang-bench")

        config_missing = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            benchmark=BenchmarkConfig(type="sglang-bench"),
            profiling=ProfilingConfig(
                type="torch",
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=10),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=10),
            ),
        )

        errors = runner.validate_config(config_missing)
        assert "benchmark.isl is required for sglang-bench" in errors
        assert "benchmark.osl is required for sglang-bench" in errors
        assert "benchmark.concurrencies is required for sglang-bench" in errors

    def test_sglang_bench_runner_build_command(self):
        from types import SimpleNamespace

        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = get_runner("sglang-bench")
        runtime = SimpleNamespace(frontend_port=8000)

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
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
