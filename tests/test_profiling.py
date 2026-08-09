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
        """Test nsys profiling with custom extra_nsys_args."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(
            type="nsys",
            extra_nsys_args=["--stats=true", "--trace=osrt"],
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
        """TRTLLM nsys wrap should honor extra_nsys_args (same ordering as default path: before -o)."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(
            type="nsys",
            extra_nsys_args=["--stats=true"],
        )
        prefix = profiling.get_nsys_prefix("/out/rank", backend_type="trtllm")
        assert "--stats=true" in prefix
        assert prefix.index("--stats=true") < prefix.index("-o")

    def test_nsys_time_vllm_dynamo_path(self, monkeypatch):
        """nsys-time on a non-TRTLLM backend (vllm) drives capture purely via
        --delay/--duration (no cudaProfilerApi range), and adds
        --trace-fork-before-exec for the dynamo frontend."""
        from srtctl.core.schema import ProfilingConfig

        monkeypatch.delenv("SRTCTL_NSYS_BIN", raising=False)
        profiling = ProfilingConfig(type="nsys-time", delay_secs=120, duration_secs=30)
        assert profiling.is_nsys_time is True

        prefix = profiling.get_nsys_prefix("/out/w0", frontend_type="dynamo", backend_type="vllm")
        assert prefix[0] == "nsys"
        assert "profile" in prefix
        assert "--delay" in prefix and "120" in prefix
        assert "--duration" in prefix and "30" in prefix
        # Time-based capture — no cudaProfilerApi capture-range trigger.
        assert "cudaProfilerApi" not in prefix
        assert "--capture-range-end" not in prefix
        assert "--trace-fork-before-exec=true" in prefix
        # Output file is the last token (-o <output>).
        assert prefix[-1] == "/out/w0"

        # sglangrouter / non-dynamo frontend omits the fork flag.
        prefix_router = profiling.get_nsys_prefix("/out/w0", frontend_type="sglangrouter", backend_type="vllm")
        assert "--trace-fork-before-exec=true" not in prefix_router

    def test_nsys_binary_override(self, monkeypatch):
        """SRTCTL_NSYS_BIN overrides the nsys executable across every code path."""
        from srtctl.core.schema import ProfilingConfig

        monkeypatch.setenv("SRTCTL_NSYS_BIN", "/opt/nsight/nsys")
        # default (vllm/sglang) path
        assert ProfilingConfig(type="nsys").get_nsys_prefix("/out/w0", backend_type="vllm")[0] == "/opt/nsight/nsys"
        # time-based path
        time_prefix = ProfilingConfig(type="nsys-time", delay_secs=1, duration_secs=1).get_nsys_prefix(
            "/out/w0", backend_type="vllm"
        )
        assert time_prefix[0] == "/opt/nsight/nsys"
        # trtllm path
        assert ProfilingConfig(type="nsys").get_nsys_prefix("/out/w0", backend_type="trtllm")[0] == "/opt/nsight/nsys"

    def test_manual_nsys_phase_parsing(self):
        """phases: decode loads as nsys-manual with built-in nsys flags."""
        from srtctl.core.schema import NSYS_MANUAL_DEFAULT_ARGS, ProfilingConfig

        prof = ProfilingConfig.Schema().load(
            {
                "type": "nsys-manual",
                "phases": "decode",
                "duration_secs": 7,
            }
        )

        assert prof.is_nsys is True
        assert prof.is_nsys_manual is True
        assert prof.phases == "decode"
        assert prof.manual_duration_secs == 7
        assert prof.get_nsys_prefix("/out", mode="decode")[2 : 2 + len(NSYS_MANUAL_DEFAULT_ARGS)] == list(
            NSYS_MANUAL_DEFAULT_ARGS
        )

    def test_manual_nsys_profiles_mode(self):
        """profiles_mode: phases selects prefill or decode; agg omits phases."""
        from srtctl.core.schema import ProfilingConfig

        disagg = ProfilingConfig(type="nsys-manual", phases="decode")
        assert disagg.profiles_mode("decode") is True
        assert disagg.profiles_mode("prefill") is False
        assert disagg.profiles_mode("agg") is False

        agg = ProfilingConfig(type="nsys-manual")
        assert agg.profiles_mode("agg") is True
        assert agg.profiles_mode("aggregated") is True
        assert agg.profiles_mode("decode") is False

    def test_manual_nsys_wraps_endpoint0_rank0(self):
        """nsys-manual always wraps endpoint 0 rank 0 only."""
        from srtctl.core.schema import ProfilingConfig

        prof = ProfilingConfig(type="nsys-manual", phases="decode")
        assert prof.should_wrap_process_with_nsys(endpoint_index=0, node_rank=0) is True
        assert prof.should_wrap_process_with_nsys(endpoint_index=0, node_rank=1) is False
        assert prof.should_wrap_process_with_nsys(endpoint_index=1, node_rank=0) is False

    def test_profile_ranks_wraps_rank(self):
        """profile_ranks limits which DP ranks get nsys wrapping (non-manual modes)."""
        from srtctl.core.schema import ProfilingConfig

        prof = ProfilingConfig(type="nsys", profile_ranks=(0,))
        assert prof.wraps_rank(0) is True
        assert prof.wraps_rank(1) is False
        assert ProfilingConfig(type="nsys").wraps_rank(3) is True

    def test_manual_nsys_prefix_uses_builtin_defaults(self, monkeypatch):
        """Manual mode uses NSYS_MANUAL_DEFAULT_ARGS."""
        from srtctl.core.schema import NSYS_MANUAL_DEFAULT_ARGS, ProfilingConfig

        monkeypatch.delenv("SRTCTL_NSYS_BIN", raising=False)
        prof = ProfilingConfig(type="nsys-manual", phases="decode")

        prefix = prof.get_nsys_prefix("/logs/p/decode/out", mode="decode")
        assert prefix == [
            "nsys",
            "profile",
            *NSYS_MANUAL_DEFAULT_ARGS,
            "--force-overwrite",
            "true",
            "-o",
            "/logs/p/decode/out",
        ]
        assert prof.get_nsys_prefix("/logs/p/prefill/out", mode="prefill") == []

    def test_manual_nsys_disagg_passes_validation(self):
        """Manual nsys with phases needs no start_step/stop_step to validate."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        cfg = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            profiling=ProfilingConfig(type="nsys-manual", phases="decode"),
        )
        assert cfg.profiling.is_nsys_manual is True

    def test_manual_nsys_disagg_requires_phases(self):
        """Disaggregated nsys-manual requires profiling.phases."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match="profiling.phases"):
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
                profiling=ProfilingConfig(type="nsys-manual"),
            )

    def test_manual_nsys_agg_rejects_phases(self):
        """Aggregated nsys-manual must not set profiling.phases."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match="must not set profiling.phases"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
                profiling=ProfilingConfig(type="nsys-manual", phases="decode"),
            )

    def test_manual_nsys_rejects_legacy_blocks(self):
        """nsys-manual rejects prefill/decode/aggregated blocks."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match="do not set prefill"):
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
                    type="nsys-manual",
                    phases="decode",
                    decode=ProfilingPhaseConfig(),
                ),
            )

    def test_manual_nsys_rejects_profile_ranks(self):
        """nsys-manual always profiles endpoint 0 rank 0; profile_ranks is rejected."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match="profile_ranks"):
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
                profiling=ProfilingConfig(type="nsys-manual", phases="decode", profile_ranks=(0,)),
            )

    def test_manual_nsys_duration_defaults_to_five_seconds(self):
        """duration_secs is shared with nsys-time and defaults to 5s for manual capture."""
        from srtctl.core.schema import NSYS_MANUAL_DEFAULT_DURATION_SECS, ProfilingConfig

        assert ProfilingConfig(type="nsys-manual", phases="decode").manual_duration_secs == (
            NSYS_MANUAL_DEFAULT_DURATION_SECS
        )
        assert ProfilingConfig(type="nsys-manual", phases="decode", duration_secs=30).manual_duration_secs == 30

    def test_manual_nsys_rejects_non_positive_duration(self):
        """duration_secs must be a positive number of seconds."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match="duration_secs must be > 0"):
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
                profiling=ProfilingConfig(type="nsys-manual", phases="decode", duration_secs=0),
            )

    def test_unknown_profiling_type_rejected(self):
        """A typo in profiling.type fails fast instead of silently doing nothing."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match="Unknown profiling.type 'nsys-manul'"):
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
                profiling=ProfilingConfig(type="nsys-manul", phases="decode"),
            )

    def test_manual_nsys_prefix_without_mode_is_empty(self):
        """Callers must pass mode; without it manual mode must not fall back to a
        full-run nsys command."""
        from srtctl.core.schema import ProfilingConfig

        assert ProfilingConfig(type="nsys-manual", phases="decode").get_nsys_prefix("/out") == []

    def test_nsys_with_flag_list_rejected(self):
        """Explicit nsys flags are no longer configurable per phase."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match="no longer configurable"):
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
                    type="nsys",
                    prefill=ProfilingPhaseConfig(nsys_args=()),
                    decode=ProfilingPhaseConfig(nsys_args=("-c", "cudaProfilerApi")),
                ),
            )

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

    def test_nsys_time_allowed_for_non_trtllm_backend(self):
        """nsys-time is no longer TRTLLM-only — it must validate for the default
        (non-TRTLLM) backend so vllm+dynamo can use time-based capture."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Should not raise (previously rejected with "only supported for trtllm").
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="gb200",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            profiling=ProfilingConfig(type="nsys-time", delay_secs=120, duration_secs=30),
        )
        assert config.profiling.is_nsys_time
        assert config.backend.type != "trtllm"

    def test_vllm_profiler_config_in_vllm_config_rejected(self):
        """profiler-config.* in vllm_config conflicts with the auto-injected one."""
        from marshmallow import ValidationError

        from srtctl.backends.vllm import VLLMProtocol, VLLMServerConfig
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match="profiler-config"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(
                    gpu_type="gb200",
                    prefill_nodes=1,
                    decode_nodes=1,
                    prefill_workers=1,
                    decode_workers=1,
                ),
                backend=VLLMProtocol(vllm_config=VLLMServerConfig(decode={"profiler-config.profiler": "cuda"})),
                profiling=ProfilingConfig(
                    type="nsys",
                    prefill=ProfilingPhaseConfig(start_step=0, stop_step=10),
                    decode=ProfilingPhaseConfig(start_step=5, stop_step=15),
                ),
            )

    def test_vllm_nsys_without_profiler_config_ok(self):
        """Steps live only in the profiling: block -> no conflict, validation passes."""
        from srtctl.backends.vllm import VLLMProtocol, VLLMServerConfig
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Should not raise.
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="gb200",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            backend=VLLMProtocol(vllm_config=VLLMServerConfig(decode={"tensor-parallel-size": 1})),
            profiling=ProfilingConfig(
                type="nsys",
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=10),
                decode=ProfilingPhaseConfig(start_step=5, stop_step=15),
            ),
        )
        assert config.backend.type == "vllm"


class TestVllmNsysProfilerConfig:
    """Tests for vLLM --profiler-config injection driven by the profiling: block."""

    def test_phase_iteration_properties(self):
        """start_step/stop_step map to vLLM delay_iterations/max_iterations."""
        from srtctl.core.schema import ProfilingPhaseConfig

        phase = ProfilingPhaseConfig(start_step=10, stop_step=30)
        assert phase.vllm_nsys_delay_iterations == 10
        assert phase.vllm_nsys_max_iterations == 20

        # Missing bounds -> no capture window (0/0).
        empty = ProfilingPhaseConfig()
        assert empty.vllm_nsys_delay_iterations == 0
        assert empty.vllm_nsys_max_iterations == 0

        # stop before start clamps to 0 instead of going negative.
        assert ProfilingPhaseConfig(start_step=30, stop_step=10).vllm_nsys_max_iterations == 0

    def _build_decode_cmd(self, profiling, monkeypatch, decode_cfg=None, endpoint_index=0, node_rank=0):
        from pathlib import Path
        from types import SimpleNamespace

        import srtctl.core.slurm as slurm_mod
        from srtctl.backends.vllm import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Process

        monkeypatch.setattr(slurm_mod, "get_hostname_ip", lambda node: "10.0.0.1")

        backend = VLLMProtocol(vllm_config=VLLMServerConfig(decode=decode_cfg or {"tensor-parallel-size": 1}))
        process = Process(
            node="node0",
            gpu_indices=frozenset({0}),
            sys_port=20000,
            http_port=0,
            endpoint_mode="decode",
            endpoint_index=endpoint_index,
            node_rank=node_rank,
        )
        runtime = SimpleNamespace(model_path=Path("/model"), is_hf_model=False, request_plane="nats")
        return backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=runtime,
            profiling=profiling,
        )

    @staticmethod
    def _profiler_config(cmd):
        import json

        if "--profiler-config" not in cmd:
            return None
        return json.loads(cmd[cmd.index("--profiler-config") + 1])

    def test_iteration_nsys_injects_profiler_config(self, monkeypatch):
        """type: nsys with phase steps -> vLLM engine drives cudaProfilerStart at those steps."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(type="nsys", decode=ProfilingPhaseConfig(start_step=10, stop_step=30))
        cmd = self._build_decode_cmd(profiling, monkeypatch)

        assert self._profiler_config(cmd) == {
            "profiler": "cuda",
            "delay_iterations": 10,
            "max_iterations": 20,
        }

    def test_nsys_time_does_not_inject_profiler_config(self, monkeypatch):
        """nsys-time drives capture by wall-clock --delay/--duration, not engine steps."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(type="nsys-time", delay_secs=120, duration_secs=30)
        cmd = self._build_decode_cmd(profiling, monkeypatch)

        assert self._profiler_config(cmd) is None

    def test_no_profiling_does_not_inject_profiler_config(self, monkeypatch):
        """Without a profiling config, nothing is injected."""
        cmd = self._build_decode_cmd(None, monkeypatch)
        assert self._profiler_config(cmd) is None

    def test_nsys_manual_requires_profiler_config(self, monkeypatch):
        """type: nsys-manual injects cuda profiler-config for on-demand capture."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(type="nsys-manual", phases="decode")
        cmd = self._build_decode_cmd(profiling, monkeypatch)

        assert self._profiler_config(cmd) == {"profiler": "cuda"}

    def test_unprofiled_process_stays_a_clean_baseline(self, monkeypatch):
        """Workers that nsys does not wrap must not load vLLM's profiler wrapper."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(type="nsys-manual", phases="decode")

        assert self._profiler_config(self._build_decode_cmd(profiling, monkeypatch, node_rank=1)) is None
        assert self._profiler_config(self._build_decode_cmd(profiling, monkeypatch, endpoint_index=1)) is None

    def test_nsys_without_phase_steps_does_not_inject(self, monkeypatch):
        """type: nsys but no phase config for this mode -> no engine-driven capture."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(type="nsys")  # no decode phase
        cmd = self._build_decode_cmd(profiling, monkeypatch)
        assert self._profiler_config(cmd) is None


class TestProfileDirectories:
    """srtctl pre-creates profiles/<mode> only for phases it actually captures."""

    def test_only_profiled_phase_gets_a_directory(self, tmp_path):
        import os
        import subprocess
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.backends.vllm import VLLMProtocol
        from srtctl.cli.mixins.worker_stage import WorkerStageMixin
        from srtctl.core.runtime import RuntimeContext
        from srtctl.core.schema import ModelConfig, ProfilingConfig, ResourceConfig, SrtConfig
        from srtctl.core.topology import Process

        model_path = tmp_path / "model"
        model_path.mkdir()
        container_path = tmp_path / "container.sqsh"
        container_path.touch()

        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOBID": "12345",
            "SLURM_NODELIST": "gpu-[01-03]",
            "SLURM_JOB_NUM_NODES": "3",
            "SRTCTL_SOURCE_DIR": str(Path(__file__).parent.parent),
        }

        def mock_scontrol(cmd, **kwargs):
            if cmd[0] == "scontrol" and "hostnames" in cmd:
                result = MagicMock()
                result.stdout = "gpu-01\ngpu-02\ngpu-03"
                result.returncode = 0
                return result
            raise subprocess.CalledProcessError(1, cmd)

        with (
            patch.dict(os.environ, slurm_env),
            patch("subprocess.run", mock_scontrol),
            patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        ):
            config = SrtConfig(
                name="test",
                model=ModelConfig(path=str(model_path), container=str(container_path), precision="fp8"),
                resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, prefill_nodes=1, decode_nodes=1),
                backend=VLLMProtocol(),
                profiling=ProfilingConfig(type="nsys-manual", phases="prefill"),
            )
            runtime = RuntimeContext.from_config(config, job_id="12345", log_dir_base=tmp_path)

            class MockWorkerStage(WorkerStageMixin):
                def __init__(self, config, runtime):
                    self.config = config
                    self.runtime = runtime

            worker_stage = MockWorkerStage(config, runtime)
            mock_backend = MagicMock()
            mock_backend.get_environment_for_mode.return_value = {}
            mock_backend.build_worker_command.return_value = ["echo", "test"]

            with (
                patch.object(worker_stage, "config") as mock_config,
                patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
            ):
                mock_config.backend = mock_backend
                mock_config.profiling = config.profiling
                mock_srun.return_value = MagicMock()
                for mode in ("prefill", "decode"):
                    process = Process(
                        node="gpu-01",
                        gpu_indices=frozenset(range(8)),
                        sys_port=8081,
                        http_port=30000,
                        endpoint_mode=mode,
                        endpoint_index=0,
                        node_rank=0,
                    )
                    worker_stage.start_worker(process, [process])

            profiles = runtime.log_dir / "profiles"
            assert (profiles / "prefill").is_dir()
            assert not (profiles / "decode").exists()


class TestNsysManualScript:
    """Tests for the generated nsys-manual.sh trigger script."""

    @staticmethod
    def _worker_stage(tmp_path, profiling, processes, frontend_type="dynamo"):
        from types import SimpleNamespace

        from srtctl.cli.mixins.worker_stage import WorkerStageMixin

        class MockWorkerStage(WorkerStageMixin):
            def __init__(self):
                self.config = SimpleNamespace(
                    profiling=profiling,
                    frontend=SimpleNamespace(type=frontend_type),
                )
                self.runtime = SimpleNamespace(
                    log_dir=tmp_path / "12345" / "logs",
                    network_interface="eth0",
                )
                self._processes = processes

            @property
            def backend_processes(self):
                return self._processes

        stage = MockWorkerStage()
        stage.runtime.log_dir.mkdir(parents=True, exist_ok=True)
        return stage

    @staticmethod
    def _process(node, sys_port, mode, endpoint_index, node_rank):
        from srtctl.core.topology import Process

        return Process(
            node=node,
            gpu_indices=frozenset({0}),
            sys_port=sys_port,
            http_port=0,
            endpoint_mode=mode,
            endpoint_index=endpoint_index,
            node_rank=node_rank,
        )

    def test_script_targets_endpoint0_rank0_with_fixed_duration(self, tmp_path, monkeypatch):
        """Only endpoint 0 rank 0 of the configured phase is triggered, for duration seconds."""
        import srtctl.cli.mixins.worker_stage as ws
        from srtctl.core.schema import ProfilingConfig

        monkeypatch.setattr(ws, "get_hostname_ip", lambda node, _iface=None: f"10.0.0.{node[-1]}")

        processes = [
            self._process("node1", 8001, "prefill", 0, 0),
            self._process("node2", 8002, "decode", 0, 0),
            self._process("node3", 8003, "decode", 0, 1),
            self._process("node4", 8004, "decode", 1, 0),
        ]
        stage = self._worker_stage(
            tmp_path, ProfilingConfig(type="nsys-manual", phases="decode", duration_secs=7), processes
        )
        stage._write_nsys_manual_script()

        script = (tmp_path / "12345" / "nsys-manual.sh").read_text()
        assert "DURATION=7" in script
        assert '"${1:-' not in script  # no CLI argument
        assert '  "10.0.0.2:8002"' in script
        assert '    local path="$1" verb="$2" ok="$3"' not in script  # 2-space indent, not 4
        for other in ("8001", "8003", "8004"):
            assert other not in script
        assert "/engine/start_profile" in script
        assert "/engine/stop_profile" in script

    def test_script_not_written_when_phase_has_no_workers(self, tmp_path, monkeypatch):
        """prefill phase selected but only decode workers exist -> no script."""
        import srtctl.cli.mixins.worker_stage as ws
        from srtctl.core.schema import ProfilingConfig

        monkeypatch.setattr(ws, "get_hostname_ip", lambda node, _iface=None: "10.0.0.1")

        stage = self._worker_stage(
            tmp_path,
            ProfilingConfig(type="nsys-manual", phases="prefill"),
            [self._process("node2", 8002, "decode", 0, 0)],
        )
        stage._write_nsys_manual_script()

        assert not (tmp_path / "12345" / "nsys-manual.sh").exists()


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
