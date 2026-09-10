# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiling configuration, validation, and benchmark runner."""

import subprocess

import pytest

from srtctl.benchmarks import get_runner
from srtctl.benchmarks.base import SCRIPTS_DIR


@pytest.mark.parametrize(
    ("frontend_type", "start_path", "stop_path"),
    [
        ("dynamo", "/engine/control/start_profile", "/engine/control/stop_profile"),
        ("sglang", "/start_profile", "/stop_profile"),
    ],
)
def test_profiling_shell_uses_frontend_control_routes(frontend_type, start_path, stop_path):
    """The shared shell helper must use each frontend's actual control API."""
    helper = SCRIPTS_DIR / "lib" / "profiling.sh"
    shell = r"""
source "$1"
SRTCTL_FRONTEND_TYPE="$2"
curl() {
    printf 'CURL %s\n' "$*" >&2
}
profiling__start_profile_on_worker "worker.example:7500" 1 3 "/profiles" "nsys" 9090
profiling__stop_profile_on_worker "worker.example:7500" 9090
"""
    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(helper), frontend_type],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"http://worker.example:7500{start_path}" in result.stderr
    assert f"http://worker.example:7500{stop_path}" in result.stderr


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
        assert profiling.nsys_trace == "cuda,nvtx"
        assert profiling.trace_fork_before_exec is None
        assert profiling.capture_range_end == "stop"
        assert profiling.nsys_library_paths is None

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

    def test_nsys_profiling_options(self):
        """Trace, fork, range-end, and library lookup settings are configurable."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(
            type="nsys",
            nsys_trace="cuda-sw,nvtx,osrt",
            trace_fork_before_exec=False,
            capture_range_end="repeat:1:async",
            nsys_library_paths=["/host/lib64", "/host/lib"],
        )

        prefix = profiling.get_nsys_prefix("/output/test", frontend_type="dynamo")
        assert prefix[prefix.index("-t") + 1] == "cuda-sw,nvtx,osrt"
        assert "--trace-fork-before-exec=true" not in prefix
        assert prefix[prefix.index("--capture-range-end") + 1] == "repeat:1:async"

        env = profiling.get_env_vars("agg", "/logs/profiles")
        assert "LD_LIBRARY_PATH" not in env

    def test_nsys_library_paths_preserve_container_environment(self):
        """Nsight paths are prepended without replacing image-provided paths."""
        from srtctl.cli.mixins.worker_stage import _nsys_library_path_preamble

        preamble = _nsys_library_path_preamble(["/host/lib64", "/host/lib", "/host/lib64"])
        assert preamble is not None
        result = subprocess.run(
            ["bash", "-c", f'export LD_LIBRARY_PATH=/image/lib && {preamble} && printf %s "$LD_LIBRARY_PATH"'],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout == "/host/lib64:/host/lib:/image/lib"

    def test_nsys_process_selector(self):
        """Each phase selects one logical worker and physical process rank."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="nsys",
            decode=ProfilingPhaseConfig(worker_index=2, worker_rank=3),
        )

        assert profiling.selects_process("decode", 2, 3)
        assert not profiling.selects_process("decode", 2, 0)
        assert not profiling.selects_process("decode", 1, 3)
        assert not profiling.selects_process("prefill", 2, 3)

        all_processes = ProfilingConfig(
            type="nsys",
            decode=ProfilingPhaseConfig(capture_scope="all"),
        )
        assert all_processes.captures_all_processes("decode")
        assert all_processes.selects_process("decode", 0, 0)
        assert all_processes.selects_process("decode", 4, 7)
        assert not all_processes.selects_process("prefill", 0, 0)

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

    @pytest.mark.parametrize(
        ("phase", "error"),
        [
            ({"worker_index": -1}, "worker_index must be non-negative"),
            ({"worker_index": 1}, "worker_index=1 is out of range"),
            ({"worker_rank": -1}, "worker_rank must be non-negative"),
            ({"worker_rank": 1}, "worker_rank=1 is not a physical process rank"),
        ],
    )
    def test_nsys_worker_selector_validation(self, phase, error):
        """Invalid physical process selectors fail during recipe validation."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match=error):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
                profiling=ProfilingConfig(
                    type="nsys",
                    aggregated=ProfilingPhaseConfig(start_step=0, stop_step=10, **phase),
                ),
            )

    def test_nsys_capture_scope_validation(self):
        """Only the documented selected/all capture scopes are accepted."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match="capture_scope"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
                profiling=ProfilingConfig(
                    type="nsys",
                    aggregated=ProfilingPhaseConfig(
                        start_step=0,
                        stop_step=10,
                        capture_scope="invalid",
                    ),
                ),
            )

    def test_nsys_worker_rank_uses_configured_physical_topology(self):
        """A multi-node worker accepts only ranks that its layout creates."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        def make_config(worker_rank):
            return SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(
                    gpu_type="h100",
                    gpus_per_node=4,
                    agg_nodes=2,
                    agg_workers=1,
                ),
                profiling=ProfilingConfig(
                    type="nsys",
                    aggregated=ProfilingPhaseConfig(
                        start_step=0,
                        stop_step=10,
                        worker_rank=worker_rank,
                    ),
                ),
            )

        assert make_config(1).profiling.aggregated.worker_rank == 1
        with pytest.raises(ValidationError, match="worker_rank=2 is not a physical process rank"):
            make_config(2)

    @pytest.mark.parametrize(
        ("profiling_kwargs", "error"),
        [
            ({"nsys_trace": ""}, "nsys_trace must not be empty"),
            ({"capture_range_end": ""}, "capture_range_end must not be empty"),
            ({"nsys_library_paths": [""]}, "nsys_library_paths must not contain empty paths"),
        ],
    )
    def test_nsys_option_validation(self, profiling_kwargs, error):
        """Invalid Nsight command settings fail during recipe validation."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        with pytest.raises(ValidationError, match=error):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
                profiling=ProfilingConfig(
                    type="nsys",
                    aggregated=ProfilingPhaseConfig(start_step=0, stop_step=10),
                    **profiling_kwargs,
                ),
            )

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

    def _build_decode_cmd(self, profiling, monkeypatch, decode_cfg=None):
        from pathlib import Path
        from types import SimpleNamespace

        import srtctl.core.slurm as slurm_mod
        from srtctl.backends.vllm import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Process

        monkeypatch.setattr(slurm_mod, "get_hostname_ip", lambda node, interface=None: "10.0.0.1")

        backend = VLLMProtocol(vllm_config=VLLMServerConfig(decode=decode_cfg or {"tensor-parallel-size": 1}))
        process = Process(
            node="node0",
            gpu_indices=frozenset({0}),
            sys_port=20000,
            http_port=0,
            endpoint_mode="decode",
            endpoint_index=0,
        )
        runtime = SimpleNamespace(
            model_path=Path("/model"),
            is_hf_model=False,
            request_plane="nats",
            network_interface="eth0",
        )
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

    def test_nsys_without_phase_steps_does_not_inject(self, monkeypatch):
        """type: nsys but no phase config for this mode -> no engine-driven capture."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(type="nsys")  # no decode phase
        cmd = self._build_decode_cmd(profiling, monkeypatch)
        assert self._profiler_config(cmd) is None


class TestProfilingTargetSelection:
    """Tests for selecting the one physical process controlled by a capture."""

    def test_worker_wrapper_targets_only_selected_process(self):
        from types import SimpleNamespace

        from srtctl.cli.mixins.worker_stage import WorkerStageMixin
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig
        from srtctl.core.topology import Process

        stage = WorkerStageMixin()
        stage.config = SimpleNamespace(
            backend=SimpleNamespace(type="vllm"),
            profiling=ProfilingConfig(
                type="nsys",
                aggregated=ProfilingPhaseConfig(worker_index=1, worker_rank=2),
            ),
        )
        selected = Process(
            node="worker-b",
            gpu_indices=frozenset({2}),
            sys_port=7002,
            http_port=8002,
            endpoint_mode="agg",
            endpoint_index=1,
            node_rank=2,
        )
        other = Process(
            node="worker-a",
            gpu_indices=frozenset({0}),
            sys_port=7000,
            http_port=8000,
            endpoint_mode="agg",
            endpoint_index=0,
            node_rank=0,
        )

        assert stage._profiling_selects_process(selected)
        assert not stage._profiling_selects_process(other)

        stage.config.profiling = ProfilingConfig(
            type="nsys",
            aggregated=ProfilingPhaseConfig(capture_scope="all"),
        )
        assert stage._profiling_selects_process(selected)
        assert stage._profiling_selects_process(other)

        stage.config.profiling = ProfilingConfig(type="nsys-time", delay_secs=1, duration_secs=1)
        assert stage._profiling_selects_process(selected)
        assert stage._profiling_selects_process(other)

        stage.config.backend.type = "trtllm"
        stage.config.profiling = ProfilingConfig(
            type="nsys",
            aggregated=ProfilingPhaseConfig(worker_index=1, worker_rank=2),
        )
        assert stage._profiling_selects_process(selected)
        assert stage._profiling_selects_process(other)

    def test_dynamo_control_uses_selected_system_port(self, monkeypatch):
        from types import SimpleNamespace

        from srtctl.cli.mixins import benchmark_stage
        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig
        from srtctl.core.topology import Process

        class Stage(BenchmarkStageMixin):
            @property
            def backend_processes(self):
                return self._processes

        stage = Stage()
        stage._processes = [
            Process(
                node="worker-a",
                gpu_indices=frozenset({0}),
                sys_port=7000,
                http_port=8000,
                endpoint_mode="agg",
                endpoint_index=0,
            ),
            Process(
                node="worker-b",
                gpu_indices=frozenset({1}),
                sys_port=7001,
                http_port=8001,
                endpoint_mode="agg",
                endpoint_index=1,
            ),
        ]
        stage.config = SimpleNamespace(
            backend_type="vllm",
            frontend=SimpleNamespace(type="dynamo"),
            dynamo=SimpleNamespace(sidecar=False),
            profiling=ProfilingConfig(
                type="nsys",
                aggregated=ProfilingPhaseConfig(
                    start_step=4,
                    stop_step=12,
                    worker_index=1,
                ),
            ),
            served_model_name="model",
        )
        stage.runtime = SimpleNamespace(
            frontend_port=9000,
            network_interface="eth0",
            nodes=SimpleNamespace(head="head"),
        )
        monkeypatch.setattr(benchmark_stage, "get_hostname_ip", lambda node, _interface: f"{node}.test")

        endpoints = stage._profiling_worker_endpoints()
        assert endpoints == [("agg", "worker-b.test", 7001)]

        env = stage._get_benchmark_profiling_env(get_runner("sa-bench"), endpoints)
        assert env["PROFILE_AGG_ENDPOINTS"] == "worker-b.test:7001"
        assert env["PROFILE_OUTPUT_DIR"] == "/logs/profiles"

        stage.config.profiling = ProfilingConfig(
            type="nsys",
            aggregated=ProfilingPhaseConfig(
                start_step=4,
                stop_step=12,
                capture_scope="all",
            ),
        )
        all_endpoints = stage._profiling_worker_endpoints()
        assert all_endpoints == [
            ("agg", "worker-a.test", 7000),
            ("agg", "worker-b.test", 7001),
        ]
        all_env = stage._get_benchmark_profiling_env(get_runner("sa-bench"), all_endpoints)
        assert all_env["PROFILE_AGG_ENDPOINTS"] == "worker-a.test:7000,worker-b.test:7001"

        stage.config.dynamo.sidecar = True
        assert stage._profiling_worker_endpoints() == all_endpoints

    @pytest.mark.parametrize(
        ("frontend_type", "sidecar", "expected_port"),
        [("dynamo", True, 7000), ("vllm", False, 9000)],
    )
    def test_leader_only_control_endpoint_skips_followers(
        self,
        monkeypatch,
        frontend_type,
        sidecar,
        expected_port,
    ):
        """All-process capture controls leader-only servers exactly once."""
        from types import SimpleNamespace

        from srtctl.cli.mixins import benchmark_stage
        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig
        from srtctl.core.topology import Process

        class Stage(BenchmarkStageMixin):
            @property
            def backend_processes(self):
                return self._processes

        stage = Stage()
        stage._processes = [
            Process(
                node="worker-a",
                gpu_indices=frozenset({0}),
                sys_port=7000,
                http_port=8000,
                endpoint_mode="agg",
                endpoint_index=0,
                node_rank=0,
            ),
            Process(
                node="worker-b",
                gpu_indices=frozenset({1}),
                sys_port=7001,
                http_port=0,
                endpoint_mode="agg",
                endpoint_index=0,
                node_rank=1,
            ),
        ]
        stage.config = SimpleNamespace(
            backend_type="vllm",
            frontend=SimpleNamespace(type=frontend_type),
            dynamo=SimpleNamespace(sidecar=sidecar),
            profiling=ProfilingConfig(
                type="nsys",
                aggregated=ProfilingPhaseConfig(capture_scope="all"),
            ),
        )
        stage.runtime = SimpleNamespace(frontend_port=9000, network_interface="eth0")
        monkeypatch.setattr(benchmark_stage, "get_hostname_ip", lambda node, _interface: f"{node}.test")

        assert stage._profiling_worker_endpoints() == [("agg", "worker-a.test", expected_port)]

        stage.config.profiling = ProfilingConfig(
            type="nsys",
            aggregated=ProfilingPhaseConfig(worker_rank=1),
        )
        with pytest.raises(ValueError, match="does not expose its own control endpoint"):
            stage._profiling_worker_endpoints()

    def test_missing_physical_rank_is_rejected(self):
        from types import SimpleNamespace

        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig
        from srtctl.core.topology import Process

        class Stage(BenchmarkStageMixin):
            @property
            def backend_processes(self):
                return self._processes

        stage = Stage()
        stage._processes = [
            Process(
                node="worker-a",
                gpu_indices=frozenset({0}),
                sys_port=7000,
                http_port=8000,
                endpoint_mode="agg",
                endpoint_index=0,
            )
        ]
        stage.config = SimpleNamespace(
            backend_type="vllm",
            frontend=SimpleNamespace(type="dynamo"),
            dynamo=SimpleNamespace(sidecar=False),
            profiling=ProfilingConfig(
                type="nsys",
                aggregated=ProfilingPhaseConfig(worker_rank=1),
            ),
        )
        stage.runtime = SimpleNamespace(frontend_port=9000, network_interface="eth0")

        with pytest.raises(ValueError, match="No physical process matches"):
            stage._profiling_worker_endpoints()


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
