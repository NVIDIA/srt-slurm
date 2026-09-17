# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM nsys capture recipe from the Dynamo Benchmark Playbook §9.5.1.1.

The flags are not stylistic: `-t cuda` SIGSEGVs the KV transceiver on nsys 2026.x,
`--cuda-graph-trace=node` hides CUDA-graph decode work, `--sample=process-tree` wedged
workers at cudaProfilerStop. These tests pin the recipe and the knobs that relax it.
"""

import warnings

import pytest
from marshmallow import ValidationError

from srtctl.core.schema import ModelConfig, ProfilingConfig, ProfilingPhaseConfig, ResourceConfig, SrtConfig

INJECTION = "/usr/local/cuda-0.gpgpu/NsightSystems-cli-2026.3.0/target-linux-sbsa-armv8/libToolsInjection64.so"


def _disagg_config(**profiling_kwargs) -> SrtConfig:
    profiling_kwargs.setdefault("type", "nsys")
    profiling_kwargs.setdefault("prefill", ProfilingPhaseConfig(start_step=1200, stop_step=1300))
    profiling_kwargs.setdefault("decode", ProfilingPhaseConfig(start_step=6000, stop_step=6600))
    return SrtConfig(
        name="nsys-playbook",
        model=ModelConfig(path="/model", container="/container", precision="fp8"),
        resources=ResourceConfig(
            gpu_type="gb200", prefill_nodes=1, decode_nodes=1, prefill_workers=1, decode_workers=1
        ),
        profiling=ProfilingConfig(**profiling_kwargs),
    )


class TestPlaybookPrefix:
    def test_trtllm_iteration_prefix_is_the_playbook_recipe(self):
        prefix = ProfilingConfig(type="nsys").get_nsys_prefix(
            "/logs/profiles/decode/n_decode_w0_profile_rank%q{SLURM_PROCID}", backend_type="trtllm"
        )
        assert prefix[:3] == ["nsys", "profile", "--force-overwrite=true"]
        assert prefix[prefix.index("-t") + 1] == "cuda-sw,nvtx,python-gil"
        for flag in (
            "--cuda-graph-trace=graph",
            "--sample=none",
            "--cpuctxsw=none",
            "--python-sampling=false",
            "--python-sampling-frequency=1000",
            "--gpu-metrics-devices=none",
            "--flush-on-cudaprofilerstop=false",
            "--cuda-flush-interval=0",
            "--capture-range-end=stop",
        ):
            assert flag in prefix, flag
        assert prefix[prefix.index("-c") + 1] == "cudaProfilerApi"
        # The pre-playbook flags must be gone: HES trace and node-level graph trace.
        assert "cuda,nvtx,ucx" not in prefix
        assert "--cuda-graph-trace=node" not in prefix
        # One nsys per srun task; the per-rank output template is passed through verbatim.
        assert prefix[-2:] == ["-o", "/logs/profiles/decode/n_decode_w0_profile_rank%q{SLURM_PROCID}"]
        assert prefix[prefix.index("--kill") + 1] == "none"
        assert prefix[prefix.index("--wait") + 1] == "all"

    def test_trtllm_time_prefix_shares_the_recipe(self):
        prefix = ProfilingConfig(type="nsys-time", delay_secs=600, duration_secs=60).get_nsys_prefix(
            "/out/rank%q{SLURM_PROCID}", backend_type="trtllm"
        )
        assert "cuda-sw,nvtx,python-gil" in prefix
        assert "--cuda-graph-trace=graph" in prefix
        assert prefix[prefix.index("--delay") + 1] == "600"
        assert prefix[prefix.index("--duration") + 1] == "60"
        assert "cudaProfilerApi" not in prefix

    def test_knobs_relax_the_recipe(self):
        prefix = ProfilingConfig(
            type="nsys",
            nsys_trace="cuda-sw,nvtx,python-gil,ucx",
            nsys_cpuctxsw="process-tree",
            nsys_python_sampling=True,
            nsys_python_sampling_frequency=500,
            extra_nsys_args=["--stats=true"],
        ).get_nsys_prefix("/out/rank%q{SLURM_PROCID}", backend_type="trtllm")
        assert prefix[prefix.index("-t") + 1] == "cuda-sw,nvtx,python-gil,ucx"
        assert "--cpuctxsw=process-tree" in prefix
        assert "--sample=none" in prefix  # IP sampling stays off unless asked
        assert "--python-sampling=true" in prefix
        assert "--python-sampling-frequency=500" in prefix
        assert prefix.index("--stats=true") < prefix.index("-o")

    def test_non_trtllm_backends_keep_their_own_prefix(self):
        prefix = ProfilingConfig(type="nsys").get_nsys_prefix("/out/x", backend_type="sglang", frontend_type="dynamo")
        assert "--trace-fork-before-exec=true" in prefix
        assert "cuda-sw" not in prefix


class TestPlaybookEnv:
    def test_worker_env_carries_window_ranks_and_nvtx(self):
        env = ProfilingConfig(
            type="nsys",
            prefill=ProfilingPhaseConfig(start_step=1200, stop_step=1300),
            decode=ProfilingPhaseConfig(start_step=6000, stop_step=6600),
            nvtx_injection_path=INJECTION,
        ).get_env_vars("decode", "/logs/profiles")
        assert env["TLLM_PROFILE_START_STOP"] == "6000-6600"
        assert env["TLLM_LLMAPI_ENABLE_NVTX"] == "1"
        assert env["TLLM_PROFILE_LOG_RANKS"] == "all"
        assert env["DYN_ENABLE_RUST_NVTX"] == "1"
        assert env["NVTX_INJECTION64_PATH"] == INJECTION

    def test_injection_path_only_when_configured(self):
        env = ProfilingConfig(type="nsys", decode=ProfilingPhaseConfig(0, 50)).get_env_vars("decode", "/p")
        assert "NVTX_INJECTION64_PATH" not in env
        assert env["DYN_ENABLE_RUST_NVTX"] == "1"

    def test_log_ranks_override(self):
        env = ProfilingConfig(type="nsys", log_ranks="0", decode=ProfilingPhaseConfig(0, 50)).get_env_vars(
            "decode", "/p"
        )
        assert env["TLLM_PROFILE_LOG_RANKS"] == "0"

    def test_disabled_profiling_sets_nothing(self):
        assert ProfilingConfig().get_env_vars("decode", "/p") == {}


class TestPlaybookValidation:
    def test_valid_playbook_config(self):
        config = _disagg_config(nvtx_injection_path=INJECTION)
        assert config.profiling.enabled and config.profiling.nsys_trace == "cuda-sw,nvtx,python-gil"

    @pytest.mark.parametrize(
        ("field", "value"),
        [("nsys_cuda_graph_trace", "kernel"), ("nsys_sample", "yes"), ("nsys_cpuctxsw", "all")],
    )
    def test_enum_fields_are_checked(self, field, value):
        with pytest.raises(ValidationError, match=f"profiling.{field}"):
            _disagg_config(**{field: value})

    def test_empty_trace_rejected(self):
        with pytest.raises(ValidationError, match="nsys_trace"):
            _disagg_config(nsys_trace="  ")

    def test_ip_sampling_warns_but_is_allowed(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = _disagg_config(nsys_sample="process-tree")
        assert config.profiling.nsys_sample == "process-tree"
        assert any("perf_event_paranoid" in str(w.message) for w in caught)


class TestFrontendProfiling:
    """profiling.frontend: nsys on the Dynamo frontend with a time window."""

    def test_prefix_env_and_defaults(self):
        from srtctl.core.schema import ProfilingFrontendConfig

        p = ProfilingConfig(
            type="nsys",
            frontend=ProfilingFrontendConfig(delay_secs=1500, duration_secs=120),
            nvtx_injection_path="/opt/nsight/libToolsInjection64.so",
        )
        assert p.profiles_frontend
        prefix = p.get_frontend_nsys_prefix("/logs/profiles/frontend/n1_frontend_0")
        assert prefix[:3] == ["nsys", "profile", "--force-overwrite=true"]
        assert prefix[prefix.index("-t") + 1] == "nvtx"  # NVTX-centric: no CUDA tracing on a CUDA-less process
        assert "--delay" in prefix and prefix[prefix.index("--delay") + 1] == "1500"
        assert "--duration" in prefix and prefix[prefix.index("--duration") + 1] == "120"
        assert prefix[-6:] == ["--kill", "none", "--wait", "all", "-o", "/logs/profiles/frontend/n1_frontend_0"]
        assert "-c" not in prefix and "--capture-range-end=stop" not in prefix
        assert p.get_frontend_env_vars() == {
            "DYN_ENABLE_RUST_NVTX": "1",
            "NVTX_INJECTION64_PATH": "/opt/nsight/libToolsInjection64.so",
        }
        # Without the block nothing changes for the frontend.
        assert ProfilingConfig(type="nsys").get_frontend_nsys_prefix("/o") == []
        assert ProfilingConfig(type="nsys").get_frontend_env_vars() == {}

    def test_validation(self):
        from srtctl.core.schema import ProfilingFrontendConfig

        cfg = _disagg_config(frontend=ProfilingFrontendConfig())
        assert cfg.profiling.profiles_frontend
        with pytest.raises(ValidationError, match="requires profiling.type nsys"):
            _disagg_config(type="torch", frontend=ProfilingFrontendConfig())
        with pytest.raises(ValidationError, match="duration_secs"):
            _disagg_config(frontend=ProfilingFrontendConfig(duration_secs=0))
        with pytest.raises(ValidationError, match="trace"):
            _disagg_config(frontend=ProfilingFrontendConfig(trace="  "))

    def test_frontend_sampling_is_frontend_only(self):
        from srtctl.core.schema import ProfilingFrontendConfig

        cfg = _disagg_config(frontend=ProfilingFrontendConfig(sample="process-tree", cpuctxsw="process-tree", trace="osrt"))
        fe = cfg.profiling.get_frontend_nsys_prefix("/logs/profiles/frontend/n_frontend_0")
        assert "--sample=process-tree" in fe and "--cpuctxsw=process-tree" in fe and fe[fe.index("-t") + 1] == "osrt"
        # workers keep the global (playbook) defaults
        wk = cfg.profiling.get_nsys_prefix("/logs/profiles/decode/x_rank%q{SLURM_PROCID}", backend_type="trtllm")
        assert "--sample=none" in wk and "--cpuctxsw=none" in wk
        # unset -> inherit the global values
        fe2 = _disagg_config(frontend=ProfilingFrontendConfig()).profiling.get_frontend_nsys_prefix("/logs/p/f")
        assert "--sample=none" in fe2
        with pytest.raises(ValidationError):
            _disagg_config(frontend=ProfilingFrontendConfig(sample="cpu"))

    def test_teardown_grace_default_and_validation(self):
        assert _disagg_config().profiling.teardown_grace_secs == 180
        assert _disagg_config(teardown_grace_secs=600).profiling.teardown_grace_secs == 600
        with pytest.raises(ValidationError):
            _disagg_config(teardown_grace_secs=0)
        assert _disagg_config().profiling.app_exit_grace_secs == 120
        with pytest.raises(ValidationError):
            _disagg_config(app_exit_grace_secs=0)
        with pytest.raises(ValidationError):
            _disagg_config(teardown_grace_secs=100, app_exit_grace_secs=120)  # must leave time for conversion

    def test_dynamo_frontend_is_wrapped(self, tmp_path):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from srtctl.core.schema import ProfilingFrontendConfig
        from srtctl.frontends.base import get_frontend

        cfg = _disagg_config(
            frontend=ProfilingFrontendConfig(delay_secs=10, duration_secs=5),
            nvtx_injection_path="/opt/nsight/libToolsInjection64.so",
        )
        topology = SimpleNamespace(frontend_nodes=["n1"], frontend_port=8000)
        runtime = SimpleNamespace(
            log_dir=tmp_path,
            nodes=SimpleNamespace(infra="n0", het_group_for=lambda node: None),
            container_image="/img.sqsh",
            container_mounts={},
            environment={},
        )
        with patch("srtctl.frontends.dynamo.start_srun_process", return_value=MagicMock()) as srun:
            procs = get_frontend("dynamo").start_frontends(
                topology=topology, runtime=runtime, config=cfg, backend=None, backend_processes=[]
            )
        assert len(procs) == 1
        kwargs = srun.call_args.kwargs
        cmd = kwargs["command"]
        # wrapped by keepalive_command: bash -c '<nsys ... frontend> & ...wait for the orphaned frontend...'
        assert cmd[:2] == ["bash", "-c"]
        script = cmd[2]
        assert "setsid nsys profile " in script  # keepalive wrapper: nsys in its own session
        assert "-o /logs/profiles/frontend/n1_frontend_0 python3 -m dynamo.frontend --http-port=8000" in script
        assert 'kill -0 "$APP"' in script
        assert kwargs["env_to_set"]["DYN_ENABLE_RUST_NVTX"] == "1"
        assert kwargs["env_to_set"]["NVTX_INJECTION64_PATH"] == "/opt/nsight/libToolsInjection64.so"
        assert (tmp_path / "profiles" / "frontend").is_dir()
        # an open frontend capture is written only after the frontend exits: cleanup must wait for it
        assert procs[0].terminate_timeout == 180.0
        # ... and signal the step, not the srun client, so the step must be named
        assert kwargs["step_name"] == "frontend_0" and procs[0].step_name == "frontend_0"

    def test_dynamo_frontend_untouched_without_block(self, tmp_path):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from srtctl.frontends.base import get_frontend

        cfg = _disagg_config()
        topology = SimpleNamespace(frontend_nodes=["n1"], frontend_port=8000)
        runtime = SimpleNamespace(
            log_dir=tmp_path,
            nodes=SimpleNamespace(infra="n0", het_group_for=lambda node: None),
            container_image="/img.sqsh",
            container_mounts={},
            environment={},
        )
        with patch("srtctl.frontends.dynamo.start_srun_process", return_value=MagicMock()) as srun:
            get_frontend("dynamo").start_frontends(
                topology=topology, runtime=runtime, config=cfg, backend=None, backend_processes=[]
            )
        cmd = srun.call_args.kwargs["command"]
        assert cmd[:3] == ["python3", "-m", "dynamo.frontend"]
        assert "DYN_ENABLE_RUST_NVTX" not in srun.call_args.kwargs["env_to_set"]
