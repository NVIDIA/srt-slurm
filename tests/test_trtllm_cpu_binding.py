# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRTLLM worker CPU/memory binding (`backend.cpu_binding`).

TRTLLM's in-process NUMA-aware affinity only re-pins the calling thread, so helper
threads that already exist keep roaming across sockets. These tests cover the
launch-time `taskset` pinning that fixes that, and pin down the legacy
`numactl -m 0,1` behavior so opting out stays byte-identical to before.
"""

import shlex
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from marshmallow import ValidationError

from srtctl.backends import TRTLLMCpuBinding, TRTLLMProtocol
from srtctl.core.topology import Process

GB200_CPU_MAP = ["0-71", "0-71", "72-143", "72-143"]
"""Measured GB200/GB300 layout: 4 GPUs and 2 CPU NUMA domains per node, GPUs 0-1
local to CPUs 0-71 and GPUs 2-3 local to CPUs 72-143."""


@pytest.fixture
def runtime(tmp_path):
    def _runtime(gpu_type="gb300"):
        rt = MagicMock()
        rt.log_dir = tmp_path
        rt.worker_model_arg = "/model"
        rt.model_path = Path("/models/DeepSeek-V4")
        rt.request_plane = "nats"
        rt.gpu_type = gpu_type
        rt.frontend_port = 8000
        return rt

    return _runtime


def make_process(gpu_indices, mode="decode"):
    return Process(
        node="node0",
        gpu_indices=frozenset(gpu_indices),
        sys_port=8081,
        http_port=8100,
        endpoint_mode=mode,
        endpoint_index=0,
        node_rank=0,
    )


def build(backend, runtime, gpu_indices=(0, 1, 2, 3), mode="decode", **kwargs):
    return backend.build_worker_command(
        process=make_process(gpu_indices, mode),
        endpoint_processes=[],
        runtime=runtime,
        **kwargs,
    )


class TestLegacyDefaults:
    """Recipes without `cpu_binding` must keep the exact pre-existing command."""

    def test_grace_prefill_decode_get_membind_only(self, runtime):
        for gpu_type in ("gb200", "gb300"):
            for mode in ("prefill", "decode"):
                cmd = build(TRTLLMProtocol(), runtime(gpu_type), mode=mode)
                assert cmd[:3] == ["numactl", "-m", "0,1"]
                assert cmd[3] == "trtllm-llmapi-launch"
                assert "taskset" not in cmd

    def test_agg_and_non_grace_get_no_prefix(self, runtime):
        assert build(TRTLLMProtocol(), runtime("gb300"), mode="agg")[0] == "trtllm-llmapi-launch"
        assert build(TRTLLMProtocol(), runtime("b200"))[0] == "trtllm-llmapi-launch"

    def test_no_affinity_env_injected(self, runtime):
        env = TRTLLMProtocol().get_environment_for_mode("decode")
        assert "TLLM_NUMA_AWARE_WORKER_AFFINITY" not in env


class TestUniformPinning:
    def test_cpus_emits_literal_taskset_before_numactl(self, runtime):
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(cpus="0-71"))
        cmd = build(backend, runtime())
        assert cmd[:6] == ["taskset", "-c", "0-71", "numactl", "-m", "0,1"]
        assert cmd[6] == "trtllm-llmapi-launch"

    def test_membind_none_drops_numactl(self, runtime):
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(cpus="0-71", membind=None))
        cmd = build(backend, runtime())
        assert cmd[:3] == ["taskset", "-c", "0-71"]
        assert "numactl" not in cmd

    def test_membind_applies_to_agg_and_non_grace_when_explicit(self, runtime):
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(cpus="0-71", membind="0"))
        cmd = build(backend, runtime("b200"), mode="agg")
        assert cmd[:6] == ["taskset", "-c", "0-71", "numactl", "-m", "0"]


class TestAffinityEnvironment:
    """taskset is useless unless TRTLLM is told not to clear the mask again."""

    def test_pinning_sets_numa_aware_worker_affinity_off(self, runtime):
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(cpus="0-71"))
        for mode in ("prefill", "decode", "agg"):
            assert backend.get_environment_for_mode(mode)["TLLM_NUMA_AWARE_WORKER_AFFINITY"] == "0"

    def test_membind_only_binding_does_not_touch_affinity_env(self, runtime):
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(membind="0,1"))
        assert "TLLM_NUMA_AWARE_WORKER_AFFINITY" not in backend.get_environment_for_mode("decode")

    def test_explicit_mode_environment_wins(self, runtime):
        backend = TRTLLMProtocol(
            cpu_binding=TRTLLMCpuBinding(cpus="0-71"),
            decode_environment={"TLLM_NUMA_AWARE_WORKER_AFFINITY": "1"},
        )
        assert backend.get_environment_for_mode("decode")["TLLM_NUMA_AWARE_WORKER_AFFINITY"] == "1"

    def test_eplb_shm_name_still_injected(self, runtime):
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(cpus="0-71"))
        assert "TRTLLM_EPLB_SHM_NAME" in backend.get_environment_for_mode("decode")


class TestPerLocalGpuPinning:
    """One srun launches every rank, so the per-GPU map resolves inside the task."""

    def test_wraps_command_in_bash_with_localid_lookup(self, runtime):
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(cpus_per_local_gpu=GB200_CPU_MAP))
        cmd = build(backend, runtime())
        assert cmd[0] == "bash"
        assert cmd[1] == "-c"
        assert cmd[2].startswith("__srt_cpu_list=(0-71 0-71 72-143 72-143); exec taskset -c ")
        assert "SLURM_LOCALID" in cmd[2]
        assert "numactl -m 0,1 trtllm-llmapi-launch python3 -m dynamo.trtllm" in cmd[2]

    def test_table_is_reordered_for_a_partial_node_endpoint(self, runtime):
        """An endpoint on GPUs 2-3 has SLURM_LOCALID 0-1, but must use the CPUs of
        the NUMA node those GPUs actually live on."""
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(cpus_per_local_gpu=GB200_CPU_MAP))
        cmd = build(backend, runtime(), gpu_indices=(2, 3))
        assert cmd[2].startswith("__srt_cpu_list=(72-143 72-143);")

    def test_nsys_prefix_stays_outermost(self, runtime):
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(cpus_per_local_gpu=GB200_CPU_MAP))
        cmd = build(backend, runtime(), nsys_prefix=["nsys", "profile", "-o", "/logs/w.nsys-rep"])
        assert "exec nsys profile -o /logs/w.nsys-rep taskset -c " in cmd[2]

    def test_trtllm_serve_frontend_is_wrapped_too(self, runtime):
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(cpus_per_local_gpu=GB200_CPU_MAP))
        cmd = build(backend, runtime(), mode="agg", frontend_type="trtllm_serve")
        assert cmd[0] == "bash"
        assert "trtllm-llmapi-launch trtllm-serve /model" in cmd[2]

    @pytest.mark.parametrize(
        ("local_id", "expected"),
        [("0", "0-71"), ("1", "0-71"), ("2", "72-143"), ("3", "72-143"), ("9", "0-71"), ("", "0-71")],
    )
    def test_resolves_to_the_right_cpu_list_under_bash(self, runtime, local_id, expected):
        """Run the emitted snippet through the same `bash -c` wrapping that
        `start_srun_process` applies, and read back the argv taskset would get."""
        backend = TRTLLMProtocol(cpu_binding=TRTLLMCpuBinding(cpus_per_local_gpu=GB200_CPU_MAP))
        cmd = build(backend, runtime())
        script = shlex.join(cmd).replace("exec taskset", "exec echo taskset", 1)
        result = subprocess.run(
            ["bash", "-c", f"export SLURM_LOCALID={local_id}; {script}"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.split()[:3] == ["taskset", "-c", expected]


class TestValidation:
    def test_cpus_and_cpus_per_local_gpu_are_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            TRTLLMCpuBinding(cpus="0-71", cpus_per_local_gpu=GB200_CPU_MAP)

    def test_empty_cpu_map_rejected(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            TRTLLMCpuBinding(cpus_per_local_gpu=[])

    def test_blank_cpu_map_entry_rejected(self):
        with pytest.raises(ValidationError, match="non-empty CPU lists"):
            TRTLLMCpuBinding(cpus_per_local_gpu=["0-71", "  "])

    def test_empty_gpu_types_rejected(self):
        with pytest.raises(ValidationError, match="gpu_types must not be empty"):
            TRTLLMCpuBinding(cpus="0-71", gpu_types=[])


class TestGpuTypeGuard:
    """CPU lists encode one node's layout; a mismatch must fail loudly, not silently
    pin ranks to the wrong socket."""

    @pytest.mark.parametrize(
        ("gpu_types", "gpu_type", "expected"),
        [
            (None, "gb300", True),
            (None, None, True),
            (["gb300"], "gb300", True),
            (["gb200", "gb300"], "gb200", True),
            (["gb300"], "gb200", False),
            (["gb300"], None, False),
        ],
    )
    def test_applies_to(self, gpu_types, gpu_type, expected):
        binding = TRTLLMCpuBinding(cpus="0-71", gpu_types=gpu_types)
        assert binding.applies_to(gpu_type) is expected

    def test_config_load_rejects_mismatched_gpu_type(self, tmp_path):
        from srtctl.core.config import validate_config_file

        recipe = tmp_path / "mismatch.yaml"
        recipe.write_text(_RECIPE.format(gpu_type="gb200", gpu_types='["gb300"]'))
        errors = validate_config_file(recipe)
        assert any("backend.cpu_binding is declared for gpu_type(s)" in str(e) for e in errors), errors

    def test_config_load_accepts_matching_gpu_type(self, tmp_path):
        from srtctl.core.config import validate_config_file

        recipe = tmp_path / "match.yaml"
        recipe.write_text(_RECIPE.format(gpu_type="gb300", gpu_types='["gb300"]'))
        assert validate_config_file(recipe) == []

    def test_config_load_accepts_unscoped_binding(self, tmp_path):
        from srtctl.core.config import validate_config_file

        recipe = tmp_path / "unscoped.yaml"
        recipe.write_text(_RECIPE.format(gpu_type="gb200", gpu_types="null"))
        assert validate_config_file(recipe) == []


_RECIPE = """
name: cpu-binding-guard
model:
  path: hf:fake/mock-model
  container: nvcr.io/fake:latest
  precision: fp8
resources:
  gpu_type: "{gpu_type}"
  gpus_per_node: 4
  agg_nodes: 1
  agg_workers: 1
backend:
  type: trtllm
  cpu_binding:
    gpu_types: {gpu_types}
    cpus_per_local_gpu: ["0-71", "0-71", "72-143", "72-143"]
benchmark:
  type: custom
  command: echo hi
"""


class TestYamlRoundTrip:
    def test_cpu_binding_loads_from_backend_yaml(self):
        backend = TRTLLMProtocol.Schema().load(
            {
                "type": "trtllm",
                "cpu_binding": {
                    "cpus_per_local_gpu": GB200_CPU_MAP,
                    "membind": "0,1",
                },
            }
        )
        assert backend.cpu_binding.cpus_per_local_gpu == GB200_CPU_MAP
        assert backend.cpu_binding.membind == "0,1"
        assert backend.cpu_binding.numa_aware_worker_affinity == "0"
