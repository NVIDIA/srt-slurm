# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mocker backend configuration and command construction."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from srtctl.backends import MockerProtocol, MockerServerConfig
from srtctl.core.schema import SrtConfig

# ============================================================================
# Helpers
# ============================================================================


def _make_process(mode="agg", bootstrap_port=None):
    from srtctl.core.topology import Process

    return Process(
        node="node0",
        gpu_indices=frozenset([0]),
        sys_port=8081,
        http_port=30000,
        endpoint_mode=mode,
        endpoint_index=0,
        node_rank=0,
        bootstrap_port=bootstrap_port,
    )


def _make_runtime(*, is_hf: bool):
    runtime = MagicMock()
    if is_hf:
        runtime.model_path = Path("Qwen/Qwen3-0.6B")
        runtime.is_hf_model = True
    else:
        runtime.model_path = Path("/models/my-model")
        runtime.is_hf_model = False
    return runtime


# ============================================================================
# YAML Deserialization
# ============================================================================


class TestMockerConfigLoading:
    """Tests for mocker backend YAML deserialization."""

    def test_minimal_mocker_config(self):
        """Minimal mocker recipe loads correctly."""
        data = {
            "name": "test-mocker",
            "model": {"path": "hf:Qwen/Qwen3-0.6B", "container": "test", "precision": "fp16"},
            "resources": {"gpu_type": "gb200", "gpus_per_node": 4, "agg_nodes": 1, "agg_workers": 1},
            "backend": {"type": "mocker"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = SrtConfig.from_yaml(Path(f.name))

        assert config.backend_type == "mocker"
        assert isinstance(config.backend, MockerProtocol)

    def test_mocker_defaults(self):
        """MockerProtocol has correct defaults."""
        backend = MockerProtocol()
        assert backend.type == "mocker"
        assert backend.engine_type == "vllm"
        assert backend.speedup_ratio == 100.0
        assert backend.decode_speedup_ratio == 1.0
        assert backend.num_gpu_blocks_override == 16384
        assert backend.max_num_seqs == 256
        assert backend.max_num_batched_tokens == 8192
        assert backend.block_size is None
        assert backend.data_parallel_size == 1
        assert backend.num_workers == 1
        assert backend.startup_time is None
        assert backend.kv_transfer_bandwidth is None
        assert backend.kv_cache_dtype is None
        assert backend.enable_prefix_caching is True
        assert backend.enable_chunked_prefill is True
        assert backend.preemption_mode is None

    def test_mocker_with_custom_fields(self):
        """Custom fields deserialize correctly."""
        data = {
            "name": "test",
            "model": {"path": "test", "container": "test", "precision": "fp16"},
            "resources": {"gpu_type": "gb200", "gpus_per_node": 4, "agg_nodes": 1, "agg_workers": 1},
            "backend": {
                "type": "mocker",
                "engine_type": "sglang",
                "speedup_ratio": 50.0,
                "num_gpu_blocks_override": 8192,
                "block_size": 32,
                "kv_cache_dtype": "fp8_e4m3",
                "enable_prefix_caching": False,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = SrtConfig.from_yaml(Path(f.name))

        backend = config.backend
        assert isinstance(backend, MockerProtocol)
        assert backend.engine_type == "sglang"
        assert backend.speedup_ratio == 50.0
        assert backend.num_gpu_blocks_override == 8192
        assert backend.block_size == 32
        assert backend.kv_cache_dtype == "fp8_e4m3"
        assert backend.enable_prefix_caching is False

    def test_mocker_with_per_mode_config(self):
        """Per-mode mocker_config deserializes correctly."""
        data = {
            "name": "test",
            "model": {"path": "test", "container": "test", "precision": "fp16"},
            "resources": {"gpu_type": "gb200", "gpus_per_node": 4, "agg_nodes": 1, "agg_workers": 1},
            "backend": {
                "type": "mocker",
                "mocker_config": {
                    "prefill": {"max-num-seqs": 512},
                    "decode": {"max-num-seqs": 128},
                },
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = SrtConfig.from_yaml(Path(f.name))

        backend = config.backend
        assert isinstance(backend, MockerProtocol)
        assert backend.get_config_for_mode("prefill") == {"max-num-seqs": 512}
        assert backend.get_config_for_mode("decode") == {"max-num-seqs": 128}
        assert backend.get_config_for_mode("agg") == {}

    def test_mocker_with_environment(self):
        """Per-mode environment vars deserialize correctly."""
        backend = MockerProtocol(
            prefill_environment={"FOO": "bar"},
            decode_environment={"BAZ": "qux"},
        )
        assert backend.get_environment_for_mode("prefill") == {"FOO": "bar"}
        assert backend.get_environment_for_mode("decode") == {"BAZ": "qux"}
        assert backend.get_environment_for_mode("agg") == {}

    def test_smoke_test_recipes_load(self):
        """Mocker recipes in recipes/mocker/ load successfully."""
        recipe_dir = Path("recipes/mocker")
        if not recipe_dir.exists():
            pytest.skip("recipes/mocker/ not found")

        for recipe in recipe_dir.glob("*.yaml"):
            config = SrtConfig.from_yaml(recipe)
            assert config.backend_type == "mocker"
            assert isinstance(config.backend, MockerProtocol)


# ============================================================================
# Command Construction
# ============================================================================


class TestMockerCommandConstruction:
    """Tests for mocker build_worker_command()."""

    def test_agg_basic_command(self):
        """Aggregated mode produces correct base command."""
        backend = MockerProtocol()
        process = _make_process(mode="agg")
        runtime = _make_runtime(is_hf=False)

        cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        assert cmd[:4] == ["python3", "-m", "dynamo.mocker", "--model-path"]
        assert cmd[4] == "/model"
        # No --disaggregation-mode for agg
        assert "--disaggregation-mode" not in cmd
        # Core params present
        assert "--engine-type" in cmd
        assert "--speedup-ratio" in cmd

    def test_prefill_mode(self):
        """Prefill mode includes disaggregation and bootstrap port."""
        backend = MockerProtocol()
        process = _make_process(mode="prefill", bootstrap_port=31000)
        runtime = _make_runtime(is_hf=False)

        cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--disaggregation-mode")
        assert cmd[idx + 1] == "prefill"
        idx = cmd.index("--bootstrap-ports")
        assert cmd[idx + 1] == "31000"

    def test_decode_mode(self):
        """Decode mode includes disaggregation, no bootstrap port."""
        backend = MockerProtocol()
        process = _make_process(mode="decode")
        runtime = _make_runtime(is_hf=False)

        cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--disaggregation-mode")
        assert cmd[idx + 1] == "decode"
        assert "--bootstrap-ports" not in cmd

    def test_hf_model_uses_model_id(self):
        """HF model passes model ID directly."""
        backend = MockerProtocol()
        process = _make_process()
        runtime = _make_runtime(is_hf=True)

        cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--model-path")
        assert cmd[idx + 1] == "Qwen/Qwen3-0.6B"

    def test_local_model_uses_container_mount(self):
        """Local model passes /model container path."""
        backend = MockerProtocol()
        process = _make_process()
        runtime = _make_runtime(is_hf=False)

        cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--model-path")
        assert cmd[idx + 1] == "/model"

    def test_core_params_always_present(self):
        """Core simulation params are always emitted."""
        backend = MockerProtocol()
        process = _make_process()
        runtime = _make_runtime(is_hf=False)

        cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        # These are always present
        idx = cmd.index("--engine-type")
        assert cmd[idx + 1] == "vllm"
        idx = cmd.index("--speedup-ratio")
        assert cmd[idx + 1] == "100.0"
        idx = cmd.index("--data-parallel-size")
        assert cmd[idx + 1] == "1"
        idx = cmd.index("--num-gpu-blocks-override")
        assert cmd[idx + 1] == "16384"
        idx = cmd.index("--max-num-seqs")
        assert cmd[idx + 1] == "256"
        idx = cmd.index("--max-num-batched-tokens")
        assert cmd[idx + 1] == "8192"

    def test_optional_flags_omitted_by_default(self):
        """Default optional fields don't produce CLI flags."""
        backend = MockerProtocol()
        process = _make_process()
        runtime = _make_runtime(is_hf=False)

        cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        # These should NOT be present with defaults
        assert "--block-size" not in cmd
        assert "--num-workers" not in cmd
        assert "--startup-time" not in cmd
        assert "--kv-transfer-bandwidth" not in cmd
        assert "--kv-cache-dtype" not in cmd
        assert "--no-enable-prefix-caching" not in cmd
        assert "--no-enable-chunked-prefill" not in cmd
        assert "--preemption-mode" not in cmd
        assert "--decode-speedup-ratio" not in cmd

    def test_optional_flags_emitted_when_set(self):
        """Non-default optional fields produce correct CLI flags."""
        backend = MockerProtocol(
            block_size=32,
            num_workers=4,
            startup_time=5.0,
            kv_transfer_bandwidth=64.0,
            kv_cache_dtype="fp8_e4m3",
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            preemption_mode="fifo",
            decode_speedup_ratio=2.0,
        )
        process = _make_process()
        runtime = _make_runtime(is_hf=False)

        cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        assert cmd[cmd.index("--block-size") + 1] == "32"
        assert cmd[cmd.index("--num-workers") + 1] == "4"
        assert cmd[cmd.index("--startup-time") + 1] == "5.0"
        assert cmd[cmd.index("--kv-transfer-bandwidth") + 1] == "64.0"
        assert cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
        assert "--no-enable-prefix-caching" in cmd
        assert "--no-enable-chunked-prefill" in cmd
        assert cmd[cmd.index("--preemption-mode") + 1] == "fifo"
        assert cmd[cmd.index("--decode-speedup-ratio") + 1] == "2.0"

    def test_per_mode_config_appended(self):
        """Per-mode mocker_config overrides are appended as CLI args."""
        backend = MockerProtocol(
            mocker_config=MockerServerConfig(
                prefill={"max-num-seqs": 512, "enable-prefix-caching": True},
            ),
        )
        process = _make_process(mode="prefill", bootstrap_port=31000)
        runtime = _make_runtime(is_hf=False)

        cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        # Per-mode overrides should appear (sorted by key)
        assert "--enable-prefix-caching" in cmd
        # Find the per-mode max-num-seqs (after the top-level one)
        idx = cmd.index("--max-num-seqs")
        assert cmd[idx + 1] == "256"  # top-level
        # The per-mode one appears later from _config_to_cli_args
        remaining = cmd[idx + 2 :]
        assert "--max-num-seqs" in remaining
        idx2 = remaining.index("--max-num-seqs")
        assert remaining[idx2 + 1] == "512"

    def test_nsys_prefix(self):
        """nsys prefix is prepended to command."""
        backend = MockerProtocol()
        process = _make_process()
        runtime = _make_runtime(is_hf=False)
        nsys = ["nsys", "profile", "-o", "output"]

        cmd = backend.build_worker_command(
            process=process, endpoint_processes=[process], runtime=runtime, nsys_prefix=nsys
        )

        assert cmd[:4] == ["nsys", "profile", "-o", "output"]
        assert cmd[4:7] == ["python3", "-m", "dynamo.mocker"]


# ============================================================================
# Endpoint Allocation & Topology
# ============================================================================


class TestMockerTopology:
    """Tests for mocker endpoint allocation and process topology."""

    def test_agg_allocation(self):
        """Aggregated mode allocates a single endpoint."""
        backend = MockerProtocol()
        endpoints = backend.allocate_endpoints(
            num_prefill=0,
            num_decode=0,
            num_agg=1,
            gpus_per_prefill=0,
            gpus_per_decode=0,
            gpus_per_agg=4,
            gpus_per_node=4,
            available_nodes=["node0"],
        )
        assert len(endpoints) == 1
        assert endpoints[0].mode == "agg"

    def test_disagg_allocation(self):
        """Disaggregated mode allocates prefill and decode endpoints."""
        backend = MockerProtocol()
        endpoints = backend.allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=1,
            gpus_per_decode=1,
            gpus_per_agg=0,
            gpus_per_node=4,
            available_nodes=["node0"],
        )
        modes = {ep.mode for ep in endpoints}
        assert modes == {"prefill", "decode"}

    def test_srun_config(self):
        """Mocker uses per-process launching, no MPI."""
        backend = MockerProtocol()
        srun_config = backend.get_srun_config()
        assert srun_config.mpi is None
        assert srun_config.oversubscribe is False
        assert srun_config.launch_per_endpoint is False

    def test_process_environment_empty(self):
        """Mocker has no per-process env vars."""
        backend = MockerProtocol()
        process = _make_process()
        assert backend.get_process_environment(process) == {}

    def test_served_model_name_default(self):
        """Mocker returns the default model name."""
        backend = MockerProtocol()
        assert backend.get_served_model_name("my-model") == "my-model"
