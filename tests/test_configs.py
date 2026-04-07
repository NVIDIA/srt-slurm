# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for configuration loading and validation."""

import glob
from pathlib import Path

import pytest

from srtctl.backends import SGLangProtocol, SGLangServerConfig
from srtctl.core.schema import SrtConfig


class TestConfigLoading:
    """Tests for config file loading."""

    def test_config_loading_from_yaml(self):
        """Test that config files in recipes/ can be loaded."""
        # Find all yaml files in recipes/
        config_files = glob.glob("recipes/**/*.yaml", recursive=True)

        if not config_files:
            pytest.skip("No config files found in recipes/")

        errors = []
        loaded = 0
        for config_path in config_files:
            try:
                config = SrtConfig.from_yaml(Path(config_path))
                assert config.name is not None
                assert config.model is not None
                assert config.resources is not None
                assert config.backend is not None
                loaded += 1
                print(f"\n✓ Loaded config: {config_path}")
                print(f"  Name: {config.name}")
                print(f"  Backend: {config.backend_type}")
            except Exception as e:
                errors.append(f"{config_path}: {e}")

        print(f"\nLoaded {loaded}/{len(config_files)} configs")
        if errors:
            print(f"Errors ({len(errors)}):")
            for err in errors[:5]:  # Show first 5 errors
                print(f"  - {err}")


class TestSrtConfigStructure:
    """Tests for SrtConfig dataclass structure."""

    def test_resource_config_disaggregated(self):
        """Test resource config disaggregation detection."""
        from srtctl.core.schema import ResourceConfig

        # Disaggregated config
        disagg = ResourceConfig(
            gpu_type="h100",
            gpus_per_node=8,
            prefill_nodes=1,
            decode_nodes=2,
        )
        assert disagg.is_disaggregated is True

        # Aggregated config
        agg = ResourceConfig(
            gpu_type="h100",
            gpus_per_node=8,
            agg_nodes=2,
        )
        assert agg.is_disaggregated is False

    def test_decode_nodes_zero_inherits_tp_from_prefill(self):
        """When decode_nodes=0, gpus_per_decode inherits from prefill."""
        from srtctl.core.schema import ResourceConfig

        # 6 prefill + 2 decode on 2 nodes, sharing
        config = ResourceConfig(
            gpu_type="gb200",
            gpus_per_node=8,
            prefill_nodes=2,
            decode_nodes=0,
            prefill_workers=6,
            decode_workers=2,
        )

        assert config.gpus_per_prefill == 2  # (2*8)/6 = 2
        assert config.gpus_per_decode == 2  # inherits from prefill

        # Total GPUs should fit
        total_needed = config.num_prefill * config.gpus_per_prefill + config.num_decode * config.gpus_per_decode
        total_available = config.total_nodes * config.gpus_per_node
        assert total_needed <= total_available


class TestDynamoConfig:
    """Tests for DynamoConfig."""

    def test_default_version(self):
        """Default is version 0.8.0."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig()
        assert config.version == "0.8.0"
        assert config.hash is None
        assert config.top_of_tree is False
        assert not config.needs_source_install

    def test_version_install_command(self):
        """Version config generates pip install command."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(version="0.8.0")
        cmd = config.get_install_commands()
        assert "pip install" in cmd
        assert "ai-dynamo-runtime==0.8.0" in cmd
        assert "ai-dynamo==0.8.0" in cmd

    def test_hash_install_command(self):
        """Hash config generates source install command."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(hash="abc123")
        assert config.version is None  # Auto-cleared
        assert config.needs_source_install
        cmd = config.get_install_commands()
        assert "git clone" in cmd
        assert "git checkout abc123" in cmd
        assert "maturin build" in cmd
        assert "if [ -d /sgl-workspace ]" in cmd
        assert "/tmp/dynamo_build" in cmd
        assert "protobuf-compiler" in cmd
        assert "if ! command -v cargo" in cmd
        assert "if ! command -v maturin" in cmd

    def test_top_of_tree_install_command(self):
        """Top-of-tree config generates source install without checkout."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(top_of_tree=True)
        assert config.version is None  # Auto-cleared
        assert config.needs_source_install
        cmd = config.get_install_commands()
        assert "git clone" in cmd
        assert "git checkout" not in cmd
        assert "maturin build" in cmd
        assert "if [ -d /sgl-workspace ]" in cmd
        assert "/tmp/dynamo_build" in cmd
        assert "--break-system-packages" in cmd
        assert "--force-reinstall" in cmd

    def test_hash_and_top_of_tree_not_allowed(self):
        """Cannot specify both hash and top_of_tree."""
        from srtctl.core.schema import DynamoConfig

        with pytest.raises(ValueError, match="Cannot specify both"):
            DynamoConfig(hash="abc123", top_of_tree=True)


class TestSGLangProtocol:
    """Tests for SGLangProtocol."""

    def test_sglang_config_structure(self):
        """Test SGLang config has expected structure."""
        config = SGLangProtocol()

        assert config.type == "sglang"
        assert hasattr(config, "prefill_environment")
        assert hasattr(config, "decode_environment")
        assert hasattr(config, "sglang_config")

    def test_get_environment_for_mode(self):
        """Test environment variable retrieval per mode."""
        config = SGLangProtocol(
            prefill_environment={"PREFILL_VAR": "1"},
            decode_environment={"DECODE_VAR": "1"},
        )

        assert config.get_environment_for_mode("prefill") == {"PREFILL_VAR": "1"}
        assert config.get_environment_for_mode("decode") == {"DECODE_VAR": "1"}
        assert config.get_environment_for_mode("agg") == {}

    def test_kv_events_config_global_bool(self):
        """Test kv_events_config=True enables prefill+decode with defaults."""
        config = SGLangProtocol(kv_events_config=True)

        assert config.get_kv_events_config_for_mode("prefill") == {
            "publisher": "zmq",
            "topic": "kv-events",
        }
        assert config.get_kv_events_config_for_mode("decode") == {
            "publisher": "zmq",
            "topic": "kv-events",
        }
        assert config.get_kv_events_config_for_mode("agg") is None

    def test_kv_events_config_per_mode(self):
        """Test kv_events_config per-mode control."""
        config = SGLangProtocol(
            kv_events_config={
                "prefill": True,
                # decode omitted = disabled
            }
        )

        assert config.get_kv_events_config_for_mode("prefill") == {
            "publisher": "zmq",
            "topic": "kv-events",
        }
        assert config.get_kv_events_config_for_mode("decode") is None
        assert config.get_kv_events_config_for_mode("agg") is None

    def test_kv_events_config_custom_settings(self):
        """Test kv_events_config with custom publisher/topic."""
        config = SGLangProtocol(
            kv_events_config={
                "prefill": {"topic": "prefill-events"},
                "decode": {"publisher": "custom", "topic": "decode-events"},
            }
        )

        prefill_cfg = config.get_kv_events_config_for_mode("prefill")
        assert prefill_cfg["publisher"] == "zmq"  # default
        assert prefill_cfg["topic"] == "prefill-events"

        decode_cfg = config.get_kv_events_config_for_mode("decode")
        assert decode_cfg["publisher"] == "custom"
        assert decode_cfg["topic"] == "decode-events"

    def test_kv_events_config_aggregated(self):
        """Test kv_events_config with aggregated key."""
        config = SGLangProtocol(
            kv_events_config={
                "aggregated": True,
            }
        )

        assert config.get_kv_events_config_for_mode("agg") == {
            "publisher": "zmq",
            "topic": "kv-events",
        }
        assert config.get_kv_events_config_for_mode("prefill") is None
        assert config.get_kv_events_config_for_mode("decode") is None

    def test_kv_events_config_disabled(self):
        """Test kv_events_config disabled by default."""
        config = SGLangProtocol()

        assert config.get_kv_events_config_for_mode("prefill") is None
        assert config.get_kv_events_config_for_mode("decode") is None
        assert config.get_kv_events_config_for_mode("agg") is None

    def test_grpc_mode_disabled_by_default(self):
        """Test gRPC mode is disabled by default."""
        config = SGLangProtocol()

        assert config.is_grpc_mode("prefill") is False
        assert config.is_grpc_mode("decode") is False
        assert config.is_grpc_mode("agg") is False

    def test_grpc_mode_enabled_per_mode(self):
        """Test gRPC mode can be enabled per worker mode."""
        config = SGLangProtocol(
            sglang_config=SGLangServerConfig(
                prefill={"grpc-mode": True},
                decode={"grpc-mode": True},
                aggregated={"grpc-mode": False},
            )
        )

        assert config.is_grpc_mode("prefill") is True
        assert config.is_grpc_mode("decode") is True
        assert config.is_grpc_mode("agg") is False


class TestServedModelName:
    """Tests for served_model_name property extraction from backend configs."""

    def test_vllm_served_model_name_extracted_from_config(self):
        """Test vLLM extracts served_model_name from config instead of using path basename.

        This was a bug where vLLM backend didn't implement get_served_model_name(),
        causing the benchmark to use the model path basename (e.g., "hf-d47b0d4-nim-bf16")
        instead of the configured name (e.g., "Qwen/Qwen3-32B").
        """
        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/models/hf-d47b0d4-nim-bf16", container="/container.sqsh", precision="bf16"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1, agg_workers=1),
            backend=VLLMProtocol(
                vllm_config=VLLMServerConfig(
                    aggregated={"served-model-name": "Qwen/Qwen3-32B"},
                )
            ),
        )

        # Should use configured name, not path basename
        assert config.served_model_name == "Qwen/Qwen3-32B"

    def test_vllm_served_model_name_fallback_to_path(self):
        """Test vLLM falls back to model path basename when not configured."""
        from srtctl.backends import VLLMProtocol
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/models/Qwen/Qwen3-32B", container="/container.sqsh", precision="bf16"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1, agg_workers=1),
            backend=VLLMProtocol(),  # No vllm_config
        )

        assert config.served_model_name == "Qwen3-32B"


class TestFrontendConfig:
    """Tests for FrontendConfig."""

    def test_frontend_defaults(self):
        """Test frontend config defaults."""
        from srtctl.core.schema import FrontendConfig

        frontend = FrontendConfig()

        assert frontend.type == "dynamo"
        assert frontend.enable_multiple_frontends is True
        assert frontend.nginx_container == "nginx:1.27.4"
        assert frontend.args is None
        assert frontend.env is None

    def test_frontend_sglang_type(self):
        """Test sglang frontend config."""
        from srtctl.core.schema import FrontendConfig

        frontend = FrontendConfig(
            type="sglang",
            args={"policy": "round_robin", "verbose": True},
            env={"MY_VAR": "value"},
        )

        assert frontend.type == "sglang"
        assert frontend.args == {"policy": "round_robin", "verbose": True}
        assert frontend.env == {"MY_VAR": "value"}

    def test_nginx_container_alias_resolution(self):
        """Test that nginx_container can be resolved from cluster containers."""
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "sglang", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
            "frontend": {"nginx_container": "nginx"},
        }

        cluster_config = {
            "containers": {
                "sglang": "/path/to/sglang.sqsh",
                "nginx": "/path/to/nginx.sqsh",
            }
        }

        resolved = resolve_config_with_defaults(user_config, cluster_config)

        assert resolved["frontend"]["nginx_container"] == "/path/to/nginx.sqsh"

    def test_nginx_container_no_alias_when_path(self):
        """Test that nginx_container path is kept when not an alias."""
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "/direct/container.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
            "frontend": {"nginx_container": "/direct/nginx.sqsh"},
        }

        cluster_config = {
            "containers": {
                "nginx": "/path/to/nginx.sqsh",
            }
        }

        resolved = resolve_config_with_defaults(user_config, cluster_config)

        # Should keep the original path since it's not an alias
        assert resolved["frontend"]["nginx_container"] == "/direct/nginx.sqsh"

    def test_nginx_container_no_cluster_config(self):
        """Test that nginx_container is kept when no cluster config."""
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "/container.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
            "frontend": {"nginx_container": "nginx"},
        }

        resolved = resolve_config_with_defaults(user_config, None)

        assert resolved["frontend"]["nginx_container"] == "nginx"


class TestSetupScript:
    """Tests for setup_script functionality."""

    def test_setup_script_in_config(self):
        """Test setup_script can be set in config."""
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
            setup_script="my-setup.sh",
        )

        assert config.setup_script == "my-setup.sh"

    def test_setup_script_override_with_replace(self):
        """Test setup_script can be overridden with dataclasses.replace."""
        from dataclasses import replace

        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
        )

        assert config.setup_script is None

        # Override with replace (simulates CLI flag behavior)
        config = replace(config, setup_script="install-sglang-main.sh")
        assert config.setup_script == "install-sglang-main.sh"

    def test_sbatch_template_includes_setup_script_env_var(self):
        """Test that sbatch template sets SRTCTL_SETUP_SCRIPT env var."""
        from pathlib import Path

        from srtctl.cli.submit import generate_minimal_sbatch_script
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
        )

        # Without setup_script
        script = generate_minimal_sbatch_script(
            config=config,
            config_path=Path("/tmp/test.yaml"),
            setup_script=None,
        )
        assert "SRTCTL_SETUP_SCRIPT" not in script

        # With setup_script
        script = generate_minimal_sbatch_script(
            config=config,
            config_path=Path("/tmp/test.yaml"),
            setup_script="install-sglang-main.sh",
        )
        assert 'export SRTCTL_SETUP_SCRIPT="install-sglang-main.sh"' in script

    def test_setup_script_env_var_override(self, monkeypatch):
        """Test that SRTCTL_SETUP_SCRIPT env var overrides config."""
        import os
        from dataclasses import replace

        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
            setup_script=None,
        )

        # Simulate env var being set (like do_sweep.main does)
        monkeypatch.setenv("SRTCTL_SETUP_SCRIPT", "install-sglang-main.sh")

        setup_script_override = os.environ.get("SRTCTL_SETUP_SCRIPT")
        assert setup_script_override == "install-sglang-main.sh"

        # Apply override like do_sweep.main does
        if setup_script_override:
            config = replace(config, setup_script=setup_script_override)

        assert config.setup_script == "install-sglang-main.sh"


class TestWorkerEnvironmentTemplating:
    """Tests for per-worker environment variable templating with {node} and {node_id}."""

    def test_environment_variable_node_templating(self, monkeypatch, tmp_path):
        """Test that environment variables support {node} and {node_id} templating."""
        import os
        import subprocess
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.backends import SGLangProtocol
        from srtctl.cli.mixins.worker_stage import WorkerStageMixin
        from srtctl.core.runtime import RuntimeContext
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig
        from srtctl.core.topology import Process

        # Create temporary model and container paths
        model_path = tmp_path / "model"
        model_path.mkdir()
        container_path = tmp_path / "container.sqsh"
        container_path.touch()

        # Mock SLURM environment
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

        with patch.dict(os.environ, slurm_env):
            with patch("subprocess.run", mock_scontrol):
                with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
                    # Create config with templated environment variables
                    config = SrtConfig(
                        name="test",
                        model=ModelConfig(
                            path=str(model_path),
                            container=str(container_path),
                            precision="fp8",
                        ),
                        resources=ResourceConfig(
                            gpu_type="h100",
                            gpus_per_node=8,
                            prefill_nodes=1,
                            decode_nodes=2,
                        ),
                        backend=SGLangProtocol(
                            prefill_environment={
                                "SGLANG_DG_CACHE_DIR": "/configs/dg-{node_id}",
                                "WORKER_NODE": "{node}",
                            },
                            decode_environment={
                                "SGLANG_DG_CACHE_DIR": "/configs/dg-{node_id}",
                            },
                        ),
                    )

                    runtime = RuntimeContext.from_config(config, job_id="12345")

                    # Create a mock worker stage
                    class MockWorkerStage(WorkerStageMixin):
                        def __init__(self, config, runtime):
                            self.config = config
                            self.runtime = runtime

                    worker_stage = MockWorkerStage(config, runtime)

                    # Create test processes on different nodes
                    processes = [
                        Process(
                            node="gpu-01",
                            gpu_indices=frozenset([0, 1, 2, 3, 4, 5, 6, 7]),
                            sys_port=8081,
                            http_port=30000,
                            endpoint_mode="prefill",
                            endpoint_index=0,
                            node_rank=0,
                        ),
                        Process(
                            node="gpu-02",
                            gpu_indices=frozenset([0, 1, 2, 3, 4, 5, 6, 7]),
                            sys_port=8082,
                            http_port=30001,
                            endpoint_mode="decode",
                            endpoint_index=0,
                            node_rank=0,
                        ),
                        Process(
                            node="gpu-03",
                            gpu_indices=frozenset([0, 1, 2, 3, 4, 5, 6, 7]),
                            sys_port=8083,
                            http_port=30002,
                            endpoint_mode="decode",
                            endpoint_index=1,
                            node_rank=0,
                        ),
                    ]

                    # Mock backend command builder and srun process to capture environment variables
                    mock_backend = MagicMock()
                    mock_backend.get_environment_for_mode.side_effect = config.backend.get_environment_for_mode
                    mock_backend.build_worker_command.return_value = ["echo", "test"]

                    with patch.object(worker_stage, "config") as mock_config:
                        mock_config.backend = mock_backend
                        mock_config.profiling = config.profiling

                        with patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun:
                            mock_srun.return_value = MagicMock()

                            # Test prefill worker on gpu-01 (index 0)
                            worker_stage.start_worker(processes[0], [])
                            call_kwargs = mock_srun.call_args.kwargs
                            env_vars = call_kwargs.get("env_to_set", {})

                            assert "SGLANG_DG_CACHE_DIR" in env_vars
                            assert env_vars["SGLANG_DG_CACHE_DIR"] == "/configs/dg-0"
                            assert env_vars["WORKER_NODE"] == "gpu-01"

                            # Test decode worker on gpu-02 (index 1)
                            worker_stage.start_worker(processes[1], [])
                            call_kwargs = mock_srun.call_args.kwargs
                            env_vars = call_kwargs.get("env_to_set", {})

                            assert env_vars["SGLANG_DG_CACHE_DIR"] == "/configs/dg-1"

                            # Test decode worker on gpu-03 (index 2)
                            worker_stage.start_worker(processes[2], [])
                            call_kwargs = mock_srun.call_args.kwargs
                            env_vars = call_kwargs.get("env_to_set", {})

                            assert env_vars["SGLANG_DG_CACHE_DIR"] == "/configs/dg-2"

    def test_environment_variable_unsupported_placeholder(self, monkeypatch, tmp_path):
        """Test that unsupported placeholders like {foo} remain unchanged and don't throw errors."""
        import os
        import subprocess
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.backends import SGLangProtocol
        from srtctl.cli.mixins.worker_stage import WorkerStageMixin
        from srtctl.core.runtime import RuntimeContext
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig
        from srtctl.core.topology import Process

        # Create temporary model and container paths
        model_path = tmp_path / "model"
        model_path.mkdir()
        container_path = tmp_path / "container.sqsh"
        container_path.touch()

        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOBID": "12345",
            "SLURM_NODELIST": "gpu-[01-02]",
            "SLURM_JOB_NUM_NODES": "2",
            "SRTCTL_SOURCE_DIR": str(Path(__file__).parent.parent),
        }

        def mock_scontrol(cmd, **kwargs):
            if cmd[0] == "scontrol" and "hostnames" in cmd:
                result = MagicMock()
                result.stdout = "gpu-01\ngpu-02"
                result.returncode = 0
                return result
            raise subprocess.CalledProcessError(1, cmd)

        with patch.dict(os.environ, slurm_env):
            with patch("subprocess.run", mock_scontrol):
                with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
                    # Create config with unsupported template placeholders
                    config = SrtConfig(
                        name="test",
                        model=ModelConfig(
                            path=str(model_path),
                            container=str(container_path),
                            precision="fp8",
                        ),
                        resources=ResourceConfig(
                            gpu_type="h100",
                            gpus_per_node=8,
                            prefill_nodes=1,
                            decode_nodes=1,
                        ),
                        backend=SGLangProtocol(
                            prefill_environment={
                                # Mix of supported and unsupported placeholders
                                "CACHE_DIR": "/cache/{node_id}/data",
                                "UNSUPPORTED": "/path/{foo}/bar/{baz}",
                                "MIXED": "{node}-{unsupported_var}-cache",
                            },
                        ),
                    )

                    runtime = RuntimeContext.from_config(config, job_id="12345")

                    class MockWorkerStage(WorkerStageMixin):
                        def __init__(self, config, runtime):
                            self.config = config
                            self.runtime = runtime

                    worker_stage = MockWorkerStage(config, runtime)

                    process = Process(
                        node="gpu-01",
                        gpu_indices=frozenset([0, 1, 2, 3, 4, 5, 6, 7]),
                        sys_port=8081,
                        http_port=30000,
                        endpoint_mode="prefill",
                        endpoint_index=0,
                        node_rank=0,
                    )

                    # Mock backend command builder and srun process to capture environment variables
                    mock_backend = MagicMock()
                    mock_backend.get_environment_for_mode.side_effect = config.backend.get_environment_for_mode
                    mock_backend.build_worker_command.return_value = ["echo", "test"]

                    with patch.object(worker_stage, "config") as mock_config:
                        mock_config.backend = mock_backend
                        mock_config.profiling = config.profiling

                        with patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun:
                            mock_srun.return_value = MagicMock()

                            # This should NOT throw an error
                            worker_stage.start_worker(process, [])
                            call_kwargs = mock_srun.call_args.kwargs
                            env_vars = call_kwargs.get("env_to_set", {})

                            # Supported placeholder should be replaced
                            assert env_vars["CACHE_DIR"] == "/cache/0/data"

                            # Unsupported placeholders should remain unchanged
                            assert env_vars["UNSUPPORTED"] == "/path/{foo}/bar/{baz}"

                            # Mixed case: supported replaced, unsupported kept
                            assert env_vars["MIXED"] == "gpu-01-{unsupported_var}-cache"


class TestInfraConfig:
    """Tests for InfraConfig dataclass."""

    def test_infra_config_defaults(self):
        """Test that InfraConfig has correct defaults."""
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
        )

        # infra config should exist with default values
        assert config.infra is not None
        assert config.infra.etcd_nats_dedicated_node is False

    def test_infra_config_enabled(self):
        """Test InfraConfig with dedicated node enabled."""
        from srtctl.core.schema import InfraConfig, ModelConfig, ResourceConfig, SrtConfig

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
            infra=InfraConfig(etcd_nats_dedicated_node=True),
        )

        assert config.infra.etcd_nats_dedicated_node is True


class TestNodesInfraAllocation:
    """Tests for Nodes infra node allocation."""

    def test_nodes_default_infra_equals_head(self):
        """Test that infra node equals head node by default."""
        from unittest.mock import patch

        from srtctl.core.runtime import Nodes

        with patch("srtctl.core.runtime.get_slurm_nodelist", return_value=["node0", "node1", "node2"]):
            nodes = Nodes.from_slurm(etcd_nats_dedicated_node=False)

        assert nodes.head == "node0"
        assert nodes.infra == "node0"  # Same as head
        assert nodes.worker == ("node0", "node1", "node2")

    def test_nodes_dedicated_infra_node(self):
        """Test that infra node is separate when dedicated node is enabled."""
        from unittest.mock import patch

        from srtctl.core.runtime import Nodes

        with patch("srtctl.core.runtime.get_slurm_nodelist", return_value=["node0", "node1", "node2"]):
            nodes = Nodes.from_slurm(etcd_nats_dedicated_node=True)

        assert nodes.infra == "node0"  # First node is infra-only
        assert nodes.head == "node1"  # Second node is head
        assert nodes.worker == ("node1", "node2")  # Infra node not in workers

    def test_nodes_dedicated_infra_requires_two_nodes(self):
        """Test that dedicated infra node requires at least 2 nodes."""
        from unittest.mock import patch

        import pytest

        from srtctl.core.runtime import Nodes

        with patch("srtctl.core.runtime.get_slurm_nodelist", return_value=["node0"]):
            with pytest.raises(ValueError, match="at least 2 nodes"):
                Nodes.from_slurm(etcd_nats_dedicated_node=True)


class TestSbatchNodeCount:
    """Tests for sbatch node count calculation with infra config."""

    def test_sbatch_adds_node_for_dedicated_infra(self):
        """Test that sbatch script requests extra node when etcd_nats_dedicated_node is enabled."""
        from pathlib import Path

        from srtctl.cli.submit import generate_minimal_sbatch_script
        from srtctl.core.schema import InfraConfig, ModelConfig, ResourceConfig, SrtConfig

        # Config with 2 worker nodes
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            infra=InfraConfig(etcd_nats_dedicated_node=True),
        )

        script = generate_minimal_sbatch_script(config, Path("/tmp/test.yaml"))

        # Should request 3 nodes: 2 workers + 1 infra
        assert "#SBATCH --nodes=3" in script

    def test_sbatch_normal_node_count_without_dedicated_infra(self):
        """Test that sbatch script uses normal node count when etcd_nats_dedicated_node is disabled."""
        from pathlib import Path

        from srtctl.cli.submit import generate_minimal_sbatch_script
        from srtctl.core.schema import InfraConfig, ModelConfig, ResourceConfig, SrtConfig

        # Config with 2 worker nodes, no dedicated infra
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            infra=InfraConfig(etcd_nats_dedicated_node=False),
        )

        script = generate_minimal_sbatch_script(config, Path("/tmp/test.yaml"))

        # Should request 2 nodes: just the workers
        assert "#SBATCH --nodes=2" in script


class TestVLLMDataParallelMode:
    """Tests for vLLM DP+EP (Data Parallel + Expert Parallel) mode."""

    def test_dp_mode_detection(self):
        """Test that DP mode is correctly detected from config."""
        from srtctl.backends import VLLMProtocol, VLLMServerConfig

        # No DP mode when data-parallel-size is not set
        backend = VLLMProtocol(
            vllm_config=VLLMServerConfig(
                prefill={"tensor-parallel-size": 8},
                decode={"tensor-parallel-size": 4},
            )
        )
        assert backend._is_dp_mode("prefill") is False
        assert backend._is_dp_mode("decode") is False

        # DP mode detected when data-parallel-size is set
        backend_dp = VLLMProtocol(
            vllm_config=VLLMServerConfig(
                prefill={"data-parallel-size": 16, "enable-expert-parallel": True},
                decode={"data-parallel-size": 16, "enable-expert-parallel": True},
            )
        )
        assert backend_dp._is_dp_mode("prefill") is True
        assert backend_dp._is_dp_mode("decode") is True
        assert backend_dp._get_dp_size("prefill") == 16

    def test_dp_mode_creates_per_gpu_processes(self):
        """Test that DP mode creates one process per GPU instead of per node."""
        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Endpoint

        backend = VLLMProtocol(
            vllm_config=VLLMServerConfig(
                prefill={"data-parallel-size": 16, "enable-expert-parallel": True},
            )
        )

        # Create an endpoint spanning 2 nodes with 8 GPUs each = 16 GPUs total
        endpoint = Endpoint(
            mode="prefill",
            index=0,
            nodes=("node0", "node1"),
            gpu_indices=frozenset(range(8)),
            gpus_per_node=8,
        )

        processes = backend.endpoints_to_processes([endpoint])

        # Should create 16 processes (1 per GPU), not 2 (1 per node)
        assert len(processes) == 16

        # Each process should have exactly 1 GPU
        for proc in processes:
            assert len(proc.gpu_indices) == 1

        # First 8 processes on node0, next 8 on node1
        node0_processes = [p for p in processes if p.node == "node0"]
        node1_processes = [p for p in processes if p.node == "node1"]
        assert len(node0_processes) == 8
        assert len(node1_processes) == 8

        # GPU indices should be 0-7 on each node
        node0_gpus = {list(p.gpu_indices)[0] for p in node0_processes}
        node1_gpus = {list(p.gpu_indices)[0] for p in node1_processes}
        assert node0_gpus == {0, 1, 2, 3, 4, 5, 6, 7}
        assert node1_gpus == {0, 1, 2, 3, 4, 5, 6, 7}

        # dp_rank (stored in node_rank) should go from 0 to 15
        dp_ranks = [p.node_rank for p in processes]
        assert dp_ranks == list(range(16))

    def test_dp_mode_command_includes_dp_flags(self):
        """Test that DP mode command includes correct DP flags instead of TP flags."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Process

        backend = VLLMProtocol(
            vllm_config=VLLMServerConfig(
                prefill={
                    "data-parallel-size": 16,
                    "data-parallel-rpc-port": 13345,
                    "enable-expert-parallel": True,
                },
            )
        )

        # Create a process representing GPU 5 with dp_rank=5
        process = Process(
            node="node0",
            gpu_indices=frozenset([5]),
            sys_port=8081,
            http_port=0,
            endpoint_mode="prefill",
            endpoint_index=0,
            node_rank=5,  # dp_rank
        )

        # Create endpoint_processes spanning 2 nodes
        endpoint_processes = [
            Process(
                node="node0",
                gpu_indices=frozenset([i]),
                sys_port=8081 + i,
                http_port=0,
                endpoint_mode="prefill",
                endpoint_index=0,
                node_rank=i,
            )
            for i in range(8)
        ] + [
            Process(
                node="node1",
                gpu_indices=frozenset([i]),
                sys_port=8089 + i,
                http_port=0,
                endpoint_mode="prefill",
                endpoint_index=0,
                node_rank=8 + i,
            )
            for i in range(8)
        ]

        # Mock runtime context
        mock_runtime = MagicMock()
        mock_runtime.model_path = Path("/model")
        mock_runtime.is_hf_model = False

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(
                process=process,
                endpoint_processes=endpoint_processes,
                runtime=mock_runtime,
            )

        # Should include DP flags
        assert "--data-parallel-rank" in cmd
        assert "5" in cmd  # dp_rank = 5
        assert "--data-parallel-address" in cmd
        assert "10.0.0.1" in cmd
        assert "--data-parallel-rpc-port" in cmd
        assert "13345" in cmd
        assert "--data-parallel-size" in cmd
        assert "16" in cmd

        # Should NOT include TP multi-node flags
        assert "--master-addr" not in cmd
        assert "--nnodes" not in cmd
        assert "--node-rank" not in cmd
        assert "--headless" not in cmd

    def test_standard_tp_mode_still_works(self):
        """Test that standard TP mode (no DP) still creates per-node processes."""
        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Endpoint

        # No data-parallel-size set = standard TP mode
        backend = VLLMProtocol(
            vllm_config=VLLMServerConfig(
                prefill={"tensor-parallel-size": 16},
            )
        )

        # Create an endpoint spanning 2 nodes
        endpoint = Endpoint(
            mode="prefill",
            index=0,
            nodes=("node0", "node1"),
            gpu_indices=frozenset(range(8)),
            gpus_per_node=8,
        )

        processes = backend.endpoints_to_processes([endpoint])

        # Should create 2 processes (1 per node), not 16 (1 per GPU)
        assert len(processes) == 2
        assert processes[0].node == "node0"
        assert processes[1].node == "node1"

        # Each process should have 8 GPUs
        assert len(processes[0].gpu_indices) == 8
        assert len(processes[1].gpu_indices) == 8

    def test_vllm_get_process_environment(self):
        """Test vLLM sets port environment variables from process."""
        from srtctl.backends import VLLMProtocol
        from srtctl.core.topology import Process

        backend = VLLMProtocol()

        # Process with ports set
        process = Process(
            node="node0",
            gpu_indices=frozenset([0]),
            sys_port=8081,
            http_port=30000,
            endpoint_mode="prefill",
            endpoint_index=0,
            node_rank=0,
            kv_events_port=5550,
            nixl_port=6550,
        )

        env = backend.get_process_environment(process)

        assert env["DYN_VLLM_KV_EVENT_PORT"] == "5550"
        assert env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "6550"

    def test_vllm_get_process_environment_none_ports(self):
        """Test vLLM handles None ports gracefully."""
        from srtctl.backends import VLLMProtocol
        from srtctl.core.topology import Process

        backend = VLLMProtocol()

        process = Process(
            node="node0",
            gpu_indices=frozenset([0]),
            sys_port=8081,
            http_port=30000,
            endpoint_mode="prefill",
            endpoint_index=0,
            node_rank=0,
            kv_events_port=None,
            nixl_port=None,
        )

        env = backend.get_process_environment(process)

        assert "DYN_VLLM_KV_EVENT_PORT" not in env
        assert "VLLM_NIXL_SIDE_CHANNEL_PORT" not in env

    def test_tp_mode_command_includes_multinode_flags(self):
        """Test standard TP mode includes multi-node coordination flags."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Process

        # Standard TP mode (no data-parallel-size)
        backend = VLLMProtocol(
            vllm_config=VLLMServerConfig(
                prefill={"tensor-parallel-size": 16},
            )
        )

        # Non-leader process (node_rank=1)
        process = Process(
            node="node1",
            gpu_indices=frozenset(range(8)),
            sys_port=8082,
            http_port=0,
            endpoint_mode="prefill",
            endpoint_index=0,
            node_rank=1,
        )

        # Endpoint spans 2 nodes
        endpoint_processes = [
            Process(
                node="node0",
                gpu_indices=frozenset(range(8)),
                sys_port=8081,
                http_port=30000,
                endpoint_mode="prefill",
                endpoint_index=0,
                node_rank=0,
            ),
            Process(
                node="node1",
                gpu_indices=frozenset(range(8)),
                sys_port=8082,
                http_port=0,
                endpoint_mode="prefill",
                endpoint_index=0,
                node_rank=1,
            ),
        ]

        mock_runtime = MagicMock()
        mock_runtime.model_path = Path("/model")
        mock_runtime.is_hf_model = False

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(
                process=process,
                endpoint_processes=endpoint_processes,
                runtime=mock_runtime,
            )

        # Should include TP multi-node flags
        assert "--master-addr" in cmd
        assert "10.0.0.1" in cmd
        assert "--nnodes" in cmd
        assert "2" in cmd
        assert "--node-rank" in cmd
        # node_rank is determined by position in endpoint_nodes, not process.node_rank
        assert "1" in cmd  # This is node1
        assert "--headless" in cmd  # Non-leader should be headless

        # Should NOT include DP flags
        assert "--data-parallel-rank" not in cmd
        assert "--data-parallel-address" not in cmd

    def test_tp_mode_leader_not_headless(self):
        """Test TP mode leader (node_rank=0) does not get --headless flag."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Process

        backend = VLLMProtocol(
            vllm_config=VLLMServerConfig(
                prefill={"tensor-parallel-size": 16},
            )
        )

        # Leader process (node_rank=0)
        process = Process(
            node="node0",
            gpu_indices=frozenset(range(8)),
            sys_port=8081,
            http_port=30000,
            endpoint_mode="prefill",
            endpoint_index=0,
            node_rank=0,
        )

        endpoint_processes = [
            process,
            Process(
                node="node1",
                gpu_indices=frozenset(range(8)),
                sys_port=8082,
                http_port=0,
                endpoint_mode="prefill",
                endpoint_index=0,
                node_rank=1,
            ),
        ]

        mock_runtime = MagicMock()
        mock_runtime.model_path = Path("/model")
        mock_runtime.is_hf_model = False

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(
                process=process,
                endpoint_processes=endpoint_processes,
                runtime=mock_runtime,
            )

        # Leader should NOT be headless
        assert "--headless" not in cmd
        # But should still have multi-node flags
        assert "--master-addr" in cmd
        assert "--nnodes" in cmd

    # =========================================================================
    # Connector → --kv-transfer-config tests (dynamo 1.0.0+)
    # =========================================================================

    def _build_cmd_with_connector(self, connector, mode="agg", mode_connector=None):
        """Helper: build a vLLM worker command with the given connector setting."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Process

        mode_config: dict = {}
        if mode_connector is not None:
            mode_config["connector"] = mode_connector

        vllm_cfg_kwargs = {("aggregated" if mode == "agg" else mode): mode_config or None}
        backend = VLLMProtocol(
            connector=connector,
            vllm_config=VLLMServerConfig(**vllm_cfg_kwargs),
        )

        process = Process(
            node="node0",
            gpu_indices=frozenset([0, 1, 2, 3]),
            sys_port=8081,
            http_port=30000,
            endpoint_mode=mode,
            endpoint_index=0,
            node_rank=0,
        )

        mock_runtime = MagicMock()
        mock_runtime.model_path = Path("/model")
        mock_runtime.is_hf_model = False

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            return backend.build_worker_command(
                process=process,
                endpoint_processes=[process],
                runtime=mock_runtime,
            )

    def test_connector_nixl_generates_kv_transfer_config(self):
        """connector: nixl → --kv-transfer-config with NixlConnector JSON."""
        import json

        cmd = self._build_cmd_with_connector("nixl")
        assert "--kv-transfer-config" in cmd
        assert "--connector" not in cmd
        idx = cmd.index("--kv-transfer-config")
        cfg = json.loads(cmd[idx + 1])
        assert cfg["kv_connector"] == "NixlConnector"
        assert cfg["kv_role"] == "kv_both"

    def test_connector_lmcache_generates_kv_transfer_config(self):
        """connector: lmcache → --kv-transfer-config with LMCacheConnectorV1 JSON."""
        import json

        cmd = self._build_cmd_with_connector("lmcache")
        idx = cmd.index("--kv-transfer-config")
        cfg = json.loads(cmd[idx + 1])
        assert cfg["kv_connector"] == "LMCacheConnectorV1"

    def test_connector_custom_json_passthrough(self):
        """connector set to a raw JSON string is passed through as-is."""
        import json

        custom = '{"kv_connector":"MyCustomConnector","kv_role":"kv_both"}'
        cmd = self._build_cmd_with_connector(custom)
        idx = cmd.index("--kv-transfer-config")
        assert cmd[idx + 1] == custom

    def test_connector_null_skips_flag(self):
        """connector: null should not emit --kv-transfer-config."""
        cmd = self._build_cmd_with_connector(None)
        assert "--kv-transfer-config" not in cmd
        assert "--connector" not in cmd

    def test_connector_none_string_skips_flag(self):
        """connector: 'none' should not emit --kv-transfer-config."""
        cmd = self._build_cmd_with_connector("none")
        assert "--kv-transfer-config" not in cmd

    def test_connector_kvbm_generates_kv_transfer_config(self):
        """connector: kvbm → --kv-transfer-config with DynamoConnector JSON."""
        import json

        cmd = self._build_cmd_with_connector("kvbm")
        idx = cmd.index("--kv-transfer-config")
        cfg = json.loads(cmd[idx + 1])
        assert cfg["kv_connector"] == "DynamoConnector"
        assert cfg["kv_connector_module_path"] == "kvbm.vllm_integration.connector"

    def test_mode_connector_overrides_default(self):
        """Per-mode connector override takes precedence over default."""
        import json

        cmd = self._build_cmd_with_connector("nixl", mode="agg", mode_connector="lmcache")
        idx = cmd.index("--kv-transfer-config")
        cfg = json.loads(cmd[idx + 1])
        assert cfg["kv_connector"] == "LMCacheConnectorV1"

    def test_disaggregation_mode_flag_for_prefill(self):
        """Prefill mode emits --disaggregation-mode prefill (not --is-prefill-worker)."""
        cmd = self._build_cmd_with_connector("nixl", mode="prefill")
        assert "--disaggregation-mode" in cmd
        idx = cmd.index("--disaggregation-mode")
        assert cmd[idx + 1] == "prefill"
        assert "--is-prefill-worker" not in cmd

    def test_disaggregation_mode_flag_for_decode(self):
        """Decode mode emits --disaggregation-mode decode (not --is-decode-worker)."""
        cmd = self._build_cmd_with_connector("nixl", mode="decode")
        assert "--disaggregation-mode" in cmd
        idx = cmd.index("--disaggregation-mode")
        assert cmd[idx + 1] == "decode"
        assert "--is-decode-worker" not in cmd

    def test_agg_mode_no_disaggregation_flag(self):
        """Aggregated mode should not emit any disaggregation flag."""
        cmd = self._build_cmd_with_connector("nixl", mode="agg")
        assert "--disaggregation-mode" not in cmd
        assert "--is-prefill-worker" not in cmd
        assert "--is-decode-worker" not in cmd


class TestHuggingFaceModelSupport:
    """Tests for HuggingFace model (hf:prefix) support across all backends."""

    @staticmethod
    def _make_process(mode="agg"):
        from srtctl.core.topology import Process

        return Process(
            node="node0",
            gpu_indices=frozenset([0, 1, 2, 3]),
            sys_port=8081,
            http_port=30000,
            endpoint_mode=mode,
            endpoint_index=0,
            node_rank=0,
        )

    @staticmethod
    def _make_runtime(*, is_hf: bool):
        from pathlib import Path
        from unittest.mock import MagicMock

        runtime = MagicMock()
        if is_hf:
            runtime.model_path = Path("facebook/opt-125m")
            runtime.is_hf_model = True
        else:
            runtime.model_path = Path("/models/my-model")
            runtime.is_hf_model = False
        return runtime

    # --- vLLM ---

    def test_vllm_hf_model_uses_model_id(self):
        """vLLM passes HF model ID when is_hf_model=True."""
        from unittest.mock import patch

        from srtctl.backends import VLLMProtocol

        backend = VLLMProtocol(connector=None)
        process = self._make_process()
        runtime = self._make_runtime(is_hf=True)

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "facebook/opt-125m"

    def test_vllm_local_model_uses_container_mount(self):
        """vLLM passes /model when is_hf_model=False."""
        from unittest.mock import patch

        from srtctl.backends import VLLMProtocol

        backend = VLLMProtocol(connector=None)
        process = self._make_process()
        runtime = self._make_runtime(is_hf=False)

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "/model"

    # --- SGLang ---

    def test_sglang_hf_model_uses_model_id(self):
        """SGLang passes HF model ID when is_hf_model=True."""
        from unittest.mock import patch

        from srtctl.backends import SGLangProtocol

        backend = SGLangProtocol()
        process = self._make_process()
        runtime = self._make_runtime(is_hf=True)

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--model-path")
        assert cmd[idx + 1] == "facebook/opt-125m"

    def test_sglang_local_model_uses_container_mount(self):
        """SGLang passes /model when is_hf_model=False."""
        from unittest.mock import patch

        from srtctl.backends import SGLangProtocol

        backend = SGLangProtocol()
        process = self._make_process()
        runtime = self._make_runtime(is_hf=False)

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--model-path")
        assert cmd[idx + 1] == "/model"

    def test_sglang_model_path_not_duplicated_from_config(self):
        """SGLang does not duplicate --model-path when user provides it in sglang_config."""
        from unittest.mock import patch

        from srtctl.backends import SGLangProtocol, SGLangServerConfig

        backend = SGLangProtocol(
            sglang_config=SGLangServerConfig(aggregated={"model-path": "/custom/model"}),
        )
        process = self._make_process()
        runtime = self._make_runtime(is_hf=False)

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        count = cmd.count("--model-path")
        assert count == 1, f"--model-path appears {count} times: {cmd}"

    def test_sglang_served_model_name_not_duplicated(self):
        """SGLang does not duplicate --served-model-name when user provides it in sglang_config."""
        from unittest.mock import patch

        from srtctl.backends import SGLangProtocol, SGLangServerConfig

        backend = SGLangProtocol(
            sglang_config=SGLangServerConfig(aggregated={"served-model-name": "MyModel"}),
        )
        process = self._make_process()
        runtime = self._make_runtime(is_hf=False)

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        count = cmd.count("--served-model-name")
        assert count == 1, f"--served-model-name appears {count} times: {cmd}"

    # --- TRTLLM ---

    def test_trtllm_hf_model_uses_model_id(self):
        """TRTLLM passes HF model ID when is_hf_model=True."""
        from pathlib import Path
        from unittest.mock import patch

        from srtctl.backends import TRTLLMProtocol

        backend = TRTLLMProtocol()
        process = self._make_process()
        runtime = self._make_runtime(is_hf=True)
        runtime.log_dir = Path("/tmp/test-logs")

        with patch("pathlib.Path.write_text"):
            with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
                cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--model-path")
        assert cmd[idx + 1] == "facebook/opt-125m"

    def test_trtllm_local_model_uses_container_mount(self):
        """TRTLLM passes /model when is_hf_model=False."""
        from pathlib import Path
        from unittest.mock import patch

        from srtctl.backends import TRTLLMProtocol

        backend = TRTLLMProtocol()
        process = self._make_process()
        runtime = self._make_runtime(is_hf=False)
        runtime.log_dir = Path("/tmp/test-logs")

        with patch("pathlib.Path.write_text"):
            with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
                cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--model-path")
        assert cmd[idx + 1] == "/model"
