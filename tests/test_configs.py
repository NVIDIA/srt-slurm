# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for configuration loading and validation."""

import glob
import json
from pathlib import Path

import pytest

from srtctl.backends import SGLangProtocol, SGLangServerConfig
from srtctl.core.schema import SrtConfig
from srtctl.ports import (
    KV_EVENTS_PORT_BASE,
    SGLANG_BOOTSTRAP_PORT_BASE,
    SGLANG_HTTP_PORT_BASE,
    SGLANG_HTTP_PORT_STRIDE,
    VLLM_DATA_PARALLEL_RPC_PORT,
    VLLM_NIXL_PORT_BASE,
)


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


class TestIdentityConfig:
    """Tests for the identity block (virtual identity for runtime verification)."""

    def test_defaults_to_empty(self):
        """IdentityConfig has empty defaults."""
        from srtctl.core.schema import IdentityConfig

        config = IdentityConfig()
        assert config.model.repo is None
        assert config.model.revision is None
        assert config.frameworks == {}

    def test_with_values(self):
        """IdentityConfig stores model and framework info."""
        from srtctl.core.schema import IdentityConfig, IdentityModelConfig

        config = IdentityConfig(
            model=IdentityModelConfig(repo="nvidia/Kimi-K2.5-NVFP4", revision="abc123"),
            frameworks={"dynamo": "1.0.0", "tensorrt_llm": "1.3.0rc9"},
        )
        assert config.model.repo == "nvidia/Kimi-K2.5-NVFP4"
        assert config.model.revision == "abc123"
        assert config.frameworks["dynamo"] == "1.0.0"
        assert config.frameworks["tensorrt_llm"] == "1.3.0rc9"

    def test_marshmallow_roundtrip(self):
        """Schema dump/load preserves identity fields."""
        from srtctl.core.schema import IdentityConfig, IdentityModelConfig

        original = IdentityConfig(
            model=IdentityModelConfig(repo="nvidia/Kimi-K2.5-NVFP4", revision="abc123"),
            frameworks={"dynamo": "1.0.0"},
        )
        schema = IdentityConfig.Schema()
        dumped = schema.dump(original)
        loaded = schema.load(dumped)
        assert loaded.model.repo == "nvidia/Kimi-K2.5-NVFP4"
        assert loaded.frameworks["dynamo"] == "1.0.0"

    def test_model_config_is_clean(self):
        """ModelConfig has no virtual identity fields (moved to IdentityConfig)."""
        from srtctl.core.schema import ModelConfig

        config = ModelConfig(path="/model", container="/c.sqsh", precision="fp8")
        assert not hasattr(config, "name")
        assert not hasattr(config, "container_image")
        assert not hasattr(config, "container_digest")


class TestDynamoConfig:
    """Tests for DynamoConfig."""

    def test_default_version(self):
        """Default is version 0.8.0."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig()
        assert config.version == "0.8.0"
        assert config.hash is None
        assert config.top_of_tree is False
        assert config.wheel is None
        assert not config.needs_source_install

    def test_version_install_command(self):
        """Version config generates pip install command."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(version="0.8.0")
        cmd = config.get_install_commands()
        assert "pip install" in cmd
        assert "ai-dynamo-runtime==0.8.0" in cmd
        assert "ai-dynamo==0.8.0" in cmd

    def test_wheel_install_command(self):
        """Wheel config installs ai-dynamo plus runtime without source build."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(wheel="1.2.0.dev20260426")
        cmd = config.get_install_commands()

        assert config.version is None
        assert config.needs_source_install is False
        assert "/srtctl-runtime/dynamo_wheels.py" in cmd
        assert "ai_dynamo-1.2.0.dev20260426-py3-none-any.whl" in cmd
        assert "install-ai-dynamo.sh" not in cmd
        assert "--find-links" not in cmd
        assert "configs/wheels" not in cmd
        assert "--extra-index-url" not in cmd
        assert "maturin" not in cmd
        assert "git clone" not in cmd

    def test_install_command_serialized_with_flock(self):
        """Install command is wrapped in a per-environment flock + sentinel.

        With --ntasks-per-node > 1 (e.g. TRTLLM), co-located tasks race
        concurrent pip installs into the shared container site-packages. The
        wrapper serializes them and lets tasks after the first skip. The lock
        is anchored in the Python env (sys.prefix), NOT /tmp, so co-located
        containers with a bind-mounted /tmp don't collide.
        """
        from srtctl.core.schema import DynamoConfig

        for config in (
            DynamoConfig(version="0.8.0"),
            DynamoConfig(wheel="1.2.0.dev20260426"),
        ):
            cmd = config.get_install_commands()
            # Lock dir resolved from the active Python env, not /tmp.
            assert "sys.prefix" in cmd
            assert "/tmp/srtctl_dynamo_install" not in cmd
            # FD 200 node-local; the hash source install nests flock -x 201 on
            # the /configs cache lock; distinct FDs keep the locks independent.
            assert "flock -x 200" in cmd
            assert '$DYN_LOCK_DIR/.srtctl_dynamo_install.lock' in cmd
            assert '$DYN_LOCK_DIR/.srtctl_dynamo_install.complete' in cmd
            # Sentinel short-circuits repeat installs; touched on success.
            assert 'touch "$DYN_LOCK_DIR/.srtctl_dynamo_install.complete"' in cmd
            assert '200>"$DYN_LOCK_DIR/.srtctl_dynamo_install.lock"' in cmd

    def test_hash_install_command(self):
        """Hash config generates a cache-aware source-install command.

        The bash should: (1) check the /configs cache, (2) clone+build under
        flock if cold, (3) install from the cache regardless. Cache is keyed
        by hash so bumping the hash forces a rebuild.
        """
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(hash="abc123")
        assert config.version is None  # Auto-cleared
        assert config.needs_source_install
        cmd = config.get_install_commands()

        # Cache lookup + flock-protected cold build
        assert "/configs/dynamo-wheels/abc123" in cmd
        assert "/configs/dynamo-wheels/abc123/.complete" in cmd
        assert "flock -x 201" in cmd
        assert "/configs/dynamo-wheels/.abc123.lock" in cmd

        # Cold-cache build still does git clone + checkout + maturin build
        assert "git clone" in cmd
        assert "git checkout abc123" in cmd
        assert "maturin build" in cmd
        assert "protobuf-compiler" in cmd

        # maturin must be force-reinstalled — a plain install no-ops on images
        # shipping the module without a console script (see schema.py).
        assert "--force-reinstall --quiet maturin" in cmd

        # Cache populate: wheel + tarball + sentinel
        assert "ai_dynamo_runtime*.whl" in cmd
        assert "dynamo-src.tar.gz" in cmd
        assert "touch /configs/dynamo-wheels/abc123/.complete" in cmd

        # Final install from cache
        assert (
            "pip install --break-system-packages --force-reinstall /configs/dynamo-wheels/abc123/ai_dynamo_runtime-*.whl"
            in cmd
        )
        assert "tar -xzf /configs/dynamo-wheels/abc123/dynamo-src.tar.gz" in cmd
        assert "pip install --break-system-packages -e /tmp/dynamo-src/dynamo" in cmd

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

        # Both branches (sglang + portable) must force-reinstall maturin — the
        # portable branch previously used a guarded plain install that no-ops on
        # images shipping maturin without a console script.
        sglang_branch, portable_branch = config._build_install_commands().split("else", 1)
        assert "--force-reinstall --quiet maturin" in sglang_branch
        assert "--force-reinstall --quiet maturin" in portable_branch

    def test_hash_and_top_of_tree_not_allowed(self):
        """Cannot specify both hash and top_of_tree."""
        from srtctl.core.schema import DynamoConfig

        with pytest.raises(ValueError, match="Cannot specify both"):
            DynamoConfig(hash="abc123", top_of_tree=True)

    def test_hash_and_wheel_not_allowed(self):
        """Cannot specify both hash and wheel."""
        from srtctl.core.schema import DynamoConfig

        with pytest.raises(ValueError, match="Cannot specify both"):
            DynamoConfig(hash="abc123", wheel="1.2.0.dev20260426")

    def test_wheel_filename_not_allowed(self):
        """Wheel config takes a package version, not an artifact filename."""
        from srtctl.core.schema import DynamoConfig

        with pytest.raises(ValueError, match="package version"):
            DynamoConfig(wheel="ai_dynamo-1.2.0.dev20260426-py3-none-any.whl")

    def test_wheel_version_required(self):
        """Wheel config must provide an exact package version."""
        from srtctl.core.schema import DynamoConfig

        with pytest.raises(ValueError, match="non-empty package version"):
            DynamoConfig(wheel="")

    def test_wheel_environment_from_version(self):
        """Wheel version is converted to setup/prefetch environment."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(wheel="1.2.0.dev20260426")

        assert config.wheel_version == "1.2.0.dev20260426"
        assert config.wheel_name == "ai_dynamo-1.2.0.dev20260426-py3-none-any.whl"
        assert config.get_wheel_environment() == {
            "DYNAMO_VERSION": "1.2.0.dev20260426",
            "DYNAMO_WHEEL_NAME": "ai_dynamo-1.2.0.dev20260426-py3-none-any.whl",
        }

    def test_request_plane_default_tcp(self):
        """Default request_plane is 'tcp'."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig()
        assert config.request_plane == "tcp"

    def test_request_plane_override_default_to_nats(self):
        """request_plane='nats' overrides the TCP default."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(request_plane="nats")
        assert config.request_plane == "nats"

    def test_request_plane_tcp(self):
        """request_plane='tcp' is accepted."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(request_plane="tcp")
        assert config.request_plane == "tcp"

    def test_request_plane_http(self):
        """request_plane='http' is accepted."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(request_plane="http")
        assert config.request_plane == "http"

    def test_request_plane_invalid(self):
        """Invalid request_plane raises ValueError."""
        from srtctl.core.schema import DynamoConfig

        with pytest.raises(ValueError, match="Invalid request_plane"):
            DynamoConfig(request_plane="grpc")


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

    def test_nginx_raise_ulimit_cluster_default(self):
        """srtslurm.yaml can set nginx_raise_ulimit when job omits frontend key."""
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "/c.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
            "frontend": {},
        }

        resolved = resolve_config_with_defaults(user_config, {"nginx_raise_ulimit": True})
        assert resolved["frontend"]["nginx_raise_ulimit"] is True

        user_explicit = {
            **user_config,
            "frontend": {"nginx_raise_ulimit": False},
        }
        resolved2 = resolve_config_with_defaults(user_explicit, {"nginx_raise_ulimit": True})
        assert resolved2["frontend"]["nginx_raise_ulimit"] is False

    def test_default_sbatch_directives_apply_as_defaults(self):
        """srtslurm.yaml can provide default sbatch directives for every job."""
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "/c.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
        }

        resolved = resolve_config_with_defaults(
            user_config,
            {"default_sbatch_directives": {"exclude": "gpu-[1,5]", "qos": "normal"}},
        )

        assert resolved["sbatch_directives"] == {
            "exclude": "gpu-[1,5]",
            "qos": "normal",
        }

    def test_default_sbatch_directives_do_not_override_job_values(self):
        """Job-level sbatch directives take precedence over srtslurm.yaml defaults."""
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "/c.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
            "sbatch_directives": {"exclude": "gpu-9"},
        }

        resolved = resolve_config_with_defaults(
            user_config,
            {"default_sbatch_directives": {"exclude": "gpu-[1,5]", "constraint": "h100"}},
        )

        assert resolved["sbatch_directives"] == {
            "exclude": "gpu-9",
            "constraint": "h100",
        }

    def test_default_health_check_applies_when_recipe_omits_it(self):
        """srtslurm.yaml can provide a default health_check block."""
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "/c.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
        }

        resolved = resolve_config_with_defaults(
            user_config,
            {"default_health_check": {"max_attempts": 540, "interval_seconds": 10}},
        )

        assert resolved["health_check"] == {"max_attempts": 540, "interval_seconds": 10}

    def test_default_health_check_does_not_override_recipe(self):
        """Recipe-level health_check wins over the cluster default."""
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "/c.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
            "health_check": {"max_attempts": 720, "interval_seconds": 10},
        }

        resolved = resolve_config_with_defaults(
            user_config,
            {"default_health_check": {"max_attempts": 540, "interval_seconds": 10}},
        )

        assert resolved["health_check"] == {"max_attempts": 720, "interval_seconds": 10}

    def test_cluster_sbatch_directives_are_not_treated_as_defaults(self):
        """srtslurm.yaml defaults must use default_sbatch_directives explicitly."""
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "/c.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
        }

        resolved = resolve_config_with_defaults(
            user_config,
            {"sbatch_directives": {"exclude": "gpu-[1,5]"}},
        )

        assert "sbatch_directives" not in resolved

    def test_telemetry_container_aliases_resolve(self):
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "sglang", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
            "telemetry": {
                "enabled": True,
                "container_image": "telemetry-scraper",
                "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401},
                "node_exporter": {"container_image": "node-exporter", "port": 9101},
            },
        }
        cluster_config = {
            "containers": {
                "sglang": "/path/to/sglang.sqsh",
                "telemetry-scraper": "/path/to/scraper.sqsh",
                "dcgm-exporter": "/path/to/dcgm.sqsh",
                "node-exporter": "/path/to/node.sqsh",
            }
        }

        resolved = resolve_config_with_defaults(user_config, cluster_config)

        assert resolved["telemetry"]["container_image"] == "/path/to/scraper.sqsh"
        assert resolved["telemetry"]["dcgm_exporter"]["container_image"] == "/path/to/dcgm.sqsh"
        assert resolved["telemetry"]["node_exporter"]["container_image"] == "/path/to/node.sqsh"

    def test_telemetry_literal_paths_pass_through(self):
        from srtctl.core.config import resolve_config_with_defaults

        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "/container.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
            "telemetry": {
                "enabled": True,
                "container_image": "/abs/scraper.sqsh",
                "dcgm_exporter": {"container_image": "/abs/dcgm.sqsh", "port": 9401},
                "node_exporter": {"container_image": "/abs/node.sqsh", "port": 9101},
            },
        }
        cluster_config = {"containers": {"dcgm-exporter": "/aliased/dcgm.sqsh"}}

        resolved = resolve_config_with_defaults(user_config, cluster_config)

        assert resolved["telemetry"]["container_image"] == "/abs/scraper.sqsh"
        assert resolved["telemetry"]["dcgm_exporter"]["container_image"] == "/abs/dcgm.sqsh"
        assert resolved["telemetry"]["node_exporter"]["container_image"] == "/abs/node.sqsh"


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

    def test_sbatch_template_prefetches_dynamo_wheel(self):
        """dynamo.wheel is exported and prefetched before orchestrator launch."""
        from pathlib import Path

        from srtctl.cli.submit import generate_minimal_sbatch_script
        from srtctl.core.schema import DynamoConfig, ModelConfig, ResourceConfig, SrtConfig

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
            dynamo=DynamoConfig(
                install=True,
                wheel="1.2.0.dev20260426",
            ),
        )

        script = generate_minimal_sbatch_script(config, Path("/tmp/test.yaml"))

        assert "export DYNAMO_VERSION=1.2.0.dev20260426" in script
        assert "export DYNAMO_WHEEL_NAME=ai_dynamo-1.2.0.dev20260426-py3-none-any.whl" in script
        assert 'uv run --no-project --python "${DYNAMO_PYTHON_VERSION:-3.12}" --with pip' in script
        assert "src/srtctl/runtime_scripts/dynamo_wheels.py" in script
        assert "configs/prefetch-ai-dynamo-wheel.sh" not in script

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

        with (
            patch.dict(os.environ, slurm_env),
            patch("subprocess.run", mock_scontrol),
            patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        ):
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

        with (
            patch.dict(os.environ, slurm_env),
            patch("subprocess.run", mock_scontrol),
            patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        ):
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

        with (
            patch("srtctl.core.runtime.get_slurm_nodelist", return_value=["node0"]),
            pytest.raises(ValueError, match="at least 2 nodes"),
        ):
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

    def test_vllm_colocation_reduces_sbatch_to_one_node_when_fit(self):
        """Test vLLM P/D colocation requests one worker node when all workers fit."""
        from pathlib import Path

        from srtctl.backends import VLLMProtocol
        from srtctl.cli.submit import generate_minimal_sbatch_script
        from srtctl.core.schema import InfraConfig, ModelConfig, ResourceConfig, SrtConfig

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
                _explicit_gpus_per_prefill=4,
                _explicit_gpus_per_decode=4,
            ),
            backend=VLLMProtocol(allow_prefill_decode_colocation=True),
            infra=InfraConfig(etcd_nats_dedicated_node=False),
        )

        assert config.resources.total_nodes == 2
        assert config.total_nodes == 1

        script = generate_minimal_sbatch_script(config, Path("/tmp/test.yaml"))

        assert "#SBATCH --nodes=1" in script

    def test_vllm_colocation_keeps_normal_node_count_when_not_fit(self):
        """Test vLLM P/D colocation does not reduce nodes when workers exceed one node."""
        from srtctl.backends import VLLMProtocol
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

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
                _explicit_gpus_per_prefill=6,
                _explicit_gpus_per_decode=4,
            ),
            backend=VLLMProtocol(allow_prefill_decode_colocation=True),
        )

        assert config.total_nodes == 2


class TestVLLMPrefillDecodeColocation:
    """Tests for vLLM prefill/decode same-node packing."""

    def test_disabled_by_default_keeps_prefill_and_decode_separate(self):
        """Test vLLM preserves default P/D node separation."""
        from srtctl.backends import VLLMProtocol

        endpoints = VLLMProtocol().allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=4,
            gpus_per_decode=4,
            gpus_per_agg=0,
            gpus_per_node=8,
            available_nodes=("node0", "node1"),
        )

        assert endpoints[0].mode == "prefill"
        assert endpoints[0].nodes == ("node0",)
        assert endpoints[1].mode == "decode"
        assert endpoints[1].nodes == ("node1",)

    def test_colocation_requires_prefill_decode_and_valid_node_size(self):
        """Test vLLM colocation stays off for incomplete or invalid P/D topology."""
        from srtctl.backends import VLLMProtocol

        backend = VLLMProtocol(allow_prefill_decode_colocation=True)

        for num_prefill, num_decode, gpus_per_node in ((0, 1, 8), (1, 0, 8), (1, 1, 0)):
            assert not backend.should_colocate_prefill_decode(
                num_prefill=num_prefill,
                num_decode=num_decode,
                num_agg=0,
                gpus_per_prefill=4,
                gpus_per_decode=4,
                gpus_per_agg=0,
                gpus_per_node=gpus_per_node,
            )

    def test_enabled_packs_prefill_and_decode_when_one_node_fits(self):
        """Test vLLM packs P/D workers together when requested and all fit."""
        from srtctl.backends import VLLMProtocol

        endpoints = VLLMProtocol(allow_prefill_decode_colocation=True).allocate_endpoints(
            num_prefill=2,
            num_decode=2,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=2,
            gpus_per_agg=0,
            gpus_per_node=8,
            available_nodes=("node0", "node1"),
        )

        prefill_eps = [ep for ep in endpoints if ep.mode == "prefill"]
        decode_eps = [ep for ep in endpoints if ep.mode == "decode"]

        assert [ep.nodes for ep in prefill_eps] == [("node0",), ("node0",)]
        assert [ep.gpu_indices for ep in prefill_eps] == [frozenset({0, 1}), frozenset({2, 3})]
        assert [ep.nodes for ep in decode_eps] == [("node0",), ("node0",)]
        assert [ep.gpu_indices for ep in decode_eps] == [frozenset({4, 5}), frozenset({6, 7})]

    def test_same_node_prefill_decode_ports_do_not_collide(self):
        """Test same-node vLLM P/D workers get distinct listener ports."""
        from srtctl.backends import VLLMProtocol

        backend = VLLMProtocol(allow_prefill_decode_colocation=True)
        endpoints = backend.allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=4,
            gpus_per_decode=4,
            gpus_per_agg=0,
            gpus_per_node=8,
            available_nodes=("node0", "node1"),
        )

        processes = backend.endpoints_to_processes(endpoints)
        prefill = next(p for p in processes if p.endpoint_mode == "prefill")
        decode = next(p for p in processes if p.endpoint_mode == "decode")

        assert prefill.node == decode.node == "node0"
        assert prefill.http_port == SGLANG_HTTP_PORT_BASE
        assert decode.http_port == SGLANG_HTTP_PORT_BASE + SGLANG_HTTP_PORT_STRIDE
        assert prefill.bootstrap_port == SGLANG_BOOTSTRAP_PORT_BASE

        bound_ports = [
            port
            for process in processes
            for port in (process.http_port, process.bootstrap_port, process.kv_events_port, process.nixl_port)
            if port
        ]
        assert len(bound_ports) == len(set(bound_ports))

    def test_same_node_dp_prefill_decode_ports_do_not_collide(self):
        """Test same-node DP P/D endpoints get distinct per-endpoint port ranges."""
        from srtctl.backends import VLLMProtocol, VLLMServerConfig

        backend = VLLMProtocol(
            allow_prefill_decode_colocation=True,
            vllm_config=VLLMServerConfig(
                prefill={"data-parallel-size": 4, "enable-expert-parallel": True},
                decode={"data-parallel-size": 4, "enable-expert-parallel": True},
            ),
        )
        endpoints = backend.allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=4,
            gpus_per_decode=4,
            gpus_per_agg=0,
            gpus_per_node=8,
            available_nodes=("node0", "node1"),
        )

        processes = backend.endpoints_to_processes(endpoints)
        prefill = [p for p in processes if p.endpoint_mode == "prefill"]
        decode = [p for p in processes if p.endpoint_mode == "decode"]

        assert len(prefill) == 4
        assert len(decode) == 4
        assert {p.node for p in prefill + decode} == {"node0"}
        assert {p.dp_rpc_port for p in prefill} == {VLLM_DATA_PARALLEL_RPC_PORT}
        assert {p.dp_rpc_port for p in decode} == {VLLM_DATA_PARALLEL_RPC_PORT + 1}
        assert {p.nixl_port for p in prefill} == {VLLM_NIXL_PORT_BASE}
        assert {p.nixl_port for p in decode} == {VLLM_NIXL_PORT_BASE + 4}

        leader_ports = [
            port for process in prefill + decode for port in (process.http_port, process.bootstrap_port) if port
        ]
        assert sorted(leader_ports) == [
            SGLANG_HTTP_PORT_BASE,
            SGLANG_HTTP_PORT_BASE + SGLANG_HTTP_PORT_STRIDE,
            SGLANG_BOOTSTRAP_PORT_BASE,
        ]

        prefill_actual_nixl_ports = {next(iter(p.nixl_port for p in prefill)) + p.node_rank for p in prefill}
        decode_actual_nixl_ports = {next(iter(p.nixl_port for p in decode)) + p.node_rank for p in decode}
        assert prefill_actual_nixl_ports == {VLLM_NIXL_PORT_BASE + i for i in range(4)}
        assert decode_actual_nixl_ports == {VLLM_NIXL_PORT_BASE + 4 + i for i in range(4)}
        assert prefill_actual_nixl_ports.isdisjoint(decode_actual_nixl_ports)

    def test_enabled_does_not_pack_when_one_node_does_not_fit(self):
        """Test vLLM falls back to separated P/D nodes when total GPUs do not fit."""
        from srtctl.backends import VLLMProtocol

        endpoints = VLLMProtocol(allow_prefill_decode_colocation=True).allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=6,
            gpus_per_decode=4,
            gpus_per_agg=0,
            gpus_per_node=8,
            available_nodes=("node0", "node1"),
        )

        assert endpoints[0].mode == "prefill"
        assert endpoints[0].nodes == ("node0",)
        assert endpoints[1].mode == "decode"
        assert endpoints[1].nodes == ("node1",)


class TestHetJobsValidation:
    """SrtConfig.__post_init__ validation for `resources.het_jobs: true`."""

    def _make(self, **resource_overrides):
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

        resources_kwargs = dict(
            gpu_type="gb200",
            gpus_per_node=4,
            prefill_nodes=12,
            decode_nodes=10,
            prefill_workers=12,
            decode_workers=10,
            het_jobs=True,
        )
        backend = resource_overrides.pop("backend", None)
        resources_kwargs.update(resource_overrides)
        kwargs = dict(
            name="t",
            model=ModelConfig(path="/m", container="/c.sqsh", precision="fp8"),
            resources=ResourceConfig(**resources_kwargs),
        )
        if backend is not None:
            kwargs["backend"] = backend
        return SrtConfig, kwargs

    def test_het_jobs_passes_with_disagg_sglang(self):
        SrtConfig, kwargs = self._make()
        cfg = SrtConfig(**kwargs)
        assert cfg.resources.het_jobs is True

    def test_het_jobs_rejected_in_agg_mode(self):
        import pytest
        from marshmallow import ValidationError

        SrtConfig, kwargs = self._make(
            prefill_nodes=None,
            decode_nodes=None,
            prefill_workers=None,
            decode_workers=None,
            agg_nodes=2,
            agg_workers=2,
        )
        with pytest.raises(ValidationError, match="disaggregated layout"):
            SrtConfig(**kwargs)

    def test_het_jobs_rejected_on_trtllm(self):
        import pytest
        from marshmallow import ValidationError

        from srtctl.backends import TRTLLMProtocol

        SrtConfig, kwargs = self._make(backend=TRTLLMProtocol())
        with pytest.raises(ValidationError, match="only supported on the sglang backend"):
            SrtConfig(**kwargs)

    def test_het_jobs_rejected_with_zero_nodes(self):
        import pytest
        from marshmallow import ValidationError

        SrtConfig, kwargs = self._make(prefill_nodes=0)
        with pytest.raises(ValidationError, match="prefill_nodes >= 1"):
            SrtConfig(**kwargs)

    def test_het_jobs_off_is_unrestricted(self):
        """Recipe with het_jobs=None or False should not trigger het validation."""
        from srtctl.backends import TRTLLMProtocol
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

        # trtllm + agg is fine when het is off — would only fail if het_jobs were True.
        cfg = SrtConfig(
            name="t",
            model=ModelConfig(path="/m", container="/c.sqsh", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="gb200",
                gpus_per_node=4,
                agg_nodes=2,
                agg_workers=2,
                het_jobs=False,
            ),
            backend=TRTLLMProtocol(),
        )
        assert cfg.resources.het_jobs is False


class TestHetComponents:
    """ResourceConfig.het_components() shape."""

    def _resources(self, **overrides):
        from srtctl.core.schema import ResourceConfig

        base = dict(
            gpu_type="gb200",
            gpus_per_node=4,
            prefill_nodes=12,
            decode_nodes=10,
            prefill_workers=12,
            decode_workers=10,
            het_jobs=True,
        )
        base.update(overrides)
        return ResourceConfig(**base)

    def test_het_components_returns_two_components(self):
        r = self._resources()
        components = r.het_components(infra_dedicated=False)
        assert components is not None
        prefill, decode = components
        assert prefill.name == "prefill"
        assert prefill.group == 0
        assert prefill.nodes == 12
        assert prefill.segment == 12
        assert decode.name == "decode"
        assert decode.group == 1
        assert decode.nodes == 10
        assert decode.segment == 10

    def test_het_components_folds_infra_into_prefill(self):
        r = self._resources()
        components = r.het_components(infra_dedicated=True)
        assert components is not None
        prefill, decode = components
        # prefill_nodes (12) + 1 dedicated infra
        assert prefill.nodes == 13
        assert prefill.segment == 13
        # decode unchanged
        assert decode.nodes == 10
        assert decode.segment == 10

    def test_het_components_none_when_off(self):
        from srtctl.core.schema import ResourceConfig

        r = ResourceConfig(
            gpu_type="gb200",
            gpus_per_node=4,
            prefill_nodes=12,
            decode_nodes=10,
            prefill_workers=12,
            decode_workers=10,
            het_jobs=False,
        )
        assert r.het_components(infra_dedicated=False) is None

    def test_het_components_cluster_default_applies_when_recipe_none(self):
        from srtctl.core.schema import ResourceConfig

        r = ResourceConfig(
            gpu_type="gb200",
            gpus_per_node=4,
            prefill_nodes=12,
            decode_nodes=10,
            prefill_workers=12,
            decode_workers=10,
            het_jobs=None,
        )
        # cluster_default=False -> off
        assert r.het_components(infra_dedicated=False) is None
        # cluster_default=True -> on
        assert r.het_components(infra_dedicated=False, cluster_default=True) is not None


class TestHetJobsSbatchScript:
    """generate_minimal_sbatch_script() emits het structure when het_jobs is True."""

    def _config(self, *, het_jobs, infra_dedicated):
        from srtctl.core.schema import InfraConfig, ModelConfig, ResourceConfig, SrtConfig

        return SrtConfig(
            name="t",
            model=ModelConfig(path="/m", container="/c.sqsh", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="gb200",
                gpus_per_node=4,
                prefill_nodes=12,
                decode_nodes=10,
                prefill_workers=12,
                decode_workers=10,
                het_jobs=het_jobs,
            ),
            infra=InfraConfig(etcd_nats_dedicated_node=infra_dedicated),
        )

    def test_emits_hetjob_separator_and_two_segments(self):
        from pathlib import Path

        from srtctl.cli.submit import generate_minimal_sbatch_script

        cfg = self._config(het_jobs=True, infra_dedicated=False)
        script = generate_minimal_sbatch_script(cfg, Path("/tmp/test.yaml"))

        assert script.count("#SBATCH hetjob") == 1
        assert "#SBATCH --segment=12" in script
        assert "#SBATCH --segment=10" in script
        # SLURM het-jobs need --account/--time/--partition repeated per component
        # (each #SBATCH directive applies to the component it follows, not the job).
        assert script.count("#SBATCH --account=") == 2
        assert script.count("#SBATCH --partition=") == 2
        # --output is job-wide (only one log file), so it appears once at the top.
        assert script.count("#SBATCH --output=") == 1
        # Per-component --nodes lines
        assert "#SBATCH --nodes=12" in script
        assert "#SBATCH --nodes=10" in script

    def test_infra_folds_into_prefill_component(self):
        from pathlib import Path

        from srtctl.cli.submit import generate_minimal_sbatch_script

        cfg = self._config(het_jobs=True, infra_dedicated=True)
        script = generate_minimal_sbatch_script(cfg, Path("/tmp/test.yaml"))

        # prefill component grows by 1 for the dedicated infra node
        assert "#SBATCH --nodes=13" in script
        assert "#SBATCH --segment=13" in script
        assert "#SBATCH --nodes=10" in script
        assert "#SBATCH --segment=10" in script

    def test_no_hetjob_block_when_off(self):
        from pathlib import Path

        from srtctl.cli.submit import generate_minimal_sbatch_script

        cfg = self._config(het_jobs=False, infra_dedicated=False)
        script = generate_minimal_sbatch_script(cfg, Path("/tmp/test.yaml"))
        assert "#SBATCH hetjob" not in script
        # Single --nodes line (12 prefill + 10 decode = 22)
        assert "#SBATCH --nodes=22" in script


class TestNodesHetGroupParsing:
    """Nodes.from_slurm reads SLURM_HET_SIZE/SLURM_JOB_NODELIST_HET_GROUP_*."""

    def test_from_slurm_returns_het_layout(self):
        from unittest.mock import patch

        from srtctl.core.runtime import Nodes

        het_lists = [
            ["gb200-01", "gb200-02", "gb200-03"],  # group 0: prefill (+ infra)
            ["gb200-04", "gb200-05"],  # group 1: decode
        ]
        with patch("srtctl.core.runtime.get_slurm_het_nodelists", return_value=het_lists):
            nodes = Nodes.from_slurm(etcd_nats_dedicated_node=False)

        assert nodes.het is True
        assert nodes.prefill_group == ("gb200-01", "gb200-02", "gb200-03")
        assert nodes.decode_group == ("gb200-04", "gb200-05")
        assert nodes.worker == ("gb200-01", "gb200-02", "gb200-03", "gb200-04", "gb200-05")

    def test_from_slurm_het_with_dedicated_infra(self):
        from unittest.mock import patch

        from srtctl.core.runtime import Nodes

        het_lists = [
            ["gb200-00", "gb200-01", "gb200-02"],  # group 0: [infra, prefill...]
            ["gb200-03", "gb200-04"],  # group 1: decode
        ]
        with patch("srtctl.core.runtime.get_slurm_het_nodelists", return_value=het_lists):
            nodes = Nodes.from_slurm(etcd_nats_dedicated_node=True)

        assert nodes.infra == "gb200-00"
        assert nodes.head == "gb200-01"
        assert nodes.prefill_group == ("gb200-01", "gb200-02")
        assert nodes.decode_group == ("gb200-03", "gb200-04")
        # Infra node carved out of worker pool
        assert "gb200-00" not in nodes.worker

    def test_het_group_for_returns_correct_group(self):
        from unittest.mock import patch

        from srtctl.core.runtime import Nodes

        het_lists = [["p0", "p1"], ["d0", "d1"]]
        with patch("srtctl.core.runtime.get_slurm_het_nodelists", return_value=het_lists):
            nodes = Nodes.from_slurm(etcd_nats_dedicated_node=False)

        assert nodes.het_group_for("p0") == 0
        assert nodes.het_group_for("d0") == 1
        assert nodes.het_group_for("unknown") is None

    def test_het_group_for_returns_none_on_non_het(self):
        from unittest.mock import patch

        from srtctl.core.runtime import Nodes

        with (
            patch("srtctl.core.runtime.get_slurm_het_nodelists", return_value=None),
            patch("srtctl.core.runtime.get_slurm_nodelist", return_value=["n0", "n1"]),
        ):
            nodes = Nodes.from_slurm(etcd_nats_dedicated_node=False)

        assert nodes.het is False
        assert nodes.het_group_for("n0") is None


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

    def test_dp_per_node_mode_creates_per_node_processes(self):
        """Per-node DP owns all local GPUs and reserves rank-sized port blocks."""
        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Endpoint

        backend = VLLMProtocol(
            dp_launch_mode="per_node",
            vllm_config=VLLMServerConfig(
                prefill={"data-parallel-size": 8, "enable-expert-parallel": True},
            ),
        )
        endpoint = Endpoint(
            mode="prefill",
            index=0,
            nodes=("node0", "node1"),
            gpu_indices=frozenset(range(4)),
            gpus_per_node=4,
            het_group=1,
        )

        processes = backend.endpoints_to_processes([endpoint])

        assert len(processes) == 2
        assert [p.node for p in processes] == ["node0", "node1"]
        assert all(p.gpu_indices == frozenset(range(4)) for p in processes)
        assert [p.node_rank for p in processes] == [0, 4]
        assert [p.kv_events_port for p in processes] == [KV_EVENTS_PORT_BASE, KV_EVENTS_PORT_BASE + 4]
        assert {p.nixl_port for p in processes} == {VLLM_NIXL_PORT_BASE}
        assert {p.dp_rpc_port for p in processes} == {VLLM_DATA_PARALLEL_RPC_PORT}
        assert {p.het_group for p in processes} == {1}
        assert all(p.http_port > 0 for p in processes)
        assert all(p.bootstrap_port is not None for p in processes)

    def test_dp_per_node_mode_allocates_non_overlapping_endpoint_ports(self):
        """Co-located per-node DP endpoints get disjoint coordination ranges."""
        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Endpoint

        backend = VLLMProtocol(
            dp_launch_mode="per_node",
            vllm_config=VLLMServerConfig(
                decode={"data-parallel-size": 4, "enable-expert-parallel": True},
            ),
        )
        endpoints = [
            Endpoint(
                mode="decode",
                index=0,
                nodes=("node0",),
                gpu_indices=frozenset(range(4)),
                gpus_per_node=8,
            ),
            Endpoint(
                mode="decode",
                index=1,
                nodes=("node0",),
                gpu_indices=frozenset(range(4, 8)),
                gpus_per_node=8,
            ),
        ]

        processes = backend.endpoints_to_processes(endpoints)

        assert [p.kv_events_port for p in processes] == [KV_EVENTS_PORT_BASE, KV_EVENTS_PORT_BASE + 4]
        assert [p.nixl_port for p in processes] == [VLLM_NIXL_PORT_BASE, VLLM_NIXL_PORT_BASE + 4]
        assert [p.dp_rpc_port for p in processes] == [
            VLLM_DATA_PARALLEL_RPC_PORT,
            VLLM_DATA_PARALLEL_RPC_PORT + 1,
        ]

    def test_dp_per_node_mode_rejects_dp_size_mismatch(self):
        """The configured global DP size must match the allocated GPUs."""
        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Endpoint

        backend = VLLMProtocol(
            dp_launch_mode="per_node",
            vllm_config=VLLMServerConfig(prefill={"data-parallel-size": 7}),
        )
        endpoint = Endpoint(
            mode="prefill",
            index=0,
            nodes=("node0", "node1"),
            gpu_indices=frozenset(range(4)),
            gpus_per_node=4,
        )

        with pytest.raises(ValueError, match="data-parallel-size=7"):
            backend.endpoints_to_processes([endpoint])

    def test_dp_launch_mode_supports_per_role_overrides(self):
        """Prefill and decode can use different DP process layouts."""
        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Endpoint

        backend = VLLMProtocol(
            dp_launch_mode="per_node",
            decode_dp_launch_mode="per_gpu",
            vllm_config=VLLMServerConfig(
                prefill={
                    "data-parallel-size": 8,
                    "data-parallel-hybrid-lb": True,
                    "enable-expert-parallel": True,
                },
                decode={"data-parallel-size": 16, "enable-expert-parallel": True},
            ),
        )
        endpoints = [
            Endpoint(
                mode="prefill",
                index=0,
                nodes=("pnode0", "pnode1"),
                gpu_indices=frozenset(range(4)),
                gpus_per_node=4,
            ),
            Endpoint(
                mode="decode",
                index=0,
                nodes=("dnode0", "dnode1", "dnode2", "dnode3"),
                gpu_indices=frozenset(range(4)),
                gpus_per_node=4,
            ),
        ]

        processes = backend.endpoints_to_processes(endpoints)
        prefill_processes = [process for process in processes if process.endpoint_mode == "prefill"]
        decode_processes = [process for process in processes if process.endpoint_mode == "decode"]

        assert backend.get_dp_launch_mode_for_mode("prefill") == "per_node"
        assert backend.get_dp_launch_mode_for_mode("decode") == "per_gpu"
        assert len(prefill_processes) == 2
        assert all(process.gpu_indices == frozenset(range(4)) for process in prefill_processes)
        assert [process.node_rank for process in prefill_processes] == [0, 4]
        assert len(decode_processes) == 16
        assert all(len(process.gpu_indices) == 1 for process in decode_processes)
        assert [process.node_rank for process in decode_processes] == list(range(16))
        assert len({process.sys_port for process in processes}) == len(processes)
        assert backend.get_expected_dynamo_worker_counts(processes) == (2, 16)

    def test_dp_mode_allocates_unique_ports_for_multiple_endpoints_per_node(self):
        """Test DP endpoints sharing a node get non-colliding coordination ports."""
        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Endpoint

        backend = VLLMProtocol(
            vllm_config=VLLMServerConfig(
                decode={"data-parallel-size": 4, "enable-expert-parallel": True},
            )
        )

        endpoints = [
            Endpoint(
                mode="decode",
                index=0,
                nodes=("node0",),
                gpu_indices=frozenset(range(4)),
                gpus_per_node=8,
            ),
            Endpoint(
                mode="decode",
                index=1,
                nodes=("node0",),
                gpu_indices=frozenset(range(4, 8)),
                gpus_per_node=8,
            ),
        ]

        processes = backend.endpoints_to_processes(endpoints)

        first_endpoint = [p for p in processes if p.endpoint_index == 0]
        second_endpoint = [p for p in processes if p.endpoint_index == 1]

        assert {p.dp_rpc_port for p in first_endpoint} == {VLLM_DATA_PARALLEL_RPC_PORT}
        assert {p.dp_rpc_port for p in second_endpoint} == {VLLM_DATA_PARALLEL_RPC_PORT + 1}
        assert {p.nixl_port for p in first_endpoint} == {VLLM_NIXL_PORT_BASE}
        assert {p.nixl_port for p in second_endpoint} == {VLLM_NIXL_PORT_BASE + 4}
        assert [p.node_rank for p in first_endpoint] == list(range(4))
        assert [p.node_rank for p in second_endpoint] == list(range(4))

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

    def test_dp_per_node_hybrid_command_targets_local_rank_range(self):
        """Hybrid per-node DP exposes the local rank range without headless."""
        from unittest.mock import MagicMock, patch

        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Process

        backend = VLLMProtocol(
            dp_launch_mode="per_node",
            vllm_config=VLLMServerConfig(
                decode={
                    "data-parallel-size": 8,
                    "data-parallel-size-local": 99,
                    "data-parallel-start-rank": 99,
                    "data-parallel-rpc-port": 13345,
                    "data-parallel-hybrid-lb": True,
                    "enable-expert-parallel": True,
                },
            ),
        )
        leader = Process(
            node="node0",
            gpu_indices=frozenset(range(4)),
            sys_port=8081,
            http_port=6100,
            endpoint_mode="decode",
            endpoint_index=0,
            node_rank=0,
            dp_rpc_port=VLLM_DATA_PARALLEL_RPC_PORT,
        )
        process = Process(
            node="node1",
            gpu_indices=frozenset(range(4)),
            sys_port=8082,
            http_port=6100,
            endpoint_mode="decode",
            endpoint_index=0,
            node_rank=4,
            dp_rpc_port=VLLM_DATA_PARALLEL_RPC_PORT,
        )
        runtime = MagicMock()
        runtime.model_path = Path("/model")
        runtime.is_hf_model = False
        runtime.request_plane = "tcp"

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(process, [leader, process], runtime)

        assert cmd.count("--data-parallel-hybrid-lb") == 1
        assert cmd.count("--data-parallel-size-local") == 1
        assert cmd.count("--data-parallel-start-rank") == 1
        assert cmd[cmd.index("--data-parallel-size-local") + 1] == "4"
        assert cmd[cmd.index("--data-parallel-start-rank") + 1] == "4"
        assert cmd[cmd.index("--data-parallel-rpc-port") + 1] == str(VLLM_DATA_PARALLEL_RPC_PORT)
        assert "--data-parallel-rank" not in cmd
        assert "--headless" not in cmd

    def test_dp_per_node_internal_lb_follower_is_headless(self):
        """Non-hybrid per-node DP uses a headless follower process."""
        from unittest.mock import MagicMock, patch

        from srtctl.backends import VLLMProtocol, VLLMServerConfig
        from srtctl.core.topology import Process

        backend = VLLMProtocol(
            dp_launch_mode="per_node",
            vllm_config=VLLMServerConfig(decode={"data-parallel-size": 8}),
        )
        leader = Process(
            node="node0",
            gpu_indices=frozenset(range(4)),
            sys_port=8081,
            http_port=6100,
            endpoint_mode="decode",
            endpoint_index=0,
            node_rank=0,
        )
        process = Process(
            node="node1",
            gpu_indices=frozenset(range(4)),
            sys_port=8082,
            http_port=6100,
            endpoint_mode="decode",
            endpoint_index=0,
            node_rank=4,
        )
        runtime = MagicMock()
        runtime.model_path = Path("/model")
        runtime.is_hf_model = False
        runtime.request_plane = "tcp"

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            cmd = backend.build_worker_command(process, [leader, process], runtime)

        assert "--data-parallel-size-local" in cmd
        assert "--data-parallel-start-rank" in cmd
        assert "--data-parallel-hybrid-lb" not in cmd
        assert "--headless" in cmd

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
        from unittest.mock import patch

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

        with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
            env = backend.get_process_environment(process)

        assert env["DYN_VLLM_KV_EVENT_PORT"] == "5550"
        assert env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "6550"
        assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.0.0.1"

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
        assert "VLLM_NIXL_SIDE_CHANNEL_HOST" not in env

    def test_vllm_kv_events_config_global_bool(self):
        """Test kv_events_config=True enables prefill+decode with vLLM defaults."""
        from srtctl.backends import VLLMProtocol

        config = VLLMProtocol(kv_events_config=True)

        assert config.get_kv_events_config_for_mode("prefill") == {
            "publisher": "zmq",
            "topic": "kv-events",
            "enable_kv_cache_events": True,
        }
        assert config.get_kv_events_config_for_mode("decode") == {
            "publisher": "zmq",
            "topic": "kv-events",
            "enable_kv_cache_events": True,
        }
        assert config.get_kv_events_config_for_mode("agg") is None

    def test_vllm_kv_events_config_custom_settings(self):
        """Test kv_events_config per-mode settings merge with vLLM defaults."""
        from srtctl.backends import VLLMProtocol

        config = VLLMProtocol(
            kv_events_config={
                "prefill": {"topic": "prefill-events"},
                "decode": {"publisher": "custom", "topic": "decode-events"},
            }
        )

        prefill_cfg = config.get_kv_events_config_for_mode("prefill")
        assert prefill_cfg["publisher"] == "zmq"
        assert prefill_cfg["topic"] == "prefill-events"
        assert prefill_cfg["enable_kv_cache_events"] is True

        decode_cfg = config.get_kv_events_config_for_mode("decode")
        assert decode_cfg["publisher"] == "custom"
        assert decode_cfg["topic"] == "decode-events"
        assert decode_cfg["enable_kv_cache_events"] is True

    def test_vllm_command_includes_kv_events_config_with_allocated_port(self):
        """Test vLLM command injects --kv-events-config with the worker port."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from srtctl.backends import VLLMProtocol
        from srtctl.core.topology import Process

        backend = VLLMProtocol(kv_events_config=True)
        process = Process(
            node="node0",
            gpu_indices=frozenset([0]),
            sys_port=8081,
            http_port=30000,
            endpoint_mode="prefill",
            endpoint_index=0,
            node_rank=0,
            kv_events_port=5550,
        )
        mock_runtime = MagicMock()
        mock_runtime.model_path = Path("/model")
        mock_runtime.is_hf_model = False

        cmd = backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=mock_runtime,
        )

        flag_index = cmd.index("--kv-events-config")
        kv_cfg = json.loads(cmd[flag_index + 1])
        assert kv_cfg == {
            "publisher": "zmq",
            "topic": "kv-events",
            "enable_kv_cache_events": True,
            "endpoint": "tcp://*:5550",
        }

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
        # build_worker_command reads runtime.worker_model_arg (a real
        # RuntimeContext property); the mock must provide it. No staging here.
        runtime.worker_model_arg = str(runtime.model_path) if is_hf else "/model"
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

        with (
            patch("pathlib.Path.write_text"),
            patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        ):
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

        with (
            patch("pathlib.Path.write_text"),
            patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        ):
            cmd = backend.build_worker_command(process=process, endpoint_processes=[process], runtime=runtime)

        idx = cmd.index("--model-path")
        assert cmd[idx + 1] == "/model"


class TestInfmaxWorkspaceMount:
    """Test that INFMAX_WORKSPACE env var creates a container mount."""

    def test_infmax_workspace_mount_added(self, tmp_path):
        """RuntimeContext includes /infmax-workspace mount when env var is set."""
        import os
        import subprocess
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.core.runtime import RuntimeContext
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

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
            "INFMAX_WORKSPACE": "/actions/runner/workspace",
        }

        def mock_scontrol(cmd, **kwargs):
            if cmd[0] == "scontrol" and "hostnames" in cmd:
                result = MagicMock()
                result.stdout = "gpu-01\ngpu-02"
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
            )
            runtime = RuntimeContext.from_config(config, job_id="12345")

            assert Path("/infmax-workspace") in runtime.container_mounts.values()

    def test_infmax_workspace_mount_not_added_without_env(self, tmp_path):
        """RuntimeContext does not include /infmax-workspace without env var."""
        import os
        import subprocess
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.core.runtime import RuntimeContext
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

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
            os.environ.pop("INFMAX_WORKSPACE", None)
            with (
                patch("subprocess.run", mock_scontrol),
                patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
            ):
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
                )
                runtime = RuntimeContext.from_config(config, job_id="12345")

                assert Path("/infmax-workspace") not in runtime.container_mounts.values()


class TestExtraMountExpansion:
    """Test path expansion for recipe extra_mount entries."""

    def test_extra_mount_host_path_expands_environment_variables(self, tmp_path):
        import os
        import subprocess
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from srtctl.core.runtime import RuntimeContext
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

        model_path = tmp_path / "model"
        model_path.mkdir()
        container_path = tmp_path / "container.sqsh"
        container_path.touch()
        extra_root = tmp_path / "extra"
        extra_root.mkdir()

        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOBID": "12345",
            "SLURM_NODELIST": "gpu-[01-02]",
            "SLURM_JOB_NUM_NODES": "2",
            "SRTCTL_SOURCE_DIR": str(Path(__file__).parent.parent),
            "SRT_EXTRA_ROOT": str(extra_root),
        }

        def mock_scontrol(cmd, **kwargs):
            if cmd[0] == "scontrol" and "hostnames" in cmd:
                result = MagicMock()
                result.stdout = "gpu-01\ngpu-02"
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
                extra_mount=("$SRT_EXTRA_ROOT:/extra",),
            )
            runtime = RuntimeContext.from_config(config, job_id="12345")

            assert extra_root.resolve() in runtime.container_mounts
            assert runtime.container_mounts[extra_root.resolve()] == Path("/extra")
