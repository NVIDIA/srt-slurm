# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pre-submit validation checks.

Every check must be fault-tolerant: network errors, timeouts, missing paths
all produce ValidationResult, never raise.
"""

from unittest.mock import patch

import requests

from srtctl.core.validation import (
    preflight_config_variants,
    run_all_validations,
    run_validations_background,
    validate_docker_image,
    validate_hf_model,
    validate_local_path,
)

# ============================================================================
# Local path validation
# ============================================================================


class TestValidateLocalPath:
    def test_existing_directory(self, tmp_path):
        (tmp_path / "file1.txt").write_text("hello")
        (tmp_path / "file2.txt").write_text("world")

        result = validate_local_path("model", str(tmp_path))
        assert result.ok is True
        assert "2 files" in result.message

    def test_existing_file(self, tmp_path):
        f = tmp_path / "model.sqsh"
        f.write_bytes(b"\x00" * 1024)

        result = validate_local_path("container", str(f))
        assert result.ok is True
        assert "GB" in result.message

    def test_missing_path(self, tmp_path):
        result = validate_local_path("model", str(tmp_path / "nonexistent"))
        assert result.ok is False
        assert "not found" in result.message


# ============================================================================
# HuggingFace validation
# ============================================================================


class TestValidateHfModel:
    def test_skipped_when_none(self):
        result = validate_hf_model(None, None)
        assert result.ok is True
        assert "skipped" in result.message

    def test_model_exists(self):
        with patch("srtctl.core.validation.requests.head") as mock_head:
            mock_head.return_value.status_code = 200
            result = validate_hf_model("deepseek-ai/DeepSeek-R1", None)

        assert result.ok is True
        assert "exists" in result.message

    def test_model_not_found(self):
        with patch("srtctl.core.validation.requests.head") as mock_head:
            mock_head.return_value.status_code = 404
            result = validate_hf_model("fake/model", None)

        assert result.ok is False
        assert "not found" in result.message

    def test_model_gated(self):
        with patch("srtctl.core.validation.requests.head") as mock_head:
            mock_head.return_value.status_code = 401
            result = validate_hf_model("meta-llama/Llama-3", None)

        assert result.ok is True
        assert "gated" in result.message

    def test_network_timeout(self):
        with patch("srtctl.core.validation.requests.head", side_effect=requests.Timeout()):
            result = validate_hf_model("some/model", None)

        assert result.ok is False
        assert "timed out" in result.message

    def test_network_error(self):
        with patch("srtctl.core.validation.requests.head", side_effect=requests.ConnectionError()):
            result = validate_hf_model("some/model", None)

        assert result.ok is False
        assert "failed" in result.message

    def test_revision_verified(self):
        with patch("srtctl.core.validation.requests.head") as mock_head:
            mock_head.return_value.status_code = 200
            result = validate_hf_model("deepseek-ai/DeepSeek-R1", "abc123def456")

        assert result.ok is True
        assert "revision" in result.message
        assert "verified" in result.message

    def test_revision_not_found(self):
        responses = iter([type("R", (), {"status_code": 200})(), type("R", (), {"status_code": 404})()])
        with patch("srtctl.core.validation.requests.head", side_effect=lambda *a, **k: next(responses)):
            result = validate_hf_model("deepseek-ai/DeepSeek-R1", "bad_revision")

        assert result.ok is False
        assert "revision" in result.message


# ============================================================================
# Docker image validation
# ============================================================================


class TestValidateDockerImage:
    def test_skipped_when_none(self):
        result = validate_docker_image(None, None)
        assert result.ok is True
        assert "skipped" in result.message

    def test_image_exists(self):
        with patch("srtctl.core.validation.requests.head") as mock_head:
            mock_head.return_value.status_code = 200
            mock_head.return_value.headers = {}
            result = validate_docker_image("lmsysorg/sglang:v0.4.6", None)

        assert result.ok is True
        assert "exists" in result.message

    def test_image_not_found(self):
        with patch("srtctl.core.validation.requests.head") as mock_head:
            mock_head.return_value.status_code = 404
            result = validate_docker_image("fake/image:v1", None)

        assert result.ok is False
        assert "not found" in result.message

    def test_network_timeout(self):
        with patch("srtctl.core.validation.requests.head", side_effect=requests.Timeout()):
            result = validate_docker_image("some/image:tag", None)

        assert result.ok is False
        assert "timed out" in result.message

    def test_digest_verified(self):
        with patch("srtctl.core.validation.requests.head") as mock_head:
            mock_head.return_value.status_code = 200
            mock_head.return_value.headers = {"Docker-Content-Digest": "sha256:abc123"}
            result = validate_docker_image("img:tag", "sha256:abc123")

        assert result.ok is True
        assert "digest verified" in result.message

    def test_digest_mismatch(self):
        with patch("srtctl.core.validation.requests.head") as mock_head:
            mock_head.return_value.status_code = 200
            mock_head.return_value.headers = {"Docker-Content-Digest": "sha256:different"}
            result = validate_docker_image("img:tag", "sha256:abc123")

        assert result.ok is False
        assert "mismatch" in result.message


# ============================================================================
# run_all_validations
# ============================================================================


class TestRunAllValidations:
    def test_never_raises(self):
        """Even with completely broken config, returns a list."""
        from srtctl.core.schema import SrtConfig

        config = SrtConfig.Schema().load(
            {
                "name": "test",
                "model": {"path": "/nonexistent", "container": "/nonexistent.sqsh", "precision": "fp8"},
                "resources": {"gpu_type": "h100", "gpus_per_node": 8, "prefill_nodes": 1, "decode_nodes": 1},
            }
        )

        results = run_all_validations(config)
        assert isinstance(results, list)
        assert len(results) >= 2  # at least model_path and container_path

    def test_all_checks_produce_results(self):
        """Each check type produces exactly one result."""
        from srtctl.core.schema import SrtConfig

        config = SrtConfig.Schema().load(
            {
                "name": "test",
                "model": {
                    "path": "/nonexistent",
                    "container": "/nonexistent.sqsh",
                    "precision": "fp8",
                },
                "resources": {"gpu_type": "h100", "gpus_per_node": 8, "prefill_nodes": 1, "decode_nodes": 1},
                "identity": {
                    "model": {"repo": "some/model"},
                },
            }
        )

        with patch("srtctl.core.validation.requests.head", side_effect=requests.ConnectionError()):
            results = run_all_validations(config)

        check_names = [r.check for r in results]
        assert "model_path" in check_names
        assert "container_path" in check_names
        assert "hf_model" in check_names


# ============================================================================
# Background thread
# ============================================================================


class TestBackgroundValidation:
    def test_thread_is_daemon(self):
        from srtctl.core.schema import SrtConfig

        config = SrtConfig.Schema().load(
            {
                "name": "test",
                "model": {"path": "/x", "container": "/x", "precision": "fp8"},
                "resources": {"gpu_type": "h100", "gpus_per_node": 8, "prefill_nodes": 1, "decode_nodes": 1},
            }
        )

        thread = run_validations_background(config)
        assert thread.daemon is True
        thread.join(timeout=10)


class TestPreflightConfigVariants:
    def test_does_not_load_host_side_srtslurm_yaml_by_default(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        container_file = tmp_path / "container.sqsh"
        container_file.write_text("sqsh")
        (tmp_path / "srtslurm.yaml").write_text(
            "model_paths:\n"
            f"  qwen32b: {model_dir}\n"
            "containers:\n"
            f"  sglang-latest: {container_file}\n"
        )
        monkeypatch.chdir(tmp_path)

        results = preflight_config_variants(
            {
                "name": "host-side-ignored",
                "model": {
                    "path": "qwen32b",
                    "container": "sglang-latest",
                    "precision": "bf16",
                },
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "agg_nodes": 1,
                    "agg_workers": 1,
                },
            },
        )

        assert results[0].ok is False
        assert results[0].model.source == "literal"
        assert results[0].container.source == "literal"

    def test_aliases_pass_when_paths_exist(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        container_file = tmp_path / "container.sqsh"
        container_file.write_text("sqsh")

        results = preflight_config_variants(
            {
                "name": "ok",
                "model": {
                    "path": "qwen32b",
                    "container": "sglang-latest",
                    "precision": "bf16",
                },
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "prefill_workers": 1,
                    "decode_workers": 1,
                },
            },
            cluster_config={
                "model_paths": {"qwen32b": str(model_dir)},
                "containers": {"sglang-latest": str(container_file)},
            },
        )

        assert len(results) == 1
        assert results[0].ok is True
        assert results[0].model.source == "srtslurm.yaml:model_paths"
        assert results[0].container.source == "srtslurm.yaml:containers"

    def test_missing_model_alias_fails(self, tmp_path):
        container_file = tmp_path / "container.sqsh"
        container_file.write_text("sqsh")

        results = preflight_config_variants(
            {
                "name": "bad-model",
                "model": {
                    "path": "Qwen/Qwen3-32B",
                    "container": "sglang-latest",
                    "precision": "bf16",
                },
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                },
            },
            cluster_config={"containers": {"sglang-latest": str(container_file)}},
        )

        assert results[0].ok is False
        assert results[0].errors[0].code == "model-not-available"

    def test_preflight_accepts_docker_uri_container(self, tmp_path):
        """Container image URIs like ``nvcr.io/fake:latest`` are accepted by
        preflight — Pyxis/enroot pulls them at srun time (mirrors the
        runtime classification in runtime.py).  The image may still fail to
        pull at runtime if it's bogus, but that's an actual srun error, not
        a preflight one."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        results = preflight_config_variants(
            {
                "name": "docker-uri",
                "model": {
                    "path": str(model_dir),
                    "container": "nvcr.io/fake:latest",
                    "precision": "bf16",
                },
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "prefill_workers": 1,
                    "decode_workers": 1,
                },
            },
        )

        assert results[0].ok is True
        assert results[0].container.source == "container-uri"
        assert results[0].container.resolved == "nvcr.io/fake:latest"

    def test_preflight_accepts_hf_prefix_model_path(self, tmp_path):
        """``hf:org/model`` model paths are accepted — the framework
        downloads via HF cache at serve time.  Mirrors runtime.py's
        ``startswith('hf:')`` classification."""
        container_file = tmp_path / "container.sqsh"
        container_file.write_text("sqsh")

        results = preflight_config_variants(
            {
                "name": "hf-model",
                "model": {
                    "path": "hf:meta-llama/Llama-3.1-8B",
                    "container": str(container_file),
                    "precision": "bf16",
                },
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "prefill_workers": 1,
                    "decode_workers": 1,
                },
            },
        )

        assert results[0].ok is True
        assert results[0].model.source == "huggingface"
        assert results[0].model.resolved == "hf:meta-llama/Llama-3.1-8B"

    def test_preflight_accepts_hf_model_and_docker_container_together(
        self, tmp_path
    ):
        """The full AIB CI shape: ``hf:`` model + Docker URI container, no
        srtslurm.yaml aliases registered."""
        results = preflight_config_variants(
            {
                "name": "aib-ci-shape",
                "model": {
                    "path": "hf:nvidia/Kimi-K2.5-NVFP4",
                    "container": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1",
                    "precision": "fp4",
                },
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "prefill_workers": 1,
                    "decode_workers": 1,
                },
            },
        )

        assert results[0].ok is True
        assert results[0].model.source == "huggingface"
        assert results[0].container.source == "container-uri"
        assert results[0].errors == []

    def test_preflight_still_rejects_typo_local_path_without_colon(
        self, tmp_path
    ):
        """A bare relative string with no ``:`` and no leading ``./`` is NOT
        a Docker URI — runtime.py would treat it as an image name too, but
        if it doesn't even look URI-shaped, that's almost certainly a typo
        of a local path.  Locks in the ``:`` guard so genuinely-broken
        configs still get caught at preflight."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        results = preflight_config_variants(
            {
                "name": "typo",
                "model": {
                    "path": str(model_dir),
                    "container": "missing-file",  # no ':' → not URI shape
                    "precision": "bf16",
                },
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "prefill_workers": 1,
                    "decode_workers": 1,
                },
            },
        )

        assert results[0].ok is False
        assert any(
            issue.code == "container-not-available" for issue in results[0].errors
        )

    def test_telemetry_aliases_resolve_and_pass_when_files_exist(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        container_file = tmp_path / "container.sqsh"
        container_file.write_text("sqsh")
        scraper_file = tmp_path / "scraper.sqsh"
        scraper_file.write_text("sqsh")
        dcgm_file = tmp_path / "dcgm.sqsh"
        dcgm_file.write_text("sqsh")
        node_file = tmp_path / "node.sqsh"
        node_file.write_text("sqsh")

        results = preflight_config_variants(
            {
                "name": "telemetry-ok",
                "model": {"path": "qwen32b", "container": "sglang-latest", "precision": "bf16"},
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "prefill_workers": 1,
                    "decode_workers": 1,
                },
                "telemetry": {
                    "enabled": True,
                    "container_image": "telemetry-scraper",
                    "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401},
                    "node_exporter": {"container_image": "node-exporter", "port": 9101},
                },
            },
            cluster_config={
                "model_paths": {"qwen32b": str(model_dir)},
                "containers": {
                    "sglang-latest": str(container_file),
                    "telemetry-scraper": str(scraper_file),
                    "dcgm-exporter": str(dcgm_file),
                    "node-exporter": str(node_file),
                },
            },
        )

        assert results[0].ok is True
        assert results[0].errors == []

    def test_telemetry_missing_sqsh_fails_preflight(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        container_file = tmp_path / "container.sqsh"
        container_file.write_text("sqsh")
        scraper_file = tmp_path / "scraper.sqsh"
        scraper_file.write_text("sqsh")
        dcgm_file = tmp_path / "dcgm.sqsh"
        dcgm_file.write_text("sqsh")
        # node.sqsh deliberately missing

        results = preflight_config_variants(
            {
                "name": "telemetry-bad",
                "model": {"path": str(model_dir), "container": str(container_file), "precision": "bf16"},
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "prefill_workers": 1,
                    "decode_workers": 1,
                },
                "telemetry": {
                    "enabled": True,
                    "container_image": str(scraper_file),
                    "dcgm_exporter": {"container_image": str(dcgm_file), "port": 9401},
                    "node_exporter": {"container_image": str(tmp_path / "node.sqsh"), "port": 9101},
                },
            },
        )

        assert results[0].ok is False
        telemetry_errors = [issue for issue in results[0].errors if issue.code == "telemetry-container-not-available"]
        assert len(telemetry_errors) == 1
        assert telemetry_errors[0].field == "telemetry.node_exporter.container_image"

    def test_telemetry_disabled_skips_preflight(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        container_file = tmp_path / "container.sqsh"
        container_file.write_text("sqsh")

        results = preflight_config_variants(
            {
                "name": "telemetry-off",
                "model": {"path": str(model_dir), "container": str(container_file), "precision": "bf16"},
                "resources": {
                    "gpu_type": "gb200",
                    "gpus_per_node": 4,
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "prefill_workers": 1,
                    "decode_workers": 1,
                },
                "telemetry": {
                    "enabled": False,
                    "container_image": "/does/not/exist.sqsh",
                    "dcgm_exporter": {"container_image": "/does/not/exist.sqsh", "port": 9401},
                    "node_exporter": {"container_image": "/does/not/exist.sqsh", "port": 9101},
                },
            },
        )

        assert results[0].ok is True
        assert not any(issue.code == "telemetry-container-not-available" for issue in results[0].errors)
