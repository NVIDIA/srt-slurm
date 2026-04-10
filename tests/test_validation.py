# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pre-submit validation checks.

Every check must be fault-tolerant: network errors, timeouts, missing paths
all produce ValidationResult, never raise.
"""

from unittest.mock import patch

import requests

from srtctl.core.validation import (
    ValidationResult,
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
        from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

        config = SrtConfig.Schema().load({
            "name": "test",
            "model": {"path": "/nonexistent", "container": "/nonexistent.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "prefill_nodes": 1, "decode_nodes": 1},
        })

        results = run_all_validations(config)
        assert isinstance(results, list)
        assert len(results) >= 2  # at least model_path and container_path

    def test_all_checks_produce_results(self):
        """Each check type produces exactly one result."""
        from srtctl.core.schema import SrtConfig

        config = SrtConfig.Schema().load({
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
        })

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

        config = SrtConfig.Schema().load({
            "name": "test",
            "model": {"path": "/x", "container": "/x", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "prefill_nodes": 1, "decode_nodes": 1},
        })

        thread = run_validations_background(config)
        assert thread.daemon is True
        thread.join(timeout=10)
