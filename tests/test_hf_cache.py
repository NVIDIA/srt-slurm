# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HuggingFace cache management in SweepOrchestrator.

Tests _get_hf_home(), _clean_stale_hf_locks(), and _ensure_model_cached().
"""

import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import load_config
from srtctl.core.runtime import RuntimeContext

# Fixtures


class TestCluster:
    """Minimal cluster for testing."""

    NUM_NODES = 5
    GPUS_PER_NODE = 4

    @classmethod
    def nodes(cls) -> list[str]:
        return [f"node-{i:02d}" for i in range(1, cls.NUM_NODES + 1)]

    @classmethod
    def slurm_env(cls) -> dict[str, str]:
        return {
            "SLURM_JOB_ID": "99999",
            "SLURM_JOBID": "99999",
            "SLURM_NODELIST": f"node-[01-{cls.NUM_NODES:02d}]",
            "SLURM_JOB_NUM_NODES": str(cls.NUM_NODES),
            "SRTCTL_SOURCE_DIR": str(Path(__file__).parent.parent),
        }

    @classmethod
    def mock_scontrol(cls):
        def mock_run(cmd, **kwargs):
            if cmd[0] == "scontrol" and "hostnames" in cmd:
                result = MagicMock()
                result.stdout = "\n".join(cls.nodes())
                result.returncode = 0
                return result
            raise subprocess.CalledProcessError(1, cmd)

        return mock_run


def _make_orchestrator(config_path: str, hf_model: bool = True) -> SweepOrchestrator:
    """Create a SweepOrchestrator with mocked SLURM environment."""
    with (
        patch.dict(os.environ, TestCluster.slurm_env()),
        patch("subprocess.run", TestCluster.mock_scontrol()),
    ):
        config = load_config(Path(config_path))
        runtime = RuntimeContext.from_config(config, "99999")
        return SweepOrchestrator(config=config, runtime=runtime)


# Tests for _get_hf_home


class TestGetHfHome:
    """Tests for _get_hf_home() method."""

    def test_returns_hf_home_from_prefill_env(self, tmp_path: Path):
        """HF_HOME set in prefill_environment should be returned."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
name: test
model:
  path: "hf:nvidia/test-model"
  container: "test-image:latest"
  precision: fp16
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "/cache/hub"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))
        assert orch._get_hf_home() == "/cache/hub"

    def test_returns_none_when_not_set(self, tmp_path: Path):
        """No HF_HOME in any environment config should return None."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
name: test
model:
  path: "hf:nvidia/test-model"
  container: "test-image:latest"
  precision: fp16
backend:
  type: vllm
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))
        assert orch._get_hf_home() is None


# Tests for _clean_stale_hf_locks


class TestCleanStaleHfLocks:
    """Tests for _clean_stale_hf_locks() method."""

    def test_removes_old_lock_files(self, tmp_path: Path):
        """Lock files older than 30 minutes should be removed."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        blobs_dir = cache_dir / "hub" / "models--test" / "blobs"
        blobs_dir.mkdir(parents=True)

        # Create a stale lock (set mtime to 1 hour ago)
        stale_lock = blobs_dir / "abc123.lock"
        stale_lock.touch()
        old_time = time.time() - 3600
        os.utime(stale_lock, (old_time, old_time))

        # Create a fresh lock (should NOT be removed)
        fresh_lock = blobs_dir / "def456.lock"
        fresh_lock.touch()

        # Create a non-lock file (should NOT be removed)
        data_file = blobs_dir / "abc123"
        data_file.write_bytes(b"model data")

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
name: test
model:
  path: "hf:nvidia/test-model"
  container: "test-image:latest"
  precision: fp16
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "{cache_dir}"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))
        orch._clean_stale_hf_locks()

        assert not stale_lock.exists(), "Stale lock should be removed"
        assert fresh_lock.exists(), "Fresh lock should be kept"
        assert data_file.exists(), "Non-lock files should be untouched"

    def test_no_hf_home_is_noop(self, tmp_path: Path):
        """No HF_HOME configured should silently do nothing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
name: test
model:
  path: "hf:nvidia/test-model"
  container: "test-image:latest"
  precision: fp16
backend:
  type: vllm
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))
        # Should not raise
        orch._clean_stale_hf_locks()

    def test_nonexistent_cache_dir_is_noop(self, tmp_path: Path):
        """HF_HOME pointing to nonexistent directory should silently do nothing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
name: test
model:
  path: "hf:nvidia/test-model"
  container: "test-image:latest"
  precision: fp16
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "{tmp_path / "nonexistent"}"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))
        # Should not raise
        orch._clean_stale_hf_locks()


# Tests for _ensure_model_cached


class TestEnsureModelCached:
    """Tests for _ensure_model_cached() method."""

    def test_skips_local_models(self, tmp_path: Path):
        """Local model paths (not hf:) should skip pre-download."""
        model_dir = tmp_path / "local_model"
        model_dir.mkdir()

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
name: test
model:
  path: "{model_dir}"
  container: "test-image:latest"
  precision: fp16
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "{tmp_path / "cache"}"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))

        with patch("srtctl.cli.do_sweep.start_srun_process") as mock_srun:
            orch._ensure_model_cached()
            mock_srun.assert_not_called()

    def test_skips_and_warns_when_no_hf_home(self, tmp_path: Path, caplog):
        """HF model without HF_HOME should skip pre-download and log a warning."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
name: test
model:
  path: "hf:nvidia/test-model"
  container: "test-image:latest"
  precision: fp16
backend:
  type: vllm
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))

        import logging

        with patch("srtctl.cli.do_sweep.start_srun_process") as mock_srun:
            with caplog.at_level(logging.WARNING):
                orch._ensure_model_cached()
            mock_srun.assert_not_called()
            assert "HF_HOME is not set" in caplog.text

    def test_skips_when_model_already_cached(self, tmp_path: Path):
        """Pre-download should skip if huggingface_hub reports model is cached."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
name: test
model:
  path: "hf:nvidia/Kimi-K2.5-NVFP4"
  container: "test-image:latest"
  precision: fp4
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "{tmp_path / "cache"}"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))

        # Mock huggingface_hub.snapshot_download to succeed (model is cached).
        # The import happens inside _ensure_model_cached via "from huggingface_hub import ...",
        # so we create a fake module and patch it into sys.modules.
        import types

        fake_hf_hub = types.ModuleType("huggingface_hub")
        fake_hf_hub.snapshot_download = MagicMock(return_value="/fake/path")

        import sys

        with (
            patch("srtctl.cli.do_sweep.start_srun_process") as mock_srun,
            patch.dict(sys.modules, {"huggingface_hub": fake_hf_hub}),
        ):
            orch._ensure_model_cached()
            mock_srun.assert_not_called()
            fake_hf_hub.snapshot_download.assert_called_once()

    def test_runs_srun_on_single_node(self, tmp_path: Path):
        """Pre-download should run huggingface-cli on exactly one node."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
name: test
model:
  path: "hf:nvidia/Kimi-K2.5-NVFP4"
  container: "test-image:latest"
  precision: fp4
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "{tmp_path / "cache"}"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=mock_proc) as mock_srun:
            orch._ensure_model_cached()

            mock_srun.assert_called_once()
            kwargs = mock_srun.call_args.kwargs

            # Should run on exactly one node (first worker node)
            assert kwargs["nodelist"] == ["node-01"]

            # Command should contain huggingface-cli download with model name
            cmd_str = " ".join(kwargs["command"])
            assert "huggingface-cli download" in cmd_str
            assert "nvidia/Kimi-K2.5-NVFP4" in cmd_str
            assert str(tmp_path / "cache") in cmd_str

            # Should use the container image
            assert kwargs["container_image"] == "test-image:latest"

            # Should pass HF env vars to srun
            assert "HF_HOME" in kwargs["env_to_set"]

            # Should wait with a timeout
            mock_proc.wait.assert_called_once()
            assert mock_proc.wait.call_args.kwargs.get("timeout") is not None

    def test_passes_hf_token_to_srun(self, tmp_path: Path):
        """HF_TOKEN from backend env should be passed to the pre-download srun."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
name: test
model:
  path: "hf:nvidia/Kimi-K2.5-NVFP4"
  container: "test-image:latest"
  precision: fp4
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "{tmp_path / "cache"}"
    HF_TOKEN: "hf_secret_token_123"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=mock_proc) as mock_srun:
            orch._ensure_model_cached()

            kwargs = mock_srun.call_args.kwargs
            assert kwargs["env_to_set"]["HF_TOKEN"] == "hf_secret_token_123"

    def test_handles_download_timeout_gracefully(self, tmp_path: Path):
        """Timed-out pre-download should kill the process and continue."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
name: test
model:
  path: "hf:nvidia/Kimi-K2.5-NVFP4"
  container: "test-image:latest"
  precision: fp4
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "{tmp_path / "cache"}"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))

        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("srun", 3600), None]

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=mock_proc):
            # Should NOT raise
            orch._ensure_model_cached()
            # Should have killed the process
            mock_proc.kill.assert_called_once()

    def test_handles_download_failure_gracefully(self, tmp_path: Path):
        """Failed pre-download should warn but not raise."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
name: test
model:
  path: "hf:nvidia/Kimi-K2.5-NVFP4"
  container: "test-image:latest"
  precision: fp4
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "{tmp_path / "cache"}"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1  # Non-zero = failure

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=mock_proc):
            # Should NOT raise, just log a warning
            orch._ensure_model_cached()

    def test_handles_srun_launch_exception_gracefully(self, tmp_path: Path):
        """Exception from start_srun_process should be caught, not abort the sweep."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
name: test
model:
  path: "hf:nvidia/Kimi-K2.5-NVFP4"
  container: "test-image:latest"
  precision: fp4
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "{tmp_path / "cache"}"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))

        with patch("srtctl.cli.do_sweep.start_srun_process", side_effect=OSError("srun not found")):
            # Should NOT raise - best-effort pre-download
            orch._ensure_model_cached()


# Tests for run() guard on is_hf_model


class TestRunHfModelGuard:
    """Tests that run() only triggers HF cache operations for HF models."""

    def test_local_model_skips_hf_cache_operations(self, tmp_path: Path):
        """Local model paths should not trigger _clean_stale_hf_locks or _ensure_model_cached."""
        model_dir = tmp_path / "local_model"
        model_dir.mkdir()

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"""
name: test
model:
  path: "{model_dir}"
  container: "test-image:latest"
  precision: fp16
backend:
  type: vllm
  prefill_environment:
    HF_HOME: "{tmp_path / "cache"}"
  vllm_config:
    prefill:
      tensor-parallel-size: 1
    decode:
      tensor-parallel-size: 1
resources:
  gpu_type: gb200
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 4
  decode_workers: 4
  gpus_per_node: 4
"""
        )
        orch = _make_orchestrator(str(config_file))

        with (
            patch.object(orch, "_clean_stale_hf_locks") as mock_clean,
            patch.object(orch, "_ensure_model_cached") as mock_ensure,
            patch.object(orch, "start_head_infrastructure"),
            patch.object(orch, "start_all_workers", return_value=[]),
            patch.object(orch, "start_frontend", return_value=[]),
            patch.object(orch, "run_benchmark", return_value=0),
            patch.object(orch, "run_postprocess"),
            patch("srtctl.cli.do_sweep.StatusReporter"),
        ):
            orch.run()
            mock_clean.assert_not_called()
            mock_ensure.assert_not_called()
