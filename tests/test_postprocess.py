# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for post-processing: benchmark extraction, S3 upload, and AI analysis."""

from unittest.mock import MagicMock, patch

from srtctl.core.schema import (
    DEFAULT_AI_ANALYSIS_PROMPT,
    AIAnalysisConfig,
    ReportingConfig,
    ReportingStatusConfig,
    S3Config,
)


class TestAIAnalysisConfig:
    """Tests for AIAnalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AIAnalysisConfig()

        assert config.enabled is False
        assert config.openrouter_api_key is None
        assert config.gh_token is None
        assert config.repos_to_search == ["sgl-project/sglang", "ai-dynamo/dynamo"]
        assert config.pr_search_days == 14
        assert config.prompt is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AIAnalysisConfig(
            enabled=True,
            openrouter_api_key="sk-or-test-key",
            gh_token="ghp_test_token",
            repos_to_search=["my-org/my-repo"],
            pr_search_days=7,
            prompt="Custom prompt: {log_dir}",
        )

        assert config.enabled is True
        assert config.openrouter_api_key == "sk-or-test-key"
        assert config.gh_token == "ghp_test_token"
        assert config.repos_to_search == ["my-org/my-repo"]
        assert config.pr_search_days == 7
        assert config.prompt == "Custom prompt: {log_dir}"

    def test_get_prompt_with_default(self):
        """Test get_prompt uses default template when prompt is None."""
        config = AIAnalysisConfig()
        prompt = config.get_prompt("/path/to/logs")

        assert "/path/to/logs" in prompt
        assert "sgl-project/sglang, ai-dynamo/dynamo" in prompt
        assert "14" in prompt  # pr_search_days

    def test_get_prompt_with_custom_template(self):
        """Test get_prompt uses custom template."""
        config = AIAnalysisConfig(
            prompt="Analyze logs in {log_dir}, search {repos} for last {pr_days} days",
            repos_to_search=["my-repo"],
            pr_search_days=7,
        )
        prompt = config.get_prompt("/my/logs")

        assert prompt == "Analyze logs in /my/logs, search my-repo for last 7 days"

    def test_get_prompt_variable_substitution(self):
        """Test all template variables are substituted."""
        config = AIAnalysisConfig(
            repos_to_search=["repo1", "repo2", "repo3"],
            pr_search_days=30,
        )
        prompt = config.get_prompt("/test/dir")

        assert "/test/dir" in prompt
        assert "repo1, repo2, repo3" in prompt
        assert "30" in prompt


class TestDefaultPrompt:
    """Tests for the default AI analysis prompt."""

    def test_default_prompt_has_placeholders(self):
        """Test default prompt has all required placeholders."""
        assert "{log_dir}" in DEFAULT_AI_ANALYSIS_PROMPT
        assert "{repos}" in DEFAULT_AI_ANALYSIS_PROMPT
        assert "{pr_days}" in DEFAULT_AI_ANALYSIS_PROMPT

    def test_default_prompt_mentions_gh_cli(self):
        """Test default prompt tells Claude about gh CLI."""
        assert "gh" in DEFAULT_AI_ANALYSIS_PROMPT.lower()
        assert "github" in DEFAULT_AI_ANALYSIS_PROMPT.lower() or "PR" in DEFAULT_AI_ANALYSIS_PROMPT

    def test_default_prompt_mentions_output_file(self):
        """Test default prompt tells Claude to write ai_analysis.md."""
        assert "ai_analysis.md" in DEFAULT_AI_ANALYSIS_PROMPT


class TestClusterConfigIntegration:
    """Tests for reporting config in cluster config."""

    def test_cluster_config_with_reporting(self):
        """Test ClusterConfig can include ReportingConfig with all sub-configs."""
        from srtctl.core.schema import ClusterConfig

        cluster_config = ClusterConfig(
            default_account="test-account",
            reporting=ReportingConfig(
                status=ReportingStatusConfig(endpoint="https://dashboard.example.com"),
                ai_analysis=AIAnalysisConfig(
                    enabled=True,
                    openrouter_api_key="sk-or-test",
                ),
                s3=S3Config(
                    bucket="test-bucket",
                    prefix="logs",
                    region="us-west-2",
                ),
            ),
        )

        assert cluster_config.reporting is not None
        assert cluster_config.reporting.status is not None
        assert cluster_config.reporting.status.endpoint == "https://dashboard.example.com"
        assert cluster_config.reporting.ai_analysis is not None
        assert cluster_config.reporting.ai_analysis.enabled is True
        assert cluster_config.reporting.ai_analysis.openrouter_api_key == "sk-or-test"
        assert cluster_config.reporting.s3 is not None
        assert cluster_config.reporting.s3.bucket == "test-bucket"

    def test_cluster_config_without_reporting(self):
        """Test ClusterConfig works without ReportingConfig."""
        from srtctl.core.schema import ClusterConfig

        cluster_config = ClusterConfig(
            default_account="test-account",
        )

        assert cluster_config.reporting is None


class TestPostProcessStageMixin:
    """Tests for PostProcessStageMixin."""

    def _create_mixin_with_mocks(self):
        """Create a mixin instance with all post-processing methods mocked."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        mixin = PostProcessStageMixin()
        mixin._copy_config_to_logs = MagicMock()
        mixin._generate_rollup = MagicMock()
        mixin._extract_benchmark_results = MagicMock(return_value=None)
        mixin._run_postprocess_container = MagicMock(return_value=(None, None))
        mixin._report_metrics = MagicMock()
        mixin._get_ai_analysis_config = MagicMock(return_value=None)
        mixin._run_ai_analysis = MagicMock()
        return mixin

    def test_resolve_secret_from_config(self):
        """Test secret resolution prefers config value."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        mixin = PostProcessStageMixin()
        result = mixin._resolve_secret("config-value", "ENV_VAR")

        assert result == "config-value"

    def test_resolve_secret_from_env(self, monkeypatch):
        """Test secret resolution falls back to environment variable."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        monkeypatch.setenv("TEST_SECRET", "env-value")

        mixin = PostProcessStageMixin()
        result = mixin._resolve_secret(None, "TEST_SECRET")

        assert result == "env-value"

    def test_resolve_secret_returns_none_when_not_found(self, monkeypatch):
        """Test secret resolution returns None when not found anywhere."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        # Ensure env var is not set
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)

        mixin = PostProcessStageMixin()
        result = mixin._resolve_secret(None, "NONEXISTENT_VAR")

        assert result is None

    def test_run_postprocess_always_runs_extraction_and_upload(self):
        """Test run_postprocess always runs benchmark extraction and S3 upload."""
        mixin = self._create_mixin_with_mocks()

        mixin.run_postprocess(0)  # success exit code

        mixin._extract_benchmark_results.assert_called_once()
        mixin._run_postprocess_container.assert_called_once()
        mixin._report_metrics.assert_called_once()

    def test_run_postprocess_skips_ai_on_success(self):
        """Test run_postprocess skips AI analysis when exit_code is 0."""
        mixin = self._create_mixin_with_mocks()

        mixin.run_postprocess(0)

        mixin._get_ai_analysis_config.assert_not_called()
        mixin._run_ai_analysis.assert_not_called()

    def test_run_postprocess_skips_ai_when_not_configured(self):
        """Test run_postprocess skips AI analysis when not configured."""
        mixin = self._create_mixin_with_mocks()
        mixin._get_ai_analysis_config.return_value = None

        mixin.run_postprocess(1)

        mixin._get_ai_analysis_config.assert_called_once()
        mixin._run_ai_analysis.assert_not_called()

    def test_run_postprocess_skips_ai_when_disabled(self):
        """Test run_postprocess skips AI analysis when disabled."""
        mixin = self._create_mixin_with_mocks()
        mixin._get_ai_analysis_config.return_value = AIAnalysisConfig(enabled=False)

        mixin.run_postprocess(1)

        mixin._get_ai_analysis_config.assert_called_once()
        mixin._run_ai_analysis.assert_not_called()

    def test_run_postprocess_calls_ai_analysis_when_enabled(self):
        """Test run_postprocess calls _run_ai_analysis when enabled and failed."""
        mixin = self._create_mixin_with_mocks()
        config = AIAnalysisConfig(enabled=True, openrouter_api_key="sk-or-test")
        mixin._get_ai_analysis_config.return_value = config

        mixin.run_postprocess(1)

        mixin._run_ai_analysis.assert_called_once_with(config)


class TestAIAnalysisConfigSchema:
    """Tests for AIAnalysisConfig marshmallow schema."""

    def test_schema_load_minimal(self):
        """Test loading minimal config from dict."""
        schema = AIAnalysisConfig.Schema()
        config = schema.load({"enabled": True})

        assert config.enabled is True
        assert config.repos_to_search == ["sgl-project/sglang", "ai-dynamo/dynamo"]

    def test_schema_load_full(self):
        """Test loading full config from dict."""
        schema = AIAnalysisConfig.Schema()
        config = schema.load(
            {
                "enabled": True,
                "openrouter_api_key": "sk-or-test",
                "gh_token": "ghp_test",
                "repos_to_search": ["my/repo"],
                "pr_search_days": 7,
                "prompt": "Custom prompt",
            }
        )

        assert config.enabled is True
        assert config.openrouter_api_key == "sk-or-test"
        assert config.gh_token == "ghp_test"
        assert config.repos_to_search == ["my/repo"]
        assert config.pr_search_days == 7
        assert config.prompt == "Custom prompt"

    def test_schema_dump(self):
        """Test dumping config to dict."""
        config = AIAnalysisConfig(
            enabled=True,
            openrouter_api_key="sk-or-test",
        )
        schema = AIAnalysisConfig.Schema()
        data = schema.dump(config)

        assert data["enabled"] is True
        assert data["openrouter_api_key"] == "sk-or-test"
        assert data["pr_search_days"] == 14  # default


class TestS3Config:
    """Tests for S3Config dataclass."""

    def test_required_bucket(self):
        """Test S3Config requires bucket."""
        config = S3Config(bucket="my-bucket")
        assert config.bucket == "my-bucket"
        assert config.prefix is None
        assert config.region is None

    def test_full_config(self):
        """Test S3Config with all fields."""
        config = S3Config(
            bucket="my-bucket",
            prefix="logs/benchmark",
            region="us-west-2",
            access_key_id="AKIA...",
            secret_access_key="secret...",
        )
        assert config.bucket == "my-bucket"
        assert config.prefix == "logs/benchmark"
        assert config.region == "us-west-2"
        assert config.access_key_id == "AKIA..."
        assert config.secret_access_key == "secret..."

    def test_schema_load(self):
        """Test loading S3Config from dict."""
        schema = S3Config.Schema()
        config = schema.load(
            {
                "bucket": "test-bucket",
                "prefix": "prefix",
                "region": "eu-west-1",
            }
        )
        assert config.bucket == "test-bucket"
        assert config.prefix == "prefix"
        assert config.region == "eu-west-1"


class TestReportingConfig:
    """Tests for ReportingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReportingConfig()
        assert config.status is None
        assert config.ai_analysis is None
        assert config.s3 is None

    def test_full_config(self):
        """Test ReportingConfig with all sub-configs."""
        config = ReportingConfig(
            status=ReportingStatusConfig(endpoint="https://api.example.com"),
            ai_analysis=AIAnalysisConfig(enabled=True),
            s3=S3Config(bucket="logs-bucket"),
        )
        assert config.status is not None
        assert config.status.endpoint == "https://api.example.com"
        assert config.ai_analysis is not None
        assert config.ai_analysis.enabled is True
        assert config.s3 is not None
        assert config.s3.bucket == "logs-bucket"

    def test_schema_load(self):
        """Test loading ReportingConfig from nested dict."""
        schema = ReportingConfig.Schema()
        config = schema.load(
            {
                "status": {"endpoint": "https://dashboard.example.com"},
                "ai_analysis": {"enabled": True, "pr_search_days": 7},
                "s3": {"bucket": "my-bucket", "prefix": "logs"},
            }
        )
        assert config.status.endpoint == "https://dashboard.example.com"
        assert config.ai_analysis.enabled is True
        assert config.ai_analysis.pr_search_days == 7
        assert config.s3.bucket == "my-bucket"
        assert config.s3.prefix == "logs"


class TestReportingStatusConfig:
    """Tests for ReportingStatusConfig dataclass."""

    def test_required_endpoint(self):
        """Test ReportingStatusConfig requires endpoint."""
        config = ReportingStatusConfig(endpoint="https://api.example.com")
        assert config.endpoint == "https://api.example.com"

    def test_schema_load(self):
        """Test loading ReportingStatusConfig from dict."""
        schema = ReportingStatusConfig.Schema()
        config = schema.load({"endpoint": "https://dashboard.example.com"})
        assert config.endpoint == "https://dashboard.example.com"


class TestRollupFaultTolerance:
    """Tests for rollup generation fault tolerance.

    These tests verify that failures in rollup generation never crash the benchmark.
    """

    def _create_mixin_with_runtime(self, tmp_path, benchmark_type="sa-bench"):
        """Create a mixin instance with real runtime and config mocks."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        mixin = PostProcessStageMixin()

        # Mock config
        mixin.config = MagicMock()
        mixin.config.benchmark.type = benchmark_type

        # Mock runtime with real tmp_path for log_dir
        mixin.runtime = MagicMock()
        mixin.runtime.log_dir = tmp_path
        mixin.runtime.job_id = "12345"

        return mixin

    def test_generate_rollup_no_script_does_not_raise(self, tmp_path):
        """Test _generate_rollup returns silently when no rollup script exists."""
        mixin = self._create_mixin_with_runtime(tmp_path, benchmark_type="nonexistent-benchmark")

        # Should not raise - just returns silently
        mixin._generate_rollup()

    def test_generate_rollup_script_failure_does_not_raise(self, tmp_path):
        """Test _generate_rollup handles script failures gracefully."""
        mixin = self._create_mixin_with_runtime(tmp_path, benchmark_type="sa-bench")

        # No sa-bench results exist, so rollup.py will fail
        # But it should not raise
        mixin._generate_rollup()

    def test_generate_rollup_timeout_does_not_raise(self, tmp_path):
        """Test _generate_rollup handles timeout gracefully."""
        import subprocess

        mixin = self._create_mixin_with_runtime(tmp_path, benchmark_type="sa-bench")

        # Mock subprocess.run to raise TimeoutExpired
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=30)

            # Should not raise
            mixin._generate_rollup()

    def test_generate_rollup_exception_does_not_raise(self, tmp_path):
        """Test _generate_rollup handles unexpected exceptions gracefully."""
        mixin = self._create_mixin_with_runtime(tmp_path, benchmark_type="sa-bench")

        # Mock subprocess.run to raise generic exception
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Unexpected error")

            # Should not raise
            mixin._generate_rollup()

    def test_extract_results_fallback_to_raw_output(self, tmp_path):
        """Test _extract_benchmark_results falls back to benchmark.out when no rollup."""
        mixin = self._create_mixin_with_runtime(tmp_path)

        # Create benchmark.out but no rollup.json
        benchmark_out = tmp_path / "benchmark.out"
        benchmark_out.write_text("Raw benchmark output here")

        result = mixin._extract_benchmark_results()

        assert result is not None
        assert result["benchmark_type"] == "unknown"
        assert result["raw_output"] == "Raw benchmark output here"

    def test_extract_results_corrupted_rollup_fallback(self, tmp_path):
        """Test _extract_benchmark_results falls back when rollup.json is corrupted."""
        mixin = self._create_mixin_with_runtime(tmp_path)

        # Create corrupted rollup.json
        rollup = tmp_path / "benchmark-rollup.json"
        rollup.write_text("not valid json {{{")

        # Create backup benchmark.out
        benchmark_out = tmp_path / "benchmark.out"
        benchmark_out.write_text("Fallback output")

        result = mixin._extract_benchmark_results()

        assert result is not None
        assert result["benchmark_type"] == "unknown"
        assert result["raw_output"] == "Fallback output"

    def test_extract_results_no_files_returns_none(self, tmp_path):
        """Test _extract_benchmark_results returns None when no files exist."""
        mixin = self._create_mixin_with_runtime(tmp_path)

        result = mixin._extract_benchmark_results()

        assert result is None

    def test_extract_results_valid_rollup(self, tmp_path):
        """Test _extract_benchmark_results reads valid rollup.json."""
        import json

        mixin = self._create_mixin_with_runtime(tmp_path)

        # Create valid rollup.json
        rollup_data = {
            "benchmark_type": "sa-bench",
            "timestamp": "2026-01-27T00:00:00Z",
            "config": {"model": "test-model", "isl": 100, "osl": 100},
            "runs": [{"concurrency": 4, "throughput_toks": 100.0}],
        }
        rollup = tmp_path / "benchmark-rollup.json"
        rollup.write_text(json.dumps(rollup_data))

        result = mixin._extract_benchmark_results()

        assert result is not None
        assert result["benchmark_type"] == "sa-bench"
        assert result["config"]["model"] == "test-model"
        assert len(result["runs"]) == 1

    def test_run_postprocess_completes_with_rollup_failure(self, tmp_path):
        """Test run_postprocess completes even when rollup fails entirely."""
        mixin = self._create_mixin_with_runtime(tmp_path, benchmark_type="sa-bench")

        # Mock all the other methods to isolate rollup behavior
        mixin._run_postprocess_container = MagicMock(return_value=(None, None))
        mixin._report_metrics = MagicMock()
        mixin._get_ai_analysis_config = MagicMock(return_value=None)

        # Mock _generate_rollup to raise (simulating worst case)
        # But actually, _generate_rollup should never raise - let's verify that
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Catastrophic failure")

            # Should complete without raising
            mixin.run_postprocess(exit_code=0)

        # Verify other methods were still called
        mixin._run_postprocess_container.assert_called_once()
        mixin._report_metrics.assert_called_once()


class TestS3UploadFaultTolerance:
    """Tests for S3 upload fault tolerance.

    These tests verify that failures in S3 upload never crash the benchmark.
    """

    def _create_mixin_with_runtime(self, tmp_path):
        """Create a mixin instance with runtime mocks."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        mixin = PostProcessStageMixin()

        # Mock config
        mixin.config = MagicMock()
        mixin.config.benchmark.type = "sa-bench"

        # Mock runtime
        mixin.runtime = MagicMock()
        mixin.runtime.log_dir = tmp_path
        mixin.runtime.job_id = "12345"
        mixin.runtime.nodes.head = "node001"

        return mixin

    def test_no_s3_config_returns_none(self, tmp_path):
        """Test _run_postprocess_container returns None when S3 not configured."""
        mixin = self._create_mixin_with_runtime(tmp_path)

        # Mock _get_s3_config to return None
        mixin._get_s3_config = MagicMock(return_value=None)

        result = mixin._run_postprocess_container()

        assert result == (None, None)

    def test_srun_failure_does_not_raise(self, tmp_path):
        """Test _run_postprocess_container handles srun failure gracefully."""
        mixin = self._create_mixin_with_runtime(tmp_path)

        # Mock S3 config
        mixin._get_s3_config = MagicMock(return_value=S3Config(bucket="test-bucket"))

        # Mock start_srun_process to raise
        with patch("srtctl.cli.mixins.postprocess_stage.start_srun_process") as mock_srun:
            mock_srun.side_effect = Exception("SLURM is down")

            result = mixin._run_postprocess_container()

        assert result == (None, None)

    def test_srun_timeout_does_not_raise(self, tmp_path):
        """Test _run_postprocess_container handles timeout gracefully."""
        import subprocess

        mixin = self._create_mixin_with_runtime(tmp_path)

        # Mock S3 config
        mixin._get_s3_config = MagicMock(return_value=S3Config(bucket="test-bucket"))

        # Mock start_srun_process to return a process that times out
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=600)
        mock_proc.kill = MagicMock()

        with patch("srtctl.cli.mixins.postprocess_stage.start_srun_process") as mock_srun:
            mock_srun.return_value = mock_proc

            result = mixin._run_postprocess_container()

        assert result == (None, None)
        mock_proc.kill.assert_called_once()

    def test_srun_nonzero_exit_does_not_raise(self, tmp_path):
        """Test _run_postprocess_container handles non-zero exit gracefully."""
        mixin = self._create_mixin_with_runtime(tmp_path)

        # Mock S3 config
        mixin._get_s3_config = MagicMock(return_value=S3Config(bucket="test-bucket"))

        # Mock start_srun_process to return a process that fails
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_proc.returncode = 1  # Non-zero exit

        with patch("srtctl.cli.mixins.postprocess_stage.start_srun_process") as mock_srun:
            mock_srun.return_value = mock_proc

            parquet_path, s3_url = mixin._run_postprocess_container()

        # Should return None for s3_url on failure
        assert s3_url is None

    def test_run_postprocess_completes_with_s3_failure(self, tmp_path):
        """Test run_postprocess completes even when S3 upload fails entirely."""
        mixin = self._create_mixin_with_runtime(tmp_path)

        # Mock _generate_rollup
        mixin._generate_rollup = MagicMock()

        # Mock _run_postprocess_container to simulate S3 failure
        mixin._run_postprocess_container = MagicMock(return_value=(None, None))

        # Mock other methods
        mixin._report_metrics = MagicMock()
        mixin._get_ai_analysis_config = MagicMock(return_value=None)

        # Should complete without raising
        mixin.run_postprocess(exit_code=0)

        # Verify _report_metrics was still called (with None for s3_url, exit_code=0)
        mixin._report_metrics.assert_called_once_with(None, None, 0)


class TestCopyConfigToLogs:
    """Tests for copying config YAML to log directory."""

    def _create_mixin_with_runtime(self, tmp_path):
        """Create a mixin instance with runtime pointing to tmp_path/logs."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        mixin = PostProcessStageMixin()
        mixin.runtime = MagicMock()
        mixin.runtime.log_dir = log_dir
        return mixin, log_dir

    def test_copies_config_yaml(self, tmp_path):
        """Test config.yaml is copied from output dir to log dir."""
        mixin, log_dir = self._create_mixin_with_runtime(tmp_path)

        config_src = tmp_path / "config.yaml"
        config_src.write_text("name: test-config\n")

        mixin._copy_config_to_logs()

        assert (log_dir / "config.yaml").exists()
        assert (log_dir / "config.yaml").read_text() == "name: test-config\n"

    def test_copies_sbatch_script(self, tmp_path):
        """Test sbatch_script.sh is also copied."""
        mixin, log_dir = self._create_mixin_with_runtime(tmp_path)

        script_src = tmp_path / "sbatch_script.sh"
        script_src.write_text("#!/bin/bash\necho hello\n")

        mixin._copy_config_to_logs()

        assert (log_dir / "sbatch_script.sh").exists()

    def test_no_config_does_not_raise(self, tmp_path):
        """Test graceful handling when config.yaml doesn't exist."""
        mixin, log_dir = self._create_mixin_with_runtime(tmp_path)

        mixin._copy_config_to_logs()

        assert not (log_dir / "config.yaml").exists()

    def test_copy_failure_does_not_raise(self, tmp_path):
        """Test graceful handling when copy fails."""
        mixin, log_dir = self._create_mixin_with_runtime(tmp_path)

        config_src = tmp_path / "config.yaml"
        config_src.write_text("name: test\n")

        log_dir.chmod(0o444)
        try:
            mixin._copy_config_to_logs()  # Should not raise
        finally:
            log_dir.chmod(0o755)
