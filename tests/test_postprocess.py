# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for artifact finalization, S3 upload, and legacy reporting config."""

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

    def _create_mixin_with_mocks(self, tmp_path=None):
        """Create a mixin instance with all post-processing methods mocked."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        mixin = PostProcessStageMixin()

        # Mock runtime and config for lockfile writing
        if tmp_path is None:
            import tempfile
            from pathlib import Path

            tmp_path = Path(tempfile.mkdtemp())
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        mixin.runtime = MagicMock()
        mixin.runtime.log_dir = log_dir
        mixin.config = MagicMock()

        mixin._copy_config_to_logs = MagicMock()
        mixin._run_postprocess_container = MagicMock(return_value=None)
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

    def test_run_postprocess_copies_artifacts_and_uploads(self):
        """Retain job inputs for upload and stash the artifact URL."""
        mixin = self._create_mixin_with_mocks()

        mixin.run_postprocess(0)  # success exit code

        mixin._copy_config_to_logs.assert_called_once()
        mixin._run_postprocess_container.assert_called_once()
        # logs_url is stashed on self for do_sweep's final report_completed PUT
        assert mixin._last_logs_url is None  # no S3 configured in this mock

    def test_run_postprocess_eagerly_pushes_logs_url_to_reporter(self):
        """When a reporter is passed and S3 sync produces a URL, push eagerly."""
        mixin = self._create_mixin_with_mocks()
        s3_url = "s3://bucket/prefix/12345/"
        mixin._run_postprocess_container = MagicMock(return_value=s3_url)
        reporter = MagicMock()

        mixin.run_postprocess(0, reporter=reporter)

        reporter.report_artifacts.assert_called_once_with(logs_url=s3_url)
        assert mixin._last_logs_url == s3_url

    def test_run_postprocess_skips_eager_push_when_no_s3_url(self):
        """No S3 URL means no eager report_artifacts call."""
        mixin = self._create_mixin_with_mocks()
        reporter = MagicMock()

        mixin.run_postprocess(0, reporter=reporter)

        reporter.report_artifacts.assert_not_called()
        assert mixin._last_logs_url is None

    def test_run_postprocess_skips_eager_push_without_reporter(self):
        """Without a reporter, stash happens but no PUT is attempted."""
        mixin = self._create_mixin_with_mocks()
        s3_url = "s3://bucket/prefix/12345/"
        mixin._run_postprocess_container = MagicMock(return_value=s3_url)

        # Should not raise even though no reporter is provided
        mixin.run_postprocess(0)

        assert mixin._last_logs_url == s3_url


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
        # None means "use the defaults"; an explicit [] means "nothing".
        assert config.exclude is None
        assert config.archive is None

    def test_schema_load_upload_policy(self):
        config = S3Config.Schema().load({"bucket": "b", "exclude": [], "archive": ["*.out"]})
        assert config.exclude == []
        assert config.archive == ["*.out"]

    def test_default_excludes_are_scoped_to_the_aiperf_artifact_roots(self):
        """A same-named file from another benchmark type (a custom runner's inputs.json) must not be dropped."""
        from srtctl.core.schema import DEFAULT_S3_EXCLUDE

        aiperf_patterns = [p for p in DEFAULT_S3_EXCLUDE if not p.startswith("perf_dashboard")]
        assert aiperf_patterns, DEFAULT_S3_EXCLUDE
        for pattern in aiperf_patterns:
            assert pattern.startswith(("artifacts/*/", "sa-bench_*/*/")), pattern
        for name in ("server_metrics_export.jsonl", "gpu_telemetry_export.jsonl", "inputs.json"):
            assert f"artifacts/*/{name}" in DEFAULT_S3_EXCLUDE
            assert f"sa-bench_*/*/{name}" in DEFAULT_S3_EXCLUDE
        assert "perf_dashboard_bundle/*" in DEFAULT_S3_EXCLUDE and "perf_dashboard.json" in DEFAULT_S3_EXCLUDE

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

        assert result is None

    def test_srun_failure_does_not_raise(self, tmp_path):
        """Test _run_postprocess_container handles srun failure gracefully."""
        mixin = self._create_mixin_with_runtime(tmp_path)

        # Mock S3 config
        mixin._get_s3_config = MagicMock(return_value=S3Config(bucket="test-bucket"))

        # Mock start_srun_process to raise
        with patch("srtctl.cli.mixins.postprocess_stage.start_srun_process") as mock_srun:
            mock_srun.side_effect = Exception("SLURM is down")

            result = mixin._run_postprocess_container()

        assert result is None

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

        assert result is None
        mock_proc.kill.assert_called_once()

    def test_upload_script_carries_the_policy(self, tmp_path):
        """The container script excludes the defaults, archives aiperf's per-request file, and is valid bash."""
        import shlex
        import subprocess

        from srtctl.core.schema import DEFAULT_S3_ARCHIVE, DEFAULT_S3_EXCLUDE

        mixin = self._create_mixin_with_runtime(tmp_path)
        mixin._get_s3_config = MagicMock(
            return_value=S3Config(bucket="test-bucket", endpoint_url="https://minio.example")
        )
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        with patch("srtctl.cli.mixins.postprocess_stage.start_srun_process") as mock_srun:
            mock_srun.return_value = mock_proc
            result = mixin._run_postprocess_container()

        assert result is not None and result.startswith("s3://test-bucket/srtslurm/") and result.endswith("/12345/")
        script = mock_srun.call_args.kwargs["command"][2]
        for pattern in DEFAULT_S3_EXCLUDE:
            assert f"--exclude {shlex.quote(pattern)}" in script, pattern
        # The archived files are kept out of the plain sync, with ** collapsed to the AWS wildcard.
        assert "--exclude 'artifacts/*/profile_export.jsonl'" in script
        assert "--exclude 'sa-bench_*/*/profile_export.jsonl'" in script
        assert shlex.quote(__import__("json").dumps(list(DEFAULT_S3_ARCHIVE))) in script
        assert 'aws s3 cp "$archive_path"' in script
        assert "--endpoint-url https://minio.example" in script
        assert '"s3_url": "s3://test-bucket/' in script and '"exclude": [' in script
        assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False).returncode == 0

    def test_upload_script_with_policy_disabled(self, tmp_path):
        """exclude: [] and archive: [] ship the whole directory as-is, like before."""
        import subprocess

        mixin = self._create_mixin_with_runtime(tmp_path)
        script = mixin._build_postprocess_script("s3://b/p/1/", "", exclude=[], archive=[])
        assert "--exclude" not in script
        assert "Packing" not in script and "aws s3 cp" in script  # the cp branch stays but archive_path is empty
        assert "aws s3 sync /logs s3://b/p/1/" in script
        assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False).returncode == 0

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

            s3_url = mixin._run_postprocess_container()

        # Should return None for s3_url on failure
        assert s3_url is None

    def test_postprocess_script_only_uploads(self, tmp_path):
        """The upload container ships the log directory as is; no log parser runs in it."""
        mixin = self._create_mixin_with_runtime(tmp_path)
        script = mixin._build_postprocess_script("s3://test-bucket/run/", "")

        assert "aws s3 sync /logs s3://test-bucket/run/" in script
        assert "srtlog" not in script
        assert '"s3_url": "s3://test-bucket/run/"' in script

    def test_run_postprocess_completes_with_s3_failure(self, tmp_path):
        """Test run_postprocess completes even when S3 upload fails entirely."""
        mixin = self._create_mixin_with_runtime(tmp_path)

        # Mock _run_postprocess_container to simulate S3 failure
        mixin._run_postprocess_container = MagicMock(return_value=None)

        reporter = MagicMock()

        # Should complete without raising
        mixin.run_postprocess(exit_code=0, reporter=reporter)

        # No S3 URL => no eager artifact push, and logs_url stash is None so the
        # final report_completed PUT sends exit_code only.
        reporter.report_artifacts.assert_not_called()
        assert mixin._last_logs_url is None


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
        mixin.runtime.job_id = "12345"
        return mixin, log_dir

    def test_copies_config_yaml(self, tmp_path):
        """Test config.yaml is copied from output dir to log dir."""
        mixin, log_dir = self._create_mixin_with_runtime(tmp_path)

        config_src = tmp_path / "config.yaml"
        config_src.write_text("name: test-config\n")

        mixin._copy_config_to_logs()

        assert (log_dir / "config.yaml").exists()
        assert (log_dir / "config.yaml").read_text() == "name: test-config\n"

    def test_copies_resolved_override_config(self, tmp_path):
        """Test resolved override/zip configs (config_{suffix}.yaml) are copied too.

        Override/zip submissions write the unresolved source as config.yaml and the
        actually-executed resolved variant as config_{suffix}.yaml. Both must reach
        the log dir so the resolved config is uploaded to S3, not just the source.
        """
        mixin, log_dir = self._create_mixin_with_runtime(tmp_path)

        (tmp_path / "config.yaml").write_text("base:\n  name: test\n")
        (tmp_path / "config_tp_0.yaml").write_text("name: test\ntensor_parallel_size: 4\n")
        (tmp_path / "config_resolved.yaml").write_text("name: test\nresolved: true\n")

        mixin._copy_config_to_logs()

        assert (log_dir / "config.yaml").exists()
        assert (log_dir / "config_tp_0.yaml").exists()
        assert (log_dir / "config_tp_0.yaml").read_text() == "name: test\ntensor_parallel_size: 4\n"
        assert (log_dir / "config_resolved.yaml").exists()

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

    def test_copies_job_id_json(self, tmp_path):
        """Test {job_id}.json is copied from output dir to log dir."""
        mixin, log_dir = self._create_mixin_with_runtime(tmp_path)

        json_src = tmp_path / "12345.json"
        json_src.write_text('{"job_id": "12345"}\n')

        mixin._copy_config_to_logs()

        assert (log_dir / "12345.json").exists()
        assert (log_dir / "12345.json").read_text() == '{"job_id": "12345"}\n'

    def test_copies_git_state_txt(self, tmp_path):
        """Test git_state.txt is copied from output dir to log dir."""
        from srtctl.core.git_state import GIT_STATE_FILENAME

        mixin, log_dir = self._create_mixin_with_runtime(tmp_path)

        git_state_src = tmp_path / GIT_STATE_FILENAME
        git_state_src.write_text("# srtctl git state snapshot\n")

        mixin._copy_config_to_logs()

        assert (log_dir / GIT_STATE_FILENAME).exists()
        assert (log_dir / GIT_STATE_FILENAME).read_text() == "# srtctl git state snapshot\n"

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


class TestArchiveScript:
    """The inline archive builder that runs inside the upload container."""

    @staticmethod
    def _tree(root):
        files = {
            "artifacts/run_c32/profile_export.jsonl": "a" * 5000,
            "artifacts/run_c32/server_metrics_export.jsonl": "m" * 5000,
            "artifacts/run_c128/profile_export.jsonl": "b" * 5000,
            "sa-bench_isl_128/conc_4/aiperf_artifacts/profile_export.jsonl": "c" * 5000,
            "perf_dashboard_bundle/profile_export.jsonl": "d" * 5000,  # a copy; not an archive default
            "sweep_1.log": "log",
        }
        for rel, body in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)

    def _run(self, root, out_dir, patterns):
        import json
        import subprocess
        import sys

        from srtctl.cli.mixins.postprocess_stage import ARCHIVE_SCRIPT

        return subprocess.run(
            [sys.executable, "-", str(root), str(out_dir), json.dumps(patterns)],
            input=ARCHIVE_SCRIPT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_packs_only_the_matching_files_with_relative_names(self, tmp_path):
        import tarfile

        from srtctl.core.schema import DEFAULT_S3_ARCHIVE

        root, out_dir = tmp_path / "logs", tmp_path / "out"
        self._tree(root)
        out_dir.mkdir()
        result = self._run(root, out_dir, list(DEFAULT_S3_ARCHIVE))
        assert result.returncode == 0, result.stderr
        archive = result.stdout.strip().splitlines()[-1]
        assert archive.startswith(str(out_dir / "bundle.tar."))
        assert "3 files" in result.stderr
        with tarfile.open(archive) as tar:
            names = sorted(tar.getnames())
        assert names == [
            "artifacts/run_c128/profile_export.jsonl",
            "artifacts/run_c32/profile_export.jsonl",
            "sa-bench_isl_128/conc_4/aiperf_artifacts/profile_export.jsonl",
        ]
        # The log directory itself is untouched: the archive lives in out_dir only.
        assert not list(root.glob("bundle.tar.*"))

    def test_no_match_writes_nothing_and_exits_zero(self, tmp_path):
        root, out_dir = tmp_path / "logs", tmp_path / "out"
        self._tree(root)
        out_dir.mkdir()
        result = self._run(root, out_dir, ["nothing/**/here.jsonl"])
        assert result.returncode == 0
        assert result.stdout.strip() == ""
        assert "no file matched" in result.stderr
        assert list(out_dir.iterdir()) == []
