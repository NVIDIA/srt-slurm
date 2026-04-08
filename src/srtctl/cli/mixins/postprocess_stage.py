# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-process stage mixin for SweepOrchestrator.

Handles:
- Benchmark result extraction
- srtlog parsing and S3 upload
- AI-powered failure analysis using Claude Code CLI

AI analysis uses Claude Code in headless mode (-p flag) with OpenRouter for authentication.
See: https://openrouter.ai/docs/guides/claude-code-integration
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from srtctl.benchmarks.base import SCRIPTS_DIR
from srtctl.core.config import load_cluster_config
from srtctl.core.schema import AIAnalysisConfig, S3Config
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)


class PostProcessStageMixin:
    """Mixin for post-process stage after benchmark completion.

    Handles AI-powered failure analysis using Claude Code CLI.
    Configuration is loaded from srtslurm.yaml (cluster config).

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    def _get_ai_analysis_config(self) -> AIAnalysisConfig | None:
        """Load AI analysis config from cluster config (reporting.ai_analysis).

        Returns:
            AIAnalysisConfig if configured, None otherwise
        """
        cluster_config = load_cluster_config()
        if not cluster_config:
            return None

        reporting = cluster_config.get("reporting")
        if not reporting:
            return None

        ai_config_dict = reporting.get("ai_analysis")
        if not ai_config_dict:
            return None

        try:
            schema = AIAnalysisConfig.Schema()
            return schema.load(ai_config_dict)
        except Exception as e:
            logger.warning("Failed to parse reporting.ai_analysis config: %s", e)
            return None

    def _get_s3_config(self) -> S3Config | None:
        """Load S3 config from cluster config (under reporting.s3).

        Returns:
            S3Config if configured, None otherwise
        """
        cluster_config = load_cluster_config()
        if not cluster_config:
            return None

        reporting = cluster_config.get("reporting")
        if not reporting:
            return None

        s3_dict = reporting.get("s3")
        if not s3_dict:
            return None

        try:
            schema = S3Config.Schema()
            return schema.load(s3_dict)
        except Exception as e:
            logger.warning("Failed to parse reporting.s3 config: %s", e)
            return None

    def _resolve_secret(self, config_value: str | None, env_var: str) -> str | None:
        """Resolve a secret from config or environment variable.

        Args:
            config_value: Value from config (may be None)
            env_var: Environment variable name to check as fallback

        Returns:
            Resolved secret value, or None if not found
        """
        if config_value:
            return config_value
        return os.environ.get(env_var)

    def _copy_config_to_logs(self) -> None:
        """Copy job artifacts into the log directory so they're included in S3 uploads.

        At submit time, config.yaml, sbatch_script.sh, and {job_id}.json are saved
        to outputs/{job_id}/, but S3 syncs outputs/{job_id}/logs/. This copies them
        into logs/ so they get uploaded alongside benchmark results and worker logs.
        """
        output_dir = self.runtime.log_dir.parent
        files_to_copy = ["config.yaml", "sbatch_script.sh", f"{self.runtime.job_id}.json"]
        for name in files_to_copy:
            src = output_dir / name
            if not src.exists():
                continue
            dst = self.runtime.log_dir / name
            try:
                shutil.copy2(src, dst)
                logger.info("Copied %s to log directory", name)
            except Exception as e:
                logger.warning("Failed to copy %s to log directory: %s", name, e)

    def run_postprocess(self, exit_code: int) -> None:
        """Run post-processing after benchmark completion.

        Handles:
        1. Copy config YAML into log directory (for S3 upload)
        2. Rollup generation (benchmark-specific normalization)
        3. Benchmark result extraction (reads rollup or falls back to raw)
        4. srtlog parsing + S3 upload (if S3 configured)
        5. Metrics reporting to dashboard (if status endpoint configured)
        6. AI-powered failure analysis (only on failures, if enabled)

        Args:
            exit_code: Exit code from the benchmark run
        """
        # Copy config into log directory so it's included in S3 upload
        self._copy_config_to_logs()

        # Generate rollup first (benchmark-specific normalization)
        self._generate_rollup()

        # Extract benchmark results (reads rollup if available)
        benchmark_results = self._extract_benchmark_results()

        # Run srtlog + S3 upload in single container (if S3 configured)
        parquet_path, s3_url = self._run_postprocess_container()

        # Report metrics to dashboard
        self._report_metrics(benchmark_results, s3_url, exit_code)

        # AI analysis only on failures
        if exit_code != 0:
            ai_config = self._get_ai_analysis_config()
            if ai_config and ai_config.enabled:
                logger.info("Running AI-powered failure analysis...")
                self._run_ai_analysis(ai_config)

    def _generate_rollup(self) -> None:
        """Run benchmark-specific rollup script to generate benchmark-rollup.json.

        Each benchmark type can have a rollup.py script that normalizes its output
        into a standardized format for historical tracking.
        """
        benchmark_type = self.config.benchmark.type
        rollup_script = SCRIPTS_DIR / benchmark_type / "rollup.py"

        if not rollup_script.exists():
            logger.debug("No rollup script for %s", benchmark_type)
            return

        try:
            result = subprocess.run(
                ["python3", str(rollup_script), str(self.runtime.log_dir)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning("Rollup failed: %s", result.stderr)
            elif result.stdout:
                logger.info(result.stdout.strip())
        except subprocess.TimeoutExpired:
            logger.warning("Rollup script timed out")
        except Exception as e:
            logger.warning("Rollup error: %s", e)

    def _extract_benchmark_results(self) -> dict[str, Any] | None:
        """Read benchmark-rollup.json if it exists, otherwise fall back to raw output.

        Returns:
            Dictionary with benchmark results, or None if not found
        """
        # Try to read the standardized rollup first
        rollup_file = self.runtime.log_dir / "benchmark-rollup.json"
        if rollup_file.exists():
            try:
                return json.loads(rollup_file.read_text())
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse rollup: %s", e)

        # Fallback to raw output for legacy/failed rollups
        benchmark_out = self.runtime.log_dir / "benchmark.out"
        if benchmark_out.exists():
            return {"benchmark_type": "unknown", "raw_output": benchmark_out.read_text(errors="replace")}

        return None

    def _run_postprocess_container(self) -> tuple[Path | None, str | None]:
        """Run srtlog and upload entire log directory to S3.

        Uploads the complete log directory including:
        - Worker logs (prefill_*.out, decode_*.out, etc.)
        - Benchmark output (benchmark.out, artifacts/)
        - Parquet files from srtlog (cached_assets/)
        - Any other artifacts

        Returns:
            (parquet_path, s3_url) tuple - s3_url points to the log directory
        """
        s3_config = self._get_s3_config()
        if not s3_config:
            logger.debug("S3 not configured, skipping srtlog/upload")
            return None, None

        # S3 path: {prefix}/{YYYY-MM-DD}/{job_id}/
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        s3_prefix = f"{s3_config.prefix or 'srtslurm'}/{date_str}/{self.runtime.job_id}"
        s3_url = f"s3://{s3_config.bucket}/{s3_prefix}/"

        # Build endpoint flag if custom endpoint provided
        endpoint_flag = f"--endpoint-url {s3_config.endpoint_url}" if s3_config.endpoint_url else ""

        # Build the post-processing script
        script = f"""
set -e

# Install uv, srtlog, and awscli
echo "Installing uv..."
pip install uv

echo "Installing srtlog and awscli..."
cd /tmp
git clone --depth 1 https://github.com/ishandhanani/srtlog.git
uv pip install --system ./srtlog awscli

# Run srtlog to generate parquet
echo "Running srtlog parse..."
cd /logs
srtlog parse .

# Upload entire log directory to S3
echo "Uploading entire log directory to S3..."
aws s3 sync /logs {s3_url} {endpoint_flag}
echo "Upload complete: {s3_url}"

# Report what was uploaded
echo ""
echo "Uploaded files:"
find /logs -type f | wc -l
echo "files total"
"""

        # Build env for AWS credentials
        env: dict[str, str] = {}
        access_key = self._resolve_secret(s3_config.access_key_id, "AWS_ACCESS_KEY_ID")
        secret_key = self._resolve_secret(s3_config.secret_access_key, "AWS_SECRET_ACCESS_KEY")
        if access_key:
            env["AWS_ACCESS_KEY_ID"] = access_key
        if secret_key:
            env["AWS_SECRET_ACCESS_KEY"] = secret_key
        if s3_config.region:
            env["AWS_DEFAULT_REGION"] = s3_config.region

        try:
            logger.info("Running post-processing container (srtlog + S3 sync)...")
            proc = start_srun_process(
                command=["bash", "-c", script],
                nodelist=[self.runtime.nodes.head],
                output=str(self.runtime.log_dir / "postprocess.log"),
                container_image="python:3.11",
                container_mounts={str(self.runtime.log_dir): "/logs"},
                env_to_set=env,
            )
            proc.wait(timeout=600)  # 10 min timeout for install + parse + full sync

            parquet_path = self.runtime.log_dir / "cached_assets" / "node_metrics.parquet"

            if proc.returncode == 0:
                logger.info("Post-processing complete: %s", s3_url)
                return parquet_path if parquet_path.exists() else None, s3_url
            else:
                logger.warning("Post-processing failed (exit code: %s)", proc.returncode)
                return parquet_path if parquet_path.exists() else None, None

        except subprocess.TimeoutExpired:
            logger.warning("Post-processing container timed out")
            proc.kill()
            return None, None
        except Exception as e:
            logger.warning("Post-processing container failed: %s", e)
            return None, None

    def _report_metrics(self, benchmark_results: dict[str, Any] | None, s3_url: str | None, exit_code: int) -> None:
        """Report metrics to dashboard via status API.

        Args:
            benchmark_results: Extracted benchmark results
            s3_url: S3 URL where logs were uploaded
            exit_code: Exit code from the benchmark run
        """
        cluster_config = load_cluster_config()
        if not cluster_config:
            return

        reporting = cluster_config.get("reporting")
        if not reporting:
            return

        status_dict = reporting.get("status")
        if not status_dict or not status_dict.get("endpoint"):
            return

        endpoint = status_dict["endpoint"]

        payload: dict[str, Any] = {}
        if benchmark_results:
            payload["benchmark_results"] = benchmark_results
        if s3_url:
            payload["logs_url"] = s3_url

        if not payload:
            return

        # Use "failed" status when exit code is non-zero
        status = "failed" if exit_code != 0 else "completed"
        payload["status"] = status

        try:
            url = f"{endpoint}/api/jobs/{self.runtime.job_id}"
            response = requests.put(url, json=payload, timeout=5)
            if response.ok:
                logger.info("Reported metrics to dashboard: %s", url)
            else:
                logger.warning("Dashboard metrics report failed: %s %s", response.status_code, response.text)
        except Exception as e:
            logger.warning("Dashboard metrics report failed: %s", e)

    def _run_ai_analysis(self, config: AIAnalysisConfig) -> None:
        """Run AI analysis using Claude Code CLI via OpenRouter.

        Uses OpenRouter for authentication which works well in headless environments.
        Installs claude CLI and gh CLI in a python container before running analysis.
        See: https://openrouter.ai/docs/guides/claude-code-integration

        Args:
            config: AI analysis configuration
        """
        # Resolve secrets
        openrouter_key = self._resolve_secret(config.openrouter_api_key, "OPENROUTER_API_KEY")
        gh_token = self._resolve_secret(config.gh_token, "GH_TOKEN")

        if not openrouter_key:
            logger.error("AI analysis requires OPENROUTER_API_KEY (set in srtslurm.yaml or environment)")
            return

        if not gh_token:
            logger.warning("GH_TOKEN not set - GitHub PR search will not work")

        # Build the prompt - escape for shell
        log_dir = str(self.runtime.log_dir)
        prompt = config.get_prompt(log_dir)
        escaped_prompt = shlex.quote(prompt)

        logger.info("Log directory: %s", log_dir)
        logger.info("Repos to search: %s", ", ".join(config.repos_to_search))

        # Build environment variables for OpenRouter integration
        # See: https://openrouter.ai/docs/guides/claude-code-integration
        env_to_set = {
            "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
            "ANTHROPIC_AUTH_TOKEN": openrouter_key,
            "ANTHROPIC_API_KEY": "",  # Must be explicitly empty to route through OpenRouter
        }
        if gh_token:
            env_to_set["GH_TOKEN"] = gh_token

        # Build the analysis script that installs tools and runs claude
        # Uses curl to install claude CLI and gh CLI without requiring apt/root
        script = f"""
set -e

echo "Installing uv..."
pip install uv

echo "Installing Claude Code CLI..."
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

echo "Installing GitHub CLI..."
GH_VERSION=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | grep '"tag_name"' | cut -d'"' -f4 | sed 's/v//')
curl -fsSL "https://github.com/cli/cli/releases/download/v${{GH_VERSION}}/gh_${{GH_VERSION}}_linux_amd64.tar.gz" | tar xz -C /tmp
export PATH="/tmp/gh_${{GH_VERSION}}_linux_amd64/bin:$PATH"

echo "Dependencies installed. Running AI analysis..."

# Run claude with explicit tool permissions
cd /logs
claude -p {escaped_prompt} \\
    --allowedTools "Read,Bash(gh *),Bash(ls *),Bash(cat *),Bash(grep *),Write(**/ai_analysis.md)"

echo "AI analysis complete."
"""

        analysis_log = self.runtime.log_dir / "ai_analysis.log"
        logger.info("Starting Claude Code analysis (log: %s)", analysis_log)

        try:
            proc = start_srun_process(
                command=["bash", "-c", script],
                nodelist=[self.runtime.nodes.head],
                output=str(analysis_log),
                container_image="python:3.11",
                container_mounts={str(self.runtime.log_dir): "/logs"},
                env_to_set=env_to_set,
            )

            # Wait for completion with timeout (15 minutes for install + analysis)
            timeout = 900
            start_time = time.time()

            while proc.poll() is None:
                if time.time() - start_time > timeout:
                    logger.warning("AI analysis timed out after %d seconds", timeout)
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return
                time.sleep(5)

            exit_code = proc.returncode or 0

            if exit_code != 0:
                logger.warning("AI analysis exited with code %d", exit_code)
            else:
                logger.info("AI analysis completed successfully")

            # Check if analysis file was created
            analysis_file = self.runtime.log_dir / "ai_analysis.md"
            if analysis_file.exists():
                logger.info("Analysis report written to: %s", analysis_file)
            else:
                logger.warning("AI analysis did not produce ai_analysis.md")

        except Exception as e:
            logger.error("Failed to run AI analysis: %s", e)
