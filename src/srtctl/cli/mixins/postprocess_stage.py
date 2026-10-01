# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Artifact finalization and upload for SweepOrchestrator.

Handles:
- Run configuration and reproducibility files
- S3 upload of the whole log directory
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.config import load_cluster_config
from srtctl.core.git_state import GIT_STATE_FILENAME
from srtctl.core.lockfile import collect_worker_fingerprints, generate_reproduction_report, write_lockfile
from srtctl.core.schema import DEFAULT_S3_ARCHIVE, DEFAULT_S3_EXCLUDE, S3Config
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.status import StatusReporter

logger = logging.getLogger(__name__)

POSTPROCESS_UPLOAD_FAILED_EXIT = 11

# Runs inside the upload container (plain python:3.11, srtctl is not installed there), so it
# is stdlib plus an optional ``zstandard``. argv: <root> <out_dir> <json list of glob patterns>.
# Packs every matching file under root into one archive, arcnames relative to root, and prints
# the archive path as the last stdout line (nothing when no file matched). zstd level 3 turned
# a 253 MB log directory into 4 MB; xz is the fallback when the zstandard wheel is unavailable.
ARCHIVE_SCRIPT = r"""
import glob, json, os, sys, tarfile
root, out_dir, patterns = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
files = sorted({p for pat in patterns for p in glob.glob(os.path.join(root, pat), recursive=True) if os.path.isfile(p)})
if not files:
    print("archive: no file matched " + ", ".join(patterns), file=sys.stderr)
    sys.exit(0)
try:
    import zstandard
    out = os.path.join(out_dir, "bundle.tar.zst")
    with open(out, "wb") as fh, zstandard.ZstdCompressor(level=3, threads=-1).stream_writer(fh) as zst, tarfile.open(fileobj=zst, mode="w|") as tar:
        for f in files:
            tar.add(f, arcname=os.path.relpath(f, root))
except ImportError:
    out = os.path.join(out_dir, "bundle.tar.xz")
    with tarfile.open(out, "w:xz") as tar:
        for f in files:
            tar.add(f, arcname=os.path.relpath(f, root))
raw = sum(os.path.getsize(f) for f in files)
print("archive: %d files, %.1f MB raw -> %.1f MB %s" % (len(files), raw / 1048576, os.path.getsize(out) / 1048576, os.path.basename(out)), file=sys.stderr)
print(out)
"""


def s3_sync_exclude_pattern(archive_pattern: str) -> str:
    """Translate a Python glob used for the archive into the AWS CLI exclude that covers it.

    AWS ``--exclude`` has no ``**`` but its ``*`` already matches across directories, so
    collapsing ``**`` to ``*`` yields a pattern at least as broad as the glob.
    """
    return archive_pattern.replace("**", "*")


class PostProcessStageMixin:
    """Finalize reproducibility files and upload captured artifacts after a run.

    Upload configuration is loaded from srtslurm.yaml (cluster config).

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

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
        except Exception as e:  # noqa: BLE001
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

        Override/zip submissions also write a resolved runtime config next to the
        source as config_{suffix}.yaml (or config_resolved.yaml). Glob all
        config*.yaml files so the actually-executed resolved config is uploaded
        too, not just the unresolved source config.yaml.
        """
        output_dir = self.runtime.log_dir.parent
        config_files = sorted(p.name for p in output_dir.glob("config*.yaml"))
        files_to_copy = [*config_files, "sbatch_script.sh", f"{self.runtime.job_id}.json", GIT_STATE_FILENAME]
        for name in files_to_copy:
            src = output_dir / name
            if not src.exists():
                continue
            dst = self.runtime.log_dir / name
            try:
                shutil.copy2(src, dst)
                logger.info("Copied %s to log directory", name)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to copy %s to log directory: %s", name, e)

    def run_postprocess(self, exit_code: int, reporter: "StatusReporter | None" = None) -> None:
        """Finalize and upload artifacts after benchmark completion.

        Handles:
        1. Copy config YAML into log directory (for S3 upload)
        2. Write the final lockfile and compare a lockfile re-run
        3. S3 upload of the whole log directory (if S3 configured)
        4. Eager push of ``logs_url`` to the status API right after the S3 sync
           completes, so downstream consumers can fetch results from S3 even
           before final completion reporting.
        5. Stash ``logs_url`` on self so the caller's final
           ``report_completed`` PUT in do_sweep can reassert the pointer.

        Analysis and dashboard generation are separate, explicit operations.

        Benchmark results themselves are NOT pushed to the status API — S3 is
        the source of truth for artifacts. The collector only stores pointers.

        Args:
            exit_code: Benchmark exit code, retained for caller compatibility.
            reporter: Optional StatusReporter for eager mid-run pushes. When
                provided, ``logs_url`` is PUT as soon as it's known (step 4);
                when None, only the stash path is used.
        """
        # Copy config into log directory so it's included in S3 upload
        self._copy_config_to_logs()

        # Write lockfile with verification
        # TODO: include benchmark results once rollup format is standardized across
        # sa-bench, trace-replay, and mooncake-router (currently only sa-bench has
        # a structured rollup with runs[].throughput_toks etc.)
        verification = getattr(self, "_identity_verification", None)
        write_lockfile(
            self.runtime.log_dir.parent,
            self.config,
            self.runtime.log_dir,
            verification=verification,
        )

        # Compare against previous lockfile if this was a lockfile re-run
        self._compare_against_previous_lock()

        # Upload the log directory to S3 (if configured)
        s3_url = self._run_postprocess_container()

        # Publish the artifact pointer as soon as the upload completes.
        if reporter is not None and s3_url:
            reporter.report_artifacts(logs_url=s3_url)

        # Stash so the final StatusReporter.report_completed PUT (in do_sweep)
        # reasserts logs_url idempotently across every configured endpoint.
        self._last_logs_url = s3_url

    def _compare_against_previous_lock(self) -> None:
        """If this run was from a lockfile, compare against previous run."""
        try:
            lock_data = getattr(self.config, "_lock_data", None)
            if not lock_data:
                return

            new_fps = collect_worker_fingerprints(self.runtime.log_dir)
            if not new_fps:
                return

            # TODO: pass benchmark results once rollup format is standardized
            summary_lines, report_lines, _issues = generate_reproduction_report(
                lock_data,
                new_fps,
            )

            # Log summary to sweep log
            if summary_lines:
                logger.info("")
                logger.info("=" * 60)
                logger.info("Comparison against previous lockfile run")
                logger.info("=" * 60)
                for line in summary_lines:
                    logger.info(line)
                logger.info("=" * 60)

            # Write full report to file
            if report_lines:
                report_path = self.runtime.log_dir / "reproduction-report.txt"
                report_path.write_text("\n".join(report_lines) + "\n")
                logger.info(f"Reproduction report: {report_path}")

        except Exception as e:  # noqa: BLE001
            logger.debug("Lockfile comparison skipped: %s", e)

    def _run_postprocess_container(self) -> str | None:
        """Upload the log directory to S3 from a small container on the head node.

        Ships the run identity (config, lockfile, job JSON, sbatch script, git
        state), every orchestrator, worker, frontend and service log, the
        benchmark results and the tachometer parquet as
        loose objects, plus one compressed archive of the patterns in
        ``reporting.s3.archive``; the patterns in ``reporting.s3.exclude`` are
        skipped (see ``DEFAULT_S3_EXCLUDE`` for why). Returns the S3 URL of the
        log directory, or None when S3 is not configured or the upload failed.
        """
        s3_config = self._get_s3_config()
        if not s3_config:
            logger.debug("S3 not configured, skipping upload")
            return None

        # S3 path: {prefix}/{YYYY-MM-DD}/{job_id}/
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        s3_prefix = f"{s3_config.prefix or 'srtslurm'}/{date_str}/{self.runtime.job_id}"
        s3_url = f"s3://{s3_config.bucket}/{s3_prefix}/"

        # Build endpoint flag if custom endpoint provided
        endpoint_flag = f"--endpoint-url {s3_config.endpoint_url}" if s3_config.endpoint_url else ""

        exclude = list(DEFAULT_S3_EXCLUDE if s3_config.exclude is None else s3_config.exclude)
        archive = list(DEFAULT_S3_ARCHIVE if s3_config.archive is None else s3_config.archive)
        logger.info(
            "S3 upload policy: %d exclude pattern(s), archive of %s",
            len(exclude),
            ", ".join(archive) if archive else "nothing",
        )
        script = self._build_postprocess_script(s3_url, endpoint_flag, exclude=exclude, archive=archive)

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
            logger.info("Uploading the log directory to %s...", s3_url)
            proc = start_srun_process(
                command=["bash", "-c", script],
                nodelist=[self.runtime.nodes.head],
                output=str(self.runtime.log_dir / "postprocess.log"),
                container_image="python:3.11",
                container_mounts={self.runtime.log_dir: Path("/logs")},
                env_to_set=env,
                het_group=self.runtime.nodes.het_group_for(self.runtime.nodes.head),
            )
            proc.wait(timeout=600)  # 10 min for the awscli install plus a full sync

            if proc.returncode == 0:
                logger.info("Upload complete: %s", s3_url)
                return s3_url
            logger.warning("S3 upload failed (exit code: %s)", proc.returncode)
            return None

        except subprocess.TimeoutExpired:
            logger.warning("S3 upload container timed out")
            proc.kill()
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("S3 upload container failed: %s", e)
            return None

    def _build_postprocess_script(
        self,
        s3_url: str,
        endpoint_flag: str,
        *,
        exclude: list[str] | None = None,
        archive: list[str] | None = None,
    ) -> str:
        """Bash for the upload container.

        Installs awscli (and zstandard, best effort), records the destination and
        policy in ``postprocess-status.json``, packs the ``archive`` patterns into
        one ``bundle.tar.zst`` under ``/tmp`` (the log directory on the cluster is
        left untouched), syncs ``/logs`` minus ``exclude`` and minus the archived
        files, then uploads the archive next to them.
        """
        exclude = list(DEFAULT_S3_EXCLUDE if exclude is None else exclude)
        archive = list(DEFAULT_S3_ARCHIVE if archive is None else archive)
        sync_excludes = exclude + [s3_sync_exclude_pattern(p) for p in archive]
        exclude_flags = " ".join(f"--exclude {shlex.quote(p)}" for p in sync_excludes)
        status_json = json.dumps({"s3_url": s3_url, "exclude": exclude, "archive": archive})

        archive_step = ""
        if archive:
            archive_step = f"""
echo "Packing {len(archive)} archive pattern(s) into one compressed bundle..."
archive_path=$(python3 - /logs /tmp {shlex.quote(json.dumps(archive))} <<'PY'
{ARCHIVE_SCRIPT}
PY
)
"""

        return f"""
set -u
set -o pipefail

echo "Installing awscli..."
if ! pip install awscli; then
  echo "Failed to install awscli"
  exit {POSTPROCESS_UPLOAD_FAILED_EXIT}
fi
pip install zstandard || echo "zstandard unavailable; the archive falls back to .tar.xz"

cat > /logs/postprocess-status.json <<'EOF'
{status_json}
EOF
archive_path=""
{archive_step}
echo "Uploading the log directory to S3 ({len(sync_excludes)} exclude pattern(s))..."
if ! aws s3 sync /logs {s3_url} {endpoint_flag} {exclude_flags}; then
  echo "Upload failed"
  exit {POSTPROCESS_UPLOAD_FAILED_EXIT}
fi
if [ -n "$archive_path" ] && [ -s "$archive_path" ]; then
  if ! aws s3 cp "$archive_path" {s3_url}$(basename "$archive_path") {endpoint_flag}; then
    echo "Archive upload failed"
    exit {POSTPROCESS_UPLOAD_FAILED_EXIT}
  fi
fi

echo "Upload complete: {s3_url}"
echo ""
echo "Uploaded objects:"
aws s3 ls --recursive {s3_url} {endpoint_flag} | wc -l
echo "objects total"
"""
