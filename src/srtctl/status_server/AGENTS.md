# Status reporting

Covers the reporter in `core/status.py` and the collector in this package. `docs/monitoring.md` is the user-facing reference.

## Status Reporting

Optional fire-and-forget HTTP status reporting to one or more collectors. Configure in `srtslurm.yaml`:

```yaml
# Cluster-level config (srtslurm.yaml)
cluster: "bruh"  # Cluster name for dashboard display
reporting:
  status:
    endpoint: "http://login-node:8080"
```

**srtctl status-server** is the in-repo collector for that endpoint (`src/srtctl/status_server/`: `store.py` is the SQLite side, `server.py` the stdlib HTTP side validating with `srtctl.contract`, `ui/index.html` the dependency-free single-page UI served at `/`, which fetches the same `/api` routes with the read token from `localStorage`). It appends an event whenever `(status, stage, message)` changes, creates a placeholder row for a PUT whose POST never arrived, and serves cursor-based feeds at `/api/events` and `/api/jobs/{id}/events`. `make_server(store, port=0)` gives tests a real server on an ephemeral port (`tests/test_status_server.py` drives it with the real `StatusReporter`). The wire contract is `docs/status-api-spec.md`; a payload field changes in `srtctl.contract`, the server, and the spec together.

**StatusReporter** - Used in `do_sweep.py` to report job lifecycle:

```python
from srtctl.core.status import StatusReporter, JobStatus, JobStage

reporter = StatusReporter.from_config(config.reporting, job_id)
reporter.report_started(config, runtime)  # Job started, with model/resources/head_node/log_dir metadata
reporter.report(JobStatus.WORKERS, JobStage.WORKERS, "Starting workers")
reporter.report_completed(exit_code, logs_url=s3_url)  # Final status
```

**Status lifecycle** (status is the stage being entered, not readiness):
```
submitted → starting → workers → frontend → benchmark → completed | failed
stages: starting, head_infrastructure, preflight, workers, frontend, benchmark, cleanup
```

**create_job_record()** - Standalone function for job submission:

```python
from srtctl.core.status import create_job_record

# Called in submit.py after sbatch succeeds
create_job_record(
    reporting=config.reporting,
    job_id=job_id,
    job_name=config.name,
    cluster=get_srtslurm_setting("cluster"),
    recipe=str(config_path),
    metadata=metadata,  # Tags go in metadata["tags"]
)
```

**Key behaviors:**
- All HTTP requests have 5-second timeout
- Failures are logged at DEBUG and silently ignored
- Job execution is never blocked by status reporting
- Tags are passed via `metadata["tags"]` (not a separate field)
- `metadata["log_dir"]` (from `report_started`) is the run's log directory on the cluster filesystem; `logs_url` is only set when `reporting.s3` uploads it
- `report_started` also repeats `job_name` and `cluster` in `metadata`: the submit-time POST is one attempt (now two) from the login node, and when it is lost the collector's placeholder row (`job-<id>`) takes its identity from the started report; a late POST fills whatever is still null, only ever moves `submitted_at` earlier, and never rewinds status; every reporter request gets two attempts
- `reporting.s3` uploads follow a policy (`DEFAULT_S3_EXCLUDE` / `DEFAULT_S3_ARCHIVE` in `core/schema.py`): aiperf's per-interval metrics scrapes, `perf_dashboard_bundle/` and `perf_dashboard.json` are skipped (tachometer parquet holds the same series; a 2 GB run becomes about 60 MB), aiperf's per-request `profile_export.jsonl` is packed into `bundle.tar.zst` by the inline `ARCHIVE_SCRIPT` in `postprocess_stage.py`, which runs in the plain `python:3.11` upload container (stdlib + optional `zstandard`, xz fallback). Change the policy in the schema constants and `docs/cluster-config.md` together
- Auth is a bearer token read from `$SRTCTL_STATUS_TOKEN` on both sides (`reporting.status.token_env` renames the variable). Never add a literal token field: `SrtConfig.Schema().dump` lands in the lockfile and resolved configs are copied into `logs/` and synced to S3. The reporter never follows redirects and warns on 3xx/401/403; the server refuses to listen beyond loopback without a token unless `--allow-unauthenticated`
