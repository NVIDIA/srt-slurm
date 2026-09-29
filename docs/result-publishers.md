# Result publishers

Use an installed command to publish a completed run to a result service or enqueue
it for a separate worker. The default is an empty list: no commands run.

```yaml
reporting:
  publishers:
    - name: team-results
      command: ["benchmark-results-publish"]
      timeout_seconds: 60
```

Set this in a recipe or as a cluster reporting default in `srtslurm.yaml`.
Recipe `reporting` replaces the cluster reporting block, as with existing status
reporting. Install the command independently on the **orchestrator host**, with
access to the run directory; it does not run inside a benchmark container.
No package is installed or downloaded by this setting. An absolute executable
path can select an isolated environment. Commands are argv arrays, without shell
expansion. Only configure trusted executables.

Keep credentials and service settings in the publisher's environment or private
configuration file. Recipe fields and command arguments can appear in saved run
artifacts, so they must not contain secrets. Each name must be unique and contain
only letters, digits, underscores and hyphens (at most 64 characters, starting
with a letter or digit).

## Protocol version 1

The command reads one JSON object from stdin (then EOF):

```json
{
  "protocol_version": 1,
  "run_dir": "/runs/42",
  "log_dir": "/runs/42/logs",
  "job_id": "42",
  "benchmark_type": "custom",
  "run_exit_code": 0
}
```

Paths are absolute host paths; the command's working directory is `run_dir`.
The job ID alone is not globally unique. Publishers own run identity,
client-specific validation, artifact selection and deduplication. The exit code
is the run's outcome after telemetry finalization, not a certificate of complete
benchmark artifacts. Publishers are invoked for failed runs too, and should
return `skipped` when a run is ineligible. Serve-only and evaluation-only runs
also need this validation.

On exit 0, stdout must contain one JSON object, at most 64 KiB:

```json
{
  "protocol_version": 1,
  "status": "accepted",
  "links": {"tracking": "https://results.example.org/jobs/42"}
}
```

`status` is `accepted` (queued/uploaded, processing unverified), `published`
(the publisher verified availability), or `skipped`. `links` is optional and
maps display names to HTTP(S) URLs without embedded credentials. Never return
presigned URLs, tokens or other credentials. Additional response fields are
ignored. Diagnostic stdout is invalid; stderr is discarded. Publishers should
keep their own diagnostic logs and durable receipts outside the run archive.

## Lifecycle and recovery

The hook runs after local rollup, dashboard and energy-report generation and
before S3 export, so S3 can include `logs/publishers/<name>.json`. It reads the
original local artifacts independently of S3's exclusions and compression.
The collector's existing `logs_url` behavior is unchanged; publication links are
in these receipts, not new collector API fields.

Commands run sequentially, each bounded by `timeout_seconds` (1–3600). They add
that time to postprocessing while the Slurm allocation is still held. For slow
uploads, use a command that durably enqueues the work for a CPU-side worker;
preserve the source artifacts until the worker has copied them. The hook does
not implement a worker or queue.

Missing executables, nonzero exits, invalid output and timeouts produce warnings
and a receipt with `state: unknown`; they never change the benchmark's exit
code. A timed-out process group is killed. The service may already have accepted
the submission, so the hook does **not** retry. A later invocation can call the
publisher again: safe deduplication belongs to the publisher, not this receipt.
The receipt records only the validated response or exception type, never raw
command output. Review the publisher's receipt before retrying an ambiguous
failure. Abrupt Slurm termination may prevent the hook from running at all;
publishers should also support recovery from an existing run directory.
