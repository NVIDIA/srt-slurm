# Cluster Config (`srtslurm.yaml`)

How srtctl finds `srtslurm.yaml`, what its defaults and aliases do to a recipe, and how to run without one. Every key with its type and default: [Cluster config](schema-reference.md#cluster-config) (generated).

## Cluster Config Discovery

srtctl looks for `srtslurm.yaml` (cluster-wide settings) in this order:

1. **`SRTSLURM_CONFIG` environment variable** (if set) - explicit path to config file
2. Current working directory
3. Parent directory (1 level up)
4. Grandparent directory (2 levels up)

For users working in deep directory structures (e.g., study directories), set `SRTSLURM_CONFIG` in your shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
export SRTSLURM_CONFIG="/path/to/srt-slurm/srtslurm.yaml"
```

This allows you to run `srtctl apply -f config.yaml` from anywhere without needing `srtslurm.yaml` nearby.

### Cluster Config Fields

Every field, with its type, default, allowed values, and description: [Cluster config](schema-reference.md#cluster-config) (generated from the code; `srtctl schema --cluster` prints it as JSON Schema). The notes below cover the keys whose behavior needs more than a line.

**reporting.s3**: After a run, a small container on the head node uploads the log directory to `s3://<bucket>/<prefix>/<YYYY-MM-DD>/<job_id>/` (`endpoint_url` for MinIO or another S3-compatible store; credentials only through `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in the submit shell, since the literal fields would land in the lockfile). Not everything is worth shipping: a benchmark run's directory is 250 MB to 2 GB on lustre, over 95% of it aiperf's per-interval scrape of the worker and DCGM `/metrics` endpoints, the same series tachometer already stores as parquet. The upload therefore follows a policy:

The resolved recipe's `reporting.s3` takes precedence over the cluster setting. A recipe-supplied S3 block replaces the cluster S3 block in full, including `exclude` and `archive`; omitted fields use the S3 defaults rather than inheriting individual cluster values. If the resolved recipe has no S3 block, the uploader falls back to `srtslurm.yaml`. See [the recipe example](../examples/features/s3-reporting.yaml).

| Shipped as-is | Packed into `bundle.tar.zst` (`archive`) | Skipped (`exclude`) |
|---|---|---|
| config, lockfile, job JSON, sbatch script, git state, fingerprints, resource snapshot; sweep, worker, frontend, service and benchmark logs; results JSON, rollup, `profile_export_aiperf.*`; `perf_dashboard.html` (if present); `tachometer/` parquet | `artifacts/**/profile_export.jsonl`, `sa-bench_*/**/profile_export.jsonl` (aiperf's per-request records, 13 to 40 MB raw, under 1 MB compressed) | `server_metrics_export.jsonl`, `server_metrics_export.json`, `gpu_telemetry_export.jsonl` and `inputs.json` under `artifacts/*/` and `sa-bench_*/*/` (the aiperf artifact roots; a same-named file from another benchmark type is not touched), `perf_dashboard_bundle/*`, `perf_dashboard.json` |

`exclude` uses `aws s3 sync` pattern rules (relative to the log directory, `*` matches across directories); `archive` uses Python glob rules with `**`. Either list replaces its default when set; `exclude: []` ships the whole directory, `archive: []` makes no archive. The archive is built under `/tmp` in the container, so nothing is added to the log directory on the cluster. One caveat: with tachometer disabled, dropping the aiperf scrape leaves no uploaded engine-metrics time series for a later dashboard build; enable tachometer, or override `exclude`.

```yaml
reporting:
  s3:
    bucket: "srt-logs"
    prefix: "sa-b200"
    endpoint_url: "https://minio.example.com"   # omit for AWS
    # exclude: []                                # ship everything
    # archive: ["artifacts/**/profile_export.jsonl", "*.out"]   # also pack the worker logs
```

**output_dir**: When set, job logs are written to `output_dir/{job_id}/logs` instead of `srtctl_root/outputs/{job_id}/logs`. Useful for CI/CD and ephemeral environments.

**containers**: A map from alias to image path or registry URI. One resolver replaces image aliases in `model.container`, `roles.<role>.container`, frontend/benchmark images, exporter images and `services[].container`. Literal paths and registry URIs pass through untouched. Free-form maps (`environment`, `roles.<role>.env`, `roles.<role>.args`, `services[].env`, `container_mounts`) and the `identity` block are never rewritten.

**default_bash_preamble**: A shell snippet (e.g. `"ulimit -n 1048576 -s unlimited -u 1048576"`) prepended to every container srun launched by srtctl: workers, frontends, telemetry, benchmark, postprocess. Runs before per-call `bash_preamble` and the main command, so cluster-wide ulimits apply to everything downstream. Silently dropped for distroless containers (e.g. `prom/node-exporter`) that bypass the bash wrapper; a WARNING log is emitted in that case.

**default_host_setup**: A [`host_setup`](runtime-env.md#host_setup) block applied to every job on the cluster, for node state that has to be set outside the container, such as locking GPU clocks. A recipe that sets its own `host_setup:` block replaces it entirely; `host_setup: {commands: []}` opts a single run out.

**preflight**: `srtctl apply` normally stats `model.path`, `model.container` and the telemetry images on the submitting node before calling sbatch. On clusters where those live only on compute nodes (node-local NVMe such as `/raid/models`), that check can never pass from the login node; set `preflight: false` and every `apply` behaves as if `--no-preflight` had been passed, with an INFO line saying so. Paths are still resolved at runtime and the framework fails loudly on the compute node if one is genuinely missing.

**nginx_raise_ulimit**: When set to `true` or `false`, this value is applied to jobs that omit `frontend.nginx_raise_ulimit` in the recipe. Use `true` on clusters where raising the nginx container's open-file limit is allowed; leave unset if each job should rely on the frontend default (`false`). A recipe that sets `frontend.nginx_raise_ulimit` always wins.

### Running without `srtslurm.yaml`

`srtslurm.yaml` is optional. A recipe can be fully self-sustaining as long as it supplies everything the cluster yaml would otherwise provide:

- Set `slurm.account`, `slurm.partition`, and `slurm.time_limit` directly in the recipe (no `default_*` fallback).
- Use absolute paths for `model.path`, `model.container`, and any other container fields; alias resolution is a no-op without the yaml's `containers:` / `model_paths:` maps.
- List every cluster-side mount the job needs in `extra_mount` (e.g. the lustre share that holds your model weights and `.sqsh` files). `default_mounts` is the only `srtslurm.yaml` field with no recipe-level equivalent until you spell mounts out yourself.
- Set `resources.gpus_per_node` explicitly.
- Supply `reporting.status` in the recipe for status updates and `reporting.s3` for S3 log upload; either can be configured without a cluster yaml. Without either recipe setting or cluster defaults, that reporting is skipped.

Workers' nats and etcd come from the dynamo/sglang container, not the yaml, so disagg/agg topologies still work end-to-end. `srtctl_root` falls back to the package install path automatically.

This is useful for portable recipes that you want to share across clusters or hand to a teammate without dragging cluster config along.
