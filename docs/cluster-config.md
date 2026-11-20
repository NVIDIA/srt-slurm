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

| Shipped as-is | Packed into `bundle.tar.zst` (`archive`) | Skipped (`exclude`) |
|---|---|---|
| config, lockfile, job JSON, sbatch script, git state, fingerprints, resource snapshot; sweep, worker, frontend, service and benchmark logs; results JSON, rollup, `profile_export_aiperf.*`; `tachometer/` parquet | `artifacts/**/profile_export.jsonl`, `sa-bench_*/**/profile_export.jsonl` (aiperf's per-request records, 13 to 40 MB raw, under 1 MB compressed) | `server_metrics_export.jsonl`, `server_metrics_export.json`, `gpu_telemetry_export.jsonl` and `inputs.json` under `artifacts/*/` and `sa-bench_*/*/` (the aiperf artifact roots; a same-named file from another benchmark type is not touched) |

`exclude` uses `aws s3 sync` pattern rules (relative to the log directory, `*` matches across directories); `archive` uses Python glob rules with `**`. Either list replaces its default when set; `exclude: []` ships the whole directory, `archive: []` makes no archive. The archive is built under `/tmp` in the container, so nothing is added to the log directory on the cluster. One caveat: with tachometer disabled, dropping the aiperf scrape leaves no uploaded engine-metrics time series for later analysis; enable tachometer, or override `exclude`.

```yaml
reporting:
  s3:
    bucket: "srt-logs"
    prefix: "sa-b200"
    endpoint_url: "https://minio.example.com"   # omit for AWS
    # exclude: []                                # ship everything
    # archive: ["artifacts/**/profile_export.jsonl", "*.out"]   # also pack the worker logs
```

**release_nodes_before_postprocess**: Off by default. When `true`, post-processing first writes the final lockfile, the lockfile comparison and the energy report (which record the allocation the run used), then shrinks the job to the node running the batch script with `scontrol update JobId=<job_id> NodeList=<batch host>`, so every other node goes back to the scheduler before the S3 upload. By then every worker, frontend and service step has stopped and host teardown has run. The S3 upload and the AI failure analysis then run on that batch host instead of the head node. srtctl checks the result with `squeue -j <job_id>`: a job on the batch host alone is a verified release; a job still on its old nodes is a failure even when scontrol exited 0 (slurmctld accepts an update to the job's submitted `--nodelist` as a no-op); when squeue cannot answer, scontrol's exit code stands and the release is recorded as unverified. If scontrol times out, only a squeue showing the batch host alone counts as a release. After a release srtctl applies the resize script scontrol writes (`SLURM_NODELIST`, `SLURM_JOB_NODELIST`, `SLURM_NNODES`, `SLURM_JOB_NUM_NODES`, `SLURM_JOB_CPUS_PER_NODE`, and unsets `SLURM_NPROCS` / `SLURM_NTASKS` / `SLURM_TASKS_PER_NODE`) to its own environment so later sruns target the one-node allocation; the script is parsed, never executed, and other names in it are ignored. Slurm refuses the shrink when the cluster sets `SchedulerParameters=disable_job_shrink` or the job uses arbitrary task distribution; the refusal is logged as a WARNING and the upload runs on the head node of the full allocation as before. Heterogeneous jobs (`use_het_jobs`, `resources.het_jobs`) are skipped, and so is any job shutting down on SIGTERM or SIGINT: at the time limit or after `scancel` Slurm refuses to resize a job that is already ending, so srtctl does not try for any signal-driven shutdown. A job that stops on a failed worker still releases. A signal that arrives while `scontrol update` is running ends the job as before, and `allocation-release.json` records the interrupted attempt. Every attempt or skip is recorded in `allocation-release.json` in the log directory (`requested`, `succeeded`, `retained`, `released`, `verified`, `reason`, `duration_s`, `job_id`, `timestamp`), which ships with the uploaded logs. `srtctl dry-run` prints a "Node release" line when the setting is on.

**output_dir**: When set, job logs are written to `output_dir/{job_id}/logs` instead of `srtctl_root/outputs/{job_id}/logs`. Useful for CI/CD and ephemeral environments.

**containers**: A map from alias to image path or registry URI. One resolver replaces image aliases in `model.container`, `roles.<role>.container`, frontend/benchmark images, exporter images and `services[].container`. Literal paths and registry URIs pass through untouched. Free-form maps (`environment`, `roles.<role>.env`, `roles.<role>.args`, `services[].env`, `container_mounts`) and the `identity` block are never rewritten.

**default_bash_preamble**: A shell snippet (e.g. `"ulimit -n 1048576 -s unlimited -u 1048576"`) prepended to every container srun launched by srtctl: workers, frontends, telemetry, benchmark, postprocess. Runs before per-call `bash_preamble` and the main command, so cluster-wide ulimits apply to everything downstream. Silently dropped for distroless containers (e.g. `prom/node-exporter`) that bypass the bash wrapper; a WARNING log is emitted in that case.

**default_host_setup**: A [`host_setup`](runtime-env.md#host_setup) block applied to every job on the cluster, for node state that has to be set outside the container, such as locking GPU clocks. A recipe that sets its own `host_setup:` block replaces it entirely; `host_setup: {commands: []}` opts a single run out.

**preflight**: `srtctl apply` normally stats `model.path`, `model.container` and the telemetry images on the submitting node before calling sbatch. On clusters where those live only on compute nodes (node-local NVMe such as `/raid/models`), that check can never pass from the login node; set `preflight: false` and every `apply` behaves as if `--no-preflight` had been passed, with an INFO line saying so. Paths are still resolved at runtime and the framework fails loudly on the compute node if one is genuinely missing. Under `launcher: host`, which runs no container images, preflight still checks `model.path`, the topology and that `model.container` is set, but does not look for the `model.container` or exporter image files, and the job does not require the image file to exist, so a recipe whose image exists only on a cluster needs neither `preflight: false` nor `--no-preflight`.

**nginx_raise_ulimit**: When set to `true` or `false`, this value is applied to jobs that omit `frontend.nginx_raise_ulimit` in the recipe. Use `true` on clusters where raising the nginx container's open-file limit is allowed; leave unset if each job should rely on the frontend default (`false`). A recipe that sets `frontend.nginx_raise_ulimit` always wins.

### Launcher

`launcher` picks where a job's processes run. `slurm` (the default) runs each process as an `srun` step inside the allocation `sbatch` hands out. `docker` runs the whole job on the machine you type `srtctl apply` on, for a single GPU VM without Slurm. `host` also runs the whole job on that machine, but starts every process straight on the host without containers, for a machine with neither Slurm nor a container runtime (see [`launcher: host`](#launcher-host)). `docker` needs Docker with the NVIDIA Container Toolkit (`docker run --gpus all` must work) and the binaries `make setup ARCH=x86_64` (or `aarch64`) installs:

```yaml
launcher: docker
# docker_args: ["--user", "1000:1000"]   # extra `docker run` arguments for every container
# docker_gpus: "device=2,3,4,5"          # `--gpus` value for every container (default: all)
```

Under `launcher: docker`, `srtctl apply` renders the same job script, stages `outputs/<job_id>/` the same way, and runs the script in the foreground with `SRTCTL_JOB_ID=docker-<timestamp>` in place of `SLURM_JOB_ID`; its output is teed to `logs/sweep_<job_id>.log`. The orchestrator, health checks, benchmarks and postprocessing are unchanged; only the launcher underneath differs:

| | `slurm` | `docker` |
|---|---|---|
| Container launch | `srun --container-image ...` (pyxis/enroot) | `docker run --rm --gpus <docker_gpus or all> --network host --ipc host --cap-add PERFMON --cap-add SYS_NICE -v <mounts> ...` |
| Container working directory | `srun_options: container-workdir` | the same option, as `docker run -w` (other `srun_options` are ignored, unknown ones with a warning) |
| Host command (`host_setup`, no container) | `srun` on each node | `bash` on this machine, in its own process group |
| GPU subset per worker | `CUDA_VISIBLE_DEVICES` from the bash wrapper | same |
| Graceful stop of a named step | `scancel --signal=TERM --full <job>.<step>` | `docker kill --signal=TERM srtctl_<job>_<step>` (or `killpg` for a host command) |
| Node and address | the allocation's nodelist, IP on `network_interface` | this host, `127.0.0.1` |

Every container sees the GPUs `docker_gpus` names (all of them by default) and shares the host network, so the per-worker GPU masks and the ports `NodePortAllocator` already hands out to colocated workers keep processes apart. On a shared or partly faulty machine, `docker_gpus: "device=2,3,4,5"` restricts every container to those devices; docker renumbers them from 0, so a worker's `CUDA_VISIBLE_DEVICES=0,1,2,3` then means physical GPUs 2-5. Containers get `--cap-add SYS_NICE` so engines can pin each rank to its GPU's NUMA node as they do in an srun step (SGLang otherwise logs "User lacks permission to set NUMA affinity" and runs unpinned), and `--cap-add PERFMON` so nsys CPU sampling (`observability.nsys.cpu_sampling`, on by default) works on a host whose `kernel.perf_event_paranoid` refuses performance counters to unprivileged processes, as a Slurm cluster's permissive setting lets it. A mounted directory whose symlinks point outside it, such as a HuggingFace hub snapshot (`models--org--name/snapshots/<rev>`, every file a relative link into the blob store), is handled by mounting the common root of the directory and its link targets at its host path and creating the recipe's container path (`/model`) as a symlink to it, so the links resolve as they do on the host. A sweep runs its points one after another. Ctrl+C stops the orchestrator, which stops its containers the same way it stops Slurm steps.

[`examples/docker/vllm-agg-1gpu.yaml`](https://github.com/NVIDIA/srt-slurm/blob/main/examples/docker/vllm-agg-1gpu.yaml) is a one-GPU recipe tested this way.

`launcher: docker` checks the recipe before running it and refuses one that needs more than this machine: more than one node, `resources.het_jobs`, a dedicated frontend, benchmark or infra node, service pools (`services[].nodes`), an engine that launches each endpoint as one multi-task MPI step (TRT-LLM), or a `model.container` / `roles.<role>.container` that is an enroot image file (`.sqsh`) rather than a Docker image name. Enroot URIs such as `nvcr.io#nvidia/x:tag` are rewritten to `nvcr.io/nvidia/x:tag`. Containers run as root unless `docker_args` sets `--user`, so files they write under `outputs/` are root-owned. `srtctl monitor` and the MCP job tools read Slurm and do not see Docker-launcher jobs.

#### `launcher: host`

```yaml
launcher: host
```

`host` runs a job the way `docker` does (the same job script in the foreground, `SRTCTL_JOB_ID=host-<timestamp>`, the same single-machine checks before it starts, one point of a sweep after another), but starts every process as a process group on this machine instead of in a container. It is for a machine with neither Slurm nor a container runtime: a Mac, a workstation, or a VM without the NVIDIA Container Toolkit.

| | `docker` | `host` |
|---|---|---|
| Process launch | `docker run ... <image> bash -c <wrapper>` | `bash -c <wrapper>` on this machine, in its own process group |
| Image (`model.container`, `roles.<role>.container`, the exporter images under `observability.tachometer` and `telemetry`) | the container each process runs in | ignored (logged once), and neither preflight nor the job checks that an image file exists, so a Slurm recipe whose `.sqsh` is only on the cluster runs here unchanged; a recipe must still name an image, and one that names a Docker image (as `examples/host/` does) also runs under `docker` and `slurm`, while `docker` refuses a `.sqsh` |
| Engine, benchmark client, tools | inside the image | on the `PATH` of the shell that ran `srtctl apply`, typically the engine installed in a virtualenv |
| Container mounts (`/logs`, `/model`, `/configs`, `extra_mount`, ...) | bind mounts | container paths in the command line, the exported environment and `container-workdir` are rewritten to the host directories behind them |
| Graceful stop of a named step | `docker kill --signal=TERM` | `killpg` on the step's process group |
| What a process leaves running when it exits | killed with the container | its process group is SIGKILLed; groups still running when the orchestrator finishes get SIGTERM, then SIGKILL |
| Working directory | the image's `WORKDIR` (or `container-workdir`) | a private scratch directory for the job (or `container-workdir`); srtctl's own host commands start in the checkout |
| A binary that is not installed | the step fails | the process fails with exit 127 and `not found` in its log |
| Node and address | this host, `127.0.0.1` | same |

Since nothing is mounted, the launcher rewrites each mount's container path to its host directory wherever it appears as a whole path in an argument (a `bash -c` script included): `--model /model` becomes `--model /data/qwen`, `/logs/sa-bench` becomes `outputs/<job_id>/logs/sa-bench`. A container path inside a longer path segment (`/models`, `/data/logs`) or a URL (`http://host:8000/model`) is left alone. A host directory whose path a shell would split (a space or a quote, as in `~/My Models/qwen`) is reached through a symlink to it in the job's private scratch directory (`$TMPDIR/srtctl-<job_id>-*/`, removed when the orchestrator exits), so the rewritten path stays one word inside `bash -c` scripts; a missing directory is created first, as `docker run -v` does. Paths written inside a script file (rather than on its command line) are not rewritten: a script that hardcodes `/logs` writes there. The bundled benchmark scripts under `src/srtctl/benchmarks/scripts/` still write their results to a fixed `/logs/...` directory, so under `host` use `benchmark.type: custom` (or `manual`) until they take their output directory from the command line.

A container also keeps whatever a step installs off the machine; on the host nothing does, so `host` guards installs instead:

- Every process starts with `PIP_REQUIRE_VIRTUALENV=1` (unless your environment sets it). `pip install`, with `--user` or `--break-system-packages` too, then refuses any Python that is not a virtualenv, and `uv pip install` already refuses one without `--system`. A step can install into the virtualenv its `pip` belongs to (the one active when you ran `srtctl apply`), never into the system or Homebrew Python.
- Every process gets the `PATH` of the shell that ran `srtctl apply`. What the job script and `uv run` put in front of it for the orchestrator (the checkout's `bin/`, `~/.local/bin`, srtctl's own virtualenv) follows it instead, srtctl's virtualenv last. `VIRTUAL_ENV` names your virtualenv: the one that shell had active or, under `uv run srtctl apply` (which replaces it with srtctl's), the virtualenv of the first `python3` on that `PATH`; `UV_PROJECT_ENVIRONMENT` no longer names srtctl's. So `python3` is your engine's interpreter, `pip` and `uv pip` install into the same virtualenv, and a process that would have run in a container starts in the job's scratch directory rather than the checkout.
- Two steps are refused before the job starts. The Dynamo install (`frontend.type: dynamo` with `dynamo.install: true`) marks itself done inside the Python it installed into, so a later job would skip its own install: install `ai-dynamo` where your engine runs and set `dynamo.install: false`. `reporting.ai_analysis` installs Claude Code and the GitHub CLI into your home directory with curl: set `reporting.ai_analysis.enabled: false`.
- A recipe's `setup_script` and `host_setup` run on this machine as you. `host_setup` starts in the srt-slurm checkout, as it does on a Slurm node, so with no virtualenv active a `uv pip install` there finds srtctl's own `.venv`; activate your engine's virtualenv before `srtctl apply`.

Do not run `make setup` on a Mac: it downloads Linux binaries into `bin/`, which the job script puts first on `PATH`. `srtctl apply` under `host` checks only for `uv` (from Homebrew or the standalone installer, on `PATH` or in `~/.local/bin`), plus the checkout files the recipe's services run, built for this machine: `configs/etcd` and `configs/nats-server` for the etcd/NATS discovery plane and `configs/otelcol-contrib` for the OTel collector (both implied by a Dynamo frontend, unless they are external or disabled). Without the Tachometer scraper, `dcgm-exporter` or `node_exporter`, which are not critical, those processes fail with exit 127 and the run goes ahead without their telemetry; set `observability.tachometer.enabled: false` to skip them. Closing the terminal `srtctl apply` runs in stops the job and its processes; run it under `nohup` to keep the job going after you log out. The job script locks its `uv sync` with `flock` where it exists; macOS has none and syncs without the lock.

[`examples/host/vllm-agg-1gpu.yaml`](https://github.com/NVIDIA/srt-slurm/blob/main/examples/host/vllm-agg-1gpu.yaml) is a one-GPU vLLM recipe for `host`, benchmarked with a custom `vllm bench serve` command; it sets nothing launcher-specific, so the same file is a valid `docker` or `slurm` recipe.

### Running without `srtslurm.yaml`

`srtslurm.yaml` is optional. A recipe can be fully self-sustaining as long as it supplies everything the cluster yaml would otherwise provide:

- Set `slurm.account`, `slurm.partition`, and `slurm.time_limit` directly in the recipe (no `default_*` fallback).
- Use absolute paths for `model.path`, `model.container`, and any other container fields; alias resolution is a no-op without the yaml's `containers:` / `model_paths:` maps.
- List every cluster-side mount the job needs in `extra_mount` (e.g. the lustre share that holds your model weights and `.sqsh` files). `default_mounts` is the only `srtslurm.yaml` field with no recipe-level equivalent until you spell mounts out yourself.
- Set `resources.gpus_per_node` explicitly.
- Status reporting and S3 log upload are skipped (their config lives under `reporting:` in the cluster yaml).

Workers' nats and etcd come from the dynamo/sglang container, not the yaml, so disagg/agg topologies still work end-to-end. `srtctl_root` falls back to the package install path automatically.

This is useful for portable recipes that you want to share across clusters or hand to a teammate without dragging cluster config along.
