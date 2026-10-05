# Paths, Mounts, and Environment

What the containers see: `FormattablePath` templates, mounts, environment variables, and the setup hooks that run before workers start.

## FormattablePath Template System

FormattablePath is a powerful templating system for paths that supports runtime placeholders and environment variable expansion.

### How It Works

FormattablePath ensures that configuration values with placeholders are always explicitly formatted before use, preventing accidental use of unformatted templates.

```yaml
# Example usage in config
output:
  log_dir: "$HOME/logs/{job_id}/{run_name}"

container_mounts:
  "$HOME/data": "/data"
  "$HOME/logs/{job_id}": "/logs"
```

### Available Placeholders

| Placeholder         | Type   | Description                          | Example                        |
| ------------------- | ------ | ------------------------------------ | ------------------------------ |
| `{job_id}`          | string | SLURM job ID                         | "12345"                        |
| `{run_name}`        | string | Job name + job ID                    | "my-benchmark_12345"           |
| `{head_node_ip}`    | string | IP address of head node              | "10.0.0.1"                     |
| `{log_dir}`         | string | Resolved log directory path          | "/home/user/outputs/12345/logs"|
| `{model_path}`      | string | Resolved model path                  | "/models/deepseek-r1"          |
| `{container_image}` | string | Resolved container image path        | "/containers/sglang.sqsh"      |
| `{gpus_per_node}`   | int    | GPUs per node                        | 8                              |

### Environment Variable Expansion

FormattablePath also expands environment variables using `$VAR` or `${VAR}` syntax:

```yaml
output:
  log_dir: "$HOME/outputs/{job_id}/logs"
  # Expands to: /home/username/outputs/12345/logs
```

Common environment variables:
- `$HOME` - User home directory
- `$USER` - Username
- `$SLURM_JOB_ID` - SLURM job ID (also available as `{job_id}`)

### Extra Placeholders

Some contexts support additional placeholders:

| Placeholder       | Context           | Description                     |
| ----------------- | ----------------- | ------------------------------- |
| `{nginx_url}`     | Frontend config   | Nginx URL for load balancing    |
| `{frontend_url}`  | Frontend config   | Frontend/router URL             |
| `{index}`         | Worker config     | Worker index                    |
| `{host}`          | Worker config     | Worker host                     |
| `{port}`          | Worker config     | Worker port                     |

### Examples

```yaml
# Log directory with job ID
output:
  log_dir: "./outputs/{job_id}/logs"

# Mount user data into container
container_mounts:
  "$HOME/datasets": "/datasets"
  "./outputs/{job_id}": "/outputs"

# Custom paths with environment variables
extra_mount:
  - "$SCRATCH/cache:/cache"
  - "${DATA_DIR}/models:/models:ro"
```

## container_mounts

Custom container mount mappings with FormattablePath support.

```yaml
container_mounts:
  "$HOME/datasets": "/datasets"
  "$HOME/outputs/{job_id}": "/outputs"
  "/shared/cache": "/cache"
```

Both keys and values support FormattablePath templating with placeholders and environment variables.

### Default Mounts

The following mounts are always added automatically:

| Host Path              | Container Path       | Description                  |
| ---------------------- | -------------------- | ---------------------------- |
| Model path             | `/model`             | Resolved model directory     |
| Log directory          | `/logs`              | Log output directory         |
| `configs/` directory   | `/configs`           | NATS, etcd binaries          |
| Benchmark scripts      | `/srtctl-benchmarks` | Bundled benchmark scripts    |

### Cluster-Level Mounts

You can also define cluster-wide mounts in `srtslurm.yaml` using the `default_mounts` field. These are applied to all jobs on the cluster, after the built-in defaults but before job-level mounts.

```yaml
# In srtslurm.yaml
default_mounts:
  "/cluster/special/libs": "/opt/libs"
  "$SCRATCH": "/scratch"
```

Environment variables (e.g., `$SCRATCH`, `$HOME`) are expanded. This is useful for mounting cluster-specific paths that are required by certain images without adding them to every job config.

### Mount Priority

Mounts have the following priority (highest to lowest):

1. **Job-level `container_mounts`** - FormattablePath dict (highest priority)
2. **Job-level `extra_mount`** - simple `host:container` strings
3. **Cluster-level** - `default_mounts` from `srtslurm.yaml`
4. **Built-in defaults** - model, logs, configs, benchmark scripts (lowest priority)

Job-level mounts always take precedence over cluster-level and built-in defaults.

## environment

Global environment variables for all worker processes.

```yaml
environment:
  MY_VAR: "value"
  CUDA_LAUNCH_BLOCKING: "1"
  NCCL_DEBUG: "INFO"
```

### Per-Worker Template Variables

Environment variable values support per-worker templating with these placeholders:

| Placeholder | Description                                    | Example      |
| ----------- | ---------------------------------------------- | ------------ |
| `{node}`    | Hostname of the node where the worker runs     | `"gpu-01"`   |
| `{node_id}` | Numeric index of the node in worker list (0-based) | `0`, `1`, `2` |

**Note**: For per-role environment variables, use `roles.prefill.env`, `roles.decode.env`, or `roles.agg.env` (see [roles](topology.md#roles)). The role's `env` is applied first and the global `environment` after it, so a key set in both takes the global value.

## extra_mount

Additional container mounts as a list of mount specifications.

```yaml
extra_mount:
  - "/local/path:/container/path"
  - "/data:/data:ro"
  - "$HOME/cache:/cache"
```

| Format                        | Description                          |
| ----------------------------- | ------------------------------------ |
| `host_path:container_path`    | Read-write mount                     |
| `host_path:container_path:ro` | Read-only mount                      |

**Note**: Unlike `container_mounts`, `extra_mount` uses simple string format, not FormattablePath. Environment variables are still expanded.

## setup_script

Run a custom script before dynamo install and worker startup.

```yaml
setup_script: "install-custom-deps.sh"
```

**Notes**:

- Script must be located in the `configs/` directory.
- Script runs inside the container before dynamo installation.
- Useful for installing custom SGLang versions, additional dependencies, or patches.

**Example setup script** (`configs/install-sglang-main.sh`):

```bash
#!/bin/bash
pip install --quiet git+https://github.com/sgl-project/sglang.git
```

## host_setup

Commands run on each allocated node's **bare host, outside the container**, before any worker starts.

This is the counterpart to [`setup_script`](#setup_script), which runs *inside* the container. Use `host_setup` for node state the container cannot reach: locking GPU clocks, loading a kernel module, dropping caches.

```yaml
host_setup:
  commands:
    - "sudo -n nvidia-smi -lmc <min>,<max>"
  teardown:
    - "sudo -n nvidia-smi -rmc"
  nodes: all
  ignore_failure: false
  timeout_seconds: 300
```

Fields: [HostSetupConfig](schema-reference.md#hostsetupconfig).

**How it runs**: the orchestrator itself runs on the host (not in a container), so it fans these out as one container-less `srun` per node, in parallel. Output lands in `<log_dir>/host_setup_<node>.out` and `<log_dir>/host_teardown_<node>.out`.

**Notes**:

- **Commands run as you, not as root.** Anything privileged needs passwordless sudo (`sudo -n ...`). A `sudo` that prompts for a password will hang until `timeout_seconds` and then fail the job; verify first with `srun --jobid <job> --overlap -w <node> sudo -n true`. If sudo prompts, no recipe change helps; the cluster's SLURM `Prolog=` (which runs as root) is the only route.
- **Prefer setting `teardown` whenever `commands` changes persistent node state.** `nvidia-smi -lmc` outlives the allocation, so without a matching `-rmc` the next job on that node inherits your locked clocks. `srtctl dry-run` warns when `commands` is set without `teardown`.
- `teardown` runs from the job's cleanup path, so it fires on failure and cancellation too, and never changes the job's exit code.
- Set cluster-wide via `default_host_setup` in `srtslurm.yaml`; that's the right home when *the cluster's machines* need this, rather than one recipe. See [Cluster Config Fields](cluster-config.md#cluster-config-fields).
- `srtctl dry-run -f config.yaml` renders the commands, their scope, and which file they came from.
- For more than a couple of commands, or when you want each one logged with its exit code, point both lists at `configs/node-hooks.sh` and write the commands one per line in `HOOK_PRE` / `HOOK_POST` block scalars under `environment:`. The job script exports those, so the host sruns inherit them. See [examples/features/node-hooks.yaml](https://github.com/NVIDIA/srt-slurm/blob/main/examples/features/node-hooks.yaml).
