# Installation

## Table of Contents

- [Prerequisites](#prerequisites)
- [Clone and Install](#clone-and-install)
- [Gather your cluster user and target partition](#gather-your-cluster-user-and-target-partition)
- [Run Setup](#run-setup)
- [Review srtslurm.yaml Defaults](#review-srtslurmyaml-defaults)
- [Install Model + Container Artifacts](#install-model--container-artifacts)
- [Configure srtslurm.yaml](#configure-srtslurmyaml)
  - [Adding Model Paths](#adding-model-paths)
  - [Adding Containers](#adding-containers)
  - [Complete srtslurm.yaml Reference](#complete-srtslurmyaml-reference)
- [Create a Job Config](#create-a-job-config)
- [Submit the Job](#submit-the-job)
- [Custom Setup Scripts](#custom-setup-scripts)

---

## Prerequisites

- Access to a SLURM cluster with GPU nodes
- Python 3.10+
- Container runtime (enroot/pyxis) configured on the cluster
- Model weights accessible from compute nodes
- SGLang container image (`.sqsh` format)

## Clone and Install

```bash
git clone https://github.com/ishandhanani/srt-slurm.git
cd srt-slurm
pip install -e .
```

## Gather your cluster user and target partition

These commands might not work on all clusters. You can use AI to figure out the right set of commands for your cluster.

```bash
# user
sacctmgr -nP show assoc where user=$(whoami) format=account
# partition
sinfo
```

## Run Setup

If you are trying to deploy onto Grace (GH200, GB200, etc.), you need to use the `aarch64` architecture. Otherwise use `x86_64`.

```bash
make setup ARCH=aarch64  # or ARCH=x86_64
```

The setup will:

1. Download NATS/ETCD binaries for your architecture
2. Prompt you for cluster settings:
   - SLURM account (default: `restricted`)
   - SLURM partition (default: `batch`)
   - GPUs per node (default: `4`)
   - Time limit (default: `4:00:00`)
3. Create `srtslurm.yaml` with your settings
4. Auto-detect and set `srtctl_root` path

## Review srtslurm.yaml Defaults

Before artifact install, open `srtslurm.yaml` once and confirm cluster-level defaults are correct for your environment:

- `default_account`
- `default_partition`
- `default_time_limit`
- `gpus_per_node`

`srtctl install` reads these values when generating the sbatch script.

## Install Model + Container Artifacts

Use `srtctl install` to download model weights, import the container `.sqsh`, and register aliases in `srtslurm.yaml`.

Flow order:

1. Run `make setup` (creates `srtslurm.yaml`)
2. Review/update cluster defaults in `srtslurm.yaml`
3. Run `srtctl install ...` (downloads artifacts and auto-registers aliases)

### Prerequisites for `srtctl install`

- Hugging Face token exported in your shell:

  ```bash
  export HF_TOKEN=hf_xxx
  ```

- For `nvcr.io/*` container images, registry credentials must be configured for enroot/docker.
  Typical locations:
  - `~/.config/enroot/.credentials`
  - `~/.docker/config.json`

### Generic install (any model + any container)

Provide all three artifact fields explicitly (`--hf-repo-id`, `--model-alias`, `--container-image`):

```bash
srtctl install my-model \
  --hf-repo-id org/model-repo \
  --model-alias org/model-alias \
  --container-image docker.io/org/runtime:tag
```

`--container-image` accepts standard Docker image references (the same value you use in `docker pull`).

Common forms:

- Docker Hub image (recommended style): `lmsysorg/sglang:gemma4-mtp-cu13`
- Docker Hub image (equivalent explicit form): `docker.io/lmsysorg/sglang:gemma4-mtp-cu13`
- NGC image (must include registry): `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.0-dev.3`

Rule of thumb:
- For Docker Hub images, use the same short form shown on Docker Hub pages.
- For `nvcr.io` images, always include the `nvcr.io/` prefix.

Optional flags:
- `--strict-auth-preflight` (fail early if common nvcr.io credential files are missing; default is warning-only and still attempts pull)

### Execution mode

`srtctl install` is SLURM-only and always submits an `sbatch` job.
The old `--slurm` flag is no longer needed.

```bash
srtctl install my-model \
  --hf-repo-id org/model-repo \
  --model-alias org/model-alias \
  --container-image docker.io/org/runtime:tag \
  --venv /path/to/venv
```

Monitor with:

```bash
squeue -u $USER
tail -f install/install_<model>_<jobid>.log
```

## Configure srtslurm.yaml

`srtslurm.yaml` contains cluster defaults plus model/container alias mappings.

If you used `srtctl install`, model/container aliases are added automatically after a successful install.
You only need to edit alias entries manually when:

- You did not run `srtctl install`, or
- You want to point aliases at pre-existing model/container paths, or
- You want to rename aliases used by recipes.

You should still review cluster-level defaults (`default_account`, `default_partition`, etc.) manually.

### Automatic alias registration (recommended)

After a successful `srtctl install ...`, aliases are written to:

- `model_paths[<model-alias>] = <downloaded-model-path>`
- `containers[<container-image>] = <downloaded-sqsh-path>`

No manual alias editing is required for the installed model/container pair.

### Manual alias configuration (optional)

Use this only when you manage models/containers yourself and want to map aliases manually.

### Adding Model Paths

The `model_paths` section maps short aliases to full filesystem paths:

```yaml
model_paths:
  deepseek-r1: "/mnt/lustre/models/DeepSeek-R1"
  deepseek-r1-fp4: "/mnt/lustre/models/deepseek-r1-0528-fp4-v2"
```

Models must be accessible from all compute nodes (typically on a shared filesystem like Lustre or GPFS).

### Adding Containers

The `containers` section maps version aliases to `.sqsh` container images:

```yaml
containers:
  container1: "/mnt/containers/lmsysorg+sglang+v0.5.5.sqsh"
  container2: "/mnt/containers/lmsysorg+sglang+v0.5.4.sqsh"
```

To create a container image from Docker:

```bash
enroot import docker://lmsysorg/sglang:v0.5.5
```

### Complete srtslurm.yaml Reference

Here's a complete example of all available options:

```yaml
# Default SLURM settings
default_account: "your-account"
default_partition: "batch"
default_time_limit: "4:00:00"

# Resource defaults
gpus_per_node: 4

# SLURM directive compatibility
use_gpus_per_node_directive: true # Set false if cluster doesn't support --gpus-per-node
use_segment_sbatch_directive: true # Set false if cluster doesn't support --segment
use_exclusive_sbatch_directive: false # Set true if cluster requires --exclusive

# Path to srtctl repo root (auto-set by make setup)
srtctl_root: "/path/to/srtctl"

# Custom output directory for job logs (optional)
# If set, job outputs go here instead of {srtctl_root}/outputs/
# Useful when running from temp dirs or CI/CD pipelines
# Can also be set via CLI: srtctl apply -f config.yaml -o /path/to/outputs
output_dir: "/persistent/path/to/outputs"

# Model path aliases
model_paths:
  deepseek-r1: "/models/DeepSeek-R1"
  llama-70b: "/models/Llama-3-70B"

# Container aliases
containers:
  latest: "/containers/sglang-latest.sqsh"
  stable: "/containers/sglang-stable.sqsh"

# Optional: default for recipes that omit frontend.nginx_raise_ulimit (nginx high-nofile tuning)
# nginx_raise_ulimit: true
```

## Create a Job Config

Create `configs/my-job.yaml`:

```yaml
name: "my-benchmark"

model:
  path: "deepseek-r1" # Uses alias from srtslurm.yaml
  container: "latest" # Uses alias from srtslurm.yaml
  precision: "fp8"

extra_mount: # add this if you need to mount extra directories to the container
  - "/local-dir1:/container-dir1"
  - "/local-dir2:/container-dir2"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 2
  prefill_workers: 1
  decode_workers: 1
  gpus_per_node: 4

slurm:
  time_limit: "02:00:00"

backend:
  prefill_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
  decode_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"

  sglang_config:
    prefill:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.84
      tensor-parallel-size: 4
    decode:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.83
      tensor-parallel-size: 8
      expert-parallel-size: 8
      data-parallel-size: 8
      enable-dp-attention: true

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [256, 512]
  req_rate: "inf"
```

See [Configuration Reference](config-reference.md) for all available options.

## Submit the Job

```bash
srtctl apply -f configs/my-job.yaml
```

Output:

```
Submitted batch job 12345
Logs: logs/12345_1P_4D_20251122_143052/
```

### Submit with Tags

You can tag runs for easier filtering in the dashboard:

```bash
srtctl apply -f configs/my-job.yaml --tags experiment,baseline,v2
```

Tags are saved in the job metadata and can be used to filter runs in analysis.

See [Monitoring](monitoring.md) for how to monitor your job and understand the detailed log structure.

## Custom Setup Scripts

You can run custom initialization scripts on worker nodes before starting SGLang workers. This is useful for:

- Setting up custom environment variables
- Installing additional dependencies
- Checking out custom code

### Creating a Setup Script

1. Create your setup script in the `configs/` directory:

   ```bash
   # configs/custom-setup.sh
   # Example of checking out a specific branch of SGLang
   #!/bin/bash
    cd /sgl-workspace/
    rm -rf sglang
    git clone https://github.com/sgl-project/sglang.git
    cd sglang
    git checkout origin/cheng/refactor/sbo
    git config --global --add safe.directory "*"
    pip install -e "python"
   ```

2. Make it executable:

   ```bash
   chmod +x configs/custom-setup.sh
   ```

3. Submit with the `--setup-script` flag:
   ```bash
   srtctl apply -f configs/my-job.yaml --setup-script custom-setup.sh
   ```

The script will be executed on each worker node (prefill, decode, or aggregated) before installing Dynamo from PyPI and starting the SGLang workers. The script must be located in the `configs/` directory, which is mounted into containers at `/configs/`.

**Note**: Setup scripts only run when you explicitly specify `--setup-script`. No default setup script will run if this flag is omitted.
