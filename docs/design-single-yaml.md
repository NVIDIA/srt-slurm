# Design: Single Self-Contained Recipe YAML

## Problem

Today srt-slurm has a **two-YAML problem**:

1. **`srtslurm.yaml`** (cluster config) — maps aliases to physical paths, sets SLURM defaults
2. **Recipe YAML** — references aliases like `dsr1-fp4` and `dynamo-sglang`

A recipe alone tells you nothing. `model.path: "dsr1-fp4"` is meaningless without the Rosetta stone of `srtslurm.yaml`. This makes recipes:
- **Not reproducible** — you can't run someone else's recipe without their srtslurm.yaml
- **Not self-documenting** — reading a recipe doesn't tell you what docker image or HF model was used
- **Painful for new users** — they must hand-curate path mappings before running anything

## Principle: No External Config File

There is no `srtslurm.yaml`, no `cluster.yaml`, no second file. The recipe is the only input to `srtctl apply`. Everything needed to understand and reproduce a run is in that one document.

Things that `srtslurm.yaml` used to provide fall into three buckets:

| What | Where it goes now |
|------|-------------------|
| Model/container aliases | Eliminated. Virtual identity (HF model ID, docker tag) goes in recipe. Physical path goes in recipe. |
| SLURM defaults (account, partition, time_limit) | Required fields in the recipe. No implicit defaults. |
| Cluster capability flags (gpus_per_node directive, segment, exclusive) | Auto-detected from the SLURM environment, or explicit in recipe. |
| Reporting config | In the recipe, or environment variables. |

---

## Recipe Format

```yaml
name: "gb200-fp4-1k1k-max-tpt"

# ============================================================
# VIRTUAL — the reproducibility contract
# These fields define WHAT ran. Portable across clusters.
# A new user reads these and knows exactly what to download/pull.
# ============================================================

model:
  name: "deepseek-ai/DeepSeek-R1"       # HuggingFace model ID
  revision: "e4e908c07378..."            # HF git commit SHA (pins exact weights)
  precision: "fp4"

container:
  image: "lmsysorg/sglang:v0.4.6.post1" # Docker image tag
  digest: "sha256:a1b2c3d4..."           # Manifest digest (pins exact image)
  pip_install:                           # Additional packages installed at runtime
    - "ai-dynamo==0.8.1"

# ============================================================
# PHYSICAL — the deployment details
# These fields define WHERE it runs. Cluster-specific.
# A new user changes these (and only these) for their cluster.
# ============================================================

model:
  path: "/lustre/fsw/.../DeepSeek-R1"

container:
  path: "/lustre/fsw/.../sglang-v0.4.6.sqsh"

slurm:
  account: "coreai_tritoninference_triton3"
  partition: "gb200"
  time_limit: "04:00:00"

resources:
  gpu_type: "gb200"
  gpus_per_node: 4
  prefill_nodes: 4
  decode_nodes: 12
  prefill_workers: 4
  decode_workers: 1

# ============================================================
# WORKLOAD — the benchmark/serving configuration
# ============================================================

backend:
  type: "sglang"
  # sglang_config, environments, etc. (unchanged from today)

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: "2048x4096"
  req_rate: "inf"
```

Note: virtual + physical fields merge under the same YAML key:

```yaml
model:
  # Virtual
  name: "deepseek-ai/DeepSeek-R1"
  revision: "e4e908c07378..."
  precision: "fp4"
  # Physical
  path: "/lustre/fsw/.../DeepSeek-R1"

container:
  # Virtual
  image: "lmsysorg/sglang:v0.4.6.post1"
  digest: "sha256:a1b2c3d4..."
  pip_install: ["ai-dynamo==0.8.1"]
  # Physical
  path: "/lustre/fsw/.../sglang-v0.4.6.sqsh"
```

### What changed from today

| Before | After | Why |
|--------|-------|-----|
| `model.path: "dsr1-fp4"` | `model.name: "deepseek-ai/DeepSeek-R1"` + `model.revision: "e4e908..."` | HF model ID + git SHA pins exact weights |
| `model.container: "dynamo-sglang"` | `container.image: "..."` + `container.digest: "sha256:..."` | Docker tag + manifest digest pins exact image |
| `dynamo.version: 0.8.1` | `container.pip_install: ["ai-dynamo==0.8.1"]` | Full software stack in one place |
| Physical paths in srtslurm.yaml | Physical paths inline in recipe | One file, self-contained |
| SLURM defaults in srtslurm.yaml | Required in recipe | No hidden state |
| Cluster flags in srtslurm.yaml | Auto-detected or in recipe | Cluster should know itself |

---

## Checksums and Verification

### Content-addressable identifiers

| Artifact | Identifier | Source | Immutable? |
|----------|-----------|--------|------------|
| HF model | Git commit SHA (`revision`) | `huggingface_hub.repo_info().sha` | Yes |
| Docker image | Manifest digest (`sha256:...`) | `Docker-Content-Digest` header or `skopeo inspect` | Yes |
| Local .sqsh file | File SHA256 | Computed at import time | Yes |

Tags and model IDs are mutable — someone can re-push `v0.4.6.post1` or update model files. The digest/revision fields are the actual pins.

### Recipe fields

```yaml
model:
  name: "deepseek-ai/DeepSeek-R1"
  revision: "e4e908c073784de20ad3af0be653421f1088922d"

container:
  image: "lmsysorg/sglang:v0.4.6.post1"
  digest: "sha256:a1b2c3d4e5f6789..."
```

Both `revision` and `digest` are optional in the recipe (you might not know them when first writing it). The lockfile fills them in after the run.

---

## Runtime Fingerprint

The recipe captures *intent*. The fingerprint captures *reality* — what was actually running inside the container. This is the equivalent of `pip freeze > requirements.txt` but for the entire runtime environment.

### What gets captured

```json
{
  "timestamp": "2026-04-09T14:30:00Z",
  "node": "lyris-gb200-001",
  "mode": "prefill",
  "worker_index": 0,
  "python_packages": "... pip freeze output ...",
  "nvidia_smi": "... gpu info ...",
  "cuda_version": "12.8",
  "driver_version": "570.86.15",
  "torch_version": "2.6.0",
  "os_release": "Ubuntu 22.04",
  "cpu_arch": "aarch64",
  "nccl_version": "2.25.1"
}
```

### Where it runs

Injected into the **worker preamble** — the bash commands that run inside the container before the actual worker process starts. This already exists in `worker_stage.py:_build_worker_preamble()`.

Critically, the fingerprint runs **after** all setup and pip installs — it captures the actual runtime state, not the base image. This means transitive deps from `pip install ai-dynamo`, anything the setup script patched, nightly wheels — all visible.

```bash
# Preamble sequence (inside container):
source /opt/setup.sh                                     # existing setup
pip install ai-dynamo==0.8.1                             # existing dynamo install
capture_fingerprint /logs/fingerprint_prefill_w0.json    # captures REAL state (~2s)
python -m sglang.launch_server ...                       # actual work
```

### Canonical ordering

**Everything written to a fingerprint or lockfile is sorted deterministically.** This is what makes `srtctl diff` work — if pip freeze comes back in random order on different nodes, every line looks changed and the diff is useless.

The ordering rules:

| Data | Sort order |
|------|-----------|
| pip packages | Alphabetical by package name (case-insensitive) |
| Environment variables | Alphabetical by key |
| sglang_config flags | Alphabetical by flag name |
| Container mounts | Alphabetical by container path |
| YAML keys in lockfile | Fixed schema order (not alphabetical — grouped by section) |
| Per-worker fingerprints | Sorted by `{mode}_{worker_index}` (prefill_w0, prefill_w1, decode_w0, ...) |

The fingerprint capture script sorts at write time — `pip freeze | sort` instead of `pip freeze`. The lockfile writer uses an ordered schema, not `yaml.dump()` with default key ordering. This means any two fingerprints from identical environments produce byte-identical output.

### Fault tolerance design

**Core rule: fingerprinting never kills a job.**

The fingerprint capture is a single bash function that wraps every probe in its own subshell with `|| true`. If `pip` doesn't exist, we note it. If `nvidia-smi` hangs, the timeout kills it. If the filesystem is read-only, we skip the write. The worker process launches regardless.

The function writes valid, sorted JSON so that diffs are clean:

```bash
capture_fingerprint() {
    local output="$1"
    # Entire function is wrapped — if anything catastrophic happens, we bail
    # but the calling script continues (called with || true)

    python3 -c '
import json, subprocess, sys, os, platform

def run(cmd, timeout=5):
    """Run a command, return stdout or None on any failure."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def get_pip_packages():
    """Return sorted list of package==version strings."""
    out = run("pip freeze")
    if not out:
        return []
    packages = sorted(out.splitlines(), key=lambda s: s.lower())
    return packages

def get_gpu_info():
    """Return GPU info from nvidia-smi."""
    out = run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    if not out:
        return {"available": False}
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    gpus = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    return {"available": True, "gpus": gpus, "driver": gpus[0]["driver"] if gpus else "unknown"}

def get_version(cmd):
    """Run a version command, return string or 'unavailable'."""
    return run(cmd) or "unavailable"

# Build fingerprint — keys in fixed order for clean diffs
fingerprint = {
    # 1. Identity
    "hostname": run("hostname") or "unknown",
    "timestamp": run("date -u +%Y-%m-%dT%H:%M:%SZ") or "unknown",
    # 2. Hardware + OS
    "arch": platform.machine(),
    "os": get_version("cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d \\\""),
    "gpu": get_gpu_info(),
    # 3. Core versions (fixed key order)
    "python_version": platform.python_version(),
    "cuda_version": get_version("nvcc --version 2>/dev/null | grep release | awk \"{print \\$6}\" | tr -d ,"),
    "torch_version": get_version("python3 -c \"import torch; print(torch.__version__)\""),
    "nccl_version": get_version("python3 -c \"import torch; print(torch.cuda.nccl.version())\""),
    # 4. Full package list (sorted)
    "pip_packages": get_pip_packages(),
}

# Write with sorted keys disabled — we control the order explicitly via insertion order
json.dump(fingerprint, sys.stdout, indent=2)
' > "$output" 2>/dev/null
}

# Called in preamble — ALWAYS continues to next command
capture_fingerprint "/logs/fingerprint_${MODE}_w${WORKER_IDX}.json" || true
```

Key properties:
- **Python script, not bash echo soup** — proper JSON output, no quoting bugs, easy to extend
- **Each probe is independent** — `run()` catches all exceptions, returns None on failure
- **`nvidia-smi` has a timeout** — protects against GPU driver hangs
- **pip packages are sorted** — diffs show only actual changes, not reordering noise
- **Keys are in fixed insertion order** — identity first, then hardware, then versions, then packages
- **The whole function is called with `|| true`** — even a syntax error won't kill the worker
- **Writes to `/logs/` mount** — persisted to host output directory automatically
- **Takes ~2 seconds total** — negligible compared to model loading

### Lockfile key ordering

The lockfile uses a fixed section order, not YAML's default alphabetical sort:

```yaml
# Section order is always:
# 1. _meta (job info, validation, checksums, aggregated fingerprint)
# 2. name
# 3. model (name, revision, precision, path)
# 4. container (image, digest, pip_install, path)
# 5. slurm (account, partition, time_limit)
# 6. resources
# 7. backend
# 8. benchmark
```

Within each section, keys follow a fixed order defined in the schema. The lockfile writer uses a custom YAML dumper that respects insertion order rather than sorting alphabetically. This means:
- Two lockfiles from identical runs are byte-identical
- Diffs between lockfiles show only meaningful changes
- Reading a lockfile top-to-bottom tells a coherent story (what → where → how)

### Per-worker vs global

Each worker writes its own fingerprint. This is intentional:
- In disaggregated setups, prefill and decode nodes may have different GPU types, driver versions, or NCCL builds
- If one node has a bad driver, you can see it in that node's fingerprint
- The lockfile aggregates them into a single view

---

## Pre-Submit Validation (Background, Non-Blocking)

When the user runs `srtctl apply -f recipe.yaml`, validation runs **concurrently with job submission**. It never blocks, never fails the submit. Results are logged.

### What gets checked

```
$ srtctl apply -f recipe.yaml

Submitting job...                                    # starts immediately
  [background] Checking deepseek-ai/DeepSeek-R1 on HuggingFace...
  [background] Checking lmsysorg/sglang:v0.4.6.post1 on Docker Hub...
  [background] Verifying local model path...
  [background] Verifying local container path...

Job 12345 submitted.

Validation results:
  Model (HuggingFace):  ✓ exists, revision e4e908c matches
  Image (Docker Hub):   ✓ exists, digest sha256:a1b2c3 matches
  Local model path:     ✓ /lustre/fsw/.../DeepSeek-R1 (100 files, 685GB)
  Local container:      ✓ /lustre/fsw/.../sglang.sqsh (4.2GB)
```

Or if something is wrong:

```
Validation results:
  Model (HuggingFace):  ⚠ revision e4e908c not found (HEAD is now f5a901b)
  Image (Docker Hub):   ⚠ tag re-pushed, digest changed (was sha256:a1b2c3, now sha256:d4e5f6)
  Local model path:     ✓ exists
  Local container:      ⚠ file not found at /lustre/fsw/.../sglang.sqsh
```

### Fault tolerance

```python
async def validate_recipe(recipe: RecipeConfig) -> ValidationReport:
    """Run all validation checks concurrently. Never raises. Never blocks submit."""
    checks = []

    if recipe.model.name:
        checks.append(("hf_model", _check_hf_model(recipe.model)))
    if recipe.container.image:
        checks.append(("docker_image", _check_docker_image(recipe.container)))
    if recipe.model.path:
        checks.append(("local_model", _check_local_path(recipe.model.path)))
    if recipe.container.path:
        checks.append(("local_container", _check_local_path(recipe.container.path)))

    results = {}
    for name, coro in checks:
        try:
            results[name] = await asyncio.wait_for(coro, timeout=10.0)
        except asyncio.TimeoutError:
            results[name] = CheckResult(status="timeout", message=f"{name} check timed out (10s)")
        except Exception as e:
            results[name] = CheckResult(status="error", message=f"{name} check failed: {e}")

    return ValidationReport(results)


async def _check_hf_model(model: ModelConfig) -> CheckResult:
    """Check HuggingFace model existence and revision match."""
    try:
        resp = requests.head(
            f"https://huggingface.co/api/models/{model.name}",
            timeout=5.0,
        )
        if resp.status_code == 404:
            return CheckResult("warning", f"Model {model.name} not found on HuggingFace")
        if resp.status_code == 401:
            return CheckResult("info", f"Model {model.name} exists (gated, needs token)")
        if resp.status_code == 200 and model.revision:
            # Check specific revision
            rev_resp = requests.head(
                f"https://huggingface.co/api/models/{model.name}/revision/{model.revision}",
                timeout=5.0,
            )
            if rev_resp.status_code == 404:
                return CheckResult("warning", f"Revision {model.revision[:12]} not found")
            return CheckResult("ok", f"Model exists, revision {model.revision[:12]} verified")
        return CheckResult("ok", f"Model {model.name} exists on HuggingFace")
    except requests.RequestException as e:
        return CheckResult("error", f"HuggingFace check failed: {e}")


async def _check_docker_image(container: ContainerConfig) -> CheckResult:
    """Check Docker image existence and digest match."""
    try:
        # Parse image into registry/repo:tag
        repo, tag = _parse_image_ref(container.image)
        resp = requests.head(
            f"https://registry.hub.docker.com/v2/{repo}/manifests/{tag}",
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
            timeout=5.0,
        )
        if resp.status_code == 404:
            return CheckResult("warning", f"Image {container.image} not found")
        if resp.status_code == 200 and container.digest:
            remote_digest = resp.headers.get("Docker-Content-Digest", "")
            if remote_digest != container.digest:
                return CheckResult(
                    "warning",
                    f"Digest mismatch: recipe has {container.digest[:20]}, "
                    f"remote has {remote_digest[:20]} (tag may have been re-pushed)"
                )
            return CheckResult("ok", f"Image exists, digest verified")
        return CheckResult("ok", f"Image {container.image} exists")
    except requests.RequestException as e:
        return CheckResult("error", f"Docker Hub check failed: {e}")


def _check_local_path(path: str) -> CheckResult:
    """Check that a local file or directory exists."""
    p = Path(path)
    if not p.exists():
        return CheckResult("warning", f"Path not found: {path}")
    if p.is_dir():
        file_count = sum(1 for _ in p.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return CheckResult("ok", f"{file_count} files, {total_size / 1e9:.1f}GB")
    else:
        size = p.stat().st_size
        return CheckResult("ok", f"{size / 1e9:.1f}GB")
```

Every check:
- Has its own `try/except` — one failing check never affects others
- Has a timeout — network checks cap at 5s each, overall validation caps at 10s
- Returns a result object, never raises
- Runs concurrently with job submission

---

## Lockfile

After a run completes, `srtctl` writes `recipe.lock.yaml` to the output directory. This is the fully-resolved state of what actually ran.

```
outputs/12345_4P_12D_gb200-fp4-1k1k-max-tpt/
  recipe.lock.yaml              # resolved recipe + runtime metadata
  fingerprint_prefill_w0.json   # per-worker environment snapshots
  fingerprint_prefill_w1.json
  fingerprint_decode_w0.json
  ...
  benchmark.out
  logs/
```

### Lockfile contents

```yaml
# AUTO-GENERATED by srtctl. This is the resolved state of the run.
_meta:
  job_id: "12345"
  submitted_at: "2026-04-09T14:30:00Z"
  completed_at: "2026-04-09T15:45:00Z"
  cluster: "lyris"
  nodes: ["lyris-gb200-001", "lyris-gb200-002", "lyris-gb200-003", ...]
  srtctl_version: "0.9.0"
  srtctl_git_sha: "abc123"

  # Pre-submit validation results
  validation:
    hf_model: {status: "ok", message: "exists, revision verified"}
    docker_image: {status: "ok", message: "exists, digest verified"}
    local_model: {status: "ok", message: "100 files, 685.2GB"}
    local_container: {status: "ok", message: "4.2GB"}

  # Checksums resolved at submit time
  resolved_checksums:
    model_revision: "e4e908c073784de20ad3af0be653421f1088922d"
    image_digest: "sha256:a1b2c3d4e5f6..."
    sqsh_sha256: "d7e8f9a0b1c2..."  # computed from local file

  # Aggregated runtime fingerprint (from per-worker captures)
  runtime_fingerprint:
    python: "3.11.9"
    torch: "2.6.0"
    cuda: "12.8"
    driver: "570.86.15"
    nccl: "2.25.1"
    arch: "aarch64"
    os: "Ubuntu 22.04.5 LTS"
    pip_packages:
      - "ai-dynamo==0.8.1"
      - "sglang==0.4.6.post1"
      - "torch==2.6.0+cu128"
      - "triton==3.2.0"
      # ... full pip freeze

# Full recipe (with all physical paths filled in)
name: "gb200-fp4-1k1k-max-tpt"
model:
  name: "deepseek-ai/DeepSeek-R1"
  revision: "e4e908c073784de20ad3af0be653421f1088922d"
  precision: "fp4"
  path: "/lustre/fsw/.../DeepSeek-R1"
container:
  image: "lmsysorg/sglang:v0.4.6.post1"
  digest: "sha256:a1b2c3d4e5f6..."
  pip_install: ["ai-dynamo==0.8.1"]
  path: "/lustre/fsw/.../sglang-v0.4.6.sqsh"
slurm:
  account: "coreai_tritoninference_triton3"
  partition: "gb200"
  time_limit: "04:00:00"
# ... rest of recipe
```

The lockfile is the answer to "exactly what ran?". The recipe is the answer to "what do I want to run?". The fingerprint files are the raw evidence.

---

## Fingerprint Operations

Fingerprints aren't just for logging — they're a tool you actively use.

### `srtctl diff` — Compare two runs

Takes two output directories (or lockfiles) and shows what changed between them.

```bash
$ srtctl diff outputs/12345 outputs/12400

Recipe changes:
  container.image:  lmsysorg/sglang:v0.4.6.post1  →  lmsysorg/sglang:v0.4.7
  container.digest: sha256:a1b2c3...               →  sha256:f7e8d9...

Runtime fingerprint changes:
  sglang:           0.4.6.post1                    →  0.4.7
  flashinfer:       0.2.1                          →  0.2.2
  torch:            2.6.0+cu128                    (unchanged)
  triton:           3.2.0                          →  3.2.1

  3 packages added:
    + flashinfer-cuda128==0.2.2
    + cutlass-extensions==3.5.1
    + nixl==0.1.3

  1 package removed:
    - flashinfer-cuda124==0.2.1

  12 packages changed version (use --verbose to list all)

Resources:     (unchanged)
Benchmark:     (unchanged)
Backend config: (unchanged)
```

The diff is structured — it groups changes by category so you can quickly see if the difference is in the software stack, the config, or the hardware.

```bash
# Compare specific sections only
$ srtctl diff outputs/12345 outputs/12400 --only pip
$ srtctl diff outputs/12345 outputs/12400 --only config
$ srtctl diff outputs/12345 outputs/12400 --only resources

# Output as JSON for scripting
$ srtctl diff outputs/12345 outputs/12400 --json

# Compare against a recipe (not a previous run)
$ srtctl diff outputs/12345 recipe.yaml
# Shows what the recipe intends vs what actually ran
```

Under the hood this is straightforward — both lockfiles are YAML, the pip packages are lists, and diffing sorted lists of `package==version` strings is trivial. The hard part is presentation, not computation.

### `srtctl check` — Verify environment matches a fingerprint

This is the "run with a fingerprint" idea. You point `srtctl check` at a lockfile and it tells you whether your current setup can reproduce that run.

```bash
$ srtctl check outputs/12345/recipe.lock.yaml

Checking recipe requirements...
  model.name: deepseek-ai/DeepSeek-R1
    Local path: /lustre/fsw/.../DeepSeek-R1       ✓ exists (100 files, 685GB)
    Revision:   e4e908c07378...                    ✓ matches local snapshot

  container.image: lmsysorg/sglang:v0.4.6.post1
    Local path: /lustre/fsw/.../sglang.sqsh        ✓ exists (4.2GB)
    Digest:     sha256:a1b2c3d4...                 ✓ matches remote

  slurm.account: coreai_tritoninference_triton3    ✓ valid account
  slurm.partition: gb200                           ✓ partition exists

Checking runtime fingerprint...
  python:  3.11.9                                  ✓ match
  torch:   2.6.0+cu128                             ⚠ current container has 2.7.0+cu128
  sglang:  0.4.6.post1                             ⚠ current container has 0.4.7
  cuda:    12.8                                    ✓ match
  driver:  570.86.15                               ✓ match

  pip packages:
    142 match
    3 version mismatches:
      sglang:     0.4.6.post1  →  0.4.7
      flashinfer: 0.2.1        →  0.2.2
      triton:     3.2.0        →  3.2.1
    2 missing packages:
      - flashinfer-cuda124==0.2.1
    1 extra package:
      + flashinfer-cuda128==0.2.2

Summary: environment DIFFERS from fingerprint (5 mismatches)
  To reproduce exactly, use container with digest sha256:a1b2c3d4...
```

How does `srtctl check` actually inspect the container? It runs a quick throwaway srun:

```bash
# srtctl check launches a short-lived container to capture current state
srun --container-image=/lustre/.../sglang.sqsh \
     --container-mounts=/logs:/logs \
     bash -c 'capture_fingerprint /logs/current_fingerprint.json' 

# Then compares current_fingerprint.json against the lockfile's fingerprint
```

This is a ~10 second operation — spin up container, pip freeze, tear down. No GPU allocation needed (just a quick CPU srun on a login/utility node if available, or `--gpus=0`).

**Fault tolerance**: If the srun fails (no nodes available, container not found), `srtctl check` falls back to checking only the recipe-level fields (paths, checksums) and reports that the runtime fingerprint couldn't be verified.

```bash
$ srtctl check outputs/12345/recipe.lock.yaml

Checking recipe requirements...
  model, container, slurm: ✓ all OK

Checking runtime fingerprint...
  ⚠ Could not launch container for verification (srun timed out)
  Skipping pip package comparison
  Recipe-level checks passed. Runtime fingerprint unverified.
```

### `srtctl check --fix` — Guided repair (future)

A natural extension: if `check` finds mismatches, `--fix` tells you exactly how to fix them.

```bash
$ srtctl check outputs/12345/recipe.lock.yaml --fix

3 mismatches found. To match fingerprint:

  1. Container image mismatch
     Current sqsh uses sglang 0.4.7, fingerprint expects 0.4.6.post1
     Fix: enroot import docker://lmsysorg/sglang:v0.4.6.post1
          # or use digest: enroot import docker://lmsysorg/sglang@sha256:a1b2c3d4...

  2. Missing package: flashinfer-cuda124==0.2.1
     Fix: pip install flashinfer-cuda124==0.2.1
          (add to container.pip_install in recipe)

  3. Model revision drift
     Local model was updated since fingerprint was taken.
     Fix: huggingface-cli download deepseek-ai/DeepSeek-R1 \
            --revision e4e908c07378... --local-dir /lustre/fsw/.../DeepSeek-R1
```

This is the "guided new user" experience — someone gets a lockfile, runs `srtctl check --fix`, and gets a step-by-step to reproduce.

---

## Validation and Helpful Errors

Since there's no fallback file, `srtctl` needs to be helpful when things are missing:

### Missing model path
```
Error: model.path is required.

Model: deepseek-ai/DeepSeek-R1
Download it:
  huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir /path/to/DeepSeek-R1

Then add to your recipe:
  model:
    path: "/path/to/DeepSeek-R1"
```

### Missing container path
```
Error: container.path is required.

Container: lmsysorg/sglang:v0.4.6.post1
Import it:
  enroot import docker://lmsysorg/sglang:v0.4.6.post1

Then add to your recipe:
  container:
    path: "/path/to/sglang-v0.4.6.sqsh"
```

### Missing SLURM account
```
Error: slurm.account is required.

Find your accounts:
  sacctmgr show associations user=$USER format=Account%-30

Then add to your recipe:
  slurm:
    account: "your-account"
```

---

## Auto-Detection of Cluster Capabilities

Instead of flags in a config file, srtctl probes the cluster:

| Capability | How to detect |
|------------|---------------|
| `--gpus-per-node` support | `scontrol show config \| grep GresTypes` |
| `--segment` support | `scontrol show config \| grep SchedulerType` |
| GPUs per node | `scontrol show node $NODE \| grep Gres` or from `resources.gpus_per_node` |
| Network interface | `ip route get 1 \| awk '{print $5}'` |

For cluster quirks that can't be auto-detected:

```yaml
cluster:
  gpus_per_node_directive: false   # e.g. Lyris doesn't register GRES
  exclusive: true
```

---

## Fault Tolerance Summary

**Nothing in the validation/fingerprint/checksum system can kill a job.**

| Component | Failure mode | Behavior |
|-----------|-------------|----------|
| HF API check | Network error, timeout, 5xx | Log warning, continue submit |
| Docker registry check | Network error, timeout, auth | Log warning, continue submit |
| Local path check | Permission denied, NFS hang | Log warning, continue submit |
| Fingerprint capture (in container) | Any command fails | `\|\| true` — next probe runs, worker launches |
| `nvidia-smi` in fingerprint | Hangs (bad driver) | `timeout 5` kills it, next probe runs |
| `pip freeze` in fingerprint | pip not installed | Logs "unavailable", next probe runs |
| Lockfile write | Disk full, permission | Log error, job results still in output dir |
| Checksum computation | Large file, slow I/O | Runs async, skipped if too slow |

The pattern is always: **try → log result → continue**. Never: try → fail → abort.

---

## Migration Path

### Phase 1: Add virtual identity fields (backward-compatible)
- Add `model.name`, `model.revision`, `container.image`, `container.digest` as new optional fields
- Keep `srtslurm.yaml` and alias resolution working
- `srtctl apply` warns if virtual identity fields are missing
- Add fingerprint capture to worker preamble
- Start writing lockfiles to output directories

### Phase 2: Make physical paths inline
- Recipes include `model.path` and `container.path` directly
- Alias resolution still works but prints deprecation warning
- SLURM fields become required in recipe
- Add background validation checks
- Add auto-detection for cluster capability flags

### Phase 3: Remove srtslurm.yaml entirely
- Delete `load_cluster_config()` and `resolve_config_with_defaults()`
- Delete `ClusterConfig` schema
- Remove srtslurm.yaml.example
- Update all recipes to be fully self-contained
- `srtctl apply -f recipe.yaml` is the only interface

---

## User Experience

### New user, day 1
```bash
$ cat recipe.yaml
# Can immediately see: model is DeepSeek-R1, container is sglang v0.4.6, needs GB200s

$ srtctl apply -f recipe.yaml
Error: model.path is required.
Model: deepseek-ai/DeepSeek-R1
Download: huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir ...

# Download, set path, repeat for container, set SLURM account. Done.
```

### Sharing a result
```bash
$ cat outputs/12345/recipe.lock.yaml
# Everything: virtual identity, physical paths, checksums, pip freeze, GPU info
```

### Debugging a regression
```bash
$ diff outputs/12345/recipe.lock.yaml outputs/12400/recipe.lock.yaml
# See exactly what changed between two runs:
# - sglang version bumped from 0.4.6 to 0.4.7
# - torch nightly changed
# - NCCL version changed
# All visible in the pip_packages diff
```

### Porting to a new cluster
```bash
# Copy recipe, change only: model.path, container.path, slurm.account, slurm.partition
# Everything else is portable
```
