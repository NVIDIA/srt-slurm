# Telemetry Setup (sgl-gb200)

End-to-end guide for getting GPU + node + scraper telemetry running on a benchmark job.

## 0. Prereqs

- A GitHub Personal Access Token with `read:packages` scope, [SSO-authorized for `NVIDIA-dev`](https://github.com/settings/tokens) (the scraper image is gated)

You'll end up with three sqsh files:

- `dcgm-exporter` (public, 351M) — GPU metrics
- `node-exporter` (public, 23M) — CPU / IB / mem metrics
- `tachometer.sqsh` (gated, 150M) — the scraper that polls all `/metrics` endpoints and writes parquet

## 1. One-time enroot credentials

enroot needs ghcr.io creds to pull the gated scraper image. Public images (dcgm, node-exporter) don't need this, but it's harmless to set up first.

```bash
ssh sgl-gb200
mkdir -p ~/.config/enroot
chmod 700 ~/.config/enroot

cat > ~/.config/enroot/.credentials <<'EOF'
machine ghcr.io login YOUR_GITHUB_USERNAME password YOUR_GITHUB_PAT
EOF

chmod 600 ~/.config/enroot/.credentials
```

If you skip the SSO-authorize step on the PAT settings page, `enroot import` will fail with `401 Unauthorized` — go back and click "Configure SSO" → authorize for `NVIDIA-dev`.

## 2. Pull the three sqsh files

GB200 is aarch64; all three images are multi-arch, so enroot picks the right manifest automatically. Naming follows the `+`-separated convention used elsewhere in the cluster's container library.

```bash
cd /mnt/lustre01/users/slurm-shared/ishan

# dcgm-exporter (public)
enroot import -o nvcr.io+nvidia+k8s+dcgm-exporter+4.2.0-4.1.0-ubuntu22.04.sqsh \
  docker://nvcr.io/nvidia/k8s/dcgm-exporter:4.2.0-4.1.0-ubuntu22.04

# node_exporter (public)
enroot import -o prom+node-exporter+v1.9.0.sqsh \
  docker://prom/node-exporter:v1.9.0

# scraper (gated, needs the credentials from step 1)
enroot import -o tachometer.sqsh \
  docker://ghcr.io/nvidia-dev/warnold-tachometer-scraper:latest
```

Pin `:latest` to a specific commit if you want a stable reference (the GH workflow tags every push as `:sha-<full-commit>`).

Verify all three landed:

```bash
ls -lh /mnt/lustre01/users/slurm-shared/ishan/{nvcr.io+nvidia+k8s+dcgm-exporter+*.sqsh,prom+node-exporter+*.sqsh,tachometer.sqsh}
```

## 3. Register aliases in `srtslurm.yaml`

Open `/mnt/lustre01/users/slurm-shared/ishan/srtslurm/srtslurm.yaml` and add these three lines under the existing `containers:` block. Aliases are optional — absolute paths also work in the recipe directly:

```yaml
containers:
  # ... existing entries ...
  "dcgm-exporter": "/mnt/lustre01/users/slurm-shared/ishan/nvcr.io+nvidia+k8s+dcgm-exporter+4.2.0-4.1.0-ubuntu22.04.sqsh"
  "node-exporter": "/mnt/lustre01/users/slurm-shared/ishan/prom+node-exporter+v1.9.0.sqsh"
  "telemetry-scraper": "/mnt/lustre01/users/slurm-shared/ishan/tachometer.sqsh"
```

After this, any recipe can refer to them by alias.

## 4. Smoke-test the binaries

Before a real benchmark, confirm enroot loads each image and the entrypoints exist:

```bash
srun --container-image=/mnt/lustre01/users/slurm-shared/ishan/nvcr.io+nvidia+k8s+dcgm-exporter+4.2.0-4.1.0-ubuntu22.04.sqsh \
     --pty bash -c 'dcgm-exporter --version'

srun --container-image=/mnt/lustre01/users/slurm-shared/ishan/prom+node-exporter+v1.9.0.sqsh \
     --pty bash -c '/bin/node_exporter --version'

srun --container-image=/mnt/lustre01/users/slurm-shared/ishan/tachometer.sqsh \
     --pty bash -c '/usr/local/bin/tachometer-scraper --help | head -5'
```

If any one of those errors, fix it before going further — the orchestrator can't recover from a missing binary at runtime.

## 5. Add telemetry to a recipe

Drop this block into any recipe yaml. The `binary_path` override is required because the upstream binary is named `tachometer-scraper`, but the schema default is `/usr/local/bin/telemetry-scraper`.

```yaml
telemetry:
  enabled: true
  container_image: telemetry-scraper
  binary_path: /usr/local/bin/tachometer-scraper
  default_frequency: 5.0 # scrape every 5s
  sync_interval_secs: 120 # parquet flush cadence
  compaction_threads: 4
  storage_subdir: telemetry # → <log_dir>/telemetry/local/
  extra_metadata:
    cluster: sgl-gb200
    experiment: my-sweep # any free-form metadata flows into every parquet row
  dcgm_exporter:
    container_image: dcgm-exporter
    port: 9401
  node_exporter:
    container_image: node-exporter
    port: 9101
```

## 6. Verify before submitting

```bash
srtctl dry-run -f your-recipe.yaml
```

Look for:

- An "Execution Extensions" panel showing `telemetry / container_image / storage_subdir / frequency`
- Preflight passing with no `telemetry-container-not-available` errors

If preflight flags a missing sqsh, fix the path before submitting — the alias likely doesn't match anything in `srtslurm.yaml`.

## 7. Submit and watch

```bash
srtctl submit -f your-recipe.yaml
```

After the job starts, three things appear under `outputs/<job_id>/logs/<run_dir>/`:

- `telemetry_dcgm_exporter.out` — dcgm-exporter srun across all worker nodes
- `telemetry_node_exporter.out` — node_exporter srun across all worker nodes
- `telemetry.out` — the scraper srun on the head node
- `telemetry_config.toml` — the generated TOML listing every `/metrics` URL being polled
- `telemetry/local/*.parquet` — the actual telemetry data, written every `sync_interval_secs`

Quick sanity check via `srun --jobid=<id> --pty bash` on a worker:

```bash
curl http://localhost:9401/metrics | head    # dcgm
curl http://localhost:9101/metrics | head    # node
```

## 8. Where the data ends up

After the run completes, srt-slurm's postprocess stage uploads the entire log dir to S3 (per `reporting.s3` in `srtslurm.yaml`). The parquets ride along, ending up at:

```
s3://sgl-gb200-logs/ssh/sglgb200/<job_id>/logs/<run_dir>/telemetry/local/*.parquet
```

Point the tachometer web UI (`ghcr.io/nvidia-dev/warnold-tachometer-web`, run off-cluster) at that bucket to browse runs.

## Common gotchas

- **`binary_path` must be set** to `/usr/local/bin/tachometer-scraper`. Without the override, the scraper srun exits with `command not found` because the schema default doesn't match the upstream binary name.
- **GH PAT needs SSO authorization** for the gated image — the most common cause of `401 Unauthorized` on the scraper pull.
- **Cluster aliases don't help bare `srun --container-image=`** — only `srtctl` recipes resolve them. For ad-hoc `srun` smoke tests, use the absolute sqsh path.
- **Disk pressure**: lustre on sgl-gb200 was at 94% last check. The three sqsh files together are ~525M; not huge, but worth a `df -h` before importing.
