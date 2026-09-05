# srtctl

Command-line tool for distributed LLM inference benchmarks on SLURM clusters and direct GPU hosts using TensorRT LLM, SGLang and vLLM. Replace complex shell scripts and 50+ CLI flags with declarative YAML configuration.

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-org/srtctl.git
cd srtctl
pip install -e .

# One-time setup (downloads NATS/ETCD, creates srtslurm.yaml)
make setup ARCH=aarch64  # or ARCH=x86_64
```

## Documentation

**Full documentation:** https://srtctl.gitbook.io/srtctl-docs/

- [Installation](docs/installation.md) - Setup and configuration
- [Monitoring](docs/monitoring.md) - Job logs and debugging
- [Parameter Sweeps](docs/sweeps.md) - Grid searches
- [Profiling](docs/profiling.md) - Torch/nsys profiling
- [Analyzing Results](docs/analyzing.md) - Dashboard and visualization
- [ruter](docs/ruter.md) - Dynamo router post-processing
- [Direct Host Lifecycle](docs/direct-host.md) - Run the same recipe directly through Docker

## Commands

```bash
# Submit job(s)
srtctl apply -f config.yaml

# Deploy an inference endpoint without running a benchmark
srtctl apply -f config.yaml --serve-only

# Submit with custom setup script
srtctl apply -f config.yaml --setup-script custom-setup.sh

# Submit with tags for filtering
srtctl apply -f config.yaml --tags experiment,baseline

# Dry-run (validate without submitting)
srtctl dry-run -f config.yaml

# Render and run one single-node recipe through Docker
srtctl apply -f config.yaml -o /absolute/path/to/runs --bash > job.sh
chmod +x job.sh
./job.sh

# Launch analysis dashboard
uv run streamlit run analysis/dashboard/app.py
```
