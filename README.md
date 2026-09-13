# srtctl

Command-line tool for distributed LLM inference benchmarks on SLURM clusters using ATOM, TensorRT LLM, SGLang and vLLM. Replace complex shell scripts and 50+ CLI flags with a declarative `schema: 2` YAML recipe: `engine:` names the engine, `roles:` describes each worker role, and `services:` covers everything launched next to the workers.

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
- [Examples](examples/README.md) - Runnable 2.0 recipes, one per frontend and topology
- [Configuration Reference](docs/config-reference.md) - Every recipe section
- [Legacy (v1) layout](docs/legacy-v1.md) - The old `backend:` recipe layout; `srtctl migrate` rewrites it
- [Monitoring](docs/monitoring.md) - Job logs and debugging
- [Parameter Sweeps](docs/sweeps.md) - Grid searches
- [Profiling](docs/profiling.md) - Torch/nsys profiling
- [Component Performance Dashboard](docs/component-dashboard.md) - the per-run HTML dashboard built from the tachometer parquet
- [ruter](docs/ruter.md) - Dynamo router post-processing

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

# Rewrite a v1 recipe into the 2.0 layout
srtctl migrate -f config.yaml
```
