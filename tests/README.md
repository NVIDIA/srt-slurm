# Tests

Unit tests for srtctl configuration, allocation, and command generation.

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_e2e.py -v

# Run single test
uv run pytest tests/test_e2e.py::TestH100Cluster::test_endpoint_allocation -v

# With coverage
uv run pytest tests/ --cov=srtctl
```

## Test Structure

### `test_e2e.py` - Cluster-Style E2E Tests

Tests the full allocation flow with mocked SLURM environments. Defines rack fixtures for different hardware:

- **`GB200NVLRack`**: 18 nodes × 4 GPUs = 72 total GPUs
- **`H100Rack`**: 13 nodes × 8 GPUs = 104 total GPUs

**Test Classes:**

- `TestGB200FP4Cluster` - Tests GB200 configs (4 GPUs/node)
- `TestH100Cluster` - Tests H100 configs (8 GPUs/node)
- `TestCIConfigs` - Tests CI configs (`ci/agg.yaml`, `ci/disagg.yaml`)

### `test_endpoint_allocation.py` - GPU Allocation Logic

Tests the core `allocate_endpoints` and `endpoints_to_processes` functions:

- Node assignment for prefill/decode/aggregated workers
- GPU slicing (multiple workers per node)
- `CUDA_VISIBLE_DEVICES` generation
- Port assignment

### `test_health.py` - Health Check Parsing

Tests health check response parsing for different backends:

- `check_dynamo_health` - Dynamo `/metrics` response parsing
- `check_static_router_health` - SGLang/vLLM Router `/workers` response parsing
- Error handling for malformed responses
- Aggregated mode (workers count as decode)

### `test_command_generation.py` - SGLang Command Building

Tests SGLang command generation from YAML configs:

- Disaggregated mode (prefill/decode commands)
- Aggregated mode (combined workers)
- Environment variable handling
- Profiling mode configuration

### `test_configs.py` - Config Loading

Tests YAML config loading and validation:

- Schema validation
- Default resolution from `srtslurm.yaml`
- Model/container path resolution

### `test_benchmarks.py` - Benchmark Runners

Tests benchmark runner implementations:

- BenchmarkRunner ABC inheritance
- Script path resolution
- Config validation

### `test_profiling.py` - Profiling

Tests profiling configuration, validation, and benchmark runner:

- `ProfilingConfig` and `ProfilingPhaseConfig` dataclasses
- Per-phase start_step/stop_step environment variables
- Validation: requires 1P+1D (disagg) or 1 agg worker
- Validation: phase configs must match serving mode
- Auto-switch to profiling benchmark when `profiling.enabled`
- Profiling runner and script

### `test_process_registry.py` - Process Management

Tests ProcessRegistry lifecycle management:

- Process registration/deregistration
- Failure detection
- Cleanup handling

## Mocking SLURM

Tests use mock SLURM environments to avoid cluster dependencies:

```python
class H100Rack:
    """H100 SLURM rack: 13 nodes × 8 GPUs = 104 total GPUs."""
    NUM_NODES = 13
    GPUS_PER_NODE = 8

    @classmethod
    def slurm_env(cls) -> dict[str, str]:
        return {
            "SLURM_JOB_ID": "67890",
            "SLURM_NODELIST": f"h100-[01-{cls.NUM_NODES:02d}]",
            "SLURM_JOB_NUM_NODES": str(cls.NUM_NODES),
            ...
        }

    @classmethod
    def mock_scontrol(cls):
        """Mock subprocess.run for scontrol hostnames."""
        def mock_run(cmd, **kwargs):
            if cmd[0] == "scontrol" and "hostnames" in cmd:
                result = MagicMock()
                result.stdout = "\n".join(cls.nodes())
                return result
            raise subprocess.CalledProcessError(1, cmd)
        return mock_run

# Usage in tests
with patch.dict(os.environ, H100Rack.slurm_env()):
    with patch("subprocess.run", H100Rack.mock_scontrol()):
        endpoints = allocate_endpoints(...)
```

## Test Philosophy

1. **Isolation**: No external dependencies (SLURM, containers, models)
2. **Focused**: Each test verifies one specific behavior
3. **Fast**: All tests run in < 5 seconds
4. **Realistic**: Use actual config files from `configs/` and `ci/`

## Adding New Tests

### For New Config Options

```python
def test_new_sglang_flag():
    from srtctl.backends import SGLangBackendConfig, SGLangServerConfig

    config = SGLangBackendConfig(
        sglang_config=SGLangServerConfig(
            prefill={"my-new-flag": "value"}
        )
    )
    flags = config.sglang_config.prefill
    assert "my-new-flag" in flags
```

### For New Allocation Logic

```python
def test_new_allocation_mode():
    endpoints = allocate_endpoints(
        num_prefill=2, num_decode=4, num_agg=0,
        gpus_per_prefill=8, gpus_per_decode=4, gpus_per_agg=8,
        gpus_per_node=8,
        available_nodes=("n0", "n1", "n2", "n3"),
    )
    assert len(endpoints) == 6  # 2 prefill + 4 decode
```

### For New Health Check Backends

```python
def test_new_backend_health():
    response = {"workers": [...], "stats": {...}}
    result = check_new_backend_health(
        response, expected_prefill=2, expected_decode=4
    )
    assert result.ready
```
