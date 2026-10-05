# Benchmarks

Runners live here; the shell scripts they launch live under `benchmarks/scripts/` and are excluded from Ruff and `ty`.

## Adding a New Benchmark

1. Create `benchmarks/<name>.py` with a class decorated `@register_benchmark("<type>")` (`benchmarks/base.py`). Subclass `BenchmarkRunner`, or `AIPerfBenchmarkRunner` for an AIPerf-driven benchmark.
2. Implement the abstract members: `name` and `script_path` (properties; the script is under `SCRIPTS_DIR`), `validate_config(config)` returning a list of error strings, and `build_command(...)`. Override `get_container_image`, `get_container_mounts`, or `get_environment` only when the benchmark needs something other than the defaults.
3. Declare the `BenchmarkConfig` fields the runner reads beyond `SHARED_BENCHMARK_FIELDS` in `config_fields` (`benchmark_config_fields()` combines the two). A field set for a type whose runner does not read it is rejected under schema 2 and warned about under schema 1 (`SrtConfig` validation in `core/schema.py`).
4. Add the script under `benchmarks/scripts/<name>/`.
5. Import the module from `benchmarks/__init__.py` so registration runs.
6. Ship it complete: a `tests/test_benchmarks.py` case, an entry under "Available Benchmark Types" in `docs/benchmarks.md`, and regenerated `docs/schema-reference.md` if a field changed.

Frontend-specific behavior (sglang-router `slow_down`, Dynamo request tracing) is one of the few name checks the Design Rules allow; do not add new ones.
