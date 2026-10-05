# CLI and orchestrator

`cli/do_sweep.py` is the `SweepOrchestrator` that runs inside the job; its stages are mixins under `cli/mixins/` (services, workers, frontend, benchmark, telemetry, postprocess). Stages hold no frontend- or backend-name branches: they ask `Frontend` or `Backend` (Design Rules in the root `CLAUDE.md`, enforced by `tests/test_design_rules.py`). `cli/submit.py` owns `srtctl apply` / `dry-run`; any config that reaches srun must show up in `show_config_details()`.

Every launch goes through `srtctl.core.slurm.start_srun_process` with a `step_name`. A module that imports it (or `get_hostname_ip`, `wait_for_health`, `wait_for_http_endpoints`) by name must also be patched in `src/srtctl/mock.py`, or the mock orchestrator and the launch snapshots will try to reach a real cluster.

## Host Setup

`host_setup` runs commands on each node's **bare host, outside the container**, before any
worker starts — the counterpart to `setup_script`, which runs *inside* the container. Use it
for node state a container cannot reach (GPU clocks, kernel modules).

```yaml
host_setup:
  commands: ["sudo -n nvidia-smi -lmc <min>,<max>"]
  teardown: ["sudo -n nvidia-smi -rmc"]   # runs on the cleanup path, success or failure
  nodes: all                              # all | workers
```

Implemented in `SweepOrchestrator._run_host_setup()` / `._run_host_teardown()` as one
container-less `start_srun_process(container_image=None, ...)` per node. Cluster-wide default
lives in `srtslurm.yaml` as `default_host_setup` (whole-block replace, like
`default_health_check`). Commands run as the submitting user, so privileged ones need
passwordless sudo. Always pair a `commands` entry that sets persistent state with a
`teardown` — otherwise it leaks to the next job on that node.
