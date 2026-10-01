---
name: validate-without-cluster
description: Verify an srt-slurm change end to end without SLURM or GPUs, using dry-run, the mock orchestrator, and the launch snapshots.
---

# Validate a change without a cluster

1. **Static checks.** `make lint` (Ruff check and format, blocking `ty check src/srtctl/`), then `uv run srtctl schema-docs --check`.
2. **Dry-run the recipes the change affects.** `uv run srtctl dry-run -f <recipe>` for an example under `examples/` or a recipe from the PR. Confirm the resolved config, mounts, env, srun options, and sbatch script show the change.
3. **Launch snapshots.** `make snapshots-check`. If the change is meant to alter what runs on the cluster, run `make snapshots` and read the diff under `tests/snapshots/launch/`: every changed line is an env var, mount, flag, or placement that will change on real hardware. A recipe whose header shows a non-zero `# exit_code:` failed under the mock; read its error before assuming the mock is wrong.
4. **Behavior.** For readiness, ordering, failure, or cleanup behavior, write a test with `srtctl.mock.run_mock_sweep` (see `tests/test_mock_sweep.py`, `tests/test_pools.py`). `MockOptions` sets the fake nodelist and child durations, and `on_srun` captures every srun call.
5. **A module that launches or resolves hosts itself.** If it imports `start_srun_process`, `get_hostname_ip`, `wait_for_health`, or `wait_for_http_endpoints` by name, add its module-level name to the patch list in `src/srtctl/mock.py`; otherwise the mock run tries to reach a real host.
6. **Full suite.** `make check`.
7. **Report honestly.** In the PR description, say which of these ran and that nothing ran on a cluster.
