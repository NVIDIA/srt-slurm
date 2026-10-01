# Tests

`tests/README.md` describes the suites. Where to add a test:

- Visible config (mounts, env, srun options, services): `test_dry_run.py`.
- Generated `sbatch_script.sh`: `test_render.py`.
- Orchestrator behavior (launch order, readiness, failure, cleanup): the mock orchestrator, `srtctl.mock.run_mock_sweep`, as in `test_mock_sweep.py` and `test_pools.py`.
- A frontend: `test_<type>_frontend.py` with `start_process` as the seam.
- Design Rules: `test_design_rules.py`. Its baselines may only shrink; fix a violation instead of listing it.

## Launch snapshots

`test_launch_snapshots.py` runs every single-job recipe under `examples/` through `run_mock_sweep` and compares each srun call (nodes, env, mounts, preamble, command) to `snapshots/launch/<dir>__<recipe>.txt`. `launch_snapshots.py` normalizes paths and random values so the files are stable. After an intended launch change, run `make snapshots` and commit the diff; reviewers read it as "what now runs on the cluster". A new example gets a snapshot the same way. A recipe that fails under the mock (`# exit_code:` other than 0) usually means a module imports a SLURM or network helper that `src/srtctl/mock.py` does not patch yet.

## Mocking SLURM

Unit tests patch the SLURM environment and `scontrol` directly:

```python
with patch.dict(os.environ, H100Rack.slurm_env()):
    with patch("subprocess.run", H100Rack.mock_scontrol()):
        ...
```

`test_e2e.py` defines the rack fixtures (`GB200NVLRack`, `H100Rack`).
