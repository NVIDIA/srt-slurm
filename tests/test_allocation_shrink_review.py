# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial review of ``release_nodes_before_postprocess``: cases the contract suites cannot see.

- ``test_allocation_shrink.py`` drives ``shrink_job_allocation`` in isolation, and
  ``test_release_idle_nodes.py`` replaces it with fakes that never touch ``os.environ``.
  The sweep tests here run the real helper (against ``FakeSlurm``) inside a mock sweep, so
  they see what post-processing reads from the environment the release rewrites.
- Tests 1-4 were defects found in review and are now regression tests. Each asserts its
  preconditions first, so a pass means the behavior changed, not the setup.
- A test still marked ``xfail(strict=True)`` is an open defect; a fix turns it into an
  XPASS failure, and the marker must go.
- Slurm references are SchedMD/slurm 521370b4.
"""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from test_allocation_shrink import NODES, FakeSlurm, run_shrink, slurm_env
from test_release_idle_nodes import EVIDENCE, JOB_ID, TWO_NODES, _allocation_shrink, _read_evidence, _run_sweep

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core import processes as processes_module
from srtctl.core import slurm as slurm_module
from srtctl.core.processes import ProcessRegistry, setup_signal_handlers
from srtctl.core.slurm import _kept_node_cpus

# What sbatch exports to a 2-node batch script, on top of the four variables the mock sets.
TWO_NODE_BATCH_ENV = {
    "SLURM_JOB_NODELIST": ",".join(TWO_NODES),
    "SLURM_JOB_NUM_NODES": "2",
    "SLURM_JOB_CPUS_PER_NODE": "64(x2)",
    "SLURM_TASKS_PER_NODE": "1(x2)",
    "SLURM_NTASKS": "2",
    "SLURM_NPROCS": "2",
}
SLURM_ENV_KEYS = (*TWO_NODE_BATCH_ENV, "SLURM_NODELIST", "SLURM_NNODES")


def _real_shrink_against(fake: FakeSlurm, calls: list[dict[str, str | None]] | None = None):
    """A ``shrink_job_allocation`` seam that runs the real helper against ``fake``.

    ``subprocess.run`` is patched only for the duration of the shrink call, so the rest of the
    sweep (rollup, git, the fake sruns) is unaffected. The real helper rewrites ``os.environ``
    exactly as it would inside a Slurm batch script. ``calls`` receives the Slurm environment
    as it stood right after each shrink.
    """
    real = slurm_module.shrink_job_allocation

    def factory(events: list, _registries: list[ProcessRegistry], _nodelist: tuple[str, ...]):
        def shrink(job_id: str, keep: str, *, timeout: float = 30.0):
            with patch("srtctl.core.slurm.subprocess.run", fake):
                result = real(job_id, keep, timeout=timeout)
            env_after = {key: os.environ.get(key) for key in SLURM_ENV_KEYS}
            if calls is not None:
                calls.append(env_after)
            events.append(("shrink", {"job_id": job_id, "keep": keep, "env_after": env_after}))
            return result

        return shrink

    return factory


def _recording_shrink(attempts: list[str]):
    """A seam that records every attempt and answers like a successful release."""

    def factory(events: list, _registries: list[ProcessRegistry], nodelist: tuple[str, ...]):
        def shrink(job_id: str, keep: str, *, timeout: float = 30.0):
            attempts.append(keep)
            events.append(("shrink", {"job_id": job_id, "keep": keep}))
            return _allocation_shrink(
                requested=True,
                succeeded=True,
                retained=keep,
                released=tuple(node for node in nodelist if node != keep),
                verified=True,
                reason=None,
                duration_s=0.1,
            )

        return shrink

    return factory


@pytest.fixture
def restore_signal_handlers():
    """``SweepOrchestrator.run`` installs SIGTERM/SIGINT handlers process-wide; put pytest's back."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    yield
    for sig, handler in previous.items():
        signal.signal(sig, handler)


# ---------------------------------------------------------------------------
# 1. The final lockfile records the allocation the run used
# ---------------------------------------------------------------------------


def test_final_lockfile_records_the_allocation_the_run_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in TWO_NODE_BATCH_ENV.items():
        monkeypatch.setenv(key, value)

    fake = FakeSlurm(
        job_id=JOB_ID, job_nodes=list(TWO_NODES), batch_host=TWO_NODES[0], cpus=dict.fromkeys(TWO_NODES, 64)
    )
    env_after_release: list[dict[str, str | None]] = []
    sweep = _run_sweep(tmp_path, monkeypatch, release=True, shrink=_real_shrink_against(fake, env_after_release))

    # Preconditions: the real helper ran, Slurm (the fake) shrank the job, and os.environ was
    # rewritten to the one-node allocation before the upload -- the exact setup that broke the lockfile.
    evidence = _read_evidence(sweep.log_dir)
    assert evidence["succeeded"] is True and evidence["verified"] is True, evidence
    assert fake.job_nodes == [TWO_NODES[0]]
    assert len(env_after_release) == 1
    assert env_after_release[0]["SLURM_JOB_NUM_NODES"] == "1"
    assert env_after_release[0]["SLURM_JOB_NODELIST"] == TWO_NODES[0]
    assert env_after_release[0]["SLURM_NTASKS"] is None
    assert sweep.only("shrink") < sweep.only("srun", "upload")
    assert list(sweep.srun("upload")["nodelist"]) == [TWO_NODES[0]]

    lock = yaml.safe_load((sweep.log_dir.parent / "recipe.lock.yaml").read_text())["lock"]["slurm"]
    assert lock.get("num_nodes") == "2", lock
    assert lock.get("nodelist") == ",".join(TWO_NODES), lock
    assert lock.get("cpus_per_node") == "64(x2)", lock
    assert lock.get("num_tasks") == "2", lock


# ---------------------------------------------------------------------------
# 2. scontrol exit 0 is not a release when squeue shows the old nodes
# ---------------------------------------------------------------------------


class _ReqNodesNoOpSlurm(FakeSlurm):
    """``scontrol update NodeList=<x>`` where ``<x>`` equals the job's submitted ``--nodelist``.

    slurmctld treats it as "new req_nodes identical to old req_nodes" and changes nothing,
    yet returns SLURM_SUCCESS (src/slurmctld/job_mgr.c:13314-13317). scontrol then exits 0
    and writes the resize script for the unchanged allocation (src/scontrol/update_job.c:1182-1186).
    A job submitted with ``sbatch -N 4 -w <batch host>`` (``sbatch_directives: {nodelist: ...}``)
    hits this: Slurm fills the other three nodes, and the release asks for exactly that name.
    """

    submitted_req_nodes: str = NODES[0]

    def _scontrol_update(self, cmd, kwargs):
        tags = {key.lower(): value for key, _, value in (arg.partition("=") for arg in cmd[2:])}
        if tags.get("nodelist") == self.submitted_req_nodes:
            # No-op on the controller: replay it as an update to the job's current nodes.
            cmd = [*cmd[:2], f"JobId={tags.get('jobid', '')}", f"NodeList={','.join(self.job_nodes)}"]
        return super()._scontrol_update(cmd, kwargs)


def test_squeue_showing_the_old_nodes_is_not_a_release() -> None:
    fake = _ReqNodesNoOpSlurm()
    env = slurm_env()
    result, after = run_shrink(fake, env, NODES[0])

    # Preconditions: scontrol was asked once, exited 0 (the no-op), and squeue answered with every node.
    assert len(fake.update_calls()) == 1
    assert fake.job_nodes == NODES
    assert len(fake.squeue_calls()) == 1
    assert result.requested is True

    assert result.succeeded is False
    assert result.released == (), f"nodes reported released while squeue shows them held: {result.released}"
    assert result.verified is False
    assert result.reason and "squeue" in result.reason
    assert after == env, "a refused release must leave os.environ untouched"


def test_squeue_runs_before_the_environment_is_rewritten() -> None:
    """squeue now gates the env rewrite, so it runs with the pre-shrink env (harmless: ``squeue -j`` ignores it)."""

    class Recording(FakeSlurm):
        seen_at_squeue: list[str | None]

        def _squeue(self, cmd, kwargs):
            self.seen_at_squeue.append(os.environ.get("SLURM_NNODES"))
            return super()._squeue(cmd, kwargs)

    fake = Recording()
    fake.seen_at_squeue = []
    result, after = run_shrink(fake, slurm_env(), NODES[0])
    assert result.succeeded is True and result.verified is True
    assert fake.seen_at_squeue == ["4"]
    assert after["SLURM_NNODES"] == "1"
    # The squeue call carries no env override, so nothing it does depends on SLURM_NODELIST.
    assert all(call.kwargs.get("env") is None for call in fake.squeue_calls())


@pytest.mark.parametrize("squeue_mode", ["fail", "raise", "timeout"])
def test_squeue_unavailable_after_exit_0_still_applies_the_resize(squeue_mode: str) -> None:
    fake = FakeSlurm(squeue_mode=squeue_mode)
    result, after = run_shrink(fake, slurm_env(), NODES[0])
    assert fake.job_nodes == [NODES[0]]
    assert result.succeeded is True
    assert result.verified is False
    assert result.released == tuple(NODES[1:])
    assert after["SLURM_NODELIST"] == NODES[0]
    assert after["SLURM_NNODES"] == "1"


# ---------------------------------------------------------------------------
# 3. The fallback SLURM_JOB_CPUS_PER_NODE is the kept node's own count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fake_kwargs", "env_overrides"),
    [
        # Client timed out after slurmctld applied the shrink, before scontrol wrote the script.
        ({"update_timeout": "after_apply"}, {}),
        # scontrol writes no script without SLURM_JOBID (update_job.c:1282-1283).
        ({}, {"SLURM_JOBID": None}),
    ],
    ids=["timeout-after-apply", "no-SLURM_JOBID"],
)
def test_fallback_cpus_per_node_is_the_kept_nodes_count(
    fake_kwargs: dict[str, Any], env_overrides: dict[str, str | None]
) -> None:
    keep = NODES[1]
    fake = FakeSlurm(batch_host=keep, cpus=dict(zip(NODES, [72, 64, 64, 64], strict=True)), **fake_kwargs)
    env = slurm_env(SLURM_JOB_CPUS_PER_NODE="72,64(x3)", SLURMD_NODENAME=keep, **env_overrides)
    result, after = run_shrink(fake, env, keep, timeout=5.0)

    # Preconditions: the shrink happened and scontrol left no resize script, so the fallback ran.
    assert result.succeeded is True
    assert fake.job_nodes == [keep]
    assert all("slurm_job_" not in " ".join(listing) for listing in fake.update_cwd_listing)
    assert after["SLURM_NODELIST"] == keep
    # What Slurm's own resize script would say for the kept node (its 64 CPUs, not nodelist[0]'s 72).
    assert after["SLURM_JOB_CPUS_PER_NODE"] == "64"


@pytest.mark.parametrize(
    ("compressed", "index", "expected"),
    [
        ("64(x4)", 3, "64"),
        ("72,64(x3)", 0, "72"),
        ("72,64(x3)", 3, "64"),
        ("16,32(x2),8", 3, "8"),
        ("128(x2),64(x2)", 2, "64"),
        ("64(x1),32(x1),16(x1),8(x1)", 2, "16"),
        (" 72, 64(x3) ", 1, "64"),
    ],
)
def test_kept_node_cpus_expands_every_compressed_form(compressed: str, index: int, expected: str) -> None:
    assert _kept_node_cpus(compressed, NODES, NODES[index]) == expected


@pytest.mark.parametrize(
    "compressed",
    [
        "64(x3)",  # three entries for four nodes
        "64(x5)",  # five entries for four nodes
        "64(x)",  # malformed repeat
        "64,",  # trailing empty group
        "",  # empty
    ],
)
def test_kept_node_cpus_falls_back_to_the_first_group_on_a_mismatch(compressed: str) -> None:
    """Never raises: a value that does not map one-to-one onto the nodelist keeps the old first-group answer."""
    assert _kept_node_cpus(compressed, NODES, NODES[2]) == compressed.split(",")[0].split("(")[0].strip()


def test_kept_node_cpus_for_a_node_outside_the_nodelist_is_the_first_group() -> None:
    assert _kept_node_cpus("72,64(x3)", NODES, "elsewhere-01") == "72"


# ---------------------------------------------------------------------------
# 4. A signal-driven shutdown (walltime, scancel, Ctrl-C) skips the release
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_signal_driven_cleanup_does_not_attempt_a_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_signal_handlers: None, signum: signal.Signals
) -> None:
    attempts: list[str] = []

    def signal_mid_run(self: SweepOrchestrator) -> None:
        # Delivered synchronously to the handler setup_signal_handlers() installed in run():
        # stop_event.set(), registry.cleanup(), sys.exit(1) -- then run()'s finally block executes.
        signal.raise_signal(signum)

    with patch.object(SweepOrchestrator, "_print_connection_info", signal_mid_run):
        sweep = _run_sweep(tmp_path, monkeypatch, release=True, shrink=_recording_shrink(attempts))

    # Preconditions: the real handler ended the run with sys.exit(1) (caught by _run_sweep), the
    # cleanup block still reached post-processing (the evidence is written inside run_postprocess)
    # with the signal passed down, and the mock restored its environment.
    assert sweep.system_exit == 1 and sweep.exit_code is None
    postprocess = sweep.events[sweep.only("run_postprocess")][1]
    assert postprocess["kwargs"].get("received_signal") == signal.Signals(signum).name, postprocess
    evidence = _read_evidence(sweep.log_dir)
    assert os.environ.get("SLURM_JOB_ID") != JOB_ID, "mock env leaked"

    assert attempts == [], "a shrink was attempted on a job Slurm has already ended"
    assert evidence["requested"] is False and evidence["succeeded"] is False
    assert signal.Signals(signum).name in (evidence["reason"] or ""), evidence


def test_received_signal_is_recorded_by_the_handler(restore_signal_handlers: None) -> None:
    registry = ProcessRegistry(job_id="1")
    stop = threading.Event()
    setup_signal_handlers(stop, registry)
    assert registry.received_signal is None
    with pytest.raises(SystemExit):
        signal.raise_signal(signal.SIGINT)
    assert registry.received_signal == "SIGINT"
    assert stop.is_set()


# ---------------------------------------------------------------------------
# 5. A job stopped by the process monitor (critical worker failure) still releases
# ---------------------------------------------------------------------------


# The real monitor ends its thread with sys.exit(1); pytest reports that SystemExit as unhandled.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_process_monitor_failure_still_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_signal_handlers: None
) -> None:
    """The monitor stops the run from its own thread without a signal; the job is still RUNNING, so release."""
    attempts: list[str] = []
    captured: dict[str, Any] = {}
    real_setup = processes_module.setup_signal_handlers

    def capture_setup(stop_event: threading.Event, registry: ProcessRegistry) -> None:
        captured["stop_event"] = stop_event
        captured["registry"] = registry
        real_setup(stop_event, registry)

    def critical_failure(self: SweepOrchestrator) -> None:
        # The real monitor loop (not the MagicMock _run_sweep installs) sees a critical failure:
        # stop_event.set(), registry.cleanup(), sys.exit(1) -- all on the monitor thread.
        registry: ProcessRegistry = captured["registry"]
        registry.check_failures = lambda: True  # type: ignore[method-assign]
        thread = processes_module.start_process_monitor(captured["stop_event"], registry, poll_interval=0.01)
        thread.join(timeout=10)
        assert not thread.is_alive()

    with patch.object(SweepOrchestrator, "_print_connection_info", critical_failure):
        sweep = _run_sweep(
            tmp_path,
            monkeypatch,
            release=True,
            benchmark_exit=1,
            shrink=_recording_shrink(attempts),
            signal_handlers=capture_setup,
        )

    # Preconditions: the monitor really stopped the run, and no signal was involved.
    assert captured["stop_event"].is_set()
    assert captured["registry"].received_signal is None
    assert sweep.system_exit is None
    assert sweep.exit_code == 1

    assert attempts == [TWO_NODES[0]]
    assert sweep.only("shrink") < sweep.only("srun", "upload")
    assert list(sweep.srun("upload")["nodelist"]) == [TWO_NODES[0]]
    assert list(sweep.srun("ai")["nodelist"]) == [TWO_NODES[0]]
    assert _read_evidence(sweep.log_dir)["succeeded"] is True


# ---------------------------------------------------------------------------
# 6. AI analysis without S3 (exit != 0) is still preceded by the release
# ---------------------------------------------------------------------------


def test_ai_analysis_without_s3_runs_on_the_retained_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[str] = []
    sweep = _run_sweep(
        tmp_path, monkeypatch, release=True, benchmark_exit=7, s3=False, shrink=_recording_shrink(attempts)
    )

    # Preconditions: S3 is not configured and no S3 upload ran, so only the AI analysis follows the release.
    reporting = yaml.safe_load((tmp_path / "srtslurm.yaml").read_text())["reporting"]
    assert "s3" not in reporting and reporting["ai_analysis"]["enabled"] is True
    assert not sweep.indices("srun", "upload")
    assert sweep.exit_code == 7

    assert attempts == [TWO_NODES[0]]
    ai_at = sweep.only("srun", "ai")
    assert sweep.only("shrink") < ai_at
    assert list(sweep.srun("ai")["nodelist"]) == [TWO_NODES[0]]
    assert sweep.srun("ai")["het_group"] is None
    assert (sweep.log_dir / EVIDENCE).is_file()


# ---------------------------------------------------------------------------
# 7. Everything run_postprocess does before the release is node-local
# ---------------------------------------------------------------------------


def test_nothing_before_the_release_launches_an_srun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Between the teardown and the release, run_postprocess (rollup, lockfile, energy report) starts no step."""
    attempts: list[str] = []
    sweep = _run_sweep(tmp_path, monkeypatch, release=True, shrink=_recording_shrink(attempts))
    shrink_at = sweep.only("shrink")
    last_teardown = max(sweep.indices("teardown_done"))
    between = [payload for event, payload in sweep.events[last_teardown + 1 : shrink_at] if event == "srun"]
    assert between == []
    # The lockfile and energy report the release now follows are already on disk when the upload starts.
    assert (sweep.log_dir.parent / "recipe.lock.yaml").is_file()
