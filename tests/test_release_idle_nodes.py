# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Releasing idle nodes before post-processing (``release_nodes_before_postprocess``).

Orchestrator, config and evidence-file contract. The Slurm helper itself
(``shrink_job_allocation``) is covered elsewhere; here it and ``get_batch_host``
are seams patched where ``postprocess_stage`` imports them by name.

Mock sweeps record one event timeline from the fake srun, the fake shrink, and
wrappers around the cleanup/teardown hooks, so ordering is proved from a single
sequence rather than from timestamps.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import signal
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

import srtctl.mock as mock_module
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins import postprocess_stage
from srtctl.core.processes import ProcessRegistry
from srtctl.core.schema import ClusterConfig
from srtctl.core.status import StatusReporter
from srtctl.mock import MockOptions, run_mock_sweep

STAGE = "srtctl.cli.mixins.postprocess_stage"
JOB_ID = "77001"
EVIDENCE = "allocation-release.json"
EVIDENCE_KEYS = {"requested", "succeeded", "retained", "released", "verified", "reason", "duration_s"}

TWO_NODES = ("mock-node-01", "mock-node-02")
THREE_NODES = ("mock-node-01", "mock-node-02", "mock-node-03")

RECIPE: dict[str, Any] = {
    "schema": 2,
    "name": "release-idle-nodes",
    "model": {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "fp8"},
    "resources": {"gpu_type": "h100", "gpus_per_node": 8},
    "engine": "sglang",
    "roles": {"agg": {"nodes": 2, "workers": 1, "gpus": 16}},
    "benchmark": {"type": "custom", "command": "echo fake-benchmark"},
    # Teardown on every node, so a release that runs before it would strand sruns on released nodes.
    "host_setup": {"nodes": "all", "commands": ["true"], "teardown": ["echo teardown"]},
}

# etcd/nats on a dedicated node: infra takes nodelist[0] (the batch host), head moves to nodelist[1].
DEDICATED_INFRA_SERVICES = [
    {"name": "etcd", "type": "etcd", "placement": {"node": "dedicated"}},
    {"name": "nats", "type": "nats", "placement": {"node": "dedicated"}},
]


def _allocation_shrink(**fields: Any):
    # Imported lazily so each test fails on the missing name, not the whole module at collection.
    from srtctl.core.slurm import AllocationShrink

    return AllocationShrink(**fields)


def _write_cluster_config(tmp_path: Path, *, release: bool | None, extra: dict | None = None, s3: bool = True) -> Path:
    """srtslurm.yaml with S3 upload and AI analysis configured, so both post-processing sruns run."""
    data: dict[str, Any] = {
        "reporting": {
            "ai_analysis": {"enabled": True, "openrouter_api_key": "sk-or-mock", "gh_token": "ghp-mock"},
        }
    }
    if s3:
        data["reporting"]["s3"] = {"bucket": "mock-bucket", "prefix": "mock"}
    if release is not None:
        data["release_nodes_before_postprocess"] = release
    data.update(extra or {})
    path = tmp_path / "srtslurm.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _write_recipe(tmp_path: Path, *, dedicated_infra: bool = False) -> Path:
    recipe = deepcopy(RECIPE)
    if dedicated_infra:
        recipe["services"] = deepcopy(DEDICATED_INFRA_SERVICES)
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path


def _srun_kind(kwargs: dict[str, Any]) -> str:
    name = Path(kwargs.get("output") or "").name
    if name == "postprocess.log":
        return "upload"
    if name == "ai_analysis.log":
        return "ai"
    if name.startswith("host_teardown_"):
        return "teardown"
    if name.startswith("host_setup_"):
        return "setup"
    return "srun"


class _QuickPostprocessPopen(mock_module.FakePopen):
    """Workers, frontends and services run for the whole test (so cleanup is what stops them);
    the one-shot host_setup, upload and AI-analysis children exit at once (they are waited on)."""

    def __init__(self, *, cmd: list[str], output: str | None, duration_s: float) -> None:
        if _srun_kind({"output": output}) != "srun":
            duration_s = 0.0
        super().__init__(cmd=cmd, output=output, duration_s=duration_s)


@dataclass
class _Sweep:
    exit_code: int | None
    events: list[tuple[str, Any]]
    log_dir: Path
    registries: list[ProcessRegistry] = field(default_factory=list)
    batch_host_fake: Any = None
    # SystemExit code when a signal handler ended the run (the real handler calls sys.exit(1)).
    system_exit: Any = None

    def indices(self, name: str, kind: str | None = None) -> list[int]:
        return [
            i
            for i, (event, payload) in enumerate(self.events)
            if event == name and (kind is None or _srun_kind(payload) == kind)
        ]

    def only(self, name: str, kind: str | None = None) -> int:
        found = self.indices(name, kind)
        assert len(found) == 1, f"expected exactly one {name}/{kind} event, got {len(found)}"
        return found[0]

    def srun(self, kind: str) -> dict[str, Any]:
        return self.events[self.only("srun", kind)][1]

    def sruns_after(self, index: int) -> list[dict[str, Any]]:
        return [payload for event, payload in self.events[index + 1 :] if event == "srun"]


def _ok_shrink(sweep_events: list, registries: list[ProcessRegistry], nodelist: tuple[str, ...]):
    def fake(job_id: str, keep: str, *, timeout: float = 30.0):
        processes = [proc for registry in registries for proc in registry.get_all_processes().values()]
        sweep_events.append(
            (
                "shrink",
                {
                    "job_id": job_id,
                    "keep": keep,
                    "processes": len(processes),
                    "still_running": sorted(proc.name for proc in processes if proc.is_running),
                },
            )
        )
        return _allocation_shrink(
            requested=True,
            succeeded=True,
            retained=keep,
            released=tuple(node for node in nodelist if node != keep),
            verified=True,
            reason=None,
            duration_s=0.25,
        )

    return fake


def _failed_shrink(sweep_events: list):
    def fake(job_id: str, keep: str, *, timeout: float = 30.0):
        sweep_events.append(("shrink", {"job_id": job_id, "keep": keep}))
        return _allocation_shrink(
            requested=True,
            succeeded=False,
            retained=keep,
            released=(),
            verified=False,
            reason="scontrol: error: Job resizing not supported (disable_job_shrink)",
            duration_s=0.1,
        )

    return fake


def _raising_shrink(sweep_events: list):
    def fake(job_id: str, keep: str, *, timeout: float = 30.0):
        sweep_events.append(("shrink", {"job_id": job_id, "keep": keep}))
        raise RuntimeError("scontrol exploded")

    return fake


def _run_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    release: bool | None,
    nodelist: tuple[str, ...] = TWO_NODES,
    dedicated_infra: bool = False,
    benchmark_exit: int = 0,
    shrink: Callable[[list, list[ProcessRegistry], tuple[str, ...]], Any] | None = None,
    batch_host: Any = None,
    patch_slurm_seams: bool = True,
    s3: bool = True,
    benchmark: Callable[..., int] | None = None,
    serve_only: bool = False,
    end_manual_hold: bool = False,
    signal_handlers: Any = None,
) -> _Sweep:
    """Run the real orchestrator under the mock with an event timeline.

    ``shrink`` builds the fake ``shrink_job_allocation`` from (events, registries, nodelist).
    The fakes are installed INSIDE ``mock_infrastructure`` so they win over whatever the mock
    itself patches onto the same names.

    ``benchmark`` replaces ``run_benchmark`` (default: return ``benchmark_exit``); pass the real
    ``SweepOrchestrator.run_benchmark`` to run it. ``signal_handlers`` replaces
    ``setup_signal_handlers`` (default: the real one, which installs real handlers).
    """
    monkeypatch.setenv("SRTSLURM_CONFIG", str(_write_cluster_config(tmp_path, release=release, s3=s3)))
    # The variables the final lockfile records; monkeypatch restores whatever a fake release rewrites.
    monkeypatch.setenv("SLURM_JOB_NODELIST", ",".join(nodelist))
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", str(len(nodelist)))
    recipe = _write_recipe(tmp_path, dedicated_infra=dedicated_infra)
    output_dir = tmp_path / "outputs" / JOB_ID
    events: list[tuple[str, Any]] = []
    registries: list[ProcessRegistry] = []

    shrink_fake = (shrink or _ok_shrink)(events, registries, nodelist)
    batch_host_fake = batch_host if batch_host is not None else MagicMock(return_value=nodelist[0])

    real_infrastructure = mock_module.mock_infrastructure

    @contextlib.contextmanager
    def infrastructure_with_seams(*, options: MockOptions, output_dir: Path) -> Iterator[Any]:
        with real_infrastructure(options=options, output_dir=output_dir) as sink:
            if not patch_slurm_seams:
                yield sink
                return
            with (
                patch(f"{STAGE}.get_batch_host", batch_host_fake),
                patch(f"{STAGE}.shrink_job_allocation", shrink_fake),
            ):
                yield sink

    real_cleanup = ProcessRegistry.cleanup
    real_failure_details = ProcessRegistry.print_failure_details
    real_teardown = SweepOrchestrator._run_host_teardown
    real_incremental = SweepOrchestrator.finalize_incremental_power_report
    real_completed = StatusReporter.report_completed
    real_lockfile = postprocess_stage.write_lockfile
    real_compare = SweepOrchestrator._compare_against_previous_lock
    real_energy = SweepOrchestrator._build_power_energy_report
    real_postprocess = SweepOrchestrator.run_postprocess

    def lockfile(*args: Any, **kwargs: Any) -> bool:
        written = real_lockfile(*args, **kwargs)
        events.append(("final_lockfile_written", os.environ.get("SLURM_JOB_NODELIST")))
        return written

    def compare(self: SweepOrchestrator) -> None:
        real_compare(self)
        events.append(("compare_lock_done", None))

    def energy(self: SweepOrchestrator) -> None:
        real_energy(self)
        events.append(("energy_report_done", None))

    def postprocess(self: SweepOrchestrator, exit_code: int, *args: Any, **kwargs: Any) -> None:
        events.append(("run_postprocess", {"exit_code": exit_code, "args": args, "kwargs": kwargs}))
        real_postprocess(self, exit_code, *args, **kwargs)

    def cleanup(self: ProcessRegistry) -> None:
        if self not in registries:
            registries.append(self)
        real_cleanup(self)
        events.append(("cleanup_done", None))

    def failure_details(self: ProcessRegistry, *args: Any, **kwargs: Any) -> None:
        real_failure_details(self, *args, **kwargs)
        events.append(("failure_details_done", None))

    def teardown(self: SweepOrchestrator) -> None:
        real_teardown(self)
        events.append(("teardown_done", None))

    def incremental(self: SweepOrchestrator) -> None:
        real_incremental(self)
        events.append(("incremental_power_done", None))

    def completed(self: StatusReporter, exit_code: int, logs_url: str | None = None) -> bool:
        events.append(("report_completed", exit_code))
        return real_completed(self, exit_code, logs_url=logs_url)

    def on_srun(kwargs: dict[str, Any]) -> None:
        if _srun_kind(kwargs) == "upload":
            # The evidence must already be on disk when the upload of the log directory starts.
            events.append(("evidence_at_upload", (output_dir / "logs" / EVIDENCE).is_file()))
        events.append(("srun", kwargs))

    run_benchmark = patch.object(SweepOrchestrator, "run_benchmark", return_value=benchmark_exit)
    if benchmark is not None:
        run_benchmark = patch.object(SweepOrchestrator, "run_benchmark", benchmark)
    handlers = (
        patch("srtctl.cli.do_sweep.setup_signal_handlers", signal_handlers)
        if signal_handlers is not None
        else contextlib.nullcontext()
    )
    orchestrator_class = (
        patch("srtctl.cli.do_sweep.SweepOrchestrator", functools.partial(SweepOrchestrator, serve_only=True))
        if serve_only
        else contextlib.nullcontext()
    )
    system_exit = None
    with (
        patch.object(mock_module, "mock_infrastructure", infrastructure_with_seams),
        patch.object(mock_module, "FakePopen", _QuickPostprocessPopen),
        patch.object(ProcessRegistry, "cleanup", cleanup),
        patch.object(ProcessRegistry, "print_failure_details", failure_details),
        patch.object(SweepOrchestrator, "_run_host_teardown", teardown),
        patch.object(SweepOrchestrator, "finalize_incremental_power_report", incremental),
        run_benchmark,
        patch.object(StatusReporter, "report_completed", completed),
        patch.object(postprocess_stage, "write_lockfile", lockfile),
        patch.object(SweepOrchestrator, "_compare_against_previous_lock", compare),
        patch.object(SweepOrchestrator, "_build_power_energy_report", energy),
        patch.object(SweepOrchestrator, "run_postprocess", postprocess),
        patch("srtctl.cli.do_sweep.start_process_monitor"),
        handlers,
        orchestrator_class,
    ):
        options = MockOptions(
            nodelist=nodelist,
            child_duration_s=600,
            phase_pause_s=0,
            on_srun=on_srun,
            end_manual_hold=end_manual_hold,
        )
        try:
            exit_code = run_mock_sweep(config_path=recipe, output_dir=output_dir, job_id=JOB_ID, options=options)
        except SystemExit as exc:
            exit_code = None
            system_exit = exc.code

    return _Sweep(
        exit_code=exit_code,
        events=events,
        log_dir=output_dir / "logs",
        registries=registries,
        batch_host_fake=batch_host_fake,
        system_exit=system_exit,
    )


def _read_evidence(log_dir: Path) -> dict[str, Any]:
    path = log_dir / EVIDENCE
    assert path.is_file(), f"{EVIDENCE} missing from {log_dir}"
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def _assert_utc_timestamp(value: Any, *, not_before: datetime, not_after: datetime) -> None:
    assert isinstance(value, str)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None, f"timestamp {value!r} has no UTC offset"
    assert parsed.utcoffset() == timedelta(0), f"timestamp {value!r} is not UTC"
    assert not_before - timedelta(seconds=5) <= parsed <= not_after + timedelta(seconds=5)


# ---------------------------------------------------------------------------
# Setting off (the default)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("release", [None, False], ids=["absent", "explicit-false"])
@pytest.mark.parametrize("benchmark_exit", [0, 7])
def test_setting_off_never_touches_the_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release: bool | None, benchmark_exit: int
) -> None:
    shrink = MagicMock(name="shrink_job_allocation")
    batch_host = MagicMock(name="get_batch_host", return_value=TWO_NODES[0])
    sweep = _run_sweep(
        tmp_path,
        monkeypatch,
        release=release,
        benchmark_exit=benchmark_exit,
        shrink=lambda *_: shrink,
        batch_host=batch_host,
    )

    assert sweep.exit_code == benchmark_exit
    shrink.assert_not_called()
    batch_host.assert_not_called()
    assert not (sweep.log_dir / EVIDENCE).exists()

    upload = sweep.srun("upload")
    assert list(upload["nodelist"]) == [TWO_NODES[0]]
    assert upload["het_group"] is None
    if benchmark_exit:
        ai = sweep.srun("ai")
        assert list(ai["nodelist"]) == [TWO_NODES[0]]
        assert ai["het_group"] is None


def test_setting_off_with_dedicated_infra_uploads_from_head_not_batch_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shrink = MagicMock(name="shrink_job_allocation")
    sweep = _run_sweep(
        tmp_path, monkeypatch, release=None, nodelist=THREE_NODES, dedicated_infra=True, shrink=lambda *_: shrink
    )
    shrink.assert_not_called()
    # Unchanged behavior: the upload follows runtime.nodes.head, which a dedicated infra node moves to nodelist[1].
    assert list(sweep.srun("upload")["nodelist"]) == [THREE_NODES[1]]


# ---------------------------------------------------------------------------
# Setting on, successful release
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("benchmark_exit", [0, 7])
def test_release_runs_after_teardown_and_before_postprocess_sruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, benchmark_exit: int
) -> None:
    started = datetime.now(timezone.utc)
    sweep = _run_sweep(tmp_path, monkeypatch, release=True, benchmark_exit=benchmark_exit)
    finished = datetime.now(timezone.utc)

    assert sweep.exit_code == benchmark_exit
    shrink_at = sweep.only("shrink")
    shrink_call = sweep.events[shrink_at][1]
    assert shrink_call["job_id"] == JOB_ID
    assert shrink_call["keep"] == TWO_NODES[0]

    # Every worker, frontend and service the registry held was stopped before the release.
    assert sweep.registries, "registry.cleanup() never ran"
    assert shrink_call["processes"] > 0
    assert shrink_call["still_running"] == []

    before = [
        *sweep.indices("cleanup_done"),
        *sweep.indices("srun", "teardown"),
        *sweep.indices("teardown_done"),
        *sweep.indices("incremental_power_done"),
        *sweep.indices("failure_details_done"),
    ]
    assert sweep.indices("cleanup_done") and sweep.indices("teardown_done")
    assert sorted({_srun_kind(sweep.events[i][1]) for i in sweep.indices("srun", "teardown")}) == ["teardown"]
    assert len(sweep.indices("srun", "teardown")) == len(TWO_NODES), "teardown ran on every node"
    if benchmark_exit:
        assert sweep.indices("failure_details_done")
    assert max(before) < shrink_at

    after = [sweep.only("srun", "upload"), sweep.only("report_completed")]
    if benchmark_exit:
        after.append(sweep.only("srun", "ai"))
    else:
        assert not sweep.indices("srun", "ai")
    assert shrink_at < min(after)

    # Nothing is launched on a released node once the release happened.
    for kwargs in sweep.sruns_after(shrink_at):
        assert kwargs.get("nodelist") is not None, f"srun without an explicit nodelist after release: {kwargs}"
        assert list(kwargs["nodelist"]) == [TWO_NODES[0]], f"srun on a released node: {kwargs.get('output')}"
        assert kwargs.get("het_group") is None

    assert sweep.events[sweep.only("report_completed")][1] == benchmark_exit
    assert sweep.events[sweep.only("evidence_at_upload")][1] is True

    evidence = _read_evidence(sweep.log_dir)
    assert EVIDENCE_KEYS | {"job_id", "timestamp"} <= set(evidence)
    assert evidence["job_id"] == JOB_ID
    assert evidence["requested"] is True
    assert evidence["succeeded"] is True
    assert evidence["retained"] == TWO_NODES[0]
    assert evidence["released"] == [TWO_NODES[1]]
    assert evidence["verified"] is True
    _assert_utc_timestamp(evidence["timestamp"], not_before=started, not_after=finished)


def test_release_keeps_the_batch_host_when_head_is_elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sweep = _run_sweep(
        tmp_path, monkeypatch, release=True, nodelist=THREE_NODES, dedicated_infra=True, benchmark_exit=3
    )
    assert sweep.exit_code == 3

    shrink_at = sweep.only("shrink")
    assert sweep.events[shrink_at][1]["keep"] == THREE_NODES[0], "the batch host, not runtime.nodes.head"
    sweep.batch_host_fake.assert_called()

    for kind in ("upload", "ai"):
        kwargs = sweep.srun(kind)
        assert list(kwargs["nodelist"]) == [THREE_NODES[0]], f"{kind} must run on the retained batch host"
        assert kwargs["het_group"] is None
    for kwargs in sweep.sruns_after(shrink_at):
        assert set(kwargs["nodelist"]) <= {THREE_NODES[0]}

    evidence = _read_evidence(sweep.log_dir)
    assert evidence["retained"] == THREE_NODES[0]
    assert evidence["released"] == list(THREE_NODES[1:])


# ---------------------------------------------------------------------------
# Setting on, release fails or raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["failed", "raised"])
@pytest.mark.parametrize("benchmark_exit", [0, 7])
def test_failed_release_changes_nothing_but_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, benchmark_exit: int
) -> None:
    builder = (lambda ev, *_: _failed_shrink(ev)) if outcome == "failed" else (lambda ev, *_: _raising_shrink(ev))
    sweep = _run_sweep(
        tmp_path,
        monkeypatch,
        release=True,
        nodelist=THREE_NODES,
        dedicated_infra=True,
        benchmark_exit=benchmark_exit,
        shrink=builder,
    )

    assert sweep.exit_code == benchmark_exit
    shrink_at = sweep.only("shrink")
    assert sweep.events[sweep.only("report_completed")][1] == benchmark_exit

    # Unchanged targeting: the head node (nodelist[1] here) with its usual het group.
    upload_at = sweep.only("srun", "upload")
    assert shrink_at < upload_at
    upload = sweep.events[upload_at][1]
    assert list(upload["nodelist"]) == [THREE_NODES[1]]
    assert upload["het_group"] is None
    if benchmark_exit:
        assert list(sweep.srun("ai")["nodelist"]) == [THREE_NODES[1]]

    evidence = _read_evidence(sweep.log_dir)
    assert evidence["job_id"] == JOB_ID
    assert evidence["succeeded"] is False
    assert evidence["released"] == []
    assert evidence["reason"]
    expected = "disable_job_shrink" if outcome == "failed" else "scontrol exploded"
    assert expected in evidence["reason"]
    assert "timestamp" in evidence


def test_exception_from_batch_host_lookup_is_contained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shrink = MagicMock(name="shrink_job_allocation")
    sweep = _run_sweep(
        tmp_path,
        monkeypatch,
        release=True,
        benchmark_exit=0,
        shrink=lambda *_: shrink,
        batch_host=MagicMock(side_effect=OSError("SLURMD_NODENAME unreadable")),
    )
    assert sweep.exit_code == 0
    shrink.assert_not_called()
    assert list(sweep.srun("upload")["nodelist"]) == [TWO_NODES[0]]
    assert sweep.indices("report_completed")


# ---------------------------------------------------------------------------
# The mock's own fake (srtctl/mock.py patches the helpers)
# ---------------------------------------------------------------------------


def test_mock_sweep_fakes_the_release_without_running_scontrol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    real_run = subprocess.run
    slurm_calls: list[list[str]] = []

    def guarded_run(cmd, *args, **kwargs):
        argv = [str(part) for part in (cmd if isinstance(cmd, (list, tuple)) else [cmd])]
        if argv and Path(argv[0]).name == "scontrol" and "update" in argv:
            slurm_calls.append(argv)
            raise AssertionError(f"mock sweep ran a real scontrol update: {argv}")
        if argv and Path(argv[0]).name == "squeue" and any("%N" in part for part in argv):
            slurm_calls.append(argv)
            raise AssertionError(f"mock sweep ran a real squeue verification: {argv}")
        return real_run(cmd, *args, **kwargs)

    with patch("srtctl.core.slurm.subprocess.run", guarded_run):
        sweep = _run_sweep(tmp_path, monkeypatch, release=True, patch_slurm_seams=False)

    assert slurm_calls == []
    assert sweep.exit_code == 0
    evidence = _read_evidence(sweep.log_dir)
    assert evidence["succeeded"] is True
    assert evidence["retained"] == TWO_NODES[0]
    assert list(sweep.srun("upload")["nodelist"]) == [TWO_NODES[0]]


# ---------------------------------------------------------------------------
# release_idle_nodes() and _postprocess_node() called directly
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    release: bool | None,
    nodelist: tuple[str, ...] = TWO_NODES,
    dedicated_infra: bool = False,
) -> Iterator[SweepOrchestrator]:
    """A real SweepOrchestrator built from a recipe and srtslurm.yaml, without running it."""
    from srtctl.core.config import load_config
    from srtctl.core.runtime import RuntimeContext

    monkeypatch.setenv("SRTSLURM_CONFIG", str(_write_cluster_config(tmp_path, release=release)))
    output_dir = tmp_path / "outputs" / JOB_ID
    monkeypatch.setenv("SRTCTL_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("SLURM_JOB_ID", JOB_ID)
    monkeypatch.setenv("SLURM_JOBID", JOB_ID)
    monkeypatch.setenv("SLURM_NODELIST", ",".join(nodelist))
    monkeypatch.setenv("SLURM_NNODES", str(len(nodelist)))
    with mock_module.mock_infrastructure(options=MockOptions(nodelist=nodelist), output_dir=output_dir):
        config = load_config(_write_recipe(tmp_path, dedicated_infra=dedicated_infra))
        runtime = RuntimeContext.from_config(config, job_id=JOB_ID)
        yield SweepOrchestrator(config=config, runtime=runtime)


def _het(orchestrator: SweepOrchestrator) -> SweepOrchestrator:
    nodes = orchestrator.runtime.nodes
    het_nodes = replace(nodes, het=True, prefill_group=nodes.worker[:1], decode_group=nodes.worker[1:])
    return SweepOrchestrator(config=orchestrator.config, runtime=replace(orchestrator.runtime, nodes=het_nodes))


def test_release_idle_nodes_returns_none_when_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shrink = MagicMock()
    batch_host = MagicMock(return_value=TWO_NODES[0])
    with (
        _orchestrator(tmp_path, monkeypatch, release=False) as orch,
        patch(f"{STAGE}.shrink_job_allocation", shrink),
        patch(f"{STAGE}.get_batch_host", batch_host),
        patch("subprocess.run", MagicMock(side_effect=AssertionError("no subprocess when the setting is off"))),
    ):
        assert orch.release_idle_nodes() is None
        assert orch._postprocess_node() == (TWO_NODES[0], None)
    shrink.assert_not_called()
    batch_host.assert_not_called()
    assert not (orch.runtime.log_dir / EVIDENCE).exists()


def test_release_success_retargets_postprocess_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    events: list = []
    started = datetime.now(timezone.utc)
    with (
        _orchestrator(tmp_path, monkeypatch, release=True, nodelist=THREE_NODES, dedicated_infra=True) as orch,
        patch(f"{STAGE}.get_batch_host", return_value=THREE_NODES[0]),
        patch(f"{STAGE}.shrink_job_allocation", side_effect=_ok_shrink(events, [], THREE_NODES)) as shrink,
        caplog.at_level(logging.INFO),
    ):
        assert orch.runtime.nodes.head == THREE_NODES[1]
        assert orch._postprocess_node() == (THREE_NODES[1], None), "before a release: head, as today"
        result = orch.release_idle_nodes()
        assert orch._postprocess_node() == (THREE_NODES[0], None)
    finished = datetime.now(timezone.utc)

    shrink.assert_called_once()
    args, kwargs = shrink.call_args
    bound = dict(zip(("job_id", "keep"), args, strict=False)) | kwargs
    assert bound["job_id"] == JOB_ID
    assert bound["keep"] == THREE_NODES[0]

    assert result is not None and result.succeeded is True
    evidence = _read_evidence(orch.runtime.log_dir)
    assert {k: v for k, v in evidence.items() if k not in ("job_id", "timestamp")} == result.to_dict()
    assert evidence["job_id"] == JOB_ID
    _assert_utc_timestamp(evidence["timestamp"], not_before=started, not_after=finished)

    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO and r.name == STAGE]
    assert any(THREE_NODES[1] in message and THREE_NODES[2] in message for message in info), info


def test_release_failure_logs_warning_and_keeps_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        _orchestrator(tmp_path, monkeypatch, release=True, nodelist=THREE_NODES, dedicated_infra=True) as orch,
        patch(f"{STAGE}.get_batch_host", return_value=THREE_NODES[0]),
        patch(f"{STAGE}.shrink_job_allocation", side_effect=_failed_shrink([])),
        caplog.at_level(logging.INFO),
    ):
        result = orch.release_idle_nodes()
        assert orch._postprocess_node() == (THREE_NODES[1], None)

    assert result is not None and result.succeeded is False
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name == STAGE]
    assert any("disable_job_shrink" in message for message in warnings), warnings
    assert _read_evidence(orch.runtime.log_dir)["succeeded"] is False


def test_release_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with (
        _orchestrator(tmp_path, monkeypatch, release=True) as orch,
        patch(f"{STAGE}.get_batch_host", return_value=TWO_NODES[0]),
        patch(f"{STAGE}.shrink_job_allocation", side_effect=_raising_shrink([])),
    ):
        result = orch.release_idle_nodes()
        assert orch._postprocess_node() == (TWO_NODES[0], None)
    assert result is None or result.succeeded is False
    evidence = _read_evidence(orch.runtime.log_dir)
    assert evidence["succeeded"] is False
    assert "scontrol exploded" in evidence["reason"]


def test_heterogeneous_job_is_skipped_with_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    shrink = MagicMock()
    with (
        _orchestrator(tmp_path, monkeypatch, release=True) as plain,
        patch(f"{STAGE}.get_batch_host", return_value=TWO_NODES[0]),
        patch(f"{STAGE}.shrink_job_allocation", shrink),
        caplog.at_level(logging.INFO),
    ):
        orch = _het(plain)
        result = orch.release_idle_nodes()
        # Het jobs keep today's targeting, het group included.
        assert orch._postprocess_node() == (orch.runtime.nodes.head, 0)

    shrink.assert_not_called()
    assert result is None or (result.requested is False and result.succeeded is False)
    info = [r.getMessage().lower() for r in caplog.records if r.levelno == logging.INFO and r.name == STAGE]
    assert any("het" in message for message in info), info
    evidence = _read_evidence(orch.runtime.log_dir)
    assert evidence["requested"] is False
    assert evidence["succeeded"] is False
    assert evidence["released"] == []
    assert "het" in (evidence["reason"] or "").lower()
    assert evidence["job_id"] == JOB_ID


def test_het_job_with_setting_off_keeps_het_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _orchestrator(tmp_path, monkeypatch, release=False) as plain:
        orch = _het(plain)
        assert orch._postprocess_node() == (orch.runtime.nodes.head, 0)
        assert orch.release_idle_nodes() is None


@pytest.mark.parametrize("missing", ["job_id", "batch_host"])
def test_no_job_id_or_batch_host_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    shrink = MagicMock()
    with (
        _orchestrator(tmp_path, monkeypatch, release=True) as orch,
        patch(f"{STAGE}.get_batch_host", return_value=None if missing == "batch_host" else TWO_NODES[0]),
        patch(f"{STAGE}.shrink_job_allocation", shrink),
    ):
        if missing == "job_id":
            # Neither the runtime nor the environment names a Slurm job.
            orch = SweepOrchestrator(config=orch.config, runtime=replace(orch.runtime, job_id=""))
            monkeypatch.delenv("SLURM_JOB_ID")
            monkeypatch.delenv("SLURM_JOBID")
        result = orch.release_idle_nodes()
        assert orch._postprocess_node() == (TWO_NODES[0], None)
    shrink.assert_not_called()
    assert result is None or result.succeeded is False
    evidence = _read_evidence(orch.runtime.log_dir)
    assert evidence["requested"] is False
    assert evidence["succeeded"] is False
    assert evidence["reason"]


def test_release_reads_the_setting_from_runtime_not_srtslurm_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load = MagicMock(return_value={"release_nodes_before_postprocess": False})
    with (
        _orchestrator(tmp_path, monkeypatch, release=True) as orch,
        patch(f"{STAGE}.get_batch_host", return_value=TWO_NODES[0]),
        patch(f"{STAGE}.shrink_job_allocation", side_effect=_ok_shrink([], [], TWO_NODES)) as shrink,
    ):
        assert orch.runtime.release_nodes_before_postprocess is True
        # srtslurm.yaml changing (or vanishing) after start must not matter: the runtime read it once.
        monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "gone.yaml"))
        with (
            patch("srtctl.core.config.load_cluster_config", load),
            patch(f"{STAGE}.load_cluster_config", load),
        ):
            orch.release_idle_nodes()
    load.assert_not_called()
    shrink.assert_called_once()

    # And the inverse: a runtime that says off is off, whatever srtslurm.yaml says now.
    shrink_off = MagicMock()
    (tmp_path / "off").mkdir()
    with (
        _orchestrator(tmp_path / "off", monkeypatch, release=None) as orch,
        patch(f"{STAGE}.shrink_job_allocation", shrink_off),
        patch(f"{STAGE}.get_batch_host", return_value=TWO_NODES[0]),
    ):
        assert orch.runtime.release_nodes_before_postprocess is False
        _write_cluster_config(tmp_path / "off", release=True)
        assert orch.release_idle_nodes() is None
    shrink_off.assert_not_called()


# ---------------------------------------------------------------------------
# Config: ClusterConfig and RuntimeContext
# ---------------------------------------------------------------------------


def test_cluster_config_field_defaults_false_and_loads_true() -> None:
    assert ClusterConfig().release_nodes_before_postprocess is False
    assert ClusterConfig.Schema().load({}).release_nodes_before_postprocess is False
    loaded = ClusterConfig.Schema().load(yaml.safe_load("release_nodes_before_postprocess: true\n"))
    assert loaded.release_nodes_before_postprocess is True


@pytest.mark.parametrize("value", ["sometimes", 7, [True], {"enabled": True}])
def test_cluster_config_rejects_non_bool(value: Any) -> None:
    with pytest.raises(ValidationError):
        ClusterConfig.Schema().load({"release_nodes_before_postprocess": value})


def test_load_cluster_config_round_trips_the_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from srtctl.core.config import get_srtslurm_setting, load_cluster_config

    monkeypatch.setenv("SRTSLURM_CONFIG", str(_write_cluster_config(tmp_path, release=True)))
    cluster = load_cluster_config()
    assert cluster is not None and cluster["release_nodes_before_postprocess"] is True
    assert get_srtslurm_setting("release_nodes_before_postprocess", False) is True


@pytest.mark.parametrize(
    ("release", "expected"),
    [(None, False), (False, False), (True, True)],
    ids=["absent", "false", "true"],
)
def test_runtime_context_reflects_srtslurm_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release: bool | None, expected: bool
) -> None:
    with _orchestrator(tmp_path, monkeypatch, release=release) as orch:
        assert orch.runtime.release_nodes_before_postprocess is expected


def test_runtime_context_defaults_false_without_srtslurm_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _orchestrator(tmp_path, monkeypatch, release=True) as orch:
        pass
    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent.yaml"))
    from srtctl.core.runtime import RuntimeContext

    with mock_module.mock_infrastructure(options=MockOptions(nodelist=TWO_NODES), output_dir=tmp_path / "o2"):
        runtime = RuntimeContext.from_config(orch.config, job_id=JOB_ID)
    assert runtime.release_nodes_before_postprocess is False
    # A dataclass field with a False default, not a property that re-reads srtslurm.yaml.
    assert "release_nodes_before_postprocess" in {f for f in RuntimeContext.__dataclass_fields__}
    assert RuntimeContext.__dataclass_fields__["release_nodes_before_postprocess"].default is False


def test_runtime_context_reads_srtslurm_yaml_once_through_get_srtslurm_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from srtctl.core import runtime as runtime_module

    seen: list[str] = []
    real = runtime_module.get_srtslurm_setting

    def recording(key: str, default: Any = None) -> Any:
        seen.append(key)
        return real(key, default)

    with (
        patch.object(runtime_module, "get_srtslurm_setting", recording),
        _orchestrator(tmp_path, monkeypatch, release=True) as orch,
    ):
        assert orch.runtime.release_nodes_before_postprocess is True
    assert seen.count("release_nodes_before_postprocess") == 1


# ---------------------------------------------------------------------------
# Release inside run_postprocess: after the final lockfile and energy report
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_signal_handlers() -> Iterator[None]:
    # The real orchestrator installs SIGTERM/SIGINT handlers bound to its registry; keep them out of other tests.
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _env_rewriting_shrink(sweep_events: list, registries: list[ProcessRegistry], nodelist: tuple[str, ...]):
    """A successful release that rewrites the job's SLURM_* environment the way Slurm's resize script does."""
    ok = _ok_shrink(sweep_events, registries, nodelist)

    def fake(job_id: str, keep: str, *, timeout: float = 30.0):
        result = ok(job_id, keep, timeout=timeout)
        os.environ.update(SLURM_NODELIST=keep, SLURM_JOB_NODELIST=keep, SLURM_NNODES="1", SLURM_JOB_NUM_NODES="1")
        return result

    return fake


def _postprocess_kwargs(sweep: _Sweep) -> dict[str, Any]:
    return sweep.events[sweep.only("run_postprocess")][1]["kwargs"]


@pytest.mark.parametrize("benchmark_exit", [0, 7])
def test_release_follows_final_lockfile_and_energy_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, benchmark_exit: int
) -> None:
    sweep = _run_sweep(tmp_path, monkeypatch, release=True, benchmark_exit=benchmark_exit, shrink=_env_rewriting_shrink)
    assert sweep.exit_code == benchmark_exit

    shrink_at = sweep.only("shrink")
    lockfile_at = sweep.only("final_lockfile_written")
    assert sweep.only("run_postprocess") < lockfile_at < shrink_at
    assert sweep.only("compare_lock_done") < shrink_at
    assert sweep.only("energy_report_done") < shrink_at
    assert shrink_at < sweep.only("srun", "upload")
    if benchmark_exit:
        assert shrink_at < sweep.only("srun", "ai")

    # The final lockfile describes the allocation the run used, not the one left after the release.
    full = ",".join(TWO_NODES)
    assert sweep.events[lockfile_at][1] == full
    lock = yaml.safe_load((sweep.log_dir.parent / "recipe.lock.yaml").read_text())["lock"]
    assert lock["slurm"]["nodelist"] == full
    assert lock["slurm"]["num_nodes"] == str(len(TWO_NODES))


def test_ai_analysis_only_path_releases_before_the_ai_srun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sweep = _run_sweep(
        tmp_path,
        monkeypatch,
        release=True,
        nodelist=THREE_NODES,
        dedicated_infra=True,
        benchmark_exit=5,
        s3=False,
    )
    assert sweep.exit_code == 5
    assert not sweep.indices("srun", "upload"), "no reporting.s3: no upload srun"
    shrink_at = sweep.only("shrink")
    ai_at = sweep.only("srun", "ai")
    assert sweep.only("final_lockfile_written") < shrink_at < ai_at
    ai = sweep.events[ai_at][1]
    assert list(ai["nodelist"]) == [THREE_NODES[0]]
    assert ai["het_group"] is None
    assert _read_evidence(sweep.log_dir)["succeeded"] is True


# ---------------------------------------------------------------------------
# How the job stopped: process monitor (release) vs. signal (skip)
# ---------------------------------------------------------------------------


# The real monitor thread ends with sys.exit(1) after its cleanup, which pytest reports from the thread.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_job_stopped_by_process_monitor_still_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from srtctl.core.processes import start_process_monitor

    seen: dict[str, Any] = {}

    def critical_worker_dies(self, registry, stop_event, reporter=None) -> int:
        workers = [proc for name, proc in registry.get_all_processes().items() if name.startswith("agg_")]
        assert workers and all(proc.critical for proc in workers)
        workers[0].popen.kill()
        # The real monitor loop: detects the failure, sets stop_event, cleans up, exits its thread.
        start_process_monitor(stop_event, registry, poll_interval=0.01).join(timeout=30)
        seen.update(stopped=stop_event.is_set(), signal=registry.received_signal, failed=registry.has_failures)
        return 1

    sweep = _run_sweep(
        tmp_path,
        monkeypatch,
        release=True,
        nodelist=THREE_NODES,
        dedicated_infra=True,
        benchmark=critical_worker_dies,
    )

    assert seen == {"stopped": True, "signal": None, "failed": True}
    assert sweep.system_exit is None
    assert sweep.exit_code == 1
    assert _postprocess_kwargs(sweep).get("received_signal") is None
    shrink_at = sweep.only("shrink")
    assert sweep.events[shrink_at][1]["keep"] == THREE_NODES[0]
    for kind in ("upload", "ai"):
        kwargs = sweep.srun(kind)
        assert list(kwargs["nodelist"]) == [THREE_NODES[0]]
        assert kwargs["het_group"] is None
    assert _read_evidence(sweep.log_dir)["succeeded"] is True


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_job_stopped_by_signal_skips_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sig) -> None:
    shrink = MagicMock(name="shrink_job_allocation")

    def signalled(self, registry, stop_event, reporter=None) -> int:
        # Delivered to this process, so the handler setup_signal_handlers installed is the one that runs.
        signal.raise_signal(sig)
        pytest.fail("the installed signal handler did not end the run")

    sweep = _run_sweep(
        tmp_path,
        monkeypatch,
        release=True,
        nodelist=THREE_NODES,
        dedicated_infra=True,
        benchmark=signalled,
        shrink=lambda *_: shrink,
    )

    assert sweep.system_exit == 1, "the handler ends the job with sys.exit(1)"
    assert sweep.registries and sweep.registries[0].received_signal == sig.name
    assert _postprocess_kwargs(sweep).get("received_signal") == sig.name
    shrink.assert_not_called()

    evidence = _read_evidence(sweep.log_dir)
    assert evidence["requested"] is False
    assert evidence["succeeded"] is False
    assert evidence["released"] == []
    assert sig.name in evidence["reason"]
    assert evidence["job_id"] == JOB_ID

    # Unchanged post-processing on the head node (nodelist[1] with a dedicated infra node).
    assert sweep.events[sweep.only("evidence_at_upload")][1] is True
    for kind in ("upload", "ai"):
        kwargs = sweep.srun(kind)
        assert list(kwargs["nodelist"]) == [THREE_NODES[1]]
        assert kwargs["het_group"] is None
    assert sweep.indices("report_completed")


def test_received_signal_defaults_to_none() -> None:
    assert ProcessRegistry(job_id=JOB_ID).received_signal is None


def test_mocked_signal_handlers_leave_received_signal_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handlers = MagicMock(name="setup_signal_handlers")
    sweep = _run_sweep(tmp_path, monkeypatch, release=True, signal_handlers=handlers)
    handlers.assert_called_once()
    assert sweep.exit_code == 0
    assert sweep.registries and all(registry.received_signal is None for registry in sweep.registries)
    assert _postprocess_kwargs(sweep).get("received_signal") is None
    sweep.only("shrink")
    assert _read_evidence(sweep.log_dir)["succeeded"] is True


# ---------------------------------------------------------------------------
# serve_only
# ---------------------------------------------------------------------------


def test_serve_only_hold_ended_without_a_signal_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # MockOptions.end_manual_hold sets stop_event at the first poll of the serve-only hold, no signal involved.
    sweep = _run_sweep(
        tmp_path,
        monkeypatch,
        release=True,
        serve_only=True,
        end_manual_hold=True,
        benchmark=SweepOrchestrator.run_benchmark,
    )
    assert sweep.exit_code == 0
    assert _postprocess_kwargs(sweep).get("received_signal") is None
    shrink_at = sweep.only("shrink")
    assert shrink_at < sweep.only("srun", "upload")
    assert list(sweep.srun("upload")["nodelist"]) == [TWO_NODES[0]]
    assert _read_evidence(sweep.log_dir)["succeeded"] is True


def test_serve_only_hold_ended_by_sigterm_skips_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # How a serve-only job really ends: scancel or walltime delivers SIGTERM during the hold.
    import time as real_time

    class _SigtermClock:
        monotonic = staticmethod(real_time.monotonic)
        time = staticmethod(real_time.time)

        def sleep(self, _seconds: float) -> None:
            signal.raise_signal(signal.SIGTERM)

    shrink = MagicMock(name="shrink_job_allocation")
    monkeypatch.setattr("srtctl.cli.mixins.benchmark_stage.time", _SigtermClock())
    sweep = _run_sweep(
        tmp_path,
        monkeypatch,
        release=True,
        nodelist=THREE_NODES,
        dedicated_infra=True,
        serve_only=True,
        benchmark=SweepOrchestrator.run_benchmark,
        shrink=lambda *_: shrink,
    )
    assert sweep.system_exit == 1
    shrink.assert_not_called()
    evidence = _read_evidence(sweep.log_dir)
    assert evidence["requested"] is False
    assert "SIGTERM" in evidence["reason"]
    assert list(sweep.srun("upload")["nodelist"]) == [THREE_NODES[1]]


def _assert_interrupted_release(sweep: _Sweep, *, requested: bool, retained: str) -> None:
    """The run ends through the handler's SystemExit(1), with a record of the interrupted release on disk."""
    assert sweep.system_exit == 1, "the handler ends the job with sys.exit(1)"
    assert sweep.exit_code is None

    evidence = _read_evidence(sweep.log_dir)
    assert EVIDENCE_KEYS | {"job_id", "timestamp"} <= set(evidence)
    assert evidence["job_id"] == JOB_ID
    assert evidence["requested"] is requested
    assert evidence["succeeded"] is False
    assert evidence["released"] == []
    assert evidence["verified"] is False
    assert evidence["retained"] == retained
    reason = (evidence["reason"] or "").lower()
    assert "interrupt" in reason and "signal" in reason, evidence["reason"]

    # The SystemExit leaves run_postprocess and the rest of the finally block, as any signal there did before.
    assert not sweep.indices("srun", "upload")
    assert not sweep.indices("srun", "ai")
    assert not sweep.indices("report_completed")
    # Everything before the release still ran.
    assert sweep.indices("final_lockfile_written")
    assert sweep.indices("energy_report_done")


def test_sigterm_during_the_shrink_still_leaves_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def shrink_interrupted(sweep_events: list, *_):
        def fake(job_id: str, keep: str, *, timeout: float = 30.0):
            sweep_events.append(("shrink", {"job_id": job_id, "keep": keep}))
            # Walltime or scancel arriving while scontrol update is in flight.
            signal.raise_signal(signal.SIGTERM)
            pytest.fail("the installed signal handler did not end the run")

        return fake

    sweep = _run_sweep(tmp_path, monkeypatch, release=True, benchmark_exit=7, shrink=shrink_interrupted)
    assert sweep.events[sweep.only("shrink")][1]["keep"] == TWO_NODES[0]
    assert sweep.registries and sweep.registries[0].received_signal == "SIGTERM"
    _assert_interrupted_release(sweep, requested=True, retained=TWO_NODES[0])


def test_sigint_before_the_shrink_is_reached_records_no_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shrink = MagicMock(name="shrink_job_allocation")

    def batch_host_interrupted() -> str:
        # Ctrl+C while the batch host is being resolved: the shrink is never reached.
        signal.raise_signal(signal.SIGINT)
        pytest.fail("the installed signal handler did not end the run")

    sweep = _run_sweep(
        tmp_path,
        monkeypatch,
        release=True,
        benchmark_exit=7,
        shrink=lambda *_: shrink,
        batch_host=MagicMock(side_effect=batch_host_interrupted),
    )
    shrink.assert_not_called()
    assert sweep.registries and sweep.registries[0].received_signal == "SIGINT"
    _assert_interrupted_release(sweep, requested=False, retained="")
