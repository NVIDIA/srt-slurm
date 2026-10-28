# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Slurm allocation-shrink helpers in ``srtctl.core.slurm``.

``FakeSlurm`` stands in for ``scontrol`` and ``squeue``. Its ``scontrol update`` follows
SchedMD/slurm 521370b4:

- ``src/slurmctld/job_mgr.c`` (~13325-13400): a NodeList update on a running job is
  rejected with ESLURM_NOT_SUPPORTED under ``disable_job_shrink``, and with
  ESLURM_INVALID_NODE_NAME when the list is empty, not a subset of the job's nodes, or
  drops the batch host.
- ``src/scontrol/update_job.c`` (~1180-1192): a failed update prints
  ``"<slurm_strerror> for job <id>"`` to stderr and exits 1.
- ``src/scontrol/update_job.c`` ``_update_job_size`` (~1276-1377): on success, and only when
  ``SLURM_JOBID`` is in scontrol's environment, scontrol writes ``slurm_job_<id>_resize.sh``
  and ``.csh`` into its current working directory. Each line is written only if that
  variable is set in scontrol's environment. It then prints the "To reset Slurm environment
  variables" hint to stdout. A file it cannot create is a message on stderr, not a failure.
- ``src/common/env.c`` ``uint32_compressed_to_str``: ``SLURM_JOB_CPUS_PER_NODE`` is written
  as ``64``, ``64(x2)`` or ``72,64(x3)``.

A client timeout is modelled at three points: before slurmctld applies the shrink, after it
applies it (``job_mgr.c`` ~13371-13385) but before scontrol writes the script, and after the
script is written. Only squeue can tell the first apart from the other two.
"""

import dataclasses
import json
import os
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from srtctl.core.slurm import AllocationShrink, get_batch_host, shrink_job_allocation

REPO_ROOT = Path(__file__).resolve().parent.parent
JOB_ID = "12345"
NODES = ["gb200-01", "gb200-02", "gb200-03", "gb200-04"]
COMPRESSED = "gb200-[01-04]"

ESLURM_NOT_SUPPORTED = "Requested operation not supported on this system"
ESLURM_INVALID_NODE_NAME = "Invalid node name specified"

# The variables scontrol's resize script may rewrite, in the order it writes them.
RESIZE_VARS = (
    "SLURM_NODELIST",
    "SLURM_JOB_NODELIST",
    "SLURM_NNODES",
    "SLURM_JOB_NUM_NODES",
    "SLURM_JOB_CPUS_PER_NODE",
)
TASK_VARS = ("SLURM_NPROCS", "SLURM_NTASKS", "SLURM_TASKS_PER_NODE")


# =============================================================================
# Fake Slurm
# =============================================================================


def expand_hostlist(raw: str) -> list[str]:
    """Expand ``a-[01-03,05],b-07`` the way ``scontrol show hostnames`` does."""
    items: list[str] = []
    depth = 0
    current = ""
    for ch in raw:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(current)
            current = ""
        else:
            current += ch
    if current:
        items.append(current)

    hosts: list[str] = []
    for item in items:
        match = re.fullmatch(r"([^\[]*)\[([^\]]+)\](.*)", item)
        if not match:
            hosts.append(item)
            continue
        prefix, ranges, suffix = match.groups()
        for part in ranges.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                hosts.extend(f"{prefix}{i:0{len(lo)}d}{suffix}" for i in range(int(lo), int(hi) + 1))
            else:
                hosts.append(f"{prefix}{part}{suffix}")
    return hosts


def compress_cpus(counts: list[int]) -> str:
    """Port of ``uint32_compressed_to_str`` (src/common/env.c)."""
    groups: list[list[int]] = []
    for count in counts:
        if groups and groups[-1][0] == count:
            groups[-1][1] += 1
        else:
            groups.append([count, 1])
    return ",".join(f"{c}(x{r})" if r > 1 else f"{c}" for c, r in groups)


@dataclasses.dataclass
class Call:
    cmd: list[str] | str
    kwargs: dict


@dataclasses.dataclass
class FakeSlurm:
    """slurmctld state for one running batch job plus the scontrol/squeue clients."""

    job_id: str = JOB_ID
    job_nodes: list[str] = dataclasses.field(default_factory=lambda: list(NODES))
    batch_host: str = NODES[0]
    cpus: dict[str, int] = dataclasses.field(default_factory=lambda: dict.fromkeys(NODES, 64))
    disable_job_shrink: bool = False
    # The job's submitted --nodelist. An update naming exactly this list is a no-op that still
    # succeeds (job_mgr.c:13314-13317), so scontrol writes a script for the unchanged allocation.
    submitted_nodelist: str | None = None
    # Exception raised by ``scontrol update`` instead of running (missing binary, permissions).
    update_raises: BaseException | None = None
    # Where the scontrol client times out (subprocess.TimeoutExpired):
    #   "before_apply": slurmctld never applied the shrink;
    #   "after_apply": slurmctld shrank the job, the client died before writing the resize script;
    #   "after_script": slurmctld shrank the job and scontrol wrote the script, then was killed.
    update_timeout: str | None = None
    # Every scontrol invocation raises FileNotFoundError (binary not on PATH).
    scontrol_missing: bool = False
    # scontrol could not create the resize files in its cwd (it still exits 0).
    cannot_write_resize: bool = False
    # Extra lines appended to the .sh / .csh files, or a full replacement of the .sh body.
    extra_sh_lines: tuple[str, ...] = ()
    extra_csh_lines: tuple[str, ...] = ()
    replace_sh_body: str | None = None
    # squeue: "ok" (real state), "fail" (non-zero), "raise" (missing binary), "timeout", or fixed output.
    squeue_mode: str = "ok"
    squeue_output: str | None = None

    calls: list[Call] = dataclasses.field(default_factory=list)
    unexpected: list[Call] = dataclasses.field(default_factory=list)
    update_cwd_listing: list[list[str]] = dataclasses.field(default_factory=list)

    # ---- helpers -----------------------------------------------------------

    def update_calls(self) -> list[Call]:
        return [c for c in self.calls if isinstance(c.cmd, list) and c.cmd[:2] == ["scontrol", "update"]]

    def squeue_calls(self) -> list[Call]:
        return [c for c in self.calls if isinstance(c.cmd, list) and c.cmd[:1] == ["squeue"]]

    @staticmethod
    def _result(cmd, kwargs, returncode: int, stdout: str, stderr: str) -> subprocess.CompletedProcess:
        """Build what the real subprocess.run would return for these kwargs."""
        captured = kwargs.get("capture_output") or kwargs.get("stdout") == subprocess.PIPE
        text = kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding")
        out: str | bytes | None = None
        err: str | bytes | None = None
        if captured:
            out = stdout if text else stdout.encode()
            err = stderr if text else stderr.encode()
        if kwargs.get("check") and returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd, output=out, stderr=err)
        return subprocess.CompletedProcess(cmd, returncode, stdout=out, stderr=err)

    # ---- subprocess.run replacement -------------------------------------------

    def __call__(self, cmd, *args, **kwargs):
        call = Call(cmd=cmd, kwargs=dict(kwargs))
        self.calls.append(call)
        if isinstance(cmd, str) or kwargs.get("shell") or args:
            self.unexpected.append(call)
            return self._result(cmd, kwargs, 127, "", "unexpected command\n")

        cwd = kwargs.get("cwd")
        if cwd is not None and not Path(cwd).is_dir():
            raise FileNotFoundError(2, "No such file or directory", str(cwd))

        if cmd[0] == "scontrol":
            if self.scontrol_missing:
                raise FileNotFoundError(2, "No such file or directory", "scontrol")
            if cmd[1:3] == ["show", "hostnames"]:
                return self._result(cmd, kwargs, 0, "\n".join(expand_hostlist(cmd[3])) + "\n", "")
            if cmd[1] == "update":
                return self._scontrol_update(cmd, kwargs)
        if cmd[0] == "squeue":
            return self._squeue(cmd, kwargs)

        self.unexpected.append(call)
        return self._result(cmd, kwargs, 127, "", f"{cmd[0]}: command not found\n")

    def _scontrol_update(self, cmd, kwargs):
        cwd = kwargs.get("cwd")
        self.update_cwd_listing.append(sorted(os.listdir(cwd)) if cwd else [])
        if self.update_raises is not None:
            raise self.update_raises
        if self.update_timeout == "before_apply":
            self._timeout(cmd, kwargs)

        tags = {}
        for arg in cmd[2:]:
            key, _, value = arg.partition("=")
            tags[key.lower()] = value
        job_id = tags.get("jobid", "")
        req_nodes = tags.get("nodelist", tags.get("reqnodelist"))
        if job_id != self.job_id:
            return self._result(cmd, kwargs, 1, "", f"Invalid job id specified for job {job_id}\n")

        # slurmctld job_mgr.c:13314-13317: identical to the submitted --nodelist, nothing changes.
        if req_nodes is not None and req_nodes == self.submitted_nodelist:
            pass
        else:
            # slurmctld job_mgr.c ~13325-13365
            if self.disable_job_shrink:
                return self._result(cmd, kwargs, 1, "", f"{ESLURM_NOT_SUPPORTED} for job {job_id}\n")
            requested = expand_hostlist(req_nodes) if req_nodes else []
            if not requested or not set(requested) <= set(self.job_nodes):
                return self._result(cmd, kwargs, 1, "", f"{ESLURM_INVALID_NODE_NAME} for job {job_id}\n")
            if self.batch_host not in requested:
                return self._result(cmd, kwargs, 1, "", f"{ESLURM_INVALID_NODE_NAME} for job {job_id}\n")
            self.job_nodes = [n for n in self.job_nodes if n in requested]
        if self.update_timeout == "after_apply":
            self._timeout(cmd, kwargs)

        # scontrol update_job.c _update_job_size ~1276-1377
        env = kwargs.get("env")
        env = os.environ if env is None else env
        if "SLURM_JOBID" not in env:
            return self._result(cmd, kwargs, 0, "", "")
        sh_name = f"slurm_job_{job_id}_resize.sh"
        csh_name = f"slurm_job_{job_id}_resize.csh"
        if self.cannot_write_resize or cwd is None:
            # cwd None would write into the caller's checkout; the cwd test fails it separately.
            return self._result(cmd, kwargs, 0, "", f"Could not create file {csh_name}: Permission denied\n")

        node_list = ",".join(self.job_nodes)
        node_cnt = len(self.job_nodes)
        sh: list[str] = []
        csh: list[str] = []
        if "SLURM_NODELIST" in env:
            sh.append(f'export SLURM_NODELIST="{node_list}"')
            csh.append(f'setenv SLURM_NODELIST "{node_list}"')
        if "SLURM_JOB_NODELIST" in env:
            sh.append(f'export SLURM_JOB_NODELIST="{node_list}"')
            csh.append(f'setenv SLURM_JOB_NODELIST "{node_list}"')
        if "SLURM_NNODES" in env:
            sh.append(f"export SLURM_NNODES={node_cnt}")
            csh.append(f"setenv SLURM_NNODES {node_cnt}")
        if "SLURM_JOB_NUM_NODES" in env:
            sh.append(f"export SLURM_JOB_NUM_NODES={node_cnt}")
            csh.append(f"setenv SLURM_JOB_NUM_NODES {node_cnt}")
        if "SLURM_JOB_CPUS_PER_NODE" in env:
            cpus = compress_cpus([self.cpus[n] for n in self.job_nodes])
            sh.append(f'export SLURM_JOB_CPUS_PER_NODE="{cpus}"')
            csh.append(f'setenv SLURM_JOB_CPUS_PER_NODE "{cpus}"')
        if "SLURM_TASKS_PER_NODE" in env:
            for var in TASK_VARS:
                sh.append(f"unset {var}")
                csh.append(f"unsetenv {var}")
        sh.extend(self.extra_sh_lines)
        csh.extend(self.extra_csh_lines)

        sh_body = self.replace_sh_body if self.replace_sh_body is not None else "".join(f"{line}\n" for line in sh)
        sh_path = Path(cwd) / sh_name
        csh_path = Path(cwd) / csh_name
        sh_path.write_text(sh_body)
        csh_path.write_text("".join(f"{line}\n" for line in csh))
        sh_path.chmod(0o700)
        csh_path.chmod(0o700)
        if self.update_timeout == "after_script":
            self._timeout(cmd, kwargs)
        stdout = (
            "To reset Slurm environment variables, execute\n"
            f"  For bash or sh shells:  . ./{sh_name}\n"
            f"  For csh shells:         source ./{csh_name}\n"
        )
        return self._result(cmd, kwargs, 0, stdout, "")

    @staticmethod
    def _timeout(cmd, kwargs):
        """subprocess.run kills the child and raises; whatever scontrol already did stays done."""
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 30.0)

    def _squeue(self, cmd, kwargs):
        if self.squeue_mode == "raise":
            raise FileNotFoundError(2, "No such file or directory", "squeue")
        if self.squeue_mode == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 1.0)
        if self.squeue_mode == "fail":
            return self._result(cmd, kwargs, 1, "", "slurm_load_jobs error: Invalid job id specified\n")
        if self.squeue_output is not None:
            return self._result(cmd, kwargs, 0, self.squeue_output, "")
        return self._result(cmd, kwargs, 0, ",".join(self.job_nodes) + "\n", "")


# =============================================================================
# Environment helpers
# =============================================================================


def slurm_env(**overrides: str | None) -> dict[str, str]:
    """A batch-script environment for a 4-node job, keeping PATH/HOME so the process still works."""
    env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
    env.update(
        {
            "SLURM_JOB_ID": JOB_ID,
            "SLURM_JOBID": JOB_ID,
            "SLURM_NODELIST": COMPRESSED,
            "SLURM_JOB_NODELIST": COMPRESSED,
            "SLURM_NNODES": "4",
            "SLURM_JOB_NUM_NODES": "4",
            "SLURM_JOB_CPUS_PER_NODE": "64(x4)",
            "SLURM_TASKS_PER_NODE": "1(x4)",
            "SLURM_NTASKS": "4",
            "SLURM_NPROCS": "4",
            "SLURMD_NODENAME": NODES[0],
            "SLURM_JOB_PARTITION": "batch",
            "SLURM_CPUS_ON_NODE": "64",
            "SRTCTL_UNRELATED": 'keep "me" = as-is',
        }
    )
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def run_shrink(fake: FakeSlurm, env: dict[str, str], keep: str, **kwargs) -> tuple[AllocationShrink, dict[str, str]]:
    """Run ``shrink_job_allocation`` against the fake; return the result and the resulting environment."""
    with (
        patch.dict(os.environ, env, clear=True),
        patch("srtctl.core.slurm.subprocess.run", fake),
    ):
        result = shrink_job_allocation(JOB_ID, keep, **kwargs)
        after = dict(os.environ)
    return result, after


def assert_temp_dirs_removed(fake: FakeSlurm) -> None:
    for call in fake.update_calls():
        cwd = call.kwargs.get("cwd")
        assert cwd is not None
        assert not Path(cwd).exists(), f"resize temp dir {cwd} was left behind"


def assert_failed_cleanly(result: AllocationShrink, before: dict[str, str], after: dict[str, str]) -> None:
    assert result.succeeded is False
    assert result.released == ()
    assert result.verified is False
    assert isinstance(result.reason, str) and result.reason
    assert after == before, "a failed shrink must leave os.environ untouched"


# =============================================================================
# get_batch_host
# =============================================================================


class TestGetBatchHost:
    def test_slurmd_nodename_wins_over_nodelist(self):
        """SLURMD_NODENAME names the batch host even when it is not the first listed node."""
        fake = FakeSlurm()
        env = slurm_env(SLURMD_NODENAME="gb200-03")
        with patch.dict(os.environ, env, clear=True), patch("srtctl.core.slurm.subprocess.run", fake):
            assert get_batch_host() == "gb200-03"

    def test_slurmd_nodename_needs_no_scontrol(self):
        """SLURMD_NODENAME is enough on its own; a missing scontrol does not matter."""
        fake = FakeSlurm(scontrol_missing=True)
        env = slurm_env(SLURMD_NODENAME="gb200-02")
        with patch.dict(os.environ, env, clear=True), patch("srtctl.core.slurm.subprocess.run", fake):
            assert get_batch_host() == "gb200-02"

    def test_falls_back_to_first_expanded_node(self):
        """Without SLURMD_NODENAME the first node of the expanded nodelist, not the ranged string."""
        fake = FakeSlurm()
        env = slurm_env(SLURMD_NODENAME=None)
        with patch.dict(os.environ, env, clear=True), patch("srtctl.core.slurm.subprocess.run", fake):
            assert get_batch_host() == "gb200-01"

    def test_empty_slurmd_nodename_falls_back(self):
        """An empty SLURMD_NODENAME is not a hostname."""
        fake = FakeSlurm()
        env = slurm_env(SLURMD_NODENAME="")
        with patch.dict(os.environ, env, clear=True), patch("srtctl.core.slurm.subprocess.run", fake):
            assert get_batch_host() == "gb200-01"

    def test_none_outside_slurm(self):
        """No SLURMD_NODENAME and no nodelist: None, no exception."""
        fake = FakeSlurm()
        env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
        with patch.dict(os.environ, env, clear=True), patch("srtctl.core.slurm.subprocess.run", fake):
            assert get_batch_host() is None
        assert fake.update_calls() == []


# =============================================================================
# Result type
# =============================================================================


class TestAllocationShrinkType:
    def test_frozen_dataclass_with_contract_fields_in_order(self):
        assert dataclasses.is_dataclass(AllocationShrink)
        names = [f.name for f in dataclasses.fields(AllocationShrink)]
        assert names[:7] == ["requested", "succeeded", "retained", "released", "verified", "reason", "duration_s"]

    def test_assignment_raises(self):
        fake = FakeSlurm()
        result, _ = run_shrink(fake, slurm_env(), NODES[0])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.succeeded = False  # type: ignore[misc]

    def test_to_dict_round_trips_through_json(self):
        fake = FakeSlurm()
        result, _ = run_shrink(fake, slurm_env(), NODES[0])
        data = result.to_dict()
        assert isinstance(data, dict)
        assert isinstance(data["released"], list)
        assert data["released"] == list(result.released)
        assert json.loads(json.dumps(data)) == data
        for field in ("requested", "succeeded", "retained", "released", "verified", "reason", "duration_s"):
            assert field in data
        assert data["retained"] == NODES[0]
        assert data["reason"] is None

    def test_to_dict_of_failure_is_json_serialisable(self):
        """stderr must be captured as text: bytes in ``reason`` would break json.dumps."""
        fake = FakeSlurm(disable_job_shrink=True)
        result, _ = run_shrink(fake, slurm_env(), NODES[0])
        data = json.loads(json.dumps(result.to_dict()))
        assert data["released"] == []
        assert ESLURM_NOT_SUPPORTED in data["reason"]


# =============================================================================
# Skips: no scontrol update at all
# =============================================================================


class TestSkips:
    def test_keep_not_in_nodelist(self):
        fake = FakeSlurm()
        env = slurm_env()
        result, after = run_shrink(fake, env, "gb200-99")
        assert fake.update_calls() == []
        assert result.requested is False
        assert result.succeeded is False
        assert result.released == ()
        assert isinstance(result.reason, str) and result.reason
        assert result.duration_s >= 0
        assert after == env

    def test_nodelist_already_only_keep(self):
        """A one-node job has nothing to release; scontrol must not be asked to 'shrink' it."""
        fake = FakeSlurm(job_nodes=[NODES[0]])
        env = slurm_env(
            SLURM_NODELIST=NODES[0],
            SLURM_JOB_NODELIST=NODES[0],
            SLURM_NNODES="1",
            SLURM_JOB_NUM_NODES="1",
            SLURM_JOB_CPUS_PER_NODE="64",
            SLURM_TASKS_PER_NODE="1",
            SLURM_NTASKS="1",
            SLURM_NPROCS="1",
        )
        result, after = run_shrink(fake, env, NODES[0])
        assert fake.update_calls() == []
        assert result.requested is False
        assert result.released == ()
        assert isinstance(result.reason, str) and result.reason
        assert after == env, "nothing was resized, so the task variables must survive"

    def test_no_nodelist_in_environment(self):
        fake = FakeSlurm()
        env = slurm_env(SLURM_NODELIST=None, SLURM_JOB_NODELIST=None)
        result, after = run_shrink(fake, env, NODES[0])
        assert fake.update_calls() == []
        assert result.requested is False
        assert result.succeeded is False
        assert result.released == ()
        assert after == env


# =============================================================================
# The scontrol update call
# =============================================================================


class TestScontrolInvocation:
    def test_exact_command_and_kwargs(self):
        fake = FakeSlurm()
        result, _ = run_shrink(fake, slurm_env(), NODES[0])
        updates = fake.update_calls()
        assert len(updates) == 1, "exactly one scontrol update, no retries"
        call = updates[0]
        assert call.cmd == ["scontrol", "update", f"JobId={JOB_ID}", f"NodeList={NODES[0]}"]
        assert call.kwargs.get("timeout") == 30.0
        assert not call.kwargs.get("shell")
        assert not call.kwargs.get("check")
        assert result.requested is True

    def test_timeout_is_passed_through(self):
        fake = FakeSlurm()
        run_shrink(fake, slurm_env(), NODES[0], timeout=7.5)
        assert fake.update_calls()[0].kwargs.get("timeout") == 7.5

    def test_cwd_is_a_fresh_private_temp_dir_that_is_removed(self, tmp_path):
        """Resize scripts must never land in the process cwd or the shared srtctl checkout."""
        fake = FakeSlurm()
        env = slurm_env(SRTCTL_SOURCE_DIR=str(REPO_ROOT))
        with (
            patch.dict(os.environ, env, clear=True),
            patch("srtctl.core.slurm.subprocess.run", fake),
        ):
            old_cwd = os.getcwd()
            os.chdir(tmp_path)
            try:
                shrink_job_allocation(JOB_ID, NODES[0])
            finally:
                os.chdir(old_cwd)

        cwd = fake.update_calls()[0].kwargs.get("cwd")
        assert cwd is not None, "scontrol update must run with an explicit cwd"
        cwd_path = Path(cwd).resolve()
        assert cwd_path != tmp_path.resolve()
        assert cwd_path != Path(old_cwd).resolve()
        assert REPO_ROOT not in (cwd_path, *cwd_path.parents)
        assert fake.update_cwd_listing == [[]], "the temp dir must be fresh and empty when scontrol runs"
        assert not cwd_path.exists(), "the temp dir must be removed afterwards"
        assert list(tmp_path.iterdir()) == []
        assert not list(REPO_ROOT.glob("slurm_job_*_resize.*"))

    def test_no_shell_or_other_process_is_spawned(self):
        """Only scontrol and squeue are run; the resize script is never handed to a shell."""
        fake = FakeSlurm()
        run_shrink(fake, slurm_env(), NODES[0])
        assert fake.unexpected == []
        for call in fake.calls:
            assert isinstance(call.cmd, list)
            assert call.cmd[0] in ("scontrol", "squeue")


# =============================================================================
# Environment after a successful shrink
# =============================================================================


class TestEnvironmentAfterSuccess:
    def test_full_environment_mirrors_resize_script(self):
        fake = FakeSlurm()
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])

        expected = dict(env)
        expected.update(
            SLURM_NODELIST=NODES[0],
            SLURM_JOB_NODELIST=NODES[0],
            SLURM_NNODES="1",
            SLURM_JOB_NUM_NODES="1",
            SLURM_JOB_CPUS_PER_NODE="64",
        )
        for var in TASK_VARS:
            expected.pop(var)
        assert after == expected, "quotes stripped, task vars unset, every other variable untouched"

        assert result.requested is True
        assert result.succeeded is True
        assert result.retained == NODES[0]
        assert result.released == tuple(NODES[1:])
        assert result.verified is True
        assert result.reason is None
        assert isinstance(result.duration_s, float) and result.duration_s >= 0
        assert_temp_dirs_removed(fake)

    def test_only_variables_that_were_set_are_rewritten(self):
        """scontrol writes a line only for a variable already in its env: nothing new may appear."""
        fake = FakeSlurm()
        env = slurm_env(
            SLURM_JOB_NODELIST=None,
            SLURM_NNODES=None,
            SLURM_JOB_NUM_NODES=None,
            SLURM_JOB_CPUS_PER_NODE=None,
        )
        result, after = run_shrink(fake, env, NODES[0])
        expected = dict(env)
        expected["SLURM_NODELIST"] = NODES[0]
        for var in TASK_VARS:
            expected.pop(var)
        assert after == expected
        for var in ("SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_JOB_NUM_NODES", "SLURM_JOB_CPUS_PER_NODE"):
            assert var not in after
        assert result.succeeded is True

    def test_task_vars_kept_when_tasks_per_node_unset(self):
        """The unset trio is written only when SLURM_TASKS_PER_NODE was set."""
        fake = FakeSlurm()
        env = slurm_env(SLURM_TASKS_PER_NODE=None, SLURM_NTASKS="8", SLURM_NPROCS="8")
        _, after = run_shrink(fake, env, NODES[0])
        assert after["SLURM_NTASKS"] == "8"
        assert after["SLURM_NPROCS"] == "8"
        assert "SLURM_TASKS_PER_NODE" not in after

    def test_unset_of_absent_task_vars_is_harmless(self):
        """SLURM_TASKS_PER_NODE set alone: the unset lines name variables that do not exist."""
        fake = FakeSlurm()
        env = slurm_env(SLURM_NTASKS=None, SLURM_NPROCS=None)
        result, after = run_shrink(fake, env, NODES[0])
        assert result.succeeded is True
        for var in TASK_VARS:
            assert var not in after

    @pytest.mark.parametrize(
        ("cpus_env", "per_node", "keep", "expected"),
        [
            ("64(x4)", [64, 64, 64, 64], "gb200-01", "64"),
            ("72,64(x3)", [72, 64, 64, 64], "gb200-01", "72"),
            # Real scontrol reports the kept node's own count: the script wins over a first-group guess.
            ("72,64(x3)", [72, 64, 64, 64], "gb200-02", "64"),
            ("64(x2),32(x2)", [64, 64, 32, 32], "gb200-04", "32"),
        ],
    )
    def test_cpus_per_node_comes_from_the_script(self, cpus_env, per_node, keep, expected):
        fake = FakeSlurm(batch_host=keep, cpus=dict(zip(NODES, per_node, strict=True)))
        env = slurm_env(SLURM_JOB_CPUS_PER_NODE=cpus_env, SLURMD_NODENAME=keep)
        result, after = run_shrink(fake, env, keep)
        assert result.succeeded is True
        assert after["SLURM_JOB_CPUS_PER_NODE"] == expected
        assert after["SLURM_NODELIST"] == keep

    def test_csh_file_is_ignored(self):
        """Only the .sh file is read; a .csh line or a stray sh-syntax line in it changes nothing."""
        fake = FakeSlurm(
            extra_csh_lines=(
                'setenv SRTCTL_CSH_POISON "1"',
                "export SRTCTL_CSH_POISON2=1",
                "unsetenv PATH",
                # Allowlisted names in sh syntax, for variables the .sh file leaves alone.
                "export SLURM_NNODES=7",
                "unset SLURM_NTASKS",
            ),
        )
        env = slurm_env(SLURM_NNODES=None, SLURM_TASKS_PER_NODE=None)
        _, after = run_shrink(fake, env, NODES[0])
        assert "SRTCTL_CSH_POISON" not in after
        assert "SRTCTL_CSH_POISON2" not in after
        assert after.get("PATH") == env.get("PATH")
        assert "SLURM_NNODES" not in after
        assert after["SLURM_NTASKS"] == "4"

    def test_keep_in_middle_releases_others_in_nodelist_order(self):
        fake = FakeSlurm(batch_host="gb200-03")
        env = slurm_env(SLURMD_NODENAME="gb200-03")
        result, after = run_shrink(fake, env, "gb200-03")
        assert result.succeeded is True
        assert result.retained == "gb200-03"
        assert result.released == ("gb200-01", "gb200-02", "gb200-04")
        assert after["SLURM_NODELIST"] == "gb200-03"
        assert fake.job_nodes == ["gb200-03"]


# =============================================================================
# Missing or malformed resize script
# =============================================================================


class TestResizeScriptFallback:
    @pytest.mark.parametrize(
        ("fake_kwargs", "env_overrides"),
        [
            # scontrol only writes the script when SLURM_JOBID (not SLURM_JOB_ID) is in its env.
            ({}, {"SLURM_JOBID": None}),
            # scontrol could not create the file: message on stderr, exit 0.
            ({"cannot_write_resize": True}, {}),
        ],
        ids=["no-SLURM_JOBID", "could-not-create-file"],
    )
    @pytest.mark.parametrize(
        ("cpus_env", "expected_cpus"),
        # The kept node is first here; "64,32" does not expand to one entry per node.
        [("64(x4)", "64"), ("64,32", "64"), ("72,64(x3)", "72")],
    )
    def test_missing_script_applies_fallback(self, fake_kwargs, env_overrides, cpus_env, expected_cpus):
        fake = FakeSlurm(**fake_kwargs)
        env = slurm_env(SLURM_JOB_CPUS_PER_NODE=cpus_env, **env_overrides)
        result, after = run_shrink(fake, env, NODES[0])

        expected = dict(env)
        expected.update(
            SLURM_NODELIST=NODES[0],
            SLURM_JOB_NODELIST=NODES[0],
            SLURM_NNODES="1",
            SLURM_JOB_NUM_NODES="1",
            SLURM_JOB_CPUS_PER_NODE=expected_cpus,
        )
        for var in TASK_VARS:
            expected.pop(var)
        assert after == expected
        assert result.succeeded is True
        assert result.released == tuple(NODES[1:])
        assert_temp_dirs_removed(fake)

    @pytest.mark.parametrize(
        ("cpus_env", "keep", "expected_cpus"),
        [
            # Expands to one entry per node: the kept node's own entry, by nodelist position.
            ("72,64(x3)", "gb200-02", "64"),
            ("72,64(x2),32", "gb200-04", "32"),
            ("16,32,48,96", "gb200-04", "96"),
            ("16,32,48,96", "gb200-03", "48"),
            # Does not match the node count: first group.
            ("72,64(x2)", "gb200-04", "72"),
            ("72,64(x4)", "gb200-04", "72"),
            ("72(x2)", "gb200-02", "72"),
            # Unparsable groups: first group, never an exception.
            ("64(x)", "gb200-02", "64"),
            ("64(x2),abc,32", "gb200-04", "64"),
            ("abc", "gb200-02", "abc"),
        ],
    )
    def test_fallback_cpus_are_the_kept_nodes_own(self, cpus_env, keep, expected_cpus):
        """Without a script, SLURM_JOB_CPUS_PER_NODE becomes the kept node's count, as real scontrol writes."""
        fake = FakeSlurm(batch_host=keep)
        env = slurm_env(SLURM_JOBID=None, SLURM_JOB_CPUS_PER_NODE=cpus_env, SLURMD_NODENAME=keep)
        result, after = run_shrink(fake, env, keep)
        assert result.succeeded is True
        assert after["SLURM_JOB_CPUS_PER_NODE"] == expected_cpus
        assert after["SLURM_NODELIST"] == keep
        assert after["SLURM_NNODES"] == "1"
        assert result.released == tuple(n for n in NODES if n != keep)

    def test_fallback_keeps_conditional_semantics(self):
        """The fallback never adds a variable that was not set, and keeps the task vars without TASKS_PER_NODE."""
        fake = FakeSlurm()
        env = slurm_env(
            SLURM_JOBID=None,
            SLURM_JOB_NODELIST=None,
            SLURM_JOB_NUM_NODES=None,
            SLURM_JOB_CPUS_PER_NODE=None,
            SLURM_TASKS_PER_NODE=None,
        )
        result, after = run_shrink(fake, env, NODES[0])
        expected = dict(env)
        expected.update(SLURM_NODELIST=NODES[0], SLURM_NNODES="1")
        assert after == expected
        assert result.succeeded is True

    def test_script_is_parsed_never_executed(self, tmp_path):
        """Shell syntax in the resize script is data: nothing runs, junk lines are ignored."""
        pwned = [tmp_path / f"pwned{i}" for i in range(5)]
        fake = FakeSlurm(
            extra_sh_lines=(
                f"touch {pwned[0]}",
                f"$(touch {pwned[1]})",
                f"`touch {pwned[2]}`",
                f"eval 'touch {pwned[3]}'",
                f"source <(echo touch {pwned[4]})",
                "export =oops",
                "export",
                "unset",
                "garbage === line",
                "# a comment",
                "",
                "   ",
            ),
        )
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        for path in pwned:
            assert not path.exists(), f"resize script content was executed ({path.name})"
        assert fake.unexpected == []

        expected = dict(env)
        expected.update(
            SLURM_NODELIST=NODES[0],
            SLURM_JOB_NODELIST=NODES[0],
            SLURM_NNODES="1",
            SLURM_JOB_NUM_NODES="1",
            SLURM_JOB_CPUS_PER_NODE="64",
        )
        for var in TASK_VARS:
            expected.pop(var)
        assert after == expected, "junk lines must not add, drop or alter variables"
        assert result.succeeded is True

    def test_only_allowlisted_names_are_applied(self):
        """The script may set or unset only the eight resize variables; anything else is ignored."""
        fake = FakeSlurm(
            extra_sh_lines=(
                'export PATH="/tmp/evil/bin"',
                "export LD_PRELOAD=/tmp/evil.so",
                'export SLURM_FOO="bar"',
                "export SLURM_JOB_ID=99999",
                "unset HOME",
                "unset SLURM_JOB_PARTITION",
            ),
        )
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        assert "LD_PRELOAD" not in after
        assert "SLURM_FOO" not in after
        assert after.get("PATH") == env.get("PATH")
        assert after.get("HOME") == env.get("HOME")
        assert after["SLURM_JOB_ID"] == JOB_ID
        assert after["SLURM_JOB_PARTITION"] == "batch"

        expected = dict(env)
        expected.update(
            SLURM_NODELIST=NODES[0],
            SLURM_JOB_NODELIST=NODES[0],
            SLURM_NNODES="1",
            SLURM_JOB_NUM_NODES="1",
            SLURM_JOB_CPUS_PER_NODE="64",
        )
        for var in TASK_VARS:
            expected.pop(var)
        assert after == expected
        assert result.succeeded is True

    def test_script_with_only_disallowed_names_applies_fallback(self):
        """No accepted line means unparsable: the fallback runs and the disallowed lines still change nothing.

        The script carries no Slurm variable at all, so every resize variable that changed (and the
        task trio being unset) can only have come from the fallback.
        """
        keep = "gb200-02"
        fake = FakeSlurm(
            batch_host=keep,
            cpus=dict(zip(NODES, [72, 64, 64, 64], strict=True)),
            replace_sh_body='export PATH="/tmp/evil/bin"\nexport LD_PRELOAD=/tmp/evil.so\nunset HOME\n',
        )
        env = slurm_env(SLURM_JOB_CPUS_PER_NODE="72,64(x3)", SLURMD_NODENAME=keep)
        result, after = run_shrink(fake, env, keep)

        # The disallowed lines were ignored.
        assert after.get("PATH") == env.get("PATH")
        assert after.get("HOME") == env.get("HOME")
        assert "LD_PRELOAD" not in after
        # The fallback ran: every resize variable that was set is rewritten, the kept node's CPUs chosen.
        expected = dict(env)
        expected.update(
            SLURM_NODELIST=keep,
            SLURM_JOB_NODELIST=keep,
            SLURM_NNODES="1",
            SLURM_JOB_NUM_NODES="1",
            SLURM_JOB_CPUS_PER_NODE="64",
        )
        for var in TASK_VARS:
            expected.pop(var)
        assert after == expected
        assert result.succeeded is True

    def test_entirely_unparsable_script_applies_fallback(self, tmp_path):
        marker = tmp_path / "pwned"
        fake = FakeSlurm(replace_sh_body=f"#!/bin/sh\nrm -rf {tmp_path}/nothing\n$(touch {marker})\n\x00\xff\n")
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        assert not marker.exists()
        expected = dict(env)
        expected.update(
            SLURM_NODELIST=NODES[0],
            SLURM_JOB_NODELIST=NODES[0],
            SLURM_NNODES="1",
            SLURM_JOB_NUM_NODES="1",
            SLURM_JOB_CPUS_PER_NODE="64",
        )
        for var in TASK_VARS:
            expected.pop(var)
        assert after == expected
        assert result.succeeded is True


# =============================================================================
# Failure paths: never raise, never touch os.environ
# =============================================================================


class TestFailures:
    @pytest.mark.parametrize(
        ("fake_kwargs", "message"),
        [
            ({"disable_job_shrink": True}, ESLURM_NOT_SUPPORTED),
            # The controller believes the batch host is another node: dropping it is rejected.
            ({"batch_host": "gb200-02"}, ESLURM_INVALID_NODE_NAME),
            # Stale env nodelist: keep is not one of the job's nodes on the controller.
            ({"job_nodes": ["gb200-02", "gb200-03"], "batch_host": "gb200-02"}, ESLURM_INVALID_NODE_NAME),
        ],
        ids=["disable_job_shrink", "batch-host-removed", "not-a-subset"],
    )
    def test_rejected_update_reports_stderr(self, fake_kwargs, message):
        fake = FakeSlurm(**fake_kwargs)
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        assert_failed_cleanly(result, env, after)
        assert result.requested is True
        assert message in result.reason
        assert len(fake.update_calls()) == 1
        assert result.duration_s >= 0
        assert_temp_dirs_removed(fake)

    def test_timeout_before_controller_applied_shrink(self):
        """The client timed out and squeue still shows every node: a failure, env untouched."""
        fake = FakeSlurm(update_timeout="before_apply")
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0], timeout=5.0)
        assert_failed_cleanly(result, env, after)
        assert result.requested is True
        assert fake.job_nodes == NODES
        assert fake.update_calls()[0].kwargs.get("timeout") == 5.0
        assert len(fake.squeue_calls()) == 1, "a timeout must be checked with squeue"
        assert_temp_dirs_removed(fake)

    @pytest.mark.parametrize(
        ("update_timeout", "expected_cpus"),
        [
            # Script written before the kill: it is authoritative (the kept node's own count).
            ("after_script", "64"),
            # No script: the fallback picks the kept node's own entry from "72,64(x3)" too.
            ("after_apply", "64"),
        ],
    )
    def test_timeout_after_controller_applied_shrink(self, update_timeout, expected_cpus):
        """slurmctld shrank the job before the client timed out (job_mgr.c ~13371-13385): a verified success."""
        keep = "gb200-02"
        fake = FakeSlurm(
            update_timeout=update_timeout, batch_host=keep, cpus=dict(zip(NODES, [72, 64, 64, 64], strict=True))
        )
        env = slurm_env(SLURM_JOB_CPUS_PER_NODE="72,64(x3)", SLURMD_NODENAME=keep)
        result, after = run_shrink(fake, env, keep, timeout=5.0)

        expected = dict(env)
        expected.update(
            SLURM_NODELIST=keep,
            SLURM_JOB_NODELIST=keep,
            SLURM_NNODES="1",
            SLURM_JOB_NUM_NODES="1",
            SLURM_JOB_CPUS_PER_NODE=expected_cpus,
        )
        for var in TASK_VARS:
            expected.pop(var)
        assert after == expected

        assert result.requested is True
        assert result.succeeded is True
        assert result.verified is True
        assert result.retained == keep
        assert result.released == ("gb200-01", "gb200-03", "gb200-04")
        assert isinstance(result.reason, str)
        assert re.search(r"time(d)?[ -]?out", result.reason, re.IGNORECASE), result.reason
        assert len(fake.squeue_calls()) == 1
        assert_temp_dirs_removed(fake)

    @pytest.mark.parametrize("squeue_mode", ["fail", "raise", "timeout"])
    def test_timeout_then_squeue_fails(self, squeue_mode):
        """The shrink may well have happened, but without squeue's word it is a failure: env untouched."""
        fake = FakeSlurm(update_timeout="after_apply", squeue_mode=squeue_mode)
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0], timeout=5.0)
        assert_failed_cleanly(result, env, after)
        assert result.requested is True
        assert_temp_dirs_removed(fake)

    def test_timeout_then_squeue_shows_other_nodes(self):
        """squeue answers, but not with exactly [keep]: a failure, env untouched."""
        fake = FakeSlurm(update_timeout="after_apply", squeue_output="gb200-[01-02]\n")
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0], timeout=5.0)
        assert_failed_cleanly(result, env, after)
        assert_temp_dirs_removed(fake)

    def test_scontrol_update_missing_binary(self):
        fake = FakeSlurm(update_raises=FileNotFoundError(2, "No such file or directory", "scontrol"))
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        assert_failed_cleanly(result, env, after)
        assert_temp_dirs_removed(fake)

    def test_scontrol_update_permission_error(self):
        fake = FakeSlurm(update_raises=PermissionError(13, "Permission denied", "scontrol"))
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        assert_failed_cleanly(result, env, after)
        assert_temp_dirs_removed(fake)

    def test_no_scontrol_at_all_never_raises(self):
        """Without scontrol the nodelist cannot be expanded; the ranged string does not contain keep."""
        fake = FakeSlurm(scontrol_missing=True)
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        assert result.succeeded is False
        assert result.released == ()
        assert after == env

    def test_failure_leaves_no_resize_script_effects(self):
        """A rejected update writes no script; any stale state must not leak into os.environ."""
        fake = FakeSlurm(disable_job_shrink=True)
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        assert after == env
        assert fake.job_nodes == NODES
        assert result.requested is True


# =============================================================================
# Verification with squeue
# =============================================================================


class TestVerification:
    def test_squeue_command(self):
        fake = FakeSlurm()
        run_shrink(fake, slurm_env(), NODES[0])
        assert [c.cmd for c in fake.squeue_calls()] == [["squeue", "-h", "-j", JOB_ID, "-o", "%N"]]

    def test_match_is_verified(self):
        fake = FakeSlurm()
        result, _ = run_shrink(fake, slurm_env(), NODES[0])
        assert result.verified is True
        assert result.reason is None

    @pytest.mark.parametrize(
        "fake_kwargs",
        [
            {"squeue_output": "\n"},
            {"squeue_mode": "fail"},
            {"squeue_mode": "raise"},
            {"squeue_mode": "timeout"},
        ],
        ids=["empty", "non-zero", "missing-binary", "timeout"],
    )
    def test_unverified_success_is_still_success(self, fake_kwargs):
        """squeue cannot answer: scontrol's exit code stands, and the missing verification is noted in ``reason``."""
        fake = FakeSlurm(**fake_kwargs)
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        assert result.requested is True
        assert result.succeeded is True
        assert result.verified is False
        assert isinstance(result.reason, str) and result.reason
        assert result.released == tuple(NODES[1:])
        assert after["SLURM_NODELIST"] == NODES[0]
        assert "SLURM_NTASKS" not in after

    @pytest.mark.parametrize(
        "squeue_output",
        ["gb200-[01-02]\n", "gb200-[01-04]\n", "gb200-02\n", "gb200-01,gb200-03\n"],
    )
    def test_squeue_showing_other_nodes_is_a_failure(self, squeue_output):
        """scontrol exited 0 but squeue says the job is not [keep]: env untouched, nothing released."""
        fake = FakeSlurm(squeue_output=squeue_output)
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        assert after == env, "env must not change before squeue confirms the shrink"
        assert result.requested is True
        assert result.succeeded is False
        assert result.verified is False
        assert result.released == ()
        assert isinstance(result.reason, str)
        shown = squeue_output.strip()
        assert shown in result.reason or any(n in result.reason for n in expand_hostlist(shown)), result.reason
        assert_temp_dirs_removed(fake)

    def test_noop_update_with_resize_script_changes_nothing(self):
        """An update equal to the submitted --nodelist is a successful no-op (job_mgr.c:13314-13317).

        scontrol still writes a resize script, for the unchanged 4-node allocation, including the
        task-variable unsets. squeue shows four nodes, so none of the script may be applied.
        """
        fake = FakeSlurm(submitted_nodelist=NODES[0])
        env = slurm_env()
        result, after = run_shrink(fake, env, NODES[0])
        assert fake.job_nodes == NODES
        assert len(fake.update_calls()) == 1
        assert after == env
        assert result.requested is True
        assert result.succeeded is False
        assert result.released == ()
        assert isinstance(result.reason, str) and result.reason
        assert_temp_dirs_removed(fake)

    def test_squeue_runs_before_env_is_touched(self):
        """When squeue runs, os.environ still holds the pre-shrink values."""
        fake = FakeSlurm()
        env = slurm_env()
        seen: list[dict[str, str]] = []
        original = fake._squeue

        def spy(cmd, kwargs):
            seen.append(dict(os.environ))
            return original(cmd, kwargs)

        fake._squeue = spy  # type: ignore[method-assign]
        result, _ = run_shrink(fake, env, NODES[0])
        assert result.succeeded is True
        assert seen == [env]

    def test_compressed_squeue_output_is_expanded(self):
        """squeue prints a ranged list; a match must compare expanded hostnames."""
        fake = FakeSlurm(batch_host="gb200-01", squeue_output="gb200-[01]\n")
        result, _ = run_shrink(fake, slurm_env(), NODES[0])
        assert result.verified is True
        assert result.reason is None
