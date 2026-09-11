# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`profiling.type: nsight-slurm` -- worker steps launched through the nsight-slurm wrapper.

The wrapper itself is exercised through a fake install: ``<home>/bin/nsight-slurm`` records
its argv and environment, ``<home>/bin/nsight-slurm-connector`` exists so `enable pyxis`
preconditions hold. Nothing here needs Slurm or the real tool.
"""

from __future__ import annotations

import json
import signal
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import TRTLLMProtocol
from srtctl.cli.mixins.nsight_slurm_stage import (
    NSIGHT_SLURM_LOG_NAME,
    NSIGHT_SLURM_REPORT_SUBDIR,
    NSIGHT_SLURM_RUNTIME_SUBDIR,
    NsightSlurmStageMixin,
)
from srtctl.core.schema import ModelConfig, ProfilingConfig, ProfilingPhaseConfig, ResourceConfig, SrtConfig
from srtctl.core.slurm import start_srun_process

FAKE_TOOL = """#!/usr/bin/env bash
# record every invocation: argv as JSON + selected env, one line each
python3 - "$@" <<'PY'
import json, os, sys
rec = {"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in ("NSIGHT_SLURM_HOME", "SLURM_SUBMIT_DIR", "SLURM_JOB_ID", "SLURMD_NODENAME")}, "cwd": os.getcwd()}
with open(os.environ["FAKE_NSIGHT_LOG"], "a") as f:
    f.write(json.dumps(rec) + "\\n")
PY
"""


def _fake_home(tmp_path: Path) -> Path:
    home = tmp_path / "nsight-slurm"
    (home / "bin").mkdir(parents=True)
    for name in ("nsight-slurm", "nsight-slurm-connector"):
        p = home / "bin" / name
        p.write_text(FAKE_TOOL)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return home


def _disagg(backend=TRTLLMProtocol(), **profiling_kwargs) -> SrtConfig:
    profiling_kwargs.setdefault("type", "nsight-slurm")
    profiling_kwargs.setdefault("prefill", ProfilingPhaseConfig(start_step=1200, stop_step=1300))
    profiling_kwargs.setdefault("decode", ProfilingPhaseConfig(start_step=6000, stop_step=6600))
    return SrtConfig(
        name="nsight",
        model=ModelConfig(path="/model", container="/container", precision="fp8"),
        resources=ResourceConfig(
            gpu_type="gb200", prefill_nodes=1, decode_nodes=1, prefill_workers=1, decode_workers=1
        ),
        profiling=ProfilingConfig(**profiling_kwargs),
        **({"backend": backend} if backend is not None else {}),
    )


class TestConfig:
    def test_type_and_helpers(self):
        p = ProfilingConfig(type="nsight-slurm", nsight_slurm_home="/shared/nsight-slurm")
        assert p.enabled and p.is_nsight_slurm and not p.is_nsys and not p.is_torch
        assert p.nsight_slurm_launcher() == ["/shared/nsight-slurm/bin/nsight-slurm", "srun"]
        assert p.nsight_slurm_bin("nsight-slurm-connector") == "/shared/nsight-slurm/bin/nsight-slurm-connector"
        assert p.nsight_slurm_process_env("/lustre/out/123/logs") == {
            "NSIGHT_SLURM_HOME": "/shared/nsight-slurm",
            "SLURM_SUBMIT_DIR": "/lustre/out/123/logs",
            "NSIGHT_SLURM_RUNTIME_DIR": "/nsrt",
        }
        # Connector-owned flags are absent from the default option set.
        opts = p.nsight_slurm_effective_tool_options()
        assert opts[:2] == ["-t", "cuda-sw,nvtx,python-gil"]
        assert "--cuda-graph-trace=graph" in opts and "--sample=none" in opts
        assert not any(o.startswith(("-o", "--force-overwrite", "-c", "--capture-range")) for o in opts)

    def test_worker_env_has_the_iteration_window_and_nvtx(self):
        p = ProfilingConfig(
            type="nsight-slurm",
            nsight_slurm_home="/s",
            decode=ProfilingPhaseConfig(6000, 6600),
            nvtx_injection_path="/usr/local/cuda-0.gpgpu/x/libToolsInjection64.so",
        )
        env = p.get_env_vars("decode", "/logs/profiles")
        assert env["TLLM_PROFILE_START_STOP"] == "6000-6600"
        assert env["TLLM_LLMAPI_ENABLE_NVTX"] == "1"
        assert env["TLLM_PROFILE_LOG_RANKS"] == "all"
        assert env["DYN_ENABLE_RUST_NVTX"] == "1"
        assert env["NVTX_INJECTION64_PATH"].endswith("libToolsInjection64.so")

    def test_no_nsys_prefix_for_nsight_slurm(self):
        p = ProfilingConfig(type="nsight-slurm", nsight_slurm_home="/s")
        assert p.get_nsys_prefix("/out", backend_type="trtllm") == []


class TestValidation:
    def test_valid(self):
        cfg = _disagg(nsight_slurm_home="/lustre/tools/nsight-slurm")
        assert cfg.profiling.is_nsight_slurm

    def test_home_required_and_absolute(self):
        with pytest.raises(ValidationError, match="nsight_slurm_home"):
            _disagg()
        with pytest.raises(ValidationError, match="nsight_slurm_home"):
            _disagg(nsight_slurm_home="relative/path")

    def test_profiling_mode_checked(self):
        with pytest.raises(ValidationError, match="nsight_slurm_profiling_mode"):
            _disagg(nsight_slurm_home="/s", nsight_slurm_profiling_mode="sometimes")

    def test_cuda_api_needs_trtllm(self):
        cfg = _disagg(nsight_slurm_home="/s")
        assert cfg.backend_type == "trtllm"
        with pytest.raises(ValidationError, match="cuda-api needs the TRT-LLM"):
            _disagg(nsight_slurm_home="/s", backend=None)
        # Other modes do not depend on the PyExecutor trigger.
        _disagg(nsight_slurm_home="/s", backend=None, nsight_slurm_profiling_mode="at-launch")

    def test_connector_owned_flags_rejected(self):
        with pytest.raises(ValidationError, match="connector-owned"):
            _disagg(nsight_slurm_home="/s", nsight_slurm_tool_options=["-t", "cuda-sw", "-o", "/x"])
        with pytest.raises(ValidationError, match="connector-owned"):
            _disagg(nsight_slurm_home="/s", nsight_slurm_tool_options=["--capture-range-end=stop"])


class TestLauncher:
    @patch("srtctl.core.slurm.subprocess.Popen")
    def test_start_srun_process_uses_the_wrapper(self, popen, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "4242")
        popen.return_value = MagicMock()
        start_srun_process(
            command=["trtllm-llmapi-launch", "python3", "-m", "dynamo.trtllm"],
            nodes=2,
            ntasks=8,
            nodelist=["n1", "n2"],
            output="/logs/w.out",
            container_image="/img.sqsh",
            container_mounts={Path("/logs"): Path("/logs")},
            env_to_set={"DYN_SYSTEM_PORT": "8081"},
            srun_export_env={"ENROOT_REMAP_ROOT": "yes"},
            mpi="pmix",
            srun_launcher=["/s/bin/nsight-slurm", "srun"],
            launcher_env={"NSIGHT_SLURM_HOME": "/s", "SLURM_SUBMIT_DIR": "/logs"},
            extra_container_mounts=[
                "/s:/s:ro",
                "/s/bin/nsight-slurm-connector:/usr/local/bin/nsight-slurm-connector:ro",
            ],
        )
        argv = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        assert argv[:2] == ["/s/bin/nsight-slurm", "srun"]
        # Exactly one --container-mounts flag, carrying the job mounts AND the wrapper's connector mounts:
        # pyxis applies only the last --container-mounts it is given, so they must never be split.
        assert argv.count("--container-mounts") == 1
        mounts = argv[argv.index("--container-mounts") + 1]
        assert mounts == "/logs:/logs,/s:/s:ro,/s/bin/nsight-slurm-connector:/usr/local/bin/nsight-slurm-connector:ro"
        # Native srun options are forwarded unchanged and terminated by `--`.
        assert "--jobid" in argv and "--mpi" in argv and "--container-image" in argv
        assert "--no-container-entrypoint" in argv
        sep = argv.index("--")
        assert argv[sep + 1 : sep + 3] == ["bash", "-c"]
        assert "export DYN_SYSTEM_PORT=8081" in argv[sep + 3]
        # The wrapper appends its own --export=ALL, so srtctl must not add one ...
        assert not any(a.startswith("--export") for a in argv)
        # ... and the task-environment variables ride along in the wrapper's process env.
        assert env["ENROOT_REMAP_ROOT"] == "yes"
        assert env["NSIGHT_SLURM_HOME"] == "/s"
        assert env["SLURM_SUBMIT_DIR"] == "/logs"

    @patch("srtctl.core.slurm.subprocess.Popen")
    def test_plain_srun_unchanged(self, popen):
        popen.return_value = MagicMock()
        start_srun_process(command=["true"], srun_export_env={"ENROOT_REMAP_ROOT": "yes"})
        argv = popen.call_args.args[0]
        assert argv[0] == "srun" and "--" not in argv
        assert any(a == "--export=ALL,ENROOT_REMAP_ROOT=yes" for a in argv)
        assert popen.call_args.kwargs["env"] is None


class TestStage:
    def _harness(self, tmp_path, home):
        class Harness(NsightSlurmStageMixin):
            def __init__(self):
                self.config = _disagg(nsight_slurm_home=str(home))
                self.runtime = SimpleNamespace(
                    log_dir=tmp_path / "logs", container_image="/img.sqsh", head_node_ip="10.0.0.7"
                )
                self.runtime.log_dir.mkdir()

        return Harness()

    def test_configures_once_and_starts_the_coordinator(self, tmp_path, monkeypatch):
        home = _fake_home(tmp_path)
        rec = tmp_path / "calls.jsonl"
        monkeypatch.setenv("FAKE_NSIGHT_LOG", str(rec))
        monkeypatch.setenv("SLURM_JOB_ID", "777")
        h = self._harness(tmp_path, home)

        kwargs = h.nsight_slurm_launch_kwargs()
        kwargs2 = h.nsight_slurm_launch_kwargs()  # idempotent: configuration runs once

        assert kwargs == kwargs2
        assert kwargs["srun_launcher"] == [str(home / "bin" / "nsight-slurm"), "srun"]
        assert kwargs["launcher_env"]["SLURM_SUBMIT_DIR"] == str(h.runtime.log_dir)
        calls = [json.loads(line) for line in rec.read_text().splitlines()]
        argvs = [c["argv"] for c in calls]
        assert argvs[0] == ["configure", "tool-path", "/usr/local/bin/nsys"]
        assert ["configure", "tool-command", "profile"] in argvs
        assert ["configure", "profiling-mode", "cuda-api"] in argvs
        tool_opts = next(a for a in argvs if a[:2] == ["configure", "tool-options"])
        assert tool_opts[2:] == list(ProfilingConfig.NSIGHT_SLURM_DEFAULT_TOOL_OPTIONS)
        assert ["configure", "report-output", str(h.runtime.log_dir / NSIGHT_SLURM_REPORT_SUBDIR)] in argvs
        assert (h.runtime.log_dir / NSIGHT_SLURM_RUNTIME_SUBDIR).is_dir()
        assert kwargs["launcher_env"]["NSIGHT_SLURM_RUNTIME_DIR"] == "/nsrt"
        assert ["disable", "pyxis"] in argvs and ["enable", "pyxis"] not in argvs
        # The connector's mounts ride in srtctl's own --container-mounts (pyxis keeps only the last flag).
        log_dir = str(h.runtime.log_dir)
        assert kwargs["extra_container_mounts"] == [
            f"{home}:{home}:ro",
            f"{home}/bin/nsight-slurm-connector:/usr/local/bin/nsight-slurm-connector:ro",
            f"{log_dir}:{log_dir}",
            f"{log_dir}/{NSIGHT_SLURM_RUNTIME_SUBDIR}:/nsrt",
        ]
        assert argvs[-1] == ["coordinator", "start"]
        # Every call ran from the log dir with the wrapper's job identity redirected there.
        assert {c["cwd"] for c in calls} == {str(h.runtime.log_dir)}
        assert {c["env"]["SLURM_SUBMIT_DIR"] for c in calls} == {str(h.runtime.log_dir)}
        assert {c["env"]["NSIGHT_SLURM_HOME"] for c in calls} == {str(home)}
        # The coordinator address is published from SLURMD_NODENAME: force the head node's IPv4 literal so
        # ranks on the coordinator's own node do not resolve its name to an unconnectable IPv6 (job 571147).
        assert {c["env"]["SLURMD_NODENAME"] for c in calls} == {"10.0.0.7"}
        # ... but the wrapper-launched srun steps must not carry that override into the tasks.
        assert "SLURMD_NODENAME" not in kwargs["launcher_env"]
        assert (h.runtime.log_dir / NSIGHT_SLURM_LOG_NAME).read_text().count("$ ") == len(calls)

        h.stop_nsight_slurm()
        calls = [json.loads(line) for line in rec.read_text().splitlines()]
        assert calls[-1]["argv"] == ["coordinator", "stop", "--force"]
        h.stop_nsight_slurm()  # second stop is a no-op
        assert len(rec.read_text().splitlines()) == len(calls)

    def test_missing_install_is_a_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "777")
        h = self._harness(tmp_path, tmp_path / "nowhere")
        with pytest.raises(RuntimeError, match="nsight-slurm executable missing"):
            h.nsight_slurm_launch_kwargs()

    def test_other_profiling_types_do_not_touch_the_wrapper(self, tmp_path):
        class Harness(NsightSlurmStageMixin):
            def __init__(self):
                self.config = SimpleNamespace(profiling=ProfilingConfig(type="nsys"))
                self.runtime = SimpleNamespace(log_dir=tmp_path)

        assert Harness().nsight_slurm_launch_kwargs() == {}


class TestFlush:
    """Reports only exist once the nsys sessions end: flush before cleanup kills the steps."""

    def _ready(self, tmp_path, monkeypatch):
        home = _fake_home(tmp_path)
        rec = tmp_path / "calls.jsonl"
        monkeypatch.setenv("FAKE_NSIGHT_LOG", str(rec))
        monkeypatch.setenv("SLURM_JOB_ID", "777")

        class Harness(NsightSlurmStageMixin):
            def __init__(self):
                self.config = _disagg(nsight_slurm_home=str(home))
                self.runtime = SimpleNamespace(
                    log_dir=tmp_path / "logs", container_image="/img.sqsh", head_node_ip="10.0.0.7"
                )
                self.runtime.log_dir.mkdir()

        h = Harness()
        h.nsight_slurm_launch_kwargs()  # configures + starts the coordinator (fake)
        rec.write_text("")  # only record the flush from here on
        return h, rec

    @staticmethod
    def _registry(*names):
        procs = {}
        for name in names:
            procs[name] = SimpleNamespace(name=name, is_running=True, popen=MagicMock())
        return SimpleNamespace(get_all_processes=lambda: dict(procs)), procs

    def test_stop_then_wait_for_reports(self, tmp_path, monkeypatch):
        h, rec = self._ready(tmp_path, monkeypatch)
        registry, procs = self._registry("prefill_0_n1", "decode_0_n2", "frontend_0_n2")
        # A report already exists (the connectors wrote it after `stop`): no worker gets signalled.
        rep = h.runtime.log_dir / NSIGHT_SLURM_REPORT_SUBDIR / "job-777" / "collection-1"
        rep.mkdir(parents=True)
        (rep / "n1-rank0.nsys-rep").write_bytes(b"x")
        n = h.flush_nsight_slurm(registry, timeout_s=2.0, first_wait_s=0.5, settle_s=0.2, poll_s=0.05)
        assert n == 1
        calls = [json.loads(line)["argv"] for line in rec.read_text().splitlines()]
        assert calls == [["stop", "--job", "777", "--timeout", "30"]]
        for p in procs.values():
            p.popen.send_signal.assert_not_called()

    def test_falls_back_to_sigterm_on_worker_steps(self, tmp_path, monkeypatch):
        h, _rec = self._ready(tmp_path, monkeypatch)
        registry, procs = self._registry("prefill_0_n1", "decode_0_n2", "frontend_0_n2", "etcd")
        # A range report left behind in a connector runtime workspace is rescued into the report root.
        scratch = (
            h.runtime.log_dir / NSIGHT_SLURM_RUNTIME_SUBDIR / "nsight-slurm-1000" / "step" / "ranks" / "3" / "reports"
        )
        scratch.mkdir(parents=True)
        (scratch / "prefill_n1_3.nsys-rep").write_bytes(b"data")
        n = h.flush_nsight_slurm(registry, timeout_s=0.6, first_wait_s=0.1, settle_s=0.1, poll_s=0.05)
        assert n == 1
        rescued = h.runtime.log_dir / NSIGHT_SLURM_REPORT_SUBDIR / "rescued"
        assert (
            rescued / "nsight-slurm-1000" / "step" / "ranks" / "3" / "reports" / "prefill_n1_3.nsys-rep"
        ).read_bytes() == b"data"
        # Only the worker steps (wrapper processes) are signalled, with a plain SIGTERM (no reap).
        procs["prefill_0_n1"].popen.send_signal.assert_called_once_with(signal.SIGTERM)
        procs["decode_0_n2"].popen.send_signal.assert_called_once_with(signal.SIGTERM)
        procs["frontend_0_n2"].popen.send_signal.assert_not_called()
        procs["etcd"].popen.send_signal.assert_not_called()

    def test_noop_without_coordinator(self, tmp_path):
        class Harness(NsightSlurmStageMixin):
            def __init__(self):
                self.config = SimpleNamespace(profiling=ProfilingConfig(type="nsys"))
                self.runtime = SimpleNamespace(log_dir=tmp_path)

        assert Harness().flush_nsight_slurm(None) == 0
