# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the retained raw host/process telemetry sampler."""


class TestHostSamplerProcessSelection:
    """Real processes must outrank launcher wrappers within the sample budget."""

    def test_wrappers_sort_after_real_processes(self, monkeypatch):
        from srtctl.analysis import host_sampler as hs

        procs = {  # pid -> (cmdline, comm)
            "10": ("srun --overlap dynamo-worker", "srun"),
            "11": ("srun --overlap dynamo-worker", "srun"),
            "12": ("python -m dynamo.trtllm", "trtllm-llmapi-l"),
            "13": ("aiperf profile --model dynamo", "aiperf system_c"),
            "14": ("bash -c dynamo", "bash"),
        }
        monkeypatch.setattr(hs.os, "listdir", lambda p: list(procs) if p == "/proc" else [])
        monkeypatch.setattr(
            hs,
            "_read",
            lambda path: (
                procs[path.split("/")[2]][0]
                if path.endswith("cmdline")
                else procs[path.split("/")[2]][1]
                if path.endswith("comm")
                else None
            ),
        )

        got = hs._interesting_pids()
        assert set(got[:2]) == {12, 13}, f"real processes must come first, got {got}"
        assert set(got[2:]) == {10, 11, 14}

    def test_budget_drops_wrappers_not_workers(self, monkeypatch):
        from srtctl.analysis import host_sampler as hs

        procs = {str(i): ("srun dynamo", "srun") for i in range(20, 60)}
        procs["12"] = ("python -m dynamo.trtllm", "trtllm-llmapi-l")
        monkeypatch.setattr(hs.os, "listdir", lambda p: list(procs) if p == "/proc" else [])
        monkeypatch.setattr(
            hs,
            "_read",
            lambda path: (
                procs[path.split("/")[2]][0]
                if path.endswith("cmdline")
                else procs[path.split("/")[2]][1]
                if path.endswith("comm")
                else None
            ),
        )

        got = hs._interesting_pids(limit=5)
        assert 12 in got, "the one real worker must survive a budget full of wrappers"
        assert len(got) == 5

    def test_pressure_parses_psi_totals(self, monkeypatch):
        from srtctl.analysis import host_sampler as hs

        psi = {
            "/proc/pressure/cpu": "some avg10=0.00 avg60=0.10 avg300=0.05 total=123456\n",
            "/proc/pressure/memory": (
                "some avg10=0.00 avg60=0.00 avg300=0.00 total=7890\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total=4200\n"
            ),
            "/proc/pressure/io": "some avg10=0.00 avg60=0.00 avg300=0.00 total=99\nfull avg10=0.00 total=55\n",
        }
        monkeypatch.setattr(hs, "_read", lambda path: psi.get(path))

        got = hs._pressure()
        assert got == {
            "cpu_some_total_us": 123456,
            "memory_some_total_us": 7890,
            "memory_full_total_us": 4200,
            "io_some_total_us": 99,
            "io_full_total_us": 55,
        }

    def test_pressure_absent_psi_yields_empty(self, monkeypatch):
        from srtctl.analysis import host_sampler as hs

        monkeypatch.setattr(hs, "_read", lambda path: None)  # kernel without CONFIG_PSI
        assert hs._pressure() == {}

