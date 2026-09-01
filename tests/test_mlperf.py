# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MLPerf benchmark driver script."""

from srtctl.benchmarks import list_benchmarks
from srtctl.benchmarks.base import SCRIPTS_DIR

BENCH = SCRIPTS_DIR / "mlperf" / "bench.sh"


class TestMLPerfScript:
    """MLPerf runs as a `custom` benchmark driving the MLPerf team's client.

    Nothing about it lives in srt-slurm's schema: the client's config has ~60
    nested settings whose shape moves with the client version, so the script
    passes it through and rewrites only what the cluster decides.
    """

    def test_no_mlperf_benchmark_type(self):
        """mlperf is deliberately not a registered benchmark type."""
        assert "mlperf" not in list_benchmarks()

    def test_script_ships_with_the_package(self):
        """scripts/ is mounted at /srtctl-benchmarks for every benchmark type."""
        assert BENCH.exists()

    def test_takes_a_client_config_rather_than_modelling_its_settings(self):
        script = BENCH.read_text()
        assert "MLPERF_CLIENT_CONFIG" in script
        # A knob for any individual client setting would be the wrapper this
        # design exists to avoid.
        for leaked in ("MLPERF_TARGET_QPS", "MLPERF_MAX_NEW_TOKENS", "MLPERF_NUM_WORKERS"):
            assert leaked not in script, f"{leaked} re-models a client config setting"

    def test_rewrites_only_endpoints_and_report_dir(self):
        """The two things the client config cannot know before the cluster exists."""
        script = BENCH.read_text()
        assert 'setdefault("endpoint_config", {})["endpoints"]' in script
        assert 'config["report_dir"]' in script

    def test_requires_an_injected_frontend(self):
        """No localhost default: benchmarking nothing must fail, not pass."""
        assert "SRT_FRONTEND_HOST is not set" in BENCH.read_text()

    def test_supports_multiple_endpoints(self):
        """The client load-balances across the list itself."""
        assert "MLPERF_ENDPOINTS" in BENCH.read_text()

    def test_writes_the_rollup_srtctl_already_consumes(self):
        """No new rollup concept: postprocess reads benchmark-rollup.json if present."""
        assert "benchmark-rollup.json" in BENCH.read_text()
