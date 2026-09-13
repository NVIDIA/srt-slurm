# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit + golden-file tests for the AgentPerf harness-run porter.

The golden e2e test pins the porter's behavior over time against a
materialized copy of a real harness run directory
(tests/fixtures/agentperf_porter/c1010 — the DSV4 c1010 GB300 baseline with
usernames anonymized, content otherwise verbatim: real engine configs, the
real 35-token worker env, the real client banner, real srun line shapes).

To regenerate the goldens after an INTENTIONAL behavior change:

    python3 src/srtctl/benchmarks/scripts/agentperf/port_harness_run.py \
        tests/fixtures/agentperf_porter/c1010 \
        --out /tmp/recipe.yaml --workload-out /tmp/workload.yaml \
        --dataset-root tests/fixtures/agentperf_porter/c1010/datasets

then replace the fixture-root prefix with <FIXTURE> and the workload output
path with <WORKLOAD_OUT> and copy over expected-{recipe,workload}.yaml.
"""

import importlib.util
from pathlib import Path

import yaml

from srtctl.benchmarks.base import SCRIPTS_DIR

_spec = importlib.util.spec_from_file_location(
    "port_harness_run_golden", SCRIPTS_DIR / "agentperf" / "port_harness_run.py"
)
porter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(porter)

FIXTURE = Path(__file__).parent / "fixtures" / "agentperf_porter"


class TestRewriteUtils:
    def test_rewrite_prefix_match(self):
        assert porter.rewrite("/lustre/a/b", [("/lustre/", "/scratch/")]) == "/scratch/a/b"

    def test_rewrite_no_match_untouched(self):
        assert porter.rewrite("/data/lustre/a", [("/lustre/", "/scratch/")]) == "/data/lustre/a"

    def test_rewrite_first_matching_rule_wins(self):
        rules = [("/a/", "/x/"), ("/a/b/", "/y/")]
        assert porter.rewrite("/a/b/c", rules) == "/x/b/c"

    def test_rewrite_tree_nested(self):
        obj = {"k": "/lustre/x", "l": ["/lustre/y", {"m": "/lustre/z"}], "n": 3, "o": None}
        out = porter.rewrite_tree(obj, [("/lustre/", "/scratch/")])
        assert out == {"k": "/scratch/x", "l": ["/scratch/y", {"m": "/scratch/z"}], "n": 3, "o": None}

    def test_absolute_paths_collects_nested_strings(self):
        obj = {"a": "/one", "b": ["x", "/two", {"c": "/three", "d": 5}]}
        assert sorted(porter.absolute_paths(obj)) == ["/one", "/three", "/two"]


class TestEnvTranslation:
    def test_drop_rename_passthrough(self):
        env = porter.translate_env(
            ["CUDA_VISIBLE_DEVICES=0,1", "PATH=/x", "DYN_KV_BLOCK_SIZE=128",
             "DYN_UCX_TLS=sm,tcp", "TLLM_LOG_LEVEL=INFO"]
        )
        assert "CUDA_VISIBLE_DEVICES" not in env and "PATH" not in env
        assert env["DYN_TRTLLM_KV_BLOCK_SIZE"] == "128"
        assert env["UCX_TLS"] == "sm,tcp"
        assert env["TLLM_LOG_LEVEL"] == "INFO"

    def test_dyn_log_pinned_but_not_overridden(self):
        assert porter.translate_env([])["DYN_LOG"] == "info"
        assert porter.translate_env(["DYN_LOG=debug"])["DYN_LOG"] == "debug"

    def test_quoted_env_tokens_picks_env_list_not_scalar_args(self):
        line = "srun x '1010' y '0-0' z 'A_B=1 C_D=2 E_F=3' w 'single'"
        assert porter._quoted_env_tokens(line) == ["A_B=1", "C_D=2", "E_F=3"]

    def test_quoted_env_tokens_none_when_no_env_list(self):
        assert porter._quoted_env_tokens("srun '1010' 'plain words here'") == []


class TestFrontendArgs:
    def test_defaults_and_kv_events_off(self):
        args = porter.build_frontend_args({})
        assert args["router-mode"] == "kv"
        assert args["no-router-kv-events"] is True
        assert args["kv-cache-block-size"] == 128
        assert args["enforce-disagg"] is True
        assert "dyn-tool-call-parser" not in args

    def test_env_driven_values(self):
        args = porter.build_frontend_args(
            {"ROUTER_MODE": "round-robin", "DYN_FRONTEND_ENABLE_KV_EVENTS": "1",
             "DYN_KV_BLOCK_SIZE": "64", "DYN_REQUEST_PLANE": "nats"}
        )
        assert args["router-mode"] == "round-robin"
        assert args["no-router-kv-events"] is False
        assert args["kv-cache-block-size"] == 64
        assert args["request-plane"] == "nats"


class TestSettleStem:
    def test_settle_regex_int_and_float(self):
        assert porter.SETTLE_RE.search("m__1010u__phase0__dur2400s__settle240s__traj3.json").group(1) == "240"
        assert porter.SETTLE_RE.search("m__8u__phase1__dur300.5s__settle2.5s__traj3.txt").group(1) == "2.5"


class TestGoldenEndToEnd:
    """Run the porter on the materialized real run dir; outputs must be byte-stable."""

    def _normalize(self, text: str, workload_out: Path) -> str:
        return text.replace(str(FIXTURE), "<FIXTURE>").replace(str(workload_out.resolve()), "<WORKLOAD_OUT>")

    def test_golden_outputs(self, tmp_path):
        out, wl = tmp_path / "recipe.yaml", tmp_path / "workload.yaml"
        rc = porter.main([
            str(FIXTURE / "c1010"), "--out", str(out), "--workload-out", str(wl),
            "--dataset-root", str(FIXTURE / "c1010" / "datasets"),
        ])
        assert rc == 0
        for generated, golden in ((out, "expected-recipe.yaml"), (wl, "expected-workload.yaml")):
            got = self._normalize(generated.read_text(), wl)
            want = (FIXTURE / golden).read_text()
            assert got == want, (
                f"{golden} drifted from the golden output.\n"
                "If the change is intentional, regenerate the goldens (see module docstring)."
            )

    def test_golden_recipe_loads_into_schema(self, tmp_path):
        """The golden recipe (with placeholders filled) must satisfy SrtConfig's schema."""
        from srtctl.core.schema import SrtConfig

        raw = (FIXTURE / "expected-recipe.yaml").read_text()
        raw = raw.replace("<FIXTURE>", str(FIXTURE)).replace("<WORKLOAD_OUT>", "/workloads/w.yaml")
        cfg = SrtConfig.Schema().load(yaml.safe_load(raw))
        assert cfg.benchmark.type == "agentperf"
        assert cfg.resources.prefill_workers == 5
        assert cfg.backend.numa_cpu_bind is False
