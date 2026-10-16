# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for power_control: EPD node_agents and host appliers per pool.

Pools and GPU groups come from the workers' GPU map; the stage renders the shared
node_agent/applier command lines, launches them the way the telemetry stage launches
its exporters, and lifts the caps on the way out.
"""

import dataclasses
import logging
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import TRTLLMBackend
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.power_control_stage import (
    DECISION_FILE_RELPATH,
    POWER_TERMINATE_TIMEOUT_SECONDS,
    build_applier_command,
    build_node_agent_command,
    compute_power_pools,
    node_agent_environment,
    node_agent_terminate_timeout,
    power_reset_script,
)
from srtctl.core.processes import ProcessRegistry
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import (
    FrontendConfig,
    ModelConfig,
    PowerControlConfig,
    PowerPoolConfig,
    ResourceConfig,
    RoleConfig,
    SrtConfig,
    validate_power_fom,
)
from srtctl.core.topology import allocate_endpoints, endpoints_to_processes

EPD_SOURCE = "/lustre/share/nvidia-epd/src/epd"
POWER_SOURCE = "/lustre/share/dynamo/components/src/dynamo/power"
SITE_PACKAGES = "/opt/dynamo/venv/lib/python3.12/site-packages"

BASE_RECIPE = {
    "schema": 2,
    "name": "power-control-test",
    "model": {"path": "/models/test", "container": "test.sqsh", "precision": "fp8"},
    "resources": {"gpu_type": "gb200", "gpus_per_node": 4},
    "roles": {
        "prefill": {"nodes": 2, "workers": 2},
        "decode": {"nodes": 4, "workers": 2},
    },
}
ENABLED = {"enabled": True, "budget_w_per_gpu": 1500, "fom": "tokens_per_gpu", "epd_source": EPD_SOURCE}


def _load_recipe(power_control: dict | None = None, **overrides) -> SrtConfig:
    data = {**BASE_RECIPE, **overrides}
    if power_control is not None:
        data["power_control"] = power_control
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        return SrtConfig.from_yaml(Path(f.name))


def _power(**overrides) -> PowerControlConfig:
    return PowerControlConfig(**{**ENABLED, **overrides})


def _processes(
    *,
    prefill: tuple[int, int] = (2, 4),
    decode: tuple[int, int] = (2, 8),
    gpus_per_node: int = 4,
    nodes: int = 8,
    pack: bool = False,
    colocate: bool = False,
):
    """Worker processes for ``(workers, gpus_per_worker)`` per role over ``nodes`` nodes."""
    endpoints = allocate_endpoints(
        num_prefill=prefill[0],
        num_decode=decode[0],
        num_agg=0,
        gpus_per_prefill=prefill[1],
        gpus_per_decode=decode[1],
        gpus_per_agg=0,
        gpus_per_node=gpus_per_node,
        available_nodes=[f"node{i}" for i in range(nodes)],
        allow_prefill_decode_colocation=colocate,
        pack_multinode_workers=pack,
    )
    return endpoints_to_processes(endpoints)


def _config(frontend: FrontendConfig | None = None, engine=None, **power_overrides) -> SrtConfig:
    return SrtConfig(
        name="power-control-test",
        model=ModelConfig(path="/models/test", container="test.sqsh", precision="fp8"),
        resources=ResourceConfig(gpu_type="gb200", gpus_per_node=4),
        roles={"prefill": RoleConfig(nodes=2, workers=2), "decode": RoleConfig(nodes=4, workers=2)},
        # One frontend on the head node unless a test asks for the nginx topology.
        frontend=frontend or FrontendConfig(enable_multiple_frontends=False),
        engine=engine,
        power_control=_power(**power_overrides),
    )


def _runtime(tmp_path: Path, environment: dict[str, str] | None = None) -> RuntimeContext:
    return RuntimeContext(
        job_id="12345",
        run_name="test-run",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=tuple(f"node{i}" for i in range(6))),
        head_node_ip="10.0.0.0",
        infra_node_ip="10.0.0.0",
        log_dir=tmp_path,
        model_path=Path("/models/test"),
        container_image=Path("/img.sqsh"),
        gpus_per_node=4,
        network_interface=None,
        container_mounts={tmp_path: Path("/logs")},
        # Non-empty so a step that forgets the recipe's srun options is told apart from one that passes them.
        srun_options={"mpi": "none"},
        environment={"UV_CACHE_DIR": "/cache"} if environment is None else environment,
    )


def _running(name: str = "step", order: list[str] | None = None) -> MagicMock:
    """A Popen stand-in that is running until terminated, recording the termination order."""
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0

    def terminate() -> None:
        if order is not None:
            order.append(name)
        proc.poll.return_value = -15

    proc.terminate.side_effect = terminate
    return proc


def _node_ip(node: str, interface: str | None = None) -> str:
    return f"10.0.0.{node.removeprefix('node')}"


class TestPools:
    def test_per_role_pools_follow_node_lines(self):
        pools = compute_power_pools(_power(pools="per-role"), _processes())

        assert [pool.name for pool in pools] == ["prefill", "decode"]
        prefill, decode = pools
        assert prefill.nodes == ("node0", "node1")
        assert decode.nodes == ("node2", "node3", "node4", "node5")
        assert all(group.gpu_indices == frozenset({0, 1, 2, 3}) for pool in pools for group in pool.groups)
        assert (prefill.gpu_count, decode.gpu_count) == (8, 16)
        # Rank 0 is the pool's first node; the buffer comes from the role.
        assert prefill.groups[0].node == "node0"
        assert (prefill.ubpc_buffer, decode.ubpc_buffer) == (2.0, 1.1)

    def test_shared_pool_spans_both_roles(self):
        (pool,) = compute_power_pools(_power(pools="shared"), _processes())

        assert pool.name == "shared"
        assert pool.nodes == ("node0", "node1", "node2", "node3", "node4", "node5")
        assert pool.roles == ("prefill", "decode")
        # A shared pool takes the largest buffer of its roles so prefill bursts are not clipped.
        assert pool.ubpc_buffer == 2.0

    def test_tep6_packing_makes_one_group_per_shared_node(self):
        """5 TEP6 workers on 8 four-GPU nodes: adjacent workers share a node (4+2 / 2+4), the last node idles 2 GPUs."""
        processes = _processes(prefill=(5, 6), decode=(0, 8), pack=True)

        (pool,) = compute_power_pools(_power(pools="per-role"), processes)

        assert pool.nodes == tuple(f"node{i}" for i in range(8))
        assert pool.gpu_count == 30
        shared = next(group for group in pool.groups if group.node == "node1")
        assert shared.gpu_indices == frozenset({0, 1, 2, 3})
        assert shared.roles == ("prefill",)
        assert {process.endpoint_index for process in shared.processes} == {0, 1}
        tail = next(group for group in pool.groups if group.node == "node7")
        assert tail.gpu_indices == frozenset({0, 1})
        assert tail.gpu_list == "0,1"

    def test_a_node_split_between_two_pools_is_refused(self):
        processes = _processes(prefill=(1, 2), decode=(1, 2), nodes=1, colocate=True)

        with pytest.raises(ValueError, match="exactly one pool"):
            compute_power_pools(_power(pools="per-role"), processes)

        (pool,) = compute_power_pools(_power(pools="shared"), processes)
        assert pool.groups[0].gpu_indices == frozenset({0, 1, 2, 3})
        assert pool.groups[0].roles == ("prefill", "decode")

    def test_explicit_role_pools_carry_their_own_knobs(self):
        power = _power(
            budget_w_per_gpu=1300,
            ubpc_buffer={"decode": 1.05},
            pools=[
                PowerPoolConfig(
                    name="ctx", roles=["prefill"], budget_w_per_gpu=1800, fom="ctx_tokens", ubpc_buffer=2.0
                ),
                PowerPoolConfig(name="gen", roles=["decode"], fom="gen_tokens"),
            ],
        )

        ctx, gen = compute_power_pools(power, _processes())

        assert (ctx.spec.budget_w_per_gpu, ctx.spec.fom, ctx.ubpc_buffer) == (1800, "ctx_tokens", 2.0)
        assert (gen.spec.budget_w_per_gpu, gen.spec.fom, gen.ubpc_buffer) == (1300, "gen_tokens", 1.05)
        assert gen.nodes == ("node2", "node3", "node4", "node5")

    def test_explicit_node_pools_take_the_buffer_of_the_roles_they_run(self):
        power = _power(
            pools=[
                PowerPoolConfig(name="feed-a", nodes=["node2", "node3"]),
                PowerPoolConfig(name="feed-b", nodes=["node4", "node5"]),
            ]
        )

        feed_a, feed_b = compute_power_pools(power, _processes())

        assert feed_a.roles == ("decode",)
        assert (feed_a.ubpc_buffer, feed_b.ubpc_buffer) == (1.1, 1.1)
        assert feed_b.gpu_count == 8

    def test_explicit_node_pool_naming_an_idle_node_is_refused(self):
        power = _power(pools=[PowerPoolConfig(name="feed-a", nodes=["node2", "node9"])])

        with pytest.raises(ValueError, match="node9 run no worker"):
            compute_power_pools(power, _processes())


class TestCommands:
    def test_node_agent_command_matches_the_contract(self):
        power = _power(
            mode="both",
            pools="per-role",
            interval_s=20,
            perf_threshold=0.05,
            min_tgp_w=900,
            max_tgp_w=1400,
            warmup_policy="fixed",
        )
        prefill = compute_power_pools(power, _processes())[0]
        urls = ["http://10.0.0.0:8000/metrics", "http://10.0.0.1:7500/metrics"]

        command = build_node_agent_command(
            power, prefill, prefill.groups[1], 1, urls, Path("/logs/power_control/prefill"), 4
        )

        assert command == [
            "python3",
            "-m",
            "dynamo.power.node_agent",
            "--pool-name",
            "prefill",
            "--rank",
            "1",
            "--nodelist",
            "node0,node1",
            "--log-dir",
            "/logs/power_control/prefill",
            "--gpu-indices",
            "0,1,2,3",
            "--mode",
            "both",
            "--budget-w-per-gpu",
            "1500",
            "--fom",
            "tokens_per_gpu",
            "--interval-s",
            "20",
            "--perf-threshold",
            "0.05",
            "--ubpc-buffer",
            "2.0",
            "--max-tgp-w",
            "1400",
            "--warmup-policy",
            "fixed",
            "--metrics-url",
            "http://10.0.0.0:8000/metrics,http://10.0.0.1:7500/metrics",
            "--gpus-per-node",
            "4",
            "--pool-gpu-count",
            "8",
        ]
        # The floor is the applier's job (EPD's cap law has none); the node_agent refuses the flag.
        assert "--min-tgp-w" not in command

    def test_node_agent_without_tgp_bounds_omits_the_flags(self):
        power = _power(pools="per-role")
        prefill = compute_power_pools(power, _processes())[0]

        command = build_node_agent_command(
            power, prefill, prefill.groups[0], 0, ["http://f:8000/metrics"], Path("/l"), 4
        )

        assert "--min-tgp-w" not in command
        assert "--max-tgp-w" not in command
        assert command[command.index("--rank") + 1] == "0"

    def test_node_agent_needs_at_least_one_metrics_url(self):
        power = _power()
        (pool,) = compute_power_pools(power, _processes())

        with pytest.raises(ValueError, match="no metrics URL"):
            build_node_agent_command(power, pool, pool.groups[0], 0, [], Path("/l"), 4)

    def test_node_agent_environment_names_the_pool_roster_and_the_file_actuator(self):
        (pool,) = compute_power_pools(_power(), _processes())

        env = node_agent_environment(pool, Path("/logs/power_control/shared"), 4)

        assert env == {
            "LOG_DIR": "/logs/power_control/shared",
            "EPD_LOG_DIR": "/logs/power_control/shared",
            "EPD_ACTUATOR": "file",
            "EPD_NODELIST": "node0,node1,node2,node3,node4,node5",
            "EPD_NUM_NODES": "6",
            "GPUS_PER_NODE": "4",
            "EPD_RUN_TRACE": "1",
            "EPD_SUMMARY_STDOUT": "1",
        }

    def test_applier_command_matches_the_contract(self):
        power = _power(pools="per-role", applier="dcgmi", stale_after_s=120, poll_s=2)
        prefill = compute_power_pools(power, _processes())[0]
        decision = Path("/runs/1/logs/power_control/prefill") / DECISION_FILE_RELPATH
        trace = Path("/runs/1/logs/power_control/prefill/applier_node1.csv")

        command = build_applier_command(power, prefill, prefill.groups[1], decision, trace, python="/venv/bin/python3")

        assert command == [
            "/venv/bin/python3",
            "-m",
            "dynamo.power.applier",
            "--decision-file",
            "/runs/1/logs/power_control/prefill/epd/actuation/power_decision.json",
            "--node",
            "node1",
            "--gpu-indices",
            "0,1,2,3",
            "--mode",
            "dcgmi",
            "--budget-w-per-gpu",
            "1500",
            "--stale-after-s",
            "120",
            "--poll-s",
            "2",
            "--restore-default-on-exit",
            "--trace-csv",
            "/runs/1/logs/power_control/prefill/applier_node1.csv",
        ]

    def test_applier_carries_the_per_gpu_floor(self):
        power = _power(pools="per-role", min_tgp_w=900)
        prefill = compute_power_pools(power, _processes())[0]

        command = build_applier_command(power, prefill, prefill.groups[0], Path("/d.json"), Path("/t.csv"))

        assert command[-2:] == ["--min-w-per-gpu", "900"]
        assert "--min-tgp-w" not in command

    def test_reset_scripts_use_the_applier_path_per_gpu_index(self):
        smi = power_reset_script("sudo-smi", frozenset({0, 1}))
        assert smi.startswith("rc=0; for i in 0 1; do ")
        # `sudo -n` fails at once without a sudoers rule; the loop's status covers every GPU, not the last one.
        assert 'sudo -n nvidia-smi -pl "${d%%.*}" -i "$i" || rc=1; done; exit $rc' in smi
        assert "sudo nvidia-smi" not in smi
        assert "power.default_limit" in smi

        dcgmi = power_reset_script("dcgmi", frozenset({0, 1, 2, 3}))
        assert "dcgmi group -c srtctl_power_reset -a 0,1,2,3" in dcgmi
        assert 'dcgmi config -g "$g" --set -P "${d%%.*}" || rc=1; dcgmi group -d "$g"; exit $rc' in dcgmi
        # A failed default-limit read or group creation fails the step instead of running `dcgmi config` blind.
        assert dcgmi.count("|| exit 1") == 3

    @staticmethod
    def _shim(directory: Path, name: str, body: str) -> None:
        shim = directory / name
        shim.write_text(f"#!/bin/bash\n{body}\n")
        shim.chmod(0o755)

    def test_sudo_smi_reset_exit_status_reports_a_cap_left_on_any_gpu(self, tmp_path):
        """With shims for nvidia-smi and sudo: GPU 0's reset fails, GPU 1's succeeds, the step still exits 1."""
        shims = tmp_path / "bin"
        shims.mkdir()
        calls = tmp_path / "calls.log"
        self._shim(shims, "nvidia-smi", 'printf "1000.00\\n"')
        self._shim(
            shims,
            "sudo",
            f'printf "%s\\n" "$*" >> {calls}; [ "$1" = "-n" ] || exit 99; case "$*" in *"-i 0") exit 1;; esac; exit 0',
        )
        env = {**os.environ, "PATH": f"{shims}:{os.environ['PATH']}"}

        result = subprocess.run(["bash", "-c", power_reset_script("sudo-smi", frozenset({0, 1}))], env=env, check=False)

        assert result.returncode == 1
        assert calls.read_text().splitlines() == ["-n nvidia-smi -pl 1000 -i 0", "-n nvidia-smi -pl 1000 -i 1"]

    def test_dcgmi_reset_deletes_its_group_and_reports_a_failed_config(self, tmp_path):
        shims = tmp_path / "bin"
        shims.mkdir()
        calls = tmp_path / "calls.log"
        self._shim(shims, "nvidia-smi", 'printf "1400.00\\n"')
        self._shim(
            shims,
            "dcgmi",
            f'printf "%s\\n" "$*" >> {calls}; case "$1" in group) [ "$2" = "-c" ] && printf "Successfully created '
            'group \\"srtctl_power_reset\\" with a group ID of 7\\n"; exit 0;; config) exit 1;; esac',
        )
        env = {**os.environ, "PATH": f"{shims}:{os.environ['PATH']}"}

        result = subprocess.run(
            ["bash", "-c", power_reset_script("dcgmi", frozenset({0, 1, 2, 3}))], env=env, check=False
        )

        assert result.returncode == 1
        assert calls.read_text().splitlines() == [
            "group -c srtctl_power_reset -a 0,1,2,3",
            "config -g 7 --set -P 1400",
            "group -d 7",
        ]

    def test_node_agent_terminate_timeout_follows_the_cadence(self):
        """EPD's stop joins an in-flight cycle for interval_s + 5 s before it writes run_summary.json."""
        assert node_agent_terminate_timeout(30) == 45.0
        assert node_agent_terminate_timeout(60) >= 75.0
        assert node_agent_terminate_timeout(5) == POWER_TERMINATE_TIMEOUT_SECONDS


class TestSchema:
    def test_defaults_to_disabled(self):
        config = _load_recipe()
        power = config.power_control
        assert power.enabled is False
        assert (power.mode, power.pools, power.applier) == ("nupd", "shared", "sudo-smi")
        assert power.ubpc_buffer == {"prefill": 2.0, "decode": 1.1, "agg": 1.1}
        assert power.site_packages == SITE_PACKAGES

    def test_loads_the_documented_block(self):
        config = _load_recipe(
            {
                **ENABLED,
                "mode": "ubpc",
                "pools": "per-role",
                "interval_s": 10,
                "perf_threshold": 0.05,
                "ubpc_buffer": {"prefill": 2.0, "decode": 1.1},
                "min_tgp_w": 900,
                "warmup_policy": "adaptive",
                "applier": "dcgmi",
                "dynamo_power_source": POWER_SOURCE,
                "critical": False,
            }
        )
        power = config.power_control
        assert (power.mode, power.pools, power.applier, power.critical) == ("ubpc", "per-role", "dcgmi", False)
        assert power.dynamo_power_source_root == "/lustre/share/dynamo/components/src"
        assert [spec.name for spec in power.resolve_pools(["prefill", "decode"])] == ["prefill", "decode"]

    def test_fom_is_required(self):
        with pytest.raises(ValidationError, match="power_control.fom is required"):
            _load_recipe({"enabled": True, "budget_w_per_gpu": 1500, "epd_source": EPD_SOURCE})

    def test_budget_is_required(self):
        with pytest.raises(ValidationError, match="power_control.budget_w_per_gpu is required"):
            _load_recipe({"enabled": True, "fom": "tokens_per_gpu", "epd_source": EPD_SOURCE})

    def test_epd_source_is_required_when_enabled(self):
        with pytest.raises(ValidationError, match="power_control.epd_source is required"):
            _load_recipe({"enabled": True, "budget_w_per_gpu": 1500, "fom": "tokens_per_gpu"})
        # Disabled blocks may leave everything unset.
        assert _load_recipe({"enabled": False}).power_control.epd_source is None

    @pytest.mark.parametrize(
        "field, value",
        [("mode", "fast"), ("pools", "everything"), ("applier", "ipmi"), ("warmup_policy", "never")],
    )
    def test_enums_are_enforced(self, field, value):
        with pytest.raises(ValidationError):
            _load_recipe({**ENABLED, field: value})

    @pytest.mark.parametrize(
        "spec",
        [
            "tokens_per_gpu",
            "prom:dynamo_frontend_output_tokens_total:counter:sum",
            "prom:trtllm_num_ctx_tokens:role=prefill,model=x:counter:sum",
            "prom:dynamo_frontend_stream_speed_p25:gauge:avg",
            "python:my_pkg.power:MyProvider",
        ],
    )
    def test_fom_forms_are_accepted(self, spec):
        validate_power_fom(spec, where="power_control.fom")

    @pytest.mark.parametrize(
        "spec",
        ["throughput", "prom:foo:rate:sum", "prom::counter:sum", "prom:foo:badlabel:counter:avg", "python:bad-mod:Cls"],
    )
    def test_fom_typos_fail_at_load(self, spec):
        with pytest.raises(ValidationError, match="power_control.fom"):
            _load_recipe({**ENABLED, "fom": spec})

    def test_explicit_pools_set_roles_or_nodes_once(self):
        with pytest.raises(ValidationError, match="exactly one of roles or nodes"):
            _load_recipe({**ENABLED, "pools": [{"name": "a", "roles": ["prefill"], "nodes": ["node0"]}]})
        with pytest.raises(ValidationError, match="unknown role"):
            _load_recipe({**ENABLED, "pools": [{"name": "a", "roles": ["ctx"]}]})
        with pytest.raises(ValidationError, match="names must be unique"):
            _load_recipe(
                {**ENABLED, "pools": [{"name": "a", "roles": ["prefill"]}, {"name": "a", "roles": ["decode"]}]}
            )
        with pytest.raises(ValidationError, match="in pools 'a' and 'b'"):
            _load_recipe(
                {**ENABLED, "pools": [{"name": "a", "roles": ["prefill"]}, {"name": "b", "roles": ["prefill"]}]}
            )
        with pytest.raises(ValidationError, match="non-empty list"):
            _load_recipe({**ENABLED, "pools": []})

    def test_explicit_pool_naming_a_role_the_recipe_does_not_run_is_refused(self):
        with pytest.raises(ValidationError, match="runs no agg workers"):
            _load_recipe({**ENABLED, "pools": [{"name": "a", "roles": ["agg"]}]})

    def test_per_pool_fom_satisfies_the_requirement(self):
        config = _load_recipe(
            {
                "enabled": True,
                "budget_w_per_gpu": 1500,
                "epd_source": EPD_SOURCE,
                "pools": [
                    {"name": "ctx", "roles": ["prefill"], "fom": "ctx_tokens"},
                    {"name": "gen", "roles": ["decode"], "fom": "gen_tokens"},
                ],
            }
        )
        assert [spec.fom for spec in config.power_control.resolve_pools(["prefill", "decode"])] == [
            "ctx_tokens",
            "gen_tokens",
        ]

    def test_colocated_decode_needs_a_shared_pool(self):
        roles = {
            "prefill": {"nodes": 1, "workers": 1, "gpus": 2},
            "decode": {"nodes": "colocate", "workers": 1, "gpus": 2},
        }
        with pytest.raises(ValidationError, match="pools: shared"):
            _load_recipe({**ENABLED, "pools": "per-role"}, roles=roles)
        assert _load_recipe({**ENABLED, "pools": "shared"}, roles=roles).power_control.enabled

    def test_dynamo_power_source_must_be_a_dynamo_power_directory(self):
        with pytest.raises(ValidationError, match="dynamo/power"):
            _load_recipe({**ENABLED, "dynamo_power_source": "/lustre/share/dynamo/components/src"})

    def test_nupd_budget_must_be_below_max_tgp_w(self):
        """EPD refuses to start a NUPD-only controller whose budget does not bind; the loader says so first."""
        with pytest.raises(ValidationError, match=r"budget_w_per_gpu \(1500 W\) must be below power_control.max_tgp_w"):
            _load_recipe({**ENABLED, "mode": "nupd", "max_tgp_w": 1500})
        with pytest.raises(ValidationError, match=r"pools\[gen\].budget_w_per_gpu \(1400 W\)"):
            _load_recipe(
                {
                    **ENABLED,
                    "budget_w_per_gpu": 1200,
                    "max_tgp_w": 1400,
                    "pools": [
                        {"name": "ctx", "roles": ["prefill"]},
                        {"name": "gen", "roles": ["decode"], "budget_w_per_gpu": 1400},
                    ],
                }
            )
        assert _load_recipe({**ENABLED, "mode": "nupd", "max_tgp_w": 1600}).power_control.max_tgp_w == 1600
        # UBPC lowers caps whatever the budget, so `both` and `ubpc` load with a non-binding budget.
        for mode in ("both", "ubpc"):
            assert _load_recipe({**ENABLED, "mode": mode, "max_tgp_w": 1500}).power_control.mode == mode

    def test_a_job_without_a_frontend_has_no_fom_source(self):
        with pytest.raises(ValidationError, match="frontend.type: none"):
            _load_recipe(ENABLED, frontend={"type": "none"})

    @pytest.mark.parametrize(
        "overrides",
        [
            {"interval_s": 0},
            {"perf_threshold": 1.5},
            {"ubpc_buffer": {"router": 1.1}},
            {"ubpc_buffer": -1.0},
            {"min_tgp_w": 1500, "max_tgp_w": 1400},
            {"site_packages": "relative/site-packages"},
        ],
    )
    def test_knob_ranges(self, overrides):
        with pytest.raises(ValidationError):
            _load_recipe({**ENABLED, **overrides})


class TestOrchestrator:
    def test_launches_appliers_on_the_host_and_node_agents_in_the_worker_image(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PYTHONPATH", raising=False)
        orch = SweepOrchestrator(
            config=_config(pools="per-role", dynamo_power_source=POWER_SOURCE), runtime=_runtime(tmp_path)
        )
        registry = ProcessRegistry(job_id="12345")

        with (
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", return_value=_running()) as srun,
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip),
        ):
            started = orch.start_power_control(registry)

        calls = {call.kwargs["step_name"]: call.kwargs for call in srun.call_args_list}
        assert sorted(name for name in calls if name.startswith("power_node_agent_")) == [
            *(f"power_node_agent_decode_node{i}" for i in (2, 3, 4, 5)),
            *(f"power_node_agent_prefill_node{i}" for i in (0, 1)),
        ]
        assert sorted(name for name in calls if name.startswith("power_applier_")) == [
            f"power_applier_node{i}" for i in range(6)
        ]
        assert len(started) == 12
        assert set(registry.get_all_processes()) == set(calls)

        agent = calls["power_node_agent_prefill_node1"]
        assert agent["nodelist"] == ["node1"]
        assert agent["container_image"] == "/img.sqsh"
        assert agent["container_mounts"][tmp_path] == Path("/logs")
        assert agent["container_mounts"][Path(EPD_SOURCE)] == Path(f"{SITE_PACKAGES}/epd:ro")
        assert agent["container_mounts"][Path(POWER_SOURCE)] == Path(f"{SITE_PACKAGES}/dynamo/power:ro")
        assert agent["srun_options"] == orch.runtime.srun_options
        assert agent["bash_preamble"] == orch._build_worker_preamble()
        command = agent["command"]
        assert command[:3] == ["python3", "-m", "dynamo.power.node_agent"]
        assert command[command.index("--rank") + 1] == "1"
        assert command[command.index("--nodelist") + 1] == "node0,node1"
        assert command[command.index("--log-dir") + 1] == "/logs/power_control/prefill"
        metrics = command[command.index("--metrics-url") + 1].split(",")
        # The frontend's /metrics first (head node), then the pool's worker system ports,
        # each tagged with the scope a FoM can select.
        assert metrics[0] == "frontend=http://10.0.0.0:8000/metrics"
        assert metrics[1:] == [
            f"worker=http://10.0.0.{process.node.removeprefix('node')}:{process.sys_port}/metrics"
            for process in orch.backend_processes
            if process.endpoint_mode == "prefill"
        ]
        assert command[command.index("--pool-gpu-count") + 1] == "8"
        assert "NVIDIA_VISIBLE_DEVICES" not in agent["env_to_set"], "a whole-node group sees every GPU"
        assert agent["env_to_set"]["EPD_NODELIST"] == "node0,node1"
        assert agent["env_to_set"]["LOG_DIR"] == "/logs/power_control/prefill"
        assert agent["env_to_set"]["UV_CACHE_DIR"] == "/cache"

        applier = calls["power_applier_node1"]
        assert applier["nodelist"] == ["node1"]
        assert applier["container_image"] is None, "the applier runs on the bare host"
        assert applier["env_to_set"] == {"PYTHONPATH": "/lustre/share/dynamo/components/src"}
        command = applier["command"]
        assert command[:3] == [sys.executable, "-m", "dynamo.power.applier"]
        pool_dir = tmp_path / "power_control" / "prefill"
        assert command[command.index("--decision-file") + 1] == str(pool_dir / DECISION_FILE_RELPATH)
        assert command[command.index("--trace-csv") + 1] == str(pool_dir / "applier_node1.csv")
        assert command[command.index("--mode") + 1] == "sudo-smi"
        assert (pool_dir / "epd" / "actuation").is_dir()
        assert (tmp_path / "power_control" / "decode" / "epd" / "actuation").is_dir()

        processes = registry.get_all_processes()
        assert all(proc.critical for proc in processes.values())
        assert processes["power_node_agent_prefill_node0"].shutdown_tier == 0
        assert processes["power_applier_node0"].shutdown_tier == 1
        # The node_agent's SIGTERM budget follows the 30 s cadence; the applier keeps the flat 30 s.
        assert processes["power_node_agent_prefill_node0"].terminate_timeout == 45.0
        assert processes["power_applier_node0"].terminate_timeout == POWER_TERMINATE_TIMEOUT_SECONDS

    def test_a_slow_cadence_widens_the_node_agent_stop_budget(self, tmp_path):
        orch = SweepOrchestrator(config=_config(interval_s=60), runtime=_runtime(tmp_path))
        registry = ProcessRegistry(job_id="12345")

        with (
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", return_value=_running()),
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip),
        ):
            orch.start_power_control(registry)

        processes = registry.get_all_processes()
        assert processes["power_node_agent_shared_node0"].terminate_timeout >= 75.0
        assert processes["power_applier_node0"].terminate_timeout == POWER_TERMINATE_TIMEOUT_SECONDS

    def test_recipe_environment_is_templated_per_node_like_the_workers(self, tmp_path):
        environment = {"CACHE_DIR": "/cache/{node_id}", "WORKER_NODE": "{node}", "KEEP": "{unsupported}-x"}
        orch = SweepOrchestrator(config=_config(), runtime=_runtime(tmp_path, environment=environment))

        with (
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", return_value=_running()) as srun,
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip),
        ):
            orch.start_power_control(ProcessRegistry(job_id="12345"))

        calls = {call.kwargs["step_name"]: call.kwargs for call in srun.call_args_list}
        env = calls["power_node_agent_shared_node1"]["env_to_set"]
        assert (env["CACHE_DIR"], env["WORKER_NODE"], env["KEEP"]) == ("/cache/1", "node1", "{unsupported}-x")
        assert env["EPD_NODELIST"] == "node0,node1,node2,node3,node4,node5"

    def test_trtllm_metrics_urls_name_only_the_mpi_leaders_and_resolve_each_node_once(self, tmp_path):
        """A TRT-LLM worker is one MPI step; the Dynamo main (system port) runs on its leader node only."""
        orch = SweepOrchestrator(config=_config(engine=TRTLLMBackend(), pools="per-role"), runtime=_runtime(tmp_path))

        with (
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", return_value=_running()) as srun,
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip) as ip,
        ):
            orch.start_power_control(ProcessRegistry(job_id="12345"))

        decode = [process for process in orch.backend_processes if process.endpoint_mode == "decode"]
        leaders = [process for process in decode if process.is_leader]
        assert len(decode) == 4 and len(leaders) == 2, "two 8-GPU decode workers span two 4-GPU nodes each"
        calls = {call.kwargs["step_name"]: call.kwargs for call in srun.call_args_list}
        command = calls["power_node_agent_decode_node2"]["command"]
        metrics = command[command.index("--metrics-url") + 1].split(",")
        assert metrics == [
            "frontend=http://10.0.0.0:8000/metrics",
            *(
                f"worker=http://10.0.0.{process.node.removeprefix('node')}:{process.sys_port}/metrics"
                for process in leaders
            ),
        ]
        # One srun-backed lookup per node at most, and none for the follower nodes that serve nothing.
        resolved = Counter(call.args[0] for call in ip.call_args_list)
        assert set(resolved.values()) == {1}
        assert set(resolved) == {"node0", "node1", "node2", "node4"}

    def test_a_partial_node_group_gets_a_restricted_gpu_view(self, tmp_path):
        """TEP6 tail: the node_agent container sees only the group's GPUs (EPD's NVML monitor has no filter)."""
        tep6 = _processes(prefill=(5, 6), decode=(0, 8), pack=True)
        runtime = dataclasses.replace(
            _runtime(tmp_path),
            nodes=Nodes(head="node0", bench="node0", infra="node0", worker=tuple(f"node{i}" for i in range(8))),
        )
        orch = SweepOrchestrator(config=_config(pools="per-role"), runtime=runtime)

        with (
            patch.object(type(orch), "backend_processes", new_callable=lambda: property(lambda self: tep6)),
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", return_value=_running()) as srun,
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip),
        ):
            orch.start_power_control(ProcessRegistry(job_id="12345"))

        calls = {call.kwargs["step_name"]: call.kwargs for call in srun.call_args_list}
        tail = calls["power_node_agent_prefill_node7"]
        assert tail["env_to_set"]["NVIDIA_VISIBLE_DEVICES"] == "0,1"
        assert tail["command"][tail["command"].index("--gpu-indices") + 1] == "0,1"
        assert tail["command"][tail["command"].index("--pool-gpu-count") + 1] == "30"
        # A node both workers fill completely keeps the default view of all four GPUs.
        assert "NVIDIA_VISIBLE_DEVICES" not in calls["power_node_agent_prefill_node1"]["env_to_set"]

    def test_validate_layout_refuses_a_split_node_before_anything_launches(self, tmp_path):
        """The job-start gate: per-role pools over a node that holds both roles fail before host setup."""
        orch = SweepOrchestrator(config=_config(pools="per-role"), runtime=_runtime(tmp_path))
        colocated = [process.__class__(**{**process.__dict__, "node": "node0"}) for process in orch.backend_processes]

        with (
            patch.object(type(orch), "backend_processes", new_callable=lambda: property(lambda self: colocated)),
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process") as srun,
            pytest.raises(ValueError, match="exactly one pool"),
        ):
            orch.validate_power_control_layout()
        srun.assert_not_called()

        assert [pool.name for pool in orch.validate_power_control_layout()] == ["prefill", "decode"]
        disabled = SweepOrchestrator(config=_config(enabled=False, epd_source=None), runtime=_runtime(tmp_path))
        assert disabled.validate_power_control_layout() == []

    def test_stop_reports_a_pool_whose_decision_file_never_appeared(self, tmp_path, caplog):
        """EPD publishes its starting caps at start(); a pool without the file ran at the static uniform budget."""
        orch = SweepOrchestrator(config=_config(pools="per-role"), runtime=_runtime(tmp_path))

        with (
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", return_value=_running()),
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip),
        ):
            orch.start_power_control(ProcessRegistry(job_id="12345"))
            (tmp_path / "power_control" / "prefill" / DECISION_FILE_RELPATH).write_text("{}")
            with caplog.at_level(logging.ERROR, logger="srtctl.cli.mixins.power_control_stage"):
                orch.stop_power_control()

        errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "Power pool decode: EPD never wrote" in errors[0]
        assert str(tmp_path / "power_control" / "decode" / DECISION_FILE_RELPATH) in errors[0]
        assert "static-cap run" in errors[0] and "node_agent_node2.out" in errors[0]

    def test_multiple_frontends_scrape_each_frontend_listener_not_nginx(self, tmp_path):
        """Behind nginx each frontend serves its own /metrics on the internal port; nginx would answer for one at random."""
        frontend = FrontendConfig(enable_multiple_frontends=True, num_additional_frontends=1)
        orch = SweepOrchestrator(config=_config(frontend, pools="shared"), runtime=_runtime(tmp_path))

        with (
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", return_value=_running()) as srun,
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip),
        ):
            orch.start_power_control(ProcessRegistry(job_id="12345"))

        topology = orch._compute_frontend_topology()
        assert topology.uses_nginx and topology.frontend_nodes == ["node1", "node2"]
        calls = {call.kwargs["step_name"]: call.kwargs for call in srun.call_args_list}
        command = calls["power_node_agent_shared_node0"]["command"]
        metrics = command[command.index("--metrics-url") + 1].split(",")
        assert metrics[:2] == ["frontend=http://10.0.0.1:8180/metrics", "frontend=http://10.0.0.2:8180/metrics"]
        assert not any(url.endswith("10.0.0.0:8000/metrics") for url in metrics)
        assert len(metrics) == 2 + len(orch.backend_processes)

    def test_non_critical_block_registers_non_critical_steps(self, tmp_path):
        orch = SweepOrchestrator(config=_config(critical=False), runtime=_runtime(tmp_path))
        registry = ProcessRegistry(job_id="12345")

        with (
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", return_value=_running()),
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip),
        ):
            orch.start_power_control(registry)

        assert registry.get_all_processes()
        assert not any(proc.critical for proc in registry.get_all_processes().values())

    def test_disabled_block_launches_nothing(self, tmp_path):
        orch = SweepOrchestrator(config=_config(enabled=False, epd_source=None), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.mixins.power_control_stage.start_srun_process") as srun:
            assert orch.start_power_control(ProcessRegistry(job_id="12345")) == []
            orch.stop_power_control()

        srun.assert_not_called()

    def test_stop_terminates_node_agents_before_appliers_then_resets_every_node(self, tmp_path):
        orch = SweepOrchestrator(config=_config(pools="per-role"), runtime=_runtime(tmp_path))
        order: list[str] = []

        def launch(**kwargs):
            if kwargs["step_name"].startswith("power_reset_"):
                return _running(kwargs["step_name"], order)
            return _running(kwargs["step_name"], order)

        with (
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", side_effect=launch) as srun,
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip),
        ):
            orch.start_power_control(ProcessRegistry(job_id="12345"))
            srun.reset_mock()
            orch.stop_power_control()

        agents = [name for name in order if name.startswith("power_node_agent_")]
        appliers = [name for name in order if name.startswith("power_applier_")]
        assert len(agents) == 6 and len(appliers) == 6
        assert order == [*agents, *appliers], "every node_agent stops before the first applier"

        resets = [call.kwargs for call in srun.call_args_list]
        assert sorted(call["step_name"] for call in resets) == [f"power_reset_node{i}" for i in range(6)]
        for call in resets:
            assert call["container_image"] is None
            assert call["srun_options"] == orch.runtime.srun_options == {"mpi": "none"}
            assert call["command"][:2] == ["bash", "-c"]
            assert 'sudo -n nvidia-smi -pl "${d%%.*}" -i "$i" || rc=1; done; exit $rc' in call["command"][2]
            assert call["nodelist"][0] in call["step_name"]

    def test_stop_resets_even_when_the_steps_already_exited(self, tmp_path):
        orch = SweepOrchestrator(config=_config(), runtime=_runtime(tmp_path))
        exited = MagicMock()
        exited.poll.return_value = 0
        exited.wait.return_value = 0

        with (
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", return_value=exited) as srun,
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip),
        ):
            orch.start_power_control(ProcessRegistry(job_id="12345"))
            srun.reset_mock()
            orch.stop_power_control()

        exited.terminate.assert_not_called()
        assert [call.kwargs["step_name"] for call in srun.call_args_list] == [f"power_reset_node{i}" for i in range(6)]

    def test_stop_never_raises_when_the_reset_cannot_launch(self, tmp_path):
        orch = SweepOrchestrator(config=_config(), runtime=_runtime(tmp_path))

        with (
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process", return_value=_running()),
            patch("srtctl.cli.mixins.power_control_stage.get_hostname_ip", side_effect=_node_ip),
        ):
            orch.start_power_control(ProcessRegistry(job_id="12345"))
        with patch("srtctl.cli.mixins.power_control_stage.start_srun_process", side_effect=OSError("srun gone")):
            orch.stop_power_control()  # does not raise

    def test_start_refuses_a_layout_it_cannot_express(self, tmp_path):
        """Per-role pools over a node that holds both roles: refused before anything launches."""
        config = SrtConfig(
            name="power-control-test",
            model=ModelConfig(path="/models/test", container="test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="gb200", gpus_per_node=4),
            roles={"prefill": RoleConfig(nodes=1, workers=1, gpus=2), "decode": RoleConfig(nodes=1, workers=1, gpus=2)},
            power_control=_power(pools="per-role"),
        )
        runtime = RuntimeContext(
            job_id="12345",
            run_name="test-run",
            nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0", "node1")),
            head_node_ip="10.0.0.0",
            infra_node_ip="10.0.0.0",
            log_dir=tmp_path,
            model_path=Path("/models/test"),
            container_image=Path("/img.sqsh"),
            gpus_per_node=4,
            network_interface=None,
            container_mounts={},
            environment={},
        )
        orch = SweepOrchestrator(config=config, runtime=runtime)
        # Two single-node roles on two nodes keep their own node; force them onto one.
        colocated = [process.__class__(**{**process.__dict__, "node": "node0"}) for process in orch.backend_processes]
        with (
            patch.object(type(orch), "backend_processes", new_callable=lambda: property(lambda self: colocated)),
            patch("srtctl.cli.mixins.power_control_stage.start_srun_process") as srun,
            pytest.raises(ValueError, match="exactly one pool"),
        ):
            orch.start_power_control(ProcessRegistry(job_id="12345"))
        srun.assert_not_called()
