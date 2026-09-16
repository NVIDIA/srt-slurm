# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``benchmarks/rl/miles/launch.sh`` (repo root): the Miles launcher against the custom-benchmark environment."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
LAUNCH = BENCHMARKS_DIR / "rl" / "miles" / "launch.sh"

BASE_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "SRT_SERVICE_TRAIN_NODES": "node1,node2",
    "SRT_SERVICE_TRAIN_IPS": "10.0.0.11,10.0.0.12",
    "SRT_SERVICE_TRAIN_NODE_COUNT": "2",
    "SRT_GPUS_PER_NODE": "8",
    "MILES_RECIPE": "scripts/run_qwen3_dense.py",
    "MILES_SCRIPT_MODEL_NAME": "Qwen3-4B",
    "MILES_SCRIPT_EXTRA_ARGS": "--num-rollout 3 --colocate",
    "MILES_LAUNCH_DRY_RUN": "1",
}


def _run(**overrides) -> subprocess.CompletedProcess[str]:
    env = {**BASE_ENV, **overrides}
    env = {k: v for k, v in env.items() if v is not None}
    return subprocess.run(["bash", str(LAUNCH)], env=env, capture_output=True, text=True, timeout=30, check=False)


def _exports(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line and not line.startswith("command:"))


def _command(stdout: str) -> list[str]:
    line = next(line for line in stdout.splitlines() if line.startswith("command:"))
    return shlex.split(line[len("command:") :])


def test_launch_script_is_executable_and_bundled() -> None:
    assert LAUNCH.is_file()
    assert os.access(LAUNCH, os.X_OK), "the container runs it by path"


def test_dry_run_resolves_the_cluster_from_the_service_env() -> None:
    result = _run()
    assert result.returncode == 0, result.stderr
    exports = _exports(result.stdout)
    assert exports["MASTER_ADDR"] == "10.0.0.11", "the first IP of the ray service is its head"
    assert exports["RAY_ADDRESS"] == "http://10.0.0.11:8265"
    assert exports["MILES_SCRIPT_EXTERNAL_RAY"] == "1"
    assert exports["MILES_SCRIPT_NUM_NODES"] == "2"
    assert exports["MILES_SCRIPT_NUM_GPUS_PER_NODE"] == "8"
    assert exports["MILES_SCRIPT_MODEL_NAME"] == "Qwen3-4B", "MILES_SCRIPT_* passes through untouched"
    assert _command(result.stdout) == ["python3", "/root/miles/scripts/run_qwen3_dense.py"], "no positional by default"


def test_subcommand_root_ports_and_service_name_are_honoured() -> None:
    result = _run(
        MILES_SUBCOMMAND="train",
        MILES_ROOT="/opt/miles",
        MILES_RAY_DASHBOARD_PORT="8300",
        MILES_RAY_SERVICE="ray-cluster",
        SRT_SERVICE_RAY_CLUSTER_IPS="10.1.1.1",
        SRT_SERVICE_RAY_CLUSTER_NODE_COUNT="1",
        SRT_SERVICE_TRAIN_IPS=None,
        SRT_SERVICE_TRAIN_NODE_COUNT=None,
    )
    assert result.returncode == 0, result.stderr
    exports = _exports(result.stdout)
    assert exports["MASTER_ADDR"] == "10.1.1.1"
    assert exports["RAY_ADDRESS"] == "http://10.1.1.1:8300"
    assert exports["MILES_SCRIPT_NUM_NODES"] == "1"
    assert _command(result.stdout) == ["python3", "/opt/miles/scripts/run_qwen3_dense.py", "train"]


def test_absolute_recipe_and_explicit_overrides_win() -> None:
    result = _run(MILES_RECIPE="/x/recipe.py", MASTER_ADDR="1.2.3.4", MILES_SCRIPT_NUM_NODES="7")
    assert result.returncode == 0, result.stderr
    exports = _exports(result.stdout)
    assert exports["MASTER_ADDR"] == "1.2.3.4"
    assert exports["MILES_SCRIPT_NUM_NODES"] == "7"
    assert _command(result.stdout) == ["python3", "/x/recipe.py"]


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("SRT_SERVICE_TRAIN_IPS", "SRT_SERVICE_TRAIN_IPS is not set"),
        ("MILES_RECIPE", "MILES_RECIPE is required"),
        ("SRT_GPUS_PER_NODE", "SRT_GPUS_PER_NODE is not set"),
    ],
)
def test_missing_inputs_fail_fast_with_the_key_to_set(missing: str, message: str) -> None:
    result = _run(**{missing: None})
    assert result.returncode != 0
    assert message in result.stderr


def test_benchmarks_dir_is_mounted_into_the_container(tmp_path: Path, monkeypatch) -> None:
    # The recipe points at /benchmarks/rl/miles/launch.sh; RuntimeContext mounts <repo>/benchmarks there
    # from SRTCTL_SOURCE_DIR, the checkout root the sbatch script exports.
    assert (BENCHMARKS_DIR / "rl" / "README.md").is_file()
    from unittest.mock import patch

    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

    config = SrtConfig.Schema().load(
        {
            "schema": 2,
            "name": "mount-test",
            "model": {"path": "hf:fake/model", "container": "nvcr.io/fake:latest", "precision": "bf16"},
            "resources": {"gpu_type": "b200", "gpus_per_node": 8},
            "frontend": {"type": "none"},
            "services": [{"name": "train", "type": "ray", "nodes": 1}],
            "benchmark": {"type": "custom", "command": "/benchmarks/rl/miles/launch.sh"},
        }
    )
    monkeypatch.setenv("SRTCTL_SOURCE_DIR", str(REPO_ROOT))
    monkeypatch.setenv("SRTCTL_OUTPUT_DIR", str(tmp_path))
    with (
        patch("srtctl.core.runtime.get_slurm_nodelist", return_value=["node1"]),
        patch("srtctl.core.runtime.get_slurm_het_nodelists", return_value=None),
        patch("srtctl.core.runtime.get_hostname_ip", return_value="10.0.0.1"),
    ):
        runtime = RuntimeContext.from_config(config, "1")
    assert runtime.container_mounts[BENCHMARKS_DIR.resolve()] == Path("/benchmarks")
