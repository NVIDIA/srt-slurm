# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise batch staging and launch sizing with fake Slurm commands."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "configs/cluster-diagnostics.sbatch"


@pytest.mark.parametrize(
    ("cluster", "image"),
    [
        ("raplab", "/home/rihuo/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh"),
        (
            "hecate",
            "/lustre/share/coreai_comparch_inferencex/rihuo/trtllm-release-f1bdd25-user_imant_glm5-rubin-30428-arm64.sqsh",
        ),
    ],
)
def test_image_presets(cluster: str, image: str) -> None:
    env = {**os.environ, "SLURM_JOB_ID": "test", "SLURM_SUBMIT_DIR": str(ROOT)}
    env.pop("CONTAINER_IMAGE", None)
    result = subprocess.run(["bash", str(SCRIPT), cluster], env=env, capture_output=True, text=True)
    # Local filesystem has no cluster images. Preset selection happens before launching Slurm.
    assert result.returncode == 1
    assert f"Missing image: {image}" in result.stderr


@pytest.mark.parametrize("fail_communication", [False, True])
@pytest.mark.parametrize("manual_endpoint", [False, True])
@pytest.mark.parametrize("cluster", ["raplab", "hecate"])
def test_spooled_batch_stages_helpers_and_uses_all_allocated_cpus(
    tmp_path: Path, fail_communication: bool, manual_endpoint: bool, cluster: str
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, body in {
        "scontrol": 'if [[ $2 == hostnames ]]; then printf "node-a\\nnode-b\\n"; fi',
        "srun": 'printf "%s\\n" "$*" >> "$CALLS"; if [[ $* == *communication* ]]; then exit "$COMM_RC"; fi',
    }.items():
        path = fake_bin / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(0o755)
    image = tmp_path / "image.sqsh"
    image.touch()
    spooled = tmp_path / "spool.sh"
    spooled.write_text(SCRIPT.read_text())
    log_dir = tmp_path / "logs"
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SLURM_JOB_ID": "test",
        "SLURM_SUBMIT_DIR": str(ROOT),
        "SLURM_JOB_NODELIST": "node-[a,b]",
        "SLURM_JOB_CPUS_PER_NODE": "352(x2)",
        "CONTAINER_IMAGE": str(image),
        "LOG_DIR": str(log_dir),
        "CALLS": str(calls),
        "COMM_RC": "9" if fail_communication else "0",
        "RUN_COMM_BENCH": "1",
        "RUN_NCCL": "1",
        "RANKS_PER_NODE": "4",
    }
    for key in ("MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    if manual_endpoint:
        env.update(MASTER_ADDR="192.0.2.1", MASTER_PORT="23456")
    result = subprocess.run(["bash", str(spooled), cluster], env=env, capture_output=True, text=True)
    assert result.returncode == int(fail_communication), result.stderr
    launches = calls.read_text().splitlines()
    assert len(launches) == 3
    assert "--cpus-per-task=352" in launches[0]
    assert "--cpus-per-task=352" in launches[1]
    assert "--ntasks=8 --ntasks-per-node=4 --cpus-per-task=88" in launches[2]
    assert "--bind-memory" in launches[2]
    assert "--nccl" in launches[2]
    if manual_endpoint:
        assert "--master-addr 192.0.2.1 --master-port 23456" in launches[2]
    else:
        assert "--master-addr" not in launches[2]
        assert "--master-port" not in launches[2]
    assert (log_dir / "numa_net_devices.py").exists()
    assert (log_dir / "runner.sh").read_text() == (ROOT / "configs/raplab-cluster-diagnostics.sbatch").read_text()
    assert (log_dir / "communication.exit-status").read_text().strip() == env["COMM_RC"]
    assert Path(f"{log_dir}.tar.gz").exists()
    recipe = yaml.safe_load(
        (ROOT / "recipes/trtllm/vr200-fp4/glm5.2/raplab-dyanmo-1004/disagg-3p-dep-1d-dep-c560.yaml").read_text()
    )
    knobs = {
        key: value
        for key, value in recipe["roles"]["decode"]["env"].items()
        if key.startswith(("MPI_UCX_", "UCX_", "NCCL_", "OMPI_MCA_"))
    }
    for key, value in knobs.items():
        assignment = f"{key}={value}"
        assert assignment not in launches[0]
        assert assignment not in launches[1]
        if cluster == "raplab":
            assert assignment in launches[2]
            assert assignment in (log_dir / "settings.txt").read_text()
        else:
            assert assignment not in launches[2]
