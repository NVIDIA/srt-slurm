# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker probe output: constructed for nvidia-smi, captured on an MI325X node for kfd."""

import json

import pytest

from srtctl.core.power.identity import is_device_identity, is_gpu_uuid, parse_node_probe

GPU_A = "GPU-11111111-1111-1111-1111-111111111111"
GPU_B = "GPU-22222222-2222-2222-2222-222222222222"
GPU_C = "GPU-33333333-3333-3333-3333-333333333333"

STEP_0 = """PID      JOBID    STEPID  LOCALID GLOBALID
24100    12345    0       0       0
24101    12345    0       -1      -1
24102    12345    0       -1      -1
"""
STEP_1 = """PID      JOBID    STEPID  LOCALID GLOBALID
25100    12345    1       0       0
25101    12345    1       -1      -1
"""
COMPUTE_APPS = f"""24101, {GPU_B}, /usr/bin/python3
25101, {GPU_C}, /usr/bin/python3
24101, {GPU_A}, /usr/bin/python3
24102, {GPU_A}, /usr/bin/python3

29999, GPU-44444444-4444-4444-4444-444444444444, /usr/bin/python3
"""


class TestParseNodeProbe:
    def test_attributes_two_steps_by_pid_with_spaced_csv(self) -> None:
        raw = json.dumps({"steps": {"12345.0": STEP_0, "12345.1": STEP_1}, "compute_apps": COMPUTE_APPS})

        assert parse_node_probe(raw, is_gpu_uuid) == {"12345.0": (GPU_A, GPU_B), "12345.1": (GPU_C,)}

    def test_uses_last_nonempty_line_after_an_earlier_probe_and_launcher_noise(self) -> None:
        earlier = json.dumps({"steps": {"12345.0": STEP_0}, "compute_apps": f"24101, {GPU_C}, python3\n"})
        current = json.dumps({"steps": {"12345.0": STEP_0}, "compute_apps": COMPUTE_APPS})

        assert parse_node_probe(f"{earlier}\nsrun: launching worker probe\n{current}\n\n", is_gpu_uuid) == {
            "12345.0": (GPU_A, GPU_B)
        }

    def test_heterogeneous_step_uses_component_job_id_in_listpids(self) -> None:
        listed = """PID      JOBID    STEPID  LOCALID GLOBALID
24101    12346    0       0       0
"""
        raw = json.dumps({"steps": {"12345+1.0": listed}, "compute_apps": f"24101, {GPU_A}, python3\n"})

        assert parse_node_probe(raw, is_gpu_uuid) == {"12345+1.0": (GPU_A,)}

    def test_rejects_mps_even_when_the_server_pid_is_outside_worker_steps(self) -> None:
        apps = COMPUTE_APPS + f"29998, {GPU_A}, /usr/bin/nvidia-cuda-mps-server\n"
        raw = json.dumps({"steps": {"12345.0": STEP_0}, "compute_apps": apps})

        with pytest.raises(ValueError, match="^MPS worker attribution is unsupported$"):
            parse_node_probe(raw, is_gpu_uuid)

    def test_rejects_a_step_without_gpu_pids_even_when_another_step_is_attributed(self) -> None:
        raw = json.dumps(
            {"steps": {"12345.0": STEP_0, "12345.1": STEP_1}, "compute_apps": f"24101, {GPU_A}, python3\n"}
        )

        with pytest.raises(ValueError, match=r"^worker step 12345\.1 has no attributable compute-app UUIDs$"):
            parse_node_probe(raw, is_gpu_uuid)

    @pytest.mark.parametrize(
        ("listed", "message"),
        [
            ("PID JOBID STEPID LOCALID\n24101 12345 0 0\n", "missing scontrol listpids header"),
            ("", "missing scontrol listpids header"),
            ("PID JOBID STEPID LOCALID GLOBALID\nunknown 12345 0 0 0\n", "invalid scontrol listpids row"),
            ("PID JOBID STEPID LOCALID GLOBALID\n0 12345 0 0 0\n", "invalid scontrol listpids row"),
            ("PID JOBID STEPID LOCALID GLOBALID\n24101 12345 0 0\n", "invalid scontrol listpids row"),
        ],
        ids=["header-drift", "empty-output", "nonnumeric-pid", "zero-pid", "missing-column"],
    )
    def test_rejects_invalid_listpids_output(self, listed: str, message: str) -> None:
        raw = json.dumps({"steps": {"12345.0": listed}, "compute_apps": COMPUTE_APPS})

        with pytest.raises(ValueError, match=f"^{message}$"):
            parse_node_probe(raw, is_gpu_uuid)

    @pytest.mark.parametrize(
        "apps",
        [
            f"N/A, {GPU_A}, /usr/bin/python3\n",
            f"0, {GPU_A}, /usr/bin/python3\n",
            "24101, MIG-11111111-1111-1111-1111-111111111111, /usr/bin/python3\n",
            f"24101, MIG-{GPU_A}/1/0, /usr/bin/python3\n",
            "24101, 11111111-1111-1111-1111-111111111111, /usr/bin/python3\n",
            f"24101, {GPU_A}\n",
        ],
        ids=["nonnumeric-pid", "zero-pid", "mig-uuid", "legacy-mig-uuid", "missing-gpu-prefix", "missing-column"],
    )
    def test_rejects_invalid_compute_app_identity(self, apps: str) -> None:
        raw = json.dumps({"steps": {"12345.0": STEP_0}, "compute_apps": apps})

        with pytest.raises(ValueError, match="^invalid compute-app identity$"):
            parse_node_probe(raw, is_gpu_uuid)


# `kfd` probe output from an MI325X node: step 0 ran with ROCR_VISIBLE_DEVICES=2,5 and
# step 1 with 0,1; KFD maps all eight GPUs into both processes, VRAM only into those.
KFD_CAPTURE = json.dumps(
    {
        "steps": {
            "3183.0": "PID      JOBID    STEPID   LOCALID GLOBALID\n1497532  3183     0        0       0        \n1495984  3183     0        -       -        \n",
            "3183.1": "PID      JOBID    STEPID   LOCALID GLOBALID\n1497531  3183     1        0       0        \n1495983  3183     1        -       -        \n",
        },
        "compute_apps": "1497532,PCB079220-1144,python3\r\n1497532,PCB079220-1156,python3\r\n1497531,PCB079220-1146,python3\r\n1497531,PCB079220-1177,python3\r\n",
    }
)


class TestKfdProbe:
    def test_captured_output_attributes_each_step_its_board_serials(self) -> None:
        assert parse_node_probe(KFD_CAPTURE, is_device_identity) == {
            "3183.0": ("PCB079220-1144", "PCB079220-1156"),
            "3183.1": ("PCB079220-1146", "PCB079220-1177"),
        }
