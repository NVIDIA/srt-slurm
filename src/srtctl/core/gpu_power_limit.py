# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply, audit, and restore per-role GPU power limits for a Slurm run."""

import csv
import logging
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srtctl.core.schema import GpuPowerLimitConfig
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

PowerAssignments = dict[tuple[str, int], float]


def build_power_assignments(config: "GpuPowerLimitConfig", processes: list["Process"]) -> PowerAssignments:
    """Map every physical worker GPU to its requested role-specific limit."""
    assignments: PowerAssignments = {}
    for process in processes:
        watts = config.watts_for_mode(process.endpoint_mode)
        if watts is None:
            continue
        for gpu_index in process.gpu_indices:
            key = (process.node, gpu_index)
            previous = assignments.get(key)
            if previous is not None and previous != watts:
                raise ValueError(
                    f"GPU {process.node}:{gpu_index} has conflicting power limits: "
                    f"{previous:g}W and {watts}W"
                )
            assignments[key] = float(watts)
    return assignments


class GpuPowerLimitManager:
    """Snapshot, set, verify, and restore persistent GPU power-limit state."""

    def __init__(
        self,
        *,
        job_id: str,
        assignments: PowerAssignments,
        log_dir: Path,
        setter: str = "auto",
        nvidia_smi_command: list[str] | None = None,
        dcgmi_command: list[str] | None = None,
        restore_on_exit: bool = True,
    ) -> None:
        self.job_id = job_id
        self.assignments = assignments
        self.log_dir = log_dir
        self.setter = setter
        self.nvidia_smi_command = nvidia_smi_command or ["nvidia-smi"]
        self.dcgmi_command = dcgmi_command or ["dcgmi"]
        self.restore_on_exit = restore_on_exit
        self._snapshot: PowerAssignments = {}
        self._active_setter: str | None = None

    @classmethod
    def from_processes(
        cls,
        *,
        config: "GpuPowerLimitConfig",
        job_id: str,
        processes: list["Process"],
        log_dir: Path,
    ) -> "GpuPowerLimitManager":
        return cls(
            job_id=job_id,
            assignments=build_power_assignments(config, processes),
            log_dir=log_dir,
            setter=config.setter,
            nvidia_smi_command=list(config.nvidia_smi_command),
            dcgmi_command=list(config.dcgmi_command),
            restore_on_exit=config.restore_on_exit,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.assignments)

    @property
    def affected_nodes(self) -> tuple[str, ...]:
        return tuple(sorted({node for node, _gpu_index in self.assignments}))

    def apply(self) -> None:
        """Capture the baseline, apply the requested limits, and fail if verification disagrees."""
        if not self.enabled:
            return

        current = self._query_limits(self.affected_nodes)
        missing = sorted(set(self.assignments) - set(current))
        if missing:
            formatted = ", ".join(f"{node}:{gpu}" for node, gpu in missing)
            raise RuntimeError(f"Could not query current GPU power limits for: {formatted}")

        self._snapshot = {key: current[key] for key in self.assignments}
        self._write_snapshot("gpu_power_limits_before.csv", self._snapshot)
        try:
            self._set_limits(self.assignments)
            self._verify_and_write("gpu_power_limits_applied.csv", self.assignments)
        except BaseException:
            self._restore_snapshot()
            raise

    def restore(self) -> None:
        """Restore the captured baseline. Safe to call repeatedly."""
        if self.restore_on_exit and self._snapshot:
            self._restore_snapshot()

    def _restore_snapshot(self) -> None:
        snapshot = self._snapshot
        if not snapshot:
            return
        # Clear first so two cleanup paths cannot race or repeat the write.
        self._snapshot = {}
        logger.info("Restoring previous GPU power limits")
        self._set_limits(snapshot, force_setter=self._active_setter)
        self._verify_and_write("gpu_power_limits_restored.csv", snapshot)

    def _verify_and_write(self, filename: str, expected: PowerAssignments) -> None:
        actual = self._query_limits(tuple(sorted({node for node, _gpu in expected})))
        self._write_snapshot(filename, actual)
        mismatches = []
        for key, expected_watts in expected.items():
            actual_watts = actual.get(key)
            if actual_watts is None or abs(actual_watts - expected_watts) > 0.5:
                node, gpu_index = key
                observed = "missing" if actual_watts is None else f"{actual_watts:g}W"
                mismatches.append(f"{node}:{gpu_index} expected {expected_watts:g}W, got {observed}")
        if mismatches:
            raise RuntimeError("GPU power-limit verification failed: " + "; ".join(mismatches))

    def _query_limits(self, nodes: tuple[str, ...]) -> PowerAssignments:
        nvidia_smi = shlex.join(self.nvidia_smi_command)
        command = (
            "node=$(hostname -s); "
            f"{nvidia_smi} --query-gpu=index,power.limit --format=csv,noheader,nounits | "
            'while IFS= read -r line; do printf \'%s,%s\\n\' "$node" "$line"; done'
        )
        result = self._run_on_nodes(nodes, command)
        aliases = {node.split(".", 1)[0]: node for node in nodes}
        aliases.update({node: node for node in nodes})
        limits: PowerAssignments = {}
        for raw_line in result.stdout.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) != 3:
                continue
            node_raw, gpu_raw, watts_raw = parts
            node = aliases.get(node_raw)
            if node is None:
                continue
            try:
                limits[(node, int(gpu_raw))] = float(watts_raw)
            except ValueError:
                logger.debug("Ignoring unparseable nvidia-smi output: %s", raw_line)
        return limits

    def _set_limits(self, assignments: PowerAssignments, *, force_setter: str | None = None) -> None:
        setter = force_setter or self.setter
        if setter == "nvidia-smi":
            self._set_limits_nvidia_smi(assignments)
            self._active_setter = "nvidia-smi"
            return
        if setter == "dcgmi":
            self._set_limits_dcgmi(assignments)
            self._active_setter = "dcgmi"
            return
        try:
            self._set_limits_nvidia_smi(assignments)
            self._active_setter = "nvidia-smi"
        except RuntimeError as exc:
            logger.warning("nvidia-smi power-limit write failed; trying dcgmi: %s", exc)
            self._set_limits_dcgmi(assignments)
            self._active_setter = "dcgmi"

    @staticmethod
    def _group_assignments(assignments: PowerAssignments) -> dict[tuple[float, tuple[int, ...]], list[str]]:
        by_node_and_limit: dict[tuple[str, float], list[int]] = defaultdict(list)
        for (node, gpu_index), watts in assignments.items():
            by_node_and_limit[(node, watts)].append(gpu_index)
        grouped: dict[tuple[float, tuple[int, ...]], list[str]] = defaultdict(list)
        for (node, watts), gpu_indices in by_node_and_limit.items():
            grouped[(watts, tuple(sorted(gpu_indices)))].append(node)
        return grouped

    def _set_limits_nvidia_smi(self, assignments: PowerAssignments) -> None:
        nvidia_smi = shlex.join(self.nvidia_smi_command)
        for (watts, gpu_indices), nodes in sorted(self._group_assignments(assignments).items()):
            indices = ",".join(str(index) for index in gpu_indices)
            command = f"{nvidia_smi} --id={shlex.quote(indices)} --power-limit={watts:g}"
            self._run_on_nodes(tuple(sorted(nodes)), command)

    def _set_limits_dcgmi(self, assignments: PowerAssignments) -> None:
        dcgmi = shlex.join(self.dcgmi_command)
        for (watts, gpu_indices), nodes in sorted(self._group_assignments(assignments).items()):
            indices = ",".join(str(index) for index in gpu_indices)
            group_name = f"srtslurm-{self.job_id}-{str(watts).replace('.', '_')}w"
            # DCGM configuration operates on groups. Create a short-lived group on
            # every node, set only its selected GPUs, and always remove the group.
            command = (
                "set -euo pipefail; unset CUDA_VISIBLE_DEVICES; "
                f"out=$({dcgmi} group -c {shlex.quote(group_name)} -a {shlex.quote(indices)} 2>&1); "
                "gid=$(printf '%s\\n' \"$out\" | sed -n 's/.*group ID of \\([0-9][0-9]*\\).*/\\1/p' | head -1); "
                'test -n "$gid" || { printf \'%s\\n\' "$out" >&2; exit 1; }; '
                f"trap '{dcgmi} group -d \"$gid\" >/dev/null 2>&1 || true' EXIT; "
                f"{dcgmi} config -g \"$gid\" --set -P {watts:g}"
            )
            self._run_on_nodes(tuple(sorted(nodes)), command)

    def _run_on_nodes(self, nodes: tuple[str, ...], shell_command: str) -> subprocess.CompletedProcess[str]:
        if not nodes:
            raise ValueError("At least one node is required for GPU power-limit operations")
        command = [
            "srun",
            "--jobid",
            self.job_id,
            "--overlap",
            "--nodes",
            str(len(nodes)),
            "--ntasks",
            str(len(nodes)),
            "--ntasks-per-node",
            "1",
            "--nodelist",
            ",".join(nodes),
            "bash",
            "-lc",
            shell_command,
        ]
        logger.info("GPU power command: %s", shlex.join(command))
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            output = "\n".join(part.strip() for part in (exc.stdout, exc.stderr) if part and part.strip())
            detail = f": {output}" if output else ""
            raise RuntimeError(f"GPU power command failed with exit code {exc.returncode}{detail}") from exc

    def _write_snapshot(self, filename: str, limits: PowerAssignments) -> None:
        path = self.log_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["node", "gpu_index", "power_limit_w"])
            for (node, gpu_index), watts in sorted(limits.items()):
                writer.writerow([node, gpu_index, f"{watts:g}"])
