# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in host-memory cleanup and diagnostics for backend nodes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

_ENABLED_VALUES = {"1", "true", "yes", "on"}


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in _ENABLED_VALUES


def stale_vllm_offload_cleanup_script(min_age_minutes: int) -> str:
    """Return a fail-closed cleanup script for stale vLLM offload mmaps."""
    return f"""
set -uo pipefail
echo \"[$(date --iso-8601=seconds)] host=$(hostname) stale-offload cleanup begin\"
if ! command -v fuser >/dev/null 2>&1; then
  echo \"fuser is unavailable; refusing to delete any mmap\"
  exit 0
fi
find /dev/shm -maxdepth 1 -type f -uid \"$(id -u)\" \\
  -name 'vllm_offload_*.mmap' -mmin +{min_age_minutes} -print0 2>/dev/null |
while IFS= read -r -d '' path; do
  if fuser -s -- \"$path\"; then
    echo \"KEEP live $path\"
  else
    echo \"REMOVE stale $path\"
    rm -f -- \"$path\"
  fi
done
echo \"[$(date --iso-8601=seconds)] host=$(hostname) stale-offload cleanup end\"
""".strip()


def host_memory_telemetry_script(interval_seconds: int) -> str:
    """Return a lightweight sampler for node, cgroup, process, and Slurm memory."""
    return f"""
set -uo pipefail
interval={interval_seconds}
while true; do
  echo \"=== sample ts=$(date --iso-8601=seconds) epoch=$(date +%s) host=$(hostname) ===\"
  echo \"--- meminfo ---\"
  grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|Shmem|Slab|SReclaimable|SUnreclaim|Unevictable|Mlocked|PageTables):' /proc/meminfo || true
  echo \"--- dev-shm ---\"
  df -B1 /dev/shm || true
  find /dev/shm -maxdepth 1 -type f -uid \"$(id -u)\" -name 'vllm_offload_*.mmap' \\
    -printf '%s %T@ %p\\n' 2>/dev/null | sort -n || true
  echo \"--- cgroup ---\"
  cg=$(awk -F: '$1 == \"0\" {{ print $3 }}' /proc/self/cgroup 2>/dev/null)
  path=\"/sys/fs/cgroup${{cg:-/}}\"
  while [[ \"$path\" == /sys/fs/cgroup* ]]; do
    echo \"cgroup_path=$path\"
    for metric in memory.current memory.max memory.peak memory.events memory.stat; do
      if [[ -r \"$path/$metric\" ]]; then
        echo \"[$metric]\"
        cat \"$path/$metric\"
      fi
    done
    [[ \"$path\" == /sys/fs/cgroup ]] && break
    path=${{path%/*}}
    [[ -n \"$path\" ]] || break
  done
  echo \"--- process-rss ---\"
  ps -eo pid,ppid,rss,vsz,stat,comm,args --sort=-rss | head -80 || true
  echo \"--- slurm-sstat ---\"
  if command -v sstat >/dev/null 2>&1 && [[ -n \"${{SLURM_JOB_ID:-}}\" ]]; then
    sstat -j \"$SLURM_JOB_ID\" --allsteps \\
      --format=JobID,State,AveRSS,MaxRSS,MaxRSSTask,MaxRSSNode,AveVMSize,MaxVMSize 2>&1 || true
  fi
  echo
  sleep \"$interval\"
done
""".strip()


class HostMemoryStageMixin:
    """Prepare optional host-memory diagnostics before backend launch."""

    config: SrtConfig
    runtime: RuntimeContext

    def _prefill_node_chunks(self) -> list[tuple[int | None, tuple[str, ...]]]:
        nodes = self.runtime.nodes.prefill_group if self.runtime.nodes.het else self.runtime.nodes.worker
        if not nodes:
            return []
        if self.runtime.nodes.het:
            return [(0, tuple(nodes))]
        return [(None, tuple(nodes))]

    def prepare_host_memory_diagnostics(self) -> list[ManagedProcess]:
        env = self.runtime.environment
        cleanup_enabled = _enabled(env.get("SRTCTL_CLEAN_STALE_VLLM_OFFLOAD"))
        telemetry_enabled = _enabled(env.get("SRTCTL_HOST_MEMORY_DIAGNOSTICS"))
        if not cleanup_enabled and not telemetry_enabled:
            return []

        min_age = max(1, int(env.get("SRTCTL_STALE_VLLM_OFFLOAD_MIN_AGE_MIN", "5")))
        interval = max(5, int(env.get("SRTCTL_HOST_MEMORY_SAMPLE_INTERVAL_S", "30")))
        managed: list[ManagedProcess] = []

        for index, (het_group, nodes) in enumerate(self._prefill_node_chunks()):
            suffix = f"g{het_group}" if het_group is not None else f"chunk{index}"
            if cleanup_enabled:
                cleanup_log = self.runtime.log_dir / f"host_memory_cleanup_{suffix}_%N.log"
                proc = start_srun_process(
                    command=["bash", "-lc", stale_vllm_offload_cleanup_script(min_age)],
                    nodes=len(nodes),
                    ntasks=len(nodes),
                    nodelist=list(nodes),
                    output=str(cleanup_log),
                    srun_options=self.runtime.srun_options,
                    het_group=het_group,
                    use_bash_wrapper=False,
                )
                rc = proc.wait(timeout=300)
                if rc != 0:
                    raise RuntimeError(f"stale vLLM offload cleanup failed with exit code {rc}")
                logger.info("Completed stale vLLM offload cleanup on %s", ",".join(nodes))

            if telemetry_enabled:
                telemetry_log = self.runtime.log_dir / f"host_memory_telemetry_{suffix}_%N.log"
                proc = start_srun_process(
                    command=["bash", "-lc", host_memory_telemetry_script(interval)],
                    nodes=len(nodes),
                    ntasks=len(nodes),
                    nodelist=list(nodes),
                    output=str(telemetry_log),
                    srun_options=self.runtime.srun_options,
                    het_group=het_group,
                    use_bash_wrapper=False,
                )
                managed.append(
                    ManagedProcess(
                        name=f"host_memory_telemetry_{suffix}",
                        popen=proc,
                        log_file=Path(telemetry_log),
                        node=",".join(nodes),
                        critical=False,
                    )
                )
                logger.info("Started host-memory telemetry on %s", ",".join(nodes))

        return managed
