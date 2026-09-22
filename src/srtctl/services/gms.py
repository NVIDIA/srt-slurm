# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: gms``: Dynamo's GPU Memory Service next to every vLLM worker (``engine.failover``).

Implied by ``engine.failover``, one instance per worker (``placement.per: worker``)
in the job container, running with the worker's ``CUDA_VISIBLE_DEVICES``. It owns
the model weights in CUDA VMM allocations that outlive any engine process, so a
shadow engine maps the copy already in HBM and a relaunched engine imports it
instead of loading from disk. Starts before the workers, is critical (a worker
without its weight server cannot recover), and is stopped after the engines at
cleanup, when it removes the worker's socket directory.

Why per worker and not per node: GMS names its sockets after the NVML UUID of
"device k" while the engine picks device k by CUDA index. With a subset
``CUDA_VISIBLE_DEVICES`` those disagree (NVML ignores it), so a node-level server
would hand the worker on GPU 1 the socket of GPU 0. In the worker's own device
view both sides name the same socket. ``gpu_memory_service.cli.server`` enumerates
GPUs with NVML too and would start a server for every GPU on the node, so the
per-device servers are launched directly. See ``docs/shadow-engine-recovery.md``.
"""

from __future__ import annotations

import re
import shlex
from typing import TYPE_CHECKING

from marshmallow import ValidationError

from srtctl.services.config import LogProbe, ServiceReadinessConfig
from srtctl.services.registry import ServiceKind, ServiceLaunchContext, register_service

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig
    from srtctl.services.config import ServiceConfig

# Line the sidecar prints once every server has bound its sockets; the stage's readiness probe.
GMS_READY_MARKER = "GMS ready:"
# Every GMS server binds one socket per logical pool (weights, kv_cache).
GMS_SOCKETS_PER_DEVICE = 2
# Argv shown by dry-run, where there is no worker to size the script for.
PREVIEW_COMMAND = ["python3", "-m", "gpu_memory_service", "--device", "<0..worker_gpu_count-1>"]


def build_gms_sidecar_command(socket_dir: str, device_count: int, startup_timeout_seconds: int) -> list[str]:
    """One GMS server per GPU of the worker, supervised by bash.

    Prints ``GMS ready: ...`` once every server has bound both of its sockets;
    exits 1 if that takes longer than ``startup_timeout_seconds`` or a server dies
    first. Afterwards any server exit stops the rest and the step exits non-zero,
    so the registry sees a dead sidecar. On SIGTERM (job cleanup, after the engines
    are gone) it stops the servers, removes the directory, and exits 0.
    """
    expected = GMS_SOCKETS_PER_DEVICE * device_count
    script = f"""set -u
dir={shlex.quote(socket_dir)}
mkdir -p "$dir" && rm -f "$dir"/gms_*.sock || exit 1
export GMS_SOCKET_DIR="$dir"
pids=()
for device in $(seq 0 {device_count - 1}); do
    python3 -m gpu_memory_service --device "$device" &
    pids+=($!)
done
stopping=0
stop() {{ stopping=1; kill -TERM "${{pids[@]}}" 2>/dev/null; }}
trap stop TERM INT
count=0
for _ in $(seq 1 {startup_timeout_seconds}); do
    for pid in "${{pids[@]}}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "GMS server pid $pid exited during startup" >&2
            stop; wait; exit 1
        fi
    done
    count=$(ls "$dir"/gms_*.sock 2>/dev/null | wc -l)
    if [ "$count" -ge {expected} ]; then break; fi
    sleep 1
done
if [ "$count" -lt {expected} ]; then
    echo "GMS startup timed out: $count of {expected} sockets in $dir after {startup_timeout_seconds}s" >&2
    stop; wait; exit 1
fi
echo "{GMS_READY_MARKER} {device_count} device(s), $count sockets in $dir"
wait -n "${{pids[@]}}"
rc=$?
if [ "$stopping" = 1 ]; then
    wait
    rm -rf -- "$dir"
    exit 0
fi
echo "GMS server exited (code=$rc); stopping the rest" >&2
stop; wait
if [ "$rc" = 0 ]; then rc=1; fi
exit "$rc"
"""
    return ["bash", "-c", script]


@register_service("gms")
class GMSService(ServiceKind):
    """GPU Memory Service sidecar, one per vLLM worker; implied by ``engine.failover``."""

    builds_command = True
    default_start = "before_workers"
    default_critical = True
    default_placement = "workers"
    default_per = "worker"
    default_readiness_timeout = 120

    def validate(self, service: ServiceConfig, config: SrtConfig) -> None:
        if config.backend.failover is None:
            raise ValidationError(
                f"services[{service.name}] (type gms) requires engine.failover, which is what the engines "
                "load their weights through"
            )
        if service.effective_per != "worker":
            raise ValidationError(
                f"services[{service.name}] (type gms) runs one instance per worker in that worker's device view; "
                "set placement.per: worker"
            )

    def _worker_dir(self, ctx: ServiceLaunchContext) -> str | None:
        failover = getattr(ctx.config.backend, "failover", None) if ctx.config is not None else None
        if failover is None or ctx.process is None:
            return None
        from srtctl.backends.vllm import failover_worker_dir

        return failover_worker_dir(failover.shared_dir, ctx.runtime.job_id, ctx.process)

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        worker_dir = self._worker_dir(ctx)
        if worker_dir is None or ctx.process is None:
            return list(PREVIEW_COMMAND)
        timeout = service.readiness.timeout_seconds if service.readiness is not None else self.default_readiness_timeout
        return build_gms_sidecar_command(worker_dir, len(ctx.process.gpu_indices), timeout)

    def forced_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        worker_dir = self._worker_dir(ctx)
        return {"GMS_SOCKET_DIR": worker_dir} if worker_dir is not None else {}

    def readiness(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> ServiceReadinessConfig | None:
        return ServiceReadinessConfig(
            log=LogProbe(pattern=re.escape(GMS_READY_MARKER)),
            timeout_seconds=self.default_readiness_timeout + 30,  # the script's own timeout fires first
            interval_seconds=1,
        )
