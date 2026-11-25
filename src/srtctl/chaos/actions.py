# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fault injection action implementations (kill_worker, kill_process, stall_process, rolling_replace)."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from srtctl.core.processes import list_step_ids, signal_step
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from pathlib import Path

    from srtctl.chaos.runner import ChaosContext
    from srtctl.chaos.targets import WorkerTarget
    from srtctl.core.schema import ChaosEventConfig

logger = logging.getLogger(__name__)

_NODE_ACTION_TIMEOUT_SECONDS = 30.0
_RESULT_PREFIX = "CHAOS_RESULT "

# Self-contained stdlib script executed on a worker node to discover and signal
# matching processes belonging to a worker step, with optional /proc/<pid>/stat verification.
_NODE_PROC_SIGNAL_SCRIPT = r"""
import json, os, random, signal, sys, time

cfg = json.loads(sys.argv[1])
pattern = cfg["pattern"]
step_num = cfg.get("step_num")
rank = cfg["rank"]
sig_name = cfg["signal"]
verify_state = cfg.get("verify_state")
verify_not_state = cfg.get("verify_not_state")
verify_timeout = float(cfg.get("verify_timeout", 5.0))
explicit_pids = cfg.get("pids")
seed = cfg.get("seed")
my_pid = os.getpid()
my_uid = os.getuid()

def read_stat_state(pid):
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        tail = raw.split(")", 1)
        if len(tail) < 2:
            return "?"
        fields = tail[1].strip().split()
        return fields[0] if fields else "?"
    except OSError:
        return "missing"

def env_has_step(pid, wanted_step):
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            data = fh.read()
    except OSError:
        return False
    needle = f"SLURM_STEP_ID={wanted_step}".encode("utf-8")
    return needle in data.split(b"\x00")

def discover(use_step_filter):
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != my_uid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                raw_cmd = fh.read()
        except OSError:
            continue
        if not raw_cmd:
            continue
        cmd = raw_cmd.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        if pattern not in cmd:
            continue
        if use_step_filter and step_num is not None and not env_has_step(pid, step_num):
            continue
        found.append((pid, cmd))
    found.sort(key=lambda item: item[0])
    return found

if explicit_pids is not None:
    chosen = [(int(p), "") for p in explicit_pids]
    matched_count = len(chosen)
else:
    matches = discover(use_step_filter=step_num is not None)
    if not matches and step_num is not None:
        # Containerized children may clear SLURM_STEP_ID; fall back to pattern-only match.
        matches = discover(use_step_filter=False)
    matched_count = len(matches)
    if not matches:
        chosen = []
    elif rank == "all":
        chosen = matches
    elif rank == "random":
        rng = random.Random(seed)
        chosen = [rng.choice(matches)]
    else:
        idx = int(rank)
        chosen = [matches[idx]] if 0 <= idx < len(matches) else []

sig_num = getattr(signal, f"SIG{sig_name}")
signalled = []
errors = []
for pid, cmd in chosen:
    try:
        os.kill(pid, sig_num)
        signalled.append({"pid": pid, "cmd": cmd[:120]})
    except ProcessLookupError:
        errors.append(f"pid {pid} exited before SIG{sig_name}")
    except OSError as exc:
        errors.append(f"pid {pid}: {exc}")

verified = {}
if signalled and (verify_state is not None or verify_not_state is not None):
    deadline = time.monotonic() + verify_timeout
    pending = {item["pid"] for item in signalled}
    while pending:
        for pid in list(pending):
            st = read_stat_state(pid)
            verified[str(pid)] = st
            ok = True
            if verify_state is not None and st != verify_state:
                ok = False
            if verify_not_state is not None and st == verify_not_state:
                ok = False
            if ok:
                pending.remove(pid)
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(0.2)

print("CHAOS_RESULT " + json.dumps({
    "matched": matched_count,
    "signalled": signalled,
    "verified_states": verified,
    "errors": errors,
}))
"""


@dataclass(frozen=True)
class StalledProcess:
    """One process currently paused with ``SIGSTOP`` that must be resumed on teardown."""

    node: str
    pid: int
    worker: str
    het_group: int | None = None


def _step_numeric_id(step_id: str | None) -> str | None:
    """Extract the Slurm numeric step id from ``<job>.<step>``."""
    if not step_id or "." not in step_id:
        return None
    return step_id.split(".", 1)[1]


def run_node_python(
    node: str,
    payload: dict[str, Any],
    *,
    log_dir: Path,
    tag: str,
    het_group: int | None = None,
    timeout: float = _NODE_ACTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the node-side process-signalling helper on ``node`` via an overlapping bare-host srun."""
    out_file = log_dir / f"chaos_{tag}_{node}.out"
    proc = start_srun_process(
        command=["python3", "-c", _NODE_PROC_SIGNAL_SCRIPT, json.dumps(payload)],
        nodelist=[node],
        output=str(out_file),
        container_image=None,
        overlap=True,
        use_bash_wrapper=False,
        het_group=het_group,
    )
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"Chaos node helper timed out after {timeout:.0f}s on {node}") from None

    text = out_file.read_text(errors="replace") if out_file.exists() else ""
    for line in reversed(text.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                parsed = json.loads(line[len(_RESULT_PREFIX) :])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Chaos node helper emitted invalid JSON on {node}: {line}") from exc
            if isinstance(parsed, dict):
                parsed["returncode"] = rc
                return parsed
    raise RuntimeError(f"Chaos node helper on {node} exited {rc} without result marker (see {out_file})")


def _prepare_destructive_target(
    ctx: ChaosContext,
    target: WorkerTarget,
    *,
    respawn: bool,
    backoff_seconds: float = 0.0,
) -> int | None:
    """Prevent a targeted kill from aborting the job; opt into supervisor relaunch when requested.

    Returns the endpoint's prior restart count when ``respawn`` is True, else None.
    """
    live_names = ctx.active_step_names(target)
    if respawn and ctx.supervisor is not None:
        return ctx.supervisor.ensure_relaunch(
            target.key,
            list(target.processes),
            live_names,
            backoff_seconds=backoff_seconds,
        )
    # Mark only the targeted worker's step(s) non-critical so surviving workers stay critical.
    for name in live_names:
        ctx.registry.set_critical(name, False)
    return None


def _signal_worker_steps(
    ctx: ChaosContext,
    step_names: list[str],
    sig: str,
    *,
    step_ids: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Deliver ``sig`` to every registered step of a worker endpoint."""
    results: list[dict[str, Any]] = []
    for name in step_names:
        proc = ctx.registry.get_process(name)
        if proc is None:
            results.append({"step": name, "delivered": False, "reason": "not_in_registry"})
            continue
        if not proc.is_running:
            results.append({"step": name, "delivered": False, "reason": "already_exited"})
            continue
        delivered = bool(proc.step_name) and signal_step(proc.step_name or "", sig, step_ids=step_ids)
        method = "signal_step"
        if not delivered:
            if sig == "TERM":
                proc.popen.terminate()
                method = "popen.terminate"
            else:
                proc.popen.kill()
                method = "popen.kill"
            delivered = True
        results.append(
            {"step": name, "node": proc.node, "signal": f"SIG{sig}", "method": method, "delivered": delivered}
        )
    return results


def _resolve_process_pattern(event: ChaosEventConfig, ctx: ChaosContext) -> str:
    if event.process_name:
        return event.process_name
    backend = ctx.config.backend_for_role(event.role)
    pattern = backend.fault_process_target(event.target)
    if not pattern:
        raise RuntimeError(
            f"No default process pattern for target={event.target!r} on backend {backend.type!r}; "
            "set process_name explicitly"
        )
    return pattern


def execute_kill_worker(
    event: ChaosEventConfig,
    targets: list[WorkerTarget],
    ctx: ChaosContext,
    event_index: int,
) -> dict[str, Any]:
    """Kill the Slurm step(s) of each targeted worker endpoint."""
    sig = event.normalized_signal
    step_ids = list_step_ids(ctx.runtime.job_id)
    outcomes: list[dict[str, Any]] = []
    for target in targets:
        live_names = ctx.active_step_names(target)
        _prepare_destructive_target(ctx, target, respawn=event.respawn)
        step_results = _signal_worker_steps(ctx, live_names, sig, step_ids=step_ids)
        logger.warning(
            "Chaos[%d] kill_worker: %s -> SIG%s (%s, respawn=%s)",
            event_index,
            target.label,
            sig,
            live_names,
            event.respawn,
        )
        outcomes.append({"worker": target.label, "steps": step_results, "respawn": event.respawn})
    return {"signal": f"SIG{sig}", "workers": outcomes}


def execute_kill_process(
    event: ChaosEventConfig,
    targets: list[WorkerTarget],
    ctx: ChaosContext,
    event_index: int,
) -> dict[str, Any]:
    """Send a signal to matched engine/rank subprocess(es) inside each targeted worker."""
    pattern = _resolve_process_pattern(event, ctx)
    sig = event.normalized_signal
    step_ids = list_step_ids(ctx.runtime.job_id) or {}
    outcomes: list[dict[str, Any]] = []

    for target in targets:
        live_names = ctx.active_step_names(target)
        _prepare_destructive_target(ctx, target, respawn=event.respawn)
        step_num = _step_numeric_id(step_ids.get(live_names[0])) if live_names else None
        node = target.leader.node
        payload: dict[str, Any] = {
            "pattern": pattern,
            "step_num": step_num,
            "rank": event.rank,
            "signal": sig,
            "seed": ctx.rng.randint(0, 2**31 - 1),
        }
        res = run_node_python(
            node,
            payload,
            log_dir=ctx.runtime.log_dir,
            tag=f"kill_{event_index}_{target.label}",
            het_group=target.leader.het_group,
        )
        signalled = res.get("signalled") or []
        if not signalled:
            raise RuntimeError(
                f"kill_process matched {res.get('matched', 0)} process(es) for pattern={pattern!r} "
                f"rank={event.rank!r} on {target.label} ({node}); nothing signalled"
            )
        logger.warning(
            "Chaos[%d] kill_process: %s (%s) pattern=%r rank=%r -> SIG%s pids=%s",
            event_index,
            target.label,
            node,
            pattern,
            event.rank,
            sig,
            [item["pid"] for item in signalled],
        )
        outcomes.append(
            {
                "worker": target.label,
                "node": node,
                "pattern": pattern,
                "matched": res.get("matched", 0),
                "signalled": signalled,
                "errors": res.get("errors") or [],
            }
        )
    return {"signal": f"SIG{sig}", "pattern": pattern, "workers": outcomes}


def resume_stalled_processes(
    stalled: list[StalledProcess],
    ctx: ChaosContext,
    *,
    tag: str,
) -> list[dict[str, Any]]:
    """Send ``SIGCONT`` to every paused process in ``stalled`` and verify state != ``T``."""
    by_node: dict[tuple[str, int | None], list[StalledProcess]] = {}
    for item in stalled:
        by_node.setdefault((item.node, item.het_group), []).append(item)

    results: list[dict[str, Any]] = []
    for (node, het_group), items in by_node.items():
        pids = [item.pid for item in items]
        payload = {
            "pattern": "",
            "rank": "all",
            "signal": "CONT",
            "pids": pids,
            "verify_not_state": "T",
            "verify_timeout": 5.0,
        }
        try:
            res = run_node_python(
                node,
                payload,
                log_dir=ctx.runtime.log_dir,
                tag=f"resume_{tag}",
                het_group=het_group,
            )
            results.append(
                {
                    "node": node,
                    "pids": pids,
                    "verified_states": res.get("verified_states") or {},
                    "errors": res.get("errors") or [],
                }
            )
            logger.info("Chaos resume: SIGCONT on %s pids=%s states=%s", node, pids, res.get("verified_states"))
        except Exception as exc:  # noqa: BLE001 - best effort on teardown
            logger.warning("Chaos resume failed on %s pids=%s: %s", node, pids, exc)
            results.append({"node": node, "pids": pids, "error": str(exc)})
    return results


def execute_stall_process(
    event: ChaosEventConfig,
    targets: list[WorkerTarget],
    ctx: ChaosContext,
    event_index: int,
) -> dict[str, Any]:
    """Pause matched subprocess(es) with ``SIGSTOP``, verify ``/proc/<pid>/stat`` == ``T``, then resume."""
    pattern = _resolve_process_pattern(event, ctx)
    step_ids = list_step_ids(ctx.runtime.job_id) or {}
    stalled_now: list[StalledProcess] = []
    worker_outcomes: list[dict[str, Any]] = []

    for target in targets:
        live_names = ctx.active_step_names(target)
        step_num = _step_numeric_id(step_ids.get(live_names[0])) if live_names else None
        node = target.leader.node
        payload: dict[str, Any] = {
            "pattern": pattern,
            "step_num": step_num,
            "rank": event.rank,
            "signal": "STOP",
            "verify_state": "T",
            "verify_timeout": 5.0,
            "seed": ctx.rng.randint(0, 2**31 - 1),
        }
        res = run_node_python(
            node,
            payload,
            log_dir=ctx.runtime.log_dir,
            tag=f"stall_{event_index}_{target.label}",
            het_group=target.leader.het_group,
        )
        signalled = res.get("signalled") or []
        if not signalled:
            raise RuntimeError(
                f"stall_process matched {res.get('matched', 0)} process(es) for pattern={pattern!r} "
                f"rank={event.rank!r} on {target.label} ({node}); nothing stalled"
            )
        verified = res.get("verified_states") or {}
        for item in signalled:
            sp = StalledProcess(node=node, pid=int(item["pid"]), worker=target.label, het_group=target.leader.het_group)
            stalled_now.append(sp)
            ctx.register_stall(sp)
        bad_states = {pid: st for pid, st in verified.items() if st != "T"}
        if bad_states:
            raise RuntimeError(f"stall_process SIGSTOP verification failed on {target.label} ({node}): {bad_states}")
        logger.warning(
            "Chaos[%d] stall_process: %s (%s) SIGSTOP verified (state=T) pids=%s",
            event_index,
            target.label,
            node,
            [item["pid"] for item in signalled],
        )
        worker_outcomes.append(
            {
                "worker": target.label,
                "node": node,
                "pattern": pattern,
                "signalled": signalled,
                "verified_states": verified,
            }
        )

    resume_results: list[dict[str, Any]] = []
    if event.duration_seconds is not None:
        logger.info("Chaos[%d] stall_process: holding for %.1fs before SIGCONT", event_index, event.duration_seconds)
        try:
            ctx.wait(event.duration_seconds)
        finally:
            resume_results = resume_stalled_processes(stalled_now, ctx, tag=f"{event_index}")
            ctx.unregister_stalls(stalled_now)

    return {
        "pattern": pattern,
        "duration_seconds": event.duration_seconds,
        "workers": worker_outcomes,
        "resumed": resume_results,
    }


def execute_rolling_replace(
    event: ChaosEventConfig,
    targets: list[WorkerTarget],
    ctx: ChaosContext,
    event_index: int,
) -> dict[str, Any]:
    """Replace every worker of ``event.role`` in a sliding window capped at ``max_unavailable``."""
    if ctx.supervisor is None:
        raise RuntimeError("rolling_replace requires WorkerSupervisor on the orchestrator")

    ordered = sorted(targets, key=lambda t: (t.index, t.key[2]))
    sig = "TERM" if event.graceful else "KILL"
    cap = min(event.max_unavailable, len(ordered))
    queue = list(ordered)
    in_flight: list[tuple[WorkerTarget, int]] = []  # (target, prev_restarts)
    replaced: list[dict[str, Any]] = []
    deadline = time.monotonic() + event.timeout_seconds

    logger.warning(
        "Chaos[%d] rolling_replace: role=%s total=%d max_unavailable=%d signal=SIG%s timeout=%.0fs",
        event_index,
        event.role,
        len(ordered),
        cap,
        sig,
        event.timeout_seconds,
    )

    while (queue or in_flight) and not ctx.should_stop():
        # Refill up to max_unavailable.
        if queue and len(in_flight) < cap:
            step_ids = list_step_ids(ctx.runtime.job_id)
            while queue and len(in_flight) < cap:
                target = queue.pop(0)
                live_names = ctx.active_step_names(target)
                prev_restarts = _prepare_destructive_target(ctx, target, respawn=True, backoff_seconds=0.0)
                assert prev_restarts is not None
                _signal_worker_steps(ctx, live_names, sig, step_ids=step_ids)
                in_flight.append((target, prev_restarts))
                logger.info(
                    "Chaos[%d] rolling_replace: stopped %s (%s); in_flight=%d/%d remaining=%d",
                    event_index,
                    target.label,
                    f"SIG{sig}",
                    len(in_flight),
                    cap,
                    len(queue),
                )

        still_in_flight: list[tuple[WorkerTarget, int]] = []
        for target, prev_restarts in in_flight:
            done, outcome = ctx.supervisor.relaunch_status(target.key, prev_restarts=prev_restarts)
            if done:
                replaced.append({"worker": target.label, "outcome": outcome})
                logger.info("Chaos[%d] rolling_replace: %s replacement %s", event_index, target.label, outcome)
            else:
                still_in_flight.append((target, prev_restarts))
        in_flight = still_in_flight

        if not queue and not in_flight:
            break
        if time.monotonic() >= deadline:
            logger.warning(
                "Chaos[%d] rolling_replace: timed out after %.0fs (%d/%d replaced)",
                event_index,
                event.timeout_seconds,
                len(replaced),
                len(ordered),
            )
            break
        ctx.wait(1.0)

    return {
        "role": event.role,
        "signal": f"SIG{sig}",
        "max_unavailable": cap,
        "total": len(ordered),
        "replaced": replaced,
        "timed_out": bool(queue or in_flight) and not ctx.should_stop(),
    }


ActionHandler = Callable[["ChaosEventConfig", list["WorkerTarget"], "ChaosContext", int], dict[str, Any]]

# Action dispatch table (names go in tables, never in branches).
ACTION_HANDLERS: dict[str, ActionHandler] = {
    "kill_worker": execute_kill_worker,
    "kill_process": execute_kill_process,
    "stall_process": execute_stall_process,
    "rolling_replace": execute_rolling_replace,
}
