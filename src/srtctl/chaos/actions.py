# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fault injection action implementations (kill, freeze, rolling_restart)."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from srtctl.core.processes import list_step_ids, signal_step
from srtctl.core.schema import CHAOS_PROCESS_ALIASES
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
dry_run = bool(cfg.get("dry_run", False))
seed = cfg.get("seed")
allow_unscoped = bool(cfg.get("allow_unscoped", False))
errors = []
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
    matches = discover(use_step_filter=True) if step_num is not None else []
    if not matches and allow_unscoped:
        # Containerized children may clear SLURM_STEP_ID. A pattern-only match is
        # safe only when no other worker shares this node.
        matches = discover(use_step_filter=False)
    elif not matches and step_num is None:
        errors.append("worker step id unknown and another worker shares this node; refusing unscoped match")
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
for pid, cmd in chosen:
    if dry_run:
        signalled.append({"pid": pid, "cmd": cmd[:120]})
        continue
    try:
        os.kill(pid, sig_num)
        signalled.append({"pid": pid, "cmd": cmd[:120]})
    except ProcessLookupError:
        errors.append(f"pid {pid} exited before SIG{sig_name}")
    except OSError as exc:
        errors.append(f"pid {pid}: {exc}")

verified = {}
if signalled and not dry_run and (verify_state is not None or verify_not_state is not None):
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
    wait_ready: bool = False,
    backoff_seconds: float = 0.0,
) -> int | None:
    """Prevent a targeted kill from aborting the job; opt into supervisor relaunch when requested.

    Returns the endpoint's prior restart count when the supervisor will relaunch it, else None.
    """
    live_names = ctx.active_step_names(target)
    if ctx.supervisor is not None and (respawn or wait_ready):
        return ctx.supervisor.ensure_relaunch(
            target.key,
            list(target.processes),
            live_names,
            backoff_seconds=backoff_seconds,
        )
    # Mark only the targeted worker's step(s) non-critical so surviving workers stay critical.
    for name in live_names:
        ctx.registry.set_critical(name, False)
        ctx.mark_step_killed(name)
    return None


def _wait_for_relaunches(
    ctx: ChaosContext,
    pending: list[tuple[WorkerTarget, int]],
    *,
    timeout_seconds: float,
    event_index: int,
    label: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Wait until ``WorkerSupervisor`` finishes relaunching every ``(target, prev_restarts)`` pair."""
    if ctx.supervisor is None or not pending:
        return [], False
    deadline = time.monotonic() + timeout_seconds
    remaining = list(pending)
    completed: list[dict[str, Any]] = []
    while remaining and not ctx.should_stop():
        still: list[tuple[WorkerTarget, int]] = []
        for target, prev_restarts in remaining:
            done, outcome = ctx.supervisor.relaunch_status(target.key, prev_restarts=prev_restarts)
            if done:
                completed.append({"worker": target.label, "outcome": outcome})
                logger.info("Chaos[%d] %s: %s relaunch %s", event_index, label, target.label, outcome)
            else:
                still.append((target, prev_restarts))
        remaining = still
        if not remaining or time.monotonic() >= deadline:
            break
        ctx.wait(1.0)
    timed_out = bool(remaining) and not ctx.should_stop()
    if timed_out:
        logger.warning(
            "Chaos[%d] %s: timed out after %.0fs waiting for %s",
            event_index,
            label,
            timeout_seconds,
            [t.label for t, _ in remaining],
        )
    return completed, timed_out


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


def _shares_node(target: WorkerTarget, ctx: ChaosContext) -> bool:
    """True when another worker endpoint has a process on ``target``'s leader node."""
    node = target.leader.node
    return any(key != target.key and any(p.node == node for p in procs) for key, procs in ctx.endpoint_groups.items())


def _resolve_process_pattern(event: ChaosEventConfig, ctx: ChaosContext) -> str | None:
    """Resolve ``event.effective_process`` to a cmdline substring, or None for step-level kills."""
    proc = event.effective_process
    if proc is None:
        return None
    if proc not in CHAOS_PROCESS_ALIASES:
        return proc
    backend = ctx.config.backend_for_role(event.role)
    pattern = backend.fault_process_target(proc)
    if not pattern:
        raise RuntimeError(
            f"No default process pattern for process={proc!r} on backend {backend.type!r}; "
            "set process to an explicit cmdline substring"
        )
    return pattern


def execute_kill(
    event: ChaosEventConfig,
    targets: list[WorkerTarget],
    ctx: ChaosContext,
    event_index: int,
) -> dict[str, Any]:
    """Kill the worker step(s) or matched engine/rank subprocess(es) for each targeted worker."""
    sig = event.normalized_signal
    pattern = _resolve_process_pattern(event, ctx)
    step_ids = list_step_ids(ctx.runtime.job_id) or {}
    outcomes: list[dict[str, Any]] = []
    relaunch_watch: list[tuple[WorkerTarget, int]] = []

    for target in targets:
        live_names = ctx.active_step_names(target)
        if not ctx.leashed:
            prev_restarts = _prepare_destructive_target(
                ctx,
                target,
                respawn=event.respawn,
                wait_ready=event.wait_ready,
            )
            if prev_restarts is not None and event.wait_ready:
                relaunch_watch.append((target, prev_restarts))

        if pattern is None:
            if ctx.leashed:
                step_results = [
                    {"step": name, "signal": f"SIG{sig}", "method": "leashed", "delivered": False}
                    for name in live_names
                ]
                logger.info(
                    "Chaos[%d] LEASHED kill (step): would send SIG%s to %s (%s)",
                    event_index,
                    sig,
                    target.label,
                    live_names,
                )
            else:
                step_results = _signal_worker_steps(ctx, live_names, sig, step_ids=step_ids or None)
                logger.warning(
                    "Chaos[%d] kill (step): %s -> SIG%s (%s, respawn=%s, wait_ready=%s)",
                    event_index,
                    target.label,
                    sig,
                    live_names,
                    event.respawn,
                    event.wait_ready,
                )
            outcomes.append({"worker": target.label, "steps": step_results, "respawn": event.respawn})
        else:
            step_num = _step_numeric_id(step_ids.get(live_names[0])) if live_names else None
            node = target.leader.node
            payload: dict[str, Any] = {
                "pattern": pattern,
                "step_num": step_num,
                "allow_unscoped": not _shares_node(target, ctx),
                "rank": event.rank,
                "signal": sig,
                "dry_run": ctx.leashed,
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
            if not signalled and not ctx.leashed:
                raise RuntimeError(
                    f"kill matched {res.get('matched', 0)} process(es) for pattern={pattern!r} "
                    f"rank={event.rank!r} on {target.label} ({node}); nothing signalled; errors={res.get('errors')}"
                )
            if ctx.leashed:
                logger.info(
                    "Chaos[%d] LEASHED kill (process): would send SIG%s to %s (%s) pattern=%r pids=%s",
                    event_index,
                    sig,
                    target.label,
                    node,
                    pattern,
                    [item["pid"] for item in signalled],
                )
            else:
                logger.warning(
                    "Chaos[%d] kill (process): %s (%s) pattern=%r rank=%r -> SIG%s pids=%s",
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

    details: dict[str, Any] = {
        "scope": "process" if pattern is not None else "step",
        "signal": f"SIG{sig}",
        "workers": outcomes,
    }
    if ctx.leashed:
        details["leashed"] = True
    if pattern is not None:
        details["pattern"] = pattern
    if not ctx.leashed and event.wait_ready and relaunch_watch:
        recovered, timed_out = _wait_for_relaunches(
            ctx,
            relaunch_watch,
            timeout_seconds=event.timeout_seconds,
            event_index=event_index,
            label="kill",
        )
        details["recovered"] = recovered
        details["wait_ready_timed_out"] = timed_out
    return details


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


def execute_freeze(
    event: ChaosEventConfig,
    targets: list[WorkerTarget],
    ctx: ChaosContext,
    event_index: int,
) -> dict[str, Any]:
    """Pause matched subprocess(es) with ``SIGSTOP``, verify ``/proc/<pid>/stat`` == ``T``, then resume."""
    pattern = _resolve_process_pattern(event, ctx)
    assert pattern is not None  # effective_process defaults to "engine" on freeze
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
            "allow_unscoped": not _shares_node(target, ctx),
            "rank": event.rank,
            "signal": "STOP",
            "dry_run": ctx.leashed,
            "verify_state": "T",
            "verify_timeout": 5.0,
            "seed": ctx.rng.randint(0, 2**31 - 1),
        }
        res = run_node_python(
            node,
            payload,
            log_dir=ctx.runtime.log_dir,
            tag=f"freeze_{event_index}_{target.label}",
            het_group=target.leader.het_group,
        )
        signalled = res.get("signalled") or []
        if not signalled and not ctx.leashed:
            raise RuntimeError(
                f"freeze matched {res.get('matched', 0)} process(es) for pattern={pattern!r} "
                f"rank={event.rank!r} on {target.label} ({node}); nothing frozen; errors={res.get('errors')}"
            )
        verified = res.get("verified_states") or {}
        if ctx.leashed:
            logger.info(
                "Chaos[%d] LEASHED freeze: would SIGSTOP %s (%s) pattern=%r pids=%s",
                event_index,
                target.label,
                node,
                pattern,
                [item["pid"] for item in signalled],
            )
        else:
            for item in signalled:
                sp = StalledProcess(
                    node=node, pid=int(item["pid"]), worker=target.label, het_group=target.leader.het_group
                )
                stalled_now.append(sp)
                ctx.register_stall(sp)
            bad_states = {pid: st for pid, st in verified.items() if st != "T"}
            if bad_states:
                raise RuntimeError(f"freeze SIGSTOP verification failed on {target.label} ({node}): {bad_states}")
            logger.warning(
                "Chaos[%d] freeze: %s (%s) SIGSTOP verified (state=T) pids=%s",
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
    if not ctx.leashed and event.duration_seconds is not None:
        logger.info("Chaos[%d] freeze: holding for %.1fs before SIGCONT", event_index, event.duration_seconds)
        try:
            ctx.wait(event.duration_seconds)
        finally:
            resume_results = resume_stalled_processes(stalled_now, ctx, tag=f"{event_index}")
            ctx.unregister_stalls(stalled_now)

    out: dict[str, Any] = {
        "pattern": pattern,
        "duration_seconds": event.duration_seconds,
        "workers": worker_outcomes,
        "resumed": resume_results,
    }
    if ctx.leashed:
        out["leashed"] = True
    return out


def execute_rolling_restart(
    event: ChaosEventConfig,
    targets: list[WorkerTarget],
    ctx: ChaosContext,
    event_index: int,
) -> dict[str, Any]:
    """Restart every worker of ``event.role`` in a sliding window capped at ``max_unavailable``."""
    ordered = sorted(targets, key=lambda t: (t.index, t.key[2]))
    sig = "TERM" if event.graceful else "KILL"
    cap = min(event.max_unavailable, len(ordered))
    if ctx.leashed:
        logger.info(
            "Chaos[%d] LEASHED rolling_restart: would restart %d %s worker(s) (max_unavailable=%d, SIG%s)",
            event_index,
            len(ordered),
            event.role,
            cap,
            sig,
        )
        return {
            "role": event.role,
            "signal": f"SIG{sig}",
            "max_unavailable": cap,
            "total": len(ordered),
            "restarted": [{"worker": t.label, "outcome": "leashed"} for t in ordered],
            "timed_out": False,
            "leashed": True,
        }

    if ctx.supervisor is None:
        raise RuntimeError("rolling_restart requires WorkerSupervisor on the orchestrator")

    queue = list(ordered)
    in_flight: list[tuple[WorkerTarget, int]] = []  # (target, prev_restarts)
    restarted: list[dict[str, Any]] = []
    deadline = time.monotonic() + event.timeout_seconds

    logger.warning(
        "Chaos[%d] rolling_restart: role=%s total=%d max_unavailable=%d signal=SIG%s timeout=%.0fs",
        event_index,
        event.role,
        len(ordered),
        cap,
        sig,
        event.timeout_seconds,
    )

    while (queue or in_flight) and not ctx.should_stop():
        room = cap - len(in_flight)
        if ctx.min_healthy > 0:
            healthy = ctx.healthy_worker_count(event.role)
            room = min(room, max(0, healthy - ctx.min_healthy))
        if queue and room > 0:
            step_ids = list_step_ids(ctx.runtime.job_id)
            while queue and room > 0:
                target = queue.pop(0)
                live_names = ctx.active_step_names(target)
                prev_restarts = _prepare_destructive_target(ctx, target, respawn=True, backoff_seconds=0.0)
                assert prev_restarts is not None
                _signal_worker_steps(ctx, live_names, sig, step_ids=step_ids)
                in_flight.append((target, prev_restarts))
                room -= 1
                logger.info(
                    "Chaos[%d] rolling_restart: stopped %s (%s); in_flight=%d/%d remaining=%d",
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
                restarted.append({"worker": target.label, "outcome": outcome})
                logger.info("Chaos[%d] rolling_restart: %s replacement %s", event_index, target.label, outcome)
            else:
                still_in_flight.append((target, prev_restarts))
        in_flight = still_in_flight

        if not queue and not in_flight:
            break
        if time.monotonic() >= deadline:
            logger.warning(
                "Chaos[%d] rolling_restart: timed out after %.0fs (%d/%d restarted)",
                event_index,
                event.timeout_seconds,
                len(restarted),
                len(ordered),
            )
            break
        ctx.wait(1.0)

    return {
        "role": event.role,
        "signal": f"SIG{sig}",
        "max_unavailable": cap,
        "total": len(ordered),
        "restarted": restarted,
        "timed_out": bool(queue or in_flight) and not ctx.should_stop(),
    }


ActionHandler = Callable[["ChaosEventConfig", list["WorkerTarget"], "ChaosContext", int], dict[str, Any]]

# Action dispatch table (names go in tables, never in branches).
ACTION_HANDLERS: dict[str, ActionHandler] = {
    "kill": execute_kill,
    "freeze": execute_freeze,
    "rolling_restart": execute_rolling_restart,
}
