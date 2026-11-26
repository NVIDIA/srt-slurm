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
import json, os, signal, sys, time

cfg = json.loads(sys.argv[1])
pattern = cfg.get("pattern", "")
step_num = cfg.get("step_num")
allow_unscoped = bool(cfg.get("allow_unscoped", False))
explicit_pids = cfg.get("pids")
sig_name = cfg.get("signal")
verify_state = cfg.get("verify_state")
verify_not_state = cfg.get("verify_not_state")
verify_timeout = float(cfg.get("verify_timeout", 5.0))
my_pid = os.getpid()
my_uid = os.getuid()
errors = []

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
        if use_step_filter and not env_has_step(pid, step_num):
            continue
        found.append((pid, cmd))
    found.sort(key=lambda item: item[0])
    return found

if explicit_pids is None:
    # Discovery only: list matching PIDs; the caller picks ranks and signals them by PID.
    matches = discover(use_step_filter=True) if step_num is not None else []
    if not matches and allow_unscoped:
        # Containerized children may clear SLURM_STEP_ID. A pattern-only match is
        # safe only when no other worker shares this node.
        matches = discover(use_step_filter=False)
    elif not matches and step_num is None:
        errors.append("worker step id unknown and another worker shares this node; refusing unscoped match")
    print("CHAOS_RESULT " + json.dumps({
        "matches": [{"pid": pid, "cmd": cmd[:120]} for pid, cmd in matches],
        "errors": errors,
    }))
    sys.exit(0)

sig_num = getattr(signal, f"SIG{sig_name}")
signalled = []
for pid in explicit_pids:
    try:
        os.kill(int(pid), sig_num)
        signalled.append(int(pid))
    except ProcessLookupError:
        errors.append(f"pid {pid} exited before SIG{sig_name}")
    except OSError as exc:
        errors.append(f"pid {pid}: {exc}")

verified = {}
if signalled and (verify_state is not None or verify_not_state is not None):
    deadline = time.monotonic() + verify_timeout
    pending = set(signalled)
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

print("CHAOS_RESULT " + json.dumps({"signalled": signalled, "verified_states": verified, "errors": errors}))
"""


@dataclass(frozen=True)
class StalledProcess:
    """One matched worker subprocess on ``node``; paused ones must be resumed on teardown."""

    node: str
    pid: int
    worker: str
    het_group: int | None = None
    cmd: str = ""


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
    # A step under roles.<role>.restart stays critical: the supervisor relaunches it and,
    # once its budget is exhausted, hands the exit back to the registry's failure check.
    for name in live_names:
        proc = ctx.registry.get_process(name)
        if proc is None or not proc.supervised:
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


def _shares_node(target: WorkerTarget, node: str, ctx: ChaosContext) -> bool:
    """True when another worker endpoint has a process on ``node``."""
    return any(key != target.key and any(p.node == node for p in procs) for key, procs in ctx.endpoint_groups.items())


def _step_for_node(ctx: ChaosContext, live_names: list[str], node: str) -> str | None:
    """The live step of a worker that runs on ``node``; a single multi-node step (MPI launch) covers every node."""
    for name in live_names:
        proc = ctx.registry.get_process(name)
        if proc is not None and proc.node == node:
            return name
    return live_names[0] if len(live_names) == 1 else None


def _discover_processes(
    target: WorkerTarget,
    pattern: str,
    ctx: ChaosContext,
    step_ids: dict[str, str],
    *,
    tag: str,
) -> tuple[list[StalledProcess], list[str]]:
    """Matching subprocesses on every node of ``target`` (node order, then PID), plus helper errors."""
    live_names = ctx.active_step_names(target)
    found: list[StalledProcess] = []
    errors: list[str] = []
    seen: set[str] = set()
    for proc in target.processes:
        if proc.node in seen:
            continue
        seen.add(proc.node)
        step_name = _step_for_node(ctx, live_names, proc.node)
        payload = {
            "pattern": pattern,
            "step_num": _step_numeric_id(step_ids.get(step_name)) if step_name else None,
            "allow_unscoped": not _shares_node(target, proc.node, ctx),
        }
        res = run_node_python(
            proc.node, payload, log_dir=ctx.runtime.log_dir, tag=f"{tag}_discover", het_group=proc.het_group
        )
        errors.extend(f"{proc.node}: {err}" for err in res.get("errors") or [])
        found.extend(
            StalledProcess(
                node=proc.node, pid=int(m["pid"]), worker=target.label, het_group=proc.het_group, cmd=m.get("cmd", "")
            )
            for m in res.get("matches") or []
        )
    return found, errors


def _select_rank(matches: list[StalledProcess], rank: int | str, ctx: ChaosContext) -> list[StalledProcess]:
    if rank == "all":
        return list(matches)
    if not matches:
        return []
    if rank == "random":
        return [ctx.rng.choice(matches)]
    idx = int(rank)
    return [matches[idx]] if idx < len(matches) else []


def _signal_processes(
    procs: list[StalledProcess],
    sig: str,
    ctx: ChaosContext,
    *,
    tag: str,
    verify_state: str | None = None,
    verify_not_state: str | None = None,
) -> list[dict[str, Any]]:
    """Send ``sig`` to ``procs`` by PID, one helper per node, and report each node's verified states."""
    by_node: dict[tuple[str, int | None], list[int]] = {}
    for item in procs:
        by_node.setdefault((item.node, item.het_group), []).append(item.pid)
    results: list[dict[str, Any]] = []
    for (node, het_group), pids in by_node.items():
        payload: dict[str, Any] = {"pids": pids, "signal": sig, "verify_timeout": 5.0}
        if verify_state is not None:
            payload["verify_state"] = verify_state
        if verify_not_state is not None:
            payload["verify_not_state"] = verify_not_state
        res = run_node_python(node, payload, log_dir=ctx.runtime.log_dir, tag=tag, het_group=het_group)
        results.append(
            {
                "node": node,
                "pids": pids,
                "signalled": res.get("signalled") or [],
                "verified_states": res.get("verified_states") or {},
                "errors": res.get("errors") or [],
            }
        )
    return results


def _proc_info(procs: list[StalledProcess]) -> list[dict[str, Any]]:
    return [{"node": p.node, "pid": p.pid, "cmd": p.cmd} for p in procs]


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

    def arm(target: WorkerTarget) -> None:
        prev_restarts = _prepare_destructive_target(ctx, target, respawn=event.respawn, wait_ready=event.wait_ready)
        if prev_restarts is not None and event.wait_ready:
            relaunch_watch.append((target, prev_restarts))

    for target in targets:
        if pattern is None:
            live_names = ctx.active_step_names(target)
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
                arm(target)
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
            continue

        # Discover before arming, so a selector that matches nothing leaves the worker untouched.
        tag = f"kill_{event_index}_{target.label}"
        matches, errors = _discover_processes(target, pattern, ctx, step_ids, tag=tag)
        chosen = _select_rank(matches, event.rank, ctx)
        if not chosen and not ctx.leashed:
            raise RuntimeError(
                f"kill matched {len(matches)} process(es) for pattern={pattern!r} rank={event.rank!r} "
                f"on {target.label}; nothing to signal; errors={errors}"
            )
        signal_results: list[dict[str, Any]] = []
        if ctx.leashed:
            logger.info(
                "Chaos[%d] LEASHED kill (process): would send SIG%s to %s pattern=%r %s",
                event_index,
                sig,
                target.label,
                pattern,
                _proc_info(chosen),
            )
        else:
            arm(target)
            signal_results = _signal_processes(chosen, sig, ctx, tag=tag)
            logger.warning(
                "Chaos[%d] kill (process): %s pattern=%r rank=%r -> SIG%s %s",
                event_index,
                target.label,
                pattern,
                event.rank,
                sig,
                _proc_info(chosen),
            )
        outcomes.append(
            {
                "worker": target.label,
                "pattern": pattern,
                "matched": len(matches),
                "chosen": _proc_info(chosen),
                "signal_results": signal_results,
                "errors": errors,
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
) -> list[StalledProcess]:
    """Send ``SIGCONT`` to ``stalled``, verify state != ``T``, and stop tracking the ones that resumed or exited.

    Returns the processes still not confirmed resumed; they stay registered so teardown retries them.
    """
    if not stalled:
        return []
    try:
        results = _signal_processes(stalled, "CONT", ctx, tag=f"resume_{tag}", verify_not_state="T")
    except Exception as exc:  # noqa: BLE001 - best effort; teardown retries what stays registered
        logger.warning("Chaos resume (%s) failed: %s", tag, exc)
        return list(stalled)
    resumed = {(r["node"], int(pid)) for r in results for pid, state in r["verified_states"].items() if state != "T"}
    for r in results:
        logger.info(
            "Chaos resume: SIGCONT on %s pids=%s states=%s errors=%s",
            r["node"],
            r["pids"],
            r["verified_states"],
            r["errors"],
        )
    done = [sp for sp in stalled if (sp.node, sp.pid) in resumed]
    ctx.unregister_stalls(done)
    return [sp for sp in stalled if (sp.node, sp.pid) not in resumed]


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

    # Resolve every target before pausing anything, so a later discovery failure cannot strand earlier pauses.
    plans: list[tuple[WorkerTarget, list[StalledProcess], list[StalledProcess], list[str]]] = []
    for target in targets:
        matches, errors = _discover_processes(
            target, pattern, ctx, step_ids, tag=f"freeze_{event_index}_{target.label}"
        )
        chosen = _select_rank(matches, event.rank, ctx)
        if not chosen and not ctx.leashed:
            raise RuntimeError(
                f"freeze matched {len(matches)} process(es) for pattern={pattern!r} rank={event.rank!r} "
                f"on {target.label}; nothing to freeze; errors={errors}"
            )
        plans.append((target, matches, chosen, errors))

    worker_outcomes: list[dict[str, Any]] = []
    if ctx.leashed:
        for target, matches, chosen, errors in plans:
            logger.info(
                "Chaos[%d] LEASHED freeze: would SIGSTOP %s pattern=%r %s",
                event_index,
                target.label,
                pattern,
                _proc_info(chosen),
            )
            worker_outcomes.append(
                {"worker": target.label, "matched": len(matches), "chosen": _proc_info(chosen), "errors": errors}
            )
        return {
            "pattern": pattern,
            "duration_seconds": event.duration_seconds,
            "workers": worker_outcomes,
            "leashed": True,
        }

    stalled_now: list[StalledProcess] = []
    try:
        for target, matches, chosen, errors in plans:
            if ctx.should_stop():
                raise RuntimeError("benchmark teardown started before the freeze was applied")
            for sp in chosen:
                ctx.register_stall(sp)
                stalled_now.append(sp)
            results = _signal_processes(
                chosen, "STOP", ctx, tag=f"freeze_{event_index}_{target.label}", verify_state="T"
            )
            bad = {
                f"{r['node']}:{pid}": state
                for r in results
                for pid, state in r["verified_states"].items()
                if state != "T"
            }
            unsignalled = [
                p.pid for p in chosen if all(p.pid not in r["signalled"] for r in results if r["node"] == p.node)
            ]
            if bad or unsignalled:
                raise RuntimeError(
                    f"freeze SIGSTOP failed on {target.label}: states={bad} unsignalled={unsignalled} "
                    f"errors={[e for r in results for e in r['errors']]}"
                )
            logger.warning(
                "Chaos[%d] freeze: %s SIGSTOP verified (state=T) %s", event_index, target.label, _proc_info(chosen)
            )
            worker_outcomes.append(
                {
                    "worker": target.label,
                    "matched": len(matches),
                    "chosen": _proc_info(chosen),
                    "verified_states": {r["node"]: r["verified_states"] for r in results},
                    "errors": errors,
                }
            )
    except BaseException:
        resume_stalled_processes(stalled_now, ctx, tag=f"{event_index}_abort")
        raise

    not_resumed: list[StalledProcess] = []
    if event.duration_seconds is not None:
        logger.info("Chaos[%d] freeze: holding for %.1fs before SIGCONT", event_index, event.duration_seconds)
        try:
            ctx.wait(event.duration_seconds)
        finally:
            not_resumed = resume_stalled_processes(stalled_now, ctx, tag=f"{event_index}")
        if not_resumed:
            raise RuntimeError(f"freeze SIGCONT not verified for {_proc_info(not_resumed)}; retrying on teardown")

    return {"pattern": pattern, "duration_seconds": event.duration_seconds, "workers": worker_outcomes}


# Relaunch outcomes after which the replacement counts as serving (see WorkerSupervisor.relaunch_status).
_SERVING_OUTCOMES = frozenset({"ready", "relaunched", "untracked"})


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
            if not done:
                still_in_flight.append((target, prev_restarts))
                continue
            restarted.append({"worker": target.label, "outcome": outcome})
            logger.info("Chaos[%d] rolling_restart: %s replacement %s", event_index, target.label, outcome)
            if outcome not in _SERVING_OUTCOMES:
                # Never take down the next worker while a replacement is not serving.
                raise RuntimeError(
                    f"rolling_restart stopped: {target.label} replacement {outcome} "
                    f"({len(restarted)}/{len(ordered)} restarted)"
                )
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
