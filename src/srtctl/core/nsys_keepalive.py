"""Keep an srun task alive after a time-windowed nsys session exits, and let nsys finish its report at teardown.

Two verified failure modes drive this wrapper (hecate, TRT-LLM disagg + Dynamo, nsys 2026.3.0):

1. ``nsys profile --delay D --duration T --kill none <app>`` writes its report when the window closes and
   then **exits**, leaving ``<app>`` orphaned. Under Slurm the exiting nsys *is* the task, so the step ends
   and slurmstepd kills the orphan (job 595056: frontend report 02:09:45, frontend step COMPLETED 02:09:48).
2. Even when the task is kept alive, the profiled process **stalls the moment the nsys process exits**
   (job 596172: the decode engine iterated normally through the window end and the report write, then
   stopped within one second of its nsys exiting, lost its etcd lease and was dropped by the router).

So the preferred mode is "capture until exit" (no ``--duration``): nsys stays attached and writes the
report when the app exits. That moves the report to teardown, where srtctl sends SIGTERM to the srun
step. This wrapper therefore (a) starts nsys in its own session/process group (``setsid``) so the step's
SIGTERM reaches only the wrapper shell, (b) forwards SIGTERM/SIGINT to the profiled app only, so the app
exits and nsys — still alive — finalises the report, and (c) exits with nsys's own code once nsys is
gone. If nsys does exit early (a ``--duration`` window), the wrapper still waits for the app, which at
least keeps the step alive. Iteration-based captures (``-c cudaProfilerApi``) do not need this wrapper:
nsys stays attached until the app exits by itself.
"""

from __future__ import annotations

import shlex

# Poll interval while waiting for an orphaned application (seconds).
_APP_POLL_SECS = 5
# How long to wait for nsys to fork the application before giving up on tracking it (seconds).
_CHILD_LOOKUP_SECS = 600
# After forwarding SIGTERM, how long the profiled app's process tree may take to exit before the wrapper
# TERMs/KILLs it (nsys itself is never signalled): TRT-LLM MPI rank processes were seen outliving the
# Dynamo worker's own shutdown by more than 10 minutes (hecate 596583).
_APP_EXIT_GRACE_SECS = 120


def keepalive_command(command: list[str], *, app_exit_grace_secs: int = _APP_EXIT_GRACE_SECS) -> list[str]:
    """Wrap an nsys-prefixed ``command`` (see module docstring).

    Returns ``["bash", "-c", script]``. On SIGTERM/SIGINT the wrapper forwards the signal to the
    profiled app; if the app's process tree (snapshotted at that moment, so nsys and its report
    importer are never touched) is still alive after ``app_exit_grace_secs``, it is TERMed and then
    KILLed so nsys can finalise the report (hecate 596583: TRT-LLM MPI rank processes outlived the
    Dynamo worker's shutdown by more than 10 minutes). Degrades gracefully: without ``setsid``/``pgrep``
    in the image the script still runs nsys and waits for it, i.e. today's behaviour.
    """
    launch = shlex.join(command)
    script = (
        # nsys (and the app it forks) in their own session: the step's SIGTERM hits only this shell
        f"if command -v setsid >/dev/null 2>&1; then setsid {launch} & else {launch} & fi; NSYS=$!; APP=''; "
        f"for _ in $(seq 1 {_CHILD_LOOKUP_SECS}); do "
        'APP=$(pgrep -P "$NSYS" 2>/dev/null | head -n1); '
        '[ -n "$APP" ] && break; kill -0 "$NSYS" 2>/dev/null || break; sleep 1; done; '
        # every descendant of nsys = the app tree (nsys itself excluded)
        'desc() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; desc "$c"; done; }; '
        # teardown: stop the app, let nsys (still running) write its report; escalate on the app tree only
        'FWD_DONE=""; ESC=""; fwd() { [ -n "$FWD_DONE" ] && return; FWD_DONE=1; TREE=$(desc "$NSYS"); '
        'echo "[srtctl] SIGTERM: stopping profiled app pid ${APP:-?} (tree: $(echo $TREE)); nsys pid $NSYS keeps running to write its report" >&2; '
        '[ -n "$APP" ] && kill -TERM "$APP" 2>/dev/null; '
        f'( sleep {app_exit_grace_secs}; alive=""; for p in $TREE; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done; '
        'if [ -n "$alive" ]; then '
        f'echo "[srtctl] app tree still alive {app_exit_grace_secs}s after SIGTERM (pids:$alive); TERM then KILL so nsys can finalise" >&2; '
        'for p in $alive; do kill -TERM "$p" 2>/dev/null; done; sleep 20; '
        'for p in $alive; do kill -KILL "$p" 2>/dev/null; done; fi ) & ESC=$!; }; '
        "trap fwd TERM INT; "
        'rc=0; while kill -0 "$NSYS" 2>/dev/null; do wait "$NSYS"; w=$?; kill -0 "$NSYS" 2>/dev/null || rc=$w; done; '
        # nsys exited before the app (a --duration window): keep the task alive while the app runs
        'if [ -n "$APP" ] && kill -0 "$APP" 2>/dev/null; then '
        'echo "[srtctl] nsys (pid $NSYS) exited with $rc; keeping task alive while pid $APP runs" >&2; '
        f'while kill -0 "$APP" 2>/dev/null; do sleep {_APP_POLL_SECS}; done; fi; '
        # drop the escalation timer (and its sleep) so the task exits as soon as nsys is done
        '[ -n "${ESC:-}" ] && { pkill -TERM -P "$ESC" 2>/dev/null; kill -TERM "$ESC" 2>/dev/null; }; '
        'exit "$rc"'
    )
    return ["bash", "-c", script]
