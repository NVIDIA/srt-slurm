"""Keep an srun task alive after a time-windowed nsys session exits.

``nsys profile --delay D --duration T --kill none <app>`` stops collecting after ``T`` seconds, writes
its report and then **exits**, leaving ``<app>`` running as an orphan. Verified on hecate job 595056:
the frontend report landed at 02:09:45, the frontend srun step completed (exit 0) at 02:09:48 and the
orphaned frontend was killed with the step; the decode worker step went the same way seconds later.
Under Slurm the exiting nsys *is* the task, so the step ends and slurmstepd kills everything left in it.

The wrapper below runs nsys in the background, records the PID of its child (the profiled app), waits
for nsys, and then keeps the shell (= the task) alive while that child still exists. Iteration-based
captures (``-c cudaProfilerApi --capture-range-end=stop``) do not need this: nsys stays attached until
the app exits.
"""

from __future__ import annotations

import shlex

# Poll interval while waiting for the orphaned application (seconds).
_APP_POLL_SECS = 5
# How long to wait for nsys to fork the application before giving up on tracking it (seconds).
_CHILD_LOOKUP_SECS = 600


def keepalive_command(command: list[str]) -> list[str]:
    """Wrap an nsys-prefixed ``command`` so the task outlives the nsys session.

    Returns ``["bash", "-c", script]``. The script exits with nsys's own exit code once the
    application has gone away, so a failed nsys launch still surfaces as a failed process. If
    ``pgrep`` is unavailable or nsys never forks a child, the script degrades to today's
    behaviour (exit when nsys exits).
    """
    launch = shlex.join(command)
    script = (
        f"{launch} & NSYS=$!; APP=''; "
        f"for _ in $(seq 1 {_CHILD_LOOKUP_SECS}); do "
        "APP=$(pgrep -P \"$NSYS\" 2>/dev/null | head -n1); "
        '[ -n "$APP" ] && break; kill -0 "$NSYS" 2>/dev/null || break; sleep 1; done; '
        'wait "$NSYS"; rc=$?; '
        'if [ -n "$APP" ]; then '
        'echo "[srtctl] nsys (pid $NSYS) exited with $rc; keeping task alive while pid $APP runs" >&2; '
        f'while kill -0 "$APP" 2>/dev/null; do sleep {_APP_POLL_SECS}; done; fi; '
        'exit "$rc"'
    )
    return ["bash", "-c", script]
