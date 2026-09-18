# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep each profiled Slurm task alive and finalize reports before applications exit.

The wrapper owns the task process group; nsys and the app use a separate session.
Signal the task without scancel --full so the wrapper can stop collection first.
SRT_NSYS_REPORT_BARRIER_DIR and SRT_NSYS_REPORT_EXPECTED coordinate the profilers
within an MPI step before any application is allowed to shut down.
"""

from __future__ import annotations

import shlex

# Poll interval while waiting for an orphaned application (seconds).
_APP_POLL_SECS = 5
# How long to wait for nsys to fork the application before giving up on tracking it (seconds).
_CHILD_LOOKUP_SECS = 600
# After forwarding SIGTERM, how long the profiled app's process tree may take to exit before the wrapper
# TERMs/KILLs it (nsys itself is never signalled): TRT-LLM MPI rank processes were seen outliving the
# parent worker during shutdown.
_APP_EXIT_GRACE_SECS = 120


def keepalive_command(command: list[str], *, app_exit_grace_secs: int = _APP_EXIT_GRACE_SECS) -> list[str]:
    """Wrap an nsys-prefixed ``command`` (see module docstring).

    Returns ``["bash", "-c", script]``. On SIGTERM/SIGINT the wrapper forwards the signal to the
    profiled app after report finalization. If the application tree is still alive
    after ``app_exit_grace_secs``, it is TERMed and then
    KILLed so nsys can finalise the report (MPI rank processes can outlive the parent
    worker during shutdown). The container must provide Bash, setsid, pgrep,
    pkill, and timeout; missing tools fail the launch explicitly.
    """
    launch = shlex.join(command)
    # The launch helper supplies one fresh directory per step. All its ranks share it,
    # so no MPI rank exits until every report has been flushed.
    report_launch = (
        shlex.join(command[:2]) + ' --session-new "$SESSION" ' + shlex.join(command[2:])
        if len(command) > 1 and command[1] == "profile"
        else launch
    )
    nsys_binary = shlex.quote(command[0])
    report_setup = (
        "REPORT_FIRST=0; REPORT_FIRST_FAILED=0; "
        '[ -n "${SRT_NSYS_REPORT_BARRIER_DIR:-}" ] && REPORT_FIRST=1; '
        'SESSION="srtctl-${SLURM_JOB_ID:-local}-${SLURM_STEP_ID:-0}-${SLURM_PROCID:-0}-$$"; '
    )
    report_stop = (
        'if [ "$REPORT_FIRST" = 1 ]; then '
        'mkdir -p "$SRT_NSYS_REPORT_BARRIER_DIR"; '
        "deadline=$((SECONDS + ${SRT_NSYS_REPORT_STOP_TIMEOUT:-900})); "
        'echo "[srtctl] stopping capture $SESSION before application shutdown" >&2; '
        f'if timeout --kill-after=5 "${{SRT_NSYS_REPORT_STOP_TIMEOUT:-900}}" {nsys_binary} stop --session="$SESSION"; then '
        'touch "$SRT_NSYS_REPORT_BARRIER_DIR/$SESSION.ready"; '
        'else REPORT_FIRST_FAILED=1; touch "$SRT_NSYS_REPORT_BARRIER_DIR/$SESSION.failed"; fi; '
        "while :; do "
        "ready=0; failed=0; "
        'for marker in "$SRT_NSYS_REPORT_BARRIER_DIR"/*.ready; do [ -f "$marker" ] && ready=$((ready+1)); done; '
        'for marker in "$SRT_NSYS_REPORT_BARRIER_DIR"/*.failed; do [ -f "$marker" ] && failed=$((failed+1)); done; '
        'if [ "$failed" -gt 0 ]; then REPORT_FIRST_FAILED=1; break; fi; '
        '[ "$ready" -ge "${SRT_NSYS_REPORT_EXPECTED:-1}" ] && break; '
        'if [ "$SECONDS" -ge "$deadline" ]; then REPORT_FIRST_FAILED=1; break; fi; '
        "sleep 1; done; "
        'echo "[srtctl] report barrier: $ready/${SRT_NSYS_REPORT_EXPECTED:-1} ready; failed=$REPORT_FIRST_FAILED" >&2; '
        "fi; "
    )
    script = (
        'for tool in setsid pgrep pkill timeout; do command -v "$tool" >/dev/null 2>&1 || '
        '{ echo "[srtctl] nsys wrapper requires $tool" >&2; exit 127; }; done; '
        + report_setup
        + 'if [ "$REPORT_FIRST" = 1 ]; then '
        + f"setsid {report_launch} & "
        + "else "
        # nsys (and the app it forks) in their own session: the step's SIGTERM hits only this shell
        f"setsid {launch} & fi; NSYS=$!; APP=''; "
        # Snapshot descendants for application cleanup; never signal nsys itself.
        'desc() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; desc "$c"; done; }; '
        # teardown: stop the app, let nsys (still running) write its report; escalate on the app tree only
        'FWD_DONE=""; ESC=""; fwd() { [ -n "$FWD_DONE" ] && return; FWD_DONE=1; TREE=$(desc "$NSYS"); '
        + report_stop
        + 'if [ -z "$APP" ]; then APP=$(pgrep -P "$NSYS" 2>/dev/null | head -n1); fi; '
        + 'echo "[srtctl] SIGTERM: stopping profiled app pid ${APP:-?} (tree: $(echo $TREE)); nsys pid $NSYS; report-first=$REPORT_FIRST" >&2; '
        '[ -n "$APP" ] && kill -TERM "$APP" 2>/dev/null; '
        f'( sleep {app_exit_grace_secs}; TREE="$TREE $(desc "$NSYS")"; alive=""; for p in $TREE; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done; '
        'if [ -n "$alive" ]; then '
        f'echo "[srtctl] app tree still alive {app_exit_grace_secs}s after SIGTERM (pids:$alive); TERM then KILL so nsys can finalise" >&2; '
        'for p in $alive; do kill -TERM "$p" 2>/dev/null; done; sleep 20; '
        'for p in $alive; do kill -KILL "$p" 2>/dev/null; done; fi ) & ESC=$!; }; '
        "trap fwd TERM INT; "
        # Install the trap before child discovery: teardown may happen during startup.
        f"for _ in $(seq 1 {_CHILD_LOOKUP_SECS}); do "
        '[ -n "$FWD_DONE" ] && break; '
        'APP=$(pgrep -P "$NSYS" 2>/dev/null | head -n1); '
        '[ -n "$APP" ] && break; kill -0 "$NSYS" 2>/dev/null || break; sleep 1; done; '
        'rc=0; while :; do wait "$NSYS"; rc=$?; kill -0 "$NSYS" 2>/dev/null || break; done; '
        # nsys exited before the app (a --duration window): keep the task alive while the app runs
        'if [ -n "$APP" ] && kill -0 "$APP" 2>/dev/null; then '
        'echo "[srtctl] nsys (pid $NSYS) exited with $rc; keeping task alive while pid $APP runs" >&2; '
        f'while kill -0 "$APP" 2>/dev/null; do sleep {_APP_POLL_SECS}; done; fi; '
        # drop the escalation timer (and its sleep) so the task exits as soon as nsys is done
        '[ -n "${ESC:-}" ] && { pkill -TERM -P "$ESC" 2>/dev/null; kill -TERM "$ESC" 2>/dev/null; }; '
        '[ "$REPORT_FIRST_FAILED" = 1 ] && rc=1; exit "$rc"'
    )
    return ["bash", "-c", script]
