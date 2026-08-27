# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host and per-process telemetry sampler -> ``host_samples.jsonl``.

The Prometheus scrape covers what Dynamo and TRT-LLM choose to publish. It says nothing
about the machine underneath them, and three documented issues live entirely there:

* **Host CPU saturation and lock convoys.** A frontend pinned at 100% of one core looks
  identical, in every published metric, to a frontend that is idle -- both report low
  queue depth and low in-flight. The discriminator is host CPU busy % together with
  *involuntary* context switches per thread: a convoy shows as a thread being descheduled
  against its will, which no application-level counter can express.
* **File-descriptor exhaustion.** An accept loop that has run out of descriptors reports
  no error anywhere; connections are simply refused before the application sees them.
  ``ulimit -n`` against the live open-fd count is the only warning available.
* **A load generator that has become the bottleneck.** Client 100% / server 0% is the
  whole diagnosis, and it is invisible from the server side by construction.

Everything here is read from ``/proc``, so there is no dependency to install and no
privilege required. The sampler runs in-process on the node driving the sweep -- the same
node that hosts the frontend -- so it observes that host and the local Dynamo/AIPerf
processes. It deliberately does NOT try to reach the worker nodes: that would need an
srun round-trip per sample, which is exactly the cost the scraper's opt-in exists to
avoid.

Best-effort by construction: every failure is logged and
swallowed. Host telemetry is observability, never a dependency of the benchmark path.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_NR_MIGRATIONS_RE = re.compile(r"se\.nr_migrations\s*:\s*(\d+)")

# Process names worth sampling individually. Matched as substrings against the
# process's own cmdline, so a wrapper script does not hide the real one.
_PROC_PATTERNS = ("dynamo", "aiperf", "http-server", "trtllm")


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def _cpu_totals() -> tuple[int, int] | None:
    """(busy_jiffies, total_jiffies) from ``/proc/stat``'s aggregate line.

    Returned as cumulative counters rather than a percentage: a percentage needs two
    samples, and computing it here would bake in this sampler's interval. The consumer
    differences consecutive rows and can use any window it likes.
    """
    txt = _read("/proc/stat")
    if not txt:
        return None
    for line in txt.splitlines():
        if line.startswith("cpu "):
            f = [int(x) for x in line.split()[1:]]
            if len(f) < 5:
                return None
            idle = f[3] + (f[4] if len(f) > 4 else 0)  # idle + iowait
            return sum(f) - idle, sum(f)
    return None


def _procs_running_blocked() -> tuple[int | None, int | None]:
    """(procs_running, procs_blocked) from ``/proc/stat``.

    Instantaneous run-queue pressure for the whole host: sustained
    procs_running above the core count is contention no per-process
    counter can localize.
    """
    txt = _read("/proc/stat") or ""
    running = blocked = None
    for line in txt.splitlines():
        if line.startswith("procs_running"):
            with contextlib.suppress(IndexError, ValueError):
                running = int(line.split()[1])
        elif line.startswith("procs_blocked"):
            with contextlib.suppress(IndexError, ValueError):
                blocked = int(line.split()[1])
    return running, blocked


def _meminfo() -> dict:
    txt = _read("/proc/meminfo") or ""
    out = {}
    for line in txt.splitlines():
        k, _, rest = line.partition(":")
        if k in ("MemTotal", "MemAvailable"):
            with contextlib.suppress(IndexError, ValueError):
                out[k] = int(rest.split()[0])  # kB
    return out


# Launcher processes whose cmdline mentions a worker without BEING one. On a real node
# `srun` appears once per launched process, so a naive cmdline match spends most of the
# budget on wrappers: observed 14 of 24 slots on theia0019 (job 2753007), crowding out
# the very workers the sampler exists to watch. Wrappers are kept, but sorted last.
_WRAPPER_COMMS = ("srun", "bash", "sh", "slurmstepd", "timeout", "env")


def _comm(pid: str) -> str:
    return (_read(f"/proc/{pid}/comm") or "").strip()


def _interesting_pids(limit: int = 32) -> list[int]:
    """PIDs worth sampling, real processes before launchers.

    Bounded so a storm of short-lived children cannot blow up a row, and ORDERED so the
    bound falls on wrappers rather than on the workers and the client -- which are the
    only processes any of PERF-37/38/40 is actually about.
    """
    real, wrappers = [], []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    own_pid = os.getpid()
    for e in entries:
        if not e.isdigit() or int(e) == own_pid:
            # Never sample self: the standalone sampler's own cmdline carries the
            # log-dir path, which routinely contains a pattern word ("dynamo").
            continue
        cmd = _read(f"/proc/{e}/cmdline")
        if not cmd or not any(p in cmd for p in _PROC_PATTERNS):
            continue
        (wrappers if _comm(e) in _WRAPPER_COMMS else real).append(int(e))
    return (real + wrappers)[:limit]


def _proc_sample(pid: int) -> dict | None:
    """Per-process CPU, RSS, context switches and open-fd count."""
    status = _read(f"/proc/{pid}/status")
    stat = _read(f"/proc/{pid}/stat")
    if not status or not stat:
        return None
    d: dict = {"pid": pid}
    for line in status.splitlines():
        k, _, rest = line.partition(":")
        rest = rest.strip()
        if k == "Name":
            d["name"] = rest
        elif k == "VmRSS":
            d["rss_kb"] = int(rest.split()[0]) if rest.split() else None
        elif k == "Threads":
            d["threads"] = int(rest)
        elif k == "voluntary_ctxt_switches":
            d["ctx_vol"] = int(rest)
        elif k == "nonvoluntary_ctxt_switches":
            # THE lock-convoy signal: the scheduler took the CPU away rather than the
            # thread yielding it. A rising rate here with flat throughput is contention.
            d["ctx_invol"] = int(rest)
    with contextlib.suppress(IndexError, ValueError):
        # utime+stime, fields 14/15 after the (possibly space-containing) comm field.
        tail = stat[stat.rfind(")") + 2 :].split()
        d["cpu_jiffies"] = int(tail[11]) + int(tail[12])
    schedstat = _read(f"/proc/{pid}/schedstat")
    if schedstat:
        with contextlib.suppress(IndexError, ValueError):
            parts = schedstat.split()
            # Field 2 is cumulative RUN-QUEUE WAIT: time the task was runnable but
            # not running. THE scheduler-contention signal — a rising rate here with
            # idle CPUs elsewhere means this task is losing its core to neighbors,
            # which is exactly what CPU pinning remedies.
            d["cpu_ns"] = int(parts[0])
            d["run_delay_ns"] = int(parts[1])
    sched = _read(f"/proc/{pid}/sched")
    if sched:
        m = _NR_MIGRATIONS_RE.search(sched)
        if m:
            # Cross-core migrations: near-zero when pinned, so this doubles as a
            # direct observable of whether pinning is in effect AND of scheduler
            # churn when it is not.
            d["nr_migrations"] = int(m.group(1))
    with contextlib.suppress(OSError):
        # Allowed-CPU count: 144 = free-floating on GB200/GB300, 36 = taskset-pinned
        # rank. Reads the pinning CONFIG state directly, no inference needed.
        d["affinity_ncpus"] = len(os.sched_getaffinity(pid))
    try:
        d["open_fds"] = len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        # Reading another user's fd dir is denied; absence is not zero.
        d["open_fds"] = None
    return d


def _fd_limit() -> int | None:
    """Soft RLIMIT_NOFILE -- the ceiling an accept loop actually hits."""
    try:
        import resource

        return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except (ImportError, OSError, ValueError):
        return None


def _established_connections() -> int | None:
    """Count of established TCP connections, from ``/proc/net/tcp``.

    State 01 is ESTABLISHED in the kernel's hex encoding. Counted rather than listed:
    the number is the signal, and the peer list would be unbounded.
    """
    total = 0
    seen = False
    for p in ("/proc/net/tcp", "/proc/net/tcp6"):
        txt = _read(p)
        if txt is None:
            continue
        seen = True
        for line in txt.splitlines()[1:]:
            parts = line.split()
            if len(parts) > 3 and parts[3] == "01":
                total += 1
    return total if seen else None


class HostSampler:
    """Samples ``/proc`` into a JSONL file until stopped."""

    def __init__(self, log_dir: Path, interval_seconds: float = 2.0, output_name: str = "host_samples.jsonl") -> None:
        self.output_path = Path(log_dir) / output_name
        # Floor of 1 s: below that the sampler's own /proc walk starts showing up in the
        # CPU figure it is trying to measure.
        self.interval_seconds = max(1.0, float(interval_seconds))
        self._thread: threading.Thread | None = None
        self._own_stop = threading.Event()
        self.samples = 0

    def start(self, stop_event: threading.Event) -> None:
        self._thread = threading.Thread(target=self._loop, args=(stop_event,), name="HostSampler", daemon=True)
        self._thread.start()
        logger.info(
            "Host sampler started: /proc every %.1fs -> %s (fd limit %s)",
            self.interval_seconds,
            self.output_path,
            _fd_limit(),
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._own_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("Host sampler stopped: %d samples -> %s", self.samples, self.output_path)

    def _loop(self, stop_event: threading.Event) -> None:
        fd_limit = _fd_limit()
        try:
            with open(self.output_path, "a") as out:
                while not stop_event.is_set() and not self._own_stop.is_set():
                    try:
                        cpu = _cpu_totals()
                        running, blocked = _procs_running_blocked()
                        row = {
                            "t": time.time(),
                            # Monotonic companion: cross-node wall clocks can disagree by
                            # seconds (observed 2.1s NVL72 skew); pairs of (t, t_mono) let
                            # the consumer estimate per-node offsets post-hoc.
                            "t_mono": time.monotonic(),
                            "host": os.uname().nodename,
                            "cpu_busy_jiffies": cpu[0] if cpu else None,
                            "cpu_total_jiffies": cpu[1] if cpu else None,
                            "procs_running": running,
                            "procs_blocked": blocked,
                            "loadavg": (_read("/proc/loadavg") or "").split()[:3],
                            "mem": _meminfo(),
                            "fd_limit": fd_limit,
                            "established_conns": _established_connections(),
                            "procs": [s for s in (_proc_sample(p) for p in _interesting_pids()) if s],
                        }
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        self.samples += 1
                    except Exception as exc:  # noqa: BLE001 - best effort
                        logger.debug("host sample failed: %s", exc)
                    stop_event.wait(self.interval_seconds) or self._own_stop.wait(0)
        except OSError as exc:
            logger.warning("Host sampler could not write %s: %s", self.output_path, exc)


def try_start_host_sampler(log_dir: Path, observability, stop_event: threading.Event) -> HostSampler | None:
    """Start the sampler if ``observability`` opts in, else return ``None``.

    Single entry point for :class:`BenchmarkStageMixin`. All failures are
    logged and swallowed -- host telemetry never blocks a benchmark.

    Gated on ``observability.enabled`` rather than a new knob: it answers the
    same question (is this run capturing observability?) and a second switch would let a
    run be half-instrumented in a way nobody intends.
    """
    if getattr(observability, "enabled", False) is not True:
        return None
    try:
        s = HostSampler(log_dir)
        s.start(stop_event)
        return s
    except Exception as exc:  # noqa: BLE001 - best effort
        logger.warning("Host sampler failed to start (continuing without it): %s", exc)
        return None


def plan_remote_sampler_nodes(all_nodes: list[str], local_host: str) -> list[str]:
    """Nodes that need a standalone sampler: every allocated node except the one
    already covered by the in-process sampler. Pure so it is unit-testable.

    ``local_host`` may be a short hostname while the nodelist carries FQDNs (or
    vice versa); match on the first dot-separated label to be safe.
    """
    local_label = local_host.split(".")[0]
    seen: set[str] = set()
    out: list[str] = []
    for node in all_nodes:
        if not node or node in seen:
            continue
        seen.add(node)
        if node.split(".")[0] != local_label:
            out.append(node)
    return out


def _main() -> int:
    """Standalone per-node entry point (``python3 host_sampler.py --log-dir D``).

    Runs on bare compute nodes with nothing but a system python3: this module
    deliberately imports only the standard library. Writes
    ``host_samples_<hostname>.jsonl`` so per-node files never collide on the
    shared log dir, and exits cleanly on SIGTERM/SIGINT (srun teardown).
    """
    import argparse
    import signal

    ap = argparse.ArgumentParser(description="Standalone /proc sampler for one node")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between samples (floored to 1.0)")
    ap.add_argument("--max-samples", type=int, default=0, help="stop after N samples (0 = until signaled)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[host_sampler %(levelname)s] %(message)s")
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    sampler = HostSampler(
        Path(args.log_dir),
        interval_seconds=args.interval,
        output_name=f"host_samples_{os.uname().nodename}.jsonl",
    )
    sampler.start(stop)
    while not stop.is_set():
        if args.max_samples and sampler.samples >= args.max_samples:
            stop.set()
            break
        if sampler._thread is not None and not sampler._thread.is_alive():
            # Sampler thread died (e.g. unwritable log dir): exit nonzero so the
            # failure is visible in the srun step output instead of idling forever.
            logger.error("sampler thread exited after %d samples; aborting", sampler.samples)
            sampler.stop()
            return 1
        stop.wait(0.5)
    sampler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
