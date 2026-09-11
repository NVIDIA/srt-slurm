# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental, per-case power/energy emission during a live benchmark.

Best-effort by design and strictly additive: the terminal
``power_energy_report.json`` remains authoritative and unchanged. Nothing here
may affect the job exit code or block the sweep.

The design is a periodic idempotent rescan -- each poll re-runs the same
``discover_run`` / ``build_concurrency_report`` code the terminal report uses,
and writes out any case that is complete and not yet emitted. That makes it
benchmark-agnostic by construction: whatever the terminal report can discover,
this discovers.

The "idempotent" guarantee above is per-process only: the co-located
``power_energy_c<N>.json`` file is overwrite-idempotent, but
``power_energy_report.jsonl`` is append-only against an in-memory set, so a
job requeued into the same log dir will append duplicate concurrency rows to
the index on restart. Consumers of the index should dedupe by ``concurrency``,
keeping the row with the latest ``emitted_at_unix``.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from pathlib import Path

from srtctl.analysis.power_energy_report import (
    ConcurrencyReport,
    ConcurrencyWindow,
    CpuSamples,
    GpuSamples,
    PowerReportError,
    RunPaths,
    aiperf_window,
    build_concurrency_report,
    discover_run,
    load_cpu_samples_from,
    load_gpu_roles,
    load_gpu_samples_from,
    report_to_dict,
    sa_bench_window,
)
from srtctl.core.power.contract import MAX_SAMPLE_GAP_SECONDS, atomic_write_json

logger = logging.getLogger(__name__)


def read_csv_tolerantly(path: Path) -> io.StringIO:
    """Read a CSV that another thread may be appending to, dropping a torn tail.

    The power collectors flush ``samples.csv`` every cycle, so a read taken
    mid-run can land between a row's bytes. Anything after the last newline is
    discarded; a file with no newline at all yields an empty stream, which the
    sample loaders turn into empty series. ``IncrementalPowerEmitter`` treats a
    discovered-but-empty sample series as not-ready-yet (see its emptiness
    guard) rather than relying on the integrator to reject it, since an empty
    series integrates to zero rather than raising.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.endswith("\n"):
        cut = text.rfind("\n")
        text = text[: cut + 1] if cut >= 0 else ""
    return io.StringIO(text)


INDEX_FILENAME = "power_energy_report.jsonl"
CO_LOCATED_FILENAME_TEMPLATE = "power_energy_c{concurrency}.json"
INCREMENTAL_SCHEMA_VERSION = 1

# Every exception a not-yet-complete artifact can plausibly raise. A case that
# trips any of these is simply retried on the next poll.
_NOT_READY = (
    PowerReportError,
    json.JSONDecodeError,
    OSError,
    KeyError,
    ValueError,
    IndexError,
    AttributeError,
    TypeError,
)


def _newest_sample_timestamp(cpu_samples: CpuSamples | None, gpu_samples: GpuSamples | None) -> float | None:
    """The newest ``timestamp_unix`` across every loaded series, or None if nothing was loaded.

    Each series in ``CpuSamples``/``GpuSamples`` is already sorted ascending
    by ``_sorted_series``, so the newest sample per series is its last point.
    """
    newest: float | None = None

    def scan(series_map: dict) -> None:
        nonlocal newest
        for times, _watts in series_map.values():
            if len(times) == 0:
                continue
            candidate = float(times[-1])
            if newest is None or candidate > newest:
                newest = candidate

    if cpu_samples is not None:
        scan(cpu_samples.per_socket)
        scan(cpu_samples.per_node)
    if gpu_samples is not None:
        scan(gpu_samples.per_device)
        scan(gpu_samples.per_node)
    return newest


class IncrementalPowerEmitter:
    """Writes each benchmark case's energy result as soon as that case completes.

    Idempotent: ``poll()`` emits a given concurrency exactly once. Never
    raises -- an incomplete artifact, an uncovered window or an unreadable file
    all mean "not ready yet, retry next tick".
    """

    def __init__(self, log_dir: Path):
        self._log_dir = log_dir
        # Keyed on source path (not concurrency alone): discover_run's sources
        # come from an rglob, so two sources could in principle share a
        # concurrency (nested run dirs, an archived run copied under log_dir,
        # a future multi-ISL layout). Keying on concurrency would silently
        # drop the second one.
        self._emitted: set[Path] = set()
        # Cases that can never become integrable -- e.g. the collector died
        # before the window closed, so the end-gap never shrinks. Tracked the
        # same way as ``_emitted`` so they are excluded from ``pending`` on
        # every later tick instead of re-parsing the full samples.csv forever.
        self._dead: set[Path] = set()

    @property
    def index_path(self) -> Path:
        """Path to the append-only ``power_energy_report.jsonl`` index.

        Append-only against an in-memory set, so it is idempotent only within
        a single process's lifetime -- a requeued job writing into the same
        log dir appends duplicate concurrency rows. Consumers should dedupe by
        ``concurrency``, last-wins by ``emitted_at_unix``.
        """
        return self._log_dir / INDEX_FILENAME

    def poll(self) -> tuple[int, ...]:
        """Emit every case that is complete and not yet written. Returns what it emitted."""
        try:
            paths = discover_run(self._log_dir)
        except _NOT_READY as exc:
            logger.debug("Incremental power: run not discoverable yet: %s", exc)
            return ()

        pending = sorted((c, s) for c, s in paths.concurrency_sources if s not in self._emitted and s not in self._dead)
        if not pending:
            return ()

        try:
            cpu_samples, gpu_samples = self._load_samples(paths)
        except _NOT_READY as exc:
            logger.debug("Incremental power: samples not readable yet: %s", exc)
            return ()
        newest_sample_unix = _newest_sample_timestamp(cpu_samples, gpu_samples)

        emitted: list[int] = []
        for concurrency, source in pending:
            window = self._build_window(concurrency, source)
            if window is None:
                continue  # window itself not ready yet; cannot judge dead-ness without it

            report = self._try_build_report(window, cpu_samples, gpu_samples)
            if report is None:
                if self._is_dead(window, newest_sample_unix):
                    self._dead.add(source)
                    logger.warning(
                        "Incremental power: concurrency %d will never be covered by samples "
                        "(newest sample at %.3f is past window end %.3f + %.1fs gap tolerance); "
                        "giving up on this case",
                        concurrency,
                        newest_sample_unix,
                        window.end_unix,
                        MAX_SAMPLE_GAP_SECONDS,
                    )
                continue

            try:
                self._write(concurrency, source, report)
            except Exception:
                logger.warning("Incremental power: failed writing concurrency %d", concurrency, exc_info=True)
                continue
            self._emitted.add(source)
            emitted.append(concurrency)

        if emitted:
            logger.info("Incremental power: emitted concurrency point(s) %s", emitted)
        return tuple(emitted)

    @staticmethod
    def _is_dead(window: ConcurrencyWindow, newest_sample_unix: float | None) -> bool:
        """A case is permanently unready once samples have grown past its window with no coverage.

        Samples only ever grow forward in time, so once the newest sample seen
        is already more than ``MAX_SAMPLE_GAP_SECONDS`` past the window end,
        no future poll can shrink that gap -- this case can never become
        integrable. Only decidable once samples have actually been loaded
        (``newest_sample_unix is not None``); a run with no samples loaded yet
        is merely not-ready, not dead.
        """
        if newest_sample_unix is None:
            return False
        return newest_sample_unix > window.end_unix + MAX_SAMPLE_GAP_SECONDS

    def _load_samples(self, paths: RunPaths) -> tuple[CpuSamples | None, GpuSamples | None]:
        """Load whichever sample series discovery found.

        A source that discovery never found (``None``) is legitimate -- a
        CPU-only or GPU-only run -- and is left as ``None``. But a source that
        *was* found and yet loaded with no series at all (e.g. the collector
        never started, or died before writing any rows) must NOT be treated as
        an empty-but-valid series: ``build_concurrency_report`` integrates
        empty series to 0 joules rather than raising, so we have to catch this
        here rather than relying on it to reject the case.
        """
        cpu_samples = None
        if paths.cpu_samples_csv is not None:
            cpu_samples = load_cpu_samples_from(read_csv_tolerantly(paths.cpu_samples_csv))
            if not cpu_samples.per_socket and not cpu_samples.per_node:
                raise PowerReportError(f"no CPU power samples yet in {paths.cpu_samples_csv}")

        gpu_samples = None
        if paths.gpu_samples_csv is not None:
            roles = load_gpu_roles(paths.gpu_manifest) if paths.gpu_manifest else None
            gpu_samples = load_gpu_samples_from(read_csv_tolerantly(paths.gpu_samples_csv), roles)
            if not gpu_samples.per_device and not gpu_samples.per_node:
                raise PowerReportError(f"no GPU power samples yet in {paths.gpu_samples_csv}")

        return cpu_samples, gpu_samples

    def _build_window(self, concurrency: int, source: Path) -> ConcurrencyWindow | None:
        """Build the case's window, or None if its own artifact is not ready yet.

        Split out from report building so a case whose window we *can* already
        read (start/end/token fields present) but whose samples don't cover it
        yet can still be judged for dead-ness against ``window.end_unix`` --
        which requires the window even when the report itself is not ready.
        """
        try:
            if source.name == "profile_export.jsonl":
                return aiperf_window(concurrency, source)
            return sa_bench_window(concurrency, source)
        except _NOT_READY as exc:
            logger.debug("Incremental power: concurrency %d window not ready: %s", concurrency, exc)
            return None

    def _try_build_report(
        self,
        window: ConcurrencyWindow,
        cpu_samples: CpuSamples | None,
        gpu_samples: GpuSamples | None,
    ) -> ConcurrencyReport | None:
        """Build the case's report, or None if it is not ready yet.

        ``build_concurrency_report`` integrates every socket, device and node in
        the window, and ``windowed_energy`` refuses any window whose nearest
        sample is more than MAX_SAMPLE_GAP_SECONDS from a boundary. So a case
        that is not yet fully bracketed by samples raises here and is withheld
        -- which is exactly the completeness guarantee we want.
        """
        try:
            return build_concurrency_report(window, cpu_samples, gpu_samples)
        except _NOT_READY as exc:
            logger.debug("Incremental power: concurrency %d not ready: %s", window.concurrency, exc)
            return None

    def _write(self, concurrency: int, source: Path, report: ConcurrencyReport) -> None:
        payload = {
            **report_to_dict(report),
            "schema_version": INCREMENTAL_SCHEMA_VERSION,
            "emitted_at_unix": time.time(),
        }
        co_located = source.parent / CO_LOCATED_FILENAME_TEMPLATE.format(concurrency=concurrency)
        atomic_write_json(co_located, payload)
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
            handle.flush()


# Deliberately much slower than the power collectors' own sampling cadence: the
# only deadline this must beat is "before the job dies", not any human-facing
# latency requirement.
DEFAULT_TICK_SECONDS = 30.0
DEFAULT_JOIN_TIMEOUT_SECONDS = 10.0


class IncrementalPowerWatcher:
    """Daemon thread that drives ``IncrementalPowerEmitter.poll()`` on a fixed tick.

    Lifecycle mirrors ``CpuPowerCollector``: nothing here raises into the
    caller, and a wedged thread cannot hang the job -- teardown joins with a
    timeout and moves on.
    """

    def __init__(
        self,
        emitter: IncrementalPowerEmitter,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
        join_timeout_seconds: float = DEFAULT_JOIN_TIMEOUT_SECONDS,
    ):
        self._emitter = emitter
        self._tick_seconds = tick_seconds
        self._join_timeout_seconds = join_timeout_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="IncrementalPowerWatcher", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self._tick_seconds)

    def _poll_once(self) -> None:
        try:
            self._emitter.poll()
        except Exception:
            logger.exception("Incremental power poll failed; continuing")

    def stop_and_finalize(self) -> None:
        """Stop the thread, then run one final poll against now-closed sample files.

        The final pass is what guarantees the index file is complete on a normal
        exit, not only after a crash.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._join_timeout_seconds)
            if thread.is_alive():
                logger.warning("Incremental power watcher did not stop within %.1fs", self._join_timeout_seconds)
                return  # a live thread still owns the emitter; do not race it
        self._poll_once()
