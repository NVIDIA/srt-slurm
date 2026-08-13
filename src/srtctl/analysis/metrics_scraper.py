# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-job Prometheus scraper -> ``raw_prometheus.jsonl`` (RAW capture).

Why this exists
---------------
Worker and frontend ``/metrics`` endpoints are served by the live processes. When
the SLURM job ends they disappear and the data is gone forever -- there is no
after-the-fact scrape. Anything that wants a time series of KV-cache occupancy,
router queue depth or frontend event-loop pressure has to capture it *during*
the run.

AIPerf already scrapes on its own ``--slice-duration`` cadence, but only for the
window it is actively benchmarking, and its aggregation is opinionated. This
scraper is the raw, unopinionated complement: it starts with the benchmark stage
and appends the response body **verbatim**, so a parser fix never requires
re-running the job.

Design rule (mirrors ``src/capture/`` in the perf-dashboard repo): *capture emits
RAW, the processor parses*. One JSON line per ``(sweep, endpoint)``::

    {"timestamp_ns": 1786194627868177662,
     "endpoint_url": "http://node1:8081/metrics",
     "role": "prefill",            # frontend | prefill | decode
     "worker_id": "node1",         # host for workers, null for the frontend
     "text": "<raw Prometheus exposition text, UNPARSED>"}

``timestamp_ns`` is ``time.time_ns()`` (wall-epoch) so it aligns with AIPerf
records and Dynamo spans on the same clock.

Best-effort by construction: every failure path is logged and swallowed. A
scrape that times out costs one missing line, never a failed benchmark.

Stdlib only -- runs on the orchestrator's host python, which reaches every
worker node over the job's network (same URLs srtctl already hands to AIPerf).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# A slow worker under load can take a while to render its exposition text; keep
# the timeout well under the sweep interval so one stuck endpoint cannot stall
# the whole sweep past its next tick.
_HTTP_TIMEOUT_SECONDS = 5.0

# Log at most one warning per endpoint per this many consecutive failures, so a
# worker that dies mid-run does not flood the sweep log with identical errors.
_WARN_EVERY = 20


@dataclass(frozen=True)
class ScrapeTarget:
    """One endpoint to poll."""

    url: str
    role: str  # "frontend" | "prefill" | "decode"
    worker_id: str | None


class RawMetricsScraper:
    """Daemon thread appending RAW ``/metrics`` bodies to a JSONL file."""

    def __init__(
        self,
        log_dir: Path,
        targets: list[ScrapeTarget],
        interval_seconds: float = 3.0,
        output_name: str = "raw_prometheus.jsonl",
    ) -> None:
        self.output_path = Path(log_dir) / output_name
        self.targets = list(targets)
        self.interval_seconds = max(0.5, float(interval_seconds))
        self._thread: threading.Thread | None = None
        # Own stop flag *in addition to* the caller's shared stop_event: the
        # benchmark stage's stop_event is only set on abort, not on normal
        # completion, so without this the scraper would outlive the client and
        # keep polling endpoints that are about to be torn down.
        self._own_stop = threading.Event()
        self.sweeps = 0
        self.lines_written = 0
        self._fail_counts: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self, stop_event: threading.Event) -> None:
        if not self.targets:
            logger.warning("Raw metrics scraper: no targets, not starting")
            return
        self._thread = threading.Thread(
            target=self._loop,
            args=(stop_event,),
            name="RawMetricsScraper",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Raw metrics scraper started: %d endpoint(s) every %.1fs -> %s",
            len(self.targets),
            self.interval_seconds,
            self.output_path,
        )
        for t in self.targets:
            logger.info("  scrape target %-8s %s", t.role, t.url)

    def stop(self, timeout: float = 15.0) -> None:
        """Signal the loop and wait for the final flush."""
        self._own_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info(
            "Raw metrics scraper stopped: %d sweeps, %d lines -> %s",
            self.sweeps,
            self.lines_written,
            self.output_path,
        )

    # -- internals ---------------------------------------------------------

    def _fetch(self, target: ScrapeTarget) -> str | None:
        try:
            with urllib.request.urlopen(target.url, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
                if resp.status != 200:
                    self._note_failure(target, f"HTTP {resp.status}")
                    return None
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, ValueError) as e:
            self._note_failure(target, repr(e))
            return None

    def _note_failure(self, target: ScrapeTarget, reason: str) -> None:
        n = self._fail_counts.get(target.url, 0) + 1
        self._fail_counts[target.url] = n
        if n == 1 or n % _WARN_EVERY == 0:
            logger.warning(
                "Raw metrics scraper: %s unreachable (%d consecutive): %s",
                target.url,
                n,
                reason,
            )

    def _sweep(self, fh) -> None:
        for target in self.targets:
            text = self._fetch(target)
            if text is None:
                continue
            self._fail_counts[target.url] = 0
            record = {
                "timestamp_ns": time.time_ns(),
                "endpoint_url": target.url,
                "role": target.role,
                "worker_id": target.worker_id,
                "text": text,
            }
            fh.write(json.dumps(record) + "\n")
            self.lines_written += 1
        # Flush per sweep, not per line: one fsync-ish cost per interval, and
        # the file stays readable from another node while the job runs (useful
        # for live progress checks over lustre).
        fh.flush()

    def _loop(self, stop_event: threading.Event) -> None:
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "a", encoding="utf-8") as fh:
                while not stop_event.is_set() and not self._own_stop.is_set():
                    started = time.monotonic()
                    try:
                        self._sweep(fh)
                        self.sweeps += 1
                    except Exception as e:  # never kill the thread mid-run  # noqa: BLE001
                        logger.warning("Raw metrics scraper sweep failed: %s", e)
                    # Drift-free pacing: wait the remainder of the interval, so a
                    # slow sweep shortens the gap instead of shifting every
                    # subsequent tick later. Waiting on the own-stop event (not
                    # sleeping) makes shutdown immediate rather than up to one
                    # interval late.
                    elapsed = time.monotonic() - started
                    self._own_stop.wait(max(0.0, self.interval_seconds - elapsed))
        except Exception as e:  # capture is best-effort, never fatal  # noqa: BLE001
            logger.warning("Raw metrics scraper aborted: %s", e)


def try_start_raw_scraper(
    log_dir: Path,
    targets: list[ScrapeTarget],
    observability,
    stop_event: threading.Event,
) -> RawMetricsScraper | None:
    """Start the scraper if ``observability`` opts in, else return ``None``.

    Single entry point for :class:`BenchmarkStageMixin`, so the mixin stays free
    of analysis-package internals -- same contract as
    :func:`srtctl.analysis.live_metrics.try_start_snapshotter`. All failures are
    logged and swallowed: RAW metric capture is best-effort observability, never
    a hard dependency on the benchmark path.
    """
    try:
        if observability is None or not getattr(observability, "scraper_enabled", False):
            return None
        scraper = RawMetricsScraper(
            log_dir=log_dir,
            targets=targets,
            interval_seconds=getattr(observability, "scrape_interval_seconds", 3.0),
            output_name=getattr(observability, "scrape_output", "raw_prometheus.jsonl"),
        )
        scraper.start(stop_event)
        return scraper
    except Exception as e:  # capture is best-effort, never fatal  # noqa: BLE001
        logger.warning("Raw metrics scraper: failed to start (continuing): %s", e)
        return None
