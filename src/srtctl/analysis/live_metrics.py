# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-flight batch-metrics snapshotter.

The orchestrator runs on the head node and shares a filesystem with the
worker logs (``outputs/<jobid>/logs/*prefill_w*.out`` etc.), so we can
poll those logs from a background daemon thread without ssh / scp /
container hops. Every ``interval_seconds`` we incrementally re-parse the
freshly appended bytes (see :class:`.batch_log_parser.LogState`) and
overwrite ``batch_metrics.png`` in place.

Usage from the orchestrator:

>>> snap = try_start_snapshotter(log_dir=runtime.log_dir, stop_event=ev)
>>> try:
...     ...  # run benchmark
... finally:
...     if snap is not None:
...         snap.stop()

The plot view is intentionally simple: one line per worker file per
metric, no cluster-wide aggregation, no DP-rank scaling. Aggregated
views can be added in a follow-up without touching the snapshotter or
the parser.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from srtctl.analysis.batch_log_parser import LogState
from srtctl.analysis.batch_plot_matrix import default_batch_plot_title, render_batch_plot_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshotter
# ---------------------------------------------------------------------------


@dataclass
class _SnapshotterParams:
    log_dir: Path
    output_path: Path
    interval_seconds: int
    downsample: int


class LiveMetricsSnapshotter:
    """Daemon thread that periodically refreshes ``batch_metrics.png``.

    Best-effort: any exception from a tick is logged and swallowed so
    the snapshotter can never fail a benchmark run. Parsing is
    incremental (file byte offsets cached in :class:`LogState`), so
    long-running benchmarks stay cheap.
    """

    def __init__(
        self,
        log_dir: Path,
        interval_seconds: int = 60,
        downsample: int = 1,
        output_filename: str = "batch_metrics.png",
        title: str | None = None,
    ) -> None:
        self._params = _SnapshotterParams(
            log_dir=Path(log_dir),
            output_path=Path(log_dir) / output_filename,
            interval_seconds=max(5, int(interval_seconds)),
            downsample=max(1, int(downsample)),
        )
        self._title = title or default_batch_plot_title(log_dir)
        self._state = LogState(log_dir=self._params.log_dir)
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._tick_count = 0

    # ---- lifecycle ----------------------------------------------------

    def start(self, stop_event: threading.Event | None = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event = stop_event or threading.Event()
        self._thread = threading.Thread(target=self._loop, name="LiveMetricsSnapshotter", daemon=True)
        self._thread.start()
        logger.info(
            "Live batch-metrics snapshotter started: log_dir=%s interval=%ds output=%s",
            self._params.log_dir,
            self._params.interval_seconds,
            self._params.output_path,
        )

    def stop(self, timeout: float = 10.0) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Live metrics snapshotter did not exit within %.1fs", timeout)
        self._thread = None

    # ---- tick loop ----------------------------------------------------

    def _tick(self) -> None:
        try:
            self._state.refresh()
            if not self._state.has_data:
                return
            render_batch_plot_matrix(
                state=self._state,
                output_path=self._params.output_path,
                title=self._title,
                downsample=self._params.downsample,
            )
            self._tick_count += 1
            logger.debug(
                "Live metrics tick #%d wrote %s (prefill_files=%d, decode_files=%d)",
                self._tick_count,
                self._params.output_path,
                len(self._state.prefill_files),
                len(self._state.decode_files),
            )
        except Exception as e:  # never let snapshot failures affect the benchmark
            logger.warning("Live metrics snapshot failed: %s", e, exc_info=False)

    def _loop(self) -> None:
        assert self._stop_event is not None
        # Give workers a few seconds to print their first batch lines so
        # the very first scheduled PNG isn't empty. Don't return early
        # on stop: still try one final tick below so a benchmark that
        # crashed before the first interval still produces a snapshot.
        warmup = min(self._params.interval_seconds, 15)
        warmed_stopped = self._stop_event.wait(timeout=warmup)

        if not warmed_stopped:
            while not self._stop_event.is_set():
                self._tick()
                if self._stop_event.wait(timeout=self._params.interval_seconds):
                    break

        # Final tick: reflects the very end of the run (post-benchmark
        # / post-cleanup), or — if we were stopped during warmup — the
        # only snapshot we'll write at all.
        self._tick()
        logger.info("Live metrics snapshotter exited after %d ticks", self._tick_count)


# ---------------------------------------------------------------------------
# Orchestrator-facing helper
# ---------------------------------------------------------------------------


def try_start_snapshotter(
    log_dir: Path,
    stop_event: threading.Event,
) -> LiveMetricsSnapshotter | None:
    """Start a snapshotter if cluster config opts in, otherwise return ``None``.

    All failures (matplotlib missing, malformed config, thread refused
    to start) are logged and swallowed: live metrics is best-effort
    visualisation, never a hard dependency on the benchmark path. This
    is the single entry point :class:`BenchmarkStageMixin` uses, which
    keeps the mixin free of any analysis-package internals.
    """
    try:
        from srtctl.core.config import load_cluster_config
    except ImportError:  # pragma: no cover - defensive
        return None

    try:
        cluster_config = load_cluster_config()
    except Exception as e:
        logger.debug("Live metrics: failed to load cluster config: %s", e)
        return None

    telemetry = cluster_config.get("telemetry") if isinstance(cluster_config, dict) else None
    if not isinstance(telemetry, dict):
        return None

    cfg = telemetry.get("live_metrics")
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return None

    # matplotlib is a required srtctl dependency (see pyproject.toml), so we
    # do not need a runtime presence check here.

    try:
        snap = LiveMetricsSnapshotter(
            log_dir=log_dir,
            interval_seconds=int(cfg.get("interval_seconds", 60)),
            downsample=int(cfg.get("downsample", 1)),
        )
        snap.start(stop_event)
        return snap
    except Exception as e:
        logger.warning("Failed to start live metrics snapshotter: %s", e)
        return None
