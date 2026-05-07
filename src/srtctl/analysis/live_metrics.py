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
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.analysis.batch_log_parser import (
    FileSeries,
    LogState,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotting (lazy matplotlib import)
# ---------------------------------------------------------------------------


def _elapsed_seconds(stamps: list[datetime], origin: datetime) -> list[float]:
    return [(t - origin).total_seconds() for t in stamps]


def _global_origin(prefill: Iterable[FileSeries], decode: Iterable[FileSeries]) -> datetime | None:
    """Earliest timestamp seen across all worker files, or ``None`` if empty."""
    first_seen: datetime | None = None
    for s in (*prefill, *decode):
        if s.empty:
            continue
        candidate = s.timestamps[0]
        if first_seen is None or candidate < first_seen:
            first_seen = candidate
    return first_seen


# ---------------------------------------------------------------------------
# Plot layout: (prefill_metric, decode_metric) row pairs.
# Use None to leave a cell empty. Only metrics listed here are plotted;
# all parsed metrics remain available in LogState for other consumers.
# To add/remove/reorder rows, edit this list — no other code needs to
# change.
# ---------------------------------------------------------------------------
_PLOT_ROWS: list[tuple[str | None, str | None]] = [
    ("input throughput (token/s)", "gen throughput (token/s)"),
    ("#new-seq", "#running-req"),
    ("#new-token", "#full token"),
    ("#cached-token", "full token usage"),
    ("#prealloc-req", "#prealloc-req"),
    ("#queue-req", "#queue-req"),
    ("#inflight-req", "#transfer-req"),
]


def _render_png(
    state: LogState,
    output_path: Path,
    title: str,
    downsample: int,
) -> None:
    """Render per-worker time-series to a PNG.

    Layout is driven by ``_PLOT_ROWS``: each entry is a
    ``(prefill_metric, decode_metric)`` pair drawn on the same row so
    semantically related metrics sit side-by-side. Either cell can be
    ``None`` to leave it blank.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pf_files = [s for s in state.prefill_files.values() if not s.empty]
    dc_files = [s for s in state.decode_files.values() if not s.empty]

    origin = _global_origin(pf_files, dc_files)
    if origin is None:
        return

    n_rows = len(_PLOT_ROWS)
    # Use tab20 so up to 20 workers each get a distinct colour.
    cmap = plt.cm.get_cmap("tab20", max(len(pf_files), len(dc_files), 1))
    colors = [cmap(i) for i in range(cmap.N)]

    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 3.0 * n_rows), squeeze=False)
    fig.suptitle(
        f"{title}\nprefill: {len(pf_files)} workers · decode: {len(dc_files)} workers",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )

    def _draw_ax(ax: plt.Axes, metric: str | None, files: list[FileSeries], side: str) -> None:
        if metric is None:
            ax.set_visible(False)
            return

        drawn = False
        for idx, s in enumerate(files):
            vs = s.metrics.get(metric)
            if not vs:
                continue
            pairs = [(t, v) for t, v in zip(s.timestamps, vs, strict=False) if v is not None]
            if downsample > 1:
                pairs = pairs[::downsample]
            if not pairs:
                continue
            elapsed = _elapsed_seconds([p[0] for p in pairs], origin)
            values = [p[1] for p in pairs]
            ax.plot(elapsed, values, color=colors[idx % len(colors)], linewidth=0.9, alpha=0.8, label=s.label)
            drawn = True

        ax.set_title(f"{side}: {metric}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Elapsed (s)", fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if not drawn:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="grey", fontsize=9)
        elif files:
            ax.legend(fontsize=7, loc="upper right", ncol=max(1, len(files) // 8 + 1))

    for row, (pf_metric, dc_metric) in enumerate(_PLOT_ROWS):
        _draw_ax(axes[row][0], pf_metric, pf_files, "Prefill")
        _draw_ax(axes[row][1], dc_metric, dc_files, "Decode")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish write: render to a sibling tmp file then rename, so
    # readers (e.g. an image viewer with auto-reload) never see a
    # half-written PNG. Append ``.tmp`` to the full filename rather than
    # replacing the suffix (which would confuse matplotlib's format
    # inference).
    tmp = output_path.parent / (output_path.name + ".tmp")
    fig.savefig(tmp, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    os.replace(tmp, output_path)


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
        self._title = title or f"{Path(log_dir).resolve().parent.name} batch metrics"
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
            _render_png(
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

    cfg = (cluster_config or {}).get("telemetry", {}).get("live_metrics") if cluster_config else None
    if not cfg or not cfg.get("enabled"):
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
