# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared renderer for SGLang prefill/decode batch metrics.

The parser lives in :mod:`.batch_log_parser`; this module only turns a
populated :class:`~srtctl.analysis.batch_log_parser.LogState` into the
per-worker 7x2 PNG view used by both the live snapshotter and the
post-mortem ``plot_batch_metrics.py`` CLI.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.analysis.batch_log_parser import FileSeries, LogState

if TYPE_CHECKING:
    from collections.abc import Iterable


# Plot layout: (prefill_metric, decode_metric) row pairs.
# Use None to leave a cell empty. Only metrics listed here are plotted;
# all parsed metrics remain available in LogState for other consumers.
PLOT_ROWS: list[tuple[str | None, str | None]] = [
    ("input throughput (token/s)", "gen throughput (token/s)"),
    ("#new-seq", "#running-req"),
    ("#new-token", "#full token"),
    ("#cached-token", "full token usage"),
    ("#prealloc-req", "#prealloc-req"),
    ("#queue-req", "#queue-req"),
    ("#inflight-req", "#transfer-req"),
]


def default_batch_plot_title(log_dir: str | Path) -> str:
    """Return the run directory name for a logs directory."""
    path = Path(log_dir).resolve()
    return path.parent.name if path.name == "logs" else path.name


def _elapsed_seconds(stamps: list[datetime], origin: datetime) -> list[float]:
    return [(t - origin).total_seconds() for t in stamps]


@dataclass
class _PlotSeries:
    label: str
    timestamps: list[datetime]
    metrics: dict[str, list[float | None]]

    @property
    def empty(self) -> bool:
        return not self.timestamps


def _series_stem(path: Path) -> str:
    stem = path.name
    for ext in (".out", ".err"):
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return stem


def _dp_label(series: FileSeries, dp_rank: int) -> str:
    stem = re.sub(r"_DP\d+$", "", _series_stem(series.path))
    return f"{stem}_DP{dp_rank}"


def _plot_series_for_files(files: Iterable[FileSeries]) -> list[_PlotSeries]:
    """Split aggregated worker logs into one plot series per DP rank."""
    out: list[_PlotSeries] = []
    for series in files:
        if series.empty:
            continue

        dp_ranks = list(series.dp_ranks)
        if len(dp_ranks) < len(series.timestamps):
            dp_ranks.extend([None] * (len(series.timestamps) - len(dp_ranks)))
        else:
            dp_ranks = dp_ranks[: len(series.timestamps)]

        known_dps = sorted({dp for dp in dp_ranks if dp is not None})
        should_split = bool(known_dps) and (len(known_dps) > 1 or "_agg_" in series.path.name)
        if not should_split:
            out.append(_PlotSeries(label=series.label, timestamps=series.timestamps, metrics=series.metrics))
            continue

        for dp_rank in known_dps:
            idxs = [i for i, dp in enumerate(dp_ranks) if dp == dp_rank]
            metrics = {
                name: [values[i] if i < len(values) else None for i in idxs] for name, values in series.metrics.items()
            }
            out.append(
                _PlotSeries(
                    label=_dp_label(series, dp_rank),
                    timestamps=[series.timestamps[i] for i in idxs],
                    metrics=metrics,
                )
            )

    return out


def _moving_average(values: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return list(values)

    half = window // 2
    out: list[float | None] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = [v for v in values[lo:hi] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def _derived_input_throughput(series: _PlotSeries, smooth_window: int) -> list[float | None]:
    """Derive prefill input throughput from ``#new-token / dt``."""
    new_tokens = series.metrics.get("#new-token")
    if not new_tokens or len(series.timestamps) < 2:
        return []

    derived: list[float | None] = [None] * len(series.timestamps)
    for i in range(1, len(series.timestamps)):
        token_count = new_tokens[i] if i < len(new_tokens) else None
        if token_count is None:
            continue

        dt = (series.timestamps[i] - series.timestamps[i - 1]).total_seconds()
        if dt > 0:
            derived[i] = token_count / dt

    if smooth_window > 1:
        derived = _moving_average(derived, smooth_window)
    return derived


def _values_for_metric(series: _PlotSeries, metric: str, smooth_input_window: int) -> list[float | None]:
    values = list(series.metrics.get(metric, []))
    if metric != "input throughput (token/s)":
        return values

    derived = _derived_input_throughput(series, smooth_input_window)
    if not derived:
        return values

    merged: list[float | None] = []
    for i in range(max(len(values), len(derived))):
        explicit = values[i] if i < len(values) else None
        fallback = derived[i] if i < len(derived) else None
        merged.append(explicit if explicit is not None else fallback)
    return merged


def _global_origin(prefill: Iterable[_PlotSeries], decode: Iterable[_PlotSeries]) -> datetime | None:
    """Earliest timestamp seen across all worker files, or ``None`` if empty."""
    first_seen: datetime | None = None
    for s in (*prefill, *decode):
        if s.empty:
            continue
        candidate = s.timestamps[0]
        if first_seen is None or candidate < first_seen:
            first_seen = candidate
    return first_seen


def render_batch_plot_matrix(
    state: LogState,
    output_path: str | Path,
    title: str | None = None,
    downsample: int = 1,
    smooth_input_window: int = 8,
) -> bool:
    """Render per-worker time-series to a PNG.

    Returns ``True`` when a PNG was written and ``False`` when the state
    has no parsed batch rows yet.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pf_files = _plot_series_for_files(state.prefill_files.values())
    dc_files = _plot_series_for_files(state.decode_files.values())

    origin = _global_origin(pf_files, dc_files)
    if origin is None:
        return False

    n_rows = len(PLOT_ROWS)
    cmap = plt.cm.get_cmap("tab20", max(len(pf_files), len(dc_files), 1))
    colors = [cmap(i) for i in range(cmap.N)]

    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 3.0 * n_rows), squeeze=False)
    fig.suptitle(
        f"{title or default_batch_plot_title(state.log_dir)}\n"
        f"prefill: {len(pf_files)} workers · decode: {len(dc_files)} workers",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )

    def _draw_ax(ax: plt.Axes, metric: str | None, files: list[_PlotSeries], side: str) -> None:
        if metric is None:
            ax.set_visible(False)
            return

        drawn = False
        for idx, s in enumerate(files):
            vs = _values_for_metric(s, metric, smooth_input_window)
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
            ax.legend(
                fontsize=7,
                loc="upper right",
                ncol=max(1, len(files) // 8 + 1),
                framealpha=0.35,
                facecolor="white",
                edgecolor="0.7",
            )

    for row, (pf_metric, dc_metric) in enumerate(PLOT_ROWS):
        _draw_ax(axes[row][0], pf_metric, pf_files, "Prefill")
        _draw_ax(axes[row][1], dc_metric, dc_files, "Decode")

    output_path = Path(output_path)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / (output_path.name + ".tmp")
    fig.savefig(tmp, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)
    os.replace(tmp, output_path)
    return True
