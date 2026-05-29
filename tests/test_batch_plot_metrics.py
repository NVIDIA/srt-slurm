# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for post-mortem batch metrics plotting."""

import importlib.util
from pathlib import Path

from srtctl.analysis.batch_log_parser import LogState
from srtctl.analysis.batch_plot_matrix import _plot_series_for_files, _values_for_metric


def _load_plot_batch_metrics_module():
    script_path = Path(__file__).parent.parent / "src/srtctl/benchmarks/scripts/plot_batch_metrics.py"
    spec = importlib.util.spec_from_file_location("plot_batch_metrics", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_batch_metrics_cli_uses_shared_live_renderer(monkeypatch, tmp_path):
    """The offline CLI should parse once and render through batch_plot_matrix."""
    module = _load_plot_batch_metrics_module()

    logs_dir = tmp_path / "453992" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "bia0004_agg_w0_DP0.out").write_text(
        "\n".join(
            [
                "[2026-05-09 12:00:00 DP0 TP0 EP0] Prefill batch, #new-seq: 2, #new-token: 8192, "
                "#cached-token: 0, full token usage: 0.10, #running-req: 0, #queue-req: 0, "
                "#prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 1000.0",
                "[2026-05-09 12:00:00 DP0 TP0 EP0] Decode batch, #running-req: 4, "
                "#full token: 10000, full token usage: 0.20, #prealloc-req: 1, #transfer-req: 0, "
                "#retracted-req: 0, gen throughput (token/s): 500.0, #queue-req: 0",
            ]
        )
    )

    calls = {}

    def fake_render_batch_plot_matrix(state, output_path, title, downsample, smooth_input_window):
        calls["state"] = state
        calls["output_path"] = Path(output_path)
        calls["title"] = title
        calls["downsample"] = downsample
        calls["smooth_input_window"] = smooth_input_window
        calls["output_path"].write_text("fake png")
        return True

    monkeypatch.setattr(module, "render_batch_plot_matrix", fake_render_batch_plot_matrix)

    output_path = tmp_path / "batch_metrics.png"
    assert module.process_single_run(
        str(logs_dir), downsample_factor=7, output_path=str(output_path), smooth_input_window=3
    )

    assert output_path.read_text() == "fake png"
    assert calls["title"] == "453992"
    assert calls["downsample"] == 7
    assert calls["smooth_input_window"] == 3
    assert len(calls["state"].prefill_files) == 1
    assert len(calls["state"].decode_files) == 1
    assert calls["state"].has_data


def test_batch_log_parser_keeps_millisecond_timestamps_and_dp_ranks(tmp_path):
    """Aggregated logs need DP rank metadata and sub-second deltas."""
    logs_dir = tmp_path / "461673" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "bia0003_agg_w0.out").write_text(
        "\n".join(
            [
                "[2026-05-10 12:00:11.896 DP1 TP1 EP1] Prefill batch [1], "
                "#new-seq: 1, #new-token: 2048, #cached-token: 0, full token usage: 0.00, "
                "#running-req: 0, #queue-req: 0",
                "[2026-05-10 12:00:11.901 DP0 TP0 EP0] Prefill batch [1], "
                "#new-seq: 1, #new-token: 2048, #cached-token: 0, full token usage: 0.00, "
                "#running-req: 0, #queue-req: 0",
            ]
        )
    )

    state = LogState(log_dir=logs_dir)
    assert state.refresh() == (2, 0)

    series = next(iter(state.prefill_files.values()))
    assert series.dp_ranks == [1, 0]
    assert series.timestamps[0].microsecond == 896000
    assert series.timestamps[1].microsecond == 901000


def test_batch_log_parser_accepts_legacy_token_field_names(tmp_path):
    """Current production SGLang logs may still use #token/token usage."""
    logs_dir = tmp_path / "461674" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "nvl72d068-T01_prefill_w0.out").write_text(
        "[2026-05-19 00:32:50 DP0 TP0] Prefill batch, #new-seq: 1, #new-token: 64, "
        "#cached-token: 0, token usage: 0.10, #running-req: 0, #queue-req: 0, "
        "#pending-token: 0, #prealloc-req: 0, #inflight-req: 1, cuda graph: False, "
        "input throughput (token/s): 0.16\n"
    )
    (logs_dir / "nvl72d068-T04_decode_w0.out").write_text(
        "[2026-05-19 00:34:42 DP0 TP0 EP0] Decode batch [2], #running-req: 1, "
        "#token: 64, token usage: 0.20, pre-allocated usage: 0.00, #prealloc-req: 0, "
        "#transfer-req: 0, #retracted-req: 0, cuda graph: True, gen throughput (token/s): 58.87, "
        "#queue-req: 0\n"
    )

    state = LogState(log_dir=logs_dir)
    assert state.refresh() == (1, 1)

    prefill = next(iter(state.prefill_files.values()))
    decode = next(iter(state.decode_files.values()))
    assert prefill.metrics["full token usage"] == [0.10]
    assert decode.metrics["#full token"] == [64.0]
    assert decode.metrics["full token usage"] == [0.20]


def test_renderer_splits_agg_logs_by_dp_and_derives_input_throughput(tmp_path):
    """The plot matrix should expose one line per DP rank in a single agg log."""
    logs_dir = tmp_path / "461673" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "bia0003_agg_w0.out").write_text(
        "\n".join(
            [
                "[2026-05-10 12:00:00.000 DP0 TP0 EP0] Prefill batch [1], "
                "#new-seq: 1, #new-token: 100, #cached-token: 0, full token usage: 0.00, "
                "#running-req: 0, #queue-req: 0",
                "[2026-05-10 12:00:00.500 DP1 TP1 EP1] Prefill batch [1], "
                "#new-seq: 1, #new-token: 50, #cached-token: 0, full token usage: 0.00, "
                "#running-req: 0, #queue-req: 0",
                "[2026-05-10 12:00:01.000 DP0 TP0 EP0] Prefill batch [2], "
                "#new-seq: 1, #new-token: 200, #cached-token: 0, full token usage: 0.00, "
                "#running-req: 0, #queue-req: 0",
                "[2026-05-10 12:00:01.000 DP1 TP1 EP1] Prefill batch [2], "
                "#new-seq: 1, #new-token: 100, #cached-token: 0, full token usage: 0.00, "
                "#running-req: 0, #queue-req: 0",
            ]
        )
    )

    state = LogState(log_dir=logs_dir)
    assert state.refresh() == (4, 0)

    plot_series = _plot_series_for_files(state.prefill_files.values())
    assert [s.label for s in plot_series] == ["bia0003_agg_w0_DP0", "bia0003_agg_w0_DP1"]

    derived = {s.label: _values_for_metric(s, "input throughput (token/s)", smooth_input_window=1) for s in plot_series}
    assert derived["bia0003_agg_w0_DP0"] == [None, 200.0]
    assert derived["bia0003_agg_w0_DP1"] == [None, 200.0]
