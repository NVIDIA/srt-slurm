# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vendored component perf dashboard (``src/ingest`` + ``src/visualization``).

These cover the seam between srt-slurm's capture layer and the vendored ingest /
renderer: the artifact *layout* srt-slurm actually writes (``raw_prometheus.jsonl``
at the log-dir root, AIPerf's export under ``artifacts/<run>/``, worker and frontend
logs named ``*.out``) has to be what the vendored defaults look for. The vendored
internals themselves are covered upstream; what breaks on a re-sync is the wiring.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# A fixed wall-epoch anchor. The renderer places everything on a run-relative axis,
# so the absolute value is irrelevant as long as every artifact shares it.
T0 = 1786194627868177662
XIDS = [f"xid{i}" for i in range(6)]


# ---------------------------------------------------------------------------
# fixtures: a miniature srt-slurm run directory
# ---------------------------------------------------------------------------


def _write_raw_prometheus(run_dir: Path, sweeps: int = 4) -> Path:
    """The RAW capture :mod:`srtctl.analysis.metrics_scraper` writes during a job.

    One line per (sweep, endpoint), body verbatim, roles drawn from the same
    ``frontend`` / ``prefill`` / ``decode`` vocabulary the scraper emits.
    """
    path = run_dir / "raw_prometheus.jsonl"
    with path.open("w") as f:
        for i in range(sweeps):
            ts = T0 + i * 1_000_000_000
            f.write(
                json.dumps(
                    {
                        "timestamp_ns": ts,
                        "endpoint_url": "http://head:8000/metrics",
                        "role": "frontend",
                        "worker_id": None,
                        "text": (
                            f'dynamo_frontend_requests_total{{worker_type="frontend"}} {10 + i}\n'
                            f"dynamo_frontend_queued_requests {i}\n"
                        ),
                    }
                )
                + "\n"
            )
            for role, host, used in (("prefill", "node1", 100 + i), ("decode", "node2", 200 + i)):
                f.write(
                    json.dumps(
                        {
                            "timestamp_ns": ts,
                            "endpoint_url": f"http://{host}:8081/metrics",
                            "role": role,
                            "worker_id": host,
                            "text": (
                                f'trtllm_kv_cache_used_blocks{{model_name="m"}} {used}\n'
                                f'trtllm_kv_cache_free_blocks{{model_name="m"}} {1000 - used}\n'
                                f'trtllm_kv_cache_max_blocks{{model_name="m"}} 1000\n'
                            ),
                        }
                    )
                    + "\n"
                )
    return path


def _write_aiperf_export(run_dir: Path) -> Path:
    """AIPerf's own per-request export, at the path srt-slurm's bench.sh puts it.

    ``<log_dir>/artifacts/<model>_<workload>_<timestamp>/profile_export.jsonl`` --
    the nesting is what the ingest default glob has to reach through.
    """
    art = run_dir / "artifacts" / "Qwen3-32B_conversation_20260820_101500"
    art.mkdir(parents=True)
    path = art / "profile_export.jsonl"
    with path.open("w") as f:
        for i, xid in enumerate(XIDS):
            start = T0 + i * 100_000_000
            f.write(
                json.dumps(
                    {
                        "metadata": {
                            "x_request_id": xid,
                            "request_start_ns": start,
                            "request_end_ns": start + 1_200_000_000,
                        },
                        "metrics": {
                            "time_to_first_token": {"value": 120.0 + i},
                            "inter_token_latency": {"value": 8.0},
                            "input_sequence_length": {"value": 1024},
                            "output_sequence_length": {"value": 128},
                            "request_latency": {"value": 1200.0 + i},
                        },
                    }
                )
                + "\n"
            )
    return path


def _write_engine_configs(run_dir: Path) -> None:
    """The resolved engine configs srt-slurm's TRTLLM backend dumps per mode.

    Values are the real ones from AgentX run 2739690 -- decode max_batch_size is 1,
    which is what makes the renderer's 256 default so badly wrong.
    """
    (run_dir / "trtllm_config_prefill.yaml").write_text("max_batch_size: 128\nmax_num_tokens: 4096\n")
    (run_dir / "trtllm_config_decode.yaml").write_text("max_batch_size: 1\nmax_num_tokens: 4\n")


def _write_frontend_log(run_dir: Path) -> Path:
    """A Dynamo frontend log in the raw-container-stdout flavour srtctl produces.

    Named the way ``DynamoFrontend.start_frontends`` names it, so the docs' glob
    (``*_frontend_*.out``) is exercised rather than an invented filename.
    """
    path = run_dir / "node0_frontend_0.out"
    lines = []
    for i, xid in enumerate(XIDS):
        rid = f"req-{i}"
        recv_s, done_s = 10 + i, 11 + i
        lines.append(
            f"2026-08-20T10:00:{recv_s:02d}.100000Z  INFO dynamo_llm::http::service::metrics: "
            f'request received request_id="{rid}" x_request_id="{xid}" model="m"'
        )
        lines.append(
            f"2026-08-20T10:00:{done_s:02d}.400000Z  INFO dynamo_llm::http::service::metrics: "
            f'request completed request_id="{rid}" x_request_id="{xid}" status="success" '
            f"ttft_ms={120.0 + i} elapsed_ms={1300.0 + i} input_tokens=1024 output_tokens=128 avg_itl_ms=8.0"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A miniature srt-slurm ``<job>/logs/`` directory with all three capture legs."""
    d = tmp_path / "logs"
    d.mkdir()
    _write_raw_prometheus(d)
    _write_aiperf_export(d)
    _write_frontend_log(d)
    _write_engine_configs(d)
    return d


def _run_ingest(run_dir: Path, bundle: Path, *extra: str) -> None:
    from src.ingest.ingest import main

    rc = main(
        [
            "--run-dir", str(run_dir),
            "--out", str(bundle),
            "--traces", "none",  # no SPAN_CLOSED lines in the fixture logs
            "--name", "smoke-run",
            "--worker", "prefill=dep:4:1",
            "--worker", "decode=tep:4:1",
            *extra,
        ]
    )
    assert rc == 0


def _render(bundle: Path | None, out_html: Path, *extra: str) -> subprocess.CompletedProcess:
    argv = [sys.executable, "-m", "src.visualization.build_dynamo_bench_dash"]
    if bundle is not None:
        argv.append(str(bundle))
    argv += [str(out_html), *extra]
    return subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True)


def _tabs(html: str) -> dict:
    """Pull the renderer's own tab-availability map back out of the page."""
    marker = '"tabs":'
    i = html.index(marker) + len(marker)
    return json.loads(html[i : html.index("}", i) + 1])


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


class TestProcessorRegistry:
    def test_vendored_processors_resolve(self):
        from src.ingest import get_processor

        assert callable(get_processor("client", "aiperf"))
        assert callable(get_processor("traces", "spanlog"))
        assert callable(get_processor("metrics", "prometheus"))

    @pytest.mark.parametrize(("axis", "name"), [("client", "agentperf"), ("traces", "tempo")])
    def test_unvendored_processors_fail_loudly(self, axis: str, name: str):
        """Upstream registers these; we deliberately did not vendor them.

        The failure has to name the valid options -- a bare KeyError would send the
        next reader looking for a module that was never copied.
        """
        from src.ingest import get_processor

        with pytest.raises(KeyError, match="valid"):
            get_processor(axis, name)


# ---------------------------------------------------------------------------
# L2: capture -> intermediate schemas
# ---------------------------------------------------------------------------


class TestMetricsPrometheus:
    def test_raw_capture_becomes_schema_2(self, tmp_path: Path):
        from src.ingest.metrics_prometheus import process

        raw = _write_raw_prometheus(tmp_path, sweeps=3)
        out = tmp_path / "server_metrics_export.jsonl"
        assert process(str(raw), str(out)) == 3

        lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        assert [ln["timestamp_ns"] for ln in lines] == sorted(ln["timestamp_ns"] for ln in lines)
        assert len(lines) == 3, "the three endpoints of one sweep merge into one line"

    def test_worker_labels_are_injected(self, tmp_path: Path):
        """TRT-LLM labels its KV gauges with only {model_name}; the KV panels pair
        them by worker_id and split by dynamo_component. Without this injection the
        Engine tab silently renders nothing."""
        from src.ingest.metrics_prometheus import process

        raw = _write_raw_prometheus(tmp_path, sweeps=1)
        out = tmp_path / "server_metrics_export.jsonl"
        process(str(raw), str(out))

        entries = json.loads(out.read_text().splitlines()[0])["metrics"]["trtllm_kv_cache_used_blocks"]
        by_worker = {e["labels"]["worker_id"]: e["labels"]["dynamo_component"] for e in entries}
        # decode maps to "backend", matching the component tag Dynamo puts on spans.
        assert by_worker == {"node1": "prefill", "node2": "backend"}


class TestFrontendInfoLogParser:
    def test_parses_srtctl_container_stdout(self, tmp_path: Path):
        from src.ingest.frontend_infolog_parser import parse_frontend_log

        log = _write_frontend_log(tmp_path)
        parsed = parse_frontend_log(str(log))

        assert set(parsed["requests"]) == set(XIDS), "records key on x_request_id, not the internal id"
        rec = parsed["requests"]["xid0"]
        assert rec["ttft"] == 120.0
        assert rec["isl"] == 1024 and rec["osl"] == 128
        assert rec["status"] == "success"

    def test_stage_rows_carry_the_shared_ir(self, tmp_path: Path):
        """The renderer draws span-sourced and log-sourced breakdowns with ONE
        renderer, so this producer must emit the same StageRow keys."""
        from src.ingest.frontend_infolog_parser import parse_frontend_log

        rows = parse_frontend_log(str(_write_frontend_log(tmp_path)))["stages"]["xid0"]
        assert rows
        for row in rows:
            assert {"name", "t", "d", "depth", "opaque"} <= set(row)


# ---------------------------------------------------------------------------
# L2 orchestration: the srt-slurm run layout
# ---------------------------------------------------------------------------


class TestIngestBundle:
    def test_builds_bundle_from_srtslurm_layout(self, run_dir: Path, tmp_path: Path):
        """The whole point of the vendored defaults: pointed at a job's logs/ dir
        with no source flags, ingest finds both capture legs on its own."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)

        profile = bundle / "profile_export.jsonl"
        assert profile.exists(), "default client glob must reach into artifacts/<run>/"
        assert len(profile.read_text().splitlines()) == len(XIDS)
        assert (bundle / "server_metrics_export.jsonl").exists()

        yaml_text = (bundle / "dashboard.yaml").read_text()
        assert "name: smoke-run" in yaml_text
        assert "aiperf_profile: profile_export.jsonl" in yaml_text
        assert "server_metrics: server_metrics_export.jsonl" in yaml_text
        # Absent leg must not be advertised, or the renderer looks for a file that
        # was never produced.
        assert "tempo_traces" not in yaml_text

    def test_engine_configs_are_carried_into_the_bundle(self, run_dir: Path, tmp_path: Path):
        """srt-slurm writes trtllm_config_<mode>.yaml into the LOG dir; the renderer
        looks for it in the BUNDLE. Without the copy the renderer silently falls back
        to --max-batch-* defaults, and on a real run decode's true ceiling is 1 against
        a default of 256 -- the Engine tab then reads as "far from saturated" while the
        engine is pinned at its limit."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)

        assert (bundle / "trtllm_config_prefill.yaml").exists()
        assert (bundle / "trtllm_config_decode.yaml").exists()
        assert "max_batch_size: 1" in (bundle / "trtllm_config_decode.yaml").read_text()

    def test_renderer_reads_real_ceilings_not_defaults(self, run_dir: Path, tmp_path: Path):
        """Assert on the DATA payload, not the log line.

        The renderer logged the config-derived ceilings correctly while the Engine
        panel still consumed the CLI defaults -- the ceilings were parsed into a
        variable only the per-iteration panel read. A log-line assertion passes
        against that bug; only the payload catches it.
        """
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert proc.returncode == 0, proc.stderr

        engine = json.loads(payload.read_text())["en"]
        assert engine["max_batch_pf"] == 128
        assert engine["max_batch_de"] == 1, "the decode in-flight panel must be drawn against 1, not the 256 default"

    def test_cli_ceilings_apply_only_without_a_run_config(self, tmp_path: Path):
        """The CLI flags stay a fallback for bundles carrying no engine config."""
        d = tmp_path / "logs"
        d.mkdir()
        _write_raw_prometheus(d)
        _write_aiperf_export(d)  # deliberately no _write_engine_configs
        bundle = tmp_path / "bundle"
        _run_ingest(d, bundle)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload),
                       "--max-batch-decode", "77")
        assert proc.returncode == 0, proc.stderr
        assert json.loads(payload.read_text())["en"]["max_batch_de"] == 77

    def test_missing_engine_configs_are_not_fatal(self, tmp_path: Path):
        """A non-TRTLLM run has no such files; ingest must still produce a bundle."""
        d = tmp_path / "logs"
        d.mkdir()
        _write_raw_prometheus(d)
        _write_aiperf_export(d)
        bundle = tmp_path / "bundle"
        _run_ingest(d, bundle)
        assert (bundle / "server_metrics_export.jsonl").exists()
        assert not list(bundle.glob("trtllm_config_*.yaml"))

    def test_span_log_default_matches_srtslurm_naming(self):
        """srt-slurm writes <node>_<mode>_w<i>.out; upstream defaulted to *.log."""
        from src.ingest.ingest import build_parser

        args = build_parser().parse_args(["--run-dir", "."])
        assert args.span_logs == []  # empty -> run_traces falls back to ["*.out"]
        assert args.client == "aiperf"
        assert args.traces == "spanlog"


# ---------------------------------------------------------------------------
# L3: render
# ---------------------------------------------------------------------------


class TestRenderComponentDashboard:
    def test_renders_component_tabs_from_server_metrics(self, run_dir: Path, tmp_path: Path):
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        out = tmp_path / "dash.html"

        proc = _render(bundle, out, "--d3-cdn")
        assert proc.returncode == 0, proc.stderr
        assert out.exists()

        tabs = _tabs(out.read_text())
        assert tabs["router"] and tabs["engine"] and tabs["frontend"]
        # No traces in this fixture, and Overview needs client AND spans joined --
        # it is dropped rather than rendered empty.
        assert tabs["overview"] is False
        assert tabs["loganalysis"] is False

    def test_frontend_log_lights_up_log_analysis(self, run_dir: Path, tmp_path: Path):
        """The leg that works with no tracing and no client export -- the only one
        available on a sa-bench run."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        out = tmp_path / "dash.html"

        proc = _render(bundle, out, "--d3-cdn", "--frontend-log", str(run_dir / "node0_frontend_0.out"))
        assert proc.returncode == 0, proc.stderr
        assert _tabs(out.read_text())["loganalysis"] is True

    def test_log_only_build_needs_no_bundle(self, run_dir: Path, tmp_path: Path):
        out = tmp_path / "dash.html"
        proc = _render(None, out, "--d3-cdn", "--frontend-log", str(run_dir / "node0_frontend_0.out"))
        assert proc.returncode == 0, proc.stderr

        tabs = _tabs(out.read_text())
        assert tabs["loganalysis"] is True
        assert not any(tabs[t] for t in ("overview", "router", "engine", "frontend"))

    def test_mismatched_frontend_log_fails_the_build(self, run_dir: Path, tmp_path: Path):
        """Mixing runs would give a page whose header describes one workload and
        whose Log-analysis tab describes another. A warning is invisible to whoever
        opens the HTML, so this must be a hard failure."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)

        other = tmp_path / "other_frontend_0.out"
        other.write_text(
            _write_frontend_log(tmp_path).read_text().replace("xid", "otherxid")
        )
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--frontend-log", str(other))
        assert proc.returncode != 0
        assert "different run" in proc.stderr

    def test_d3_is_inlined_by_default(self, run_dir: Path, tmp_path: Path):
        """The HTML is routinely read after being pulled off a cluster or synced to
        S3, where a CDN fetch is not guaranteed."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        out = tmp_path / "dash.html"

        assert _render(bundle, out).returncode == 0
        html = out.read_text()
        assert "d3js.org v7" in html, "vendored d3.v7.min.js should be inlined"
        assert "<script src=" not in html, "a self-contained page must not fetch anything"


# ---------------------------------------------------------------------------
# The single-run pipeline: srtctl.analysis.perf_dashboard
# ---------------------------------------------------------------------------


def _stub_config(*, enabled=True, build_dashboard=None, prefill=(1, 4), decode=(3, 8)):
    """Minimal stand-in for SrtConfig carrying only what perf_dashboard reads."""
    from srtctl.core.schema import ObservabilityConfig

    obs = ObservabilityConfig.Schema().load(
        {"enabled": enabled} if build_dashboard is None else {"enabled": enabled, "build_dashboard": build_dashboard}
    )
    return SimpleNamespace(
        name="unit-test-run",
        observability=obs,
        resources=SimpleNamespace(
            num_prefill=prefill[0],
            gpus_per_prefill=prefill[1],
            num_decode=decode[0],
            gpus_per_decode=decode[1],
            num_agg=0,
            gpus_per_agg=0,
        ),
    )


class TestPerfDashboardPipeline:
    def test_repo_root_discovery_finds_the_vendored_tree(self):
        from srtctl.analysis.perf_dashboard import find_repo_root

        root = find_repo_root()
        assert root is not None, "must locate the checkout under an editable install"
        assert (root / "src" / "ingest" / "ingest.py").is_file()
        assert (root / "src" / "visualization" / "build_dynamo_bench_dash.py").is_file()

    def test_worker_specs_describe_the_topology(self):
        """These become --worker args, which are what give the page its GPU count.
        Getting them wrong silently rescales every per-GPU number on the page."""
        from srtctl.analysis.perf_dashboard import _worker_specs

        specs = _worker_specs(_stub_config(prefill=(2, 4), decode=(4, 8)))
        assert specs == ["prefill=dep:4:2", "decode=tep:8:4"]
        # 4*2 + 8*4 = 40 GPUs is what the renderer will divide by.

    def test_agg_only_topology_emits_one_spec(self):
        from srtctl.analysis.perf_dashboard import _worker_specs

        cfg = _stub_config(prefill=(0, 0), decode=(0, 0))
        cfg.resources.num_agg, cfg.resources.gpus_per_agg = 6, 4
        assert _worker_specs(cfg) == ["agg=tep:4:6"]

    def test_try_build_is_a_noop_when_not_opted_in(self, tmp_path: Path):
        from srtctl.analysis.perf_dashboard import try_build

        runtime = SimpleNamespace(log_dir=tmp_path, job_id="12345")
        assert try_build(_stub_config(enabled=False), runtime) is None
        assert not (tmp_path / "perf_dashboard.html").exists()

    def test_build_dashboard_false_overrides_observability_enabled(self, tmp_path: Path):
        from srtctl.analysis.perf_dashboard import try_build

        runtime = SimpleNamespace(log_dir=tmp_path, job_id="12345")
        assert try_build(_stub_config(enabled=True, build_dashboard=False), runtime) is None
        assert not (tmp_path / "perf_dashboard.html").exists()

    def test_end_to_end_one_run_produces_the_dashboard(self, run_dir: Path):
        """The whole point of the wiring: a finished log dir in, artifacts out, with
        no hand-driven step from a checkout."""
        from srtctl.analysis.perf_dashboard import BUNDLE_DIRNAME, try_build

        runtime = SimpleNamespace(log_dir=run_dir, job_id="2739690")
        html = try_build(_stub_config(), runtime)

        assert html is not None, "pipeline should have produced a dashboard"
        assert html.is_file() and html.stat().st_size > 100_000, "D3 should be inlined"

        payload = run_dir / "perf_dashboard.json"
        assert payload.is_file(), "the no-browser payload must be written"
        data = json.loads(payload.read_text())
        # The JSON is the same object the HTML embeds, so it can be asserted on.
        assert data["tabs"]["router"] and data["tabs"]["engine"] and data["tabs"]["frontend"]
        assert data["tabs"]["loganalysis"], "frontend log in the run dir should be picked up"
        assert data["meta"]["scrapes"] == 4

        bundle = run_dir / BUNDLE_DIRNAME
        assert (bundle / "server_metrics_export.jsonl").is_file()
        assert (bundle / "trtllm_config_decode.yaml").is_file(), "engine ceilings must reach the bundle"

    def test_missing_vendored_tree_is_survivable(self, run_dir: Path, monkeypatch):
        """A wheel install with no checkout must degrade to a warning, not an exception
        — this runs inside post-processing of a benchmark that already succeeded."""
        import srtctl.analysis.perf_dashboard as pd

        monkeypatch.setattr(pd, "find_repo_root", lambda: None)
        runtime = SimpleNamespace(log_dir=run_dir, job_id="12345")
        assert pd.try_build(_stub_config(), runtime) is None


# ---------------------------------------------------------------------------
# request-trace leg (schema 4)
# ---------------------------------------------------------------------------


def _trace_record(xid, sid, received_ms, *, prefill_wait, prefill, kv_transfer, total,
                  avg_itl, osl, hashes, turn_hashes_shared=None):
    """One dynamo.request.trace.v1 request_end record, real field paths."""
    return {
        "timestamp": 700000,
        "event": {
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_source": "dynamo",
            "agent_context": {"session_id": sid},
            "request": {
                "request_id": f"rid-{xid}",
                "x_request_id": xid,
                "model": "DeepSeek-V4-Pro",
                "input_tokens": 1024,
                "output_tokens": osl,
                "cached_tokens": 512,
                "request_received_ms": received_ms,
                "prefill_wait_time_ms": prefill_wait,
                "prefill_time_ms": prefill,
                "ttft_ms": prefill_wait + prefill,
                "total_time_ms": total,
                "avg_itl_ms": avg_itl,
                "kv_hit_rate": 0.5,
                "kv_transfer_estimated_latency_ms": kv_transfer,
                "queue_depth": 0,
                "worker": {
                    "prefill_worker_id": 111, "prefill_dp_rank": 2,
                    "decode_worker_id": 222, "decode_dp_rank": 0,
                },
                "replay": {"trace_block_size": 32, "input_length": 1024,
                           "input_sequence_hashes": hashes},
                "finish_reason_metadata": {"finish_reason": "length"},
            },
        },
    }


class TestRequestTraceProcessor:
    def test_ttft_is_corrected_for_kv_transfer(self, tmp_path: Path):
        """`ttft_ms` in the file is the PREFILL worker's first token. On disagg the
        client waits for decode, which cannot start until KV has transferred. Measured
        over 557 joined requests, adding kv_transfer moves the client-TTFT residual
        from 'wrong on 510/557' to 'wrong on 0/557'."""
        from src.ingest.request_trace import flatten

        row = flatten(_trace_record("x1", "s1", 1000, prefill_wait=4.0, prefill=680.0,
                                    kv_transfer=130.0, total=2500.0, avg_itl=66.0,
                                    osl=11, hashes=[1, 2, 3]))
        assert row["ttft_prefill_ms"] == 684.0
        assert row["client_ttft_ms"] == 814.0, "must add KV transfer"
        assert row["steady_decode_ms"] == pytest.approx(2500.0 - 814.0)

    def test_itl_is_decontaminated(self, tmp_path: Path):
        """avg_itl averages over a decode span starting at the PREFILL first token, so
        it absorbs the whole KV transfer. On the reference run that inflates p50 ITL
        roughly 11x (66.4ms raw vs 5.9ms clean)."""
        from src.ingest.request_trace import flatten

        row = flatten(_trace_record("x1", "s1", 1000, prefill_wait=4.0, prefill=680.0,
                                    kv_transfer=130.0, total=2500.0, avg_itl=66.0,
                                    osl=11, hashes=[1]))
        assert row["clean_itl_ms"] == pytest.approx(66.0 - 130.0 / 10)

    def test_aggregated_run_has_no_kv_transfer_correction(self):
        """Non-disagg: no transfer, so client TTFT IS the prefill-side TTFT."""
        from src.ingest.request_trace import flatten

        rec = _trace_record("x1", "s1", 1000, prefill_wait=4.0, prefill=680.0,
                            kv_transfer=None, total=2500.0, avg_itl=66.0, osl=11, hashes=[1])
        del rec["event"]["request"]["kv_transfer_estimated_latency_ms"]
        row = flatten(rec)
        assert row["client_ttft_ms"] == row["ttft_prefill_ms"] == 684.0
        assert row["clean_itl_ms"] == 66.0

    def test_non_request_end_records_are_skipped(self):
        from src.ingest.request_trace import flatten

        rec = _trace_record("x1", "s1", 1000, prefill_wait=1.0, prefill=1.0,
                            kv_transfer=1.0, total=1.0, avg_itl=1.0, osl=2, hashes=[1])
        rec["event"]["event_type"] = "tool_start"
        assert flatten(rec) is None

    def test_prefix_reuse_and_turn_order(self, tmp_path: Path):
        """Turn order is by received_ms (verified to reproduce the client's own
        turn_index on 33/33 sessions). Turn 0 has no predecessor, so its reuse is
        None -- 'no previous turn' is not 'no reuse'."""
        from src.ingest.request_trace import process

        src = tmp_path / "dynamo-request-trace"
        with src.open("w") as f:
            # deliberately out of chronological order in the file
            f.write(json.dumps(_trace_record("x2", "s1", 2000, prefill_wait=1.0, prefill=1.0,
                                             kv_transfer=1.0, total=10.0, avg_itl=1.0, osl=3,
                                             hashes=[1, 2, 3, 9])) + "\n")
            f.write(json.dumps(_trace_record("x1", "s1", 1000, prefill_wait=1.0, prefill=1.0,
                                             kv_transfer=1.0, total=10.0, avg_itl=1.0, osl=3,
                                             hashes=[1, 2, 3])) + "\n")
        out = tmp_path / "request_trace.jsonl"
        assert process(str(src), str(out)) == 2

        rows = {r["x_request_id"]: r for r in (json.loads(x) for x in out.read_text().splitlines())}
        assert rows["x1"]["turn_index"] == 0 and rows["x1"]["prefix_reuse_ratio"] is None
        assert rows["x2"]["turn_index"] == 1
        assert rows["x2"]["prefix_reuse_ratio"] == pytest.approx(0.75)  # 3 of 4 blocks shared

    def test_bulky_hashes_never_reach_the_bundle(self, tmp_path: Path):
        """input_sequence_hashes is 463k entries on a 20-min run and would dominate any
        embedded payload. Only the derived reuse ratio survives."""
        from src.ingest.request_trace import process

        src = tmp_path / "dynamo-request-trace"
        src.write_text(json.dumps(_trace_record("x1", "s1", 1000, prefill_wait=1.0, prefill=1.0,
                                                kv_transfer=1.0, total=10.0, avg_itl=1.0, osl=3,
                                                hashes=list(range(5000)))) + "\n")
        out = tmp_path / "request_trace.jsonl"
        process(str(src), str(out))
        row = json.loads(out.read_text().strip())
        assert "input_sequence_hashes" not in json.dumps(row)
        assert row["n_hash_blocks"] == 5000


class TestClientLayouts:
    @pytest.mark.parametrize("relpath", [
        "agentic/conc_3/aiperf_artifacts/profile_export.jsonl",  # AgentX
        "artifacts/Qwen3-32B_conversation_20260820/profile_export.jsonl",  # AIPerf bench.sh
    ])
    def test_both_client_layouts_are_found(self, tmp_path: Path, relpath: str):
        """AgentX nests one level deeper than the AIPerf harness. Missing the AgentX
        layout silently kills the Overview tab AND the trace leg, which is gated on
        the client xids."""
        from src.ingest.ingest import main

        d = tmp_path / "logs"
        d.mkdir()
        _write_raw_prometheus(d)
        export = d / relpath
        export.parent.mkdir(parents=True)
        export.write_text(
            json.dumps({
                "metadata": {"x_request_id": "xid0", "request_start_ns": T0,
                             "request_end_ns": T0 + 1_000_000_000},
                "metrics": {"time_to_first_token": {"value": 120.0},
                            "request_latency": {"value": 1200.0},
                            "input_sequence_length": {"value": 1024},
                            "output_sequence_length": {"value": 128}},
            }) + "\n"
        )
        bundle = tmp_path / "bundle"
        assert main(["--run-dir", str(d), "--out", str(bundle), "--traces", "none"]) == 0
        assert (bundle / "profile_export.jsonl").read_text().count("xid0") == 1


class TestRequestAndSessionEntities:
    """The two non-time-series dashboard entities, both built from the request trace."""

    @staticmethod
    def _bundle_with_trace(run_dir: Path, tmp_path: Path) -> Path:
        src = run_dir / "dynamo-request-trace"
        recs = []
        for i, xid in enumerate(XIDS):
            recs.append(_trace_record(
                xid, f"sess-{i % 2}", 1_787_174_000_000 + i * 1000,
                prefill_wait=2.0, prefill=500.0 + i, kv_transfer=50.0,
                total=1000.0 + i, avg_itl=20.0, osl=11,
                hashes=list(range(10 + i)),
            ))
        src.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        return bundle

    def test_request_bands_sum_to_total(self, run_dir: Path, tmp_path: Path):
        """The waterfall must be gapless -- on the reference runs there are 0/560
        negative residuals, and a band that silently absorbs a gap makes the card lie
        about where the time went."""
        bundle = self._bundle_with_trace(run_dir, tmp_path)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert proc.returncode == 0, proc.stderr

        rt = json.loads(payload.read_text())["rt"]
        assert rt["requests"], "request cards should be populated"
        for card in rt["requests"].values():
            total = sum(v for _, _, v in card["bands"] if v is not None)
            assert total == pytest.approx(card["attrs"]["total_ms"], abs=0.01)

    def test_sessions_carry_turn_ordered_series(self, run_dir: Path, tmp_path: Path):
        bundle = self._bundle_with_trace(run_dir, tmp_path)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert proc.returncode == 0, proc.stderr

        data = json.loads(payload.read_text())
        assert data["tabs"]["session"] is True
        sessions = data["rt"]["sessions"]
        assert len(sessions) == 2, "XIDS split across two sessions"
        for s in sessions:
            assert s["turns"] == len(s["ttft_ms"]) == len(s["xids"])
            # busy + idle must reconstruct the session's wall-clock span.
            assert s["busy_ms"] + s["idle_ms"] == pytest.approx(s["span_ms"], abs=0.2)
            assert s["idle_ms"] >= 0

    def test_constant_fields_are_reported_not_hidden(self, run_dir: Path, tmp_path: Path):
        """queue_depth reads a constant 0 on both reference runs. Reporting it as
        constant is the finding ('this run never queued'); dropping it silently would
        make that unaskable, and drawing it as a flat line looks like a broken chart."""
        bundle = self._bundle_with_trace(run_dir, tmp_path)
        payload = tmp_path / "dash.json"
        _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))

        const = json.loads(payload.read_text())["rt"]["const"]
        assert const["queue_depth"] == 0
        assert const["decode_dp_rank"] == 0

    def test_session_tab_absent_without_a_request_trace(self, run_dir: Path, tmp_path: Path):
        """Tab gating is on source availability: no trace file, no session tab."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)  # run_dir fixture has no dynamo-request-trace
        payload = tmp_path / "dash.json"
        _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))

        data = json.loads(payload.read_text())
        assert data["tabs"]["session"] is False
        assert data["rt"]["sessions"] == []


class TestPanelSpec:
    """The declarative panel layer: one evaluator, no per-panel code."""

    def test_every_panel_row_is_well_formed(self):
        """A malformed row would fail at render time on a cluster, hours after the
        run it was meant to explain."""
        from src.visualization.panels import KINDS, PANELS

        ids = [p["id"] for p in PANELS]
        assert len(ids) == len(set(ids)), "panel ids must be unique"
        for p in PANELS:
            assert p["kind"] in KINDS, f"{p['id']}: bad kind"
            assert p["metrics"], f"{p['id']}: no source metric"
            assert p["title"] and p["why"], f"{p['id']}: missing title/why"
            if p["kind"] == "ratio":
                assert len(p["metrics"]) == 2, f"{p['id']}: ratio needs exactly two counters"

    def test_titles_are_run_independent(self):
        """Principle: no panel title may encode the run it was written against."""
        import re

        from src.visualization.panels import PANELS

        for p in PANELS:
            assert not re.search(r"\d", p["title"]), f"{p['id']}: digit in title suggests a hardcoded count"
            for banned in ("agentx", "2739690", "deepseek", "lyris", "gb200", "gb300"):
                assert banned not in p["title"].lower(), f"{p['id']}: run-specific title"

    def test_no_engine_metric_is_split_by_dp_rank(self):
        """Engine-side dp_rank is a replicated broadcast -- every rank reports
        identical values. Splitting on it renders a flat family of lines that reads as
        'no imbalance' on a run that had a 12x spread."""
        from src.visualization.panels import PANELS

        for p in PANELS:
            if any(m.startswith("trtllm_") for m in p["metrics"]):
                assert p["split_by"] != "dp_rank", f"{p['id']}: would render a false flat"

    def test_counter_rate_ignores_a_counter_reset(self):
        """A restart makes a counter go backwards. Emitting the negative delta draws a
        throughput cliff that never happened."""
        from src.visualization.panels import evaluate

        spec = [{"id": "c", "tab": "t", "title": "T", "unit": "u", "kind": "counter_rate",
                 "metrics": ["m"], "split_by": None, "why": "w", "issues": [], "caveat": None}]
        scrapes = [(i * 1_000_000_000, {"m": [{"labels": {}, "value": v}]})
                   for i, v in enumerate([10, 20, 5, 15])]
        series = evaluate(scrapes, spec)["c"]["series"]["all"]
        assert [v for _, v in series] == [10.0, 10.0], "the reset must be dropped, not plotted"

    def test_hist_mean_is_an_interval_mean(self):
        """A cumulative mean flattens over a long run and hides the late degradation
        these panels exist to catch."""
        from src.visualization.panels import evaluate

        spec = [{"id": "h", "tab": "t", "title": "T", "unit": "s", "kind": "hist_mean",
                 "metrics": ["lat"], "split_by": None, "why": "w", "issues": [], "caveat": None}]
        # interval 1: 10 events / 10s -> 1.0 ; interval 2: 10 events / 100s -> 10.0
        scrapes = [
            (0, {"lat_sum": [{"labels": {}, "value": 0.0}], "lat_count": [{"labels": {}, "value": 0}]}),
            (1_000_000_000, {"lat_sum": [{"labels": {}, "value": 10.0}], "lat_count": [{"labels": {}, "value": 10}]}),
            (2_000_000_000, {"lat_sum": [{"labels": {}, "value": 110.0}], "lat_count": [{"labels": {}, "value": 20}]}),
        ]
        assert [v for _, v in evaluate(scrapes, spec)["h"]["series"]["all"]] == [1.0, 10.0]

    def test_split_by_produces_one_series_per_worker(self):
        from src.visualization.panels import evaluate

        spec = [{"id": "g", "tab": "t", "title": "T", "unit": "u", "kind": "gauge",
                 "metrics": ["m"], "split_by": "worker_id", "why": "w", "issues": [], "caveat": None}]
        scrapes = [(0, {"m": [{"labels": {"worker_id": "a"}, "value": 1},
                              {"labels": {"worker_id": "b"}, "value": 2}]})]
        assert set(evaluate(scrapes, spec)["g"]["series"]) == {"a", "b"}

    def test_panels_with_no_data_are_omitted(self):
        """Omission is what lets a tab drop cleanly instead of rendering empty."""
        from src.visualization.panels import evaluate

        spec = [{"id": "absent", "tab": "t", "title": "T", "unit": "u", "kind": "gauge",
                 "metrics": ["never_emitted"], "split_by": None, "why": "w", "issues": [], "caveat": None}]
        assert evaluate([(0, {"other": [{"labels": {}, "value": 1}]})], spec) == {}

    def test_panels_reach_the_payload(self, run_dir: Path, tmp_path: Path):
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert proc.returncode == 0, proc.stderr

        panels = json.loads(payload.read_text())["panels"]
        assert panels, "the fixture emits KV-cache metrics, so panels must be present"
        for p in panels.values():
            assert {"tab", "title", "unit", "kind", "why", "source", "series"} <= set(p)


class TestGeneratedPageStructure:
    """Dependency-free structural checks on the emitted HTML.

    These cannot prove the page RENDERS -- that needs a DOM, and
    ``tools/dashboard_render_check.js`` does it with jsdom. What they do catch is the
    page losing a whole feature (a tab, the spec-panel renderer, the session table)
    without anyone noticing, which is cheap to check and has no npm cost.
    """

    def test_page_carries_every_tab_and_renderer(self, run_dir: Path, tmp_path: Path):
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        out = tmp_path / "dash.html"
        assert _render(bundle, out, "--d3-cdn").returncode == 0

        html = out.read_text()
        for marker in ("['session','Session']", "declarative spec panels",
                       "DATA.panels", "DATA.rt", ".tbl th{", "attr('class','tbl')"):
            assert marker in html, f"page lost: {marker}"

    def test_page_js_has_balanced_script_tags(self, run_dir: Path, tmp_path: Path):
        """A truncated template would produce a page that silently renders nothing."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        out = tmp_path / "dash.html"
        _render(bundle, out, "--d3-cdn")

        html = out.read_text()
        assert html.rstrip().endswith("</html>")
        # `<script` not `<script>`: with --d3-cdn one tag is `<script src=...>`, so
        # counting only the bare form is asymmetric by construction.
        assert html.count("<script") == html.count("</script>")


class TestDoubleMetricPollingWarning:
    """A run can poll /metrics twice without saying so, which has previously made a
    benchmark submission irreproducible."""

    @staticmethod
    def _mixin(benchmark_type: str, scraper_enabled: bool):
        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin

        obj = BenchmarkStageMixin()
        obj.config = SimpleNamespace(
            benchmark=SimpleNamespace(type=benchmark_type),
            observability=SimpleNamespace(scraper_enabled=scraper_enabled),
        )
        return obj

    def test_warns_for_an_aiperf_benchmark(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            self._mixin("mooncake-router", True)._warn_on_double_metric_polling()
        assert any("Double /metrics polling" in r.message for r in caplog.records)

    def test_silent_for_a_non_aiperf_benchmark(self, caplog):
        """sa-bench drives no client-side polling, so there is nothing to warn about."""
        import logging

        with caplog.at_level(logging.WARNING):
            self._mixin("sa-bench", True)._warn_on_double_metric_polling()
        assert not any("Double /metrics polling" in r.message for r in caplog.records)

    def test_unknown_benchmark_type_is_not_an_error(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            self._mixin("no-such-benchmark", True)._warn_on_double_metric_polling()


class TestWaterfallKvTransferBand:
    def test_run_level_waterfall_has_the_kv_transfer_band(self, run_dir: Path, tmp_path: Path):
        """The decode `handle_payload` span starts AFTER the KV cache has transferred,
        so a spans-only waterfall butts prefill compute against decode compute and
        silently absorbs the wait between them. On the reference run that gap is 83%
        of TTFT at p90 -- the single largest phase, invisible."""
        src = run_dir / "dynamo-request-trace"
        src.write_text("\n".join(
            json.dumps(_trace_record(xid, "s1", 1_787_174_000_000 + i * 1000,
                                     prefill_wait=2.0, prefill=500.0, kv_transfer=250.0,
                                     total=1000.0, avg_itl=20.0, osl=11, hashes=[1, 2]))
            for i, xid in enumerate(XIDS)) + "\n")
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert proc.returncode == 0, proc.stderr

        wf = json.loads(payload.read_text())["waterfall"]
        assert wf, "fixture has no spans, so the waterfall may be empty; guard the assert"
        for band in wf.values():
            assert "kvt" in band, "the run-level waterfall must carry the KV-transfer band"

    def test_kvt_band_is_zero_without_a_request_trace(self, run_dir: Path, tmp_path: Path):
        """Aggregated runs have no transfer, and older captures have no trace file.
        The band must still exist so the waterfall's shape is run-independent."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)  # no dynamo-request-trace in the fixture
        payload = tmp_path / "dash.json"
        _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        for band in json.loads(payload.read_text())["waterfall"].values():
            assert band["kvt"] == 0


class TestInstantaneousImbalancePanels:
    """Balance over a phase and balance at an instant are different properties.

    Reported in the field as: "Over the whole benchmark phase, I don't see a large
    imbalance between DP ranks. However, there is a large instantaneously imbalance."
    A run-total per rank is exactly the aggregate that hides it.
    """

    @staticmethod
    def _trace(run_dir: Path, ranks_by_turn):
        recs = []
        for i, rank in enumerate(ranks_by_turn):
            r = _trace_record(f"x{i}", "s1", 1_787_174_000_000 + i * 10,
                              prefill_wait=1.0, prefill=10.0, kv_transfer=1.0,
                              total=1000.0, avg_itl=2.0, osl=11, hashes=[1])
            r["event"]["request"]["worker"]["prefill_dp_rank"] = rank
            recs.append(r)
        (run_dir / "dynamo-request-trace").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n")

    def _panels(self, run_dir: Path, tmp_path: Path):
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert proc.returncode == 0, proc.stderr
        return json.loads(payload.read_text())["panels"]

    def test_concurrent_skew_is_detected(self, run_dir: Path, tmp_path: Path):
        """Every request lands on rank 0 while ranks 1-3 idle."""
        self._trace(run_dir, [0, 0, 0, 0, 0, 1])
        p = self._panels(run_dir, tmp_path)["bal_prefill_dp_rank"]
        assert max(v for _, v in p["series"]["max-min"]) >= 4

    def test_balanced_load_shows_no_spread(self, run_dir: Path, tmp_path: Path):
        """Overlapping requests spread evenly across ranks must not raise an alarm."""
        self._trace(run_dir, [0, 1, 2, 3])
        p = self._panels(run_dir, tmp_path)["bal_prefill_dp_rank"]
        assert max(v for _, v in p["series"]["max-min"]) <= 1

    def test_single_rank_run_emits_no_panel(self, run_dir: Path, tmp_path: Path):
        """With one rank there is no imbalance to express; a flat zero line would
        imply the run was checked and found balanced."""
        self._trace(run_dir, [0, 0, 0])
        assert "bal_prefill_dp_rank" not in self._panels(run_dir, tmp_path)

    def test_derived_panel_matches_the_spec_panel_shape(self, run_dir: Path, tmp_path: Path):
        """It must be indistinguishable from a spec panel to the renderer -- a
        different source, not a different kind of panel."""
        self._trace(run_dir, [0, 0, 1, 2])
        p = self._panels(run_dir, tmp_path)["bal_prefill_dp_rank"]
        assert {"tab", "title", "unit", "kind", "why", "source", "caveat", "series"} <= set(p)


class TestRoutingOutcomeAndBeliefCheck:
    """Entity type 2 asks the card to explain the routing decision, not just the timing."""

    @staticmethod
    def _bundle(run_dir: Path, tmp_path: Path):
        src = run_dir / "dynamo-request-trace"
        src.write_text("\n".join(
            json.dumps(_trace_record(xid, "s1", 1_787_174_000_000 + i * 1000,
                                     prefill_wait=2.0, prefill=500.0, kv_transfer=50.0,
                                     total=1000.0, avg_itl=20.0, osl=11, hashes=[1, 2]))
            for i, xid in enumerate(XIDS)) + "\n")
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert proc.returncode == 0, proc.stderr
        return json.loads(payload.read_text())

    def test_cards_exist_without_spans(self, run_dir: Path, tmp_path: Path):
        """The fixture has no SPAN_CLOSED logs, so routing is unjoinable. The card must
        still be produced with routing None rather than fabricating a decision."""
        rt = self._bundle(run_dir, tmp_path)["rt"]
        assert rt["requests"]
        assert all(c["routing"] is None for c in rt["requests"].values())

    def test_belief_summary_absent_without_routing(self, run_dir: Path, tmp_path: Path):
        """No overlap_blocks means the belief check has nothing to compare; it must
        report nothing rather than a vacuous 100% agreement."""
        assert self._bundle(run_dir, tmp_path)["rt"]["belief"] is None


_ITER_LINE = (
    "[08/19/2026-14:11:49] [TRT-LLM] [I] [_torch][RANK 0] iter = {i}, global_rank = 0, "
    "rank = 0, num_scheduled_requests = {sched}, kv_cache_util = {kv}, "
    "currank_total_requests = 0/1, host_step_time = {host}ms, "
    "prev_device_step_time = {dev}, timestamp = 2026-08-19 14:11:{sec:02d}, "
    "states = {{'num_ctx_requests': 1, 'num_ctx_tokens': 4, "
    "'num_generation_tokens': {gen}, 'cached_kv_tokens': 0}}"
)


class TestIterLogProcessor:
    """TRT-LLM print_iter_log -> iter_bins.json. The only source for batch composition."""

    def test_parses_a_real_line(self):
        from src.ingest.iter_log import parse_line

        r = parse_line(_ITER_LINE.format(i=2, sched=3, kv="0.112", host="4.08",
                                         dev="243.13ms", sec=49, gen=7))
        assert r["iter"] == 2 and r["num_scheduled_requests"] == 3
        assert r["kv_cache_util"] == 0.112
        assert r["host_step_time_ms"] == 4.08 and r["device_step_time_ms"] == 243.13
        assert r["num_generation_tokens"] == 7

    def test_na_device_time_is_none_not_zero(self):
        """N/A on the first iteration of a lifecycle. Zero would be a real measurement
        and would drag any aggregate toward it."""
        from src.ingest.iter_log import parse_line

        assert parse_line(_ITER_LINE.format(i=1, sched=1, kv="0.001", host="243.17",
                                            dev="N/A", sec=49, gen=0))["device_step_time_ms"] is None

    def test_non_iter_lines_are_ignored(self):
        from src.ingest.iter_log import parse_line

        assert parse_line("[TRT-LLM] [I] some other log line entirely") is None

    def test_timezone_offset_is_derived_not_hardcoded(self, tmp_path: Path):
        """TRT-LLM stamps worker-LOCAL time while every other bundle source is UTC.
        Reading it naively puts engine events hours from the frontend events they
        belong to. The offset is derived from the run window, so it follows the
        cluster and DST instead of assuming one zone."""
        from src.ingest.iter_log import derive_offset_hours

        recs = [{"local_ts": "2026-08-19 14:11:49"}]
        # Log says 14:11 local; the run window is centred 7h later in UTC.
        base = int(datetime(2026, 8, 19, 21, 11, 49, tzinfo=timezone.utc).timestamp() * 1e9)
        assert derive_offset_hours(recs, base - 10**11, base + 10**11) == 7
        # A window centred on the log time itself needs no shift.
        base0 = int(datetime(2026, 8, 19, 14, 11, 49, tzinfo=timezone.utc).timestamp() * 1e9)
        assert derive_offset_hours(recs, base0 - 10**11, base0 + 10**11) == 0

    def test_no_window_means_no_guess(self):
        """An underived offset beats an invented one."""
        from src.ingest.iter_log import derive_offset_hours

        assert derive_offset_hours([{"local_ts": "2026-08-19 14:11:49"}], None, None) == 0

    def test_bins_carry_the_scheduled_distribution(self, tmp_path: Path):
        """A median alone cannot answer "did it ever batch" -- the distribution can."""
        from src.ingest.iter_log import process

        log = tmp_path / "node_decode_w0.out"
        lines = [_ITER_LINE.format(i=i, sched=s, kv="0.01", host="13.0", dev="12.0ms",
                                   sec=49, gen=1)
                 for i, s in enumerate([0, 1, 1, 4], start=1)]
        log.write_text("\n".join(lines) + "\n")
        out = tmp_path / "iter_bins.json"
        assert process(str(out), [str(log)]) >= 1

        d = json.loads(out.read_text())
        rows = d["bins"]["node_decode_w0"]
        hist = {}
        for r in rows:
            for k, v in r["sched_hist"].items():
                hist[k] = hist.get(k, 0) + v
        assert hist == {"0": 1, "1": 2, "4": 1}
        assert max(r["sched_max"] for r in rows) == 4

    def test_no_iter_lines_is_not_an_error(self, tmp_path: Path):
        from src.ingest.iter_log import process

        log = tmp_path / "node_decode_w0.out"
        log.write_text("nothing of interest here\n")
        assert process(str(tmp_path / "iter_bins.json"), [str(log)]) == 0


class TestRenderedEntities:
    """Payload without a renderer is not coverage. These pin the JS consumers."""

    def test_request_cards_have_a_renderer(self, run_dir: Path, tmp_path: Path):
        src = run_dir / "dynamo-request-trace"
        src.write_text("\n".join(
            json.dumps(_trace_record(x, "s1", 1_787_174_000_000 + i * 1000,
                                     prefill_wait=2.0, prefill=500.0, kv_transfer=50.0,
                                     total=1000.0, avg_itl=20.0, osl=11, hashes=[1, 2]))
            for i, x in enumerate(XIDS)) + "\n")
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        out = tmp_path / "dash.html"
        assert _render(bundle, out, "--d3-cdn").returncode == 0

        html = out.read_text()
        # DATA.rt.requests carried the per-request card for several commits with no JS
        # consumer at all -- the payload existed and nothing drew it.
        for marker in ("Per-request decomposition", "reqcard", "DATA.rt.requests",
                       "Router belief vs engine reality"):
            assert marker in html, f"per-request entity lost its renderer: {marker}"

    def test_high_cardinality_panels_state_what_they_omit(self, run_dir: Path, tmp_path: Path):
        """A per-thread runtime gauge is ~144 series. Drawing all is unreadable and
        drawing some silently reads as complete."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        out = tmp_path / "dash.html"
        _render(bundle, out, "--d3-cdn")
        assert "highest-peak of" in out.read_text()


class TestKvHitRateGating:
    def test_reuse_disabled_components_are_excluded(self, run_dir: Path, tmp_path: Path):
        """A worker with enable_block_reuse:false reports a hard 0 -- correctly, it does
        no reuse. Averaging that into the run-level rate makes the number fall as the
        deployment adds decode workers. On the reference run the naive mean reported
        0.0 where the true prefill rate was 65.1."""
        (run_dir / "trtllm_config_prefill.yaml").write_text(
            "max_batch_size: 128\nmax_num_tokens: 4096\nkv_cache_config:\n  enable_block_reuse: true\n")
        (run_dir / "trtllm_config_decode.yaml").write_text(
            "max_batch_size: 1\nmax_num_tokens: 4\nkv_cache_config:\n  enable_block_reuse: false\n")
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert proc.returncode == 0, proc.stderr
        assert json.loads(payload.read_text())["en"]["reuse_components"] == ["prefill"]

    def test_absent_config_keeps_every_component(self, run_dir: Path, tmp_path: Path):
        """No engine config means "cannot tell", which must not silently drop data."""
        for f in run_dir.glob("trtllm_config_*.yaml"):
            f.unlink()
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert json.loads(payload.read_text())["en"]["reuse_components"] is None

    def test_header_block_size_comes_from_the_stream(self, run_dir: Path, tmp_path: Path):
        """The yaml carries only ingest's --block-size fallback; on the reference run
        that fallback (512) disagreed with the measured value (32) and the engine
        config (256). The header must not show the fallback."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        topo = json.loads(payload.read_text())["meta"]["topo"]
        assert "__BLK__" not in topo
        assert "block 512" not in topo


class TestBlockSizeMismatch:
    """Router block size vs engine tokens-per-block.

    The e2e self-build run 2751593 reported blk_router=None blk_engine=None -- "cannot
    tell" -- on a run whose engine config plainly said 256. The gauge the check read
    (`trtllm_kv_cache_tokens_per_block`) is only exported once the engine publishes its
    KV-cache family, so a run without it lost a check whose whole purpose is a failure
    mode that is invisible in the metrics.
    """

    def test_engine_block_size_falls_back_to_the_run_config(self, run_dir: Path, tmp_path: Path):
        """No gauge in the scrape, but tokens_per_block in the config -> still decided."""
        (run_dir / "trtllm_config_decode.yaml").write_text(
            "max_batch_size: 1\nmax_num_tokens: 4\nkv_cache_config:\n  tokens_per_block: 256\n")
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert proc.returncode == 0, proc.stderr
        load = json.loads(payload.read_text())["load"]
        assert load["blk_engine"] == 256, "engine size must come from the config when the gauge is absent"
        assert load["blk_src"] == "engine config"

    def test_a_mismatch_is_reported_not_averaged_away(self, run_dir: Path, tmp_path: Path):
        """Router 32 vs engine 256 is the reference run's real configuration."""
        (run_dir / "trtllm_config_decode.yaml").write_text(
            "max_batch_size: 1\nkv_cache_config:\n  tokens_per_block: 256\n")
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        load = json.loads(payload.read_text())["load"]
        if load["blk_router"] is not None:
            assert load["blk_router"] != load["blk_engine"], "the fixture encodes a mismatch"

    def test_no_config_and_no_gauge_stays_unknown(self, run_dir: Path, tmp_path: Path):
        """A guessed all-clear is worse than an admitted unknown: the consequence of a
        mismatch is a silent router index, so 'no mismatch' must never be inferred from
        missing data."""
        for f in run_dir.glob("trtllm_config_*.yaml"):
            f.unlink()
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert json.loads(payload.read_text())["load"]["blk_engine"] is None


class TestHeaderProvenance:
    def test_header_explains_an_empty_profiling_population(self, run_dir: Path, tmp_path: Path):
        """A run can end before the profiling phase, leaving every client record marked
        warmup. meta.n is then legitimately 0 -- but a bare "0 requests" beside a
        Session tab listing dozens of sessions reads as a broken page. Observed for
        real: 118 warmup records, 0 profiling, 65 sessions."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        out = tmp_path / "dash.html"
        assert _render(bundle, out, "--d3-cdn").returncode == 0

        html = out.read_text()
        assert "profiling requests" in html, "the header must name which population it counts"
        assert "are warmup and are shown on the" in html, "and name the other one"


class TestJsonlSelectorRecovery:
    """On a DYN_LOGGING_JSONL=true run every record is JSON, and the parser used to
    discard every non-SPAN_CLOSED JSON line. That silently threw away the entire
    router family: 597 real selector records on the reference log counted as 0."""

    @staticmethod
    def _log(tmp_path: Path, msgs) -> Path:
        p = tmp_path / "node_frontend_0.out"
        p.write_text("\n".join(json.dumps({
            "time": "2026-08-19T21:15:34.05374%dZ" % (i % 10),
            "level": "DEBUG",
            "target": "dynamo_kv_router::scheduling::selector",
            "message": m,
        }) for i, m in enumerate(msgs)) + "\n")
        return p

    def test_jsonl_selector_decisions_are_counted(self, tmp_path: Path):
        from src.ingest.frontend_infolog_parser import parse_frontend_log

        log = self._log(tmp_path, [
            "Selected pinned worker: worker_type=prefill, worker_id=7587 dp_rank=1, logit=0.5",
            "Selected pinned worker: worker_type=decode, worker_id=7588 dp_rank=0, logit=0.5",
            "Selected worker",
        ])
        s = parse_frontend_log(str(log))["stats"]
        assert s["selector_without_request_id"] == 3, "JSON selector records must be seen"
        assert s["selector_decisions_pinned"] == 2

    def test_candidate_scores_are_not_counted_as_decisions(self, tmp_path: Path):
        """A per-candidate score is not a choice. Conflating them would inflate the
        decision count by the number of workers considered."""
        from src.ingest.frontend_infolog_parser import parse_frontend_log

        log = self._log(tmp_path, [
            "Formula for worker_id=7587 dp_rank=0 with 0.00 effective cached blocks: 8094.094 = a * b",
            "Pinned formula for worker_id=7588 dp_rank=1 with 12.00 effective cached blocks: 42.0 = a * b",
            "Selected pinned worker: worker_type=prefill, worker_id=7588 dp_rank=1, logit=0.5",
        ])
        s = parse_frontend_log(str(log))["stats"]
        assert s["selector_candidate_scores"] == 2
        assert s["selector_without_request_id"] == 1

    def test_span_closed_records_still_parse(self, tmp_path: Path):
        """The fix must not disturb the SPAN_CLOSED path it sits next to."""
        from src.ingest.frontend_infolog_parser import parse_frontend_log

        p = tmp_path / "node_frontend_0.out"
        p.write_text(json.dumps({
            "time": "2026-08-19T21:15:34.053746Z", "level": "DEBUG",
            "target": "request_span", "message": "SPAN_CLOSED",
            "request_id": "r1", "x_request_id": "xid0", "time.duration_us": 1_000_000,
            "ttft_ms": 120.0, "input_tokens": 10, "output_tokens": 5,
        }) + "\n")
        parsed = parse_frontend_log(str(p))
        assert "xid0" in parsed["requests"]


class TestSpanBusyIdleRetained:
    def test_busy_and_idle_survive_as_attributes(self, tmp_path: Path):
        """time.busy_us / time.idle_us split a span's wall time into work vs waiting --
        the difference between a span that is slow because it computed and one that is
        slow because it blocked. They were being dropped as envelope metadata."""
        from src.ingest.traces_spanlog import parse_line

        span = parse_line(json.dumps({
            "time": "2026-08-19T21:15:34.053746Z", "message": "SPAN_CLOSED",
            "span_name": "handle_payload", "span_id": "a", "parent_id": "b",
            "trace_id": "t", "request_id": "r", "x_request_id": "x",
            "time.duration_us": 1000, "time.busy_us": 200, "time.idle_us": 800,
            "component": "prefill",
        }))
        assert span["attrs"]["time.busy_us"] == 200
        assert span["attrs"]["time.idle_us"] == 800
        assert "time.duration_us" not in span["attrs"], "duration is the span length, not an attr"


class TestMetricNameResolution:
    """Prometheus counters conventionally end in `_total`, and ingest keeps the scraped
    name verbatim. A lookup written without the suffix finds nothing and the caller
    reads a hard zero -- indistinguishable downstream from a metric that really was
    zero. Five families were being read this way."""

    def test_counter_total_suffix_is_resolved(self, run_dir: Path, tmp_path: Path):
        """The tokenizer-cache KPI read 0.0% on a run whose real hit rate was 99.2%."""
        # 40 hits, 10 misses -> 80%
        path = run_dir / "raw_prometheus.jsonl"
        lines = []
        for i in range(4):
            ts = T0 + i * 1_000_000_000
            lines.append(json.dumps({
                "timestamp_ns": ts, "endpoint_url": "http://head:8000/metrics",
                "role": "frontend", "worker_id": None,
                "text": (f"dynamo_frontend_tokenizer_cache_hits_total {10 * (i + 1)}\n"
                         f"dynamo_frontend_tokenizer_cache_misses_total {2.5 * (i + 1)}\n"
                         f'dynamo_frontend_requests_total{{model="m"}} {i}\n'),
            }))
        path.write_text("\n".join(lines) + "\n")

        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        payload = tmp_path / "dash.json"
        proc = _render(bundle, tmp_path / "dash.html", "--d3-cdn", "--dump-json", str(payload))
        assert proc.returncode == 0, proc.stderr

        kpi = json.loads(payload.read_text())["kpi"]
        assert kpi["tok_cache"] == pytest.approx(80.0, abs=0.1), (
            "a counter named <x>_total must resolve from a lookup written as <x>")

    def test_no_renderer_metric_name_omits_a_needed_suffix(self):
        """Guards the whole class: every metric name the renderer looks up must either
        exist verbatim or be resolvable. Names are checked against the catalogued
        families so a newly-added lookup cannot silently read zero."""
        import re

        src = (REPO_ROOT / "src/visualization/build_dynamo_bench_dash.py").read_text()
        # Counter families that only ever exist with the _total suffix.
        counters_needing_total = {
            "dynamo_frontend_requests",
            "dynamo_frontend_tokenizer_cache_hits",
            "dynamo_frontend_tokenizer_cache_misses",
            "dynamo_frontend_tokenizer_cache_cached_tokens",
            "dynamo_frontend_tokenizer_cache_uncached_tokens",
        }
        referenced = set(re.findall(r'["\'](dynamo_[a-z0-9_]+|trtllm_[a-z0-9_]+)["\']', src))
        bare = referenced & counters_needing_total
        # They MAY appear bare -- but only because _entries() resolves the suffix.
        assert "_entries" in src and "name + \"_total\"" in src, (
            f"bare counter names {sorted(bare)} are referenced, so the central "
            "_total-resolution helper must still exist")


class TestKpiProvenance:
    def test_kpi_items_carry_a_source(self, run_dir: Path, tmp_path: Path):
        """A KPI with no provenance is indistinguishable from one computed off the
        wrong population -- which has happened here more than once (the tokenizer-cache
        KPI read 0.0% against a real 99.2% because of a metric-name mismatch)."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        out = tmp_path / "dash.html"
        assert _render(bundle, out, "--d3-cdn",
                       "--frontend-log", str(run_dir / "node0_frontend_0.out")).returncode == 0

        html = out.read_text()
        assert "'routing decisions seen'" in html or "routing decisions seen" in html
        assert ".kpi .src{" in html, "the per-KPI source annotation style must exist"

    def test_no_branching_prose_in_the_log_tab_note(self, run_dir: Path, tmp_path: Path):
        """One caption when a feature is present and a different one when it is absent
        is the varying-description defect: the caption becomes a function of the run
        instead of the value doing that job."""
        bundle = tmp_path / "bundle"
        _run_ingest(run_dir, bundle)
        out = tmp_path / "dash.html"
        _render(bundle, out, "--d3-cdn", "--frontend-log", str(run_dir / "node0_frontend_0.out"))

        html = out.read_text()
        assert "not available in this run, so TTFT is a single opaque block" not in html
