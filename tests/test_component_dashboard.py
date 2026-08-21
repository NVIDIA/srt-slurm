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
