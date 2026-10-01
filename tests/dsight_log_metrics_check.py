# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise log capacity charts through source fixtures and the real offline UI."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets
from dsight_browser_check import browser_targets
from dsight_metrics_check import Browser
from test_dsight import write_run
from test_dsight_log_metrics import batch, config, log_run
from test_dsight_sglang_log_metrics import PREFILL_REQUEST

from srtctl.dsight.build import build_dashboard
from srtctl.dsight.log_metrics.tokenspeed import ACTIVE_DECODE, ACTIVE_PAGES, DECODE_LIMIT


async def run(out: Path, port: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    page = next(p for p in await asyncio.to_thread(browser_targets, port) if p["type"] == "page")
    report = {"cases": []}
    cases = {
        "constant": [config(), batch(), batch(36, active=6, pages=96)],
        "changed": [config(), batch(), config(35, "16"), batch(36, active=12)],
        "missing": [batch(), batch(36)],
        "wrong-rank": [config(rank=1), batch(), batch(36)],
        "late": [batch(), config(35), batch(36)],
        "invalidated": [config(), batch(), config(35, "unknown"), batch(36)],
        "conflicting": [config(), config(maximum="16"), batch(), batch(36)],
    }
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=100_000_000) as socket:
        browser = Browser(socket, out)
        try:
            for domain in ("Page", "Runtime", "Network"):
                await browser.call(f"{domain}.enable")
            await browser.call(
                "Emulation.setDeviceMetricsOverride", width=1500, height=1000, deviceScaleFactor=1, mobile=False
            )
            for case, lines in cases.items():
                imported, _ = log_run(out / case, lines)
                summary = build_dashboard(
                    imported.logs, out / case / "report", single_file=True, iteration_timezone="UTC"
                )
                await browser.call("Page.navigate", url="about:blank")
                await browser.wait("!window.traceExplorer")
                await browser.call("Page.navigate", url=Path(summary["html"]).resolve().as_uri())
                await browser.wait("Boolean(window.traceExplorer?.ready)")
                await browser.js("traceExplorer.whenMetricsReady()")
                # Observe the public chart boundary and actual uPlot state; don't replace behavior.
                await browser.js("""(()=>{
                  window.__plots=[];window.__options=[];
                  const Plot=window.uPlot;
                  window.uPlot=Object.assign(function(...args){const plot=new Plot(...args);__plots.push(plot);return plot},Plot);
                  const charts=DSightMetricCharts;
                  window.DSightMetricCharts={mount(host,options){__options.push(options);return charts.mount(host,options)}};
                })()""")
                await browser.js(
                    "traceExplorer.setState("
                    + json.dumps(
                        {"metric": ACTIVE_DECODE, "pinnedMetrics": [ACTIVE_DECODE, ACTIVE_PAGES], "from": 0, "to": 10}
                    )
                    + ")"
                )
                await browser.js("traceExplorer.whenMetricsReady()")
                texts = await browser.js(
                    "Array.from(document.querySelectorAll('.ds-metric-capacity'),e=>e.textContent)"
                )
                assert len(texts) == 2
                active = texts[0]
                assert await browser.js("__plots[0].scales.y.min") == 0
                assert not await browser.js("document.getElementById('error').textContent")
                if case == "constant":
                    assert (
                        "Peak observed 6 requests" in active
                        and "Configured batch limit 8" in active
                        and "75%" in active
                    )
                    assert "Peak observed 96 pages" in texts[1] and "KV pool size 128" in texts[1]
                    assert await browser.js("__plots[0].series.length") == 3
                    assert await browser.js("__plots[0].series[2].dash") == [6, 4]
                    assert await browser.js("__options[0].references[0].points[0][0]") == -1
                    assert await browser.js("__plots[0].data[0][0]") == 0  # Display projection only.
                    assert await browser.js("__plots[1].data[0][0]") == 2.23  # Sampled pool never projected.
                    evidence = await browser.js(
                        "traceExplorer.queryMetrics({name:" + json.dumps(DECODE_LIMIT) + ",points:true})"
                    )
                    assert evidence[0]["points"] == [] and evidence[0]["carried_setting"][0][:2] == [-1, 8]
                    await browser.rectangle("#metricsSection")
                    await browser.screenshot("01-capacity-panels.png")
                    await browser.click(".ds-metric-legend-toggle")
                    assert await browser.js("__plots[0].series.slice(1).map(s=>s.show)") == [False, False]
                    await browser.click(".ds-metric-legend-toggle")
                    await browser.js("traceExplorer.setState({from:0,to:3})")
                    await browser.js("traceExplorer.whenMetricsReady()")
                    zoomed = await browser.js("document.querySelector('.ds-metric-capacity').textContent")
                    assert "Peak observed 4 requests" in zoomed and "50%" in zoomed
                    await browser.call(
                        "Emulation.setDeviceMetricsOverride", width=390, height=1000, deviceScaleFactor=1, mobile=False
                    )
                    await asyncio.sleep(0.2)
                    await browser.rectangle("#metricsSection")
                    await browser.screenshot("02-narrow-capacity.png")
                    assert await browser.js("document.documentElement.scrollWidth<=innerWidth+1")
                    await browser.call(
                        "Emulation.setDeviceMetricsOverride", width=1500, height=1000, deviceScaleFactor=1, mobile=False
                    )
                elif case == "changed":
                    assert "8–16 (changed)" in active and "75%" in active and "Peak observed 12 requests" in active
                    await browser.rectangle("#metricsSection")
                    await browser.screenshot("03-changing-limit.png")
                elif case in {"missing", "wrong-rank"}:
                    assert "Configured batch limit unavailable" in active and "2/2 samples" in active
                    assert await browser.js("__plots[0].series.length") == 2
                elif case in {"late", "invalidated"}:
                    assert "1/2 samples" in active
                    if case == "late":
                        assert await browser.js("__options[0].references[0].points[0][0]") == 4
                    else:
                        assert "50%" in active
                else:
                    assert "2/2 samples" in active and "Highest observed usage" not in active
                    assert await browser.js("__options[0].references[0].conflict_timestamps") == [-1]
                report["cases"].append({"case": case, "summaries": texts})
            # uPlot needs unique x coordinates; event evidence must retain all
            # source lines while the plot explicitly labels its display median.
            logs, _ = write_run(out / "request-events")
            base = PREFILL_REQUEST.replace("2026-09-24 01:47:50.616", "2026-09-17 10:58:33.232")
            lines = [
                base,
                base.replace("rid=request-one", "rid=request-two"),
                base.replace("rid=request-one", "rid=request-three").replace(
                    "queue_duration=0.41ms", "queue_duration=1.23ms"
                ),
                base.replace("10:58:33.232", "10:58:34.232").replace("rid=request-one", "rid=request-four"),
                base.replace("10:58:33.232", "10:58:34.232")
                .replace("rid=request-one", "rid=request-five")
                .replace("queue_duration=0.41ms", "queue_duration=1.23ms"),
            ]
            (logs / "prefill-host_prefill_w0.out").write_text("\n".join(lines) + "\n")
            summary = build_dashboard(
                logs, out / "request-events" / "report", single_file=True, iteration_timezone="UTC"
            )
            await browser.call("Page.navigate", url="about:blank")
            await browser.wait("!window.traceExplorer")
            await browser.call("Page.navigate", url=Path(summary["html"]).resolve().as_uri())
            await browser.wait("Boolean(window.traceExplorer?.ready)")
            await browser.js("traceExplorer.whenMetricsReady()")
            await browser.js("""(()=>{
              window.__plots=[];
              const Plot=window.uPlot;
              window.uPlot=Object.assign(function(...args){const plot=new Plot(...args);__plots.push(plot);return plot},Plot);
            })()""")
            metric = "log_sglang_request_queue_duration_ms"
            await browser.js(
                "traceExplorer.setState("
                + json.dumps({"metric": metric, "pinnedMetrics": [metric], "from": 0, "to": 10})
                + ")"
            )
            await browser.js("traceExplorer.whenMetricsReady()")
            query = "traceExplorer.queryMetrics({name:" + json.dumps(metric) + ",points:true})"
            raw = await browser.js(query)
            assert len(raw) == 1 and [p[1] for p in raw[0]["points"]] == [0.41, 0.41, 1.23, 0.41, 1.23]
            assert [p[3] for p in raw[0]["points"]] == [1, 2, 3, 4, 5]
            plotted = await browser.js("__plots.at(-1).data")
            assert plotted[0] == [2.232, 3.232]
            assert abs(plotted[1][0] - 0.41) < 1e-12 and abs(plotted[1][1] - 0.82) < 1e-12
            for time, count in [(2.232, 3), (3.232, 2)]:
                await browser.js(
                    f"(()=>{{const p=__plots.at(-1);p.setCursor({{left:p.valToPos({time},'x'),top:10}})}})()"
                )
                value = await browser.js("document.querySelector('.ds-metric-value').textContent")
                assert "median" in value and f"({count} events)" in value, value
            assert await browser.js(query) == raw
            assert not await browser.js("document.getElementById('error').textContent")
            await browser.rectangle("#metricsSection")
            await browser.screenshot("04-request-events.png")
            report["cases"].append(
                {"case": "request-events", "raw_points": len(raw[0]["points"]), "display_points": plotted}
            )
            # Stock SGLang logging omits ranks, milliseconds, and batch counters.
            # Multiple batches at one second are separate observations, not conflicts.
            logs, _ = write_run(out / "stock-batch-events")
            (logs / "prefill-host_prefill_w0.out").write_text(
                "".join(
                    f"[2026-09-17 10:58:{second}] Prefill batch, #new-token: {value}\n"
                    for second, value in [(33, 64), (33, 64), (33, 128), (34, 64), (34, 128)]
                )
            )
            summary = build_dashboard(
                logs, out / "stock-batch-events" / "report", single_file=True, iteration_timezone="UTC"
            )
            await browser.call("Page.navigate", url="about:blank")
            await browser.wait("!window.traceExplorer")
            await browser.call("Page.navigate", url=Path(summary["html"]).resolve().as_uri())
            await browser.wait("Boolean(window.traceExplorer?.ready)")
            await browser.js("traceExplorer.whenMetricsReady()")
            await browser.js("""(()=>{
              window.__plots=[];
              const Plot=window.uPlot;
              window.uPlot=Object.assign(function(...args){const plot=new Plot(...args);__plots.push(plot);return plot},Plot);
            })()""")
            metric = "log_sglang_new_tokens"
            await browser.js(
                "traceExplorer.setState("
                + json.dumps({"metric": metric, "pinnedMetrics": [metric], "from": 0, "to": 10})
                + ")"
            )
            await browser.js("traceExplorer.whenMetricsReady()")
            query = "traceExplorer.queryMetrics({name:" + json.dumps(metric) + ",points:true})"
            raw = await browser.js(query)
            assert len(raw) == 1 and [p[1] for p in raw[0]["points"]] == [64, 64, 128, 64, 128]
            assert [p[3] for p in raw[0]["points"]] == [1, 2, 3, 4, 5]
            assert raw[0]["rank"] is None and raw[0]["time_resolution_s"] == 1
            assert not raw[0]["conflict_timestamps"]
            plotted = await browser.js("__plots.at(-1).data")
            assert plotted == [[2, 3], [64, 96]]
            for time, count in [(2, 3), (3, 2)]:
                await browser.js(
                    f"(()=>{{const p=__plots.at(-1);p.setCursor({{left:p.valToPos({time},'x'),top:10}})}})()"
                )
                value = await browser.js("document.querySelector('.ds-metric-value').textContent")
                assert "median" in value and f"({count} events)" in value, value
            assert await browser.js(query) == raw
            assert not await browser.js("document.getElementById('error').textContent")
            await browser.rectangle("#metricsSection")
            await browser.screenshot("05-stock-batch-events.png")
            report["cases"].append(
                {"case": "stock-batch-events", "raw_points": len(raw[0]["points"]), "display_points": plotted}
            )
            errors = [event for event in browser.events if event.get("method") == "Runtime.exceptionThrown"]
            assert not errors, errors
            external = [
                event
                for event in browser.events
                if event.get("method") == "Network.requestWillBeSent"
                and event["params"]["request"]["url"].startswith(("http:", "https:"))
            ]
            assert not external, external
            report["runtime_errors"] = errors
            report["external_requests"] = external
        except Exception as error:
            report["failure"] = str(error)
            await browser.screenshot("failure.png")
            raise
        finally:
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9222)
    args = parser.parse_args()
    asyncio.run(run(args.out, args.port))
