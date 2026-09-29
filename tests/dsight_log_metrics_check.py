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
from test_dsight_log_metrics import batch, config, log_run

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
                summary = build_dashboard(imported.logs, out / case / "report", single_file=True, iteration_timezone="UTC")
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
