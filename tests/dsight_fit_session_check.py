# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the Fit Session button with real pointer input in offline Chrome."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
from pathlib import Path

import websockets
from dsight_browser_check import browser_targets
from dsight_metrics_check import Browser
from test_dsight import CLIENT, ORIGIN, write_run

from srtctl.dsight.build import build_dashboard


async def run(output: Path, port: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    logs, profiles = write_run(output / "inputs")
    client = next(logs.rglob("profile_export.jsonl"))
    with client.open("a") as stream:
        stream.write(json.dumps({
            "metadata": {
                "request_id": "subagent-turn", "root_correlation_id": "session-a",
                "x_correlation_id": "child-agent", "parent_correlation_id": "agent-a", "agent_depth": 1,
                "request_start_ns": ORIGIN + 2_000_000_000, "request_end_ns": ORIGIN + 4_000_000_000,
                "benchmark_phase": "profiling",
            },
            "metrics": {"time_to_first_token": {"value": 100}},
        }) + "\n")
    metrics_only = output / "metrics-only-input"
    shutil.copytree(logs / "tachometer", metrics_only / "tachometer")
    example = Path(__file__).resolve().parents[1] / "examples/dsight/fit-session"
    reports = {
        "traced": build_dashboard(logs, output / "traced", sqlites=profiles, iteration_timezone="UTC"),
        "client-only": build_dashboard(example, output / "client-only"),
        "metrics-only": build_dashboard(metrics_only, output / "metrics-only"),
    }
    page = next(t for t in await asyncio.to_thread(browser_targets, port) if t["type"] == "page")
    results = []
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=20_000_000) as ws:
        browser = Browser(ws, output)
        for domain in ("Page", "Runtime", "Network"):
            await browser.call(domain + ".enable")
        await browser.call("Network.setBlockedURLs", urls=["http://*", "https://*"])
        await browser.call("Emulation.setDeviceMetricsOverride", width=1600, height=1100, deviceScaleFactor=1, mobile=False)

        async def navigate(name: str) -> None:
            await browser.call("Page.navigate", url="about:blank")
            await browser.wait("!window.traceExplorer")
            await browser.call("Page.navigate", url=Path(reports[name]["html"]).as_uri())
            await browser.wait("Boolean(window.traceExplorer?.ready)")
            await browser.js("traceExplorer.whenMetricsReady()")

        async def fit(request: str, expected: tuple[float, float]) -> None:
            await browser.js("traceExplorer.selectRequest(" + json.dumps(request) + ", {fit:true,expand:true})")
            # Only this request is visible, but its siblings still determine the session envelope.
            await browser.js("traceExplorer.setState({search:" + json.dumps(request) + "})")
            if not await browser.js("document.querySelector('#fitTTFT').disabled"):
                await browser.click("#fitTTFT")
            else:
                await browser.click("#zoomIn")
            await browser.js("traceExplorer.whenMetricsReady()")
            before = await browser.js("traceExplorer.getState()")
            assert await browser.js("document.querySelector('#fitSession').textContent") == "Fit Session"
            await browser.click("#fitSession")
            await browser.js("traceExplorer.whenMetricsReady()")
            after = await browser.js("traceExplorer.getState()")
            assert math.isclose(after["from"], expected[0], abs_tol=1e-8), after
            assert math.isclose(after["to"], expected[1], abs_tol=1e-8), after
            for key in ("request", "search", "span", "tab", "profile", "nsys", "pinnedMetrics", "metricCharts",
                        "expandedRequests", "expandedSessions", "expandedAgents"):
                assert after[key] == before[key], key
            assert not await browser.js("document.querySelector('#error').textContent")
            # A second click at the same range must not add a duplicate history entry.
            await browser.click("#fitSession")
            await browser.click("#rangeBack")
            restored = await browser.js("traceExplorer.getState()")
            assert (restored["from"], restored["to"]) == (before["from"], before["to"])
            await browser.click("#fitSession")
            await browser.js("traceExplorer.whenMetricsReady()")
            results.append({"request": request, "session_range": expected, "passed": True})

        await navigate("traced")
        await browser.js("traceExplorer.setState({pinnedMetrics:['trtllm_num_requests_running'],metricCharts:{"
                         "'[\"metric\",\"trtllm_num_requests_running\"]':{hidden:['0']}}})")
        await fit(CLIENT, (0, 8.48))
        await fit("subagent-turn", (0, 8.48))
        # Missing TTFT and OTel still allow fitting this single-request session.
        await fit("client-only", (8.94, 10))
        await navigate("client-only")
        await fit("parent-turn", (.52, 9.48))
        await fit("child-turn", (.52, 9.48))
        await fit("sibling-turn", (.52, 9.48))
        await browser.screenshot("session-fit-desktop.png")
        await browser.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=1, mobile=False)
        await fit("child-turn", (.52, 9.48))
        assert not await browser.js("document.documentElement.scrollWidth > innerWidth")
        await browser.screenshot("session-fit-narrow.png")
        await fit("earlier-turn", (0, .265))
        await fit("later-turn", (10.94, 12))
        await navigate("metrics-only")
        assert not await browser.js("Boolean(document.querySelector('#fitSession'))")
        assert not await browser.js("document.querySelector('#error').textContent")
        errors = [e for e in browser.events if e.get("method") == "Runtime.exceptionThrown"]
        assert not errors, errors
        report = {"cases": results, "request_free_button_absent": True, "runtime_errors": errors}
        (output / "fit-session-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.out, args.port))
