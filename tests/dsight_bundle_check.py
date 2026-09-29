# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progressive static delivery in isolated Chrome, with exact evidence comparisons.

uv run --with websockets python tests/dsight_bundle_check.py --out /path/to/artifacts
Pass --report and --baseline to additionally measure preserved reports served over HTTP.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import shutil
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import websockets
from dsight_browser_check import browser_targets
from dsight_metrics_check import Browser
from test_dsight import write_run

from srtctl.dsight.build import write_dashboard
from srtctl.dsight.importer import Importer
from srtctl.dsight.query import TraceDataset


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


async def check(args, port, origin, root):
    page = next(p for p in browser_targets(port) if p["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=100_000_000) as socket:
        browser = Browser(socket, args.out)
        for domain in ("Runtime", "Page", "Network", "Performance"):
            await browser.call(domain + ".enable")
        await browser.call("Network.setCacheDisabled", cacheDisabled=True)
        await browser.call(
            "Page.addScriptToEvaluateOnNewDocument",
            source="""
            window.addEventListener('trace-explorer:ready', () => {window.readyMs = performance.now()});
        """,
        )
        measurements = []

        async def navigate(path, *, throttle=False):
            await browser.call(
                "Page.navigate", url=origin + "/" + (args.out / "warmup.html").relative_to(root).as_posix()
            )
            await browser.wait("document.title === 'warmup'")
            if throttle:
                await browser.call(
                    "Network.emulateNetworkConditions",
                    offline=False,
                    latency=20,
                    downloadThroughput=1_250_000,
                    uploadThroughput=1_250_000,
                )
            await browser.call("Page.navigate", url=origin + "/" + path.resolve().relative_to(root).as_posix())
            await browser.wait("Boolean(window.readyMs)", timeout=120)
            await browser.js("traceExplorer.whenMetricsReady()")
            await browser.js("traceExplorer.whenDetailsReady?.()")
            await browser.js("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
            measurement = await browser.js("""({
                ready_ms: readyMs, with_metrics_ms: performance.now(),
                navigation: performance.getEntriesByType('navigation').map(e => ({bytes:e.transferSize,body:e.encodedBodySize,duration:e.duration})),
                resources: performance.getEntriesByType('resource').map(e => ({name:e.name,bytes:e.transferSize,body:e.encodedBodySize})),
                heap: performance.memory?.usedJSHeapSize,
                error: document.getElementById('error').textContent,
            })""")
            assert not measurement["error"], measurement
            measurement.update(path=str(path), throttled_10mbit=throttle)
            measurements.append(measurement)
            if throttle:
                await browser.call(
                    "Network.emulateNetworkConditions",
                    offline=False,
                    latency=0,
                    downloadThroughput=-1,
                    uploadThroughput=-1,
                )
            return measurement

        report = args.out / "fixture"
        dataset = TraceDataset.from_path(report)
        initial = await navigate(report / "index.html")
        # The initial overview must not fetch NVTX shards.
        assert len(initial["resources"]) < 10
        profile = dataset.query("profiles")["items"][0]["id"]
        await browser.js(f"traceExplorer.inspectNsys({{profile:{profile},from:0,to:10}})")
        await browser.js("traceExplorer.whenDetailsReady()")
        assert "Density overview" in await browser.js("document.getElementById('nsysTracks').textContent")
        before = len(await browser.js("performance.getEntriesByType('resource')"))
        result = await browser.js(f"traceExplorer.inspectNsys({{profile:{profile},from:6,to:6.1}})")
        await browser.js("traceExplorer.whenDetailsReady()")
        expected = dataset.query("nsys", profile=profile, start=6, end=6.1, limit=100)
        assert result["total"] == expected["total"]
        assert [r["rowid"] for r in result["items"]] == [r["rowid"] for r in expected["items"]]
        assert "Loading selected" not in await browser.js("document.getElementById('nsysTracks').textContent")
        assert "Density overview" not in await browser.js("document.getElementById('nsysTracks').textContent")
        # Neighboring range transitions may abort reads. Only the final view wins.
        await browser.js(
            "traceExplorer.selectRange(0,1);traceExplorer.selectRange(8,8.1);traceExplorer.selectRange(6,6.1)"
        )
        await browser.js("traceExplorer.whenDetailsReady()")
        await browser.js("traceExplorer.whenMetricsReady()")
        assert (await browser.js("traceExplorer.getState()"))["from"] == 6
        again = await browser.js("traceExplorer.queryNsys({offset:10,limit:15})")
        assert again["items"] == result["items"][10:25]
        exported = await browser.js("traceExplorer.exportSelection()")
        assert exported["profile"]["total"] == expected["total"]
        assert exported["sources"] == dataset.data["sources"]
        cache = await browser.js("traceExplorer.detailDataStatus()")
        assert cache["decodedBytes"] <= cache["maxDecodedBytes"]
        assert not await browser.js("document.getElementById('error').textContent")
        await browser.screenshot("fixture-window.png")
        # Confirm real browser metric rows, including same-timestamp conflicts,
        # null samples, carried settings, and source references, across shards.
        metrics = await browser.js("traceExplorer.queryMetrics({from:3,to:4,points:true})")
        python_metrics = dataset.query("metrics", start=3, end=4, points=True)["items"]
        for actual, expected_series in zip(metrics, python_metrics, strict=True):
            for field in ("samples", "min", "max", "mean", "last", "points"):
                assert actual[field] == expected_series[field], (field, actual, expected_series)
            if expected_series.get("temporal") == "setting":
                assert actual["carried_setting"] == expected_series["carried_setting"]
        cpu = await browser.js("traceExplorer.queryCpu({from:1,to:4})")
        expected_cpu = dataset.query("cpu", start=1, end=4)
        assert cpu["total_samples"] == expected_cpu["total_samples"]
        assert cpu["hotspots"] == expected_cpu["items"]
        # Cold reload with a pinned metric/window must restore the same state.
        state = await browser.js("traceExplorer.getState()")
        await browser.call(
            "Page.navigate",
            url=origin
            + "/"
            + (report / "index.html").relative_to(root).as_posix()
            + "#view="
            + quote(json.dumps(state)),
        )
        await browser.wait("Boolean(window.readyMs)")
        await browser.js("traceExplorer.whenDetailsReady()")
        await browser.js("traceExplorer.whenMetricsReady()")
        assert (await browser.js("traceExplorer.getState()"))["from"] == state["from"]

        for path in (args.baseline, args.report):
            if path:
                for _ in range(args.repeats):
                    await navigate(path / "index.html")
                if path == args.report:
                    await navigate(path / "index.html", throttle=True)
                    stored = TraceDataset.from_path(args.dataset or path)
                    p = max(stored.query("profiles")["items"], key=lambda x: x["event_count"])
                    start = min(600, stored.data["meta"]["duration"] / 2)
                    began = time.perf_counter()
                    exact = await browser.js(
                        f"traceExplorer.inspectNsys({{profile:{p['id']},from:{start},to:{start + 1}}})"
                    )
                    await browser.js("traceExplorer.whenDetailsReady()")
                    elapsed = time.perf_counter() - began
                    expected = stored.query("nsys", profile=p["id"], start=start, end=start + 1)
                    assert exact["total"] == expected["total"]
                    assert [r["rowid"] for r in exact["items"]] == [r["rowid"] for r in expected["items"]]
                    measurements.append({"selected_window_s": elapsed, "profile": p["id"], "total": exact["total"]})
                    await browser.screenshot("preserved-window.png")
        errors = [e for e in browser.events if e.get("method") == "Runtime.exceptionThrown"]
        assert not errors, errors
        return {
            "measurements": measurements,
            "cache": cache,
            "fixture_query_total": result["total"],
            "fixture_resources_before_window": before,
            "browser_errors": errors,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--dataset", type=Path, help="Local query cache when --report contains only browser files")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    logs, sqlites = write_run(args.out / "sources")
    data = Importer(logs, sqlites, iteration_timezone="UTC").run()
    data["profiles"][0]["events"] = [[i / 12000, i / 12000 + 0.0001, 0, "17", i] for i in range(120000)]
    data["profiles"][0]["cpu"] = {
        "pid": 0,
        "names": ["a", "b"],
        "stacks": [[0, 0, 1], [1]],
        "samples": [[1, "17", 0, 1], [3, "17", 1, 2]],
        "attribution": "samples",
    }
    data["metrics"][0]["temporal"] = "setting"
    data["metrics"][0]["points"] = [[1, i % 2, 0, i] for i in range(8200)] + [[3, None, 0, 8201], [4, 5, 0, 8202]]
    data["metrics"][0]["conflict_timestamps"] = [1]
    write_dashboard(data, args.out / "fixture")
    root = Path(os.path.commonpath([args.out, *[p.resolve() for p in (args.report, args.baseline) if p]]))
    (args.out / "warmup.html").write_text("<title>warmup</title>")
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    chrome_log = (args.out / "chrome.log").open("w")
    profile = args.out / "chrome"
    process = subprocess.Popen(
        [
            shutil.which("google-chrome") or "chromium",
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdout=chrome_log,
        stderr=chrome_log,
    )
    try:
        for _ in range(100):
            if (profile / "DevToolsActivePort").exists():
                break
            time.sleep(0.1)
        port = int((profile / "DevToolsActivePort").read_text().splitlines()[0])
        result = asyncio.run(check(args, port, f"http://127.0.0.1:{server.server_port}", root))
        (args.out / "browser-results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        process.terminate()
        process.wait(timeout=15)
        chrome_log.close()
        server.shutdown()


if __name__ == "__main__":
    main()
