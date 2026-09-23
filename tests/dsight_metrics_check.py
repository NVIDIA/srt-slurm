# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify metric interactions on a real offline DSight report through Chrome."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import websockets
from dsight_browser_check import browser_targets


class Browser:
    """Small CDP client retaining runtime and network events for the audit."""

    def __init__(self, socket: Any, output: Path) -> None:
        self.socket = socket
        self.output = output
        self.sequence = 0
        self.events: list[dict[str, Any]] = []

    async def call(self, method: str, **params: Any) -> dict[str, Any]:
        self.sequence += 1
        identifier = self.sequence
        await self.socket.send(json.dumps({"id": identifier, "method": method, "params": params}))
        while True:
            message = json.loads(await self.socket.recv())
            if message.get("id") == identifier:
                if "error" in message:
                    raise AssertionError(message["error"])
                return message.get("result", {})
            self.events.append(message)

    async def js(self, expression: str) -> Any:
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if result.get("exceptionDetails"):
            raise AssertionError(result["exceptionDetails"])
        return result["result"].get("value")

    async def wait(self, expression: str, *, timeout: float = 45) -> None:
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            if await self.js(expression):
                return
            error = await self.js("window.traceExplorerError || null")
            if error:
                raise AssertionError(error)
            await asyncio.sleep(0.1)
        raise AssertionError(f"Timed out waiting for {expression}")

    async def rectangle(self, selector: str) -> dict[str, float]:
        return await self.js(
            "(()=>{const e=document.querySelector("
            + json.dumps(selector)
            + ");if(!e)throw Error('Missing control');e.scrollIntoView({block:'center'});"
            "const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})()"
        )

    async def click(self, selector: str) -> None:
        rectangle = await self.rectangle(selector)
        position = {"x": rectangle["x"] + rectangle["width"] / 2, "y": rectangle["y"] + rectangle["height"] / 2}
        await self.call("Input.dispatchMouseEvent", type="mousePressed", button="left", clickCount=1, **position)
        await self.call("Input.dispatchMouseEvent", type="mouseReleased", button="left", clickCount=1, **position)

    async def screenshot(self, name: str) -> None:
        result = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        (self.output / name).write_bytes(base64.b64decode(result["data"]))


async def run(html: Path, output: Path, port: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    targets = await asyncio.to_thread(browser_targets, port)
    page = next(target for target in targets if target["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=100_000_000) as socket:
        browser = Browser(socket, output)
        report: dict[str, Any] = {
            "html": str(html.resolve()),
            "html_sha256": hashlib.sha256(html.read_bytes()).hexdigest(),
            "tests": [],
        }
        try:
            for domain in ("Page", "Runtime", "Network"):
                await browser.call(f"{domain}.enable")
            await browser.call(
                "Emulation.setDeviceMetricsOverride", width=1600, height=1200, deviceScaleFactor=1, mobile=False
            )
            await browser.call("Page.navigate", url="about:blank")
            await browser.wait("!window.traceExplorer")
            started = time.monotonic()
            await browser.call("Page.navigate", url=html.resolve().as_uri())
            await browser.wait("Boolean(window.traceExplorer?.ready)")
            report["load_seconds"] = time.monotonic() - started
            report["description"] = await browser.js("traceExplorer.describe()")
            assert not await browser.js("document.getElementById('error').textContent")
            await check_metrics(browser, report)
            report["runtime_errors"] = [
                event for event in browser.events if event.get("method") == "Runtime.exceptionThrown"
            ]
            report["external_requests"] = [
                event["params"]["request"]["url"]
                for event in browser.events
                if event.get("method") == "Network.requestWillBeSent"
                and event["params"]["request"]["url"].startswith(("http://", "https://"))
            ]
            assert not report["runtime_errors"], report["runtime_errors"]
            assert not report["external_requests"], report["external_requests"]
            report["tests"].append(
                "The complete offline interaction sequence makes no external requests or runtime errors"
            )
        except Exception as error:
            report["failure"] = str(error)
            await browser.screenshot("failure.png")
            raise
        finally:
            (output / "metrics-browser-report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"load_seconds": report["load_seconds"], "tests": report["tests"]}, indent=2))


async def check_metrics(browser: Browser, report: dict[str, Any]) -> None:
    """Check one metric chart, legend visibility, recorded values, brushing, and links."""
    await browser.wait("Boolean(window.DSightMetricCharts)")
    duration = report["description"]["meta"]["duration"]
    await browser.js(
        "traceExplorer.setState("
        + json.dumps({"from": 0, "to": duration, "hardware": False, "nsys": False, "expandedSessions": []})
        + ")"
    )
    original_metrics = await browser.js("traceExplorer.queryMetrics()")
    report["metric_count"] = len(original_metrics)

    def metric_key(name: str) -> str:
        return json.dumps(["metric", name], separators=(",", ":"))

    def panel(key: str) -> str:
        return f".dsight-metric-panel[data-metric-key={json.dumps(key)}]"

    def toggle(key: str, identifier: str) -> str:
        return panel(key) + f" .ds-metric-legend-row[data-series-id={json.dumps(identifier)}] .ds-metric-legend-toggle"

    async def drawn(key: str) -> list[str]:
        return await browser.js(
            f"JSON.parse(document.querySelector({json.dumps(panel(key) + ' .ds-metric-chart')}).dataset.drawnIds)"
        )

    async def selection(key: str) -> dict[str, Any]:
        return await browser.js(f"traceExplorer.getState().metricCharts[{json.dumps(key)}]")

    async def visible(key: str) -> list[str]:
        return await browser.js(
            f"Array.from(document.querySelectorAll({json.dumps(panel(key) + ' .ds-metric-legend-row')}))"
            ".filter(e=>e.querySelector('.ds-metric-legend-toggle').getAttribute('aria-pressed')==='true')"
            ".map(e=>e.dataset.seriesId)"
        )

    async def choose_metric(value: str) -> None:
        await browser.js(
            f"(()=>{{const e=document.getElementById('workerMetric');e.value={json.dumps(value)};"
            "e.dispatchEvent(new Event('change',{bubbles:true}))})()"
        )

    async def no_extra_controls() -> None:
        assert await browser.js("document.querySelectorAll('.ds-metric-filters,.ds-metric-chooser').length") == 0
        assert not await browser.js(
            "Array.from(document.querySelectorAll('.ds-metric-chart')).some(e=>"
            "/Filter labels|Choose series/.test(e.textContent))"
        )

    name = await browser.js("traceExplorer.getState().metric")
    key = metric_key(name)
    expected = sorted(
        (metric for metric in original_metrics if metric["name"] == name), key=lambda metric: metric["id"]
    )
    expected_ids = [str(metric["id"]) for metric in expected]
    assert len({metric["worker"] for metric in expected}) == 4, "Use the preserved four-worker TRT-LLM run"
    assert await browser.js("document.querySelectorAll('.dsight-metric-panel').length") == 1
    assert await drawn(key) == expected_ids
    assert await visible(key) == expected_ids
    await no_extra_controls()
    report["combined_chart"] = {
        "metric": name,
        "ids": expected_ids,
        "workers": [metric["worker"] for metric in expected],
    }
    report["tests"].append(
        "One chart includes the selected metric from all four workers, with no filter or chooser controls"
    )
    assert not await browser.js("Boolean(document.getElementById('workerActivity'))")
    context_worker = await browser.js("traceExplorer.getRequest(traceExplorer.getState().request).workers[0]")
    await browser.click(f"[data-path-worker={json.dumps(context_worker)}]")
    assert await browser.js("Boolean(document.getElementById('workerActivity'))")
    assert await browser.js("document.querySelectorAll('.dsight-metric-panel').length") == 1
    await browser.click("#workerActivity [data-worker-nsys]")
    assert await browser.js("document.querySelectorAll('.nsys-track').length") > 0
    await browser.click("#workerActivity [data-toggle='worker']")
    assert not await browser.js("Boolean(document.getElementById('workerActivity'))")
    await browser.js("traceExplorer.setState({nsys:false,tab:'request'})")
    assert await browser.js("document.querySelectorAll('.dsight-metric-panel').length") == 1
    report["tests"].append(
        "Inspector worker activity and Nsight remain accessible without adding separate worker metric charts"
    )

    scope = panel(key)
    identifier = expected_ids[0]
    await browser.click(scope + " .ds-metric-label-details summary")
    labels = await browser.js(
        f"JSON.parse(document.querySelector({json.dumps(scope + ' .ds-metric-label-details pre')}).textContent)"
    )
    assert labels["labels"] == expected[0]["labels"]
    assert labels["endpoint"] == expected[0]["endpoint"]
    assert labels["rank"] == expected[0]["rank"]
    report["tests"].append("Legend label details expose the exact recorded labels, endpoint, and rank")
    await browser.screenshot("01-combined-worker-labels.png")
    await browser.click(scope + " .ds-metric-label-details summary")

    # All sources remain accessible, including after every visible line is hidden.
    for source_id in expected_ids:
        await browser.click(toggle(key, source_id))
    assert await visible(key) == []
    assert set((await selection(key))["hidden"]) == set(expected_ids)
    assert await drawn(key) == expected_ids
    assert await browser.js(f"document.querySelectorAll({json.dumps(scope + ' .ds-metric-legend-row')}).length") == len(
        expected_ids
    )
    await browser.click(toggle(key, identifier))
    assert await visible(key) == [identifier]
    for source_id in expected_ids[1:]:
        await browser.click(toggle(key, source_id))
    assert await visible(key) == expected_ids
    assert (await selection(key))["hidden"] == []
    report["tests"].append(
        "Legend clicks hide every line and restore them without losing access to any recorded source"
    )

    # Old chooser/filter fields no longer restrict the legend or its source set.
    await browser.js(
        "traceExplorer.setState("
        + json.dumps({"metricCharts": {key: {"ids": [], "filters": {"worker": "missing"}, "hidden": []}}})
        + ")"
    )
    assert await drawn(key) == expected_ids
    assert await visible(key) == expected_ids
    await no_extra_controls()
    report["tests"].append("Legacy filter and chooser state cannot remove sources from the combined chart")

    # Each selected family owns its visibility state.
    await browser.click(toggle(key, identifier))
    next_name = next(
        candidate
        for candidate in ("trtllm_num_requests_running", "trtllm_num_requests_waiting")
        if candidate != name and any(metric["name"] == candidate for metric in original_metrics)
    )
    next_key = metric_key(next_name)
    next_ids = [str(metric["id"]) for metric in original_metrics if metric["name"] == next_name]
    await choose_metric(next_name)
    assert await browser.js("document.querySelectorAll('.dsight-metric-panel').length") == 1
    assert await drawn(next_key) == next_ids
    assert await visible(next_key) == next_ids
    await browser.click(toggle(next_key, next_ids[-1]))
    await choose_metric(name)
    assert await drawn(key) == expected_ids
    assert await visible(key) == expected_ids[1:]
    await choose_metric(next_name)
    assert await visible(next_key) == next_ids[:-1]
    await choose_metric(name)
    await browser.click(toggle(key, identifier))
    report["tests"].append(
        "The metric dropdown replaces the complete source set and preserves visibility separately for each metric"
    )

    # Keep actual JavaScript objects alive across clicks; CDP returnByValue alone hides aliasing bugs.
    await browser.js(
        "window.__metricSnapshot=traceExplorer.getState();"
        "window.__metricSnapshotJson=JSON.stringify(window.__metricSnapshot)"
    )
    await browser.click(toggle(key, identifier))
    assert await browser.js("JSON.stringify(window.__metricSnapshot)===window.__metricSnapshotJson")
    await browser.js("traceExplorer.setState(window.__metricSnapshot)")
    assert await visible(key) == expected_ids
    await browser.js(
        "window.__metricRestoreInput=structuredClone(window.__metricSnapshot);"
        "traceExplorer.setState(window.__metricRestoreInput);"
        "window.__metricRestoreJson=JSON.stringify(window.__metricRestoreInput)"
    )
    await browser.click(toggle(key, identifier))
    assert await browser.js("JSON.stringify(window.__metricRestoreInput)===window.__metricRestoreJson")
    await browser.js("traceExplorer.setState(window.__metricSnapshot)")
    assert await visible(key) == expected_ids
    await browser.js(
        "delete window.__metricSnapshot;delete window.__metricSnapshotJson;"
        "delete window.__metricRestoreInput;delete window.__metricRestoreJson"
    )
    report["tests"].append("Saved JavaScript state snapshots and restore inputs stay immutable across legend clicks")

    # Hover a native uPlot canvas and compare its full precision tooltip to source samples.
    values_selector = scope + f" .ds-metric-legend-row[data-series-id={json.dumps(identifier)}] .ds-metric-value"
    before_hover = await browser.js(f"document.querySelector({json.dumps(values_selector)}).title")
    box = await browser.rectangle(scope + " .u-over")
    await browser.call("Input.dispatchMouseEvent", type="mouseMoved", x=box["x"] + box["width"] * 0.4, y=box["y"] + 40)
    await browser.wait(f"document.querySelector({json.dumps(values_selector)}).title !== {json.dumps(before_hover)}")
    hover = await browser.js(f"document.querySelector({json.dumps(values_selector)}).title")
    source_points = await browser.js(
        f"traceExplorer.queryMetrics({{name:{json.dumps(name)},points:true}})"
        f".find(s=>String(s.id)==={json.dumps(identifier)}).points"
    )
    sample_text = hover.removeprefix("Recorded sample at ")
    timestamp_text, value_text = sample_text.split(" elapsed seconds: ", 1)
    sample_time = float(timestamp_text)
    sample_value = float(value_text.split()[0])
    assert any(point[0] == sample_time and point[1] == sample_value for point in source_points), hover
    report["hover_sample"] = {"id": identifier, "time": sample_time, "value": sample_value}
    report["tests"].append("Native plot hover reports an exact recorded timestamp and value from the source series")
    await browser.screenshot("02-combined-hover.png")

    # Brushing the metric canvas drives every time-aligned panel through the public range.
    box = await browser.rectangle(scope + " .u-over")
    start = {"x": box["x"] + box["width"] * 0.25, "y": box["y"] + 30}
    end = {"x": box["x"] + box["width"] * 0.60, "y": start["y"]}
    await browser.call("Input.dispatchMouseEvent", type="mousePressed", button="left", clickCount=1, **start)
    await browser.call("Input.dispatchMouseEvent", type="mouseMoved", button="left", buttons=1, **end)
    await browser.call("Input.dispatchMouseEvent", type="mouseReleased", button="left", clickCount=1, **end)
    await browser.wait("traceExplorer.getState().from > 0")
    state = await browser.js("traceExplorer.getState()")
    assert abs(state["from"] - duration * 0.25) < duration * 0.015, state
    assert abs(state["to"] - duration * 0.60) < duration * 0.015, state
    assert await browser.js(
        "(()=>{const s=traceExplorer.getState();return traceExplorer.queryMetrics({points:true})"
        ".every(m=>m.points.every(p=>p[0]>=s.from&&p[0]<=s.to))})()"
    )
    report["brushed_range"] = {"from": state["from"], "to": state["to"]}
    report["tests"].append("Native chart brushing updates the shared range and all metric queries")

    await browser.click(toggle(key, identifier))
    saved = await browser.js("traceExplorer.getState()")
    url = await browser.js(
        "location.href.split('#')[0]+'#view='+encodeURIComponent(JSON.stringify(traceExplorer.getState()))"
    )
    await browser.call("Page.navigate", url="about:blank")
    await browser.wait("!window.traceExplorer")
    await browser.call("Page.navigate", url=url)
    await browser.wait("Boolean(window.traceExplorer?.ready && document.querySelector('.ds-metric-chart'))")
    restored = await browser.js("traceExplorer.getState()")
    assert restored["metricCharts"] == saved["metricCharts"]
    assert restored["from"] == saved["from"] and restored["to"] == saved["to"]
    assert restored["metric"] == saved["metric"]
    assert await drawn(key) == expected_ids
    assert await visible(key) == expected_ids[1:]
    await no_extra_controls()
    report["tests"].append("Opening a saved-view URL restores the metric, per-line visibility, and shared time range")
    await browser.rectangle(scope)
    await browser.screenshot("03-restored-combined-chart.png")
    await browser.call("Emulation.setDeviceMetricsOverride", width=390, height=1150, deviceScaleFactor=1, mobile=False)
    await asyncio.sleep(0.3)
    await browser.rectangle(scope)
    await browser.screenshot("04-narrow-combined-chart.png")
    assert await browser.js("document.documentElement.scrollWidth <= innerWidth + 1")
    assert await browser.js(
        f"(()=>{{const e=document.querySelector({json.dumps(scope)});return e.scrollWidth<=e.clientWidth+1}})()"
    )
    report["tests"].append("The combined chart and legend stay within a 390-pixel viewport")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    asyncio.run(run(arguments.html, arguments.out, arguments.port))
