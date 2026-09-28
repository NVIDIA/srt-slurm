# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify pinned metric comparisons on an offline DSight report through Chrome."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import websockets
from dsight_browser_check import browser_targets
from dsight_metrics_check import Browser


def card(name: str) -> str:
    return f".metric-card[data-metric-name={json.dumps(name)}]"


def chart_key(name: str) -> str:
    return json.dumps(["metric", name], separators=(",", ":"))


def legend(name: str, identifier: str) -> str:
    return card(name) + f" .ds-metric-legend-row[data-series-id={json.dumps(identifier)}] .ds-metric-legend-toggle"


async def check_pinning(browser: Browser, report: dict[str, Any]) -> None:
    """Compare complete source sets while preserving ordered pins and independent visibility."""
    families = await browser.js("traceExplorer.listMetricFamilies()")
    metadata = await browser.js("traceExplorer.listMetricSeries()")
    duration = report["description"]["meta"]["duration"]
    initial = await browser.js("traceExplorer.getState().metric")
    # Keep point queries small: this suite tests pin composition, not full catalog import again.
    candidates = sorted(
        (family for family in families if family["samples"] and family["series_count"]),
        key=lambda family: (family["series_count"], family["samples"], family["name"]),
    )
    preferred = [initial, "trtllm_num_requests_running", "trtllm_num_requests_waiting"]
    names = list(dict.fromkeys(name for name in preferred if any(f["name"] == name for f in candidates)))
    names.extend(f["name"] for f in candidates if f["name"] not in names)
    assert len(names) >= 4, "Use a report with at least four sampled metric families"
    first, second, third, cold = names[:4]
    ids = {
        name: sorted(str(series["id"]) for series in metadata if series["name"] == name)
        for name in (first, second, third, cold)
    }
    report["families"] = {name: len(values) for name, values in ids.items()}

    async def ready() -> None:
        await browser.js("traceExplorer.whenMetricsReady()")
        await browser.wait("!document.querySelector('.ds-metric-loading')")

    async def state(update: dict[str, Any]) -> None:
        await browser.js("traceExplorer.setState(" + json.dumps(update) + ")")
        await ready()

    async def choose(name: str) -> None:
        await browser.js(
            "(()=>{const e=document.getElementById('workerMetric');e.value="
            + json.dumps(name)
            + ";e.dispatchEvent(new Event('change',{bubbles:true}))})()"
        )
        await ready()

    async def assert_cards(expected: list[str], pinned: list[str]) -> None:
        await ready()
        actual = await browser.js("Array.from(document.querySelectorAll('.metric-card'),e=>e.dataset.metricName)")
        assert actual == expected, {"actual": actual, "expected": expected}
        assert await browser.js("traceExplorer.getState().pinnedMetrics") == pinned
        assert await browser.js("document.querySelectorAll('.dsight-metric-panel').length") == len(expected)
        for name in expected:
            if name not in ids:
                continue
            details = await browser.js(
                "(()=>{const e=document.querySelector("
                + json.dumps(card(name))
                + ");return {key:e.querySelector('[data-metric-key]').dataset.metricKey,"
                "ids:JSON.parse(e.querySelector('.ds-metric-chart').dataset.drawnIds)}})()"
            )
            assert details["key"] == chart_key(name)
            assert sorted(details["ids"]) == ids[name]

    async def hidden(name: str) -> list[str]:
        await ready()
        return await browser.js(
            f"Array.from(document.querySelectorAll({json.dumps(card(name) + ' .ds-metric-legend-row')}))"
            ".filter(e=>e.querySelector('.ds-metric-legend-toggle').getAttribute('aria-pressed')==='false')"
            ".map(e=>e.dataset.seriesId).sort()"
        )

    await state(
        {"metric": first, "pinnedMetrics": [], "metricCharts": {}, "from": 0, "to": duration, "hardware": False}
    )
    # Observe the public mount boundary without changing chart behavior or its range callbacks.
    await browser.js(
        "window.__pinChartLibrary=DSightMetricCharts;window.__pinMountRanges={};"
        "window.DSightMetricCharts={mount(host,options){"
        "__pinMountRanges[host.dataset.metricKey]=[options.from,options.to];"
        "return __pinChartLibrary.mount(host,options)}}"
    )
    await assert_cards([first], [])
    await browser.click("#pinMetric")
    await assert_cards([first], [first])
    await choose(second)
    await assert_cards([first, second], [first])
    report["tests"].append("Pinning retains a complete metric chart when the selector changes to another family")

    await browser.click("#pinMetric")
    await choose(third)
    await assert_cards([first, second, third], [first, second])
    await choose(first)
    await assert_cards([first, second], [first, second])
    await browser.click("#pinMetric")
    await assert_cards([second, first], [second])
    await browser.click("#pinMetric")
    await assert_cards([second, first], [second, first])
    await browser.click(card(second) + " [data-action='unpin-metric']")
    await assert_cards([first], [first])
    await choose(second)
    await browser.click("#pinMetric")
    await choose(third)
    await assert_cards([first, second, third], [first, second])
    report["tests"].append(
        "Multiple pins keep insertion order; selecting a pin does not duplicate it, and both unpin controls remove it"
    )

    await browser.click(legend(first, ids[first][0]))
    await browser.click(legend(second, ids[second][-1]))
    assert await hidden(first) == [ids[first][0]]
    assert await hidden(second) == [ids[second][-1]]
    assert await hidden(third) == []
    await choose(first)
    await choose(third)
    assert await hidden(first) == [ids[first][0]]
    assert await hidden(second) == [ids[second][-1]]
    assert await hidden(third) == []
    report["tests"].append("Every pinned metric preserves its own legend visibility while other charts change")
    await browser.rectangle(card(first))
    await browser.screenshot("01-pinned-comparison.png")

    # Keep real objects in the page: CDP returnByValue alone would hide shallow-copy errors.
    await browser.js(
        "window.__pinSnapshot=traceExplorer.getState();window.__pinSnapshotJson=JSON.stringify(__pinSnapshot);"
        "window.__pinInput=structuredClone(__pinSnapshot);traceExplorer.setState(__pinInput);"
        "window.__pinInputJson=JSON.stringify(__pinInput)"
    )
    await ready()
    await browser.click("#pinMetric")
    await assert_cards([first, second, third], [first, second, third])
    assert await browser.js("JSON.stringify(__pinSnapshot)===__pinSnapshotJson")
    assert await browser.js("JSON.stringify(__pinInput)===__pinInputJson")
    await browser.js("traceExplorer.setState(__pinSnapshot)")
    await assert_cards([first, second, third], [first, second])
    await browser.js("__pinSnapshot.pinnedMetrics.push('caller-only');__pinInput.pinnedMetrics.length=0")
    await assert_cards([first, second, third], [first, second])
    await browser.js("delete window.__pinSnapshot;delete window.__pinSnapshotJson;delete window.__pinInput")
    report["tests"].append("Exported state snapshots and restored input arrays cannot mutate live pins or vice versa")

    await state({"pinnedMetrics": [second, first, second, "__missing_metric__", None, 42, {}, first]})
    await assert_cards([second, first, third], [second, first])
    await state({"pinnedMetrics": "not-an-array"})
    await assert_cards([second, first, third], [second, first])
    await state({"pinnedMetrics": [first, second]})
    report["tests"].append(
        "Restored pins discard unknown or malformed entries and duplicate names while preserving order"
    )

    empty = next((family["name"] for family in families if not family["samples"]), None)
    if empty:
        await state({"pinnedMetrics": [first, empty], "metric": second})
        await assert_cards([first, empty, second], [first, empty])
        assert await browser.js(
            f"/no .*samples?|no recorded/i.test(document.querySelector({json.dumps(card(empty))}).textContent)"
        )
        report["tests"].append(
            "A captured family without in-window samples can stay pinned with explicit empty coverage"
        )
    await state({"pinnedMetrics": [first, second], "metric": third, "from": 0, "to": duration})

    await browser.rectangle(card(third) + " .u-over")
    scroll_before = await browser.js("document.getElementById('tracks').scrollTop")
    assert scroll_before > 0
    await state({"pinnedMetrics": [first, second, third]})
    assert abs(await browser.js("document.getElementById('tracks').scrollTop") - scroll_before) <= 2
    # Activate the lower card's control using Enter; focus must survive replacing its DOM.
    unpin_selector = card(third) + " [data-action='unpin-metric']"
    await browser.rectangle(unpin_selector)
    await browser.js(f"document.querySelector({json.dumps(unpin_selector)}).focus({{preventScroll:true}})")
    scroll_before = await browser.js("document.getElementById('tracks').scrollTop")
    await browser.call(
        "Input.dispatchKeyEvent",
        type="keyDown",
        key="Enter",
        code="Enter",
        windowsVirtualKeyCode=13,
        text="\r",
        unmodifiedText="\r",
    )
    await browser.call("Input.dispatchKeyEvent", type="keyUp", key="Enter", code="Enter", windowsVirtualKeyCode=13)
    await assert_cards([first, second, third], [first, second])
    focus = await browser.js(
        "(()=>{const e=document.activeElement,r=e.getBoundingClientRect(),"
        "t=document.getElementById('tracks').getBoundingClientRect();return {connected:e.isConnected,"
        "name:e.closest('.metric-card')?.dataset.metricName,"
        "visible:r.bottom>Math.max(0,t.top)&&r.top<Math.min(innerHeight,t.bottom)}})()"
    )
    assert focus == {"connected": True, "name": third, "visible": True}, focus
    report["tests"].append("Lower-card rerenders retain scroll position, and keyboard unpin retains usable focus")

    # A native brush in one mounted uPlot must update every peer without stale callbacks.
    rectangle = await browser.rectangle(card(second) + " .u-over")
    start = {"x": rectangle["x"] + rectangle["width"] * 0.2, "y": rectangle["y"] + 30}
    end = {"x": rectangle["x"] + rectangle["width"] * 0.65, "y": start["y"]}
    await browser.call("Input.dispatchMouseEvent", type="mousePressed", button="left", clickCount=1, **start)
    await browser.call("Input.dispatchMouseEvent", type="mouseMoved", button="left", buttons=1, **end)
    await browser.call("Input.dispatchMouseEvent", type="mouseReleased", button="left", clickCount=1, **end)
    await browser.wait("traceExplorer.getState().from > 0")
    await assert_cards([first, second, third], [first, second])
    selected = await browser.js("traceExplorer.getState()")
    assert abs(selected["from"] - duration * 0.2) < duration * 0.02, selected
    assert abs(selected["to"] - duration * 0.65) < duration * 0.02, selected
    range_checks = await browser.js(
        "(async()=>{const s=traceExplorer.getState();return Promise.all("
        + json.dumps([first, second, third])
        + ".map(async name=>{const series=await traceExplorer.queryMetrics({name,points:true});"
        "return {name,count:series.length,samples:series.reduce((n,x)=>n+x.points.length,0),"
        "valid:series.every(x=>x.points.every(p=>p[0]>=s.from&&p[0]<=s.to))}}))})()"
    )
    assert all(item["valid"] and item["samples"] and item["count"] == len(ids[item["name"]]) for item in range_checks)
    mounted_ranges = await browser.js("window.__pinMountRanges")
    for name in (first, second, third):
        assert mounted_ranges[chart_key(name)] == [selected["from"], selected["to"]]
    report["brushed_range"] = {key: selected[key] for key in ("from", "to")}
    report["tests"].append("Brushing a pinned chart updates the shared range, peer axes, and bounded source queries")

    # One export checks pins in the asynchronous snapshot; return only the view, not all metric summaries.
    exported = await browser.js(
        "(async()=>{const before=traceExplorer.getState();const pending=traceExplorer.exportSelection();"
        "traceExplorer.setState({pinnedMetrics:[]});const result=await pending;"
        "return {before,view:result.view,metrics:result.metrics.length}})()"
    )
    assert exported["before"] == exported["view"]
    assert exported["view"]["pinnedMetrics"] == [first, second]
    assert exported["metrics"] == len(metadata)
    await state(exported["before"])
    report["tests"].append(
        "Selection export captures the original ordered pins even when live pins change while loading"
    )

    failure = await browser.js(
        "(async()=>{const original=DSightMetricCharts;const target="
        + json.dumps(chart_key(first))
        + ";window.DSightMetricCharts={mount(host,options){if(host.dataset.metricKey===target)"
        "throw Error('Injected pinning chart failure');return original.mount(host,options)}};"
        "try{traceExplorer.setState({});try{await traceExplorer.whenMetricsReady();return {rejected:false}}"
        "catch(error){return {rejected:true,error:error.message,remaining:"
        "Array.from(document.querySelectorAll('.metric-card .ds-metric-chart'),"
        "e=>e.closest('.metric-card').dataset.metricName),loading:document.querySelectorAll('.ds-metric-loading').length}}}"
        "finally{window.DSightMetricCharts=original}})()"
    )
    assert failure["rejected"] and failure["error"] == "Injected pinning chart failure", failure
    assert failure["remaining"] == [second, third] and failure["loading"] == 0, failure
    await browser.js("document.getElementById('error').textContent=''")
    await state({})
    await assert_cards([first, second, third], [first, second])
    report["tests"].append(
        "A failed pinned chart rejects readiness after its peers settle, and a later render recovers"
    )

    saved = await browser.js("traceExplorer.getState()")
    url = await browser.js(
        "location.href.split('#')[0]+'#view='+encodeURIComponent(JSON.stringify(traceExplorer.getState()))"
    )
    await browser.call("Page.navigate", url="about:blank")
    await browser.wait("!window.traceExplorer")
    await browser.call("Page.navigate", url=url)
    await browser.wait("Boolean(window.traceExplorer?.ready)")
    await assert_cards([first, second, third], [first, second])
    restored = await browser.js("traceExplorer.getState()")
    for key in ("pinnedMetrics", "metric", "metricCharts", "from", "to"):
        assert restored[key] == saved[key], key
    assert await hidden(first) == [ids[first][0]]
    assert await hidden(second) == [ids[second][-1]]
    report["tests"].append(
        "A saved-view URL restores ordered pins, current metric, independent legends, and the common range"
    )

    # Trigger superseded asynchronous generations before any of their loads can complete.
    updates = [
        {"pinnedMetrics": [cold, first], "metric": second},
        {"pinnedMetrics": [third], "metric": first},
        {"pinnedMetrics": [second, first], "metric": third, "from": duration * 0.1, "to": duration * 0.4},
    ]
    await browser.js("(()=>{for(const update of " + json.dumps(updates) + ")traceExplorer.setState(update)})()")
    await assert_cards([second, first, third], [second, first])
    await browser.js("traceExplorer.queryMetrics({name:" + json.dumps(cold) + "})")
    await assert_cards([second, first, third], [second, first])
    assert await hidden(first) == [ids[first][0]]
    assert await hidden(second) == [ids[second][-1]]
    report["tests"].append(
        "Rapid pin, metric, and range updates cannot resurrect removed cards or replace the final view"
    )

    await browser.rectangle(card(second))
    await browser.screenshot("02-restored-pins.png")
    await browser.call("Emulation.setDeviceMetricsOverride", width=390, height=1150, deviceScaleFactor=1, mobile=False)
    await asyncio.sleep(0.2)
    await browser.rectangle(card(second))
    await browser.screenshot("03-narrow-pins.png")
    assert await browser.js("document.documentElement.scrollWidth<=innerWidth+1")
    assert await browser.js(
        "Array.from(document.querySelectorAll('.metric-card')).every(e=>e.scrollWidth<=e.clientWidth+1)"
    )
    report["tests"].append("Stacked pinned charts and their controls remain within a 390-pixel viewport")


async def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    targets = await asyncio.to_thread(browser_targets, args.port)
    page = next(target for target in targets if target["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=100_000_000) as socket:
        browser = Browser(socket, args.out)
        report: dict[str, Any] = {
            "html": str(args.html.resolve()),
            "html_sha256": hashlib.sha256(args.html.read_bytes()).hexdigest(),
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
            await browser.call("Page.navigate", url=args.html.resolve().as_uri())
            await browser.wait("Boolean(window.traceExplorer?.ready)")
            await browser.js("traceExplorer.whenMetricsReady()")
            report["load_seconds"] = time.monotonic() - started
            report["description"] = await browser.js("traceExplorer.describe()")
            await check_pinning(browser, report)
            report["runtime_errors"] = [e for e in browser.events if e.get("method") == "Runtime.exceptionThrown"]
            report["external_requests"] = [
                event["params"]["request"]["url"]
                for event in browser.events
                if event.get("method") == "Network.requestWillBeSent"
                and event["params"]["request"]["url"].startswith(("http://", "https://"))
            ]
            assert not report["runtime_errors"], report["runtime_errors"]
            assert not report["external_requests"], report["external_requests"]
            report["tests"].append("All pinning interactions complete offline without runtime exceptions")
        except Exception as error:
            report["failure"] = str(error)
            await browser.screenshot("failure.png")
            raise
        finally:
            (args.out / "pinning-browser-report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"load_seconds": report["load_seconds"], "tests": report["tests"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    asyncio.run(run(parser.parse_args()))
