# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the full metric catalog and lazy browser API against an offline report."""

from __future__ import annotations

import argparse
import asyncio
import gc
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import websockets
from dsight_browser_check import browser_targets
from dsight_metrics_check import Browser


def source_expectations(path: Path | None) -> dict[str, Any]:
    """Retain a few independent point witnesses per family, not millions of samples."""
    if path is None:
        return {}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    families: dict[str, Any] = {}
    for series in data["metrics"]:
        family = families.setdefault(series["name"], {"ids": [], "samples": 0})
        family["ids"].append(str(series["id"]))
        points = series["points"]
        family["samples"] += len(points)
        if points and "witness" not in family:
            family["witness"] = {
                "id": str(series["id"]),
                "unit": series["unit"],
                "labels": series["labels"],
                "points": [points[index] for index in sorted({0, len(points) // 2, len(points) - 1})],
            }
    result = {
        "families": families,
        "series": len(data["metrics"]),
        "catalog": data.get("metric_catalog", []),
        "source": {"path": str(path.resolve()), "sha256": digest.hexdigest()},
    }
    del data
    gc.collect()
    return result


async def check_catalog(browser: Browser, report: dict[str, Any], expected: dict[str, Any]) -> None:
    families = await browser.js("traceExplorer.listMetricFamilies()")
    metadata = await browser.js("traceExplorer.listMetricSeries()")
    names = {family["name"] for family in families}
    assert len(names) == len(families), "Catalog family names must be unique"
    if expected:
        assert len(metadata) == expected["series"]
        assert {str(series["id"]) for series in metadata} == {
            identifier for family in expected["families"].values() for identifier in family["ids"]
        }
        expected_names = {family["name"] for family in expected["catalog"]} or set(expected["families"])
        assert names == expected_names
    assert all("points" not in series for series in metadata)
    assert all(
        {"name", "component", "group", "title", "unit", "samples", "series_count"} <= family.keys()
        for family in families
    )
    components = {family["component"] for family in families}
    if len(families) > 100:
        assert components == {"Frontend", "Router", "Workers", "GPU", "Host"}, components
    report["catalog"] = {
        "families": len(families),
        "nonempty_families": sum(family["samples"] > 0 for family in families),
        "series": len(metadata),
        "components": sorted(components),
    }
    if report.get("expected_families") is not None:
        assert len(families) == report["expected_families"]
    if report.get("expected_nonempty") is not None:
        assert report["catalog"]["nonempty_families"] == report["expected_nonempty"]
    status = await browser.js("traceExplorer.metricDataStatus()")
    report["initial_data_status"] = status
    assert status["cachedPoints"] <= status["maxCachedPoints"]
    assert len(status["cachedFamilies"]) < len(families)
    options = await browser.js(
        "Array.from(document.querySelectorAll('#workerMetric option')).map(o=>({name:o.value,text:o.textContent}))"
    )
    assert {option["name"] for option in options} == names
    assert all(option["name"] in option["text"] for option in options)
    groups = await browser.js("Array.from(document.querySelectorAll('#workerMetric optgroup')).map(g=>g.label)")
    assert all(any(group.startswith(component + " / ") for group in groups) for component in components)
    report["tests"].append(
        "Catalog and grouped selector expose every captured family with raw names and synchronous series metadata"
    )

    def key(name: str) -> str:
        return json.dumps(["metric", name], separators=(",", ":"))

    def panel(name: str) -> str:
        return f".dsight-metric-panel[data-metric-key={json.dumps(key(name))}]"

    def toggle(name: str, identifier: str) -> str:
        return panel(name) + f" .ds-metric-legend-row[data-series-id={json.dumps(identifier)}] .ds-metric-legend-toggle"

    async def select(name: str) -> float:
        started = time.monotonic()
        await browser.js(
            f"(()=>{{const e=document.getElementById('workerMetric');e.value={json.dumps(name)};"
            "e.dispatchEvent(new Event('change',{bubbles:true}))})()"
        )
        await browser.js("traceExplorer.whenMetricsReady()")
        assert await browser.js("traceExplorer.getState().metric") == name
        assert await browser.js("document.querySelectorAll('.dsight-metric-panel').length") == 1
        assert await browser.js(f"Boolean(document.querySelector({json.dumps(panel(name) + ' .ds-metric-chart')}))")
        return time.monotonic() - started

    async def search(text: str) -> list[str]:
        await browser.click("#metricSearch")
        await browser.js("document.getElementById('metricSearch').select()")
        await browser.call("Input.insertText", text=text)
        return await browser.js("Array.from(document.querySelectorAll('#workerMetric option')).map(o=>o.value)")

    selected = await browser.js("traceExplorer.getState().metric")
    target = next(family for family in families if family["name"] != selected and family["samples"])
    results = await search(target["name"])
    assert target["name"] in results
    assert set(results) <= {
        family["name"]
        for family in families
        if target["name"].lower()
        in " ".join(str(family[field]) for field in ("name", "title", "component", "group")).lower()
    } | {selected}
    for component in sorted(components):
        results = await search(component.lower())
        component_names = {family["name"] for family in families if family["component"] == component}
        assert component_names <= set(results)
    results = await search("__no_matching_recorded_metric__")
    assert set(results) <= {selected}
    await browser.js(
        "document.getElementById('metricSearch').value='';document.getElementById('metricSearch').dispatchEvent(new Event('input',{bubbles:true}))"
    )
    assert (
        set(await browser.js("Array.from(document.querySelectorAll('#workerMetric option')).map(o=>o.value)")) == names
    )
    report["tests"].append(
        f"Metric search finds raw names and all {len(components)} available categories, handles no matches, and restores the complete list"
    )

    # Query a cold family before displaying it, proving the public API is complete beyond the cached chart.
    cold_name = target["name"]
    cold = await browser.js(
        "(async()=>{const result=await traceExplorer.queryMetrics({name:"
        + json.dumps(cold_name)
        + "});return {ids:result.map(s=>String(s.id)),samples:result.reduce((n,s)=>n+s.samples,0),hasPoints:result.some(s=>'points'in s)}})()"
    )
    expected_ids = {str(series["id"]) for series in metadata if series["name"] == cold_name}
    assert set(cold["ids"]) == expected_ids
    assert not cold["hasPoints"]
    assert cold["samples"] == target["samples"]
    assert await browser.js("traceExplorer.getState().metric") == selected
    report["tests"].append(
        "Async queryMetrics returns an undisplayed family's complete source set without changing the selected chart"
    )

    # Representative value semantics use normalized-source witnesses when a dataset is supplied.
    representatives = []
    for kind in ("histogram", "counter", "stored"):
        family = next((family for family in families if family.get("value_kind") == kind and family["samples"]), None)
        if family:
            representatives.append(family)
    if not representatives:
        representatives = [target]
    report["semantic_checks"] = []
    for family in representatives:
        name = family["name"]
        elapsed = await select(name)
        series = [series for series in metadata if series["name"] == name]
        assert all(series["unit"] == family["unit"] for series in series)
        if family.get("value_kind") == "histogram":
            assert family["unit"] == "observations", family
            assert any("bound" in label or label.endswith("le") for series in series for label in series["labels"])
        witness = expected.get("families", {}).get(name, {}).get("witness")
        if witness:
            check = await browser.js(
                "(async()=>{const witness=" + json.dumps(witness) + ";"
                "const series=(await traceExplorer.queryMetrics({name:"
                + json.dumps(name)
                + ",points:true})).find(s=>String(s.id)===witness.id);"
                "return {unit:series.unit,labels:series.labels,points:witness.points.map(w=>series.points.find(p=>"
                "p.length===w.length&&p.every((v,i)=>v===w[i])))}})()"
            )
            assert check == {field: witness[field] for field in ("unit", "labels", "points")}, name
        report["semantic_checks"].append(
            {"name": name, "kind": family.get("value_kind"), "unit": family["unit"], "load_seconds": elapsed}
        )
    report["tests"].append(
        f"Representative {', '.join(str(family.get('value_kind', 'stored')) for family in representatives)} metrics retain units, labels, and exact source sample witnesses"
    )

    empty = next((family for family in families if family["samples"] == 0), None)
    if empty:
        await select(empty["name"])
        text = await browser.js(f"document.querySelector({json.dumps(panel(empty['name']))}).textContent")
        assert "no" in text.lower() and ("sample" in text.lower() or "recorded" in text.lower()), text
        counts = await browser.js(
            "(async()=>{const s=await traceExplorer.queryMetrics({name:"
            + json.dumps(empty["name"])
            + ",points:true});return {samples:s.reduce((n,x)=>n+x.samples,0),points:s.reduce((n,x)=>n+x.points.length,0)}})()"
        )
        assert counts == {"samples": 0, "points": 0}
        report["tests"].append(
            "Captured families outside the trace window remain selectable with explicit empty coverage and no fabricated samples"
        )

    largest = max(families, key=lambda family: family["series_count"])
    largest_status = await browser.js("traceExplorer.metricDataStatus()")
    heap_before = await browser.call("Runtime.getHeapUsage")
    largest_seconds = await select(largest["name"])
    heap_after = await browser.call("Runtime.getHeapUsage")
    largest_ids = [str(series["id"]) for series in metadata if series["name"] == largest["name"]]
    shown = await browser.js(
        f"JSON.parse(document.querySelector({json.dumps(panel(largest['name']) + ' .ds-metric-chart')}).dataset.drawnIds)"
    )
    assert set(shown) == set(largest_ids)
    assert len(shown) == largest["series_count"]
    assert await browser.js(
        f"document.querySelectorAll({json.dumps(panel(largest['name']) + ' .ds-metric-legend-row')}).length"
    ) == len(largest_ids)
    for identifier in (largest_ids[0], largest_ids[-1]):
        await browser.click(toggle(largest["name"], identifier))
        assert (
            await browser.js(
                f"document.querySelector({json.dumps(toggle(largest['name'], identifier))}).getAttribute('aria-pressed')"
            )
            == "false"
        )
        await browser.click(toggle(largest["name"], identifier))
    report["largest_family"] = {
        "name": largest["name"],
        "series": len(largest_ids),
        "load_seconds": largest_seconds,
        "was_cached": largest["name"] in largest_status["cachedFamilies"],
        "heap_before": heap_before,
        "heap_after": heap_after,
    }
    await browser.rectangle(panel(largest["name"]))
    await browser.screenshot("01-largest-family.png")
    report["tests"].append(
        "The largest family exposes every source with no series cap; first and last legend entries both control their lines"
    )

    # Fire multiple changes in one JS turn, then explicitly wait for the superseded query too.
    first, last = sorted(
        (family for family in families if family["samples"]), key=lambda family: family["samples"], reverse=True
    )[:2]
    await browser.js(
        "(()=>{for(const name of "
        + json.dumps([first["name"], last["name"]])
        + "){const e=document.getElementById('workerMetric');e.value=name;e.dispatchEvent(new Event('change',{bubbles:true}))}})()"
    )
    await browser.js("traceExplorer.whenMetricsReady()")
    await browser.js("traceExplorer.queryMetrics({name:" + json.dumps(first["name"]) + "})")
    assert await browser.js("traceExplorer.getState().metric") == last["name"]
    assert await browser.js("document.querySelectorAll('.dsight-metric-panel').length") == 1
    assert await browser.js(f"Boolean(document.querySelector({json.dumps(panel(last['name']) + ' .ds-metric-chart')}))")
    report["tests"].append("Rapid metric selection cannot let a superseded lazy load replace the final chosen chart")

    duration = report["description"]["meta"]["duration"]
    lo, hi = duration * 0.2, duration * 0.4
    await browser.js(f"traceExplorer.selectRange({lo},{hi})")
    bounded = await browser.js(
        "(async()=>{const s=await traceExplorer.queryMetrics({name:"
        + json.dumps(largest["name"])
        + ",points:true});return {count:s.length,samples:s.reduce((n,x)=>n+x.samples,0),valid:s.every(x=>x.points.every(p=>p[0]>="
        + str(lo)
        + "&&p[0]<="
        + str(hi)
        + "))}})()"
    )
    assert bounded["valid"] and bounded["count"] == largest["series_count"]
    # exportSelection must capture its original view despite a range change while loading other families.
    export = await browser.js(
        "(async()=>{const before=traceExplorer.getState();const promise=traceExplorer.exportSelection();"
        "traceExplorer.selectRange(0," + str(duration) + ");const result=await promise;"
        "return {before,view:result.view,count:result.metrics.length,names:[...new Set(result.metrics.map(s=>s.name))],"
        "boundsValid:result.metrics.every(s=>!s.samples||(s.first_time>=before.from&&s.last_time<=before.to))}})()"
    )
    assert export["view"] == export["before"]
    assert export["count"] == len(metadata)
    assert set(export["names"]) == {series["name"] for series in metadata}
    assert export["boundsValid"]
    report["tests"].append(
        "Async metric queries clip source samples, and exportSelection preserves its captured view while returning all source series"
    )

    chosen = representatives[0]["name"]
    await select(chosen)
    chosen_id = next(str(series["id"]) for series in metadata if series["name"] == chosen)
    await browser.click(toggle(chosen, chosen_id))
    saved = await browser.js("traceExplorer.getState()")
    url = await browser.js(
        "location.href.split('#')[0]+'#view='+encodeURIComponent(JSON.stringify(traceExplorer.getState()))"
    )
    await browser.call("Page.navigate", url="about:blank")
    await browser.wait("!window.traceExplorer")
    await browser.call("Page.navigate", url=url)
    await browser.wait("Boolean(window.traceExplorer?.ready)")
    await browser.js("traceExplorer.whenMetricsReady()")
    restored = await browser.js("traceExplorer.getState()")
    assert restored["metric"] == chosen and restored["metricCharts"] == saved["metricCharts"]
    assert (
        await browser.js(
            f"document.querySelector({json.dumps(toggle(chosen, chosen_id))}).getAttribute('aria-pressed')"
        )
        == "false"
    )
    await browser.rectangle(panel(chosen))
    await browser.screenshot("02-restored-catalog-metric.png")
    await browser.call("Emulation.setDeviceMetricsOverride", width=390, height=1150, deviceScaleFactor=1, mobile=False)
    await asyncio.sleep(0.2)
    await browser.rectangle(panel(chosen))
    await browser.screenshot("03-narrow-catalog.png")
    assert await browser.js("document.documentElement.scrollWidth <= innerWidth+1")
    report["final_data_status"] = await browser.js("traceExplorer.metricDataStatus()")
    report["tests"].append(
        "Saved catalog selections lazy-load with their visibility intact and remain usable in a narrow viewport"
    )


async def run(args: argparse.Namespace) -> None:
    expected = source_expectations(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    targets = await asyncio.to_thread(browser_targets, args.port)
    page = next(target for target in targets if target["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=100_000_000) as socket:
        browser = Browser(socket, args.out)
        report: dict[str, Any] = {
            "html": str(args.html.resolve()),
            "html_sha256": hashlib.sha256(args.html.read_bytes()).hexdigest(),
            "expected_families": args.expected_families,
            "expected_nonempty": args.expected_nonempty,
            "source_dataset": expected.get("source"),
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
            await check_catalog(browser, report, expected)
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
            report["tests"].append("Catalog loading and every interaction stay offline with no runtime exceptions")
        except Exception as error:
            report["failure"] = str(error)
            await browser.screenshot("failure.png")
            raise
        finally:
            (args.out / "catalog-browser-report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {"catalog": report["catalog"], "tests": report["tests"], "largest_family": report["largest_family"]},
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-families", type=int)
    parser.add_argument("--expected-nonempty", type=int)
    asyncio.run(run(parser.parse_args()))
