# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the offline artifact through Chrome DevTools and capture screenshots."""

import argparse
import asyncio
import base64
import json
import time
import urllib.request
from pathlib import Path

import websockets
from dsight_drag_check import check_client_drag


def browser_targets(port):
    with urllib.request.urlopen(f"http://localhost:{port}/json", timeout=5) as response:
        return json.load(response)


async def run(html, output, port, request_id):
    output.mkdir(parents=True, exist_ok=True)
    targets = await asyncio.to_thread(browser_targets, port)
    page = next(p for p in targets if p["type"] == "page")
    events = []
    seq = 0
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=100_000_000) as ws:

        async def call(method, **params):
            nonlocal seq
            seq += 1
            ident = seq
            await ws.send(json.dumps({"id": ident, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == ident:
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    return msg.get("result", {})
                events.append(msg)

        async def js(expression):
            r = await call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
            if r.get("exceptionDetails"):
                raise AssertionError(r["exceptionDetails"])
            return r["result"].get("value")

        async def click(selector):
            rect = await js(
                f"(()=>{{const e=document.querySelector({json.dumps(selector)});if(!e)throw Error('Missing control');e.scrollIntoView({{block:'nearest'}});const r=e.getBoundingClientRect();return {{x:r.x+r.width/2,y:r.y+r.height/2}}}})()"
            )
            await call("Input.dispatchMouseEvent", type="mousePressed", button="left", clickCount=1, **rect)
            await call("Input.dispatchMouseEvent", type="mouseReleased", button="left", clickCount=1, **rect)

        async def screenshot(name):
            shot = await call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
            (output / name).write_bytes(base64.b64decode(shot["data"]))

        await call("Page.enable")
        await call("Runtime.enable")
        await call("Network.enable")
        await call("Emulation.setDeviceMetricsOverride", width=1600, height=1200, deviceScaleFactor=1, mobile=False)
        await call("Page.navigate", url=html.resolve().as_uri())
        start = time.monotonic()
        for _ in range(150):
            if await js("Boolean(window.traceExplorer?.ready && document.querySelector('.track'))"):
                break
            error = await js("window.traceExplorerError || null")
            if error:
                raise AssertionError(error)
            await asyncio.sleep(0.2)
        else:
            raise AssertionError("Explorer did not initialize")
        report = {
            "load_seconds": time.monotonic() - start,
            "description": await js("traceExplorer.describe()"),
            "tests": [],
        }
        assert not await js("document.getElementById('error').textContent")
        await screenshot("01-overview.png")
        report["initial_state"] = await js("traceExplorer.getState()")
        await js("document.getElementById('rangeFrom').value='1';document.getElementById('rangeTo').value='3'")
        await click("#applyRange")
        state = await js("traceExplorer.getState()")
        assert (state["from"], state["to"]) == (1, 3)
        report["tests"].append("Human From/To inputs update the shared selection")
        await click("#fullRun")
        assert (await js("traceExplorer.getState().from")) == 0
        # Brush gesture uses native pointer input, not a call to selectRange.
        box = await js(
            "(()=>{const r=document.getElementById('overview').getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()"
        )
        a = {"x": box["x"] + box["w"] * 0.2, "y": box["y"] + box["h"] * 0.5}
        b = {"x": box["x"] + box["w"] * 0.4, "y": a["y"]}
        await call("Input.dispatchMouseEvent", type="mousePressed", button="left", clickCount=1, **a)
        await call("Input.dispatchMouseEvent", type="mouseMoved", button="left", buttons=1, **b)
        await call("Input.dispatchMouseEvent", type="mouseReleased", button="left", clickCount=1, **b)
        state = await js("traceExplorer.getState()")
        duration = report["description"]["meta"]["duration"]
        assert abs(state["from"] - duration * 0.2) < 0.01 and abs(state["to"] - duration * 0.4) < 0.01
        report["tests"].append("Human overview brush updates the shared selection")
        # Select a real request through the same public action the timeline uses.
        if request_id is None:
            request_id = await js(
                "traceExplorer.queryRequests({from:0,to:traceExplorer.describe().meta.duration,limit:1000}).items.find(r=>r.span_count&&r.first!==null)?.id"
            )
        assert request_id, "Browser drill-down checks need one joined request with measured TTFT"
        r = await js("traceExplorer.getRequest(" + json.dumps(request_id) + ")")
        await js(f"traceExplorer.selectRequest({json.dumps(r['id'])},{{fit:true}})")
        await click("#expandTTFT")
        assert r["id"] in (await js("traceExplorer.getState().expandedRequests"))
        assert await js("document.querySelectorAll('[data-span]').length") > 0
        await click("#fitTTFT")
        assert await js("document.querySelectorAll('[data-span]').length") > 0
        await screenshot("02-ttft.png")
        report["tests"].append("Human TTFT expansion reveals joined lifecycle intervals")
        report["selected_request"] = {k: r[k] for k in ("id", "session", "server_ids", "workers", "ttft_ms")}
        layout = await js("traceExplorer.getLifecycle(" + json.dumps(r["id"]) + ")")
        report["stage_layout_audit"] = await js("""(()=>{
          let checked=0;const failures=[];const end=traceExplorer.describe().meta.duration;
          const total=traceExplorer.queryRequests({from:0,to:end,limit:0}).total;
          for(let offset=0;offset<total;offset+=1000){for(const r of traceExplorer.queryRequests({from:0,to:end,offset,limit:1000}).items){
            const detail=traceExplorer.getRequest(r.id),model=traceExplorer.getLifecycle(r.id),ids=model.stages.map(s=>s.id),raw=new Map(detail.spans.map(s=>[s.id,s]));
            if(new Set(ids).size!==ids.length||model.rows.length!==ids.length)failures.push(r.id+": duplicate stage or missing row");
            for(let i=0;i<model.rows.length;i++)if(JSON.stringify(model.rows[i])!==JSON.stringify(ids.slice(0,i+1)))failures.push(r.id+": incorrect cumulative prefix");
            for(let i=1;i<model.stages.length;i++)if(model.stages[i-1].end!==model.stages[i].start)failures.push(r.id+": progress gap/overlap");
            const elapsed=model.stages.reduce((n,s)=>n+s.end-s.start,0);
            if(Math.abs(elapsed-(r.end-r.start))>1e-7)failures.push(r.id+": wrong elapsed partition");
            for(const s of model.activities)if(s.start!==raw.get(s.id).start||s.end!==raw.get(s.id).end)failures.push(r.id+": changed raw timing");
            checked++;
          }}return {requests:checked,failures};})()""")
        assert not report["stage_layout_audit"]["failures"]
        report["tests"].append("Every request has disjoint cumulative milestones and unchanged inclusive source spans")
        await js("traceExplorer.setState({search:" + json.dumps(r["id"]) + ',nsys:false,tab:"request"})')
        await click("#fitRequest")
        rendered = await js("""Array.from(document.querySelectorAll("[data-stage-row]")).map(e=>({
          index:+e.dataset.stageRow,
          ids:Array.from(e.querySelectorAll("[data-stage-span]")).map(b=>b.dataset.stageSpan),
          current:Array.from(e.querySelectorAll("[data-current-stage=true]")).map(b=>b.dataset.stageSpan)
        }))""")
        assert len(rendered) == len(layout["stages"])
        for i, row in enumerate(rendered):
            assert row["index"] == i
            assert row["ids"] == [s["id"] for s in layout["stages"][: i + 1]]
            assert row["current"] == [layout["stages"][i]["id"]]
        await click('[data-stage-row="4"] .stage-row-label')
        assert await js("traceExplorer.getState().span") == layout["stages"][4]["id"]
        assert "between milestones" in await js("document.querySelector('.phase-focus').textContent")
        report["tests"].append("Human expansion appends one milestone per row and drills into provenance")
        await js("document.getElementById('tracks').scrollTop=0;document.getElementById('inspectorBody').scrollTop=0")
        await screenshot("11-cumulative-lifecycle.png")
        await js("traceExplorer.setState({tab:'iterations'})")
        iterations = await js(
            "traceExplorer.queryIterations({worker:traceExplorer.getRequest(traceExplorer.getState().request).workers[0],rank:0,limit:3})"
        )
        assert iterations["items"] or iterations["unaligned_rows"]
        await screenshot("12-iterations.png")
        report["tests"].append("Iteration context is queryable without claiming per-request batch ownership")
        # Exercise every imported rank via query, and a concrete worker through the UI.
        profiles = await js("traceExplorer.listProfiles()")
        coverage = await js(
            "traceExplorer.listProfiles().map(p=>({id:p.id,worker:p.worker,rank:p.rank,count:traceExplorer.queryNsys({profile:p.id,from:0,to:traceExplorer.describe().meta.duration,limit:1}).total}))"
        )
        report["profile_queries"] = coverage
        assert all(p["count"] > 0 for p in coverage)
        target = next((p for p in profiles if p["worker"] in r["workers"]), profiles[0])
        await js(f"traceExplorer.inspectNsys({{profile:{target['id']}}})")
        assert await js("document.querySelectorAll('.nsys-track').length") > 0
        await screenshot("03-nsight.png")
        report["tests"].append("Agent API selects worker/rank and returns exact Nsight intervals")
        ns = await js("traceExplorer.queryNsys({limit:5})")
        assert ns["total"] >= len(ns["items"])
        report["nsight_query_sample"] = ns
        front = next((p for p in profiles if p["worker"] == "frontend"), None)
        if front:
            await js(f"traceExplorer.inspectNsys({{profile:{front['id']}}})")
            cpu = await js("traceExplorer.queryCpu({limit:6})")
            assert cpu["available"] and cpu["total_samples"] > 0
            report["cpu_query_sample"] = cpu
            await screenshot("06-frontend-cpu.png")
            report["tests"].append("Frontend CPU sample hotspots follow the same selected window")
        await js("traceExplorer.setState({tab:'request'})")
        phase = next(s for s in r["spans"] if s["name"] == "worker.operation.prefill")
        await js(f"traceExplorer.selectSpan({json.dumps(phase['id'])})")
        await click("#inspectSpan")
        phase_state = await js("traceExplorer.getState()")
        assert phase_state["nsys"] and phase_state["from"] <= phase["start"] < phase["end"] <= phase_state["to"]
        report["tests"].append("Human phase-to-Nsight action selects the mapped worker and phase window")
        await click("#compareNsys")
        assert await js("document.querySelectorAll('[data-nsys-heading]').length") == len(
            {p["worker"] for p in profiles if p["worker"] in ["frontend", *r["workers"]]}
        )
        report["tests"].append("Frontend, prefill, and decode Nsight tracks compare on one shared range")
        await screenshot("07-cross-process-nsight.png")
        # Refusal cases protect agents from silent clamping / bogus IDs.
        rejects = await js(
            "(()=>{let n=0;for(const f of [()=>traceExplorer.selectRange(10,1),()=>traceExplorer.selectRange(NaN,3),()=>traceExplorer.selectRequest('not-a-request'),()=>traceExplorer.inspectNsys({worker:'missing'})]){try{f()}catch(e){n++}}return n})()"
        )
        assert rejects == 4
        report["tests"].append("Invalid ranges and missing identities fail explicitly")
        metrics = await js("traceExplorer.queryMetrics({points:true})")
        state = await js("traceExplorer.getState()")
        assert all(state["from"] <= p[0] <= state["to"] for m in metrics for p in m.get("points", []))
        report["tests"].append("Metric API preserves labels and respects the selected range")
        # Narrow viewport must not create a horizontally scrolling page.
        await call("Emulation.setDeviceMetricsOverride", width=390, height=1150, deviceScaleFactor=1, mobile=False)
        await screenshot("04-narrow.png")
        assert await js("document.documentElement.scrollWidth <= innerWidth+1")
        report["tests"].append("Narrow layout stays within the viewport")
        await call("Emulation.setDeviceMetricsOverride", width=1600, height=1200, deviceScaleFactor=1, mobile=False)
        await js("traceExplorer.setState({tab:'api'})")
        await screenshot("05-agent-api.png")
        # Saved state round-trip in the same public interface.
        before = await js("traceExplorer.getState()")
        await js("traceExplorer.selectRange(0,1)")
        await js("traceExplorer.setState(" + json.dumps(before) + ")")
        assert await js("traceExplorer.getState()") == before
        report["tests"].append("Saved selection restores range, identity, and expansions")
        await check_client_drag(call, js, click, screenshot, r, report)
        report["export"] = await js(
            "(()=>{const x=traceExplorer.exportSelection();return {request:x.request.id,sources:x.sources.length,metrics:x.metrics.length,view:x.view}})()"
        )
        assert report["export"]["request"] == r["id"]
        report["errors"] = [e for e in events if e.get("method") == "Runtime.exceptionThrown"]
        report["external_requests"] = [
            e["params"]["request"]["url"]
            for e in events
            if e.get("method") == "Network.requestWillBeSent"
            and e["params"]["request"]["url"].startswith(("http://", "https://"))
        ]
        assert not report["errors"], report["errors"]
        assert not report["external_requests"], report["external_requests"]
        (output / "browser-report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    "load_seconds": report["load_seconds"],
                    "tests": report["tests"],
                    "profiles": len(profiles),
                    "errors": 0,
                    "external_requests": 0,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("html", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=9337)
    p.add_argument("--request")
    a = p.parse_args()
    asyncio.run(run(a.html, a.out, a.port, a.request))
