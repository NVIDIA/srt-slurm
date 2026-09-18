# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check traced and untraced request UI behavior in an isolated Chrome session."""

import argparse
import asyncio
import base64
import json
import tempfile
from pathlib import Path
from urllib.parse import quote

import websockets
from dsight_browser_check import browser_targets
from test_dsight import CLIENT, SERVER, write_run

from srtctl.dsight.build import build_dashboard


async def run(output: Path, port: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    inputs = Path(tempfile.mkdtemp(prefix="inputs-", dir=output))
    targets = await asyncio.to_thread(browser_targets, port)
    page = next(p for p in targets if p["type"] == "page")
    events, results = [], []
    seq = 0
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=10_000_000) as ws:

        async def call(method, **params):
            nonlocal seq
            seq += 1
            ident = seq
            await ws.send(json.dumps({"id": ident, "method": method, "params": params}))
            while True:
                message = json.loads(await ws.recv())
                if message.get("id") == ident:
                    if "error" in message:
                        raise AssertionError(message["error"])
                    return message.get("result", {})
                events.append(message)

        async def js(expression):
            result = await call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
            assert not result.get("exceptionDetails"), result
            return result["result"].get("value")

        async def click(selector):
            rect = await js(
                "(()=>{const e=document.querySelector(" + json.dumps(selector) + ");"
                "e.scrollIntoView({block:'nearest'});const r=e.getBoundingClientRect();"
                "return {x:r.x+r.width/2,y:r.y+r.height/2}})()"
            )
            await call("Input.dispatchMouseEvent", type="mousePressed", button="left", clickCount=1, **rect)
            await call("Input.dispatchMouseEvent", type="mouseReleased", button="left", clickCount=1, **rect)

        async def navigate(url, job):
            await call("Page.navigate", url="about:blank")
            await call("Page.navigate", url=url)
            for _ in range(100):
                if await js(
                    "Boolean(window.traceExplorer?.ready && "
                    f"traceExplorer.describe().meta.job === {json.dumps(job)} && document.querySelector('.track'))"
                ):
                    return
                await asyncio.sleep(0.1)
            raise AssertionError("Explorer did not initialize")

        async def check_request(request_id, available):
            await js(f"traceExplorer.selectRequest({json.dumps(request_id)}, {{expand:true,fit:true}})")
            actual = await js(
                "(()=>{const x=traceExplorer,s=x.getState(),r=x.getRequest(s.request),m=x.getLifecycle(s.request);"
                "return {available:m.available,stages:m.stages.length,expanded:s.expandedRequests.includes(r.id),"
                "button:!!document.querySelector('#expandTTFT'),"
                "toggle:!!document.querySelector('[data-toggle=request][data-id=\"'+r.id+'\"]'),"
                "rows:document.querySelectorAll('.lifecycle-chain[data-owner-request=\"'+r.id+'\"]').length,"
                "placeholder:/No worker ID mapping recorded|host not mapped/.test(document.querySelector('#inspectorBody').innerText),"
                "breakdown:/Progress milestones|Source measurements/.test(document.querySelector('#inspectorBody').innerText)}})()"
            )
            assert actual["available"] is available, actual
            assert actual["button"] is available and actual["toggle"] is available, actual
            assert actual["expanded"] is available and actual["breakdown"] is available, actual
            assert not actual["placeholder"], actual
            assert actual["rows"] == actual["stages"], actual
            if not available:
                assert actual["stages"] == 0, actual
                await js(f"traceExplorer.expandRequest({json.dumps(request_id)})")
                assert request_id not in (await js("traceExplorer.getState()"))["expandedRequests"]
            return actual

        await call("Page.enable")
        await call("Runtime.enable")
        await call("Network.enable")
        await call("Emulation.setDeviceMetricsOverride", width=1600, height=1100, deviceScaleFactor=1, mobile=False)
        for mode in ("mixed", "missing", "empty", "unjoined", "disabled", "client-only"):
            logs, sqlites = write_run(inputs / mode)
            trace = next(logs.glob("otel/*/traces.jsonl"))
            if mode == "missing":
                trace.unlink()
            elif mode == "empty":
                trace.write_text("")
            elif mode == "unjoined":
                trace.write_text(trace.read_text().replace(SERVER, "33333333-3333-4333-8333-333333333333"))
            elif mode == "disabled":
                trace.write_text("invalid OTLP JSON; --no-otel must skip this file\n")
            elif mode == "client-only":
                client = next(logs.rglob("profile_export.jsonl"))
                logs = inputs / "bare"
                logs.mkdir()
                (logs / "profile_export.jsonl").write_bytes(client.read_bytes())
                sqlites = None
            report = build_dashboard(
                logs, output / mode, sqlites=sqlites, otel=mode != "disabled", iteration_timezone="UTC", job=mode
            )
            url = Path(report["html"]).as_uri()
            await navigate(url, mode)
            await check_request("client-only", False)
            assert not await js(
                "/Recorded request path|Identity bridge/.test(document.querySelector('#inspectorBody').innerText)"
            )
            available = mode == "mixed"
            check = await check_request(CLIENT, available)
            await click("#fitTTFT")
            state = await js("traceExplorer.getState()")
            assert state["from"] == 0 and state["to"] < 4, state
            assert (CLIENT in state["expandedRequests"]) is available, state
            # Restored links cannot force an unavailable breakdown or stage selection.
            saved = {
                "from": 0,
                "to": 10,
                "request": CLIENT,
                "expandedRequests": [CLIENT, "client-only"],
                "span": "progress:0",
            }
            await navigate(url + "#view=" + quote(json.dumps(saved)), mode)
            state = await js("traceExplorer.getState()")
            assert "client-only" not in state["expandedRequests"], state
            assert (CLIENT in state["expandedRequests"]) is available, state
            assert (state["span"] is not None) is available, state
            await js("document.querySelector('#rangeFrom').value=0;document.querySelector('#rangeTo').value=4")
            await click("#applyRange")
            state = await js("traceExplorer.getState()")
            assert (state["from"], state["to"]) == (0, 4), state
            # The client panel still accepts a native drag without lifecycle rows.
            rect = await js(
                "(()=>{document.querySelector('#tracks').scrollTop=0;"
                "const r=document.querySelector('#clientTracks .lane').getBoundingClientRect();"
                "return {x:r.x+r.width*.25,end:r.x+r.width*.75,y:r.y+r.height/2}})()"
            )
            await call(
                "Input.dispatchMouseEvent", type="mousePressed", button="left", clickCount=1, x=rect["x"], y=rect["y"]
            )
            await call(
                "Input.dispatchMouseEvent", type="mouseMoved", button="left", buttons=1, x=rect["end"], y=rect["y"]
            )
            await call(
                "Input.dispatchMouseEvent",
                type="mouseReleased",
                button="left",
                clickCount=1,
                x=rect["end"],
                y=rect["y"],
            )
            state = await js("traceExplorer.getState()")
            assert abs(state["from"] - 1) < 1e-5 and abs(state["to"] - 3) < 1e-5, state
            if mode != "client-only":
                assert len(await js("traceExplorer.queryMetrics()")) == 2
                assert (await js("traceExplorer.queryIterations({from:0,to:10})"))["total"] == 2
                assert (await js("traceExplorer.inspectNsys({worker:'decode-0',rank:0,from:2,to:3})"))["total"] == 2
                await click("[data-tab=request]")
            else:
                assert await js("getComputedStyle(document.querySelector('#joinBadge')).display === 'none'")
            await check_request(CLIENT, available)
            if not available:
                assert not await js("document.querySelector('#coverageNotice').innerText.includes('OTel')")
            await js("document.querySelector('#tracks').scrollTop=0")
            shot = await call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
            (output / f"{mode}.png").write_bytes(base64.b64decode(shot["data"]))
            results.append(
                {"mode": mode, "request": check, "range_and_saved_view": "passed", "independent_sources": "passed"}
            )
        errors = [e for e in events if e.get("method") == "Runtime.exceptionThrown"]
        external = [
            e
            for e in events
            if e.get("method") == "Network.requestWillBeSent" and e["params"]["request"]["url"].startswith("http")
        ]
        assert not errors, errors
        assert not external, external
        summary = {"cases": results, "errors": errors, "external_requests": external}
        (output / "browser-report.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.out.resolve(), args.port))
