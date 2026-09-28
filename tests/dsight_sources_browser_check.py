# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise independent optional sources and overlapping activity in real Chrome."""

import argparse
import asyncio
import base64
import json
import shutil
from pathlib import Path
from urllib.parse import quote

import websockets
from dsight_browser_check import browser_targets
from test_dsight import CLIENT, write_run
from test_dsight_engines import add_empty_cpu_profile, tokenspeed_run

from srtctl.dsight.build import build_dashboard


async def run(output: Path, port: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    page = next(p for p in await asyncio.to_thread(browser_targets, port) if p["type"] == "page")
    results, events = [], []
    seq = 0
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=20_000_000) as ws:

        async def call(method, **params):
            nonlocal seq
            seq += 1
            await ws.send(json.dumps({"id": seq, "method": method, "params": params}))
            while True:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                if message.get("id") == seq:
                    assert "error" not in message, message
                    return message.get("result", {})
                events.append(message)

        async def js(expression):
            result = await call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
            assert not result.get("exceptionDetails"), result
            return result["result"].get("value")

        async def click(selector):
            rect = await js(
                "(()=>{const e=document.querySelector(" + json.dumps(selector) + ");"
                "if(!e) throw Error('missing button');e.scrollIntoView({block:'nearest'});"
                "const r=e.getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2}})()"
            )
            await call("Input.dispatchMouseEvent", type="mousePressed", button="left", clickCount=1, **rect)
            await call("Input.dispatchMouseEvent", type="mouseReleased", button="left", clickCount=1, **rect)

        async def navigate(url, job):
            await call("Page.navigate", url="about:blank")
            await call("Page.navigate", url=url)
            for _ in range(200):
                if await js(
                    f"Boolean(window.traceExplorer?.ready && traceExplorer.describe().meta.job === {json.dumps(job)})"
                ):
                    assert not await js("window.traceExplorerError || document.querySelector('#error').textContent")
                    return
                await asyncio.sleep(0.05)
            raise AssertionError("Dashboard did not initialize: " + str(await js("window.traceExplorerError")))

        await call("Page.enable")
        await call("Runtime.enable")
        await call("Network.enable")
        await call("Network.setBlockedURLs", urls=["http://*", "https://*"])
        await call("Emulation.setDeviceMetricsOverride", width=1700, height=1100, deviceScaleFactor=1, mobile=False)
        modes = (
            "tokenspeed",
            "empty-cpu",
            "no-otel",
            "no-nsight",
            "no-metrics",
            "client-only",
            "metrics-only",
            "nsight-only",
            "otel-only",
            "batch-only",
        )
        for mode in modes:
            original, profiles = tokenspeed_run(output / "inputs" / mode)
            logs, options = original, {"sqlites": profiles, "iteration_timezone": "UTC", "job": mode}
            if mode == "empty-cpu":
                add_empty_cpu_profile(profiles)
            elif mode == "no-otel":
                options["otel"] = False
            elif mode == "no-nsight":
                options["sqlites"] = None
            elif mode == "no-metrics":
                shutil.rmtree(logs / "tachometer")
            elif mode.endswith("-only"):
                logs = output / "subsets" / mode
                logs.mkdir(parents=True)
                options["sqlites"] = None
                if mode == "client-only":
                    shutil.copy(next(original.rglob("profile_export.jsonl")), logs / "profile_export.jsonl")
                elif mode == "nsight-only":
                    options["sqlites"] = profiles
                elif mode == "otel-only":
                    shutil.copytree(original / "otel", logs / "otel")
                elif mode == "batch-only":
                    for path in original.glob("*_w*.out"):
                        shutil.copy(path, logs / path.name)
                else:
                    # This fixture contains distinct sample timestamps.
                    metrics_logs, _ = write_run(output / "metric-source")
                    shutil.copytree(metrics_logs / "tachometer", logs / "tachometer")
            report = build_dashboard(logs, output / "reports" / mode, **options)
            cap, url = report["capabilities"], Path(report["html"]).as_uri()
            await navigate(url, mode)
            visibility = await js("""(()=> {
              const shown = (q) => [...document.querySelectorAll(q)].some(e => e.getClientRects().length);
              return {requestTab:shown('.tabs [data-tab=request]'), nsightTab:shown('.tabs [data-tab=nsys]'),
                batchesTab:shown('.tabs [data-tab=iterations]'), clientTracks:!!document.querySelector('#clientTracks'),
                hardware:shown('#hardwareToggle'), metrics:shown('#workerMetric'), cpu:/CPU sample hotspots/.test(document.body.innerText)};
            })()""")
            assert visibility["requestTab"] == cap["requests"], (mode, visibility, cap)
            assert visibility["clientTracks"] == cap["requests"], (mode, visibility, cap)
            assert visibility["nsightTab"] == cap["nsight"], (mode, visibility, cap)
            assert visibility["batchesTab"] == cap["iterations"], (mode, visibility, cap)
            assert visibility["hardware"] == cap["hardware_metrics"], (mode, visibility, cap)
            assert visibility["metrics"] == cap["worker_metrics"], (mode, visibility, cap)
            assert not visibility["cpu"]
            assert (await js("traceExplorer.describe().available")) == cap
            if cap["requests"]:
                await js(f"traceExplorer.selectRequest({json.dumps(CLIENT)},{{expand:true,fit:true}})")
                assert bool(await js("document.querySelector('#expandTTFT') !== null")) == cap["request_breakdown"]
                if cap["request_breakdown"]:
                    model = await js(f"traceExplorer.getLifecycle({json.dumps(CLIENT)})")
                    assert await js("document.querySelectorAll('[data-activity-row]').length") == len(
                        model["activities"]
                    )
                    await click('[data-lifecycle-view="milestones"]')
                    assert await js("document.querySelectorAll('.lifecycle-chain').length") == len(model["stages"])
                    await click('[data-lifecycle-view="activities"]')
                    assert not await js("document.querySelectorAll('.lifecycle-chain').length")
                await js("traceExplorer.selectRequest('client-only',{expand:true})")
                assert not await js("document.querySelector('#expandTTFT') !== null")
            else:
                assert (await js("traceExplorer.queryRequests().total")) == 0
            if mode == "tokenspeed":
                await js(f"traceExplorer.selectRequest({json.dumps(CLIENT)},{{expand:true,fit:true}})")
                await js("traceExplorer.selectSpan('dop',{nsys:true})")
                result = await js("traceExplorer.queryNsys({limit:10})")
                assert result["rank"] is None and result["total"] == 3
                assert await js("document.querySelectorAll('.nsys-track').length") > 0
                await click('.tabs [data-tab="iterations"]')
                header = await js("document.querySelector('#inspectorBody').innerText")
                assert "Queued" in header and "Prev. device ms" not in header and "TRT-LLM" not in header, header
            await js("traceExplorer.selectRange(0, traceExplorer.describe().meta.duration)")
            exported = await js("traceExplorer.exportSelection()")
            assert bool(exported["server_spans"]["total"]) == cap["server_activity"]
            assert bool(exported["batch_observations"]["total"]) == cap["iterations"]
            if cap["server_activity"]:
                assert await js("document.querySelector('.server-activity') !== null")
                page1 = await js("traceExplorer.queryServerSpans({limit:1})")
                page2 = await js("traceExplorer.queryServerSpans({offset:1,limit:1})")
                assert page1["items"][0]["id"] != page2["items"][0]["id"]
            await click('.tabs [data-tab="api"]')
            examples = await js("document.querySelector('#inspectorBody .code').textContent")
            await js(examples)  # Every advertised example uses an available source and recorded identity.
            # Stale links cannot enable unavailable sources or retain bogus identities.
            saved = {
                "from": 0,
                "to": report["meta"]["duration"],
                "request": "absent-request",
                "tab": "nsys",
                "nsys": True,
                "hardware": True,
                "profile": 999,
                "metric": "absent_metric",
                "expandedRequests": ["client-only"],
                "span": "missing",
            }
            await navigate(url + "#view=" + quote(json.dumps(saved)), mode)
            state = await js("traceExplorer.getState()")
            assert state["request"] is None and state["span"] is None and state["expandedRequests"] == [], state
            assert state["nsys"] == cap["nsight"] and state["hardware"] == cap["hardware_metrics"], state
            assert not await js("/CPU sample hotspots/.test(document.body.innerText)")
            if not cap["nsight"]:
                assert state["tab"] != "nsys"
                assert (await js("traceExplorer.queryNsys()"))["total"] == 0
            assert not await js("window.traceExplorerError || document.querySelector('#error').textContent")
            shot = await call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
            (output / f"{mode}.png").write_bytes(base64.b64decode(shot["data"]))
            results.append({"mode": mode, "capabilities": cap, "visibility": visibility, "passed": True})
        errors = [e for e in events if e.get("method") == "Runtime.exceptionThrown"]
        assert not errors, errors
        (output / "browser-report.json").write_text(json.dumps({"cases": results, "errors": errors}, indent=2))
        print(json.dumps({"passed": len(results), "modes": list(modes), "errors": errors}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.out.resolve(), args.port))
