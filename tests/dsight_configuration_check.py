# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional config lineage and capacity overlays through the actual offline viewer."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets
from dsight_browser_check import browser_targets
from dsight_metrics_check import Browser
from test_dsight_configuration import configured_run, recipe, scheduler
from test_dsight_log_metrics import batch

from srtctl.dsight.build import build_dashboard
from srtctl.dsight.log_metrics.tokenspeed import ACTIVE_DECODE, ACTIVE_PAGES, DECODE_LIMIT


async def run(out: Path, port: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    page = next(p for p in await asyncio.to_thread(browser_targets, port) if p["type"] == "page")
    report = {"cases": []}
    cases = {
        "matching": (recipe("8"), [scheduler(), batch(), batch(36, active=6)]),
        "different": (recipe("16"), [scheduler(), batch(), batch(36, active=6)]),
        "absent": (None, [scheduler(), batch(), batch(36)]),
        "malformed": ("roles: [broken", [scheduler(), batch(), batch(36)]),
        "multiple-dp": (recipe(), [scheduler(dp=2), batch(), batch(36)]),
        "missing-scope": (recipe(), [batch(), batch(36)]),
        "late-scope": (recipe(), [batch(), scheduler(35), batch(36)]),
        "changed-runtime": (recipe("8"), [scheduler(), batch(), scheduler(35, maximum=12), batch(36)]),
    }
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=100_000_000) as socket:
        browser = Browser(socket, out)
        try:
            for domain in ("Page", "Runtime", "Network"):
                await browser.call(f"{domain}.enable")
            await browser.call("Emulation.setDeviceMetricsOverride", width=1500, height=1100, deviceScaleFactor=1, mobile=False)
            for case, (content, lines) in cases.items():
                importer, _data, config_path = configured_run(out / case, content, lines)
                built = build_dashboard(importer.logs, out / case / "report", iteration_timezone="UTC")
                await browser.call("Page.navigate", url="about:blank")
                await browser.wait("!window.traceExplorer")
                await browser.call("Page.navigate", url=Path(built["html"]).as_uri())
                await browser.wait("Boolean(window.traceExplorer?.ready)")
                await browser.js("traceExplorer.whenMetricsReady()")
                await browser.js("""(()=>{
                  window.__plots=[];
                  const Plot=window.uPlot;
                  window.uPlot=Object.assign(function(...args){const plot=new Plot(...args);__plots.push(plot);return plot},Plot);
                })()""")
                await browser.js("traceExplorer.setState(" + json.dumps({"metric":ACTIVE_DECODE,"pinnedMetrics":[ACTIVE_DECODE,ACTIVE_PAGES],"from":0,"to":10}) + ")")
                await browser.js("traceExplorer.whenMetricsReady()")
                series = await browser.js("traceExplorer.queryMetrics({name:" + json.dumps(ACTIVE_DECODE) + ",points:true})")
                assert series[0]["max"] in (4,6)
                assert not await browser.js("document.getElementById('error').textContent")
                assert await browser.js("traceExplorer.listMetricFamilies().map(f=>f.name).filter(n=>n.startsWith('log_tokenspeed_')).length") == (3 if case=="missing-scope" else 4)
                boxes = await browser.js("document.querySelector('.metric-card').querySelectorAll('.ds-metric-configuration').length")
                if case in {"absent","malformed"}:
                    assert boxes == 0
                    assert "configuration" not in series[0]
                    assert await browser.js("__plots[0].series.length") == 3
                else:
                    assert boxes == 1, {"case": case, "configuration": series[0].get("configuration"), "boxes": boxes}
                    text = await browser.js("document.querySelector('.ds-metric-configuration').textContent")
                    assert "roles.decode.args.max-num-seqs" in text and "Configured max requests" in text
                    assert str(config_path) in text
                    metadata = series[0]["configuration"][0]
                    if case in {"multiple-dp","missing-scope"}:
                        assert metadata["comparison"] is None
                        assert await browser.js("__plots[0].series.length") == (2 if case=="missing-scope" else 3)
                        assert "per-scheduler comparison requires" in text
                    else:
                        assert await browser.js("__plots[0].series.length") == 4
                        assert await browser.js("__plots[0].series[3].dash") == [2,3]
                        assert await browser.js("__plots[0].scales.y.max") > metadata["source"]["value"]
                        assert "dp_size=1" in text and "SHA-256" in text
                        if case=="late-scope":
                            assert metadata["comparison"]["start"] == 4
                            assert await browser.js("__plots[0].data[0].every((t,i)=>t>=4 || !Number.isFinite(__plots[0].data[3][i]))")
                        else:
                            assert metadata["comparison"]["start"] == -1
                            assert ("Recorded limits match" if case=="matching" else "Recorded limits differ") in text
                        await browser.click(".ds-metric-legend-toggle")
                        assert await browser.js("__plots[0].series.slice(1).every(s=>s.show===false)")
                        await browser.click(".ds-metric-legend-toggle")
                    if case=="different":
                        logged = await browser.js("traceExplorer.queryMetrics({name:"+json.dumps(DECODE_LIMIT)+",points:true})")
                        assert logged[0]["carried_setting"][0][1] == 8
                        assert metadata["source"]["value"] == 16
                        await browser.click(".ds-metric-configuration summary")
                        await browser.rectangle("#metricsSection")
                        await browser.screenshot("01-config-runtime-lineage.png")
                        await browser.js("traceExplorer.setState({from:0,to:3})")
                        await browser.js("traceExplorer.whenMetricsReady()")
                        await browser.click(".ds-metric-configuration summary")
                        assert "4 / 16 (25%)" in await browser.js("document.querySelector('.ds-metric-configuration').textContent")
                        await browser.call("Emulation.setDeviceMetricsOverride",width=390,height=1100,deviceScaleFactor=1,mobile=False)
                        await asyncio.sleep(.2)
                        await browser.rectangle("#metricsSection")
                        await browser.screenshot("02-narrow-config-lineage.png")
                        assert await browser.js("document.documentElement.scrollWidth<=innerWidth+1")
                        await browser.call("Emulation.setDeviceMetricsOverride",width=1500,height=1100,deviceScaleFactor=1,mobile=False)
                report["cases"].append({"case":case,"passed":True,"configuration":series[0].get("configuration")})
            errors = [event for event in browser.events if event.get("method")=="Runtime.exceptionThrown"]
            assert not errors, errors
            requests = [event for event in browser.events if event.get("method")=="Network.requestWillBeSent" and event["params"]["request"]["url"].startswith(("http:","https:"))]
            assert not requests, requests
            report.update(runtime_errors=errors, external_requests=requests)
        except Exception as error:
            report["failure"] = str(error)
            await browser.screenshot("failure.png")
            raise
        finally:
            (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":len(report["cases"]),"runtime_errors":report["runtime_errors"]}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--port",type=int,default=9222)
    args=parser.parse_args()
    asyncio.run(run(args.out,args.port))
