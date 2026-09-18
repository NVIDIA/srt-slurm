# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real pointer regression checks for the client timeline range selector."""

import json


async def check_client_drag(call, js, click, screenshot, request, report):
    baseline = {
        "from": max(0, request["start"] - 0.5),
        "to": min(request["end"], request["first"] + 0.5),
        "request": request["id"],
        "span": None,
        "tab": "request",
        "nsys": False,
        "hardware": False,
        "search": request["id"],
        "page": 0,
        "expandedSessions": [request["session"]],
        "expandedAgents": [request["agent"]],
        "expandedRequests": [request["id"]],
    }

    async def reset():
        await js("traceExplorer.setState(" + json.dumps(baseline) + ")")

    async def geometry(selector):
        return await js(
            """(async()=>{const e=document.querySelector("""
            + json.dumps(selector)
            + """);
          if(!e)throw Error("Missing drag target");e.scrollIntoView({block:"nearest"});
          await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
          const r=e.getBoundingClientRect(),l=e.closest(".lane").getBoundingClientRect();
          return {x:r.x,y:r.y,w:r.width,h:r.height,left:l.left,width:l.width};})()"""
        )

    async def press(point):
        await call("Input.dispatchMouseEvent", type="mousePressed", button="left", clickCount=1, **point)

    async def move(point):
        await call("Input.dispatchMouseEvent", type="mouseMoved", button="left", buttons=1, **point)

    async def release(point):
        await call("Input.dispatchMouseEvent", type="mouseReleased", button="left", clickCount=1, **point)

    async def assert_range(box, start, end):
        def t(x):
            return baseline["from"] + max(0, min(1, (x - box["left"]) / box["width"])) * (
                baseline["to"] - baseline["from"]
            )

        expected = sorted((t(start["x"]), t(end["x"])))
        state = await js("traceExplorer.getState()")
        assert abs(state["from"] - expected[0]) < 1e-7 and abs(state["to"] - expected[1]) < 1e-7, (state, expected)
        assert state["request"] == request["id"]
        assert state["expandedSessions"] == baseline["expandedSessions"]
        assert not await js("Boolean(document.querySelector('.client-range-preview'))")
        inputs = await js("[+document.getElementById('rangeFrom').value,+document.getElementById('rangeTo').value]")
        assert all(abs(actual - want) < 1e-6 for actual, want in zip(inputs, expected, strict=False))
        query = await js("traceExplorer.queryRequests({limit:0})")
        assert query["range"] == [state["from"], state["to"]]
        return state

    await reset()
    await js(
        "window.clientRangeEvents=[];window.addEventListener('trace-explorer:state',e=>window.clientRangeEvents.push(e.detail))"
    )
    box = await geometry("#clientTracks .lane")
    # Empty lane pixels above the bar, then move down across another client row.
    start = {"x": box["left"] + box["width"] * 0.2, "y": box["y"] + 2}
    end = {"x": box["left"] + box["width"] * 0.7, "y": box["y"] + box["h"] + 15}
    await press(start)
    await move(end)
    preview = await js(
        "(()=>{const e=document.querySelector('.client-range-preview');if(!e)return null;const r=e.getBoundingClientRect();return {x:r.x,w:r.width,text:e.textContent}})()"
    )
    assert preview and abs(preview["x"] - start["x"]) < 1 and abs(preview["w"] - (end["x"] - start["x"])) < 1
    during = await js("traceExplorer.getState()")
    assert (during["from"], during["to"]) == (baseline["from"], baseline["to"])
    await screenshot("09-client-range-preview.png")
    await release(end)
    await assert_range(box, start, end)
    assert await js("clientRangeEvents.length") == 1
    report["tests"].append("Client-panel empty-space drag previews and commits one shared range on release")
    await screenshot("10-client-range-selected.png")
    await click("#rangeBack")
    restored = await js("traceExplorer.getState()")
    assert (restored["from"], restored["to"]) == (baseline["from"], baseline["to"])
    report["tests"].append("Client-panel drag participates in Previous time range history")

    await reset()
    box = await geometry("#clientTracks .bar.session")
    start = {"x": box["x"] + box["w"] * 0.8, "y": box["y"] + box["h"] / 2}
    end = {"x": box["x"] + box["w"] * 0.3, "y": start["y"]}
    await press(start)
    await move(end)
    await release(end)
    await assert_range(box, start, end)
    report["tests"].append("Right-to-left drag over a session bar selects a range without toggling the session")

    await reset()
    phase = max(request["lifecycle"]["stages"][:-1], key=lambda s: s["end"] - s["start"])
    box = await geometry('#clientTracks [data-stage-span="' + phase["id"] + '"]')
    start = {"x": box["x"] + box["w"] * 0.2, "y": box["y"] + box["h"] / 2}
    end = {"x": box["x"] + box["w"] * 0.8, "y": start["y"]}
    await press(start)
    await move(end)
    await release(end)
    state = await assert_range(box, start, end)
    assert state["span"] is None
    report["tests"].append("Dragging a lifecycle block changes the range without selecting its span")

    await reset()
    box = await geometry("#clientTracks .lane")
    start = {"x": box["left"] + box["width"] * 0.4, "y": box["y"] + 2}
    end = {"x": box["left"] + box["width"] + 90, "y": start["y"]}
    await press(start)
    await move(end)
    await release(end)
    await assert_range(box, start, end)
    report["tests"].append("Pointer capture finishes outside the panel and clamps to the visible time boundary")

    await reset()
    box = await geometry("#clientTracks .lane")
    start = {"x": box["left"] + box["width"] * 0.2, "y": box["y"] + 2}
    end = {"x": box["left"] + box["width"] * 0.7, "y": start["y"]}
    await press(start)
    await move(end)
    await call("Input.dispatchKeyEvent", type="keyDown", key="Escape", code="Escape", windowsVirtualKeyCode=27)
    await call("Input.dispatchKeyEvent", type="keyUp", key="Escape", code="Escape", windowsVirtualKeyCode=27)
    await release(end)
    after = await js("traceExplorer.getState()")
    assert (after["from"], after["to"]) == (baseline["from"], baseline["to"])
    assert not await js("Boolean(document.querySelector('.client-range-preview'))")
    report["tests"].append("Escape cancels the drag and clears its preview without changing the range")

    await reset()
    await js("traceExplorer.selectSpan(" + json.dumps(phase["id"]) + ")")
    box = await geometry('#clientTracks .track.selected [data-request="' + request["id"] + '"]')
    start = {"x": box["x"] + box["w"] * 0.4, "y": box["y"] + box["h"] / 2}
    end = {"x": start["x"] + 2, "y": start["y"]}
    await press(start)
    await move(end)
    await release(end)
    after = await js("traceExplorer.getState()")
    assert (after["from"], after["to"]) == (baseline["from"], baseline["to"])
    assert after["request"] == request["id"] and after["span"] is None
    await click('#clientTracks [data-stage-span="' + phase["id"] + '"]')
    assert (await js("traceExplorer.getState()"))["span"] == phase["id"]
    report["tests"].append("Small pointer movement preserves request clicks and lifecycle block drill-down")

    await reset()
    box = await geometry("#clientTracks .lane")
    start = {"x": box["left"] + box["width"] * 0.2, "y": box["y"] + 2}
    end = {"x": box["left"] + box["width"] * 0.7, "y": start["y"]}
    await press(start)
    await move(end)
    await js("traceExplorer.selectRange(0,.001)")
    await release(end)
    after = await js("traceExplorer.getState()")
    assert (after["from"], after["to"]) == (0, 0.001)
    assert not await js("Boolean(document.querySelector('.client-range-preview'))")
    report["tests"].append("An agent API range change cancels a pending gesture without a stale overwrite")
    await reset()
