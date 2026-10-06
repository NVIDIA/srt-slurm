# Standalone SGLang KV-aware router

`frontend.type: sgl-router` launches upstream's standalone
`experimental/sgl-router` executable. It is distinct from
[`sglang-router`](sglang-router.md), which launches the Python Model Gateway.
The stock router image is distroless: srtctl invokes its binary directly and
does not add a shell wrapper. `frontend.numa_bind` is unsupported by that image.

**Container runtime limitation:** Pyxis itself starts an in-container shell to
establish its namespaces, even for a direct executable command. The pinned
distroless image therefore fails before the router starts with
`enroot-switchroot: failed to execute: /bin/sh: No such file or directory`.
This frontend does not make that image compatible with Pyxis; it requires a
runtime-compatible upstream image. See [Pyxis's startup implementation](https://github.com/NVIDIA/pyxis/blob/107519944221822ea1dace4db8e7234b2eaa4cd5/pyxis_slurmstepd.c#L1056).

The router discovers each HTTP worker's aggregate, prefill or decode role and
bootstrap port from `/server_info`. srtctl supplies the static worker URLs and
served model name. `/readyz`, registry counts and circuit-breaker health from
`/metrics`, and direct worker health together gate benchmark startup. A usable
single worker is not sufficient when the recipe requested several.

See [the MI300X topology example](https://github.com/NVIDIA/srt-slurm/blob/main/examples/sglang/sgl-router-disagg.yaml), subject to the container-runtime limitation above.
The router and engine need compatible versions: the example pins the stock
router build at [ed75fe6](https://github.com/sgl-project/sglang/tree/ed75fe67f119f14afa7fd6b7491055ece381b9aa/experimental/sgl-router)
and an engine containing [730f1f3](https://github.com/sgl-project/sglang/tree/730f1f3e5be9c781f523f856fc39c18c2ad266d6/python/sglang/srt).

## Cache events and queued work

Set `roles.<role>.kv_events` to enable the engine publisher. Set
`roles.<role>.args.load-publish-endpoint: auto` to also publish engine load;
KV events alone do not enable load publication. To enable dropped-event
replay, set `roles.<role>.kv_events.replay_endpoint: auto`.
srtctl replaces these endpoint placeholders with independently allocated
KV, replay and load ranges, reserving a port per DP cache. Multiple workers
on a node therefore do not compete for an adjacent port chosen by the engine.

The example uses upstream `--chat-routing reorg --policy cache_aware
--affinity-mode balanced`. This compares queued uncached prefill tokens with
the prefix already available on each candidate. The native default thresholds
are unchanged. Other upstream router arguments pass through `frontend.args`.

Use the same tokenizer snapshot and rendering defaults as the workers.
`disable-input-ids-forwarding: true` keeps engine-owned prompt rendering;
it does not prove that the router's cache hashes match. Verify actual cache
events and routing decisions before making performance claims.

The pinned engine advertises one host/base for each worker's DP publishers.
Attention DP spread across several nodes does not expose every rank through
that single host. Prefer node-local DP for this telemetry path; do not infer
complete remote-rank load coverage from a healthy router.
