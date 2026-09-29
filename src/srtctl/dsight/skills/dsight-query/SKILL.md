---
name: dsight-query
description: Analyze existing DSight reports through the read-only SQLite-backed CLI, Python or MCP query interface. Use when inspecting request latency, lifecycle, metrics, Nsight ranges or CPU samples, comparing runs, or gathering evidence for a dashboard view.
---

# Query DSight evidence

Read the [DSight guide](../../../../../docs/dsight.md) before using this workflow.
Use the [storage reference](../../../../../docs/dsight-storage.md) for the current
schema, query filters, pagination and browser API. The
[data-flow guide](../../../../../docs/dsight-data-flow.md) explains source joins
and which evidence reaches each UI view.

## Select the matching artifact

1. Locate the report directory corresponding to the user's run or dashboard.
   Inspect its manifest and source inventory to verify the inputs and generation;
   a URL version parameter or an older local copy is not proof of a match.
2. Prefer the report's `trace-data.sqlite` for analysis. Passing the directory to
   DSight queries automatically selects it. This is the normalized DSight cache;
   original Nsight SQLite exports are build inputs with a different schema.
3. A legacy single-file report uses `trace-data.json.gz`; the same query interface
   accepts that file or its directory. State which format was inspected.
4. A browser publication may contain only `index.html`, compressed detail files
   and a manifest. The browser does not read SQLite. Locate the matching local
   query cache from available artifact records; do not invent its location. If
   only the browser is accessible, use the documented browser API and state this
   limitation. Await detail queries before interpreting their results.

Query preserved artifacts for analysis. Do not rebuild a report or rerun a
benchmark merely to inspect it. If a dataset reports that its generation changed,
reopen it before continuing; do not combine evidence from different generations.

## Discover, then narrow

From the repository root, replace the quoted placeholders with recorded values:

```bash
uv run --no-dev srtctl dsight query "<report_directory>" summary
uv run --no-dev srtctl dsight query "<report_directory>" sources --limit 20
uv run --no-dev srtctl dsight query "<report_directory>" profiles --limit 20
uv run --no-dev srtctl dsight query "<report_directory>" metrics --limit 20
uv run --no-dev srtctl dsight query "<report_directory>" requests --limit 20
```

Read the summary's duration, warnings and coverage before making claims. Discover
exact metric names, profile IDs and client request IDs; never infer them from a
chart label or filename. Paginate a catalog when the desired item is not on its
first page. Select only the sources relevant to the question.

```bash
uv run --no-dev srtctl dsight query "<report_directory>" metrics \
  --name "<recorded_metric_name>" --from "<start_seconds>" --to "<end_seconds>" --points
uv run --no-dev srtctl dsight query "<report_directory>" lifecycle \
  --request "<recorded_client_request_id>"
uv run --no-dev srtctl dsight query "<report_directory>" nsys \
  --profile "<recorded_profile_id>" --from "<start_seconds>" --to "<end_seconds>" --limit 100
```

Times are seconds relative to the report's exact `meta.origin_ns`, not wall-clock
seconds or raw Nsight timestamps. Choose `0 <= start < end <= duration`. Separate
runs have separate origins; compare equivalent phases or load windows rather
than assuming the same offsets represent the same workload.

For repeated queries, use Python to open the catalog once:

```python
from srtctl.dsight.query import TraceDataset

trace = TraceDataset.from_path("<report_directory>")
summary = trace.query("summary")
profiles = trace.query("profiles", limit=20)
metrics = trace.query("metrics", limit=20)
```

MCP provides `query_trace(dataset, kind="summary", **filters)` with the same
response shapes. Its dataset path is on the MCP server's filesystem. CLI
`--from`, `--to` and `--request` map to Python/MCP `start`, `end` and `request_id`.

## Preserve query semantics

- Use `TraceDataset.query` for evidence. In a SQLite-backed dataset,
  `TraceDataset.data` contains catalog metadata with empty event, sample and
  metric-point arrays. Those arrays do **not** mean the evidence is absent.
  Prefer the query interface to ad hoc SQL; it resolves dictionaries, source
  references and indexed interval overlap. Direct SQL inspection must open the
  database read-only, as shown in the storage reference.
- Bound detailed queries by time and worker/profile; use offsets and limits to
  paginate rows. Read totals and truncation indicators before describing a page
  as the full capture. `--limit 0` returns counts for list queries.
- Metric pagination selects series, not points. `--points` returns at most 1,000
  points per series; check the `"points_total"` and `"points_truncated"` fields.
  Narrow the time window for more detail. Min/max/mean/last use all non-null points
  in the selected window even when the returned point list is truncated.
- Windows include both boundaries. Avoid counting boundary samples twice when
  combining adjacent windows. Settings can include `carried_setting` from before
  the window; retain conflicting values instead of arbitrarily choosing one.
- Inspect labels and worker/profile identity. Recorded ranks may describe
  different process layouts; a router rank does not establish a GPU-profile
  join. Use discovered profile IDs when worker/rank attribution is ambiguous.
- Missing OTel, metrics or profiles mean unavailable evidence, not zero activity.
  An Nsight `partial` result describes the imported subset. NVTX intervals are
  annotations, not CUDA kernel measurements or exclusive per-request GPU time.

## Explain the finding

Report the artifact/generation, query window, exact metric or profile identity,
and relevant source IDs/rows with the measured values. Distinguish observations
from inferences and name missing evidence that limits the conclusion. For
capacity questions, keep configured settings, logged effective limits and
runtime samples distinct; inspect their available provenance before comparing.

When sharing a dashboard view, select metric names and request IDs actually found
in that report and use the saved-view interface documented in the DSight guide.
The link supports the explanation; keep the evidence and interpretation in the
answer so the user can understand it without opening the viewer.
