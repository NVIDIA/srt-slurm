# Build and explore a preserved run

From a checkout with DSight installed:

```bash
uv run srtctl dsight build /path/to/run --output /path/to/report \
  --nsys-sqlite /path/to/exported-profiles

# Exact local evidence without opening the complete dataset in a browser:
uv run srtctl dsight query /path/to/report profiles
uv run srtctl dsight query /path/to/report nsys \
  --profile 0 --from 10 --to 11 --limit 100

# Serve the complete generated directory on your own machine:
python -m http.server 8000 --bind 127.0.0.1 --directory /path/to/report
```

Open `http://127.0.0.1:8000/index.html`. Metrics load for the selected family and
time window. Broad Nsight views show labeled density bins; zoom to load exact
ranges. The browser API also supports `await traceExplorer.queryNsys(...)` and
`await traceExplorer.queryCpu(...)` for exact queries.

For a portable single HTML file opened directly from disk, rebuild with
`--single-file`. This embeds all imported data and can be much larger. All inputs
are preserved artifacts; these commands do not launch a benchmark or profiler.
See [DSight](../../docs/dsight.md) for import limits and source semantics.
