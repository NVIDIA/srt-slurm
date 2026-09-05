# ruter

`ruter` prepares the raw artifacts from a Dynamo KV-router benchmark, then optionally serves the local route-decision dashboard. It never alters raw logs or Tachometer Parquet.

1. Enable the signals in the recipe:

   ```yaml
   frontend:
     type: dynamo

   observability:
     enabled: true
     tachometer:
       enabled: true
   ```

2. Run the recipe through either lifecycle. The post-process step is the same for SLURM and the [direct Docker lifecycle](direct-host.md):

   ```bash
   uv run srtctl apply -f recipe.yaml
   uv run srtctl apply -f recipe.yaml --bash > benchmark.sh
   bash benchmark.sh
   ```

3. At completion, find the generated bundle in `logs/.ruter/`:

   ```text
   logs/.ruter/
   ├── manifest.json
   ├── router-events.jsonl
   └── worker-events.jsonl
   ```

   `manifest.json` records the original Tachometer `final.parquet` path. ruter leaves that Parquet file and all raw logs untouched.

4. Re-run only the post-processing later, if needed:

   ```bash
   uv run ruter init /path/to/run-output
   ```

5. Install the optional viewer and open the same bundle:

   ```bash
   uv sync --extra ruter
   uv run srtctl view /path/to/run-output
   ```

   The viewer binds only to `127.0.0.1:8877`; use `--port 9000` to choose another port. It loads the normalized events, request trace, AIPerf results, and Tachometer worker snapshots before serving the static dashboard. Use `--refresh` to rebuild `logs/.ruter/` after updating a parser.

The JSONL records are deliberately small and direct: router records include the exact Dynamo KV routing formula or selected worker, and worker records include batch, request, and lifecycle events.
