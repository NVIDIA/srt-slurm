# ruter

`ruter` prepares the raw artifacts from a Dynamo KV-router benchmark. It does not serve a dashboard and it does not interpret or alter the benchmark data.

1. Enable the signals in the recipe:

   ```yaml
   frontend:
     type: dynamo

   observability:
     enabled: true
     tachometer:
       enabled: true
   ```

2. Run the recipe through either lifecycle. The post-process step is the same for SLURM and direct Bash:

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

The JSONL records are deliberately small and direct: router records include the exact Dynamo KV routing formula or selected worker, and worker records include batch, request, and lifecycle events. A future viewer can read these files together with Tachometer Parquet without re-parsing logs.
