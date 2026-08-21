# ruter

`ruter` explains a completed Dynamo KV-router run. It is built and run from this srt-slurm checkout; it does not run or compare SGLang routers.

## Use it

### 1. Render and run the E2E example

From the srt-slurm checkout, build the two local tools once:

```bash
make setup ARCH="$(uname -m)"
make ruter
```

For aggregation, use [the 8×TP1 recipe](examples/qwen3-32b-fp8-dynamo-mooncake.yaml). For a 3-prefill / 2-decode run, use [the 3P2D recipe](examples/qwen3-32b-fp8-dynamo-mooncake-3p2d.yaml). The 3P2D recipe takes the two host paths through environment variables so the YAML stays shareable:

```bash
cd /path/to/srt-slurm
export RUTER_MODEL_PATH=/absolute/path/to/Qwen3-32B-FP8
export RUTER_MOONCAKE_TRACE=/absolute/path/to/mooncake_refined.jsonl
uv run srtctl apply -f ruter/examples/qwen3-32b-fp8-dynamo-mooncake-3p2d.yaml --bash > benchmark.sh
SRTCTL_PYTHON=/path/to/dynamo/.venv/bin/python bash benchmark.sh
```

The rendered Bash launches all lifecycle pieces: role-qualified worker logs, router log, Tachometer Parquet, AIPerf trace, and the Dynamo request trace. `observability.enabled` supplies the debug records ruter needs.

### 2. Initialize the completed run

Run this from the srt-slurm output directory after the benchmark completes:

```bash
cd /path/to/benchmark-output
/path/to/srt-slurm/bin/ruter init
```

This creates `.ruter/` beside the raw artifacts. It parses and materializes the data once; it never edits the benchmark artifacts.

### 3. Open the dashboard

```bash
cd /path/to/benchmark-output
/path/to/srt-slurm/bin/ruter view
```

Open `http://127.0.0.1:8877`.

### 4. Share it

Share the srt-slurm output directory with its `.ruter/` folder. The recipient can open the prepared analysis without rerunning parsing:

```bash
/path/to/srt-slurm/bin/ruter view --analysis /path/to/shared-output/.ruter
```

`view` reads only the prepared `.ruter/ruter.db`; raw logs and Parquet stay out of the browser.
