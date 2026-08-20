# ruter

`ruter` explains Dynamo KV-router decisions after a benchmark has finished. It does not run or compare SGLang routers.

## Use it

### 1. Render and run the E2E example

Use srt-slurm after [PR 317](https://github.com/NVIDIA/srt-slurm/pull/317) has merged. On the benchmark host, install the native Tachometer scraper once:

```bash
make setup ARCH="$(uname -m)"
```

Start with [the 8×TP1 Qwen3-32B FP8 Dynamo KV/Mooncake example](examples/qwen3-32b-fp8-dynamo-mooncake.yaml). Set its `model.path` and Mooncake `--input-file` to local paths, then render it:

```bash
cd /path/to/srt-slurm
uv run srtctl apply -f /path/to/ruter/examples/qwen3-32b-fp8-dynamo-mooncake.yaml --bash > benchmark.sh
bash benchmark.sh
```

The example launches eight aggregated TP1 SGLang workers behind the Dynamo KV router. `observability.enabled` supplies the Dynamo debug logging ruter needs; the output bundle contains the router log, individual `worker-*.log` files, Tachometer `final.parquet`, AIPerf trace, and Dynamo request trace.

### 2. Initialize the completed run

Run this from the srt-slurm output directory after the benchmark completes:

```bash
cd /path/to/benchmark-output
uvx ruter init
```

This creates `.ruter/` beside the raw artifacts. It parses and materializes the data once; it never edits the benchmark artifacts.

### 3. Open the dashboard

```bash
cd /path/to/benchmark-output
uvx ruter view
```

Open `http://127.0.0.1:8877`.

### 4. Share it

Share the srt-slurm output directory with its `.ruter/` folder. The recipient can open the prepared analysis without rerunning parsing:

```bash
uvx ruter view --analysis /path/to/shared-output/.ruter
```

`view` reads only the prepared `.ruter/ruter.db`; raw logs and Parquet stay out of the browser.
