# Parameter Sweeps

Parameter sweeps let you run multiple configurations with a single command. Sweeps are automatically detected from config files that contain a `sweep:` section.

## Table of Contents

- [How It Works](#how-it-works)
- [Simple Walkthrough](#simple-walkthrough)
- [Multiple Parameters](#multiple-parameters)
- [Where Placeholders Can Go](#where-placeholders-can-go)
- [Auto-Detection](#auto-detection)
- [Tips](#tips)

---

## How It Works

1. Add a `sweep:` section to your YAML config with parameter values
2. Add `{placeholder}` markers where you want values substituted
3. Run `srtctl apply -f <config>` - sweep mode is auto-detected
4. `srtctl` generates and submits one job per parameter combination

## Simple Walkthrough

### Step 1: Create a sweep config

```yaml
# configs/concurrency-sweep.yaml
schema: 2
name: "concurrency-sweep"

model:
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  gpus_per_node: 4

engine: sglang
roles:
  prefill:
    nodes: 1
    workers: 1
  decode:
    nodes: 4
    workers: 1

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: "{concurrency}" # <-- placeholder

sweep:
  concurrency: [128, 256, 512] # <-- sweep values
```

### Step 2: Preview with dry-run

```bash
srtctl dry-run -f configs/concurrency-sweep.yaml
```

This shows you what will be generated without submitting. Sweep mode is automatically detected from the `sweep:` section.

### Step 3: Submit

```bash
srtctl apply -f configs/concurrency-sweep.yaml
```

This submits 3 separate jobs, one for each concurrency value (128, 256, 512).

## Multiple Parameters

Multiple parameters create a Cartesian product:

```yaml
roles:
  decode:
    args:
      mem-fraction-static: "{mem}"

benchmark:
  concurrencies: "{conc}"

sweep:
  mem: [0.85, 0.90]
  conc: [256, 512]
```

```bash
srtctl apply -f config.yaml
```

This generates 4 jobs (2 x 2):

- mem=0.85, conc=256
- mem=0.85, conc=512
- mem=0.90, conc=256
- mem=0.90, conc=512

## Where Placeholders Can Go

Placeholders work anywhere in the YAML, quoted so YAML reads them as strings:

```yaml
name: "sweep-{param}"
roles:
  decode:
    args:
      mem-fraction-static: "{mem}"
      dp-size: "{dp}"
benchmark:
  concurrencies: "{conc}"
```

## Auto-Detection

Sweep configs are automatically detected by the presence of a `sweep:` section. You don't need to pass `--sweep` flag:

```bash
# Auto-detected sweep
srtctl apply -f sweep-config.yaml

# Force sweep mode (if auto-detection fails)
srtctl apply -f config.yaml --sweep
```

## Tips

- Always use `srtctl dry-run -f <config>` first to verify
- Start with 2-3 values before running large sweeps
- Cartesian products grow fast: 3 params x 4 values = 64 jobs
- Each job gets a unique name based on parameter values
