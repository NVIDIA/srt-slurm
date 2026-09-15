#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Port a script-harness AgentPerf run directory to an srt-slurm recipe.

Host-side developer tool (not executed inside jobs). Given the ``c<N>/`` run
directory of a standalone disagg-harness AgentPerf run, it reads the readable
artifacts the harness leaves behind and emits:

  * an srt-slurm recipe (server topology, engine configs, worker/frontend env)
  * the agentperf-client workload YAML (settle/timeout/datasets/knobs)
  * a provenance report on stdout: every generated field -> the artifact it
    came from, plus a TODO list of what could not be inferred.

Sources parsed (all verified against real harness runs):
  job_params.env         container image, model path, concurrencies, backend
  ctx_config.yaml        -> backend.trtllm_config.prefill (verbatim, paths rewritten)
  gen_config.yaml        -> backend.trtllm_config.decode  (verbatim, paths rewritten)
  client_cmds_base.sh    pinned agentperf-client checkout path
  client.log             the client's resolved-config banner (workload knobs)
  job.log                srun lines: worker env, frontend env, topology
  agentperf/*traj*.json  settling time (encoded in the output filename stem)
  3_output_CTX_0.log     cpu-pinning detection (taskset -c lines)

Known translations applied automatically:
  DYN_KV_BLOCK_SIZE -> DYN_TRTLLM_KV_BLOCK_SIZE  (harness env is inert under
      srt-slurm; dynamo's own env fallback must carry the block size)
  DYN_UCX_TLS       -> UCX_TLS                    (the harness script exported it)
  path rewrites (default /lustre/ -> /scratch/)   (on clusters where /lustre is
      a symlink, containers only mount /scratch)
  drops: CUDA_VISIBLE_DEVICES, ETCD_ENDPOINTS, PATH, LD_PRELOAD, VIRTUAL_ENV,
      NIXL_PLUGIN_DIR, HOME (srt-slurm or the image manage these)
  frontend --dyn-*-parser flags are NOT ported to frontend.args (the frontend
      rejects them; parsers stay worker-side via env)

Usage:
  python3 port_harness_run.py <old_run_dir> \\
      --out recipe.yaml --workload-out workload.yaml \\
      [--dataset-root DIR ...] [--path-rewrite OLD=NEW ...] [--no-default-rewrites]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

DEFAULT_REWRITES = [("/lustre/", "/scratch/")]
ENV_DROP = {
    "CUDA_VISIBLE_DEVICES",
    "ETCD_ENDPOINTS",
    "PATH",
    "LD_PRELOAD",
    "VIRTUAL_ENV",
    "NIXL_PLUGIN_DIR",
    "HOME",
}
ENV_RENAME = {
    "DYN_KV_BLOCK_SIZE": "DYN_TRTLLM_KV_BLOCK_SIZE",
    "DYN_UCX_TLS": "UCX_TLS",
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SETTLE_RE = re.compile(r"__settle(\d+(?:\.\d+)?)s__")


class Provenance:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []
        self.todos: list[str] = []

    def add(self, field: str, source: str) -> None:
        self.rows.append((field, source))

    def todo(self, msg: str) -> None:
        self.todos.append(msg)

    def report(self) -> str:
        out = ["", "=== provenance ==="]
        out += [f"  {f:<42} <- {s}" for f, s in self.rows]
        if self.todos:
            out += ["", "=== TODO (could not be inferred — edit before submitting) ==="]
            out += [f"  ! {t}" for t in self.todos]
        return "\n".join(out)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def rewrite(value: str, rewrites: list[tuple[str, str]]) -> str:
    for old, new in rewrites:
        if value.startswith(old):
            return new + value[len(old):]
    return value


def rewrite_tree(obj, rewrites):
    """Apply path-prefix rewrites to every string in a nested YAML structure."""
    if isinstance(obj, str):
        return rewrite(obj, rewrites)
    if isinstance(obj, list):
        return [rewrite_tree(v, rewrites) for v in obj]
    if isinstance(obj, dict):
        return {k: rewrite_tree(v, rewrites) for k, v in obj.items()}
    return obj


def absolute_paths(obj) -> list[str]:
    found = []
    if isinstance(obj, str) and obj.startswith("/"):
        found.append(obj)
    elif isinstance(obj, list):
        for v in obj:
            found += absolute_paths(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            found += absolute_paths(v)
    return found


def parse_job_params(run_dir: Path, prov: Provenance) -> dict:
    out: dict = {}
    txt = _read(run_dir / "job_params.env")
    if txt is None:
        prov.todo("job_params.env unreadable: set model.path / model.container by hand")
        return out
    for line in txt.splitlines():
        m = re.match(r'([A-Z_]+)=["\']?([^"\']*)["\']?\s*$', line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    for key in ("MODEL_PATH", "CONTAINER_IMAGE", "CONCURRENCIES", "SERVER_BACKEND"):
        if key in out:
            prov.add(key, "job_params.env")
    return out


def parse_client_dir(run_dir: Path, prov: Provenance) -> str | None:
    txt = _read(run_dir / "client_cmds_base.sh")
    if not txt:
        prov.todo("client_cmds_base.sh unreadable: set benchmark.agentperf_client_dir by hand")
        return None
    m = re.search(r"cd ([^;]+);\s*uv run python agentperf/run\.py", txt)
    if m:
        prov.add("benchmark.agentperf_client_dir", "client_cmds_base.sh (cd before run.py)")
        return m.group(1).strip()
    prov.todo("could not find the client checkout in client_cmds_base.sh")
    return None


def parse_client_banner(run_dir: Path, prov: Provenance) -> dict:
    """Workload knobs from the client's resolved-config startup banner."""
    txt = _read(run_dir / "client.log")
    if not txt:
        prov.todo("client.log unreadable: fill the workload YAML knobs by hand")
        return {}
    knobs: dict = {}
    patterns = {
        "server_type": r"Server type:\s*(\S+)",
        "max_workers": r"Max workers:\s*(\d+)",
        "phase_timeout_seconds": r"Phase Timeout:\s*([\d.]+)s",
        "max_starting_line_offset": r"Max ISL offset:\s*(\d+)",
        "max_tokens": r"Max tokens:\s*(\d+)",
        "reasoning_effort": r"Reasoning effort:\s*(\S+)",
        "seed": r"Seed:\s*(\d+)",
    }
    for line in txt.splitlines():
        line = ANSI.sub("", re.sub(r"^\d+:\s?", "", line))
        for key, pat in patterns.items():
            m = re.search(pat, line)
            if m and key not in knobs:
                v = m.group(1)
                knobs[key] = float(v) if "." in v else (v if not v.isdigit() else int(v))
        if "Conversation routing headers:" in line:
            knobs["send_conversation_routing_headers"] = "enabled" in line
        if "Dynamo conv-aware routing" in line:
            knobs["use_dynamo_conv_aware_routing"] = "enabled" in line
        m = re.search(r"User assignments:\s*(\S+)", line)
        if m:
            knobs["_assignments_basename"] = Path(m.group(1)).name
        m = re.search(r"Loading trajectories from\s*(\S+?)\.{0,3}$", line)
        if m:
            knobs["_trajectory_basename"] = Path(m.group(1)).name
    for k in sorted(k for k in knobs if not k.startswith("_")):
        prov.add(f"workload.{k}", "client.log resolved-config banner")
    return knobs


def parse_settle(run_dir: Path, prov: Provenance) -> float | None:
    for p in sorted((run_dir / "agentperf").glob("*traj*")) if (run_dir / "agentperf").is_dir() else []:
        m = SETTLE_RE.search(p.name)
        if m:
            prov.add("workload.settling_time_seconds", f"output filename stem ({p.name})")
            return float(m.group(1))
    prov.todo("settling_time_seconds not recoverable (no agentperf/*traj* outputs); it is YAML-only — set it by hand")
    return None


def _quoted_env_tokens(line: str) -> list[str]:
    """Pick the single-quoted argument that looks like a KEY=val env token list."""
    best: list[str] = []
    for quoted in re.findall(r"'([^']*)'", line):
        toks = quoted.split()
        if toks and sum(bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)) for t in toks) >= max(2, len(toks) - 1):
            if len(toks) > len(best):
                best = toks
    return best


def translate_env(tokens: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for tok in tokens:
        key, _, val = tok.partition("=")
        if key in ENV_DROP:
            continue
        env[ENV_RENAME.get(key, key)] = val
    # The harness set no DYN_LOG; srt-slurm setdefaults a noisier filter on
    # workers. Pin plain 'info' so the ported run matches the reference logs.
    env.setdefault("DYN_LOG", "info")
    return env


def parse_job_log(run_dir: Path, prov: Provenance) -> dict:
    """Worker/frontend env + topology from the harness's srun launch lines."""
    txt = _read(run_dir / "job.log")
    out: dict = {"prefill_env": {}, "decode_env": {}, "frontend_env": {}, "topology": {}}
    if not txt:
        prov.todo("job.log unreadable: set worker/frontend env and resources by hand")
        return out
    ctx_nodes, gen_nodelists, gpus_per_node = set(), [], None
    for line in txt.splitlines():
        if "Executing: srun" not in line:
            continue
        nodelist = re.search(r"--nodelist[ =]([\w.,-]+)", line)
        if re.search(r"3_output_CTX_\d+\.log", line):
            toks = _quoted_env_tokens(line)
            if toks and not out["prefill_env"]:
                out["prefill_env"] = translate_env(toks)
                prov.add("backend.prefill_environment", "job.log CTX srun line (translated)")
                cvd = next((t for t in toks if t.startswith("CUDA_VISIBLE_DEVICES=")), None)
                if cvd:
                    gpus_per_node = len(cvd.split("=", 1)[1].split(","))
            if nodelist:
                ctx_nodes.add(nodelist.group(1))
        elif re.search(r"3_output_GEN_\d+\.log", line):
            toks = _quoted_env_tokens(line)
            if toks and not out["decode_env"]:
                out["decode_env"] = translate_env(toks)
                prov.add("backend.decode_environment", "job.log GEN srun line (translated)")
            if nodelist:
                gen_nodelists.append(nodelist.group(1).split(","))
        elif "4_output_frontend" in line:
            m = re.search(r"--export=ALL,(\S+)", line)
            if m and not out["frontend_env"]:
                env = {}
                for tok in m.group(1).split(","):
                    key, _, val = tok.partition("=")
                    if key and key not in ENV_DROP:
                        env[key] = val
                # The harness's infra script exported HOME=/tmp unconditionally
                # (the image HOME is read-only); carry that for the frontend.
                env.setdefault("HOME", "/tmp")
                out["frontend_env"] = env
                prov.add("frontend.env", "job.log frontend srun line --export list")
    out["topology"] = {
        "prefill_workers": len(ctx_nodes),
        "prefill_nodes": len(ctx_nodes),
        "decode_workers": len(gen_nodelists),
        "decode_nodes": len(gen_nodelists[0]) if gen_nodelists else None,
        "gpus_per_node": gpus_per_node,
    }
    prov.add("resources (topology)", "job.log srun nodelists")
    return out


def detect_cpu_pinning(run_dir: Path, prov: Provenance) -> bool:
    txt = _read(run_dir / "3_output_CTX_0.log")
    if txt and re.search(r"taskset -c \d", txt):
        prov.add("backend.numa_cpu_bind=true", "3_output_CTX_0.log (taskset -c lines present)")
        return True
    return False


def find_dataset(basename: str | None, roots: list[Path], prov: Provenance, label: str) -> str:
    placeholder = f"/TODO/path/to/{basename or label}"
    if not basename:
        prov.todo(f"{label}: not found in client.log; set it by hand")
        return placeholder
    for root in roots:
        hits = list(root.rglob(basename)) if root.is_dir() else []
        if hits:
            prov.add(f"workload.{label}", f"found under --dataset-root {root}")
            return str(hits[0])
    prov.todo(
        f"{label}: the banner only shows the node-local staged copy ({basename}); "
        f"locate the original (try --dataset-root) and replace the placeholder"
    )
    return placeholder


def build_frontend_args(frontend_env: dict[str, str]) -> dict:
    """Reconstruct the dynamo.frontend flags the harness's infra script derived from env."""
    return {
        "router-mode": frontend_env.get("ROUTER_MODE", "kv"),
        "no-router-kv-events": frontend_env.get("DYN_FRONTEND_ENABLE_KV_EVENTS", "0") != "1",
        "kv-cache-block-size": int(frontend_env.get("DYN_KV_BLOCK_SIZE", "128")),
        "enforce-disagg": True,
        "request-plane": frontend_env.get("DYN_REQUEST_PLANE", "tcp"),
        "event-plane": "zmq",
        # NO dyn-*-parser flags: the frontend rejects them; parsers are worker-side env.
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="the harness c<N>/ run directory")
    ap.add_argument("--out", type=Path, required=True, help="output recipe YAML path")
    ap.add_argument("--workload-out", type=Path, required=True, help="output agentperf workload YAML path")
    ap.add_argument("--dataset-root", type=Path, action="append", default=[],
                    help="root(s) to search for the original trajectory/assignments files")
    ap.add_argument("--path-rewrite", action="append", default=[], metavar="OLD=NEW",
                    help="path prefix rewrite applied to all ported paths (repeatable)")
    ap.add_argument("--no-default-rewrites", action="store_true",
                    help=f"disable the default rewrites {DEFAULT_REWRITES}")
    args = ap.parse_args(argv)

    rewrites = [] if args.no_default_rewrites else list(DEFAULT_REWRITES)
    for spec in args.path_rewrite:
        old, _, new = spec.partition("=")
        if not old or not new:
            ap.error(f"--path-rewrite must be OLD=NEW, got: {spec}")
        rewrites.append((old, new))

    run_dir, prov = args.run_dir, Provenance()
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        return 1

    params = parse_job_params(run_dir, prov)
    if params.get("SERVER_BACKEND", "dynamo") != "dynamo":
        prov.todo(f"SERVER_BACKEND={params.get('SERVER_BACKEND')}: this porter only handles the dynamo "
                  "frontend path; the trtllm-serve path needs a hand-written recipe")

    ctx = yaml.safe_load(_read(run_dir / "ctx_config.yaml") or "") or {}
    gen = yaml.safe_load(_read(run_dir / "gen_config.yaml") or "") or {}
    if ctx:
        prov.add("backend.trtllm_config.prefill", "ctx_config.yaml (verbatim, paths rewritten)")
    else:
        prov.todo("ctx_config.yaml unreadable")
    if gen:
        prov.add("backend.trtllm_config.decode", "gen_config.yaml (verbatim, paths rewritten)")
    else:
        prov.todo("gen_config.yaml unreadable")
    ctx, gen = rewrite_tree(ctx, rewrites), rewrite_tree(gen, rewrites)
    for p in absolute_paths({"prefill": ctx, "decode": gen}):
        prov.todo(f"engine config references {p} — confirm it is readable from the compute containers "
                  "(copy to your own scratch if it lives under another user)")

    client_dir = parse_client_dir(run_dir, prov)
    knobs = parse_client_banner(run_dir, prov)
    settle = parse_settle(run_dir, prov)
    joblog = parse_job_log(run_dir, prov)
    cpu_pinned = detect_cpu_pinning(run_dir, prov)

    concurrencies = [int(c) for c in str(params.get("CONCURRENCIES", "")).split()] or None
    if concurrencies is None:
        prov.todo("CONCURRENCIES not found in job_params.env; set benchmark.concurrency by hand")

    workload = {
        "base_url": "http://placeholder:8000/v1",
        "api_key": "dummy",
        "model": "placeholder",
        "server_type": knobs.get("server_type", "trtllm"),
        "concurrencies": concurrencies or [1],
        "phase_timeout_seconds": knobs.get("phase_timeout_seconds", 3600.0),
        "settling_time_seconds": settle if settle is not None else 240.0,
        "trajectory_path": find_dataset(knobs.get("_trajectory_basename"), args.dataset_root, prov, "trajectory_path"),
        "user_assignments_path": find_dataset(knobs.get("_assignments_basename"), args.dataset_root, prov,
                                              "user_assignments_path"),
        "max_starting_line_offset": knobs.get("max_starting_line_offset", 10),
        "seed": knobs.get("seed", 42),
        "timeout_seconds": 300.0,
        "max_workers": knobs.get("max_workers", 8),
        "max_tokens": knobs.get("max_tokens", 16384),
        "reasoning_effort": knobs.get("reasoning_effort"),
        "client_impl": "rust",
        "send_conversation_routing_headers": knobs.get("send_conversation_routing_headers", False),
        "use_dynamo_conv_aware_routing": knobs.get("use_dynamo_conv_aware_routing", False),
        "power_dcgm_url": None,
    }

    topo = joblog["topology"]
    model_path = rewrite(params.get("MODEL_PATH", "/TODO/model/path"), rewrites)
    recipe = {
        "name": re.sub(r"[^a-zA-Z0-9-]+", "-", run_dir.resolve().parent.name)[:64] or "agentperf-ported-run",
        "model": {
            "path": model_path,
            "container": rewrite(params.get("CONTAINER_IMAGE", "/TODO/container.sqsh"), rewrites),
            "precision": "fp4" if "fp4" in model_path.lower() or "nvfp4" in model_path.lower() else "fp8",
        },
        "resources": {
            "gpu_type": "gb300",  # TODO'd below — not recoverable from the run dir
            "gpus_per_node": topo.get("gpus_per_node") or 4,
            "prefill_nodes": topo.get("prefill_nodes") or 1,
            "prefill_workers": topo.get("prefill_workers") or 1,
            "decode_nodes": topo.get("decode_nodes") or 1,
            "decode_workers": topo.get("decode_workers") or 1,
        },
        "dynamo": {"install": False, "request_plane": joblog["frontend_env"].get("DYN_REQUEST_PLANE", "tcp"),
                   "event_plane": "zmq"},
        "backend": {
            "type": "trtllm",
            "numa_memory_bind": True,
            "numa_cpu_bind": cpu_pinned,
            "prefill_environment": joblog["prefill_env"],
            "decode_environment": joblog["decode_env"],
            "trtllm_config": {"prefill": ctx, "decode": gen},
        },
        "frontend": {
            "type": "dynamo",
            "enable_multiple_frontends": False,
            "args": build_frontend_args(joblog["frontend_env"]),
            "env": joblog["frontend_env"],
        },
        "infra": {"etcd_nats_dedicated_node": False},
        "observability": {"enabled": False},
        "benchmark": {
            "type": "agentperf",
            "client_placement": "last_decode",
            "concurrency": concurrencies[0] if concurrencies and len(concurrencies) == 1 else None,
            "concurrencies": concurrencies if concurrencies and len(concurrencies) > 1 else None,
            "agentperf_client_dir": rewrite(client_dir, rewrites) if client_dir else "/TODO/agentperf-client",
            "agentperf_config": str(args.workload_out.resolve()),
        },
        "extra_mount": ["/scratch:/scratch"],
    }
    recipe["benchmark"] = {k: v for k, v in recipe["benchmark"].items() if v is not None}
    prov.todo("resources.gpu_type is guessed as gb300 — set it for your cluster")
    prov.todo(f"benchmark.agentperf_config is set to {args.workload_out.resolve()} — "
              "move the workload YAML somewhere container-visible and update the path")
    prov.todo("review benchmark.client_placement (last_decode = client on the decode leader node, "
              "matching the harness's client-on-GEN placement)")

    args.out.write_text(yaml.safe_dump(recipe, sort_keys=False))
    args.workload_out.write_text(yaml.safe_dump(workload, sort_keys=False))
    print(f"wrote recipe:   {args.out}")
    print(f"wrote workload: {args.workload_out}")
    print(prov.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
