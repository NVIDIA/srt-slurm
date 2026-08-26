# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-run bridge: a finished run's log dir -> component perf dashboard.

Runs on **every** job, with no opt-in. Reading a run used to mean hand-driving two
scripts from a checkout against a log dir; this module runs them at the end of the
job instead, so one submission produces the page.

Unconditional because the page is not a special-occasion artifact: the question it
answers -- where did the time go, which component was the ceiling -- is the one
asked of every run, and it is asked *after* the run, when opting in is no longer
possible. A knob would only ever be discovered by the person who already knew.

``observability.enabled`` decides which capture legs exist and therefore which tabs
the page carries; it never decides whether the page exists. A run with no
server-side capture at all still renders from the client's own metrics export, the
per-iteration log and the frontend log -- which is the shape most runs have.

It drives the two vendored layers as SUBPROCESSES:

    L2  python3 -m src.ingest.ingest                 log dir -> bundle
    L3  python3 -m src.visualization.build_dynamo_bench_dash   bundle -> HTML + JSON

Subprocess rather than import, for three reasons: the renderer is a script with
module-level argparse and no importable entry point; the vendored tree is
deliberately excluded from lint/typecheck and is not part of the ``srtctl`` wheel;
and a crash in third-party rendering code must not be able to take down the job's
post-processing.

Best-effort by construction, matching ``srtctl.analysis.metrics_scraper``: every
failure path is logged and swallowed. Visualisation is never a hard dependency of a
benchmark that has already produced its results.

Outputs, all under the run's log dir:

    perf_dashboard.html        self-contained page (D3 inlined)
    perf_dashboard.json        the page's DATA payload, for diffing/CI/no-browser reads
    perf_dashboard_bundle/     the intermediate schemas, kept deliberately (see below)

The bundle is kept rather than cleaned up because it is the reproducible middle of
the pipeline: re-rendering from it is seconds, whereas re-deriving it means another
pass over multi-GB logs -- and on a run whose logs have since been deleted, it is the
only surviving form of the data.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

# Ingest walks every worker log to pre-grep SPAN_CLOSED lines; on a multi-hour run
# those are tens of GB. The pre-grep is parallel and grep-backed, but give it room.
INGEST_TIMEOUT_SEC = 1800
# Rendering is pure Python over the bundle -- fast, but a very long run can carry
# millions of client records.
RENDER_TIMEOUT_SEC = 900

BUNDLE_DIRNAME = "perf_dashboard_bundle"
HTML_FILENAME = "perf_dashboard.html"
JSON_FILENAME = "perf_dashboard.json"


def find_repo_root() -> Path | None:
    """Locate the checkout holding the vendored ``src/ingest`` + ``src/visualization``.

    Those live at the repo root and are NOT installed into the ``srtctl`` wheel
    (``pyproject.toml`` packages only ``src/srtctl``), so they have to be found on
    disk. Two strategies, in order:

    1. Relative to the installed ``srtctl`` package. Under the editable install that
       compute nodes use, ``srtctl/__init__.py`` is at ``<root>/src/srtctl/``, so the
       root is two parents up. This is the common case and needs no configuration.
    2. ``srtctl_root`` from ``srtslurm.yaml`` -- the same setting
       ``PostProcessStageMixin._export_node_metrics_csv`` uses to reach ``analysis/``.

    Both are validated by checking the vendored entry point actually exists, so a
    non-editable install falling back to a site-packages path is rejected rather than
    producing a confusing ``No module named src.ingest`` later.
    """
    candidates: list[Path] = []

    try:
        import srtctl

        pkg = Path(srtctl.__file__).resolve()
        candidates.append(pkg.parents[2])  # <root>/src/srtctl/__init__.py -> <root>
    except Exception as e:  # noqa: BLE001 - discovery is best-effort
        logger.debug("perf dashboard: package-relative root discovery failed: %s", e)

    try:
        from srtctl.core.config import get_srtslurm_setting

        configured = get_srtslurm_setting("srtctl_root")
        if configured:
            candidates.append(Path(configured))
    except Exception as e:  # noqa: BLE001
        logger.debug("perf dashboard: srtslurm.yaml root discovery failed: %s", e)

    for root in candidates:
        try:
            if (root / "src" / "ingest" / "ingest.py").is_file() and (
                root / "src" / "visualization" / "build_dynamo_bench_dash.py"
            ).is_file():
                return root.resolve()
        except OSError:
            continue

    logger.warning(
        "perf dashboard: could not locate the vendored src/ingest + src/visualization tree "
        "(tried: %s); skipping dashboard build",
        ", ".join(str(c) for c in candidates) or "no candidates",
    )
    return None


def _worker_specs(config: SrtConfig) -> list[str]:
    """``--worker ROLE=PARALLELISM:RANK:COUNT`` args describing the run's topology.

    This is what gives the page its GPU count: the renderer sums ``rank x worker_count``
    for the tok/s/GPU denominator and otherwise falls back to 1 GPU with a warning,
    which would silently inflate every per-GPU number by the size of the job.

    ``rank`` is GPUs-per-worker. ``ResourceConfig`` exposes that as the computed
    ``gpus_per_*`` properties, so explicit recipe overrides are honoured for free.
    Parallelism is reported as a label only; it is not used in any arithmetic, so an
    imprecise tag cannot corrupt a number.
    """
    res = config.resources
    specs: list[str] = []
    for role, count_attr, gpus_attr, label in (
        ("prefill", "num_prefill", "gpus_per_prefill", "dep"),
        ("decode", "num_decode", "gpus_per_decode", "tep"),
        ("agg", "num_agg", "gpus_per_agg", "tep"),
    ):
        count = getattr(res, count_attr, 0) or 0
        gpus = getattr(res, gpus_attr, 0) or 0
        if count and gpus:
            specs.append(f"{role}={label}:{gpus}:{count}")
    return specs


def _run(argv: list[str], cwd: Path, timeout: int, stage: str) -> bool:
    """Run one pipeline stage, logging its output. Returns success."""
    logger.info("perf dashboard [%s]: %s", stage, " ".join(argv))
    try:
        result = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        logger.warning("perf dashboard [%s]: timed out after %ds", stage, timeout)
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning("perf dashboard [%s]: failed to launch: %s", stage, e)
        return False

    # Both stages log progress to stderr at INFO; surface it so the sweep log records
    # what was found (source availability, scrape counts, resolved engine ceilings).
    for line in (result.stderr or "").rstrip().splitlines():
        logger.info("perf dashboard [%s] %s", stage, line)
    if result.returncode != 0:
        for line in (result.stdout or "").rstrip().splitlines()[-20:]:
            logger.warning("perf dashboard [%s] %s", stage, line)
        logger.warning("perf dashboard [%s]: exit %d", stage, result.returncode)
        return False
    return True


def build(config: SrtConfig, runtime: RuntimeContext) -> Path | None:
    """Build the component dashboard for a finished run. Returns the HTML path or None.

    Never raises: a visualisation failure must not change the outcome of a benchmark
    that has already produced its results.
    """
    root = find_repo_root()
    if root is None:
        return None

    log_dir = Path(runtime.log_dir)
    bundle = log_dir / BUNDLE_DIRNAME
    html = log_dir / HTML_FILENAME

    ingest_argv = [
        sys.executable,
        "-m",
        "src.ingest.ingest",
        "--run-dir",
        str(log_dir),
        "--out",
        str(bundle),
        "--name",
        f"{runtime.job_id} {config.name}".strip(),
    ]
    for spec in _worker_specs(config):
        ingest_argv += ["--worker", spec]
    if not _run(ingest_argv, root, INGEST_TIMEOUT_SEC, "ingest"):
        return None

    render_argv = [
        sys.executable,
        "-m",
        "src.visualization.build_dynamo_bench_dash",
        str(bundle),
        str(html),
        "--dump-json",
        str(log_dir / JSON_FILENAME),
    ]
    # The Log-analysis tab needs the frontend log, which the bundle does not contain
    # (it stays in the log dir -- it is the raw multi-GB artifact, not an intermediate
    # schema). Pass the first one found; correspondence with the bundle is enforced by
    # the renderer on the x_request_id pivot, so a wrong file fails loudly.
    frontend_logs = sorted(log_dir.glob("*_frontend_*.out"))
    if frontend_logs:
        render_argv += ["--frontend-log", str(frontend_logs[0])]
    else:
        logger.info("perf dashboard: no *_frontend_*.out in %s; Log-analysis tab omitted", log_dir)

    if not _run(render_argv, root, RENDER_TIMEOUT_SEC, "render"):
        return None

    if not html.is_file():
        logger.warning("perf dashboard: render reported success but %s is missing", html)
        return None
    logger.info("perf dashboard: %s", html)
    return html


def try_build(config: SrtConfig, runtime: RuntimeContext) -> Path | None:
    """Build the dashboard for a finished run. Returns the HTML path, or None.

    Single entry point for :class:`PostProcessStageMixin`, so the mixin stays free of
    analysis-package internals -- the same contract as
    :func:`srtctl.analysis.metrics_scraper.try_start_raw_scraper`.

    Unconditional: there is no opt-in to check. What differs between runs is which
    legs :func:`build` finds, and that is the ingest's decision to make from the log
    dir rather than a recipe's to declare in advance.

    With no gate left, this ``except`` is the only thing standing between a bug in
    third-party rendering code and the post-processing of a benchmark that has
    already produced its results -- so it stays broad on purpose.
    """
    try:
        return build(config, runtime)
    except Exception as e:  # noqa: BLE001 - visualisation is never fatal
        logger.warning("perf dashboard: skipped after error: %s", e)
        return None
