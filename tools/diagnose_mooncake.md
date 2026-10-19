# Diagnose Mooncake memory transfers on available nodes

Submit from the repository root on the cluster:

```bash
sbatch tools/diagnose_mooncake.sbatch
```

Keep `tools/diagnose_mooncake.py` beside the batch script. The job copies both
files into its run directory before launching container steps.

The defaults request account `root`, partition `batch-r42`, exclusive access to
`vr-r42-cn[01-03,05-13,15-18]` (16 nodes; cn04 excluded), and 25 minutes. The job waits
for the requested nodes to become available. This tests host memory and RDMA;
it does not request GPUs or load GLM-5.3. Each step has 64 CPUs for Mooncake's
multi-NIC progress threads. The container defaults to:

```text
/home/_shared/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh
```

Four fresh master/donor cases compare cn05 and cn02, each with 512 GiB and
1100 GiB of donated memory. The other 14 nodes are readers in every case. Each
reader uses two threads and batches of 512 objects of 2,006,016 bytes, matching
the observed cache-page size and batch configuration. Clients have no storage
segments, so keys must be allocated on the case's donor. The script uses the
same eight NICs and Mooncake environment settings as the recipe.

The diagnostic records registration, seed writes, verified reads, per-key error
codes, batch latencies, donor process/cgroup memory health, InfiniBand counters,
external TCP reachability of the actual handshake listener, master metrics, and
Slurm step exit status. Readers finish before the donor is stopped. All stop
requests are timestamped so teardown errors can be separated from traffic errors.

Outputs are saved under:

```text
outputs/mooncake-diag-<job_id>/logs/
```

Read `perf-analysis.md`, `events.log`, each case's `donor.err`,
`donor-health.jsonl`, `external-health.jsonl`, and `client-*.jsonl` together.
TTFT, ITL, token throughput and agentic metrics are explicitly N/A for this
host-only test. Missing kernel/cgroup evidence is recorded rather than treated
as a clean result.

This is a reduced diagnostic, sized with the repository's
[job-sizing skill](../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md).
A pass does not reproduce or clear the original failure after approximately
40 minutes: there is no decoder/model memory pressure. Registering 1100 GiB
with a small working set also does not prove that the second memory-registration
chunk was read. Key replica addresses are recorded for coverage inspection.

Environment overrides are optional:

```bash
MC_DIAG_SECONDS=180 sbatch tools/diagnose_mooncake.sbatch
```

`MC_DIAG_IMAGE`, `MC_DIAG_BATCH_SIZE`, `MC_DIAG_CLIENT_WORKERS`,
`MC_DIAG_OUTPUT_ROOT`, and `MC_DIAG_HELPER` override their corresponding defaults.
Keep the total workload within the allocation's walltime. For a smaller first
check, retain both donor candidates and the original prefill nodes:

```bash
sbatch --nodes=5 --nodelist=vr-r42-cn[02-03,05,17-18] tools/diagnose_mooncake.sbatch
```
