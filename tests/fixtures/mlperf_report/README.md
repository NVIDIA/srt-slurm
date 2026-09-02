# inference-endpoint report fixtures

Unedited output from real MLPerf runs, copied from `endpoints-launch/sflow_output`
on the hecate cluster. Kept verbatim so the parser is tested against what the
client actually wrote, not against a hand-made approximation of it.

| File | Source run |
|---|---|
| `performance/result_summary.json` | `474440-trtllm_disagg_dynamo_perf_only-20260825` (perf only) |
| `accuracy/accuracy_results.json` | `412798-trtllm_disagg_dynamo_perf_only-20260813` (has an accuracy phase) |

**The two files come from different runs.** The most recent run was performance
only, so its directory has no accuracy report; the accuracy file is from the most
recent run that produced one. Their numbers therefore do not describe a single
run, and no test should assert a relationship between the two.

The directory layout mirrors the newer of the two layouts observed. An older run
(`286161`, July) wrote `result_summary.json` at the top level instead — that
variant is covered by constructed cases in `test_mlperf.py` rather than by a
second copied fixture.

Expected values in the tests are cross-checked against the `report.txt` that the
client wrote beside each summary, so they assert agreement with the client's own
rendering of its numbers.
