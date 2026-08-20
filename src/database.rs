//! SQLite materialization for the small, portable static viewer.

use crate::artifacts::Manifest;
use crate::model::{Event, EventKind, LogSource};
use crate::tables::{AiperfRequest, RequestTrace, WorkerMetricSample};
use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::collections::BTreeMap;
use std::path::Path;

const ROUTER_PREFILL: &str = "dynamo_frontend_worker_active_prefill_tokens";
const ROUTER_TTFT: &str = "dynamo_frontend_worker_last_time_to_first_token_seconds";
const ROUTER_ITL: &str = "dynamo_frontend_worker_last_inter_token_latency_seconds";
const ENGINE_RUNNING: &str = "sglang:num_running_reqs";
const ENGINE_QUEUED: &str = "sglang:num_queue_reqs";
const ENGINE_KV_USED: &str = "dynamo_component_gpu_cache_usage_percent";

#[derive(Clone, Copy)]
struct MetricPoint {
    timestamp_ns: i64,
    value: f64,
}

pub fn write_database(
    path: &Path,
    manifest: &Manifest,
    events: &[Event],
    aiperf_requests: &[AiperfRequest],
    request_traces: &[RequestTrace],
    worker_metric_samples: &[WorkerMetricSample],
) -> Result<()> {
    let mut connection =
        Connection::open(path).with_context(|| format!("open {}", path.display()))?;
    connection.execute_batch(
        "
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = NORMAL;
        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS aiperf_requests;
        DROP TABLE IF EXISTS request_traces;
        DROP TABLE IF EXISTS routing_decisions;
        DROP TABLE IF EXISTS routing_candidates;
        DROP TABLE IF EXISTS worker_state_snapshots;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE aiperf_requests (
            arm TEXT NOT NULL, request_id TEXT, credit_issued_ns INTEGER, request_start_ns INTEGER,
            request_end_ns INTEGER, input_tokens INTEGER, output_tokens INTEGER, ttft_ms REAL,
            itl_ms REAL, e2e_ms REAL
        );
        CREATE TABLE request_traces (
            arm TEXT NOT NULL, request_id TEXT, x_request_id TEXT, event_time_ns INTEGER,
            request_received_ns INTEGER, input_tokens INTEGER, output_tokens INTEGER,
            cached_tokens INTEGER, kv_hit_rate REAL, ttft_ms REAL, total_time_ms REAL,
            avg_itl_ms REAL, queue_depth INTEGER, prefill_worker_id TEXT, prefill_dp_rank INTEGER,
            decode_worker_id TEXT, decode_dp_rank INTEGER, raw_json TEXT NOT NULL
        );
        CREATE TABLE routing_decisions (
            timestamp_ns INTEGER, x_request_id TEXT, dynamo_request_id TEXT,
            worker_id TEXT, dp_rank INTEGER,
            overlap_blocks INTEGER, total_blocks INTEGER, lower_prefix_selected INTEGER NOT NULL,
            raw TEXT NOT NULL
        );
        CREATE TABLE routing_candidates (
            dynamo_request_id TEXT NOT NULL, timestamp_ns INTEGER NOT NULL,
            worker_id TEXT NOT NULL, dp_rank INTEGER,
            cost_blocks REAL, effective_cached_blocks REAL,
            prefill_load_scale REAL, adjusted_prefill_blocks REAL, raw_prefill_blocks REAL,
            overlap_credit_blocks REAL, overlap_credit_decay REAL,
            decode_blocks REAL, active_request_cost_blocks REAL,
            PRIMARY KEY(dynamo_request_id, worker_id, dp_rank)
        );
        CREATE TABLE worker_state_snapshots (
            dynamo_request_id TEXT NOT NULL, worker_id TEXT NOT NULL,
            router_sample_age_ms REAL, active_prefill_tokens REAL, last_ttft_ms REAL, last_itl_ms REAL,
            engine_sample_age_ms REAL, running_reqs REAL, queued_reqs REAL, gpu_cache_usage_fraction REAL,
            PRIMARY KEY(dynamo_request_id, worker_id)
        );
        CREATE INDEX request_traces_time ON request_traces(request_received_ns, event_time_ns);
        CREATE INDEX routing_decisions_time ON routing_decisions(timestamp_ns);
        CREATE INDEX routing_candidates_request ON routing_candidates(dynamo_request_id, cost_blocks);
        CREATE INDEX worker_state_request ON worker_state_snapshots(dynamo_request_id);
        ",
    )?;

    let decision_times = routing_decision_times(events);
    let worker_indices = worker_indices(events);
    let metric_series = metric_series(worker_metric_samples);

    let transaction = connection.transaction()?;
    transaction.execute(
        "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
        params!["run_root", &manifest.run_root],
    )?;
    transaction.execute(
        "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
        params!["schema_version", manifest.schema_version.to_string()],
    )?;
    for arm in &manifest.arms {
        if let Some(settings) = &arm.router_settings {
            transaction.execute(
                "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
                params![
                    format!("{}.router_settings", arm.arm.name()),
                    serde_json::to_string(settings)?
                ],
            )?;
        }
    }
    {
        let mut insert = transaction.prepare(
            "INSERT INTO aiperf_requests VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )?;
        for row in aiperf_requests {
            insert.execute(params![
                row.arm,
                row.request_id,
                row.credit_issued_ns,
                row.request_start_ns,
                row.request_end_ns,
                row.input_tokens,
                row.output_tokens,
                row.ttft_ms,
                row.itl_ms,
                row.e2e_ms,
            ])?;
        }
    }
    {
        let mut insert = transaction.prepare(
            "INSERT INTO request_traces VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
        )?;
        for row in request_traces {
            insert.execute(params![
                row.arm,
                row.request_id,
                row.x_request_id,
                row.event_time_ns,
                row.request_received_ns,
                row.input_tokens,
                row.output_tokens,
                row.cached_tokens,
                row.kv_hit_rate,
                row.ttft_ms,
                row.total_time_ms,
                row.avg_itl_ms,
                row.queue_depth,
                row.prefill_worker_id,
                row.prefill_dp_rank,
                row.decode_worker_id,
                row.decode_dp_rank,
                row.raw_json,
            ])?;
        }
    }
    {
        let mut insert = transaction
            .prepare("INSERT INTO routing_decisions VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)")?;
        for event in events.iter().filter(|event| {
            event.source == LogSource::DynamoRouter && event.kind == EventKind::RoutingDecision
        }) {
            insert.execute(params![
                event.timestamp_ns,
                event.request_id,
                event.fields.get("dynamo_request_id"),
                event.fields.get("worker_id"),
                integer_field(event, "dp_rank"),
                integer_field(event, "overlap_blocks"),
                integer_field(event, "total_blocks"),
                0,
                event.raw,
            ])?;
        }
    }
    {
        let mut insert_candidate = transaction.prepare(
            "INSERT OR REPLACE INTO routing_candidates VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        )?;
        let mut insert_snapshot = transaction.prepare(
            "INSERT OR REPLACE INTO worker_state_snapshots VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )?;
        for event in events.iter().filter(|event| {
            event.source == LogSource::DynamoRouter && event.kind == EventKind::RoutingFormula
        }) {
            let Some(dynamo_request_id) = event.request_id.as_deref() else {
                continue;
            };
            let Some(&decision_timestamp_ns) = decision_times.get(dynamo_request_id) else {
                continue;
            };
            // The formula is emitted immediately before selection, so it is
            // the only honest time at which to sample pre-routing state.
            let timestamp_ns = event.timestamp_ns.unwrap_or(decision_timestamp_ns);
            let Some(worker_id) = event.fields.get("worker_id") else {
                continue;
            };
            insert_candidate.execute(params![
                dynamo_request_id,
                timestamp_ns,
                worker_id,
                integer_field(event, "dp_rank"),
                float_field(event, "cost_blocks"),
                float_field(event, "effective_cached_blocks"),
                float_field(event, "prefill_load_scale"),
                float_field(event, "adjusted_prefill_blocks"),
                float_field(event, "raw_prefill_blocks"),
                float_field(event, "overlap_credit_blocks"),
                float_field(event, "overlap_credit_decay"),
                float_field(event, "decode_blocks"),
                float_field(event, "active_request_cost_blocks"),
            ])?;
            let snapshot = snapshot(
                &metric_series,
                &worker_indices,
                dynamo_request_id,
                worker_id,
                timestamp_ns,
            );
            insert_snapshot.execute(params![
                dynamo_request_id,
                worker_id,
                snapshot.router_sample_age_ms,
                snapshot.active_prefill_tokens,
                snapshot.last_ttft_ms,
                snapshot.last_itl_ms,
                snapshot.engine_sample_age_ms,
                snapshot.running_reqs,
                snapshot.queued_reqs,
                snapshot.gpu_cache_usage_fraction,
            ])?;
        }
    }
    transaction.execute(
        "
        UPDATE routing_decisions
        SET lower_prefix_selected = EXISTS(
            SELECT 1
            FROM routing_candidates selected
            JOIN routing_candidates candidate
              ON candidate.dynamo_request_id = selected.dynamo_request_id
            WHERE selected.dynamo_request_id = routing_decisions.dynamo_request_id
              AND selected.worker_id = routing_decisions.worker_id
              AND candidate.effective_cached_blocks > selected.effective_cached_blocks
        )
        ",
        [],
    )?;
    transaction.commit()?;
    Ok(())
}

fn routing_decision_times(events: &[Event]) -> BTreeMap<String, i64> {
    events
        .iter()
        .filter(|event| {
            event.source == LogSource::DynamoRouter && event.kind == EventKind::RoutingDecision
        })
        .filter_map(|event| {
            Some((
                event.fields.get("dynamo_request_id")?.clone(),
                event.timestamp_ns?,
            ))
        })
        .collect()
}

/// Tie an opaque Dynamo frontend worker ID to the `worker-N.log` which saw it.
fn worker_indices(events: &[Event]) -> BTreeMap<String, u32> {
    let mut indices = BTreeMap::new();
    for event in events.iter().filter(|event| {
        event.source == LogSource::DynamoWorker && event.kind == EventKind::WorkerRequest
    }) {
        let (Some(worker_id), Some(worker_index)) =
            (event.fields.get("instance_id"), event.worker_index)
        else {
            continue;
        };
        indices.entry(worker_id.clone()).or_insert(worker_index);
    }
    indices
}

type MetricSeries = BTreeMap<(String, String), Vec<MetricPoint>>;

fn metric_series(samples: &[WorkerMetricSample]) -> MetricSeries {
    let mut series = MetricSeries::new();
    for sample in samples {
        let subject = if sample.scraper_endpoint == "router" {
            let Some(worker_id) = sample.router_worker_id.as_ref() else {
                continue;
            };
            format!("router:{worker_id}")
        } else {
            format!("engine:{}", sample.scraper_endpoint)
        };
        series
            .entry((subject, sample.metric_name_clean.clone()))
            .or_default()
            .push(MetricPoint {
                timestamp_ns: sample.timestamp_ns,
                value: sample.metric_value,
            });
    }
    for points in series.values_mut() {
        points.sort_by_key(|point| point.timestamp_ns);
    }
    series
}

#[derive(Default)]
struct WorkerSnapshot {
    router_sample_age_ms: Option<f64>,
    active_prefill_tokens: Option<f64>,
    last_ttft_ms: Option<f64>,
    last_itl_ms: Option<f64>,
    engine_sample_age_ms: Option<f64>,
    running_reqs: Option<f64>,
    queued_reqs: Option<f64>,
    gpu_cache_usage_fraction: Option<f64>,
}

fn snapshot(
    series: &MetricSeries,
    worker_indices: &BTreeMap<String, u32>,
    _dynamo_request_id: &str,
    worker_id: &str,
    decision_ns: i64,
) -> WorkerSnapshot {
    let router_subject = format!("router:{worker_id}");
    let prefill = sample_before(series, &router_subject, ROUTER_PREFILL, decision_ns);
    let last_ttft = sample_before(series, &router_subject, ROUTER_TTFT, decision_ns);
    let last_itl = sample_before(series, &router_subject, ROUTER_ITL, decision_ns);

    let engine_subject = worker_indices
        .get(worker_id)
        .map(|index| format!("engine:worker_agg_{index}_0"));
    let running = engine_subject
        .as_deref()
        .and_then(|subject| sample_before(series, subject, ENGINE_RUNNING, decision_ns));
    let queued = engine_subject
        .as_deref()
        .and_then(|subject| sample_before(series, subject, ENGINE_QUEUED, decision_ns));
    let kv_used = engine_subject
        .as_deref()
        .and_then(|subject| sample_before(series, subject, ENGINE_KV_USED, decision_ns));

    WorkerSnapshot {
        router_sample_age_ms: max_age_ms(decision_ns, [prefill, last_ttft, last_itl]),
        active_prefill_tokens: prefill.map(|point| point.value),
        last_ttft_ms: last_ttft.map(|point| point.value * 1_000.0),
        last_itl_ms: last_itl.map(|point| point.value * 1_000.0),
        engine_sample_age_ms: max_age_ms(decision_ns, [running, queued, kv_used]),
        running_reqs: running.map(|point| point.value),
        queued_reqs: queued.map(|point| point.value),
        gpu_cache_usage_fraction: kv_used.map(|point| point.value),
    }
}

fn sample_before<'a>(
    series: &'a MetricSeries,
    subject: &str,
    metric: &str,
    decision_ns: i64,
) -> Option<MetricPoint> {
    let points = series.get(&(subject.to_owned(), metric.to_owned()))?;
    let insertion = points.partition_point(|point| point.timestamp_ns <= decision_ns);
    insertion.checked_sub(1).map(|index| points[index])
}

fn max_age_ms<const N: usize>(decision_ns: i64, points: [Option<MetricPoint>; N]) -> Option<f64> {
    points
        .into_iter()
        .flatten()
        .map(|point| ((decision_ns - point.timestamp_ns).max(0) as f64) / 1_000_000.0)
        .max_by(f64::total_cmp)
}

fn integer_field(event: &Event, field: &str) -> Option<i64> {
    event.fields.get(field).and_then(|value| value.parse().ok())
}

fn float_field(event: &Event, field: &str) -> Option<f64> {
    event.fields.get(field).and_then(|value| value.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::Manifest;
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    #[test]
    fn materializes_formula_and_last_preceding_worker_state() {
        let temp = tempdir().unwrap();
        let manifest = Manifest {
            schema_version: 5,
            run_root: "/run".to_owned(),
            arms: vec![],
            warnings: vec![],
        };
        let decision = Event {
            source: LogSource::DynamoRouter,
            kind: EventKind::RoutingDecision,
            timestamp_ns: Some(2_000_000_000),
            worker_index: None,
            request_id: Some("client-1".to_owned()),
            fields: BTreeMap::from([
                ("dynamo_request_id".to_owned(), "internal-1".to_owned()),
                ("worker_id".to_owned(), "42".to_owned()),
            ]),
            raw: "routing".to_owned(),
        };
        let formula = Event {
            source: LogSource::DynamoRouter,
            kind: EventKind::RoutingFormula,
            timestamp_ns: Some(1_999_000_000),
            worker_index: None,
            request_id: Some("internal-1".to_owned()),
            fields: BTreeMap::from([
                ("worker_id".to_owned(), "42".to_owned()),
                ("cost_blocks".to_owned(), "10.5".to_owned()),
                ("effective_cached_blocks".to_owned(), "32".to_owned()),
            ]),
            raw: "formula".to_owned(),
        };
        let higher_prefix = Event {
            source: LogSource::DynamoRouter,
            kind: EventKind::RoutingFormula,
            timestamp_ns: Some(1_999_000_000),
            worker_index: None,
            request_id: Some("internal-1".to_owned()),
            fields: BTreeMap::from([
                ("worker_id".to_owned(), "43".to_owned()),
                ("cost_blocks".to_owned(), "11".to_owned()),
                ("effective_cached_blocks".to_owned(), "64".to_owned()),
            ]),
            raw: "formula".to_owned(),
        };
        let worker_event = Event {
            source: LogSource::DynamoWorker,
            kind: EventKind::WorkerRequest,
            timestamp_ns: None,
            worker_index: Some(2),
            request_id: None,
            fields: BTreeMap::from([("instance_id".to_owned(), "42".to_owned())]),
            raw: "payload".to_owned(),
        };
        let samples = vec![
            WorkerMetricSample {
                timestamp_ns: 1_500_000_000,
                scraper_endpoint: "router".to_owned(),
                metric_name_clean: ROUTER_PREFILL.to_owned(),
                router_worker_id: Some("42".to_owned()),
                metric_value: 64.0,
            },
            WorkerMetricSample {
                timestamp_ns: 2_500_000_000,
                scraper_endpoint: "router".to_owned(),
                metric_name_clean: ROUTER_PREFILL.to_owned(),
                router_worker_id: Some("42".to_owned()),
                metric_value: 99.0,
            },
            WorkerMetricSample {
                timestamp_ns: 1_700_000_000,
                scraper_endpoint: "worker_agg_2_0".to_owned(),
                metric_name_clean: ENGINE_RUNNING.to_owned(),
                router_worker_id: None,
                metric_value: 3.0,
            },
        ];
        write_database(
            &temp.path().join("ruter.db"),
            &manifest,
            &[decision, formula, higher_prefix, worker_event],
            &[],
            &[],
            &samples,
        )
        .unwrap();
        let connection = Connection::open(temp.path().join("ruter.db")).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT cost_blocks FROM routing_candidates", [], |row| row
                    .get::<_, f64>(
                    0
                ))
                .unwrap(),
            10.5
        );
        assert!(
            connection
                .query_row(
                    "SELECT lower_prefix_selected FROM routing_decisions",
                    [],
                    |row| { row.get::<_, bool>(0) }
                )
                .unwrap()
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT active_prefill_tokens FROM worker_state_snapshots",
                    [],
                    |row| row.get::<_, f64>(0)
                )
                .unwrap(),
            64.0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT running_reqs FROM worker_state_snapshots",
                    [],
                    |row| row.get::<_, f64>(0)
                )
                .unwrap(),
            3.0
        );
    }
}
