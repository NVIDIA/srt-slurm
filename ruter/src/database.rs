//! SQLite materialization for the small, portable static viewer.

use crate::artifacts::Manifest;
use crate::model::{Event, EventKind, LogSource};
use crate::tables::{AiperfRequest, RequestTrace, WorkerMetricSample};
use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::collections::{BTreeMap, BTreeSet};
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

#[derive(Clone)]
struct DecisionRecord<'a> {
    decision_id: String,
    event: &'a Event,
    dynamo_request_id: String,
    phase: String,
}

#[derive(Clone)]
struct WorkerLocator {
    role: String,
    index: u32,
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
            decision_id TEXT PRIMARY KEY, timestamp_ns INTEGER, x_request_id TEXT,
            dynamo_request_id TEXT NOT NULL, phase TEXT NOT NULL, worker_id TEXT, dp_rank INTEGER,
            overlap_blocks INTEGER, total_blocks INTEGER, lower_prefix_selected INTEGER NOT NULL,
            raw TEXT NOT NULL
        );
        CREATE TABLE routing_candidates (
            decision_id TEXT NOT NULL, timestamp_ns INTEGER NOT NULL, worker_id TEXT NOT NULL,
            dp_rank INTEGER, cost_blocks REAL, effective_cached_blocks REAL,
            prefill_load_scale REAL, adjusted_prefill_blocks REAL, raw_prefill_blocks REAL,
            overlap_credit_blocks REAL, overlap_credit_decay REAL, decode_blocks REAL,
            active_request_cost_blocks REAL, PRIMARY KEY(decision_id, worker_id, dp_rank)
        );
        CREATE TABLE worker_state_snapshots (
            decision_id TEXT NOT NULL, worker_id TEXT NOT NULL, router_sample_age_ms REAL,
            active_prefill_tokens REAL, last_ttft_ms REAL, last_itl_ms REAL,
            engine_sample_age_ms REAL, running_reqs REAL, queued_reqs REAL,
            gpu_cache_usage_fraction REAL, PRIMARY KEY(decision_id, worker_id)
        );
        CREATE INDEX request_traces_time ON request_traces(request_received_ns, event_time_ns);
        CREATE INDEX routing_decisions_time ON routing_decisions(timestamp_ns);
        CREATE INDEX routing_decisions_request ON routing_decisions(dynamo_request_id, phase);
        CREATE INDEX routing_candidates_decision ON routing_candidates(decision_id, cost_blocks);
        CREATE INDEX worker_state_decision ON worker_state_snapshots(decision_id);
        ",
    )?;

    let decisions = decision_records(events);
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
        let mut insert = transaction.prepare(
            "INSERT INTO routing_decisions VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        )?;
        for decision in &decisions {
            insert.execute(params![
                decision.decision_id,
                decision.event.timestamp_ns,
                decision.event.fields.get("x_request_id"),
                decision.dynamo_request_id,
                decision.phase,
                decision.event.fields.get("worker_id"),
                integer_field(decision.event, "dp_rank"),
                integer_field(decision.event, "overlap_blocks"),
                integer_field(decision.event, "total_blocks"),
                0,
                decision.event.raw,
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
        for decision in &decisions {
            for event in formulas_for_decision(events, decision) {
                let Some(timestamp_ns) = event.timestamp_ns.or(decision.event.timestamp_ns) else {
                    continue;
                };
                let Some(worker_id) = event.fields.get("worker_id") else {
                    continue;
                };
                insert_candidate.execute(params![
                    decision.decision_id,
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
                let snapshot = snapshot(&metric_series, &worker_indices, worker_id, timestamp_ns);
                insert_snapshot.execute(params![
                    decision.decision_id,
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
    }
    transaction.execute(
        "
        UPDATE routing_decisions
        SET lower_prefix_selected = CASE WHEN phase = 'prefill' THEN EXISTS(
            SELECT 1
            FROM routing_candidates selected
            JOIN routing_candidates candidate ON candidate.decision_id = selected.decision_id
            WHERE selected.decision_id = routing_decisions.decision_id
              AND selected.worker_id = routing_decisions.worker_id
              AND candidate.effective_cached_blocks > selected.effective_cached_blocks
        ) ELSE 0 END
        ",
        [],
    )?;
    transaction.commit()?;
    Ok(())
}

fn decision_records(events: &[Event]) -> Vec<DecisionRecord<'_>> {
    let phases_by_request = formula_phases(events);
    let is_disaggregated = phases_by_request
        .values()
        .any(|phases| phases.contains("prefill") && phases.contains("decode"));
    let mut ordinals = BTreeMap::<(String, String), usize>::new();
    events
        .iter()
        .filter(|event| {
            event.source == LogSource::DynamoRouter && event.kind == EventKind::RoutingDecision
        })
        .filter_map(|event| {
            let dynamo_request_id = dynamo_request_id(event)?.to_owned();
            let phase = decision_phase(
                event,
                &dynamo_request_id,
                &phases_by_request,
                is_disaggregated,
            );
            let ordinal = ordinals
                .entry((dynamo_request_id.clone(), phase.clone()))
                .or_default();
            let decision_id = format!("{dynamo_request_id}:{phase}:{}", *ordinal);
            *ordinal += 1;
            Some(DecisionRecord {
                decision_id,
                event,
                dynamo_request_id,
                phase,
            })
        })
        .collect()
}

fn formula_phases(events: &[Event]) -> BTreeMap<String, BTreeSet<String>> {
    let mut phases = BTreeMap::<String, BTreeSet<String>>::new();
    for event in events.iter().filter(|event| {
        event.source == LogSource::DynamoRouter && event.kind == EventKind::RoutingFormula
    }) {
        let (Some(request_id), Some(phase)) = (dynamo_request_id(event), normalized_phase(event))
        else {
            continue;
        };
        phases
            .entry(request_id.to_owned())
            .or_default()
            .insert(phase);
    }
    phases
}

fn decision_phase(
    event: &Event,
    _request_id: &str,
    _phases_by_request: &BTreeMap<String, BTreeSet<String>>,
    is_disaggregated: bool,
) -> String {
    if !is_disaggregated {
        return "aggregated".to_owned();
    }
    if let Some(phase) = normalized_phase(event) {
        return phase;
    }
    // `[ROUTING] Best` originates in Dynamo's prefill push router. Decode
    // routing emits a structured `Selected worker` event with worker_type.
    "prefill".to_owned()
}

fn normalized_phase(event: &Event) -> Option<String> {
    let phase = event
        .fields
        .get("worker_type")
        .or_else(|| event.fields.get("phase"))?
        .to_ascii_lowercase();
    match phase.as_str() {
        "prefill" | "decode" | "aggregated" | "agg" => Some(if phase == "agg" {
            "aggregated".to_owned()
        } else {
            phase
        }),
        _ => None,
    }
}

fn dynamo_request_id(event: &Event) -> Option<&str> {
    event
        .fields
        .get("dynamo_request_id")
        .map(String::as_str)
        .or(event.request_id.as_deref())
}

fn formulas_for_decision<'a>(
    events: &'a [Event],
    decision: &DecisionRecord<'_>,
) -> impl Iterator<Item = &'a Event> {
    events.iter().filter(move |event| {
        event.source == LogSource::DynamoRouter
            && event.kind == EventKind::RoutingFormula
            && dynamo_request_id(event) == Some(decision.dynamo_request_id.as_str())
            && (decision.phase == "aggregated"
                || normalized_phase(event).as_deref() == Some(decision.phase.as_str()))
    })
}

/// Tie an opaque Dynamo frontend worker ID to its role-qualified direct-host
/// worker log, e.g. `worker_prefill_1_0` rather than the aggregate-only name.
fn worker_indices(events: &[Event]) -> BTreeMap<String, WorkerLocator> {
    let mut indices = BTreeMap::new();
    for event in events.iter().filter(|event| {
        event.source == LogSource::DynamoWorker && event.kind == EventKind::WorkerRequest
    }) {
        let (Some(worker_id), Some(worker_index), Some(worker_role)) = (
            event.fields.get("instance_id"),
            event.worker_index,
            event.fields.get("worker_role"),
        ) else {
            continue;
        };
        indices
            .entry(worker_id.clone())
            .or_insert_with(|| WorkerLocator {
                role: worker_role.clone(),
                index: worker_index,
            });
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
    worker_indices: &BTreeMap<String, WorkerLocator>,
    worker_id: &str,
    decision_ns: i64,
) -> WorkerSnapshot {
    let router_subject = format!("router:{worker_id}");
    let prefill = sample_before(series, &router_subject, ROUTER_PREFILL, decision_ns);
    let last_ttft = sample_before(series, &router_subject, ROUTER_TTFT, decision_ns);
    let last_itl = sample_before(series, &router_subject, ROUTER_ITL, decision_ns);
    let engine_subject = worker_indices
        .get(worker_id)
        .map(|worker| format!("engine:worker_{}_{}_0", worker.role, worker.index));
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

fn sample_before(
    series: &MetricSeries,
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

    fn event(
        kind: EventKind,
        timestamp_ns: i64,
        request_id: &str,
        fields: &[(&str, &str)],
    ) -> Event {
        Event {
            source: LogSource::DynamoRouter,
            kind,
            timestamp_ns: Some(timestamp_ns),
            worker_index: None,
            request_id: Some(request_id.to_owned()),
            fields: fields
                .iter()
                .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
                .collect(),
            raw: "event".to_owned(),
        }
    }

    #[test]
    fn materializes_two_phase_decisions_and_role_qualified_worker_state() {
        let temp = tempdir().unwrap();
        let manifest = Manifest {
            schema_version: 6,
            run_root: "/run".to_owned(),
            arms: vec![],
            warnings: vec![],
        };
        let events = vec![
            event(
                EventKind::RoutingFormula,
                1_999_000_000,
                "internal-1",
                &[
                    ("worker_id", "42"),
                    ("worker_type", "prefill"),
                    ("cost_blocks", "10.5"),
                    ("effective_cached_blocks", "32"),
                ],
            ),
            event(
                EventKind::RoutingFormula,
                1_999_000_000,
                "internal-1",
                &[
                    ("worker_id", "43"),
                    ("worker_type", "prefill"),
                    ("cost_blocks", "11"),
                    ("effective_cached_blocks", "64"),
                ],
            ),
            event(
                EventKind::RoutingFormula,
                2_009_000_000,
                "internal-1",
                &[
                    ("worker_id", "99"),
                    ("worker_type", "decode"),
                    ("cost_blocks", "4"),
                    ("effective_cached_blocks", "0"),
                ],
            ),
            event(
                EventKind::RoutingFormula,
                2_009_000_000,
                "internal-1",
                &[
                    ("worker_id", "100"),
                    ("worker_type", "decode"),
                    ("cost_blocks", "5"),
                    ("effective_cached_blocks", "0"),
                ],
            ),
            event(
                EventKind::RoutingDecision,
                2_000_000_000,
                "internal-1",
                &[("worker_id", "42")],
            ),
            event(
                EventKind::RoutingDecision,
                2_010_000_000,
                "internal-1",
                &[("worker_id", "99"), ("worker_type", "decode")],
            ),
            Event {
                source: LogSource::DynamoWorker,
                kind: EventKind::WorkerRequest,
                timestamp_ns: None,
                worker_index: Some(2),
                request_id: None,
                fields: BTreeMap::from([
                    ("instance_id".to_owned(), "42".to_owned()),
                    ("worker_role".to_owned(), "prefill".to_owned()),
                ]),
                raw: "payload".to_owned(),
            },
        ];
        let samples = vec![
            WorkerMetricSample {
                timestamp_ns: 1_500_000_000,
                scraper_endpoint: "router".to_owned(),
                metric_name_clean: ROUTER_PREFILL.to_owned(),
                router_worker_id: Some("42".to_owned()),
                metric_value: 64.0,
            },
            WorkerMetricSample {
                timestamp_ns: 1_700_000_000,
                scraper_endpoint: "worker_prefill_2_0".to_owned(),
                metric_name_clean: ENGINE_RUNNING.to_owned(),
                router_worker_id: None,
                metric_value: 3.0,
            },
        ];
        write_database(
            &temp.path().join("ruter.db"),
            &manifest,
            &events,
            &[],
            &[],
            &samples,
        )
        .unwrap();
        let connection = Connection::open(temp.path().join("ruter.db")).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT COUNT(*) FROM routing_decisions", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT phase FROM routing_decisions WHERE worker_id = '42'",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            "prefill"
        );
        assert!(
            connection
                .query_row(
                    "SELECT lower_prefix_selected FROM routing_decisions WHERE worker_id = '42'",
                    [],
                    |row| row.get::<_, bool>(0),
                )
                .unwrap()
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT running_reqs FROM worker_state_snapshots WHERE worker_id = '42'",
                    [],
                    |row| row.get::<_, f64>(0),
                )
                .unwrap(),
            3.0
        );
    }
}
