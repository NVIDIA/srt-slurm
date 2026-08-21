//! Source-specific log parsers plus shared normalization helpers.

mod dynamo_router;
mod worker;

use crate::model::{Event, LogSource};
use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use regex::Regex;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::LazyLock;

static ANSI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b\[[0-?]*[ -/]*[@-~]").unwrap());
static RFC3339_TS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?P<ts>20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)").unwrap()
});
static SGLANG_TS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(?P<ts>20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]").unwrap());
static LOGFMT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(?:\"(?P<quoted>[^\"]*)\"|(?P<bare>[^,\s}]+))"#,
    )
    .unwrap()
});
static SCHEDULER_FIELD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?P<key>#[A-Za-z][A-Za-z0-9-]*)\s*:\s*(?P<value>[^,]+)").unwrap()
});
static RID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\brid\s*(?:=|:)\s*(?:\"(?P<quoted>[^\"]+)\"|(?P<bare>[^,\s}]+))"#).unwrap()
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserKind {
    DynamoRouter,
    DynamoWorker {
        worker_index: u32,
        /// The direct lifecycle renderer puts the role in the log filename.
        /// Keep it with every normalized worker event so a Dynamo instance ID
        /// can be joined to Tachometer's role-qualified endpoint later.
        worker_role: &'static str,
    },
}

impl ParserKind {
    pub fn source(self) -> LogSource {
        match self {
            Self::DynamoRouter => LogSource::DynamoRouter,
            Self::DynamoWorker { .. } => LogSource::DynamoWorker,
        }
    }

    fn worker_index(self) -> Option<u32> {
        match self {
            Self::DynamoWorker { worker_index, .. } => Some(worker_index),
            _ => None,
        }
    }
}

/// Parse one Dynamo router or Dynamo-hosted SGLang worker log line.
pub fn parse_line(kind: ParserKind, raw: &str) -> Vec<Event> {
    let line = strip_ansi(raw);
    let mut fields = parse_fields(&line);
    let timestamp_ns = parse_timestamp_ns(&line);
    match kind {
        ParserKind::DynamoRouter => dynamo_router::parse(&line, fields, timestamp_ns),
        ParserKind::DynamoWorker { worker_role, .. } => {
            fields.insert("worker_role".to_owned(), worker_role.to_owned());
            worker::parse(kind, &line, fields, timestamp_ns)
        }
    }
}

pub fn parse_file(kind: ParserKind, path: &Path) -> Result<Vec<Event>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut events = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.with_context(|| format!("read {}", path.display()))?;
        events.extend(parse_line(kind, &line));
    }
    Ok(events)
}

/// Return the first parseable UTC timestamp in a text log. Tachometer's
/// `time_since_start` values are explicitly anchored to this value at ingest.
pub fn first_timestamp_ns(path: &Path) -> Result<Option<i64>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line.with_context(|| format!("read {}", path.display()))?;
        if let Some(timestamp_ns) = parse_timestamp_ns(&strip_ansi(&line)) {
            return Ok(Some(timestamp_ns));
        }
    }
    Ok(None)
}

pub(super) fn event(
    source: LogSource,
    kind: crate::model::EventKind,
    timestamp_ns: Option<i64>,
    worker_index: Option<u32>,
    request_id: Option<String>,
    fields: BTreeMap<String, String>,
    raw: &str,
) -> Event {
    Event {
        source,
        kind,
        timestamp_ns,
        worker_index,
        request_id,
        fields,
        raw: raw.to_owned(),
    }
}

pub(super) fn request_id(fields: &BTreeMap<String, String>) -> Option<String> {
    fields
        .get("x_request_id")
        .or_else(|| fields.get("request_id"))
        .cloned()
}

pub(super) fn parse_rid(line: &str) -> Option<String> {
    RID.captures(line).and_then(|captures| {
        captures
            .name("quoted")
            .or_else(|| captures.name("bare"))
            .map(|value| value.as_str().to_owned())
    })
}

fn strip_ansi(line: &str) -> String {
    ANSI.replace_all(line, "").into_owned()
}

fn parse_fields(line: &str) -> BTreeMap<String, String> {
    let mut fields = BTreeMap::new();
    // `observability.enabled` uses Dynamo's JSONL logging. A tracing event
    // normally stores its useful structured fields under `fields`; accepting
    // those directly keeps the parser equally usable with JSONL and text logs.
    if let Ok(Value::Object(record)) = serde_json::from_str::<Value>(line) {
        insert_json_fields(&mut fields, &record);
        if let Some(Value::Object(record_fields)) = record.get("fields") {
            insert_json_fields(&mut fields, record_fields);
        }
    }
    for captures in LOGFMT.captures_iter(line) {
        let key = captures.name("key").unwrap().as_str().to_owned();
        let value = captures
            .name("quoted")
            .or_else(|| captures.name("bare"))
            .unwrap()
            .as_str()
            .trim_end_matches('}')
            .to_owned();
        fields.insert(key, value);
    }
    for captures in SCHEDULER_FIELD.captures_iter(line) {
        fields.insert(
            captures.name("key").unwrap().as_str().to_owned(),
            captures.name("value").unwrap().as_str().trim().to_owned(),
        );
    }
    fields
}

fn insert_json_fields(
    fields: &mut BTreeMap<String, String>,
    record: &serde_json::Map<String, Value>,
) {
    for (key, value) in record {
        match value {
            Value::String(value) => {
                fields.entry(key.clone()).or_insert_with(|| value.clone());
            }
            Value::Number(value) => {
                fields
                    .entry(key.clone())
                    .or_insert_with(|| value.to_string());
            }
            Value::Bool(value) => {
                fields
                    .entry(key.clone())
                    .or_insert_with(|| value.to_string());
            }
            _ => {}
        }
    }
}

fn parse_timestamp_ns(line: &str) -> Option<i64> {
    if let Some(captures) = RFC3339_TS.captures(line) {
        return DateTime::parse_from_rfc3339(captures.name("ts").unwrap().as_str())
            .ok()
            .and_then(|time| time.timestamp_nanos_opt());
    }
    SGLANG_TS.captures(line).and_then(|captures| {
        NaiveDateTime::parse_from_str(captures.name("ts").unwrap().as_str(), "%Y-%m-%d %H:%M:%S")
            .ok()
            .and_then(|time| Utc.from_utc_datetime(&time).timestamp_nanos_opt())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::EventKind;

    #[test]
    fn parses_dynamo_selector_and_egress_as_candidates() {
        let candidate = "2026-08-20T04:00:30.062165Z INFO selector: Selected worker router_mode=\"kv\" request_id=78ef worker_id=7587896704094643198 dp_rank=0 logit=0.875";
        let final_decision = "2026-08-20T04:00:30.063348Z INFO request: Selected worker router_mode=\"direct\" request_id=e578 request_id=78ef worker_id=7587896704094643198 dp_rank=0 overlap_blocks=0 phase=Aggregated";
        assert_eq!(
            parse_line(ParserKind::DynamoRouter, candidate)[0].kind,
            EventKind::RoutingCandidate
        );
        let event = &parse_line(ParserKind::DynamoRouter, final_decision)[0];
        assert_eq!(event.kind, EventKind::RoutingCandidate);
        assert_eq!(event.request_id.as_deref(), Some("78ef"));
        assert_eq!(event.fields["worker_id"], "7587896704094643198");
    }

    #[test]
    fn parses_dynamo_debug_routing_record() {
        let line = "2026-08-20T04:00:30.0Z DEBUG router: request_id=req-1 worker_id=42 dp_rank=0 overlap_blocks=8 total_blocks=12 [ROUTING] Best: worker_42 dp_rank=0 with 8/12 blocks overlap";
        let event = &parse_line(ParserKind::DynamoRouter, line)[0];
        assert_eq!(event.kind, EventKind::RoutingDecision);
        assert_eq!(event.request_id.as_deref(), Some("req-1"));
        assert_eq!(event.fields["total_blocks"], "12");
    }

    #[test]
    fn parses_exact_kv_cost_formula() {
        let line = "2026-08-20T04:00:30.0Z DEBUG selector: Formula for worker_id=42 dp_rank=0 with 32.00 effective cached blocks: 80.500 = prefill_load_scale * adjusted_prefill_blocks + decode_blocks + active_request_cost_blocks = 1.000 * 24.500 + 56.000 + 0.000 (raw_prefill_blocks: 56.500, overlap_credit_blocks: 32.000, overlap_credit_decay: 1.000) request_id=internal-1";
        let event = &parse_line(ParserKind::DynamoRouter, line)[0];
        assert_eq!(event.kind, EventKind::RoutingFormula);
        assert_eq!(event.request_id.as_deref(), Some("internal-1"));
        assert_eq!(event.fields["worker_id"], "42");
        assert_eq!(event.fields["cost_blocks"], "80.500");
        assert_eq!(event.fields["overlap_credit_blocks"], "32.000");
    }

    #[test]
    fn parses_debug_routing_fields_emitted_only_in_the_message() {
        let line = "2026-08-20T04:00:30.0Z DEBUG router: request_id=req-2 [ROUTING] Best: worker_99 dp_rank=0 with 8/12 blocks overlap";
        let event = &parse_line(ParserKind::DynamoRouter, line)[0];
        assert_eq!(event.fields["worker_id"], "worker_99");
        assert_eq!(event.fields["overlap_blocks"], "8");
        assert_eq!(event.fields["total_blocks"], "12");
    }

    #[test]
    fn preserves_dynamo_and_aiperf_request_ids_from_a_routing_line() {
        let line = "2026-08-20T04:00:30.0Z DEBUG router: [ROUTING] Best: worker_99 dp_rank=0 with 8/12 blocks overlap request_id=internal-1 worker_id=99 overlap_blocks=8 total_blocks=12 x_request_id=\"client-1\" request_id=http-1";
        let event = &parse_line(ParserKind::DynamoRouter, line)[0];
        assert_eq!(event.request_id.as_deref(), Some("client-1"));
        assert_eq!(event.fields["dynamo_request_id"], "internal-1");
    }

    #[test]
    fn parses_dynamo_worker_batch_grammar() {
        let line = "[2026-08-20 04:40:20] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, token usage: 0.09, #running-req: 0, #queue-req: 0, #pending-token: 14906, cuda graph: False";
        let events = parse_line(
            ParserKind::DynamoWorker {
                worker_index: 0,
                worker_role: "agg",
            },
            line,
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventKind::WorkerPrefillBatch);
        assert_eq!(events[0].fields["#new-token"], "8192");
        assert_eq!(events[0].fields["#pending-token"], "14906");
    }

    #[test]
    fn preserves_worker_id_from_payload_lifecycle() {
        let line = "[2026-08-20 04:40:20] INFO handle_payload: request received instance_id=2228894916226259721 request_id=internal-1";
        let event = &parse_line(
            ParserKind::DynamoWorker {
                worker_index: 3,
                worker_role: "prefill",
            },
            line,
        )[0];
        assert_eq!(event.kind, EventKind::WorkerRequest);
        assert_eq!(event.worker_index, Some(3));
        assert_eq!(event.fields["worker_role"], "prefill");
        assert_eq!(event.fields["instance_id"], "2228894916226259721");
    }

    #[test]
    fn parses_jsonl_structured_router_fields() {
        let line = r#"{"timestamp":"2026-08-20T04:00:30.0Z","fields":{"message":"[ROUTING] Best: worker_42 dp_rank=0 with 8/12 blocks overlap","request_id":"internal-1","x_request_id":"client-1","worker_id":42,"dp_rank":0}}"#;
        let event = &parse_line(ParserKind::DynamoRouter, line)[0];
        assert_eq!(event.kind, EventKind::RoutingDecision);
        assert_eq!(event.request_id.as_deref(), Some("client-1"));
        assert_eq!(event.fields["dynamo_request_id"], "internal-1");
    }
}
