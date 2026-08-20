use crate::model::{Event, EventKind, LogSource};
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::LazyLock;

static ROUTING_WORKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[ROUTING\] Best:\s*(?P<worker>[^\s,]+)").unwrap());
static ROUTING_OVERLAP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"with\s+(?P<overlap>\d+)\s*/\s*(?P<total>\d+)\s+blocks").unwrap());
static ROUTING_DYNAMO_REQUEST: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"\[ROUTING\] Best:.*?\brequest_id\s*=\s*(?:\"(?P<quoted>[^\"]+)\"|(?P<bare>[^,\s}]+))"#,
    )
    .unwrap()
});
static ROUTING_FORMULA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"Formula for worker_id=(?P<worker>\d+) dp_rank=(?P<dp_rank>\d+) with (?P<effective_cached_blocks>[\d.]+) effective cached blocks:\s*(?P<cost_blocks>[\d.]+) = prefill_load_scale \* adjusted_prefill_blocks \+ decode_blocks \+ active_request_cost_blocks =\s*(?P<prefill_load_scale>[\d.]+) \* (?P<adjusted_prefill_blocks>[\d.]+) \+ (?P<decode_blocks>[\d.]+) \+ (?P<active_request_cost_blocks>[\d.]+)\s*\(raw_prefill_blocks: (?P<raw_prefill_blocks>[\d.]+), overlap_credit_blocks: (?P<overlap_credit_blocks>[\d.]+), overlap_credit_decay: (?P<overlap_credit_decay>[\d.]+)\)",
    )
    .unwrap()
});

/// Parse Dynamo frontend/KV-router logs. `[ROUTING] Best` is the one exact
/// decision record per request; selector and egress records remain candidates.
pub(super) fn parse(
    line: &str,
    mut fields: BTreeMap<String, String>,
    timestamp_ns: Option<i64>,
) -> Vec<Event> {
    if let Some(captures) = ROUTING_FORMULA.captures(line) {
        for field in [
            "worker",
            "dp_rank",
            "effective_cached_blocks",
            "cost_blocks",
            "prefill_load_scale",
            "adjusted_prefill_blocks",
            "decode_blocks",
            "active_request_cost_blocks",
            "raw_prefill_blocks",
            "overlap_credit_blocks",
            "overlap_credit_decay",
        ] {
            let key = if field == "worker" {
                "worker_id"
            } else {
                field
            };
            fields.insert(key.to_owned(), captures[field].to_owned());
        }
        return vec![super::event(
            LogSource::DynamoRouter,
            EventKind::RoutingFormula,
            timestamp_ns,
            None,
            super::request_id(&fields),
            fields,
            line,
        )];
    }
    if line.contains("Selected worker") || line.contains("[ROUTING] Best:") {
        if let Some(captures) = ROUTING_WORKER.captures(line) {
            fields
                .entry("worker_id".to_owned())
                .or_insert_with(|| captures["worker"].to_owned());
        }
        if let Some(captures) = ROUTING_OVERLAP.captures(line) {
            fields
                .entry("overlap_blocks".to_owned())
                .or_insert_with(|| captures["overlap"].to_owned());
            fields
                .entry("total_blocks".to_owned())
                .or_insert_with(|| captures["total"].to_owned());
        }
        // The structured HTTP span later in the same line can contain another
        // `request_id`. Keep this leading KV-router ID too: it is the ID emitted
        // by the context-free Dynamo request trace, while `x_request_id` joins
        // the same row to AIPerf.
        if let Some(captures) = ROUTING_DYNAMO_REQUEST.captures(line) {
            let request_id = captures
                .name("quoted")
                .or_else(|| captures.name("bare"))
                .expect("routing request ID has a value")
                .as_str()
                .to_owned();
            fields
                .entry("dynamo_request_id".to_owned())
                .or_insert(request_id);
        }
        let kind = if line.contains("[ROUTING] Best:") {
            EventKind::RoutingDecision
        } else {
            EventKind::RoutingCandidate
        };
        return vec![super::event(
            LogSource::DynamoRouter,
            kind,
            timestamp_ns,
            None,
            super::request_id(&fields),
            fields,
            line,
        )];
    }
    if line.contains("request received") {
        return vec![super::event(
            LogSource::DynamoRouter,
            EventKind::RouterAdmission,
            timestamp_ns,
            None,
            super::request_id(&fields),
            fields,
            line,
        )];
    }
    Vec::new()
}
