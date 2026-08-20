use crate::logs::ParserKind;
use crate::model::{Event, EventKind};
use std::collections::BTreeMap;

/// Parse Dynamo-hosted SGLang engine logs while preserving the worker index.
pub(super) fn parse(
    kind: ParserKind,
    line: &str,
    mut fields: BTreeMap<String, String>,
    timestamp_ns: Option<i64>,
) -> Vec<Event> {
    let event_kind = if line.contains("Prefill batch,") {
        Some(EventKind::WorkerPrefillBatch)
    } else if line.contains("Decode batch,") {
        Some(EventKind::WorkerDecodeBatch)
    } else if let Some(rid) = super::parse_rid(line) {
        fields.insert("rid".to_owned(), rid);
        Some(EventKind::WorkerRequest)
    } else if fields.contains_key("instance_id")
        && (line.contains("request received") || line.contains("request completed"))
    {
        // Dynamo's SGLang payload log contains the opaque frontend worker ID.
        // Preserve it with the physical worker-log index so `ruter init` can
        // join candidate IDs to Tachometer's worker_agg_<index>_0 scrape.
        Some(EventKind::WorkerRequest)
    } else if line.contains("Model registration succeeded") || line.contains("server ready") {
        Some(EventKind::WorkerLifecycle)
    } else {
        None
    };

    event_kind
        .map(|event_kind| {
            let request_id = fields
                .get("rid")
                .cloned()
                .or_else(|| super::request_id(&fields));
            super::event(
                kind.source(),
                event_kind,
                timestamp_ns,
                kind.worker_index(),
                request_id,
                fields,
                line,
            )
        })
        .into_iter()
        .collect()
}
