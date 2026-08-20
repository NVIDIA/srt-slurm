use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogSource {
    DynamoRouter,
    DynamoWorker,
}

impl LogSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DynamoRouter => "dynamo_router",
            Self::DynamoWorker => "dynamo_worker",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    RouterAdmission,
    /// One candidate's exact KV-router block-cost formula from `DYN_LOG=debug`.
    RoutingFormula,
    RoutingCandidate,
    RoutingDecision,
    RouterRequestStart,
    RouterResponse,
    WorkerPrefillBatch,
    WorkerDecodeBatch,
    WorkerRequest,
    WorkerLifecycle,
}

impl EventKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RouterAdmission => "router_admission",
            Self::RoutingFormula => "routing_formula",
            Self::RoutingCandidate => "routing_candidate",
            Self::RoutingDecision => "routing_decision",
            Self::RouterRequestStart => "router_request_start",
            Self::RouterResponse => "router_response",
            Self::WorkerPrefillBatch => "worker_prefill_batch",
            Self::WorkerDecodeBatch => "worker_decode_batch",
            Self::WorkerRequest => "worker_request",
            Self::WorkerLifecycle => "worker_lifecycle",
        }
    }
}

/// A normalized event. `fields` holds parser-specific facts without discarding
/// attributes that a later visualization may need.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub source: LogSource,
    pub kind: EventKind,
    /// UTC Unix nanoseconds, when the source line carries a parseable timestamp.
    pub timestamp_ns: Option<i64>,
    pub worker_index: Option<u32>,
    pub request_id: Option<String>,
    pub fields: BTreeMap<String, String>,
    pub raw: String,
}
