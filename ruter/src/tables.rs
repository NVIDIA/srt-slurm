//! Flat Parquet tables used by the hackable Streamlit/DuckDB UI.

use crate::logs::first_timestamp_ns;
use crate::model::Event;
use anyhow::{Context, Result};
use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int64Array, LargeStringArray, RecordBatch,
    StringArray, StringViewArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use flate2::read::MultiGzDecoder;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::{Arc, LazyLock};

const BATCH_ROWS: usize = 16_384;
static WORKER_ID_LABEL: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r#"(?:^|[,{])worker_id=\"(?P<worker_id>[^\"]+)\""#).unwrap()
});

/// A narrowly-selected Tachometer sample used by the per-decision inspector.
/// Keeping only the useful metrics out of the browser keeps the static view
/// responsive even when the source Parquet has millions of samples.
#[derive(Debug, Clone)]
pub struct WorkerMetricSample {
    pub timestamp_ns: i64,
    pub scraper_endpoint: String,
    pub metric_name_clean: String,
    pub router_worker_id: Option<String>,
    pub metric_value: f64,
}

/// One load-generator result. It remains separate from a Dynamo request trace:
/// context-free Dynamo traces deliberately do not capture user-provided headers.
#[derive(Debug, Clone)]
pub struct AiperfRequest {
    pub arm: String,
    pub request_id: Option<String>,
    pub session_id: Option<String>,
    pub conversation_id: Option<String>,
    pub credit_issued_ns: Option<i64>,
    pub request_start_ns: Option<i64>,
    pub request_ack_ns: Option<i64>,
    pub request_end_ns: Option<i64>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub ttft_ms: Option<f64>,
    pub itl_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    pub raw_json: String,
}

/// The fields needed to graph a context-free Dynamo `request_end` record.
#[derive(Debug, Clone)]
pub struct RequestTrace {
    pub arm: String,
    pub request_id: Option<String>,
    pub x_request_id: Option<String>,
    pub event_time_ns: Option<i64>,
    pub request_received_ns: Option<i64>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
    pub kv_hit_rate: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub total_time_ms: Option<f64>,
    pub avg_itl_ms: Option<f64>,
    pub queue_depth: Option<i64>,
    pub prefill_worker_id: Option<String>,
    pub prefill_dp_rank: Option<i64>,
    pub decode_worker_id: Option<String>,
    pub decode_dp_rank: Option<i64>,
    pub raw_json: String,
}

pub fn write_events(path: &Path, events: &[Event]) -> Result<usize> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("timestamp_ns", DataType::Int64, true),
        Field::new("worker_index", DataType::UInt32, true),
        Field::new("request_id", DataType::Utf8, true),
        Field::new("fields_json", DataType::Utf8, false),
        Field::new("raw", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(
                events
                    .iter()
                    .map(|event| event.source.as_str())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                events
                    .iter()
                    .map(|event| event.kind.as_str())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                events
                    .iter()
                    .map(|event| event.timestamp_ns)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt32Array::from(
                events
                    .iter()
                    .map(|event| event.worker_index)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                events
                    .iter()
                    .map(|event| event.request_id.as_deref())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                events
                    .iter()
                    .map(|event| serde_json::to_string(&event.fields).expect("BTreeMap serializes"))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                events
                    .iter()
                    .map(|event| event.raw.as_str())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(events.len())
}

pub fn load_aiperf_requests(sources: &[(&str, &Path)]) -> Result<Vec<AiperfRequest>> {
    let mut rows = Vec::new();
    for (arm, source) in sources {
        for line in jsonl_reader(source)?.lines() {
            let line = line.with_context(|| format!("read {}", source.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            let record: Value = serde_json::from_str(&line)
                .with_context(|| format!("parse {}", source.display()))?;
            rows.push(AiperfRequest {
                arm: (*arm).to_owned(),
                request_id: string_at(&record, &["metadata", "x_request_id"]),
                session_id: string_at(&record, &["metadata", "session_num"]),
                conversation_id: string_at(&record, &["metadata", "conversation_id"]),
                credit_issued_ns: i64_at(&record, &["metadata", "credit_issued_ns"]),
                request_start_ns: i64_at(&record, &["metadata", "request_start_ns"]),
                request_ack_ns: i64_at(&record, &["metadata", "request_ack_ns"]),
                request_end_ns: i64_at(&record, &["metadata", "request_end_ns"]),
                input_tokens: i64_at(&record, &["metrics", "input_sequence_length", "value"]),
                output_tokens: i64_at(&record, &["metrics", "output_sequence_length", "value"]),
                ttft_ms: f64_at(&record, &["metrics", "time_to_first_token", "value"]),
                itl_ms: f64_at(&record, &["metrics", "inter_token_latency", "value"]),
                e2e_ms: f64_at(&record, &["metrics", "request_latency", "value"]),
                raw_json: line,
            });
        }
    }
    Ok(rows)
}

fn jsonl_reader(source: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(source).with_context(|| format!("open {}", source.display()))?;
    let reader: Box<dyn Read> = if source.extension().is_some_and(|ext| ext == "gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::new(reader)))
}

pub fn write_requests(path: &Path, source_rows: &[AiperfRequest]) -> Result<usize> {
    let schema = requests_schema();
    let mut rows = RequestRows::default();
    for row in source_rows {
        rows.push(row)
    }
    let batch = rows.into_batch(schema.clone())?;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(batch.num_rows())
}

pub fn load_request_traces(sources: &[(&str, &Path)]) -> Result<Vec<RequestTrace>> {
    let mut rows = Vec::new();
    for (arm, source) in sources {
        for line in jsonl_reader(source)?.lines() {
            let line = line.with_context(|| format!("read {}", source.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            let envelope: Value = serde_json::from_str(&line)
                .with_context(|| format!("parse {}", source.display()))?;
            let event = envelope.get("event").unwrap_or(&envelope);
            if event.get("schema").and_then(Value::as_str) != Some("dynamo.request.trace.v1")
                || event.get("event_type").and_then(Value::as_str) != Some("request_end")
            {
                continue;
            }
            let Some(request) = event.get("request") else {
                continue;
            };
            rows.push(RequestTrace {
                arm: (*arm).to_owned(),
                request_id: string_at(request, &["request_id"]),
                x_request_id: string_at(request, &["x_request_id"]),
                event_time_ns: i64_at(event, &["event_time_unix_ms"])
                    .map(|value| value.saturating_mul(1_000_000)),
                request_received_ns: i64_at(request, &["request_received_ms"])
                    .map(|value| value.saturating_mul(1_000_000)),
                input_tokens: i64_at(request, &["input_tokens"]),
                output_tokens: i64_at(request, &["output_tokens"]),
                cached_tokens: i64_at(request, &["cached_tokens"]),
                kv_hit_rate: f64_at(request, &["kv_hit_rate"]),
                ttft_ms: f64_at(request, &["ttft_ms"]),
                total_time_ms: f64_at(request, &["total_time_ms"]),
                avg_itl_ms: f64_at(request, &["avg_itl_ms"]),
                queue_depth: i64_at(request, &["queue_depth"]),
                prefill_worker_id: string_at(request, &["worker", "prefill_worker_id"]),
                prefill_dp_rank: i64_at(request, &["worker", "prefill_dp_rank"]),
                decode_worker_id: string_at(request, &["worker", "decode_worker_id"]),
                decode_dp_rank: i64_at(request, &["worker", "decode_dp_rank"]),
                raw_json: line,
            });
        }
    }
    Ok(rows)
}

pub fn write_request_traces(path: &Path, rows: &[RequestTrace]) -> Result<usize> {
    let schema = request_traces_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.arm.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.request_id.as_deref())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.x_request_id.as_deref())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.event_time_ns).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| row.request_received_ns)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.input_tokens).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.output_tokens).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.cached_tokens).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.kv_hit_rate).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.ttft_ms).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.total_time_ms).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.avg_itl_ms).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.queue_depth).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.prefill_worker_id.as_deref())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| row.prefill_dp_rank)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.decode_worker_id.as_deref())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| row.decode_dp_rank)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.raw_json.as_str())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(batch.num_rows())
}

/// Copy Tachometer's long-form metrics into a stable, arm-attributed schema.
/// The original `time_since_start` is retained alongside its explicit log anchor.
pub fn write_metrics(path: &Path, sources: &[(&str, &Path, Option<&Path>)]) -> Result<usize> {
    let schema = metrics_schema();
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    let mut rows = MetricRows::default();
    let mut count = 0;
    for (arm, source, tachometer_log) in sources {
        let anchor_ns = tachometer_log
            .map(first_timestamp_ns)
            .transpose()?
            .flatten();
        let file = File::open(source).with_context(|| format!("open {}", source.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("open Parquet {}", source.display()))?;
        let mut reader = builder.with_batch_size(BATCH_ROWS).build()?;
        while let Some(batch) = reader.next() {
            let batch = batch?;
            for index in 0..batch.num_rows() {
                rows.push(*arm, anchor_ns, &batch, index);
                if rows.len() == BATCH_ROWS {
                    rows.flush(&mut writer, schema.clone())?;
                }
                count += 1;
            }
        }
    }
    rows.flush(&mut writer, schema)?;
    writer.close()?;
    Ok(count)
}

/// Read only the telemetry needed to contextualize a routing decision. This
/// runs after normalization, so it has the explicit Tachometer log time anchor
/// rather than relying on an implicit scrape-clock interpretation.
pub fn load_worker_metric_samples(path: &Path) -> Result<Vec<WorkerMetricSample>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open Parquet {}", path.display()))?;
    let mut reader = builder.with_batch_size(BATCH_ROWS).build()?;
    let mut samples = Vec::new();
    while let Some(batch) = reader.next() {
        let batch = batch?;
        for index in 0..batch.num_rows() {
            let Some(metric_name_clean) = string(&batch, "metric_name_clean", index) else {
                continue;
            };
            if !matches!(
                metric_name_clean.as_str(),
                "dynamo_frontend_worker_active_prefill_tokens"
                    | "dynamo_frontend_worker_last_time_to_first_token_seconds"
                    | "dynamo_frontend_worker_last_inter_token_latency_seconds"
                    | "sglang:num_running_reqs"
                    | "sglang:num_queue_reqs"
                    | "dynamo_component_gpu_cache_usage_percent"
            ) {
                continue;
            }
            let (Some(timestamp_ns), Some(scraper_endpoint), Some(metric_name), Some(metric_value)) = (
                integer(&batch, "timestamp_ns", index),
                string(&batch, "scraper_endpoint", index),
                string(&batch, "metric_name", index),
                number(&batch, "metric_value", index),
            ) else {
                continue;
            };
            let router_worker_id = if scraper_endpoint == "router" {
                WORKER_ID_LABEL
                    .captures(&metric_name)
                    .and_then(|captures| captures.name("worker_id"))
                    .map(|value| value.as_str().to_owned())
            } else {
                None
            };
            samples.push(WorkerMetricSample {
                timestamp_ns,
                scraper_endpoint,
                metric_name_clean,
                router_worker_id,
                metric_value,
            });
        }
    }
    Ok(samples)
}

fn requests_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("arm", DataType::Utf8, false),
        Field::new("request_id", DataType::Utf8, true),
        Field::new("session_id", DataType::Utf8, true),
        Field::new("conversation_id", DataType::Utf8, true),
        Field::new("credit_issued_ns", DataType::Int64, true),
        Field::new("request_start_ns", DataType::Int64, true),
        Field::new("request_ack_ns", DataType::Int64, true),
        Field::new("request_end_ns", DataType::Int64, true),
        Field::new("input_tokens", DataType::Int64, true),
        Field::new("output_tokens", DataType::Int64, true),
        Field::new("ttft_ms", DataType::Float64, true),
        Field::new("itl_ms", DataType::Float64, true),
        Field::new("e2e_ms", DataType::Float64, true),
        Field::new("raw_json", DataType::Utf8, false),
    ]))
}

fn request_traces_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("arm", DataType::Utf8, false),
        Field::new("request_id", DataType::Utf8, true),
        Field::new("x_request_id", DataType::Utf8, true),
        Field::new("event_time_ns", DataType::Int64, true),
        Field::new("request_received_ns", DataType::Int64, true),
        Field::new("input_tokens", DataType::Int64, true),
        Field::new("output_tokens", DataType::Int64, true),
        Field::new("cached_tokens", DataType::Int64, true),
        Field::new("kv_hit_rate", DataType::Float64, true),
        Field::new("ttft_ms", DataType::Float64, true),
        Field::new("total_time_ms", DataType::Float64, true),
        Field::new("avg_itl_ms", DataType::Float64, true),
        Field::new("queue_depth", DataType::Int64, true),
        Field::new("prefill_worker_id", DataType::Utf8, true),
        Field::new("prefill_dp_rank", DataType::Int64, true),
        Field::new("decode_worker_id", DataType::Utf8, true),
        Field::new("decode_dp_rank", DataType::Int64, true),
        Field::new("raw_json", DataType::Utf8, false),
    ]))
}

fn metrics_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("arm", DataType::Utf8, false),
        Field::new("timestamp_ns", DataType::Int64, true),
        Field::new("time_since_start_s", DataType::Float64, true),
        Field::new("scraper_endpoint", DataType::Utf8, true),
        Field::new("endpoint", DataType::Utf8, true),
        Field::new("metric_name", DataType::Utf8, true),
        Field::new("metric_name_clean", DataType::Utf8, true),
        Field::new("metric_value", DataType::Float64, true),
        Field::new("histogram_lower_bound", DataType::Float64, true),
        Field::new("histogram_upper_bound", DataType::Float64, true),
        Field::new("histogram_sum", DataType::Float64, true),
        Field::new("histogram_count", DataType::Float64, true),
        Field::new("node", DataType::Utf8, true),
        Field::new("process_index", DataType::Utf8, true),
        Field::new("router", DataType::Utf8, true),
        Field::new("endpoint_index", DataType::Utf8, true),
        Field::new("index", DataType::Utf8, true),
    ]))
}

#[derive(Default)]
struct RequestRows {
    arm: Vec<String>,
    request_id: Vec<Option<String>>,
    session_id: Vec<Option<String>>,
    conversation_id: Vec<Option<String>>,
    credit_issued_ns: Vec<Option<i64>>,
    request_start_ns: Vec<Option<i64>>,
    request_ack_ns: Vec<Option<i64>>,
    request_end_ns: Vec<Option<i64>>,
    input_tokens: Vec<Option<i64>>,
    output_tokens: Vec<Option<i64>>,
    ttft_ms: Vec<Option<f64>>,
    itl_ms: Vec<Option<f64>>,
    e2e_ms: Vec<Option<f64>>,
    raw_json: Vec<String>,
}

impl RequestRows {
    fn push(&mut self, row: &AiperfRequest) {
        self.arm.push(row.arm.clone());
        self.request_id.push(row.request_id.clone());
        self.session_id.push(row.session_id.clone());
        self.conversation_id.push(row.conversation_id.clone());
        self.credit_issued_ns.push(row.credit_issued_ns);
        self.request_start_ns.push(row.request_start_ns);
        self.request_ack_ns.push(row.request_ack_ns);
        self.request_end_ns.push(row.request_end_ns);
        self.input_tokens.push(row.input_tokens);
        self.output_tokens.push(row.output_tokens);
        self.ttft_ms.push(row.ttft_ms);
        self.itl_ms.push(row.itl_ms);
        self.e2e_ms.push(row.e2e_ms);
        self.raw_json.push(row.raw_json.clone());
    }

    fn into_batch(self, schema: Arc<Schema>) -> Result<RecordBatch> {
        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(self.arm)) as ArrayRef,
                Arc::new(StringArray::from(self.request_id)) as ArrayRef,
                Arc::new(StringArray::from(self.session_id)) as ArrayRef,
                Arc::new(StringArray::from(self.conversation_id)) as ArrayRef,
                Arc::new(Int64Array::from(self.credit_issued_ns)) as ArrayRef,
                Arc::new(Int64Array::from(self.request_start_ns)) as ArrayRef,
                Arc::new(Int64Array::from(self.request_ack_ns)) as ArrayRef,
                Arc::new(Int64Array::from(self.request_end_ns)) as ArrayRef,
                Arc::new(Int64Array::from(self.input_tokens)) as ArrayRef,
                Arc::new(Int64Array::from(self.output_tokens)) as ArrayRef,
                Arc::new(Float64Array::from(self.ttft_ms)) as ArrayRef,
                Arc::new(Float64Array::from(self.itl_ms)) as ArrayRef,
                Arc::new(Float64Array::from(self.e2e_ms)) as ArrayRef,
                Arc::new(StringArray::from(self.raw_json)) as ArrayRef,
            ],
        )?)
    }
}

#[derive(Default)]
struct MetricRows {
    arm: Vec<String>,
    timestamp_ns: Vec<Option<i64>>,
    time_since_start_s: Vec<Option<f64>>,
    scraper_endpoint: Vec<Option<String>>,
    endpoint: Vec<Option<String>>,
    metric_name: Vec<Option<String>>,
    metric_name_clean: Vec<Option<String>>,
    metric_value: Vec<Option<f64>>,
    histogram_lower_bound: Vec<Option<f64>>,
    histogram_upper_bound: Vec<Option<f64>>,
    histogram_sum: Vec<Option<f64>>,
    histogram_count: Vec<Option<f64>>,
    node: Vec<Option<String>>,
    process_index: Vec<Option<String>>,
    router: Vec<Option<String>>,
    endpoint_index: Vec<Option<String>>,
    index: Vec<Option<String>>,
}

impl MetricRows {
    fn len(&self) -> usize {
        self.arm.len()
    }

    fn push(&mut self, arm: &str, anchor_ns: Option<i64>, batch: &RecordBatch, row: usize) {
        let elapsed = number(batch, "time_since_start", row);
        self.arm.push(arm.to_owned());
        self.timestamp_ns.push(
            anchor_ns
                .zip(elapsed)
                .map(|(anchor, seconds)| anchor.saturating_add((seconds * 1_000_000_000.0) as i64)),
        );
        self.time_since_start_s.push(elapsed);
        self.scraper_endpoint
            .push(string(batch, "scraper_endpoint", row));
        self.endpoint.push(string(batch, "endpoint", row));
        self.metric_name.push(string(batch, "metric_name", row));
        self.metric_name_clean
            .push(string(batch, "metric_name_clean", row));
        self.metric_value.push(number(batch, "metric_value", row));
        self.histogram_lower_bound
            .push(number(batch, "histogram_bucket_lower", row));
        self.histogram_upper_bound
            .push(number(batch, "histogram_bucket_upper", row));
        self.histogram_sum.push(number(batch, "histogram_sum", row));
        self.histogram_count
            .push(number(batch, "histogram_count", row));
        self.node.push(string(batch, "node", row));
        self.process_index
            .push(scalar_string(batch, "process_index", row));
        self.router.push(string(batch, "router", row));
        self.endpoint_index
            .push(scalar_string(batch, "endpoint_index", row));
        self.index.push(scalar_string(batch, "index", row));
    }

    fn flush(&mut self, writer: &mut ArrowWriter<File>, schema: Arc<Schema>) -> Result<()> {
        if self.len() == 0 {
            return Ok(());
        }
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(std::mem::take(&mut self.arm))) as ArrayRef,
                Arc::new(Int64Array::from(std::mem::take(&mut self.timestamp_ns))) as ArrayRef,
                Arc::new(Float64Array::from(std::mem::take(
                    &mut self.time_since_start_s,
                ))) as ArrayRef,
                Arc::new(StringArray::from(std::mem::take(
                    &mut self.scraper_endpoint,
                ))) as ArrayRef,
                Arc::new(StringArray::from(std::mem::take(&mut self.endpoint))) as ArrayRef,
                Arc::new(StringArray::from(std::mem::take(&mut self.metric_name))) as ArrayRef,
                Arc::new(StringArray::from(std::mem::take(
                    &mut self.metric_name_clean,
                ))) as ArrayRef,
                Arc::new(Float64Array::from(std::mem::take(&mut self.metric_value))) as ArrayRef,
                Arc::new(Float64Array::from(std::mem::take(
                    &mut self.histogram_lower_bound,
                ))) as ArrayRef,
                Arc::new(Float64Array::from(std::mem::take(
                    &mut self.histogram_upper_bound,
                ))) as ArrayRef,
                Arc::new(Float64Array::from(std::mem::take(&mut self.histogram_sum))) as ArrayRef,
                Arc::new(Float64Array::from(std::mem::take(
                    &mut self.histogram_count,
                ))) as ArrayRef,
                Arc::new(StringArray::from(std::mem::take(&mut self.node))) as ArrayRef,
                Arc::new(StringArray::from(std::mem::take(&mut self.process_index))) as ArrayRef,
                Arc::new(StringArray::from(std::mem::take(&mut self.router))) as ArrayRef,
                Arc::new(StringArray::from(std::mem::take(&mut self.endpoint_index))) as ArrayRef,
                Arc::new(StringArray::from(std::mem::take(&mut self.index))) as ArrayRef,
            ],
        )?;
        writer.write(&batch)?;
        Ok(())
    }
}

fn value_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
}

fn string_at(value: &Value, path: &[&str]) -> Option<String> {
    value_at(value, path).and_then(|value| match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn i64_at(value: &Value, path: &[&str]) -> Option<i64> {
    value_at(value, path).and_then(Value::as_i64)
}
fn f64_at(value: &Value, path: &[&str]) -> Option<f64> {
    value_at(value, path).and_then(Value::as_f64)
}

fn string(batch: &RecordBatch, name: &str, index: usize) -> Option<String> {
    let array = batch.column_by_name(name)?;
    if array.is_null(index) {
        return None;
    }
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return Some(array.value(index).to_owned());
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Some(array.value(index).to_owned());
    }
    array
        .as_any()
        .downcast_ref::<StringViewArray>()
        .map(|array| array.value(index).to_owned())
}

fn number(batch: &RecordBatch, name: &str, index: usize) -> Option<f64> {
    let array = batch.column_by_name(name)?;
    if array.is_null(index) {
        return None;
    }
    if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
        return Some(array.value(index));
    }
    if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
        return Some(array.value(index) as f64);
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        return Some(array.value(index) as f64);
    }
    if let Some(array) = array.as_any().downcast_ref::<UInt32Array>() {
        return Some(array.value(index) as f64);
    }
    if let Some(array) = array.as_any().downcast_ref::<UInt64Array>() {
        return Some(array.value(index) as f64);
    }
    None
}

fn integer(batch: &RecordBatch, name: &str, index: usize) -> Option<i64> {
    let array = batch.column_by_name(name)?;
    if array.is_null(index) {
        return None;
    }
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .map(|array| array.value(index))
}

fn scalar_string(batch: &RecordBatch, name: &str, index: usize) -> Option<String> {
    string(batch, name, index).or_else(|| number(batch, name, index).map(|value| value.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn reads_tachometer_large_strings_and_floats() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("metric_name", DataType::LargeUtf8, true),
            Field::new("metric_value", DataType::Float32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(LargeStringArray::from(vec![Some(
                    "sglang:num_running_reqs",
                )])) as ArrayRef,
                Arc::new(Float32Array::from(vec![Some(3.0)])) as ArrayRef,
            ],
        )
        .unwrap();
        assert_eq!(
            string(&batch, "metric_name", 0).as_deref(),
            Some("sglang:num_running_reqs")
        );
        assert_eq!(number(&batch, "metric_value", 0), Some(3.0));
    }

    #[test]
    fn loads_gzip_dynamo_request_trace() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("dynamo-request-trace.000000.jsonl.gz");
        let file = File::create(&path).unwrap();
        let mut writer = GzEncoder::new(file, Compression::default());
        writer
            .write_all(
                br#"{"timestamp":0,"event":{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":2000,"request":{"request_id":"req-1","kv_hit_rate":0.5}}}
"#,
            )
            .unwrap();
        writer.finish().unwrap();

        let rows = load_request_traces(&[("dynamo", path.as_path())]).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].request_id.as_deref(), Some("req-1"));
        assert_eq!(rows[0].kv_hit_rate, Some(0.5));
    }
}
