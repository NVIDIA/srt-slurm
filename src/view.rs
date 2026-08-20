//! Small static dashboard backed only by the SQLite database made by `ruter init`.

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;
use tiny_http::{Header, Response, Server, StatusCode};

const INDEX: &str = include_str!("../ui/index.html");
const APP: &str = include_str!("../ui/app.js");
const BEAVER_BADGE: &[u8] = include_bytes!("../ui/assets/certified-beaver.png");
const BEAVER_AUDIO: &[u8] = include_bytes!("../ui/assets/ruter.mp3");

pub fn launch(analysis_dir: &Path, port: Option<u16>) -> Result<()> {
    for file in ["manifest.json", "ruter.db"] {
        if !analysis_dir.join(file).is_file() {
            bail!(
                "{} is not a complete ruter analysis directory; missing {file}",
                analysis_dir.display()
            );
        }
    }
    let port = port.unwrap_or(8877);
    let address = format!("127.0.0.1:{port}");
    let server =
        Server::http(&address).map_err(|error| anyhow::anyhow!("listen on {address}: {error}"))?;
    println!("ruter view: http://{address}");
    for request in server.incoming_requests() {
        let url = request.url();
        let response = match url.split('?').next().unwrap_or("/") {
            "/" | "/index.html" => response(INDEX, "text/html; charset=utf-8"),
            "/app.js" => response(APP, "application/javascript; charset=utf-8"),
            "/assets/certified-beaver.png" => bytes_response(BEAVER_BADGE, "image/png"),
            "/assets/ruter.mp3" => bytes_response(BEAVER_AUDIO, "audio/mpeg"),
            "/api/summary" => json_response(summary(analysis_dir)),
            "/api/timeline" => json_response(timeline(analysis_dir)),
            "/api/decisions" => json_response(decisions(analysis_dir)),
            "/api/decision" => json_response(decision(analysis_dir, query_value(url, "id"))),
            _ => Response::from_string("not found").with_status_code(StatusCode(404)),
        };
        request.respond(response)?;
    }
    Ok(())
}

fn query_value<'a>(url: &'a str, wanted: &str) -> Option<&'a str> {
    url.split_once('?')?.1.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        (key == wanted).then_some(value)
    })
}

fn response(body: &str, content_type: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_data(body.as_bytes().to_vec())
        .with_header(Header::from_bytes("Content-Type", content_type).expect("static header"))
        .with_header(cache_control())
}

fn bytes_response(body: &[u8], content_type: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_data(body.to_vec())
        .with_header(Header::from_bytes("Content-Type", content_type).expect("static header"))
        .with_header(cache_control())
}

fn cache_control() -> Header {
    Header::from_bytes("Cache-Control", "no-store").expect("valid cache header")
}

fn json_response(result: Result<Value>) -> Response<std::io::Cursor<Vec<u8>>> {
    match result {
        Ok(value) => response(
            &serde_json::to_string(&value).expect("JSON value serializes"),
            "application/json",
        ),
        Err(error) => Response::from_string(format!("database query failed: {error:#}"))
            .with_status_code(StatusCode(500)),
    }
}

fn open(analysis_dir: &Path) -> Result<Connection> {
    Connection::open(analysis_dir.join("ruter.db")).context("open ruter.db")
}

fn metadata(connection: &Connection, key: &str) -> Result<Option<Value>> {
    let value = connection
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            params![key],
            |row| row.get(0),
        )
        .optional()?;
    value
        .map(|value: String| serde_json::from_str(&value).context("parse metadata JSON"))
        .transpose()
}

fn benchmark_start(connection: &Connection) -> Result<Option<i64>> {
    connection
        .query_row(
            "
            SELECT COALESCE(
                (SELECT MIN(credit_issued_ns) FROM aiperf_requests WHERE credit_issued_ns IS NOT NULL),
                (SELECT MIN(request_received_ns) FROM request_traces WHERE request_received_ns IS NOT NULL),
                (SELECT MIN(timestamp_ns) FROM routing_decisions WHERE timestamp_ns IS NOT NULL)
            )
            ",
            [],
            |row| row.get(0),
        )
        .context("read benchmark start")
}

/// Stable, compact display names for opaque Dynamo instance IDs. The raw ID is
/// retained in every API row, so the presentation alias never loses evidence.
fn worker_aliases(connection: &Connection) -> Result<BTreeMap<String, String>> {
    let mut statement = connection.prepare(
        "SELECT DISTINCT worker_id FROM routing_decisions WHERE worker_id IS NOT NULL ORDER BY worker_id",
    )?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    let mut aliases = BTreeMap::new();
    for (index, worker_id) in rows.enumerate() {
        aliases.insert(worker_id?, alphabetic_alias(index));
    }
    Ok(aliases)
}

fn alphabetic_alias(mut index: usize) -> String {
    let mut alias = String::new();
    loop {
        alias.insert(0, char::from(b'A' + (index % 26) as u8));
        if index < 26 {
            return alias;
        }
        index = index / 26 - 1;
    }
}

fn summary(analysis_dir: &Path) -> Result<Value> {
    let connection = open(analysis_dir)?;
    let worker_aliases = worker_aliases(&connection)?;
    let traces: i64 =
        connection.query_row("SELECT COUNT(*) FROM request_traces", [], |row| row.get(0))?;
    let aiperf_requests: i64 =
        connection.query_row("SELECT COUNT(*) FROM aiperf_requests", [], |row| row.get(0))?;
    let decisions: i64 =
        connection.query_row("SELECT COUNT(*) FROM routing_decisions", [], |row| {
            row.get(0)
        })?;
    let workers: i64 = connection.query_row(
        "SELECT COUNT(DISTINCT worker_id) FROM routing_decisions WHERE worker_id IS NOT NULL",
        [],
        |row| row.get(0),
    )?;
    let avg_kv_hit_rate: Option<f64> =
        connection.query_row("SELECT AVG(kv_hit_rate) FROM request_traces", [], |row| {
            row.get(0)
        })?;
    let avg_ttft_ms: Option<f64> =
        connection.query_row("SELECT AVG(ttft_ms) FROM request_traces", [], |row| {
            row.get(0)
        })?;
    let router_settings = metadata(&connection, "dynamo.router_settings")?;
    Ok(json!({
        "requestTraces": traces,
        "aiperfRequests": aiperf_requests,
        "decisions": decisions,
        "workers": workers,
        "workerAliases": worker_aliases.values().collect::<Vec<_>>(),
        "avgKvHitRate": avg_kv_hit_rate,
        "avgTtftMs": avg_ttft_ms,
        "routerSettings": router_settings,
    }))
}

fn timeline(analysis_dir: &Path) -> Result<Value> {
    let connection = open(analysis_dir)?;
    let worker_aliases = worker_aliases(&connection)?;
    let Some(start_ns) = benchmark_start(&connection)? else {
        return Ok(json!({"traces": [], "aiperf": []}));
    };
    let mut traces = Vec::new();
    let mut statement = connection.prepare(
        "
        SELECT (COALESCE(request_traces.request_received_ns, request_traces.event_time_ns) - ?1) / 1000000000.0,
               COALESCE(request_traces.x_request_id, request_traces.request_id), request_traces.request_id,
               request_traces.input_tokens, request_traces.cached_tokens, request_traces.kv_hit_rate,
               request_traces.ttft_ms, request_traces.total_time_ms, request_traces.queue_depth,
               request_traces.prefill_worker_id, request_traces.prefill_dp_rank,
               COALESCE(d.lower_prefix_selected, 0)
        FROM request_traces
        LEFT JOIN routing_decisions d ON d.dynamo_request_id = request_traces.request_id
        WHERE COALESCE(request_traces.request_received_ns, request_traces.event_time_ns) IS NOT NULL
        ORDER BY COALESCE(request_traces.request_received_ns, request_traces.event_time_ns)
        ",
    )?;
    let rows = statement.query_map(params![start_ns], |row| {
        Ok(json!({
            "benchS": row.get::<_, f64>(0)?, "requestId": row.get::<_, Option<String>>(1)?,
            "dynamoRequestId": row.get::<_, Option<String>>(2)?,
            "inputTokens": row.get::<_, Option<i64>>(3)?, "cachedTokens": row.get::<_, Option<i64>>(4)?,
            "kvHitRate": row.get::<_, Option<f64>>(5)?, "ttftMs": row.get::<_, Option<f64>>(6)?,
            "e2eMs": row.get::<_, Option<f64>>(7)?, "queueDepth": row.get::<_, Option<i64>>(8)?,
            "prefillWorkerId": row.get::<_, Option<String>>(9)?, "dpRank": row.get::<_, Option<i64>>(10)?,
            "lowerPrefixSelected": row.get::<_, bool>(11)?,
        }))
    })?;
    for row in rows {
        let mut row = row?;
        let raw_worker_id = row
            .get("prefillWorkerId")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let worker_alias = raw_worker_id
            .as_ref()
            .and_then(|worker_id| worker_aliases.get(worker_id));
        row["prefillWorkerAlias"] = json!(worker_alias);
        row.as_object_mut()
            .expect("timeline row is an object")
            .remove("prefillWorkerId");
        traces.push(row)
    }

    let mut aiperf = Vec::new();
    let mut statement = connection.prepare(
        "
        SELECT (credit_issued_ns - ?1) / 1000000000.0, request_id, input_tokens, output_tokens, ttft_ms, e2e_ms
        FROM aiperf_requests WHERE credit_issued_ns IS NOT NULL ORDER BY credit_issued_ns
        ",
    )?;
    let rows = statement.query_map(params![start_ns], |row| {
        Ok(json!({
            "benchS": row.get::<_, f64>(0)?, "requestId": row.get::<_, Option<String>>(1)?,
            "inputTokens": row.get::<_, Option<i64>>(2)?, "outputTokens": row.get::<_, Option<i64>>(3)?,
            "ttftMs": row.get::<_, Option<f64>>(4)?, "e2eMs": row.get::<_, Option<f64>>(5)?,
        }))
    })?;
    for row in rows {
        aiperf.push(row?)
    }
    Ok(json!({"traces": traces, "aiperf": aiperf}))
}

fn decisions(analysis_dir: &Path) -> Result<Value> {
    let connection = open(analysis_dir)?;
    let worker_aliases = worker_aliases(&connection)?;
    let Some(start_ns) = benchmark_start(&connection)? else {
        return Ok(json!([]));
    };
    let mut statement = connection.prepare(
        "
        SELECT (d.timestamp_ns - ?1) / 1000000000.0,
               d.dynamo_request_id,
               d.worker_id, d.dp_rank, d.overlap_blocks, d.total_blocks, c.cost_blocks,
               d.lower_prefix_selected
        FROM routing_decisions d
        LEFT JOIN routing_candidates c
          ON c.dynamo_request_id = d.dynamo_request_id AND c.worker_id = d.worker_id
        WHERE d.timestamp_ns IS NOT NULL
        ORDER BY d.timestamp_ns LIMIT 5000
        ",
    )?;
    let rows = statement.query_map(params![start_ns], |row| {
        Ok(json!({
            "benchS": row.get::<_, f64>(0)?, "dynamoRequestId": row.get::<_, Option<String>>(1)?,
            "workerId": row.get::<_, Option<String>>(2)?, "dpRank": row.get::<_, Option<i64>>(3)?,
            "overlapBlocks": row.get::<_, Option<i64>>(4)?, "totalBlocks": row.get::<_, Option<i64>>(5)?,
            "costBlocks": row.get::<_, Option<f64>>(6)?,
            "lowerPrefixSelected": row.get::<_, bool>(7)?,
        }))
    })?;
    let mut decisions = Vec::new();
    for row in rows {
        let mut row = row?;
        let raw_worker_id = row
            .get("workerId")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let worker_alias = raw_worker_id
            .as_ref()
            .and_then(|worker_id| worker_aliases.get(worker_id));
        row["workerAlias"] = json!(worker_alias);
        row.as_object_mut()
            .expect("decision row is an object")
            .remove("workerId");
        decisions.push(row);
    }
    Ok(Value::Array(decisions))
}

/// One request's exact router formula and the last Tachometer observation from
/// before it. The endpoint intentionally returns only the selected request so
/// the browser never needs to sift through millions of raw metric rows.
fn decision(analysis_dir: &Path, dynamo_request_id: Option<&str>) -> Result<Value> {
    let Some(dynamo_request_id) = dynamo_request_id.filter(|value| !value.is_empty()) else {
        return Ok(json!({"found": false}));
    };
    let connection = open(analysis_dir)?;
    let worker_aliases = worker_aliases(&connection)?;
    let Some(start_ns) = benchmark_start(&connection)? else {
        return Ok(json!({"found": false}));
    };
    let decision = connection
        .query_row(
            "
            SELECT (d.timestamp_ns - ?1) / 1000000000.0,
                   d.worker_id, d.dp_rank, d.overlap_blocks, d.total_blocks,
                   t.kv_hit_rate, t.ttft_ms, t.total_time_ms, t.input_tokens, t.cached_tokens
            FROM routing_decisions d
            LEFT JOIN request_traces t ON t.request_id = d.dynamo_request_id
            WHERE d.dynamo_request_id = ?2
            ORDER BY d.timestamp_ns
            LIMIT 1
            ",
            params![start_ns, dynamo_request_id],
            |row| {
                Ok((
                    row.get::<_, f64>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                    row.get::<_, Option<i64>>(4)?,
                    row.get::<_, Option<f64>>(5)?,
                    row.get::<_, Option<f64>>(6)?,
                    row.get::<_, Option<f64>>(7)?,
                    row.get::<_, Option<i64>>(8)?,
                    row.get::<_, Option<i64>>(9)?,
                ))
            },
        )
        .optional()?;
    let Some((
        bench_s,
        selected_worker_id,
        dp_rank,
        overlap_blocks,
        total_blocks,
        kv_hit_rate,
        ttft_ms,
        e2e_ms,
        input_tokens,
        cached_tokens,
    )) = decision
    else {
        return Ok(json!({"found": false}));
    };

    let mut statement = connection.prepare(
        "
        SELECT c.worker_id, c.dp_rank, c.cost_blocks, c.effective_cached_blocks,
               c.prefill_load_scale, c.adjusted_prefill_blocks, c.raw_prefill_blocks,
               c.overlap_credit_blocks, c.overlap_credit_decay, c.decode_blocks,
               c.active_request_cost_blocks,
               s.router_sample_age_ms, s.active_prefill_tokens, s.last_ttft_ms, s.last_itl_ms,
               s.engine_sample_age_ms, s.running_reqs, s.queued_reqs, s.gpu_cache_usage_fraction
        FROM routing_candidates c
        LEFT JOIN worker_state_snapshots s
          ON s.dynamo_request_id = c.dynamo_request_id AND s.worker_id = c.worker_id
        WHERE c.dynamo_request_id = ?1
        ORDER BY c.cost_blocks, c.worker_id
        ",
    )?;
    let rows = statement.query_map(params![dynamo_request_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<i64>>(1)?,
            row.get::<_, Option<f64>>(2)?,
            row.get::<_, Option<f64>>(3)?,
            row.get::<_, Option<f64>>(4)?,
            row.get::<_, Option<f64>>(5)?,
            row.get::<_, Option<f64>>(6)?,
            row.get::<_, Option<f64>>(7)?,
            row.get::<_, Option<f64>>(8)?,
            row.get::<_, Option<f64>>(9)?,
            row.get::<_, Option<f64>>(10)?,
            row.get::<_, Option<f64>>(11)?,
            row.get::<_, Option<f64>>(12)?,
            row.get::<_, Option<f64>>(13)?,
            row.get::<_, Option<f64>>(14)?,
            row.get::<_, Option<f64>>(15)?,
            row.get::<_, Option<f64>>(16)?,
            row.get::<_, Option<f64>>(17)?,
            row.get::<_, Option<f64>>(18)?,
        ))
    })?;
    let mut candidates = Vec::new();
    for row in rows {
        let (
            worker_id,
            candidate_dp_rank,
            cost_blocks,
            effective_cached_blocks,
            prefill_load_scale,
            adjusted_prefill_blocks,
            raw_prefill_blocks,
            overlap_credit_blocks,
            overlap_credit_decay,
            decode_blocks,
            active_request_cost_blocks,
            router_sample_age_ms,
            active_prefill_tokens,
            last_ttft_ms,
            last_itl_ms,
            engine_sample_age_ms,
            running_reqs,
            queued_reqs,
            gpu_cache_usage_fraction,
        ) = row?;
        let selected = selected_worker_id.as_deref() == Some(worker_id.as_str());
        candidates.push(json!({
            "workerAlias": worker_aliases.get(&worker_id),
            "selected": selected,
            "dpRank": candidate_dp_rank,
            "costBlocks": cost_blocks,
            "effectiveCachedBlocks": effective_cached_blocks,
            "prefillLoadScale": prefill_load_scale,
            "adjustedPrefillBlocks": adjusted_prefill_blocks,
            "rawPrefillBlocks": raw_prefill_blocks,
            "overlapCreditBlocks": overlap_credit_blocks,
            "overlapCreditDecay": overlap_credit_decay,
            "decodeBlocks": decode_blocks,
            "activeRequestCostBlocks": active_request_cost_blocks,
            "routerSampleAgeMs": router_sample_age_ms,
            "activePrefillTokens": active_prefill_tokens,
            "lastTtftMs": last_ttft_ms,
            "lastItlMs": last_itl_ms,
            "engineSampleAgeMs": engine_sample_age_ms,
            "runningReqs": running_reqs,
            "queuedReqs": queued_reqs,
            "gpuCacheUsageFraction": gpu_cache_usage_fraction,
        }));
    }
    Ok(json!({
        "found": true,
        "benchS": bench_s,
        "selectedWorkerAlias": selected_worker_id.as_ref().and_then(|id| worker_aliases.get(id)),
        "dpRank": dp_rank,
        "overlapBlocks": overlap_blocks,
        "totalBlocks": total_blocks,
        "kvHitRate": kv_hit_rate,
        "ttftMs": ttft_ms,
        "e2eMs": e2e_ms,
        "inputTokens": input_tokens,
        "cachedTokens": cached_tokens,
        "candidates": candidates,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serves_a_static_page_and_no_streamlit_runtime() {
        assert!(INDEX.contains("route decision"));
        assert!(INDEX.contains("beaver-badge"));
        assert!(APP.contains("/api/timeline"));
        assert!(APP.contains("/api/decision"));
        assert!(APP.contains("setupBeaverAudio"));
        assert_eq!(&BEAVER_BADGE[..8], b"\x89PNG\r\n\x1a\n");
        assert!(!BEAVER_AUDIO.is_empty());
        assert!(!INDEX.contains("streamlit"));
    }

    #[test]
    fn makes_worker_aliases_human_readable() {
        assert_eq!(alphabetic_alias(0), "A");
        assert_eq!(alphabetic_alias(7), "H");
        assert_eq!(alphabetic_alias(25), "Z");
        assert_eq!(alphabetic_alias(26), "AA");
    }
}
