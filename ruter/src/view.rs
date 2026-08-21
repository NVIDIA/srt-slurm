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

type WorkerAliases = BTreeMap<(String, String), String>;

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

/// Opaque Dynamo instance IDs become role-specific compact display names.
/// The phase is part of the identity: a 3P2D run shows P-A..P-C and D-A..D-B.
fn worker_aliases(connection: &Connection) -> Result<WorkerAliases> {
    let mut statement = connection.prepare(
        "SELECT DISTINCT phase, worker_id FROM routing_decisions WHERE worker_id IS NOT NULL ORDER BY phase, worker_id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut aliases = WorkerAliases::new();
    let mut next_by_phase = BTreeMap::<String, usize>::new();
    for row in rows {
        let (phase, worker_id) = row?;
        let index = next_by_phase.entry(phase.clone()).or_default();
        let prefix = match phase.as_str() {
            "prefill" => "P-",
            "decode" => "D-",
            _ => "",
        };
        aliases.insert(
            (phase, worker_id),
            format!("{prefix}{}", alphabetic_alias(*index)),
        );
        *index += 1;
    }
    Ok(aliases)
}

fn alias(aliases: &WorkerAliases, phase: &str, worker_id: Option<&str>) -> Option<String> {
    worker_id.and_then(|worker_id| {
        aliases
            .get(&(phase.to_owned(), worker_id.to_owned()))
            .cloned()
    })
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
    let aliases = worker_aliases(&connection)?;
    let traces: i64 =
        connection.query_row("SELECT COUNT(*) FROM request_traces", [], |row| row.get(0))?;
    let aiperf_requests: i64 =
        connection.query_row("SELECT COUNT(*) FROM aiperf_requests", [], |row| row.get(0))?;
    let decisions: i64 =
        connection.query_row("SELECT COUNT(*) FROM routing_decisions", [], |row| {
            row.get(0)
        })?;
    let avg_kv_hit_rate: Option<f64> =
        connection.query_row("SELECT AVG(kv_hit_rate) FROM request_traces", [], |row| {
            row.get(0)
        })?;
    let avg_ttft_ms: Option<f64> =
        connection.query_row("SELECT AVG(ttft_ms) FROM request_traces", [], |row| {
            row.get(0)
        })?;
    Ok(json!({
        "requestTraces": traces,
        "aiperfRequests": aiperf_requests,
        "decisions": decisions,
        "workerAliases": aliases.values().collect::<Vec<_>>(),
        "avgKvHitRate": avg_kv_hit_rate,
        "avgTtftMs": avg_ttft_ms,
        "routerSettings": metadata(&connection, "dynamo.router_settings")?,
    }))
}

fn timeline(analysis_dir: &Path) -> Result<Value> {
    let connection = open(analysis_dir)?;
    let aliases = worker_aliases(&connection)?;
    let Some(start_ns) = benchmark_start(&connection)? else {
        return Ok(json!({"traces": []}));
    };
    let mut statement = connection.prepare(
        "
        SELECT (COALESCE(t.request_received_ns, t.event_time_ns) - ?1) / 1000000000.0,
               COALESCE(t.x_request_id, t.request_id), t.request_id, t.kv_hit_rate,
               p.phase, p.worker_id, COALESCE(p.lower_prefix_selected, 0)
        FROM request_traces t
        LEFT JOIN routing_decisions p ON p.decision_id = (
            SELECT decision_id FROM routing_decisions d
            WHERE d.dynamo_request_id = t.request_id AND d.phase IN ('prefill', 'aggregated')
            ORDER BY CASE d.phase WHEN 'prefill' THEN 0 ELSE 1 END, d.timestamp_ns
            LIMIT 1
        )
        WHERE COALESCE(t.request_received_ns, t.event_time_ns) IS NOT NULL
        ORDER BY COALESCE(t.request_received_ns, t.event_time_ns)
        ",
    )?;
    let rows = statement.query_map(params![start_ns], |row| {
        Ok((
            row.get::<_, f64>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, Option<f64>>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, bool>(6)?,
        ))
    })?;
    let mut traces = Vec::new();
    for row in rows {
        let (
            bench_s,
            request_id,
            dynamo_request_id,
            kv_hit_rate,
            phase,
            worker_id,
            lower_prefix_selected,
        ) = row?;
        let phase = phase.unwrap_or_else(|| "prefill".to_owned());
        traces.push(json!({
            "benchS": bench_s,
            "requestId": request_id,
            "dynamoRequestId": dynamo_request_id,
            "kvHitRate": kv_hit_rate,
            "prefillWorkerAlias": alias(&aliases, &phase, worker_id.as_deref()),
            "lowerPrefixSelected": lower_prefix_selected,
        }));
    }
    Ok(json!({"traces": traces}))
}

fn decisions(analysis_dir: &Path) -> Result<Value> {
    let connection = open(analysis_dir)?;
    let aliases = worker_aliases(&connection)?;
    let Some(start_ns) = benchmark_start(&connection)? else {
        return Ok(json!([]));
    };
    let mut statement = connection.prepare(
        "
        SELECT (d.timestamp_ns - ?1) / 1000000000.0, d.dynamo_request_id, d.phase,
               d.worker_id, d.dp_rank, d.overlap_blocks, d.total_blocks, c.cost_blocks,
               d.lower_prefix_selected
        FROM routing_decisions d
        LEFT JOIN routing_candidates c ON c.decision_id = d.decision_id AND c.worker_id = d.worker_id
        WHERE d.timestamp_ns IS NOT NULL
        ORDER BY d.timestamp_ns LIMIT 5000
        ",
    )?;
    let rows = statement.query_map(params![start_ns], |row| {
        Ok((
            row.get::<_, f64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<i64>>(4)?,
            row.get::<_, Option<i64>>(5)?,
            row.get::<_, Option<i64>>(6)?,
            row.get::<_, Option<f64>>(7)?,
            row.get::<_, bool>(8)?,
        ))
    })?;
    let mut output = Vec::new();
    for row in rows {
        let (
            bench_s,
            dynamo_request_id,
            phase,
            worker_id,
            dp_rank,
            overlap_blocks,
            total_blocks,
            cost_blocks,
            lower_prefix_selected,
        ) = row?;
        output.push(json!({
            "benchS": bench_s,
            "dynamoRequestId": dynamo_request_id,
            "phase": phase,
            "workerAlias": alias(&aliases, &phase, worker_id.as_deref()),
            "dpRank": dp_rank,
            "overlapBlocks": overlap_blocks,
            "totalBlocks": total_blocks,
            "costBlocks": cost_blocks,
            "lowerPrefixSelected": lower_prefix_selected,
        }));
    }
    Ok(Value::Array(output))
}

/// One request's exact P and D decision sheets. The browser receives only the
/// selected request's small precomputed snapshots, never raw scrape series.
fn decision(analysis_dir: &Path, dynamo_request_id: Option<&str>) -> Result<Value> {
    let Some(dynamo_request_id) = dynamo_request_id.filter(|value| !value.is_empty()) else {
        return Ok(json!({"found": false}));
    };
    let connection = open(analysis_dir)?;
    let aliases = worker_aliases(&connection)?;
    let Some(start_ns) = benchmark_start(&connection)? else {
        return Ok(json!({"found": false}));
    };
    let request_facts = connection
        .query_row(
            "
            SELECT kv_hit_rate, ttft_ms, total_time_ms, input_tokens, cached_tokens
            FROM request_traces WHERE request_id = ?1 LIMIT 1
            ",
            params![dynamo_request_id],
            |row| {
                Ok(json!({
                    "kvHitRate": row.get::<_, Option<f64>>(0)?,
                    "ttftMs": row.get::<_, Option<f64>>(1)?,
                    "e2eMs": row.get::<_, Option<f64>>(2)?,
                    "inputTokens": row.get::<_, Option<i64>>(3)?,
                    "cachedTokens": row.get::<_, Option<i64>>(4)?,
                }))
            },
        )
        .optional()?;
    let mut statement = connection.prepare(
        "
        SELECT decision_id, (timestamp_ns - ?1) / 1000000000.0, phase, worker_id, dp_rank,
               overlap_blocks, total_blocks, lower_prefix_selected
        FROM routing_decisions WHERE dynamo_request_id = ?2 ORDER BY timestamp_ns
        ",
    )?;
    let rows = statement.query_map(params![start_ns, dynamo_request_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, f64>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<i64>>(4)?,
            row.get::<_, Option<i64>>(5)?,
            row.get::<_, Option<i64>>(6)?,
            row.get::<_, bool>(7)?,
        ))
    })?;
    let mut phases = Vec::new();
    for row in rows {
        let (
            decision_id,
            bench_s,
            phase,
            worker_id,
            dp_rank,
            overlap_blocks,
            total_blocks,
            lower_prefix_selected,
        ) = row?;
        phases.push(json!({
            "phase": phase,
            "benchS": bench_s,
            "selectedWorkerAlias": alias(&aliases, &phase, worker_id.as_deref()),
            "dpRank": dp_rank,
            "overlapBlocks": overlap_blocks,
            "totalBlocks": total_blocks,
            "lowerPrefixSelected": lower_prefix_selected,
            "candidates": candidates(&connection, &aliases, &decision_id, &phase, worker_id.as_deref())?,
        }));
    }
    if phases.is_empty() {
        return Ok(json!({"found": false}));
    }
    Ok(json!({
        "found": true,
        "facts": request_facts.unwrap_or_else(|| json!({})),
        "phases": phases,
    }))
}

fn candidates(
    connection: &Connection,
    aliases: &WorkerAliases,
    decision_id: &str,
    phase: &str,
    selected_worker_id: Option<&str>,
) -> Result<Vec<Value>> {
    let mut statement = connection.prepare(
        "
        SELECT c.worker_id, c.dp_rank, c.cost_blocks, c.effective_cached_blocks,
               c.prefill_load_scale, c.adjusted_prefill_blocks, c.decode_blocks,
               c.active_request_cost_blocks, s.running_reqs, s.queued_reqs,
               s.gpu_cache_usage_fraction, s.router_sample_age_ms, s.engine_sample_age_ms
        FROM routing_candidates c
        LEFT JOIN worker_state_snapshots s ON s.decision_id = c.decision_id AND s.worker_id = c.worker_id
        WHERE c.decision_id = ?1 ORDER BY c.cost_blocks, c.worker_id
        ",
    )?;
    let rows = statement.query_map(params![decision_id], |row| {
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
        ))
    })?;
    let mut output = Vec::new();
    for row in rows {
        let (
            worker_id,
            dp_rank,
            cost_blocks,
            effective_cached_blocks,
            prefill_load_scale,
            adjusted_prefill_blocks,
            decode_blocks,
            active_request_cost_blocks,
            running_reqs,
            queued_reqs,
            gpu_cache_usage_fraction,
            router_sample_age_ms,
            engine_sample_age_ms,
        ) = row?;
        output.push(json!({
            "workerAlias": alias(aliases, phase, Some(&worker_id)),
            "selected": selected_worker_id == Some(worker_id.as_str()),
            "dpRank": dp_rank,
            "costBlocks": cost_blocks,
            "effectiveCachedBlocks": effective_cached_blocks,
            "prefillLoadScale": prefill_load_scale,
            "adjustedPrefillBlocks": adjusted_prefill_blocks,
            "decodeBlocks": decode_blocks,
            "activeRequestCostBlocks": active_request_cost_blocks,
            "runningReqs": running_reqs,
            "queuedReqs": queued_reqs,
            "gpuCacheUsageFraction": gpu_cache_usage_fraction,
            "routerSampleAgeMs": router_sample_age_ms,
            "engineSampleAgeMs": engine_sample_age_ms,
        }));
    }
    Ok(output)
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
