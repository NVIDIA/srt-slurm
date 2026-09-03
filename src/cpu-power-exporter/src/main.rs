// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal Prometheus exporter for Grace CPU power via ACPI hwmon.
//!
//! Reads /sys/class/hwmon/hwmon*/power*_average, exposes:
//!   cpu_power_acpi_watts{sensor, type, socket, oem_info, source}
//!
//! Endpoints:
//!   GET /metrics  — Prometheus text format
//!   GET /health   — "ok\n"

use anyhow::{Context, Result};
use clap::Parser;
use std::collections::btree_map::{BTreeMap, Entry};
use std::fmt::Write as _;
use std::io::ErrorKind;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, SyncSender};
use std::{fs, process, thread};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::signal;
use tokio::signal::unix::{signal as unix_signal, SignalKind};
use tokio::sync::oneshot;
use tokio::task::JoinSet;
use tokio::time::{timeout, Duration};

/// Channels we surface. Matched against the hwmon OEM string in this order;
/// the socket ID must follow the phrase directly, so "Grace Power Socket 1"
/// stays `grace` even when the string also mentions CPU power elsewhere.
const OEM_KINDS: [(&str, &str); 3] = [
    ("cpu power socket ", "cpu"),
    ("grace power socket ", "grace"),
    ("sysio power socket ", "sysio"),
];

const READ_TIMEOUT: Duration = Duration::from_secs(5);
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);
const ACCEPT_BACKOFF_MIN: Duration = Duration::from_millis(10);
const ACCEPT_BACKOFF_MAX: Duration = Duration::from_secs(1);
/// A wedged sysfs read must not hold a scrape open longer than the client waits.
const COLLECT_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_REQUEST_BYTES: usize = 8192;
/// Backpressure, not a happy-path size: a scrape target sees ~1 request/s.
const MAX_CONNECTIONS: usize = 64;

#[derive(Parser)]
#[command(
    version,
    about = "Grace CPU power exporter for Prometheus / AIPerf server-metrics"
)]
struct Args {
    /// TCP port to listen on.
    #[arg(long, default_value_t = 9401)]
    port: u16,

    /// Address to bind.
    #[arg(long, default_value = "0.0.0.0")]
    bind: IpAddr,

    /// Root of hwmon sysfs (override for testing).
    #[arg(long, default_value = "/sys/class/hwmon")]
    hwmon_root: PathBuf,
}

struct Sensor {
    path: PathBuf,
    /// `<hwmon node>/<channel>`: the one label guaranteed distinct per rail,
    /// so two rails sharing an OEM string are still two Prometheus series.
    id: String,
    oem_info: String,
    kind: &'static str,
    socket: String,
    /// Escaped `sensor=..,type=..,socket=..,oem_info=..` rendered once at discovery.
    labels: String,
}

/// One `power*_average` file before alias resolution picks a winner.
struct Candidate {
    path: PathBuf,
    node: String,
    chan: String,
    oem_info: Option<String>,
}

fn read_text(p: &Path) -> Option<String> {
    fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn classify_oem(oem: &str) -> (&'static str, String) {
    let lower = oem.to_lowercase();
    for (needle, kind) in OEM_KINDS {
        let Some((_, rest)) = lower.split_once(needle) else {
            continue;
        };
        let socket: String = rest.chars().take_while(char::is_ascii_digit).collect();
        if !socket.is_empty() {
            return (kind, socket);
        }
    }
    ("other", String::new())
}

fn escape_label(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Channel `*_average` files under one hwmon directory.
///
/// A missing `device/` alias is normal. Any other I/O error means rails we
/// cannot see, so it is reported rather than silently shortening the list.
fn average_paths(dir: &Path) -> Vec<PathBuf> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            tracing::warn!(error = %e, dir = %dir.display(), "hwmon directory unreadable; its rails are not exported");
            return Vec::new();
        }
    };

    let mut paths = Vec::new();
    for entry in entries {
        match entry {
            Ok(entry) => {
                let path = entry.path();
                let is_average = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("power") && n.ends_with("_average"));
                if is_average && path.is_file() {
                    paths.push(path);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, dir = %dir.display(), "hwmon entry unreadable; a rail may be missing")
            }
        }
    }
    paths.sort();
    paths
}

fn discover_sensors(hwmon_root: &Path) -> Result<Vec<Sensor>> {
    let entries =
        fs::read_dir(hwmon_root).with_context(|| format!("list {}", hwmon_root.display()))?;
    let mut dirs = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| format!("list {}", hwmon_root.display()))?;
        dirs.push(entry.path());
    }
    dirs.sort();

    // `hwmonN/` and `hwmonN/device/` alias the same files, so dedup on the
    // resolved path; distinct channels are never collapsed. The alias that
    // carries the channel's metadata wins regardless of scan order.
    let mut by_identity: BTreeMap<PathBuf, Candidate> = BTreeMap::new();

    for hwmon_dir in dirs {
        let Some(node) = hwmon_dir.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !node.starts_with("hwmon") {
            continue;
        }
        if read_text(&hwmon_dir.join("name")).as_deref() != Some("power_meter") {
            continue;
        }
        let node = node.to_owned();

        for root in [hwmon_dir.clone(), hwmon_dir.join("device")] {
            for path in average_paths(&root) {
                let Some(chan) = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_suffix("_average"))
                    .map(str::to_owned)
                else {
                    continue;
                };
                let oem_info = read_text(&root.join(format!("{chan}_oem_info")))
                    .or_else(|| read_text(&root.join(format!("{chan}_label"))));
                let identity = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                let candidate = Candidate {
                    path,
                    node: node.clone(),
                    chan,
                    oem_info,
                };
                match by_identity.entry(identity) {
                    Entry::Vacant(slot) => {
                        slot.insert(candidate);
                    }
                    Entry::Occupied(mut slot) => {
                        if slot.get().oem_info.is_none() && candidate.oem_info.is_some() {
                            slot.insert(candidate);
                        }
                    }
                }
            }
        }
    }

    let mut candidates: Vec<Candidate> = by_identity.into_values().collect();
    candidates.sort_by(|a, b| (&a.node, &a.chan).cmp(&(&b.node, &b.chan)));

    Ok(candidates
        .into_iter()
        .map(|c| {
            let id = format!("{}/{}", c.node, c.chan);
            let oem_info = c.oem_info.unwrap_or(c.chan);
            let (kind, socket) = classify_oem(&oem_info);
            let labels = format!(
                "sensor=\"{}\",type=\"{kind}\",socket=\"{socket}\",oem_info=\"{}\"",
                escape_label(&id),
                escape_label(&oem_info),
            );
            Sensor {
                path: c.path,
                id,
                oem_info,
                kind,
                socket,
                labels,
            }
        })
        .collect())
}

fn read_watts(path: &Path) -> Option<f64> {
    let raw = read_text(path)?;
    let microwatts: f64 = raw.parse().ok()?;
    let watts = microwatts / 1_000_000.0;
    if watts.is_finite() && watts >= 0.0 {
        Some(watts)
    } else {
        None
    }
}

fn build_metrics(sensors: &[Sensor]) -> String {
    let mut out = String::from(
        "# HELP cpu_power_acpi_watts Grace CPU power rail reading from ACPI hwmon (W).\n\
         # TYPE cpu_power_acpi_watts gauge\n",
    );
    for s in sensors {
        match read_watts(&s.path) {
            Some(watts) => {
                let _ = writeln!(
                    out,
                    "cpu_power_acpi_watts{{{},source=\"acpi\"}} {watts:.6}",
                    s.labels
                );
            }
            // Every scrape re-reads, so a persistently bad channel would warn
            // forever; the gap in the series is the signal a consumer acts on.
            None => {
                tracing::debug!(path = %s.path.display(), "unreadable channel; skipping sample")
            }
        }
    }
    out
}

/// Reads until the end of the request headers, so a request split across
/// segments is not mistaken for a request for `/`.
///
/// The timeout bounds the whole header, not each read: a client dribbling one
/// byte at a time must not hold a connection slot open indefinitely.
async fn read_request(stream: &mut TcpStream) -> Option<String> {
    let read_headers = async {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 1024];
        loop {
            let n = stream.read(&mut chunk).await.ok()?;
            if n == 0 {
                return None;
            }
            buf.extend_from_slice(&chunk[..n]);
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                return String::from_utf8(buf).ok();
            }
            if buf.len() > MAX_REQUEST_BYTES {
                return None;
            }
        }
    };
    timeout(READ_TIMEOUT, read_headers).await.ok()?
}

/// A pending scrape: the collection thread renders the body and replies here.
type CollectRequest = oneshot::Sender<String>;

/// Owns every blocking hwmon read on one dedicated OS thread.
///
/// sysfs reads can stall indefinitely on a wedged firmware path. Doing them on
/// a Tokio worker would take that thread out of the runtime and, with enough
/// concurrent scrapes, stop the accept loop and the signal handlers from ever
/// being polled. Handing them to a thread of our own keeps every async task
/// cancellable and shutdown bounded: the thread is never joined, so a stuck
/// read delays nothing at exit.
struct Collector {
    requests: SyncSender<CollectRequest>,
}

impl Collector {
    fn spawn(sensors: &'static [Sensor]) -> Result<&'static Collector> {
        // Bounded by the connection limit, so a backlog is refused rather than queued.
        let (requests, inbox) = mpsc::sync_channel::<CollectRequest>(MAX_CONNECTIONS);
        thread::Builder::new()
            .name("hwmon-collector".to_owned())
            .spawn(move || {
                for reply in inbox {
                    // The receiver is gone when the client timed out; render anyway
                    // is wasteful, so check first.
                    if reply.is_closed() {
                        continue;
                    }
                    let _ = reply.send(build_metrics(sensors));
                }
            })
            .context("spawn hwmon collector thread")?;
        Ok(Box::leak(Box::new(Collector { requests })))
    }

    /// `None` when the collector is saturated or wedged past `COLLECT_TIMEOUT`.
    async fn snapshot(&self) -> Option<String> {
        let (reply, answer) = oneshot::channel();
        self.requests.try_send(reply).ok()?;
        timeout(COLLECT_TIMEOUT, answer).await.ok()?.ok()
    }
}

async fn handle_connection(mut stream: TcpStream, collector: &'static Collector) {
    let Some(req) = read_request(&mut stream).await else {
        return;
    };
    let mut request_line = req.split_whitespace();
    let method = request_line.next().unwrap_or("");
    // `/metrics?collect[]=x` addresses the same endpoint; the query is not part of it.
    let path = request_line
        .next()
        .unwrap_or("")
        .split('?')
        .next()
        .unwrap_or("");

    let mut extra_headers = "";
    let (status, content_type, body) = match (method, path) {
        // /health deliberately touches no sensor: it must still answer while
        // the collection thread is stuck on a firmware read.
        ("GET" | "HEAD", "/health") => ("200 OK", "text/plain", "ok\n".to_owned()),
        ("GET" | "HEAD", "/metrics") => match collector.snapshot().await {
            Some(body) => ("200 OK", "text/plain; version=0.0.4; charset=utf-8", body),
            None => (
                "503 Service Unavailable",
                "text/plain",
                "sensor collection unavailable\n".to_owned(),
            ),
        },
        ("GET" | "HEAD", _) => ("404 Not Found", "text/plain", "Not Found\n".to_owned()),
        _ => {
            extra_headers = "Allow: GET, HEAD\r\n";
            (
                "405 Method Not Allowed",
                "text/plain",
                "Method Not Allowed\n".to_owned(),
            )
        }
    };

    // HEAD carries exactly the headers its GET would, and no body.
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{extra_headers}Connection: close\r\n\r\n{}",
        body.len(),
        if method == "HEAD" { "" } else { body.as_str() },
    );
    match timeout(WRITE_TIMEOUT, stream.write_all(response.as_bytes())).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::debug!(error = %e, "response write failed"),
        Err(_) => tracing::debug!("response write timed out"),
    }
}

/// Accepts one connection, backing off while the listener keeps erroring.
///
/// `select!` drops this future on a shutdown signal, so a backoff sleep can
/// never delay shutdown.
async fn accept(listener: &TcpListener, backoff: &mut Duration) -> TcpStream {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                *backoff = Duration::ZERO;
                return stream;
            }
            Err(e) => {
                *backoff = (*backoff * 2).clamp(ACCEPT_BACKOFF_MIN, ACCEPT_BACKOFF_MAX);
                tracing::warn!(error = %e, backoff_ms = backoff.as_millis(), "accept failed");
                tokio::time::sleep(*backoff).await;
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Without the `env-filter` feature this honours RUST_LOG via `Targets` and
    // defaults to INFO. With it, the default directive would be ERROR, which
    // silently drops every line below unless the caller sets RUST_LOG.
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Discovered once: hwmon topology is fixed for the process lifetime.
    let sensors = discover_sensors(&args.hwmon_root)?;
    if sensors.is_empty() {
        anyhow::bail!(
            "no ACPI power_meter hwmon sensors found under {}",
            args.hwmon_root.display()
        );
    }

    tracing::info!(count = sensors.len(), "discovered sensors");
    for s in &sensors {
        tracing::debug!(sensor = %s.id, kind = s.kind, socket = %s.socket, oem_info = %s.oem_info, path = %s.path.display(), "sensor");
    }
    let sensors: &'static [Sensor] = Vec::leak(sensors);
    let collector = Collector::spawn(sensors)?;

    let addr = SocketAddr::new(args.bind, args.port);
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    // Consumers (AIPerf discovers its endpoints once at startup) treat this
    // line as readiness, so it is emitted only after the port is accepting.
    tracing::info!(%addr, "listening");

    let mut sigterm = unix_signal(SignalKind::terminate())?;
    let mut conns = JoinSet::new();
    let mut backoff = Duration::ZERO;
    loop {
        let at_capacity = conns.len() >= MAX_CONNECTIONS;
        tokio::select! {
            // Reaping stays inside the select! so a signal is honoured while
            // the exporter is sitting at its connection limit.
            Some(_) = conns.join_next(), if at_capacity => {}
            stream = accept(&listener, &mut backoff), if !at_capacity => {
                conns.spawn(handle_connection(stream, collector));
            }
            _ = signal::ctrl_c() => {
                tracing::info!("received SIGINT, shutting down");
                break;
            }
            _ = sigterm.recv() => {
                tracing::info!("received SIGTERM, shutting down");
                break;
            }
        }
    }

    // Let in-flight scrapes finish rather than truncating a response mid-write.
    if timeout(DRAIN_TIMEOUT, async {
        while conns.join_next().await.is_some() {}
    })
    .await
    .is_err()
    {
        // A handler blocked in a sysfs read reaches no await point, so dropping
        // its task would not bound anything. Exiting does.
        tracing::warn!("timed out draining in-flight connections; exiting");
        process::exit(0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_hwmon(root: &Path, node: &str, sensors: &[(&str, Option<&str>, &str)]) -> PathBuf {
        let hwmon = root.join(node);
        fs::create_dir_all(&hwmon).unwrap();
        fs::write(hwmon.join("name"), "power_meter").unwrap();
        for (chan, oem, microwatts) in sensors {
            fs::write(hwmon.join(format!("{chan}_average")), microwatts).unwrap();
            if let Some(oem) = oem {
                fs::write(hwmon.join(format!("{chan}_oem_info")), oem).unwrap();
            }
        }
        hwmon
    }

    #[test]
    fn discover_sensors_finds_power_meter_channels() {
        let dir = TempDir::new().unwrap();
        write_hwmon(
            dir.path(),
            "hwmon0",
            &[
                ("power1", Some("CPU Power Socket 0"), "150000000"),
                ("power2", Some("Grace Power Socket 1"), "80000000"),
            ],
        );
        let sensors = discover_sensors(dir.path()).unwrap();
        assert_eq!(sensors.len(), 2);
        assert_eq!((sensors[0].kind, sensors[0].socket.as_str()), ("cpu", "0"));
        assert_eq!(
            (sensors[1].kind, sensors[1].socket.as_str()),
            ("grace", "1")
        );
    }

    #[test]
    fn alias_dedup_keeps_the_side_that_carries_the_metadata() {
        let dir = TempDir::new().unwrap();
        let hwmon = write_hwmon(dir.path(), "hwmon0", &[("power1", None, "100000000")]);
        // Real Grace nodes publish the reading on the hwmon node and the OEM
        // string only under device/. The bare hwmon side is scanned first and
        // must not win, or every rail would be labelled "power1".
        let device = hwmon.join("device");
        fs::create_dir_all(&device).unwrap();
        std::os::unix::fs::symlink(hwmon.join("power1_average"), device.join("power1_average"))
            .unwrap();
        fs::write(device.join("power1_oem_info"), "CPU Power Socket 0").unwrap();

        let sensors = discover_sensors(dir.path()).unwrap();
        assert_eq!(sensors.len(), 1, "an alias is one sensor, not two");
        assert_eq!(sensors[0].oem_info, "CPU Power Socket 0");
        assert_eq!(sensors[0].kind, "cpu");
    }

    #[test]
    fn distinct_rails_get_distinct_series_without_usable_metadata() {
        let dir = TempDir::new().unwrap();
        // Duplicated OEM text on one node and no metadata at all on another:
        // both collapse to one Prometheus series unless `sensor` disambiguates.
        write_hwmon(
            dir.path(),
            "hwmon0",
            &[
                ("power1", Some("CPU Power"), "100000000"),
                ("power2", Some("CPU Power"), "110000000"),
            ],
        );
        write_hwmon(dir.path(), "hwmon1", &[("power1", None, "120000000")]);

        let output = build_metrics(&discover_sensors(dir.path()).unwrap());
        let series: Vec<&str> = output
            .lines()
            .filter(|line| line.starts_with("cpu_power_acpi_watts{"))
            .collect();
        assert_eq!(series.len(), 3, "{output}");
        let identities: std::collections::HashSet<&str> = series
            .iter()
            .map(|line| line.split_once(' ').unwrap().0)
            .collect();
        assert_eq!(identities.len(), 3, "duplicate label sets: {output}");
    }

    #[test]
    fn classify_oem_requires_a_socket_id_adjacent_to_the_rail_name() {
        assert_eq!(classify_oem("CPU Power Socket 0"), ("cpu", "0".into()));
        // Rail name wins by adjacency, not by scan order.
        assert_eq!(
            classify_oem("Grace Power Socket 1 CPU Power"),
            ("grace", "1".into())
        );
        // No numeric socket ID -> unclassified, so no unescaped text reaches a label.
        assert_eq!(
            classify_oem("CPU Power Socket \"x"),
            ("other", String::new())
        );
        assert_eq!(classify_oem("Module Socket A"), ("other", String::new()));
    }

    #[test]
    fn build_metrics_formats_and_escapes_prometheus_text() {
        let dir = TempDir::new().unwrap();
        write_hwmon(
            dir.path(),
            "hwmon0",
            &[
                ("power1", Some("CPU Power Socket 0"), "150000000"),
                ("power2", Some("Odd \"rail\""), "not-a-number"),
            ],
        );
        let sensors = discover_sensors(dir.path()).unwrap();
        let output = build_metrics(&sensors);
        assert!(output.contains("# TYPE cpu_power_acpi_watts gauge"));
        assert!(output.contains(
            "cpu_power_acpi_watts{sensor=\"hwmon0/power1\",type=\"cpu\",socket=\"0\",oem_info=\"CPU Power Socket 0\",source=\"acpi\"} 150.000000\n"
        ), "{output}");
        assert!(
            !output.contains("not-a-number"),
            "unparseable channel must be skipped"
        );
        assert!(sensors
            .iter()
            .any(|s| s.labels.contains(r#"oem_info="Odd \"rail\"""#)));
    }

    #[test]
    fn discover_sensors_reports_an_unreadable_root() {
        let dir = TempDir::new().unwrap();
        assert!(discover_sensors(&dir.path().join("absent")).is_err());
    }

    async fn request(addr: SocketAddr, head: &str) -> String {
        let mut client = TcpStream::connect(addr).await.unwrap();
        // Split across writes: a correct reader waits for the header terminator.
        client.write_all(head.as_bytes()).await.unwrap();
        client.write_all(b"Host: localhost\r\n\r\n").await.unwrap();
        let mut resp = String::new();
        client.read_to_string(&mut resp).await.unwrap();
        resp
    }

    #[tokio::test]
    async fn serves_metrics_health_and_rejects_unsupported_methods() {
        let dir = TempDir::new().unwrap();
        write_hwmon(
            dir.path(),
            "hwmon0",
            &[("power1", Some("CPU Power Socket 0"), "150000000")],
        );
        let sensors: &'static [Sensor] = Vec::leak(discover_sensors(dir.path()).unwrap());
        let addr = serve(Collector::spawn(sensors).expect("spawn collector")).await;

        for (head, expected) in [
            ("GET /health HTTP/1.1\r\n", "200 OK"),
            // A query string addresses the same endpoint.
            ("GET /metrics?collect%5B%5D=x HTTP/1.1\r\n", "200 OK"),
            ("GET /nope HTTP/1.1\r\n", "404 Not Found"),
            ("POST /metrics HTTP/1.1\r\n", "405 Method Not Allowed"),
        ] {
            let resp = request(addr, head).await;
            assert!(
                resp.starts_with(&format!("HTTP/1.1 {expected}")),
                "{head}: {resp}"
            );
        }

        let get = request(addr, "GET /metrics HTTP/1.1\r\n").await;
        let head = request(addr, "HEAD /metrics HTTP/1.1\r\n").await;
        let body = "cpu_power_acpi_watts{sensor=\"hwmon0/power1\"";
        assert!(get.contains(body), "{get}");
        assert!(!head.contains(body), "HEAD must carry no body: {head}");
        // Same Content-Length as the GET it mirrors.
        let length = |resp: &str| {
            resp.lines()
                .find(|l| l.starts_with("Content-Length:"))
                .unwrap()
                .to_owned()
        };
        assert_eq!(length(&get), length(&head));

        let rejected = request(addr, "POST /metrics HTTP/1.1\r\n").await;
        assert!(rejected.contains("Allow: GET, HEAD"), "{rejected}");
    }

    /// Accepts serially, so a handler that blocked its thread would stall the next request.
    async fn serve(collector: &'static Collector) -> SocketAddr {
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                handle_connection(stream, collector).await;
            }
        });
        addr
    }

    #[tokio::test]
    async fn a_wedged_collector_degrades_metrics_without_blocking_the_runtime() {
        // A collector whose thread never answers stands in for a sysfs read
        // stuck in the kernel: /metrics must give up, and /health must not care.
        let (requests, inbox) = mpsc::sync_channel::<CollectRequest>(MAX_CONNECTIONS);
        let stuck = thread::spawn(move || {
            let held: Vec<CollectRequest> = inbox.into_iter().collect();
            drop(held);
        });
        let collector: &'static Collector = Box::leak(Box::new(Collector { requests }));
        let addr = serve(collector).await;

        let metrics = tokio::time::timeout(
            COLLECT_TIMEOUT * 2,
            request(addr, "GET /metrics HTTP/1.1\r\n"),
        )
        .await
        .expect("handler must not hold the connection open indefinitely");
        assert!(
            metrics.starts_with("HTTP/1.1 503 Service Unavailable"),
            "{metrics}"
        );

        let health = request(addr, "GET /health HTTP/1.1\r\n").await;
        assert!(health.starts_with("HTTP/1.1 200 OK"), "{health}");
        let _ = stuck;
    }
}
