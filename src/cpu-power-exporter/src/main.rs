// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal Prometheus exporter for Grace CPU power via ACPI hwmon.
//!
//! Reads /sys/class/hwmon/hwmon*/power*_average, exposes:
//!   cpu_power_acpi_watts{type, socket, oem_info, source}
//!
//! Endpoints:
//!   GET /metrics  — Prometheus text format
//!   GET /health   — "ok\n"

use anyhow::{Context, Result};
use clap::Parser;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::signal;
use tokio::signal::unix::{signal as unix_signal, SignalKind};
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
    oem_info: String,
    kind: &'static str,
    socket: String,
    /// Escaped `type=..,socket=..,oem_info=..` rendered once at discovery.
    labels: String,
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

fn discover_sensors(hwmon_root: &Path) -> Vec<Sensor> {
    let mut sensors = Vec::new();
    // hwmon_dir/device/ and hwmon_dir/ alias the same files; dedup on the
    // resolved path so genuinely distinct channels are never collapsed.
    let mut seen = HashSet::<PathBuf>::new();

    let Ok(entries) = fs::read_dir(hwmon_root) else {
        return sensors;
    };
    let mut dirs: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    dirs.sort();

    for hwmon_dir in dirs {
        if !hwmon_dir
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with("hwmon"))
            .unwrap_or(false)
        {
            continue;
        }
        if read_text(&hwmon_dir.join("name")).as_deref() != Some("power_meter") {
            continue;
        }
        let device_root = hwmon_dir.join("device");
        for root in [device_root, hwmon_dir] {
            let Ok(files) = fs::read_dir(&root) else {
                continue;
            };
            let mut avg_paths: Vec<_> = files
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("power") && n.ends_with("_average"))
                        .unwrap_or(false)
                })
                .collect();
            avg_paths.sort();

            for avg_path in avg_paths {
                if !avg_path.is_file() {
                    continue;
                }
                let identity = fs::canonicalize(&avg_path).unwrap_or_else(|_| avg_path.clone());
                if !seen.insert(identity) {
                    continue;
                }

                let chan = avg_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_suffix("_average"))
                    .unwrap_or_default()
                    .to_owned();

                let oem_info = read_text(&root.join(format!("{chan}_oem_info")))
                    .or_else(|| read_text(&root.join(format!("{chan}_label"))))
                    .unwrap_or(chan);

                let (kind, socket) = classify_oem(&oem_info);
                let labels = format!(
                    "type=\"{}\",socket=\"{}\",oem_info=\"{}\"",
                    kind,
                    escape_label(&socket),
                    escape_label(&oem_info),
                );
                sensors.push(Sensor {
                    path: avg_path,
                    oem_info,
                    kind,
                    socket,
                    labels,
                });
            }
        }
    }
    sensors
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
async fn read_request(stream: &mut TcpStream) -> Option<String> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let n = timeout(READ_TIMEOUT, stream.read(&mut chunk))
            .await
            .ok()?
            .ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if buf.len() > MAX_REQUEST_BYTES {
            return None;
        }
    }
    String::from_utf8(buf).ok()
}

async fn handle_connection(mut stream: TcpStream, sensors: &'static [Sensor]) {
    let Some(req) = read_request(&mut stream).await else {
        return;
    };
    let path = req.split_whitespace().nth(1).unwrap_or("");

    let (status, content_type, body): (&str, &str, String) = match path {
        "/health" => ("200 OK", "text/plain", "ok\n".into()),
        "/metrics" => match tokio::task::spawn_blocking(move || build_metrics(sensors)).await {
            Ok(body) => ("200 OK", "text/plain; version=0.0.4; charset=utf-8", body),
            // An empty 200 would look to Prometheus like a host with no power
            // rails rather than a broken exporter.
            Err(e) => {
                tracing::error!(error = %e, "metrics collection failed");
                (
                    "500 Internal Server Error",
                    "text/plain",
                    "collection failed\n".into(),
                )
            }
        },
        _ => ("404 Not Found", "text/plain", "Not Found\n".into()),
    };

    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len(),
    );
    if let Err(e) = timeout(WRITE_TIMEOUT, stream.write_all(response.as_bytes())).await {
        tracing::trace!(error = %e, "response write timed out");
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
    let sensors = discover_sensors(&args.hwmon_root);
    if sensors.is_empty() {
        anyhow::bail!(
            "no ACPI power_meter hwmon sensors found under {}",
            args.hwmon_root.display()
        );
    }

    tracing::info!(count = sensors.len(), "discovered sensors");
    for s in &sensors {
        tracing::debug!(kind = s.kind, socket = %s.socket, oem_info = %s.oem_info, path = %s.path.display(), "sensor");
    }
    let sensors: &'static [Sensor] = Vec::leak(sensors);

    let addr = SocketAddr::new(args.bind, args.port);
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    tracing::info!(%addr, "listening");

    let mut sigterm = unix_signal(SignalKind::terminate())?;
    let mut conns = JoinSet::new();
    loop {
        if conns.len() >= MAX_CONNECTIONS {
            conns.join_next().await;
        }
        tokio::select! {
            res = listener.accept() => match res {
                Ok((stream, _)) => { conns.spawn(handle_connection(stream, sensors)); }
                Err(e) => tracing::warn!(error = %e, "accept failed"),
            },
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
        tracing::warn!("timed out draining in-flight connections");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_hwmon(dir: &TempDir, sensors: &[(&str, &str, &str)]) {
        let hwmon = dir.path().join("hwmon0");
        fs::create_dir_all(&hwmon).unwrap();
        fs::write(hwmon.join("name"), "power_meter").unwrap();
        for (chan, oem, microwatts) in sensors {
            fs::write(hwmon.join(format!("{chan}_average")), microwatts).unwrap();
            fs::write(hwmon.join(format!("{chan}_oem_info")), oem).unwrap();
        }
    }

    #[test]
    fn discover_sensors_finds_power_meter_channels() {
        let dir = TempDir::new().unwrap();
        make_hwmon(
            &dir,
            &[
                ("power1", "CPU Power Socket 0", "150000000"),
                ("power2", "Grace Power Socket 1", "80000000"),
            ],
        );
        let sensors = discover_sensors(dir.path());
        assert_eq!(sensors.len(), 2);
        assert_eq!((sensors[0].kind, sensors[0].socket.as_str()), ("cpu", "0"));
        assert_eq!(
            (sensors[1].kind, sensors[1].socket.as_str()),
            ("grace", "1")
        );
    }

    #[test]
    fn discover_sensors_dedups_alias_but_keeps_distinct_channels() {
        let dir = TempDir::new().unwrap();
        let hwmon = dir.path().join("hwmon0");
        let device = hwmon.join("device");
        fs::create_dir_all(&device).unwrap();
        fs::write(hwmon.join("name"), "power_meter").unwrap();
        fs::write(hwmon.join("power1_average"), "100000000").unwrap();
        fs::write(hwmon.join("power1_oem_info"), "CPU Power Socket 0").unwrap();
        // Same channel reached through the device/ alias.
        std::os::unix::fs::symlink(hwmon.join("power1_average"), device.join("power1_average"))
            .unwrap();
        // A distinct channel with no oem_info: falls back to the same labels as
        // any other bare power1, and must still be kept.
        let hwmon1 = dir.path().join("hwmon1");
        fs::create_dir_all(&hwmon1).unwrap();
        fs::write(hwmon1.join("name"), "power_meter").unwrap();
        fs::write(hwmon1.join("power1_average"), "200000000").unwrap();

        let sensors = discover_sensors(dir.path());
        assert_eq!(
            sensors.len(),
            2,
            "alias must dedup, distinct channels must not"
        );
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
        make_hwmon(
            &dir,
            &[
                ("power1", "CPU Power Socket 0", "150000000"),
                ("power2", "Odd \"rail\"", "not-a-number"),
            ],
        );
        let sensors = discover_sensors(dir.path());
        let output = build_metrics(&sensors);
        assert!(output.contains("# TYPE cpu_power_acpi_watts gauge"));
        assert!(output.contains(
            "cpu_power_acpi_watts{type=\"cpu\",socket=\"0\",oem_info=\"CPU Power Socket 0\",source=\"acpi\"} 150.000000\n"
        ));
        assert!(
            !output.contains("not-a-number"),
            "unparseable channel must be skipped"
        );
        assert!(sensors
            .iter()
            .any(|s| s.labels.contains(r#"oem_info="Odd \"rail\"""#)));
    }

    #[tokio::test]
    async fn serves_metrics_health_and_404() {
        let dir = TempDir::new().unwrap();
        make_hwmon(&dir, &[("power1", "CPU Power Socket 0", "150000000")]);
        let sensors: &'static [Sensor] = Vec::leak(discover_sensors(dir.path()));

        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                handle_connection(stream, sensors).await;
            }
        });

        for (path, expected) in [
            ("/health", "200 OK"),
            ("/metrics", "200 OK"),
            ("/nope", "404 Not Found"),
        ] {
            let mut client = TcpStream::connect(addr).await.unwrap();
            // Split across writes: a correct reader waits for the header terminator.
            client
                .write_all(format!("GET {path} HTTP/1.1\r\n").as_bytes())
                .await
                .unwrap();
            client.write_all(b"Host: localhost\r\n\r\n").await.unwrap();
            let mut resp = String::new();
            client.read_to_string(&mut resp).await.unwrap();
            assert!(
                resp.starts_with(&format!("HTTP/1.1 {expected}")),
                "{path}: {resp}"
            );
        }
    }
}
