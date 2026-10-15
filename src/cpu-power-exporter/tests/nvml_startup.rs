// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "linux")]

use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tempfile::TempDir;

struct Exporter(Child);

impl Drop for Exporter {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_until(timeout: Duration, mut ready: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if ready() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn get(addr: SocketAddr, path: &str) -> std::io::Result<String> {
    let timeout = Duration::from_millis(200);
    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    write!(stream, "GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n")?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

#[test]
fn blocked_nvml_setup_keeps_cpu_metrics_available_and_allows_sigterm() {
    let dir = TempDir::new().unwrap();
    let library = dir.path().join("libnvidia-ml.so.1");
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/nvml_stall.c");
    let compiled = Command::new("cc")
        .args(["-shared", "-fPIC", fixture, "-o"])
        .arg(&library)
        .status()
        .unwrap();
    assert!(compiled.success());

    let hwmon = dir.path().join("hwmon/hwmon0");
    fs::create_dir_all(&hwmon).unwrap();
    fs::write(hwmon.join("name"), "power_meter\n").unwrap();
    fs::write(hwmon.join("power1_oem_info"), "Grace Power Socket 0\n").unwrap();
    fs::write(hwmon.join("power1_average"), "88000000\n").unwrap();

    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = reservation.local_addr().unwrap();
    drop(reservation);
    let entered = dir.path().join("nvml-entered");
    let log_path = dir.path().join("exporter.log");
    let log = fs::File::create(&log_path).unwrap();
    let mut exporter = Exporter(
        Command::new(env!("CARGO_BIN_EXE_cpu-power-exporter"))
            .args(["--source", "acpi", "--bind", "127.0.0.1", "--port"])
            .arg(addr.port().to_string())
            .arg("--hwmon-root")
            .arg(dir.path().join("hwmon"))
            .env("LD_LIBRARY_PATH", dir.path())
            .env("NVML_TEST_ENTERED", &entered)
            .env("RUST_LOG", "info")
            .stdout(Stdio::from(log.try_clone().unwrap()))
            .stderr(Stdio::from(log))
            .spawn()
            .unwrap(),
    );
    assert!(wait_until(Duration::from_secs(5), || entered.exists()));
    assert!(
        wait_until(Duration::from_secs(3), || get(addr, "/metrics").is_ok_and(
            |body| body.contains("} 88.000000\n") && !body.contains("cpu_power_nvml_watts")
        )),
        "CPU metrics unavailable while NVML initialization is blocked"
    );
    let health = get(addr, "/health").unwrap();
    assert!(health.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(health.ends_with("\r\n\r\nok\n"));

    let warning = "NVML module power missing or stale";
    assert!(wait_until(Duration::from_secs(5), || fs::read_to_string(
        &log_path
    )
    .unwrap()
    .contains(warning)));
    std::thread::sleep(Duration::from_millis(1100));
    assert_eq!(
        fs::read_to_string(&log_path)
            .unwrap()
            .matches(warning)
            .count(),
        1
    );

    assert!(Command::new("kill")
        .args(["-TERM", &exporter.0.id().to_string()])
        .status()
        .unwrap()
        .success());
    assert!(wait_until(Duration::from_secs(2), || exporter
        .0
        .try_wait()
        .unwrap()
        .is_some_and(|status| status.success())));
    assert!(fs::read_to_string(log_path)
        .unwrap()
        .contains("received SIGTERM, shutting down"));
}
