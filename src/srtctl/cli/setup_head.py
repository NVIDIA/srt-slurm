# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Head node infrastructure setup.

This script is called by the orchestrator to start NATS and etcd on the head node.
It runs inside the container and starts the infrastructure services.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Network configurations
ETCD_CLIENT_PORT = 2379
ETCD_PEER_PORT = 2380
NATS_PORT = 4222
ETCD_LISTEN_ADDR = "http://0.0.0.0"

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """Get local IP address using multiple fallback methods.

    Methods tried (in order):
    1. hostname -I (gets first non-loopback IP)
    2. ip route get 8.8.8.8 (finds default source IP)
    3. socket.gethostbyname (fallback)
    """
    import socket
    import subprocess

    def _is_bad_ip(ip: str) -> bool:
        return not ip or ip == "0.0.0.0" or ip.startswith("127.") or ip.startswith("169.254.")

    def _is_private_ip(ip: str) -> bool:
        if ip.startswith("10.") or ip.startswith("192.168."):
            return True
        if ip.startswith("172."):
            try:
                second = int(ip.split(".", 2)[1])
            except Exception:
                return False
            return 16 <= second <= 31
        return False

    def _select_best_ip(candidates: list[str]) -> str | None:
        for ip in candidates:
            if _is_bad_ip(ip):
                continue
            if _is_private_ip(ip):
                return ip
        for ip in candidates:
            if _is_bad_ip(ip):
                continue
            return ip
        return None

    # Method 1: hostname -I
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            ips = [s for s in result.stdout.strip().split() if s]
            if (ip := _select_best_ip(ips)) is not None:
                return ip
    except Exception:
        pass

    # Method 2: ip route get 8.8.8.8
    try:
        result = subprocess.run(
            ["ip", "route", "get", "8.8.8.8"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse "8.8.8.8 via X.X.X.X dev ethX src Y.Y.Y.Y"
            parts = result.stdout.split("src ")
            if len(parts) > 1:
                ip = parts[1].split()[0]
                if not _is_bad_ip(ip):
                    return ip
    except Exception:
        pass

    # Method 3: socket fallback
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if not _is_bad_ip(ip):
            return ip
    except socket.gaierror:
        pass

    # Last resort
    logger.warning("Could not determine local IP, using 127.0.0.1")
    return "127.0.0.1"


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def start_nats(binary_path: str = "/configs/nats-server", max_payload_mb: int | None = None) -> subprocess.Popen:
    """Start NATS server.

    Args:
        binary_path: Path to nats-server binary
        max_payload_mb: Maximum message payload size in MB, or None for NATS default (1MB).
            Set to 24+ for disaggregated serving with long ISL (65K+ tokens).

    Returns:
        Popen object for the NATS process
    """
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"NATS binary not found: {binary_path}")

    # Use /tmp for JetStream storage - avoids "Temporary storage directory" warning
    # and ensures we're using fast local storage
    if os.path.exists("/tmp/nats"):
        shutil.rmtree("/tmp/nats")
    nats_store_dir = "/tmp/nats"
    os.makedirs(nats_store_dir, exist_ok=True)

    if max_payload_mb is not None:
        # Write NATS config with custom max_payload
        nats_config_path = "/tmp/nats.conf"
        max_payload_bytes = max_payload_mb * 1024 * 1024
        with open(nats_config_path, "w") as f:
            f.write(f"max_payload: {max_payload_bytes}\n")
            f.write(f"jetstream {{ store_dir: \"{nats_store_dir}\" }}\n")
        logger.info("Starting NATS server (max_payload: %dMB)...", max_payload_mb)
        cmd = [binary_path, "-c", nats_config_path]
    else:
        logger.info("Starting NATS server...")
        cmd = [binary_path, "-js", "-sd", nats_store_dir]

    proc = subprocess.Popen(
        cmd,
    )

    logger.info("NATS server started (PID: %d)", proc.pid)
    return proc


def start_etcd(
    host_ip: str,
    binary_path: str = "/configs/etcd",
    log_dir: Path | None = None,
) -> subprocess.Popen:
    """Start etcd server.

    Args:
        host_ip: IP address of this node (for peer URLs)
        binary_path: Path to etcd binary
        log_dir: Optional log directory

    Returns:
        Popen object for the etcd process
    """
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"etcd binary not found: {binary_path}")

    logger.info("Starting etcd server...")

    # Use /tmp for etcd data directory - this is typically on fast local storage
    # (often tmpfs on HPC systems). Without this, etcd uses "default.etcd" in CWD
    # which may be on slow network storage, causing Raft consensus timeouts.
    if os.path.exists("/tmp/etcd"):
        shutil.rmtree("/tmp/etcd")
    etcd_data_dir = "/tmp/etcd"
    os.makedirs(etcd_data_dir, exist_ok=True)

    cmd = [
        binary_path,
        "--data-dir",
        etcd_data_dir,
        "--listen-client-urls",
        f"{ETCD_LISTEN_ADDR}:{ETCD_CLIENT_PORT}",
        "--advertise-client-urls",
        f"http://{host_ip}:{ETCD_CLIENT_PORT}",  # Must be reachable IP, not 0.0.0.0
    ]

    # Set up output handling
    stdout = None
    if log_dir:
        etcd_log = log_dir / "etcd.log"
        stdout = open(etcd_log, "w")  # noqa: SIM115, F841 - stays open for subprocess

    proc = subprocess.Popen(
        cmd,
    )

    logger.info("etcd server started (PID: %d)", proc.pid)
    return proc


def wait_for_service(host: str, port: int, name: str, timeout: float = 300.0) -> bool:
    """Wait for a service to become available on a port.

    Args:
        host: Hostname or IP
        port: Port number
        name: Service name for logging
        timeout: Maximum time to wait

    Returns:
        True if service is ready, False if timeout
    """
    import socket

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                logger.info("%s is ready on port %d", name, port)
                return True
        except (TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(0.5)

    logger.error("%s did not become ready on port %d", name, port)
    return False


def main():
    """Main entry point for head node setup."""
    parser = argparse.ArgumentParser(description="Setup head node infrastructure")
    parser.add_argument("--name", type=str, required=True, help="Run name")
    parser.add_argument("--log-dir", type=str, required=True, help="Log directory")
    parser.add_argument(
        "--nats-binary",
        type=str,
        default="/configs/nats-server",
        help="Path to NATS binary",
    )
    parser.add_argument(
        "--etcd-binary",
        type=str,
        default="/configs/etcd",
        help="Path to etcd binary",
    )
    parser.add_argument(
        "--nats-max-payload-mb",
        type=int,
        default=None,
        help="NATS max message payload in MB (default: NATS default 1MB)",
    )

    args = parser.parse_args()

    setup_logging()
    logger.info("Setting up head node infrastructure for: %s", args.name)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get our IP address using multiple fallback methods
    host_ip = get_local_ip()
    logger.info("Host IP: %s", host_ip)

    # Start services
    nats_proc = None
    etcd_proc = None

    try:
        nats_proc = start_nats(args.nats_binary, max_payload_mb=args.nats_max_payload_mb)
        etcd_proc = start_etcd(host_ip, args.etcd_binary, log_dir)

        # Wait for services
        if not wait_for_service("localhost", NATS_PORT, "NATS"):
            logger.error("NATS failed to start")
            sys.exit(1)

        if not wait_for_service("localhost", ETCD_CLIENT_PORT, "etcd"):
            logger.error("etcd failed to start")
            sys.exit(1)

        logger.info("Head node infrastructure is ready")
        logger.info("  NATS: nats://localhost:%d", NATS_PORT)
        logger.info("  etcd: http://localhost:%d", ETCD_CLIENT_PORT)

        # Keep running - wait for either process to exit
        while True:
            if nats_proc and nats_proc.poll() is not None:
                logger.error("NATS exited with code %d", nats_proc.returncode)
                sys.exit(1)
            if etcd_proc and etcd_proc.poll() is not None:
                logger.error("etcd exited with code %d", etcd_proc.returncode)
                sys.exit(1)
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)
    finally:
        # Cleanup
        if nats_proc and nats_proc.poll() is None:
            nats_proc.terminate()
            nats_proc.wait(timeout=5)
        if etcd_proc and etcd_proc.poll() is None:
            etcd_proc.terminate()
            etcd_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
