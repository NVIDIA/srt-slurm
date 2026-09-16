# Status API Specification v1

srtslurm can optionally report job status to one or more HTTP collectors via fire-and-forget POST/PUT requests. `srtctl status-server` is a collector that ships with srtctl; any server implementing the endpoints below works.

## Configuration

In `srtslurm.yaml` or recipe YAML:

```yaml
reporting:
  status:
    endpoint: "http://login-node:8080"
    # Optional: several collectors, each receives every request
    endpoints:
      - "http://login-node:8080"
      - "https://status.example.com"
    # Optional: which environment variable holds the bearer token (default SRTCTL_STATUS_TOKEN)
    token_env: SRTCTL_STATUS_TOKEN
```

If not configured, status reporting is disabled and jobs run normally.

## Running the native collector

```bash
srtctl status-server                                  # loopback only, port 8080
srtctl status-server --host 0.0.0.0                   # reachable from compute nodes
srtctl status-server --port 9000 --db /lustre/shared/srtctl-status.db
```

Jobs and events live in one SQLite file (default `~/.local/state/srtctl/status.db`). It survives restarts and other tools can read it directly. The process logs one line per lifecycle transition, so leaving it in a terminal or under systemd gives a live feed of every job pointed at it. Run it where the cluster can reach it: `srtctl apply` POSTs from the submitting host, and the sweep PUTs from the head node of the allocation.

Behaviors of the native collector on top of the contract:

- A PUT for a job that was never POSTed creates a placeholder row (`job_name` is `job-<id>`), so a sweep whose submit-time POST was lost still lands every later update.
- A repeated POST leaves the row alone and returns its current status.
- An event is appended whenever `(status, stage, message)` differs from the job's last event. Same-status transitions are kept (`frontend / Starting frontend`, then `frontend / Inference endpoint ready`); pure `artifacts` or `metadata` patches emit nothing.
- `status` and `stage` are validated against `srtctl.contract.JobStatus` and `JobStage`; anything else is HTTP 422.
- Bodies over 1 MiB are rejected with 413 before they are read.

## Web UI

`GET /` serves a single-page UI with no external dependencies: a jobs table (filter by text, status and cluster; elapsed time ticks for active jobs), a detail pane per job (cluster, exit code, duration, model, resources, head node, recipe, log dir, logs URL, the event timeline with deltas, and the raw metadata), and a live global event feed that follows `/api/events` with the cursor. Poll interval is selectable (2 s, 5 s, 15 s, paused). Arrow keys move between jobs; clicking a job id in the feed opens it.

The page itself needs no token (it is static and reveals nothing). It sends the read token the viewer pastes once as `Authorization: Bearer` on every API call and keeps it in the browser's `localStorage`. Opening `/#token=<read token>` seeds it and strips the fragment from the URL; fragments are never sent to the server. `HEAD` is answered like `GET` without a body, for uptime checkers.

### Hosting the page elsewhere

The same `index.html` can be served by any static web server (a Caddy on a corporate network, `python -m http.server`) or opened from a file, and pointed at a collector on another host: set the API base in the header field or open the page with `#api=https://collector.example.com` (also remembered in `localStorage`). Browsers then need the collector's permission for that origin, which is off by default:

```bash
srtctl status-server --host 0.0.0.0 --cors-origin https://zhongshan.example      # repeatable
srtctl status-server --host 0.0.0.0 --cors-origin '*'                            # any origin, including a page opened from a file
```

With a matching `Origin`, GET and HEAD responses (errors included, so the page can show a 401) carry `Access-Control-Allow-Origin`, and the `OPTIONS` preflight is answered before auth with `Access-Control-Allow-Headers: Authorization` and `Access-Control-Allow-Methods: GET, HEAD, OPTIONS`. Writes are never offered cross-origin. This is safe to enable because the API uses no cookies and a token stored by one origin's `localStorage` cannot be read by another; a page from an origin that is not listed simply cannot call the API from the browser, and every call still needs the token.

## Authentication

Tokens are bearer tokens read from the environment on both sides. Nothing token-shaped ever goes into a recipe or `srtslurm.yaml`: the resolved config is written to the lockfile and copied into the log directory that `reporting.s3` uploads.

| Side | Variable | Effect |
|------|----------|--------|
| Server | `SRTCTL_STATUS_TOKEN` (`--token-env`) | Write token. Required for POST, PUT and DELETE; also grants GET |
| Server | `SRTCTL_STATUS_READ_TOKEN` (`--read-token-env`) | Optional read-only token for GET routes (dashboards, humans) |
| Reporter | `SRTCTL_STATUS_TOKEN` (`reporting.status.token_env` renames it) | Sent as `Authorization: Bearer` on every POST and PUT |

Rules:

- `GET /api/health` never needs a token and returns only `{"status": "ok"}`. `GET /` and `/index.html` (the UI) are static and open too.
- Authentication runs before body parsing and routing, so an unauthenticated caller gets 401 and learns nothing else: not whether a job exists, not whether the body parsed.
- Missing or wrong token: 401 with `WWW-Authenticate: Bearer`. Read token on a write route: 403. Tokens are compared in constant time.
- With no write token the server is open. That is only allowed on loopback, or with `--allow-unauthenticated` for a network that is trusted end to end (a cluster login node reachable only from its compute nodes). A read token without a write token is a startup error.
- The reporter never follows redirects (`allow_redirects=False`). A 3xx means the endpoint is behind a login page or proxy and is logged at WARNING as a failure; so are 401 and 403. Network errors stay at DEBUG because reporting is fire-and-forget.

The reporter reads the token from the shell that runs `srtctl apply`; SLURM's default `--export=ALL` carries it to the orchestrator on the head node, which runs outside the container.

```bash
# collector host
export SRTCTL_STATUS_TOKEN=$(openssl rand -hex 32)
export SRTCTL_STATUS_READ_TOKEN=$(openssl rand -hex 32)
srtctl status-server --host 0.0.0.0 --port 8080

# login node, before srtctl apply (same write token)
export SRTCTL_STATUS_TOKEN=<write token>

# reading
curl -H "Authorization: Bearer $SRTCTL_STATUS_READ_TOKEN" https://collector.example.com/api/events
```

Exposing the collector on the public internet also needs TLS in front of it (a reverse proxy, or a platform that terminates TLS) and, where possible, a source allow-list for the cluster's egress addresses on the write path. The collector itself never speaks TLS.

## Endpoints

### POST /api/jobs

Create a job record. Called at submission time.

**Request:**
```json
{
  "job_id": "12345",
  "job_name": "benchmark-run",
  "cluster": "gpu-cluster-01",
  "recipe": "configs/benchmark.yaml",
  "submitted_at": "2025-01-26T10:30:00Z",
  "metadata": {
    "tags": ["pipeline:98765", "suite:kv-router-comparison"]
  }
}
```

**Response:** `201 Created`
```json
{
  "job_id": "12345",
  "status": "submitted"
}
```

### PUT /api/jobs/{job_id}

Update job status. Called during execution and at completion.

**Request (during execution):**
```json
{
  "status": "workers",
  "stage": "workers",
  "message": "Starting workers",
  "updated_at": "2025-01-26T10:35:00Z"
}
```

**Request (at completion):**
```json
{
  "status": "completed",
  "stage": "cleanup",
  "exit_code": 0,
  "logs_url": "s3://bucket/outputs/12345/",
  "updated_at": "2025-01-26T11:02:00Z",
  "completed_at": "2025-01-26T11:02:00Z"
}
```

All fields except `status` and `updated_at` are optional.

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | **Required.** New job status |
| `updated_at` | string | **Required.** ISO 8601 timestamp of this update |
| `stage` | string | Current execution stage |
| `message` | string | Human-readable status message |
| `started_at` | string | Job start timestamp |
| `completed_at` | string | Job completion timestamp |
| `exit_code` | int | Process exit code |
| `logs_url` | string | URL where logs were uploaded (S3 today) |
| `benchmark_results` | object | Parsed benchmark metrics (replaces) |
| `artifacts` | object | Collector-side artifact pointers (merged with existing) |
| `metadata` | object | Additional metadata (merged with existing) |

**Response:** `200 OK`
```json
{
  "job_id": "12345",
  "status": "completed"
}
```

### GET /api/jobs/{job_id}

Full job record with its ordered event history.

```json
{
  "job_id": "12345",
  "job_name": "benchmark-run",
  "status": "completed",
  "stage": "cleanup",
  "cluster": "gpu-cluster-01",
  "recipe": "configs/benchmark.yaml",
  "message": "Benchmark completed successfully",
  "submitted_at": "2025-01-26T10:30:00Z",
  "started_at": "2025-01-26T10:33:00Z",
  "completed_at": "2025-01-26T11:02:00Z",
  "updated_at": "2025-01-26T11:02:00Z",
  "exit_code": 0,
  "logs_url": "s3://bucket/outputs/12345/",
  "benchmark_results": null,
  "artifacts": null,
  "metadata": {"tags": ["suite:kv-router-comparison"], "head_node": "node-01", "log_dir": "/lustre/outputs/12345/logs/12345_1P_4D"},
  "events": [
    {"id": 1, "job_id": "12345", "status": "submitted", "stage": null, "message": null, "created_at": "2025-01-26T10:30:00Z"},
    {"id": 2, "job_id": "12345", "status": "starting", "stage": "starting", "message": "Job started on node-01", "created_at": "2025-01-26T10:33:00Z"}
  ]
}
```

### GET /api/jobs

List jobs, newest first, with pagination and filters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `per_page` | int | 50 | Results per page (max 100) |
| `status` | string | - | Filter by status |
| `cluster` | string | - | Filter by cluster |

Response: `{"jobs": [JobSummary, ...], "total": N, "page": 1, "per_page": 50}`.

### GET /api/jobs/{job_id}/events

Incremental event feed for one job. Events carry a monotonically increasing `id`; pass the last one you saw as `after` to resume.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `after` | int | 0 | Return events with `id > after` |
| `limit` | int | 100 | Max events per response (max 1000) |

```json
{
  "job_id": "12345",
  "events": [
    {"id": 7, "job_id": "12345", "status": "benchmark", "stage": "benchmark", "message": "Running benchmark", "created_at": "2025-01-26T10:40:00Z"}
  ],
  "next_cursor": 7
}
```

`next_cursor` is the last `id` returned, or the `after` you passed when nothing new arrived (null on an empty feed). A poll loop is `after = next_cursor` between requests.

### GET /api/events

Same as above across every job, with an optional `job_id` filter. This is the feed for dashboards and agents that want to react to job transitions without polling each job.

### DELETE /api/jobs/{job_id}

Remove a job and its events. `200 {"deleted": true, "job_id": ...}` or `404`.

### GET /api/health

`200 {"status": "ok"}`.

## Status Values

```text
submitted -> starting -> workers -> frontend -> benchmark -> completed | failed
```

Status reflects which stage is currently executing, not readiness.

## Started metadata

The first PUT of a run (`StatusReporter.report_started`) carries `metadata` with the model path and precision, the resource shape (`gpu_type`, worker counts, CPU allocation), the benchmark type, `backend_type`, `frontend_type`, `head_node`, and `log_dir`, the run's log directory on the cluster filesystem. A collector on the same filesystem can open the logs from `log_dir` straight away; `logs_url` is only set later, and only when `reporting.s3` uploads the directory.

## Contract Models

The canonical Pydantic models live in `srtctl.contract`:

```python
from srtctl.contract import (
    JobStatus,              # Status enum
    JobStage,               # Stage enum
    JobCreatePayload,       # POST request body
    JobUpdatePayload,       # PUT request body
    JobResponse,            # POST/PUT response
    JobSummary,             # List endpoint item
    JobDetail,              # GET endpoint response
    JobListResponse,        # List endpoint wrapper
    JobEventRecord,         # One event in either feed
    JobEventListResponse,   # Per-job events response
    EventFeedResponse,      # Global events response
)
```

## Behavior

- All requests have a 5-second timeout
- Redirects are never followed; a 3xx, 401 or 403 is logged at WARNING and counts as a failure
- Other failures are logged at DEBUG level and ignored
- Job execution is never blocked by status reporting failures
