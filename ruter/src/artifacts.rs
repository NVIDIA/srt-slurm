//! `srt-slurm` run-layout discovery and immutable input manifest generation.

use crate::database::write_database;
use crate::logs::{ParserKind, parse_file};
use crate::tables::{
    load_aiperf_requests, load_request_traces, load_worker_metric_samples, write_events,
    write_metrics, write_request_traces, write_requests,
};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const SCHEMA_VERSION: u32 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Dynamo,
}

impl Arm {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Dynamo => "dynamo",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InputFile {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct ArmManifest {
    pub arm: Arm,
    pub router_log: InputFile,
    pub worker_logs: Vec<InputFile>,
    pub tachometer_parquet: Option<InputFile>,
    pub aiperf_trace: Option<InputFile>,
    pub dynamo_request_traces: Vec<InputFile>,
    pub tachometer_log: Option<InputFile>,
    /// Resolved routing knobs captured by Dynamo's startup CONFIG_DUMP.
    /// These are more trustworthy than launch flags because they include defaults.
    pub router_settings: Option<RouterSettings>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouterSettings {
    pub router_mode: Option<String>,
    pub overlap_score_credit: Option<f64>,
    pub overlap_score_credit_decay: Option<f64>,
    pub prefill_load_scale: Option<f64>,
    pub decode_active_request_weight: Option<f64>,
    pub router_temperature: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub run_root: String,
    pub arms: Vec<ArmManifest>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct InitReport {
    pub analysis_dir: String,
    pub arms: Vec<Arm>,
    pub normalized: NormalizedTables,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedTables {
    pub events: usize,
    pub requests: usize,
    pub metrics: usize,
    pub request_traces: usize,
}

struct DiscoveredArm {
    arm: Arm,
    router_log: PathBuf,
    worker_logs: Vec<PathBuf>,
    tachometer_parquet: Option<PathBuf>,
    aiperf_trace: Option<PathBuf>,
    dynamo_request_traces: Vec<PathBuf>,
    tachometer_log: Option<PathBuf>,
}

/// Discover inputs and write normalized log events without modifying raw inputs.
/// `analysis_dir` must be a separate `.ruter` directory, usually under `run_root`.
pub fn initialize(run_root: &Path, analysis_dir: &Path) -> Result<InitReport> {
    let run_root = run_root
        .canonicalize()
        .with_context(|| format!("resolve run root {}", run_root.display()))?;
    let discovered = discover_arm(&run_root, Arm::Dynamo)?
        .into_iter()
        .collect::<Vec<_>>();
    if discovered.is_empty() {
        bail!(
            "no Dynamo run found below {}; expected logs/router.log",
            run_root.display()
        );
    }

    fs::create_dir_all(analysis_dir)
        .with_context(|| format!("create {}", analysis_dir.display()))?;
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    let mut arms = Vec::new();

    for arm in &discovered {
        for event in parse_file(router_parser(arm.arm), &arm.router_log)? {
            events.push(event);
        }
        for (worker_index, worker_log) in arm.worker_logs.iter().enumerate() {
            for event in parse_file(worker_parser(arm.arm, worker_index as u32), worker_log)? {
                events.push(event);
            }
        }

        if arm.tachometer_parquet.is_none() {
            warnings.push(format!(
                "{}: Tachometer final.parquet was not found",
                arm.arm.name()
            ));
        }
        if arm.aiperf_trace.is_none() {
            warnings.push(format!(
                "{}: AIPerf trace.jsonl was not found",
                arm.arm.name()
            ));
        }
        arms.push(ArmManifest {
            arm: arm.arm,
            router_log: input_file(&run_root, &arm.router_log)?,
            worker_logs: arm
                .worker_logs
                .iter()
                .map(|path| input_file(&run_root, path))
                .collect::<Result<_>>()?,
            tachometer_parquet: arm
                .tachometer_parquet
                .as_deref()
                .map(|path| input_file(&run_root, path))
                .transpose()?,
            aiperf_trace: arm
                .aiperf_trace
                .as_deref()
                .map(|path| input_file(&run_root, path))
                .transpose()?,
            dynamo_request_traces: arm
                .dynamo_request_traces
                .iter()
                .map(|path| input_file(&run_root, path))
                .collect::<Result<_>>()?,
            tachometer_log: arm
                .tachometer_log
                .as_deref()
                .map(|path| input_file(&run_root, path))
                .transpose()?,
            router_settings: router_settings(&arm.router_log)?,
        });
    }
    let event_count = write_events(&analysis_dir.join("events.parquet"), &events)?;
    let request_sources = discovered
        .iter()
        .filter_map(|arm| {
            arm.aiperf_trace
                .as_deref()
                .map(|path| (arm.arm.name(), path))
        })
        .collect::<Vec<_>>();
    let aiperf_requests = load_aiperf_requests(&request_sources)?;
    let request_count = write_requests(&analysis_dir.join("requests.parquet"), &aiperf_requests)?;
    let request_trace_sources = discovered
        .iter()
        .flat_map(|arm| {
            arm.dynamo_request_traces
                .iter()
                .map(|path| (arm.arm.name(), path.as_path()))
        })
        .collect::<Vec<_>>();
    let mut request_traces = load_request_traces(&request_trace_sources)?;
    enrich_request_traces(&mut request_traces, &events);
    let request_trace_count = write_request_traces(
        &analysis_dir.join("request_traces.parquet"),
        &request_traces,
    )?;
    let metric_sources = discovered
        .iter()
        .filter_map(|arm| {
            arm.tachometer_parquet
                .as_deref()
                .map(|path| (arm.arm.name(), path, arm.tachometer_log.as_deref()))
        })
        .collect::<Vec<_>>();
    let metrics = write_metrics(&analysis_dir.join("metrics.parquet"), &metric_sources)?;
    let worker_metric_samples = load_worker_metric_samples(&analysis_dir.join("metrics.parquet"))?;

    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        run_root: run_root.display().to_string(),
        arms,
        warnings,
    };
    let manifest_path = analysis_dir.join("manifest.json");
    let manifest_writer = BufWriter::new(
        File::create(&manifest_path)
            .with_context(|| format!("create {}", manifest_path.display()))?,
    );
    serde_json::to_writer_pretty(manifest_writer, &manifest)?;
    write_database(
        &analysis_dir.join("ruter.db"),
        &manifest,
        &events,
        &aiperf_requests,
        &request_traces,
        &worker_metric_samples,
    )?;
    Ok(InitReport {
        analysis_dir: analysis_dir.display().to_string(),
        arms: manifest.arms.iter().map(|arm| arm.arm).collect(),
        normalized: NormalizedTables {
            events: event_count,
            requests: request_count,
            metrics: metrics,
            request_traces: request_trace_count,
        },
        warnings: manifest.warnings.clone(),
    })
}

fn router_settings(router_log: &Path) -> Result<Option<RouterSettings>> {
    let file = File::open(router_log).with_context(|| format!("open {}", router_log.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line.with_context(|| format!("read {}", router_log.display()))?;
        let Some((_, dump)) = line.split_once("CONFIG_DUMP: ") else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(dump) else {
            continue;
        };
        let Some(config) = value.get("config") else {
            continue;
        };
        return Ok(Some(RouterSettings {
            router_mode: string_setting(config, "router_mode"),
            overlap_score_credit: number_setting(config, "overlap_score_credit"),
            overlap_score_credit_decay: number_setting(config, "overlap_score_credit_decay"),
            prefill_load_scale: number_setting(config, "prefill_load_scale"),
            decode_active_request_weight: number_setting(config, "decode_active_request_weight"),
            router_temperature: number_setting(config, "router_temperature"),
        }));
    }
    Ok(None)
}

fn string_setting(config: &Value, name: &str) -> Option<String> {
    config.get(name)?.as_str().map(ToOwned::to_owned)
}

fn number_setting(config: &Value, name: &str) -> Option<f64> {
    config.get(name)?.as_f64()
}

/// Link Dynamo's context-free trace to the client request ID without assuming
/// session headers. The KV-router debug line is the stable bridge: it carries
/// the internal Dynamo request ID and inbound `x_request_id` together.
fn enrich_request_traces(
    request_traces: &mut [crate::tables::RequestTrace],
    events: &[crate::model::Event],
) {
    let x_request_ids = events
        .iter()
        .filter(|event| {
            event.source == crate::model::LogSource::DynamoRouter
                && event.kind == crate::model::EventKind::RoutingDecision
        })
        .filter_map(|event| {
            let dynamo_request_id = event.fields.get("dynamo_request_id")?;
            let x_request_id = event.fields.get("x_request_id")?;
            Some((dynamo_request_id.clone(), x_request_id.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    for trace in request_traces {
        if trace.x_request_id.is_none() {
            trace.x_request_id = trace
                .request_id
                .as_ref()
                .and_then(|request_id| x_request_ids.get(request_id))
                .cloned();
        }
    }
}

fn discover_arm(run_root: &Path, arm: Arm) -> Result<Option<DiscoveredArm>> {
    let dir = [
        run_root.join("artifacts").join(arm.name()),
        run_root.join(arm.name()),
    ]
    .into_iter()
    .find(|path| path.is_dir())
    .or_else(|| {
        (arm == Arm::Dynamo && run_root.join("logs/router.log").is_file())
            .then(|| run_root.to_path_buf())
    });
    let Some(dir) = dir else { return Ok(None) };
    let router_log = first_file(&dir, |path| {
        path.file_name().is_some_and(|name| name == "router.log")
    })
    .with_context(|| {
        format!(
            "{} arm has no router.log below {}",
            arm.name(),
            dir.display()
        )
    })?;
    let mut worker_logs = WalkDir::new(&dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy();
            (name.starts_with("worker-") && name.ends_with(".log")).then(|| entry.into_path())
        })
        .collect::<Vec<_>>();
    worker_logs.sort();
    if worker_logs.is_empty() {
        bail!(
            "{} arm has no worker-*.log below {}",
            arm.name(),
            dir.display()
        );
    }
    Ok(Some(DiscoveredArm {
        arm,
        router_log,
        worker_logs,
        tachometer_parquet: first_file(&dir, |path| {
            path.file_name().is_some_and(|name| name == "final.parquet")
        }),
        aiperf_trace: first_file(&dir, |path| {
            path.file_name().is_some_and(|name| name == "trace.jsonl")
        }),
        dynamo_request_traces: (arm == Arm::Dynamo)
            .then(|| request_trace_files(&dir))
            .unwrap_or_default(),
        tachometer_log: first_file(&dir, |path| {
            path.file_name()
                .is_some_and(|name| name == "tachometer.log")
        }),
    }))
}

fn first_file<F>(root: &Path, predicate: F) -> Option<PathBuf>
where
    F: Fn(&Path) -> bool,
{
    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .find(|path| predicate(path))
}

/// Dynamo's default `jsonl_gz` writer creates numbered segments such as
/// `dynamo-request-trace.000000.jsonl.gz`. Keep all segments: a long run can
/// rotate after 256 MiB, and dropping an earlier segment silently loses rows.
fn request_trace_files(root: &Path) -> Vec<PathBuf> {
    let mut paths = WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy();
            (name == "dynamo-request-trace.jsonl"
                || name == "dynamo-request-trace.jsonl.gz"
                || (name.starts_with("dynamo-request-trace.") && name.ends_with(".jsonl.gz")))
            .then(|| entry.into_path())
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn router_parser(arm: Arm) -> ParserKind {
    match arm {
        Arm::Dynamo => ParserKind::DynamoRouter,
    }
}

fn worker_parser(arm: Arm, worker_index: u32) -> ParserKind {
    match arm {
        Arm::Dynamo => ParserKind::DynamoWorker { worker_index },
    }
}

fn input_file(root: &Path, path: &Path) -> Result<InputFile> {
    let metadata = path
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?;
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(InputFile {
        path: path
            .strip_prefix(root)
            .unwrap_or(path)
            .display()
            .to_string(),
        sha256: format!("{:x}", hasher.finalize()),
        bytes: metadata.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use std::io::Write;
    use tempfile::tempdir;

    fn write(root: &Path, name: &str, contents: &str) {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn init_discovers_arm_writes_manifest_and_only_analysis_outputs() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        write(
            root,
            "artifacts/dynamo/logs/router.log",
            "2026-08-20T04:00:30.0Z INFO: Selected worker request_id=req-1 worker_id=1 overlap_blocks=0 phase=Aggregated\n",
        );
        write(
            root,
            "artifacts/dynamo/logs/worker-0.log",
            "[2026-08-20 04:00:31] Prefill batch, #new-seq: 1, #new-token: 16\n",
        );
        write(
            root,
            "artifacts/dynamo/artifacts/aiperf/trace.jsonl",
            "{}\n",
        );
        write(
            root,
            "artifacts/dynamo/artifacts/dynamo-request-trace.jsonl",
            r#"{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":2000,"request":{"request_id":"req-1","input_tokens":100,"cached_tokens":80,"kv_hit_rate":0.8}}"#,
        );
        let report = initialize(root, &root.join(".ruter")).unwrap();
        assert_eq!(report.arms.len(), 1);
        assert_eq!(report.normalized.events, 2);
        assert_eq!(report.normalized.request_traces, 1);
        assert!(root.join(".ruter/manifest.json").is_file());
        assert!(root.join(".ruter/events.parquet").is_file());
        assert!(root.join(".ruter/request_traces.parquet").is_file());
        assert!(root.join(".ruter/ruter.db").is_file());
        assert!(root.join("artifacts/dynamo/logs/router.log").is_file());
    }

    #[test]
    fn init_discovers_a_direct_lifecycle_output_as_dynamo() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        write(
            root,
            "logs/router.log",
            "2026-08-20T04:00:30.0Z INFO: Selected worker request_id=req-1 worker_id=1 overlap_blocks=0 phase=Aggregated\n",
        );
        write(
            root,
            "logs/worker-0.log",
            "[2026-08-20 04:00:31] Prefill batch, #new-seq: 1, #new-token: 16\n",
        );
        write(root, "artifacts/aiperf/trace.jsonl", "{}\n");
        write(
            root,
            "artifacts/dynamo-request-trace.jsonl",
            r#"{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":2000,"request":{"request_id":"req-1","kv_hit_rate":0.5}}"#,
        );
        let report = initialize(root, &root.join(".ruter")).unwrap();
        assert_eq!(report.arms, vec![Arm::Dynamo]);
        assert_eq!(report.normalized.request_traces, 1);
    }

    #[test]
    fn init_discovers_default_gzip_request_trace_segments() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        write(root, "logs/router.log", "router started\n");
        write(root, "logs/worker-0.log", "worker started\n");
        let trace = root.join("artifacts/dynamo-request-trace.000000.jsonl.gz");
        fs::create_dir_all(trace.parent().unwrap()).unwrap();
        let file = File::create(&trace).unwrap();
        let mut writer = GzEncoder::new(file, Compression::default());
        writer
            .write_all(
                br#"{"timestamp":0,"event":{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":2000,"request":{"request_id":"req-1","kv_hit_rate":0.5}}}
"#,
            )
            .unwrap();
        writer.finish().unwrap();

        let analysis = root.join(".ruter");
        let report = initialize(root, &analysis).unwrap();
        assert_eq!(report.normalized.request_traces, 1);
        let manifest: serde_json::Value =
            serde_json::from_reader(File::open(analysis.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(
            manifest["arms"][0]["dynamo_request_traces"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn reads_resolved_router_settings_from_config_dump() {
        let temp = tempdir().unwrap();
        let log = temp.path().join("router.log");
        fs::write(
            &log,
            "INFO CONFIG_DUMP: {\"config\":{\"router_mode\":\"kv\",\"overlap_score_credit\":1.0,\"overlap_score_credit_decay\":0.25,\"prefill_load_scale\":2.0,\"decode_active_request_weight\":3.0,\"router_temperature\":0.5}}\n",
        )
        .unwrap();
        let settings = router_settings(&log).unwrap().unwrap();
        assert_eq!(settings.router_mode.as_deref(), Some("kv"));
        assert_eq!(settings.overlap_score_credit, Some(1.0));
        assert_eq!(settings.overlap_score_credit_decay, Some(0.25));
        assert_eq!(settings.prefill_load_scale, Some(2.0));
        assert_eq!(settings.decode_active_request_weight, Some(3.0));
        assert_eq!(settings.router_temperature, Some(0.5));
    }

    #[test]
    fn init_joins_dynamo_trace_to_aiperf_request_id_from_routing_debug() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        write(
            root,
            "logs/router.log",
            "2026-08-20T04:00:30.0Z DEBUG: [ROUTING] Best: worker_1 dp_rank=0 with 1/2 blocks overlap request_id=internal-1 worker_id=1 overlap_blocks=1 total_blocks=2 x_request_id=\"client-1\" request_id=http-1\n",
        );
        write(root, "logs/worker-0.log", "worker started\n");
        write(
            root,
            "artifacts/dynamo-request-trace.jsonl",
            r#"{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":2000,"request":{"request_id":"internal-1","kv_hit_rate":0.5}}"#,
        );
        let analysis = root.join(".ruter");
        initialize(root, &analysis).unwrap();
        let database = rusqlite::Connection::open(analysis.join("ruter.db")).unwrap();
        assert_eq!(
            database
                .query_row("SELECT x_request_id FROM request_traces", [], |row| row
                    .get::<_, String>(
                    0
                ))
                .unwrap(),
            "client-1"
        );
    }
}
