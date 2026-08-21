use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use ruter::logs::{ParserKind, parse_file};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Post-hoc router benchmark observability")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Discover a benchmark run and create its immutable-input analysis directory.
    Init {
        /// srt-slurm run root. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Output directory. Defaults to <root>/.ruter.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Serve the static SQLite-backed view for a previously initialized run.
    View {
        /// srt-slurm run root. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Analysis directory. Defaults to <root>/.ruter.
        #[arg(long)]
        analysis: Option<PathBuf>,
        #[arg(long)]
        port: Option<u16>,
    },
    /// Parse a single saved router or worker log to normalized JSON Lines.
    ParseLog {
        #[arg(long)]
        kind: LogKind,
        #[arg(long)]
        input: PathBuf,
        /// Required for a worker log.
        #[arg(long)]
        worker_index: Option<u32>,
        /// Physical worker role, as encoded by srt-slurm's worker log name.
        #[arg(long, value_enum, default_value_t = WorkerRole::Agg)]
        worker_role: WorkerRole,
    },
}

#[derive(Clone, ValueEnum)]
enum LogKind {
    DynamoRouter,
    DynamoWorker,
}

#[derive(Clone, Copy, ValueEnum)]
enum WorkerRole {
    Prefill,
    Decode,
    Agg,
}

impl WorkerRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Agg => "agg",
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Init { root, output } => {
            let output = output.unwrap_or_else(|| root.join(".ruter"));
            let report = ruter::artifacts::initialize(&root, &output)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::View {
            root,
            analysis,
            port,
        } => {
            ruter::view::launch(&analysis.unwrap_or_else(|| root.join(".ruter")), port)?;
        }
        Command::ParseLog {
            kind,
            input,
            worker_index,
            worker_role,
        } => {
            let kind = match kind {
                LogKind::DynamoRouter => ParserKind::DynamoRouter,
                LogKind::DynamoWorker => ParserKind::DynamoWorker {
                    worker_index: worker_index
                        .context("--worker-index is required for a worker log")?,
                    worker_role: worker_role.as_str(),
                },
            };
            for event in parse_file(kind, &input)? {
                println!("{}", serde_json::to_string(&event)?);
            }
        }
    }
    Ok(())
}
