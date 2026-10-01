use crate::Result;
use arrow::array::*;
use arrow::datatypes::*;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use log::{error, info};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Row {
    pub scraper_endpoint: String,
    pub metric_name: String,
    pub metric_value: f64,
    pub histogram_bucket_lower: Option<f64>,
    pub histogram_bucket_upper: Option<f64>,
    pub histogram_sum: Option<f64>,
    pub histogram_count: Option<f64>,
    pub extras: Vec<(String, String)>, // Sorted metadata key-value pairs
}

pub struct DatasetWriter {
    local_dir: PathBuf,
    buffer: Arc<Mutex<RecordBatchBuffer>>,
    row_count: Arc<Mutex<usize>>,
    parquet_index: Arc<Mutex<usize>>,
    rows_per_parquet: usize,
    outbox_dir: Option<PathBuf>,
    save_handle: tokio::task::JoinHandle<()>,
    start_time: Instant,
    /// Wall-clock time at writer start, as nanoseconds since UNIX_EPOCH.
    /// Captured at the same moment as `start_time` so that
    /// `start_epoch_ns + start_time.elapsed()` yields a monotonic epoch
    /// timestamp even if the wall clock steps afterwards.
    start_epoch_ns: i64,
}

struct RecordBatchBuffer {
    scraper_endpoints: Vec<String>,
    metric_names: Vec<String>,
    metric_values: Vec<f64>,
    histogram_bucket_lowers: Vec<Option<f64>>,
    histogram_bucket_uppers: Vec<Option<f64>>,
    histogram_sums: Vec<Option<f64>>,
    histogram_counts: Vec<Option<f64>>,
    time_since_starts: Vec<f64>,
    timestamp_ns: Vec<i64>,
    extras_columns: Vec<Vec<String>>, // One Vec<String> per extra column
    extra_column_names: Vec<String>,  // Names of extra columns (sorted)
}

impl RecordBatchBuffer {
    fn new(extra_column_names: Vec<String>) -> Self {
        let extras_columns = extra_column_names.iter().map(|_| Vec::new()).collect();
        Self {
            scraper_endpoints: Vec::new(),
            metric_names: Vec::new(),
            metric_values: Vec::new(),
            histogram_bucket_lowers: Vec::new(),
            histogram_bucket_uppers: Vec::new(),
            histogram_sums: Vec::new(),
            histogram_counts: Vec::new(),
            time_since_starts: Vec::new(),
            timestamp_ns: Vec::new(),
            extras_columns,
            extra_column_names,
        }
    }

    fn len(&self) -> usize {
        self.scraper_endpoints.len()
    }

    fn append_row(&mut self, row: &Row, time_since_start: f64, timestamp_ns: i64) {
        self.scraper_endpoints.push(row.scraper_endpoint.clone());
        self.metric_names.push(row.metric_name.clone());
        self.metric_values.push(row.metric_value);
        self.histogram_bucket_lowers
            .push(row.histogram_bucket_lower);
        self.histogram_bucket_uppers
            .push(row.histogram_bucket_upper);
        self.histogram_sums.push(row.histogram_sum);
        self.histogram_counts.push(row.histogram_count);
        self.time_since_starts.push(time_since_start);
        self.timestamp_ns.push(timestamp_ns);

        // Populate extras columns - ensure all columns have values (empty string if missing)
        for (col_idx, col_name) in self.extra_column_names.iter().enumerate() {
            let value = row
                .extras
                .iter()
                .find(|(k, _)| k == col_name)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(String::new);
            self.extras_columns[col_idx].push(value);
        }
    }

    fn clear(&mut self) {
        self.scraper_endpoints.clear();
        self.metric_names.clear();
        self.metric_values.clear();
        self.histogram_bucket_lowers.clear();
        self.histogram_bucket_uppers.clear();
        self.histogram_sums.clear();
        self.histogram_counts.clear();
        self.time_since_starts.clear();
        self.timestamp_ns.clear();
        for col in &mut self.extras_columns {
            col.clear();
        }
    }

    fn to_record_batch(&self) -> Result<RecordBatch> {
        if self.len() == 0 {
            return Err(crate::NoMoreError::InvalidSchema(
                "Cannot create empty record batch".to_string(),
            ));
        }

        let mut fields = vec![
            Field::new("scraper_endpoint", DataType::Utf8, false),
            Field::new("metric_name", DataType::Utf8, false),
            Field::new("metric_value", DataType::Float64, false),
            Field::new("histogram_bucket_lower", DataType::Float64, true),
            Field::new("histogram_bucket_upper", DataType::Float64, true),
            Field::new("histogram_sum", DataType::Float64, true),
            Field::new("histogram_count", DataType::Float64, true),
            Field::new("time_since_start", DataType::Float64, false),
            Field::new("timestamp_ns", DataType::Int64, false),
        ];

        // Add extra columns to schema
        for col_name in &self.extra_column_names {
            fields.push(Field::new(col_name, DataType::Utf8, false));
        }

        let schema = Arc::new(Schema::new(fields));

        let mut arrays: Vec<Arc<dyn Array>> = vec![
            Arc::new(StringArray::from(self.scraper_endpoints.clone())),
            Arc::new(StringArray::from(self.metric_names.clone())),
            Arc::new(Float64Array::from(self.metric_values.clone())),
            Arc::new(Float64Array::from(self.histogram_bucket_lowers.clone())),
            Arc::new(Float64Array::from(self.histogram_bucket_uppers.clone())),
            Arc::new(Float64Array::from(self.histogram_sums.clone())),
            Arc::new(Float64Array::from(self.histogram_counts.clone())),
            Arc::new(Float64Array::from(self.time_since_starts.clone())),
            Arc::new(Int64Array::from(self.timestamp_ns.clone())),
        ];

        // Add extra column arrays
        for col in &self.extras_columns {
            arrays.push(Arc::new(StringArray::from(col.clone())));
        }

        RecordBatch::try_new(schema, arrays).map_err(|e| crate::NoMoreError::Arrow(e.to_string()))
    }
}

impl DatasetWriter {
    /// Create a new DatasetWriter that writes intermediate files to a local directory.
    ///
    /// # Arguments
    /// * `local_dir` - Local directory for intermediate files (current.arrow, out-N.parquet)
    /// * `rows_per_parquet` - Number of rows before creating a numbered parquet file
    /// * `save_interval_secs` - Interval in seconds for periodic Arrow file saves
    /// * `extra_column_names` - Names of extra columns to include in the schema
    pub fn new(
        local_dir: PathBuf,
        rows_per_parquet: usize,
        save_interval_secs: u64,
        extra_column_names: Vec<String>,
    ) -> Result<Self> {
        Self::with_outbox(
            local_dir,
            rows_per_parquet,
            save_interval_secs,
            extra_column_names,
            None,
        )
    }

    /// Like [`DatasetWriter::new`], but with an outbox for live streaming.
    ///
    /// Every save interval seals the buffered rows into the next immutable
    /// `out-N.parquet` instead of rewriting `current.arrow`, and links it into
    /// `outbox_dir`. Compaction may delete `out-N.parquet` from `local_dir`; the
    /// outbox link belongs to the uploader, which unlinks it once delivered.
    pub fn with_outbox(
        local_dir: PathBuf,
        rows_per_parquet: usize,
        save_interval_secs: u64,
        extra_column_names: Vec<String>,
        outbox_dir: Option<PathBuf>,
    ) -> Result<Self> {
        // Create the local directory if it doesn't exist
        std::fs::create_dir_all(&local_dir).map_err(|e| {
            crate::NoMoreError::Io(std::io::Error::other(format!(
                "Failed to create local directory {}: {}",
                local_dir.display(),
                e
            )))
        })?;

        if let Some(outbox) = &outbox_dir {
            std::fs::create_dir_all(outbox)?;
        }

        let buffer = Arc::new(Mutex::new(RecordBatchBuffer::new(extra_column_names)));
        let row_count = Arc::new(Mutex::new(0));
        let parquet_index = Arc::new(Mutex::new(1));
        // Capture the monotonic and wall-clock start times at the same moment.
        // All per-row epoch timestamps are derived as start_epoch_ns + elapsed
        // (monotonic), so rows stay monotonic even if the wall clock steps.
        let start_time = Instant::now();
        let start_epoch_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                crate::NoMoreError::Io(std::io::Error::other(format!(
                    "System clock is before UNIX_EPOCH: {}",
                    e
                )))
            })?
            .as_nanos() as i64;

        let segments = SegmentState {
            buffer: buffer.clone(),
            row_count: row_count.clone(),
            parquet_index: parquet_index.clone(),
            local_dir: local_dir.clone(),
            outbox_dir: outbox_dir.clone(),
        };

        let save_handle = tokio::spawn(async move {
            periodic_save_task(segments, save_interval_secs).await;
        });

        Ok(Self {
            local_dir,
            buffer,
            row_count,
            parquet_index,
            rows_per_parquet,
            outbox_dir,
            save_handle,
            start_time,
            start_epoch_ns,
        })
    }

    /// Get the local directory where intermediate files are stored.
    pub fn local_dir(&self) -> &Path {
        &self.local_dir
    }

    pub async fn append_row(&self, row: Row) -> Result<()> {
        let elapsed = self.start_time.elapsed();
        let time_since_start = elapsed.as_secs_f64();
        let timestamp_ns = self.start_epoch_ns + elapsed.as_nanos() as i64;
        let mut buffer = self.buffer.lock().await;
        let mut row_count = self.row_count.lock().await;

        buffer.append_row(&row, time_since_start, timestamp_ns);
        *row_count += 1;

        // Check if we need to create a numbered parquet file
        if *row_count >= self.rows_per_parquet {
            let mut parquet_idx = self.parquet_index.lock().await;
            seal_segment(
                &mut buffer,
                &mut parquet_idx,
                &self.local_dir,
                self.outbox_dir.as_deref(),
            )?;
            *row_count = 0;
        }

        Ok(())
    }

    pub async fn append_rows(&self, rows: Vec<Row>) -> Result<()> {
        let elapsed = self.start_time.elapsed();
        let time_since_start = elapsed.as_secs_f64();
        let timestamp_ns = self.start_epoch_ns + elapsed.as_nanos() as i64;
        let mut buffer = self.buffer.lock().await;
        let mut row_count = self.row_count.lock().await;

        for row in rows {
            buffer.append_row(&row, time_since_start, timestamp_ns);
            *row_count += 1;
        }

        // Check if we need to create a numbered parquet file
        if *row_count >= self.rows_per_parquet {
            let mut parquet_idx = self.parquet_index.lock().await;
            seal_segment(
                &mut buffer,
                &mut parquet_idx,
                &self.local_dir,
                self.outbox_dir.as_deref(),
            )?;
            *row_count = 0;
        }

        Ok(())
    }

    /// Shutdown the writer, flushing any remaining data to current.arrow, or to
    /// a final segment when streaming to an outbox.
    /// Returns the local directory path for subsequent compaction.
    pub async fn shutdown(&self) -> Result<PathBuf> {
        // Stop the periodic save task
        self.save_handle.abort();

        let mut buffer = self.buffer.lock().await;
        if buffer.len() > 0 && self.outbox_dir.is_some() {
            let mut row_count = self.row_count.lock().await;
            let mut parquet_idx = self.parquet_index.lock().await;
            seal_segment(
                &mut buffer,
                &mut parquet_idx,
                &self.local_dir,
                self.outbox_dir.as_deref(),
            )?;
            *row_count = 0;
        } else if buffer.len() > 0 {
            let batch = buffer.to_record_batch()?;
            let num_rows = batch.num_rows();
            write_arrow_file_local(&self.local_dir, "current.arrow", &batch)?;
            info!("Flushed current.arrow with {} rows on shutdown", num_rows);
        }

        Ok(self.local_dir.clone())
    }
}

/// Shared writer state for the periodic save task.
struct SegmentState {
    buffer: Arc<Mutex<RecordBatchBuffer>>,
    row_count: Arc<Mutex<usize>>,
    parquet_index: Arc<Mutex<usize>>,
    local_dir: PathBuf,
    outbox_dir: Option<PathBuf>,
}

/// Write the buffered rows to the next `out-N.parquet` and clear the buffer.
///
/// The caller holds the buffer lock throughout, so shutdown cannot observe a
/// half-sealed segment. A failed outbox link is logged: the rows still reach
/// final.parquet through compaction.
fn seal_segment(
    buffer: &mut RecordBatchBuffer,
    parquet_index: &mut usize,
    local_dir: &Path,
    outbox_dir: Option<&Path>,
) -> Result<()> {
    let batch = buffer.to_record_batch()?;
    let filename = format!("out-{}.parquet", *parquet_index);
    *parquet_index += 1;

    write_parquet_file_local(local_dir, &filename, &batch)?;
    info!(
        "Saved parquet file {} with {} rows",
        filename,
        batch.num_rows()
    );
    buffer.clear();
    if let Some(outbox) = outbox_dir {
        if let Err(e) = link_into_outbox(local_dir, outbox, &filename) {
            error!("Error publishing {} to outbox: {}", filename, e);
        }
    }
    Ok(())
}

/// Publish a sealed segment to the outbox, hard-linking when possible. A copy
/// is renamed into place so the uploader never observes a partial file.
fn link_into_outbox(local_dir: &Path, outbox_dir: &Path, filename: &str) -> Result<()> {
    let source = local_dir.join(filename);
    let target = outbox_dir.join(filename);
    if std::fs::hard_link(&source, &target).is_ok() {
        return Ok(());
    }
    let pending = tempfile::Builder::new()
        .prefix(".tachometer-")
        .suffix(".tmp")
        .tempfile_in(outbox_dir)?;
    std::fs::copy(&source, pending.path())?;
    pending
        .persist(&target)
        .map_err(|e| crate::NoMoreError::Io(e.error))?;
    Ok(())
}

async fn periodic_save_task(state: SegmentState, save_interval_secs: u64) {
    let mut interval = tokio::time::interval(Duration::from_secs(save_interval_secs));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        interval.tick().await;

        if state.outbox_dir.is_some() {
            let mut buffer = state.buffer.lock().await;
            if buffer.len() == 0 {
                continue;
            }
            let mut row_count = state.row_count.lock().await;
            let mut parquet_idx = state.parquet_index.lock().await;
            match seal_segment(
                &mut buffer,
                &mut parquet_idx,
                &state.local_dir,
                state.outbox_dir.as_deref(),
            ) {
                Ok(()) => *row_count = 0,
                Err(e) => error!("Error sealing segment: {}", e),
            }
            continue;
        }

        let batch = {
            let buffer_guard = state.buffer.lock().await;
            if buffer_guard.len() == 0 {
                continue;
            }
            match buffer_guard.to_record_batch() {
                Ok(batch) => batch,
                Err(e) => {
                    error!("Error creating record batch for periodic save: {}", e);
                    continue;
                }
            }
        };

        // Save to current.arrow on local disk
        if let Err(e) = write_arrow_file_local(&state.local_dir, "current.arrow", &batch) {
            error!("Error saving current.arrow: {}", e);
        } else {
            info!("Saved current.arrow with {} rows", batch.num_rows());
        }
    }
}

/// Write an Arrow IPC stream file to local disk.
fn write_arrow_file_local(local_dir: &Path, filename: &str, batch: &RecordBatch) -> Result<()> {
    publish_capture_file(local_dir, filename, |file| {
        let mut writer = BufWriter::new(file);
        {
            let mut stream_writer = StreamWriter::try_new(&mut writer, batch.schema().as_ref())
                .map_err(|e| crate::NoMoreError::Arrow(e.to_string()))?;
            stream_writer
                .write(batch)
                .map_err(|e| crate::NoMoreError::Arrow(e.to_string()))?;
            stream_writer
                .finish()
                .map_err(|e| crate::NoMoreError::Arrow(e.to_string()))?;
        }
        writer.flush()?;
        Ok(())
    })
}

/// Write a Parquet file to local disk.
fn write_parquet_file_local(local_dir: &Path, filename: &str, batch: &RecordBatch) -> Result<()> {
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_write_batch_size(100_000)
        .build();

    publish_capture_file(local_dir, filename, |file| {
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
            .map_err(|e| crate::NoMoreError::Parquet(e.to_string()))?;
        writer
            .write(batch)
            .map_err(|e| crate::NoMoreError::Parquet(e.to_string()))?;
        writer
            .close()
            .map_err(|e| crate::NoMoreError::Parquet(e.to_string()))?;
        Ok(())
    })
}

/// Publish complete captures atomically. Open readers keep the previous snapshot
/// during replacement, and the compactor cannot observe unfinished output.
fn publish_capture_file(
    local_dir: &Path,
    filename: &str,
    write: impl FnOnce(&mut File) -> Result<()>,
) -> Result<()> {
    // A same-directory temporary file guarantees the rename stays on one
    // filesystem. Dropping it removes partial output on write/close errors.
    let mut pending = tempfile::Builder::new()
        .prefix(".tachometer-")
        .suffix(".tmp")
        .tempfile_in(local_dir)?;
    write(pending.as_file_mut())?;
    pending
        .persist(local_dir.join(filename))
        .map_err(|e| crate::NoMoreError::Io(e.error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic_compact_and_sync;
    use object_store::local::LocalFileSystem;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::ArrowWriter;
    use std::io::Write;

    fn batch(first: usize) -> RecordBatch {
        let mut buffer = RecordBatchBuffer::new(vec![]);
        for value in first..first + 3 {
            buffer.append_row(
                &Row {
                    scraper_endpoint: "worker0".into(),
                    metric_name: "requests_total".into(),
                    metric_value: value as f64,
                    histogram_bucket_lower: None,
                    histogram_bucket_upper: None,
                    histogram_sum: None,
                    histogram_count: None,
                    extras: vec![],
                },
                value as f64,
                1_700_000_000_000_000_000 + value as i64,
            );
        }
        buffer.to_record_batch().unwrap()
    }

    #[test]
    fn arrow_snapshot_readers_survive_replacement() {
        use std::io::Read;

        let local = tempfile::tempdir().unwrap();
        let path = local.path().join("current.arrow");
        write_arrow_file_local(local.path(), "current.arrow", &batch(0)).unwrap();
        let expected = std::fs::read(&path).unwrap();
        let mut open_snapshot = File::open(&path).unwrap();

        write_arrow_file_local(local.path(), "current.arrow", &batch(3)).unwrap();
        let mut previous = Vec::new();
        open_snapshot.read_to_end(&mut previous).unwrap();
        assert_eq!(previous, expected);
        assert_ne!(std::fs::read(&path).unwrap(), previous);
        let mut reader =
            arrow::ipc::reader::StreamReader::try_new(File::open(&path).unwrap(), None).unwrap();
        assert_eq!(reader.next().unwrap().unwrap(), batch(3));
        assert!(reader.next().is_none());
    }

    #[test]
    fn compaction_cannot_observe_an_unfinished_parquet_file() {
        let local = tempfile::tempdir().unwrap();
        let remote = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalFileSystem::new_with_prefix(remote.path()).unwrap());
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        write_parquet_file_local(local.path(), "out-1.parquet", &batch(0)).unwrap();

        publish_capture_file(local.path(), "out-2.parquet", |file| {
            let next = batch(3);
            let mut writer = ArrowWriter::try_new(file, next.schema(), None).unwrap();
            writer.write(&next).unwrap();
            // Flush actual row data, leaving the writer open without a footer.
            // No timing or scheduler luck is needed to exercise this interleaving.
            writer.flush().unwrap();
            let pending: Vec<_> = std::fs::read_dir(local.path())
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| path.file_name().is_some_and(|name| name != "out-1.parquet"))
                .collect();
            assert_eq!(pending.len(), 1);
            assert!(
                ParquetRecordBatchReaderBuilder::try_new(File::open(&pending[0]).unwrap()).is_err()
            );
            // The real compactor consumes the completed first part, but must
            // neither read nor delete the part whose writer is still open.
            let compacted = runtime
                .block_on(periodic_compact_and_sync(
                    local.path(),
                    store.clone(),
                    "run",
                ))
                .unwrap();
            assert!(
                pending[0].exists(),
                "compactor deleted the unfinished write"
            );
            assert_eq!(compacted, 1);
            assert!(!local.path().join("out-2.parquet").exists());
            writer.close().unwrap();
            Ok(())
        })
        .unwrap();

        assert!(local.path().join("out-2.parquet").exists());
        assert_eq!(
            runtime
                .block_on(periodic_compact_and_sync(
                    local.path(),
                    store.clone(),
                    "run"
                ))
                .unwrap(),
            2
        );
        assert_eq!(
            runtime
                .block_on(periodic_compact_and_sync(local.path(), store, "run"))
                .unwrap(),
            0
        );
        let files: Vec<_> = std::fs::read_dir(local.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(files.len(), 1);
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&files[0]).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let mut values = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let column = batch
                .column_by_name("metric_value")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            values.extend_from_slice(column.values());
        }
        values.sort_by(f64::total_cmp);
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn failed_parquet_write_does_not_publish_or_leave_a_partial_file() {
        let local = tempfile::tempdir().unwrap();
        let result = publish_capture_file(local.path(), "out-1.parquet", |file| {
            file.write_all(b"PAR1unfinished parquet")?;
            Err(crate::NoMoreError::Io(std::io::Error::other(
                "injected write failure",
            )))
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read_dir(local.path()).unwrap().count(), 0);
    }

    #[test]
    fn failed_parquet_publication_does_not_leave_a_partial_file() {
        let local = tempfile::tempdir().unwrap();
        // Renaming a regular file over an existing directory must fail.
        let destination = local.path().join("out-1.parquet");
        std::fs::create_dir(&destination).unwrap();
        assert!(write_parquet_file_local(local.path(), "out-1.parquet", &batch(0)).is_err());
        assert!(destination.is_dir());
        assert_eq!(std::fs::read_dir(local.path()).unwrap().count(), 1);
    }
}
