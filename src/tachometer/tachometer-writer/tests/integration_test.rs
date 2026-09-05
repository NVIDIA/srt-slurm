use arrow::array::Array;
use bytes::Bytes;
use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::path::PathBuf;
use std::sync::Arc;
use tachometer_writer::{compact_and_upload, periodic_compact_and_sync, DatasetWriter, Row};
use tempfile::TempDir;

/// Create a test setup with separate local and remote directories
async fn create_test_setup() -> (TempDir, TempDir, PathBuf, Arc<dyn ObjectStore>, String) {
    let local_temp_dir = TempDir::new().unwrap();
    let remote_temp_dir = TempDir::new().unwrap();
    let local_dir = local_temp_dir.path().join("test_run");
    let remote_store = Arc::new(LocalFileSystem::new_with_prefix(remote_temp_dir.path()).unwrap())
        as Arc<dyn ObjectStore>;
    let remote_path = "test_run".to_string();
    (
        local_temp_dir,
        remote_temp_dir,
        local_dir,
        remote_store,
        remote_path,
    )
}

fn create_test_rows(count: usize, metric_prefix: &str) -> Vec<Row> {
    (0..count)
        .map(|i| Row {
            scraper_endpoint: "test_endpoint".to_string(),
            metric_name: format!("{}_{}", metric_prefix, i),
            metric_value: i as f64,
            histogram_bucket_lower: None,
            histogram_bucket_upper: None,
            histogram_sum: None,
            histogram_count: None,
            extras: vec![],
        })
        .collect()
}

async fn read_parquet_file(
    store: &Arc<dyn ObjectStore>,
    path: &str,
) -> Vec<arrow::record_batch::RecordBatch> {
    let path = object_store::path::Path::from(path);
    let data = store.get(&path).await.unwrap().bytes().await.unwrap();
    let bytes = Bytes::from(data.to_vec());
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
    let reader = builder.build().unwrap();
    let batches: Vec<_> = reader.map(|batch_result| batch_result.unwrap()).collect();
    batches
}

fn read_local_parquet_file(path: &std::path::Path) -> Vec<arrow::record_batch::RecordBatch> {
    let file = std::fs::File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.build().unwrap();
    let batches: Vec<_> = reader.map(|batch_result| batch_result.unwrap()).collect();
    batches
}

#[tokio::test]
async fn test_writer_creates_parquet_on_threshold() {
    let (_local_temp, _remote_temp, local_dir, _remote_store, _remote_path) =
        create_test_setup().await;

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            5,    // rows_per_parquet
            1000, // save_interval_secs (long to avoid periodic saves during test)
            vec![],
        )
        .unwrap(),
    );

    // Write exactly rows_per_parquet rows
    let rows = create_test_rows(5, "metric");
    writer.append_rows(rows).await.unwrap();

    // Give a moment for async operations
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Check that parquet file was created locally
    let path = local_dir.join("out-1.parquet");
    assert!(path.exists(), "Parquet file should exist at {:?}", path);

    let batches = read_local_parquet_file(&path);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 5);

    // Verify the data
    let batch = &batches[0];
    let metric_names = batch
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap();
    assert_eq!(metric_names.value(0), "metric_0");
    assert_eq!(metric_names.value(4), "metric_4");

    writer.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_writer_flushes_on_shutdown() {
    let (_local_temp, _remote_temp, local_dir, _remote_store, _remote_path) =
        create_test_setup().await;

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            100,  // rows_per_parquet (high threshold)
            1000, // save_interval_secs
            vec![],
        )
        .unwrap(),
    );

    // Write fewer rows than threshold
    let rows = create_test_rows(3, "metric");
    writer.append_rows(rows).await.unwrap();

    // Shutdown should flush remaining data
    writer.shutdown().await.unwrap();

    // Check that current.arrow was created locally
    let path = local_dir.join("current.arrow");
    assert!(path.exists(), "current.arrow should exist");

    let data = std::fs::read(&path).unwrap();
    assert!(!data.is_empty());

    let mut reader =
        arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(data), None).unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert_eq!(batch.num_rows(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_compact_and_upload_combines_files() {
    let (_local_temp, _remote_temp, local_dir, remote_store, remote_path) =
        create_test_setup().await;

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            3,    // rows_per_parquet
            1000, // save_interval_secs
            vec![],
        )
        .unwrap(),
    );

    // Write rows that will create multiple parquet files
    let rows1 = create_test_rows(3, "metric_a");
    let rows2 = create_test_rows(3, "metric_b");
    let rows3 = create_test_rows(3, "metric_c");

    writer.append_rows(rows1).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    writer.append_rows(rows2).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    writer.append_rows(rows3).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    writer.shutdown().await.unwrap();

    // Verify we have multiple parquet files locally
    let entries: Vec<_> = std::fs::read_dir(&local_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("out-"))
        .collect();
    assert!(
        entries.len() >= 3,
        "Expected at least 3 parquet files locally"
    );

    // Run compaction and upload
    compact_and_upload(&local_dir, remote_store.clone(), &remote_path)
        .await
        .unwrap();

    // Verify final.parquet exists in remote storage and has all rows
    let final_path = format!("{}/final.parquet", remote_path);
    let batches = read_parquet_file(&remote_store, &final_path).await;
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 9);

    // Verify intermediate files are deleted from local disk
    let remaining_files: Vec<_> = std::fs::read_dir(&local_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("out-"))
        .collect();
    assert_eq!(
        remaining_files.len(),
        0,
        "Intermediate parquet files should be deleted from local disk"
    );
}

#[tokio::test]
async fn test_compact_and_upload_sorts_by_metric_name_and_time() {
    let (_local_temp, _remote_temp, local_dir, remote_store, remote_path) =
        create_test_setup().await;

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            100,  // rows_per_parquet
            1000, // save_interval_secs
            vec![],
        )
        .unwrap(),
    );

    // Create rows with different metric names in non-sorted order
    let rows = vec![
        Row {
            scraper_endpoint: "ep1".to_string(),
            metric_name: "zebra".to_string(),
            metric_value: 1.0,
            histogram_bucket_lower: None,
            histogram_bucket_upper: None,
            histogram_sum: None,
            histogram_count: None,
            extras: vec![],
        },
        Row {
            scraper_endpoint: "ep1".to_string(),
            metric_name: "alpha".to_string(),
            metric_value: 2.0,
            histogram_bucket_lower: None,
            histogram_bucket_upper: None,
            histogram_sum: None,
            histogram_count: None,
            extras: vec![],
        },
        Row {
            scraper_endpoint: "ep1".to_string(),
            metric_name: "alpha".to_string(),
            metric_value: 3.0,
            histogram_bucket_lower: None,
            histogram_bucket_upper: None,
            histogram_sum: None,
            histogram_count: None,
            extras: vec![],
        },
        Row {
            scraper_endpoint: "ep1".to_string(),
            metric_name: "beta".to_string(),
            metric_value: 4.0,
            histogram_bucket_lower: None,
            histogram_bucket_upper: None,
            histogram_sum: None,
            histogram_count: None,
            extras: vec![],
        },
    ];

    writer.append_rows(rows).await.unwrap();
    writer.shutdown().await.unwrap();

    // Run compaction
    compact_and_upload(&local_dir, remote_store.clone(), &remote_path)
        .await
        .unwrap();

    // Read back and verify sorting
    let final_path = format!("{}/final.parquet", remote_path);
    let batches = read_parquet_file(&remote_store, &final_path).await;
    let batch = &batches[0];

    // After compaction, schema has metric_name_clean added, so use column_by_name
    // Polars uses LargeUtf8 (LargeStringArray) for strings
    let metric_names = batch
        .column_by_name("metric_name")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
        .unwrap();
    let time_values = batch
        .column_by_name("time_since_start")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();

    // Verify sorting: should be sorted by metric_name first, then time_since_start
    assert_eq!(metric_names.value(0), "alpha");
    assert_eq!(metric_names.value(1), "alpha");
    assert_eq!(metric_names.value(2), "beta");
    assert_eq!(metric_names.value(3), "zebra");

    // For rows with same metric_name, verify they're sorted by time_since_start
    assert!(
        time_values.value(0) <= time_values.value(1),
        "Same metric_name rows should be sorted by time"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_compact_and_upload_combines_parquet_and_arrow() {
    let (_local_temp, _remote_temp, local_dir, remote_store, remote_path) =
        create_test_setup().await;

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            3,    // rows_per_parquet
            1000, // save_interval_secs
            vec![],
        )
        .unwrap(),
    );

    // Write enough to create a parquet file
    let rows1 = create_test_rows(3, "parquet_metric");
    writer.append_rows(rows1).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Write less than threshold (will be in current.arrow after shutdown)
    let rows2 = create_test_rows(2, "arrow_metric");
    writer.append_rows(rows2).await.unwrap();
    writer.shutdown().await.unwrap();

    // Verify both files exist locally
    let parquet_exists = local_dir.join("out-1.parquet").exists();
    let arrow_exists = local_dir.join("current.arrow").exists();
    assert!(parquet_exists, "Parquet file should exist");
    assert!(arrow_exists, "Arrow file should exist");

    // Run compaction and upload
    compact_and_upload(&local_dir, remote_store.clone(), &remote_path)
        .await
        .unwrap();

    // Verify final.parquet has all rows in remote storage
    let final_path = format!("{}/final.parquet", remote_path);
    let batches = read_parquet_file(&remote_store, &final_path).await;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 5);

    // Verify current.arrow is deleted locally
    let arrow_exists_after = local_dir.join("current.arrow").exists();
    assert!(
        !arrow_exists_after,
        "current.arrow should be deleted after compaction"
    );
}

#[tokio::test]
async fn test_compact_and_upload_empty_dataset() {
    let (_local_temp, _remote_temp, local_dir, remote_store, remote_path) =
        create_test_setup().await;

    // Create the local directory (would normally be done by DatasetWriter)
    std::fs::create_dir_all(&local_dir).unwrap();

    // Run compaction on empty dataset
    compact_and_upload(&local_dir, remote_store.clone(), &remote_path)
        .await
        .unwrap();

    // Should not create final.parquet in remote storage
    let final_exists = remote_store
        .head(&object_store::path::Path::from(format!(
            "{}/final.parquet",
            remote_path
        )))
        .await
        .is_ok();
    assert!(
        !final_exists,
        "final.parquet should not be created for empty dataset"
    );
}

#[tokio::test]
async fn test_writer_handles_histogram_rows() {
    let (_local_temp, _remote_temp, local_dir, _remote_store, _remote_path) =
        create_test_setup().await;

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            100,  // rows_per_parquet
            1000, // save_interval_secs
            vec![],
        )
        .unwrap(),
    );

    let rows = vec![
        Row {
            scraper_endpoint: "ep1".to_string(),
            metric_name: "histogram_metric".to_string(),
            metric_value: 10.0,
            histogram_bucket_lower: Some(0.0),
            histogram_bucket_upper: Some(1.0),
            histogram_sum: Some(123.4),
            histogram_count: Some(100.0),
            extras: vec![],
        },
        Row {
            scraper_endpoint: "ep1".to_string(),
            metric_name: "histogram_metric".to_string(),
            metric_value: 20.0,
            histogram_bucket_lower: Some(1.0),
            histogram_bucket_upper: Some(2.0),
            histogram_sum: Some(123.4),
            histogram_count: Some(100.0),
            extras: vec![],
        },
    ];

    writer.append_rows(rows).await.unwrap();
    writer.shutdown().await.unwrap();

    // Verify data was written correctly to local disk
    let path = local_dir.join("current.arrow");
    let data = std::fs::read(&path).unwrap();

    let mut reader =
        arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(data), None).unwrap();
    let batch = reader.next().unwrap().unwrap();

    // Verify histogram columns
    let lowers = batch
        .column(3)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();
    let uppers = batch
        .column(4)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();

    // Check that values are not null and have correct values
    assert!(!lowers.is_null(0));
    assert!(!uppers.is_null(0));
    assert!(!lowers.is_null(1));
    assert!(!uppers.is_null(1));

    assert_eq!(lowers.value(0), 0.0);
    assert_eq!(uppers.value(0), 1.0);
    assert_eq!(lowers.value(1), 1.0);
    assert_eq!(uppers.value(1), 2.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_compact_and_upload_preserves_all_columns() {
    let (_local_temp, _remote_temp, local_dir, remote_store, remote_path) =
        create_test_setup().await;

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            3,    // rows_per_parquet
            1000, // save_interval_secs
            vec![],
        )
        .unwrap(),
    );

    let rows = create_test_rows(3, "test");
    writer.append_rows(rows).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    writer.shutdown().await.unwrap();

    compact_and_upload(&local_dir, remote_store.clone(), &remote_path)
        .await
        .unwrap();

    // Verify schema is preserved
    let final_path = format!("{}/final.parquet", remote_path);
    let batches = read_parquet_file(&remote_store, &final_path).await;
    let batch = &batches[0];
    let schema = batch.schema();

    // Should have 10 columns with required names (order may vary with polars)
    assert_eq!(schema.fields().len(), 10);

    // Verify all required columns exist by name
    let required_columns = [
        "scraper_endpoint",
        "metric_name",
        "metric_name_clean",
        "metric_value",
        "histogram_bucket_lower",
        "histogram_bucket_upper",
        "histogram_sum",
        "histogram_count",
        "time_since_start",
        "timestamp_ns",
    ];

    for col_name in &required_columns {
        assert!(
            schema.fields().iter().any(|f| f.name() == *col_name),
            "Schema should have column '{}'",
            col_name
        );
    }
}

#[tokio::test]
async fn test_compact_and_upload_extracts_metric_name_clean() {
    let (_local_temp, _remote_temp, local_dir, remote_store, remote_path) =
        create_test_setup().await;

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            100,  // rows_per_parquet
            1000, // save_interval_secs
            vec![],
        )
        .unwrap(),
    );

    // Create rows with Prometheus-style labels in metric names
    let rows = vec![
        Row {
            scraper_endpoint: "ep1".to_string(),
            metric_name: "cpu_usage{host=\"server1\",cpu=\"0\"}".to_string(),
            metric_value: 50.0,
            histogram_bucket_lower: None,
            histogram_bucket_upper: None,
            histogram_sum: None,
            histogram_count: None,
            extras: vec![],
        },
        Row {
            scraper_endpoint: "ep1".to_string(),
            metric_name: "cpu_usage{host=\"server1\",cpu=\"1\"}".to_string(),
            metric_value: 60.0,
            histogram_bucket_lower: None,
            histogram_bucket_upper: None,
            histogram_sum: None,
            histogram_count: None,
            extras: vec![],
        },
        Row {
            scraper_endpoint: "ep1".to_string(),
            metric_name: "memory_used".to_string(), // No labels
            metric_value: 1024.0,
            histogram_bucket_lower: None,
            histogram_bucket_upper: None,
            histogram_sum: None,
            histogram_count: None,
            extras: vec![],
        },
    ];

    writer.append_rows(rows).await.unwrap();
    writer.shutdown().await.unwrap();

    // Run compaction
    compact_and_upload(&local_dir, remote_store.clone(), &remote_path)
        .await
        .unwrap();

    // Read back and verify metric_name_clean extraction
    let final_path = format!("{}/final.parquet", remote_path);
    let batches = read_parquet_file(&remote_store, &final_path).await;
    let batch = &batches[0];

    // Polars uses LargeUtf8 (LargeStringArray) for strings
    let metric_names = batch
        .column_by_name("metric_name")
        .expect("Should have metric_name column")
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
        .expect("metric_name should be LargeStringArray");
    let metric_names_clean = batch
        .column_by_name("metric_name_clean")
        .expect("Should have metric_name_clean column")
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
        .expect("metric_name_clean should be LargeStringArray");

    // Verify metric_name_clean strips the labels
    for i in 0..batch.num_rows() {
        let name = metric_names.value(i);
        let clean = metric_names_clean.value(i);

        // Clean name should be everything before '{'
        let expected_clean = name.split('{').next().unwrap();
        assert_eq!(
            clean, expected_clean,
            "metric_name_clean should strip labels"
        );
    }

    // Verify specific values
    assert!(metric_names_clean.iter().any(|v| v == Some("cpu_usage")));
    assert!(metric_names_clean.iter().any(|v| v == Some("memory_used")));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_compact_and_upload_skips_corrupt_files() {
    let (_local_temp, _remote_temp, local_dir, remote_store, remote_path) =
        create_test_setup().await;

    // Create the local directory
    std::fs::create_dir_all(&local_dir).unwrap();

    // Create a valid parquet file manually
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    let schema = Arc::new(Schema::new(vec![
        Field::new("scraper_endpoint", DataType::Utf8, false),
        Field::new("metric_name", DataType::Utf8, false),
        Field::new("metric_value", DataType::Float64, false),
        Field::new("histogram_bucket_lower", DataType::Float64, true),
        Field::new("histogram_bucket_upper", DataType::Float64, true),
        Field::new("histogram_sum", DataType::Float64, true),
        Field::new("histogram_count", DataType::Float64, true),
        Field::new("time_since_start", DataType::Float64, false),
        Field::new("timestamp_ns", DataType::Int64, false),
    ]));

    // Write a valid out-1.parquet
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["ep1", "ep1"])),
            Arc::new(StringArray::from(vec!["valid_metric_1", "valid_metric_2"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![None, None])),
            Arc::new(Float64Array::from(vec![None, None])),
            Arc::new(Float64Array::from(vec![None, None])),
            Arc::new(Float64Array::from(vec![None, None])),
            Arc::new(Float64Array::from(vec![0.1, 0.2])),
            Arc::new(Int64Array::from(vec![
                1_700_000_000_000_000_000i64,
                1_700_000_000_100_000_000i64,
            ])),
        ],
    )
    .unwrap();

    let valid_path = local_dir.join("out-1.parquet");
    let file = std::fs::File::create(&valid_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Create a corrupt parquet file (truncated - missing footer)
    let corrupt_path = local_dir.join("out-2.parquet");
    std::fs::write(&corrupt_path, b"PAR1corrupted data without proper footer").unwrap();

    // Run compaction - should skip corrupt file and process valid one
    compact_and_upload(&local_dir, remote_store.clone(), &remote_path)
        .await
        .unwrap();

    // Verify final.parquet has rows only from the valid file
    let final_path = format!("{}/final.parquet", remote_path);
    let batches = read_parquet_file(&remote_store, &final_path).await;
    let batch = &batches[0];

    // Should have 2 rows from the valid file only
    assert_eq!(
        batch.num_rows(),
        2,
        "Should only have rows from valid parquet file"
    );

    // Verify the corrupt file was removed
    assert!(!corrupt_path.exists(), "Corrupt file should be removed");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_periodic_compact_and_sync_creates_incomplete_file() {
    let (_local_temp, _remote_temp, local_dir, remote_store, remote_path) =
        create_test_setup().await;

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            3,    // rows_per_parquet
            1000, // save_interval_secs
            vec![],
        )
        .unwrap(),
    );

    // Write enough to create multiple parquet files
    for i in 0..3 {
        let rows = create_test_rows(3, &format!("metric_batch_{}", i));
        writer.append_rows(rows).await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }
    writer.shutdown().await.unwrap();

    // Verify we have multiple out-*.parquet files
    let out_files: Vec<_> = std::fs::read_dir(&local_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("out-"))
        .collect();
    assert!(
        out_files.len() >= 3,
        "Expected at least 3 out-*.parquet files"
    );

    // Run periodic sync (should create incomplete-*.parquet)
    let files_processed = periodic_compact_and_sync(&local_dir, remote_store.clone(), &remote_path)
        .await
        .unwrap();
    assert!(
        files_processed >= 3,
        "Should have processed at least 3 files"
    );

    // Verify incomplete-*.parquet was created locally
    let incomplete_files: Vec<_> = std::fs::read_dir(&local_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("incomplete-"))
        .collect();
    assert_eq!(
        incomplete_files.len(),
        1,
        "Should have one incomplete-*.parquet file"
    );

    // Verify out-*.parquet files were deleted
    let remaining_out_files: Vec<_> = std::fs::read_dir(&local_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("out-"))
        .collect();
    assert_eq!(
        remaining_out_files.len(),
        0,
        "out-*.parquet files should be deleted after sync"
    );

    // Now run final compaction - should include the incomplete file
    compact_and_upload(&local_dir, remote_store.clone(), &remote_path)
        .await
        .unwrap();

    // Verify final.parquet has all 9 rows
    let final_path = format!("{}/final.parquet", remote_path);
    let batches = read_parquet_file(&remote_store, &final_path).await;
    assert_eq!(
        batches[0].num_rows(),
        9,
        "final.parquet should have all 9 rows"
    );

    // Verify incomplete-*.parquet was deleted after final compaction
    let remaining_incomplete: Vec<_> = std::fs::read_dir(&local_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("incomplete-"))
        .collect();
    assert_eq!(
        remaining_incomplete.len(),
        0,
        "incomplete-*.parquet should be deleted after final compaction"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_compact_handles_multiple_incomplete_files() {
    let (_local_temp, _remote_temp, local_dir, remote_store, remote_path) =
        create_test_setup().await;

    // Create the local directory
    std::fs::create_dir_all(&local_dir).unwrap();

    // Create multiple incomplete-*.parquet files manually
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    let schema = Arc::new(Schema::new(vec![
        Field::new("scraper_endpoint", DataType::Utf8, false),
        Field::new("metric_name", DataType::Utf8, false),
        Field::new("metric_value", DataType::Float64, false),
        Field::new("histogram_bucket_lower", DataType::Float64, true),
        Field::new("histogram_bucket_upper", DataType::Float64, true),
        Field::new("histogram_sum", DataType::Float64, true),
        Field::new("histogram_count", DataType::Float64, true),
        Field::new("time_since_start", DataType::Float64, false),
        Field::new("timestamp_ns", DataType::Int64, false),
    ]));

    // Create incomplete-1.parquet and incomplete-2.parquet
    for idx in 1..=2 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["ep1", "ep1", "ep1"])),
                Arc::new(StringArray::from(vec![
                    format!("metric_{}_a", idx),
                    format!("metric_{}_b", idx),
                    format!("metric_{}_c", idx),
                ])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![None, None, None])),
                Arc::new(Float64Array::from(vec![None, None, None])),
                Arc::new(Float64Array::from(vec![None, None, None])),
                Arc::new(Float64Array::from(vec![None, None, None])),
                Arc::new(Float64Array::from(vec![
                    idx as f64 * 0.1,
                    idx as f64 * 0.2,
                    idx as f64 * 0.3,
                ])),
                Arc::new(Int64Array::from(vec![
                    1_700_000_000_000_000_000i64 + idx as i64,
                    1_700_000_000_000_000_001i64 + idx as i64,
                    1_700_000_000_000_000_002i64 + idx as i64,
                ])),
            ],
        )
        .unwrap();

        let file_path = local_dir.join(format!("incomplete-{}.parquet", idx));
        let file = std::fs::File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    // Run compaction
    compact_and_upload(&local_dir, remote_store.clone(), &remote_path)
        .await
        .unwrap();

    // Verify final.parquet has all 6 rows (3 from each incomplete file)
    let final_path = format!("{}/final.parquet", remote_path);
    let batches = read_parquet_file(&remote_store, &final_path).await;
    assert_eq!(
        batches[0].num_rows(),
        6,
        "Should have 6 rows total from 2 incomplete files"
    );

    // Verify both incomplete files are deleted
    let remaining_incomplete: Vec<_> = std::fs::read_dir(&local_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("incomplete-"))
        .collect();
    assert_eq!(
        remaining_incomplete.len(),
        0,
        "All incomplete files should be deleted"
    );
}

/// Regression guard for the two schema defects:
/// (a) rows must carry an epoch-nanosecond `timestamp_ns` join key that is
///     plausibly recent and non-decreasing across appended rows;
/// (b) metric values must be stored as Float64 so that integers above f32's
///     24-bit mantissa (2^24 = 16,777,216) round-trip exactly. Under the old
///     Float32 schema, 16_777_217.0 would come back as 16_777_216.0.
#[tokio::test]
async fn test_timestamp_ns_and_f64_precision() {
    let (_local_temp, _remote_temp, local_dir, _remote_store, _remote_path) =
        create_test_setup().await;

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            3,    // rows_per_parquet: cut a parquet file after 3 rows
            1000, // save_interval_secs (long to avoid periodic saves during test)
            vec![],
        )
        .unwrap(),
    );

    // 2^24 + 1: the first integer NOT representable in f32
    const F32_UNREPRESENTABLE: f64 = 16_777_217.0;

    let make_row = |name: &str, value: f64| Row {
        scraper_endpoint: "test_endpoint".to_string(),
        metric_name: name.to_string(),
        metric_value: value,
        histogram_bucket_lower: None,
        histogram_bucket_upper: None,
        histogram_sum: Some(F32_UNREPRESENTABLE),
        histogram_count: Some(F32_UNREPRESENTABLE),
        extras: vec![],
    };

    // Append one row at a time with small sleeps so timestamps advance
    writer
        .append_row(make_row("big_counter_bytes", F32_UNREPRESENTABLE))
        .await
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    writer.append_row(make_row("metric_b", 1.5)).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    writer.append_row(make_row("metric_c", 2.5)).await.unwrap();

    // 3 rows == rows_per_parquet, so out-1.parquet must exist
    let path = local_dir.join("out-1.parquet");
    assert!(path.exists(), "Parquet file should exist at {:?}", path);

    let batches = read_local_parquet_file(&path);
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 3);

    // (a) timestamp_ns: present, Int64, non-nullable, recent, non-decreasing
    let schema = batch.schema();
    let ts_field = schema
        .field_with_name("timestamp_ns")
        .expect("Schema should contain timestamp_ns column");
    assert_eq!(ts_field.data_type(), &arrow::datatypes::DataType::Int64);
    assert!(
        !ts_field.is_nullable(),
        "timestamp_ns should be non-nullable"
    );

    let timestamps = batch
        .column_by_name("timestamp_ns")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("timestamp_ns should be an Int64Array");

    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as i64;
    const ONE_HOUR_NS: i64 = 3_600 * 1_000_000_000;

    for i in 0..timestamps.len() {
        assert!(!timestamps.is_null(i), "timestamp_ns must not be null");
        let ts = timestamps.value(i);
        assert!(
            (now_ns - ts).abs() < ONE_HOUR_NS,
            "timestamp_ns {} should be within an hour of now {}",
            ts,
            now_ns
        );
    }
    for i in 1..timestamps.len() {
        assert!(
            timestamps.value(i) >= timestamps.value(i - 1),
            "timestamp_ns must be non-decreasing across appended rows: row {} ({}) < row {} ({})",
            i,
            timestamps.value(i),
            i - 1,
            timestamps.value(i - 1)
        );
    }
    // Rows appended with sleeps in between must have strictly increasing timestamps
    assert!(
        timestamps.value(2) > timestamps.value(0),
        "timestamps of rows appended later should advance"
    );

    // (b) f64 round-trip: 2^24 + 1 must survive exactly (fails under Float32)
    let values = batch
        .column_by_name("metric_value")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .expect("metric_value should be a Float64Array");
    assert_eq!(
        values.value(0),
        F32_UNREPRESENTABLE,
        "metric_value must round-trip 2^24+1 exactly (f32 would truncate to 16_777_216)"
    );

    let sums = batch
        .column_by_name("histogram_sum")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .expect("histogram_sum should be a Float64Array");
    let counts = batch
        .column_by_name("histogram_count")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .expect("histogram_count should be a Float64Array");
    assert_eq!(sums.value(0), F32_UNREPRESENTABLE);
    assert_eq!(counts.value(0), F32_UNREPRESENTABLE);

    writer.shutdown().await.unwrap();
}
