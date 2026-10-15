// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{self, Write};
use std::sync::{Arc, Mutex};

struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl Write for LogBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn capture_logs(run: impl FnOnce()) -> String {
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::clone(&bytes);
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_target(false)
        .with_writer(move || LogBuffer(Arc::clone(&writer)))
        .finish();
    tracing::subscriber::with_default(subscriber, run);
    let captured = bytes.lock().unwrap().clone();
    String::from_utf8(captured).unwrap()
}
