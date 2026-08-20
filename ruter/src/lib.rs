//! Ruter turns benchmark artifacts into timestamped, source-attributed events.
//!
//! Parsers deliberately retain their source and raw line: a dashboard may show a
//! derived field, but every conclusion must remain traceable to an artifact.

pub mod artifacts;
pub mod database;
pub mod logs;
pub mod model;
pub mod tables;
pub mod view;
