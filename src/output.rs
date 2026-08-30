//! CSV output shared by the tier modules.
//!
//! [`write_matrix_csv`] writes a dense matrix as a `row` header followed by
//! one line per matrix row. Tier 1's weight-untied cosine dumps and Tier 2's
//! cosine and adjacency heatmaps both call it.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use nalgebra::DMatrix;

use crate::error::Result;

/// Writes `matrix` to `path` as one CSV row per matrix row, each line opening
/// with its row index under a `row` header followed by one `column_j` field
/// per column.
///
/// Floats are written in Rust's shortest round-tripping form.
///
/// # Errors
///
/// Returns [`crate::error::Error::Io`] from creating, writing, or flushing
/// `path`.
pub fn write_matrix_csv(path: &Path, matrix: &DMatrix<f64>) -> Result<()> {
    let mut sink = BufWriter::new(File::create(path)?);
    write!(sink, "row")?;
    for j in 0..matrix.ncols() {
        write!(sink, ",column_{j}")?;
    }
    writeln!(sink)?;

    for i in 0..matrix.nrows() {
        write!(sink, "{i}")?;
        for j in 0..matrix.ncols() {
            write!(sink, ",{}", matrix[(i, j)])?;
        }
        writeln!(sink)?;
    }
    sink.flush()?;
    Ok(())
}
