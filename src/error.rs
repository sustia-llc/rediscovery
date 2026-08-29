//! Typed error surface for the library.
//!
//! Library code returns [`Result`] (aliasing this crate's [`Error`]).
//! Application code (`main.rs`, integration tests) uses `anyhow`, which
//! converts any variant via `anyhow::Error`'s blanket `From` impl for
//! `std::error::Error` types that are `Send + Sync + 'static` — true of
//! every `#[from]` source here.

/// All fallible operations in the library funnel through this enum.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load configuration")]
    Config(#[from] config::ConfigError),

    #[error("I/O operation failed")]
    Io(#[from] std::io::Error),

    #[error("graph parameter `{parameter}` must be at least {minimum}, got {value}")]
    InvalidGraphParameter {
        parameter: &'static str,
        minimum: usize,
        value: usize,
    },

    #[error("edge ({u}, {v}) leaves the vertex range of a {order}-vertex graph")]
    EdgeOutOfBounds { u: usize, v: usize, order: usize },

    #[error("self-loop at vertex {vertex}: graphs in this crate are loop-free")]
    SelfLoop { vertex: usize },

    #[error("`{constructor}` was asked for a vertex count beyond the representable range")]
    GraphTooLarge { constructor: &'static str },

    #[error("vertex {vertex} has degree zero, so D⁻¹ does not exist")]
    IsolatedVertex { vertex: usize },

    #[error("expected a square matrix, got {rows}×{columns}")]
    NotSquare { rows: usize, columns: usize },

    #[error("expected a non-empty matrix, got 0×0")]
    EmptyMatrix,

    #[error("matrix entry ({row}, {column}) is not finite")]
    NonFinite { row: usize, column: usize },

    #[error("matrix entries ({row}, {column}) and ({column}, {row}) differ")]
    NotSymmetric { row: usize, column: usize },

    #[error("run parameter `{parameter}` must be positive and finite, got {value}")]
    InvalidRunParameter { parameter: &'static str, value: f64 },

    #[error("embedding dimension must be at least 1, got {dimension}")]
    InvalidDimension { dimension: usize },

    #[error("a {rows}×{columns} embedding is beyond the representable range")]
    EmbeddingTooLarge { rows: usize, columns: usize },

    #[error("embedding has {rows} rows but the graph has {order} vertices")]
    EmbeddingOrderMismatch { rows: usize, order: usize },

    #[error("embeddings differ in shape: {rows}×{columns} against {other_rows}×{other_columns}")]
    EmbeddingShapeMismatch {
        rows: usize,
        columns: usize,
        other_rows: usize,
        other_columns: usize,
    },
}

/// Crate-wide result alias over [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
