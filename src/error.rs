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

    #[error("vertex {vertex} leaves the vertex range of a {order}-vertex graph")]
    VertexOutOfBounds { vertex: usize, order: usize },

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

    #[error("run parameter `{parameter}` must be non-negative and finite, got {value}")]
    NegativeRunParameter { parameter: &'static str, value: f64 },

    #[error("run parameter `{parameter}` must lie in [0, 1), got {value}")]
    RunParameterNotAFraction { parameter: &'static str, value: f64 },

    #[error("embedding dimension must be at least 1, got {dimension}")]
    InvalidDimension { dimension: usize },

    #[error("embedding has {rows} rows but the graph has {order} vertices")]
    EmbeddingOrderMismatch { rows: usize, order: usize },

    #[error("embedding factors differ in width: {columns} against {other_columns}")]
    EmbeddingShapeMismatch {
        columns: usize,
        other_columns: usize,
    },

    #[error("run parameter `max_steps` must be at least 1, got 0")]
    ZeroMaxSteps,

    #[error("a {rows}×{columns} matrix is beyond the representable range")]
    MatrixTooLarge { rows: usize, columns: usize },

    #[error("weight matrix is {rows}×{columns}, expected {width}×{width}")]
    WeightShapeMismatch {
        rows: usize,
        columns: usize,
        width: usize,
    },

    #[error("graph has vertex pairs at only {available} of the required distinct distances")]
    InsufficientDistanceShells { available: usize },
}

/// Crate-wide result alias over [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
