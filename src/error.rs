//! Typed error surface for the library.
//!
//! Library code returns [`Result`] (aliasing this crate's [`Error`]).
//! Application code (`main.rs`, integration tests) uses `anyhow`, which
//! converts any variant via `anyhow::Error`'s blanket `From` impl for
//! `std::error::Error` types that are `Send + Sync + 'static` — true of
//! every `#[from]` source here.

/// Names the graph-constructor parameter that [`Error::InvalidGraphParameter`]
/// rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphParameter {
    /// Number of disjoint paths leaving a path-star root.
    Arms,
    /// Vertices per path-star arm, excluding the root.
    ArmLength,
    /// Grid row count.
    Rows,
    /// Grid column count.
    Columns,
    /// Vertex count of a cycle.
    CycleOrder,
    /// Degree of a tree-star's central vertex.
    RootDegree,
    /// Root-to-leaf path length of a tree-star, in edges.
    PathLength,
    /// Vertex count of a complete graph.
    Order,
}

impl std::fmt::Display for GraphParameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Arms => "arms",
            Self::ArmLength => "arm_len",
            Self::Rows => "rows",
            Self::Columns => "cols",
            Self::CycleOrder | Self::Order => "n",
            Self::RootDegree => "d",
            Self::PathLength => "ell",
        };
        f.write_str(name)
    }
}

/// All fallible operations in the library funnel through this enum.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load configuration")]
    Config(#[from] config::ConfigError),

    #[error("I/O operation failed")]
    Io(#[from] std::io::Error),

    #[error("graph parameter `{parameter}` must be at least {minimum}, got {value}")]
    InvalidGraphParameter {
        parameter: GraphParameter,
        minimum: usize,
        value: usize,
    },

    #[error("edge ({u}, {v}) leaves the vertex range of a {order}-vertex graph")]
    EdgeOutOfBounds { u: usize, v: usize, order: usize },

    #[error("self-loop at vertex {vertex}: graphs in this crate are loop-free")]
    SelfLoop { vertex: usize },

    #[error("the requested graph exceeds the addressable vertex count")]
    GraphTooLarge,

    #[error("vertex {vertex} has degree zero, so D⁻¹ does not exist")]
    IsolatedVertex { vertex: usize },

    #[error("expected a square matrix, got {rows}×{columns}")]
    NotSquare { rows: usize, columns: usize },

    #[error("symmetric eigendecomposition of a {order}×{order} matrix did not converge")]
    EigenNotConverged { order: usize },
}

/// Crate-wide result alias over [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
