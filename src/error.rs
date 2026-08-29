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
}

/// Crate-wide result alias over [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
