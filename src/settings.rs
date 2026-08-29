//! Application settings loaded via the `config` crate.
//!
//! Sources layer in this precedence order: `config/default.toml`, an
//! optional `config/<RUN_MODE>.toml` override (`RUN_MODE` defaults to
//! `development`), then `__`-separated environment variables.

use crate::error::Result;
use config::{Config, Environment, File};
use serde::Deserialize;
use std::env;
use std::path::PathBuf;
use std::sync::LazyLock;

pub static SETTINGS: LazyLock<Settings> =
    LazyLock::new(|| Settings::new().expect("invariant: settings must be valid at startup"));

#[derive(Debug, Clone, Deserialize)]
pub struct Logger {
    pub level: String,
}

/// Where per-run experiment CSV logs are written.
#[derive(Debug, Clone, Deserialize)]
pub struct Output {
    pub dir: PathBuf,
}

/// Global RNG seed for deterministic, bit-reproducible runs.
#[derive(Debug, Clone, Deserialize)]
pub struct Rng {
    pub seed: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Settings {
    pub environment: String,
    pub logger: Logger,
    pub output: Output,
    pub rng: Rng,
}

impl Settings {
    /// Builds `Settings` from `config/default.toml`, the `RUN_MODE` override
    /// file, and environment variables.
    ///
    /// # Errors
    ///
    /// Returns `Error::Config` if a config source cannot be read or the
    /// merged result cannot be deserialized into `Settings`.
    pub fn new() -> Result<Self> {
        let run_mode = env::var("RUN_MODE").unwrap_or_else(|_| "development".into());

        let builder = Config::builder()
            .add_source(File::with_name("config/default"))
            .add_source(File::with_name(&format!("config/{run_mode}")).required(false))
            .add_source(Environment::default().separator("__"));

        let config = builder.build()?;
        Ok(config.try_deserialize()?)
    }

    /// Creates the experiment output directory (`self.output.dir`) if it does
    /// not already exist.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` if directory creation fails.
    pub fn ensure_output_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.output.dir)?;
        Ok(())
    }
}
