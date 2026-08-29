//! Thin daemon entry point.
//!
//! Wires up logging and settings, constructs a [`Runner`], waits for ctrl-c
//! (Pattern 1 from the `rust-practical:async-lifecycle` skill: bare ctrl-c,
//! no signal handling inside the library), then drains it via
//! `Runner::shutdown`.

use anyhow::Result;

use rediscovery::logger;
use rediscovery::settings::SETTINGS;
use rediscovery::subsystems::runner::Runner;

#[tokio::main]
async fn main() -> Result<()> {
    logger::setup();
    SETTINGS.ensure_output_dir()?;

    let runner = Runner::new();

    tokio::signal::ctrl_c().await?;
    tracing::info!("ctrl-c received, shutting down.");

    // Three-step drain: cancel the root token, close the tracker, wait for
    // every spawned job to finish.
    runner.shutdown().await;

    tracing::info!("daemon stopped cleanly.");
    Ok(())
}
