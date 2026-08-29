use crate::settings::SETTINGS;
use std::str::FromStr;
use tracing::Level;

/// Initializes the global `tracing` subscriber at the level configured in
/// `SETTINGS`. Falls back to `INFO`, with a message on stderr, if the
/// configured level string does not parse.
pub fn setup() {
    let level = if let Ok(level) = Level::from_str(SETTINGS.logger.level.as_str()) {
        level
    } else {
        eprintln!(
            "Invalid log level: {}, defaulting to INFO",
            SETTINGS.logger.level
        );
        Level::INFO
    };

    tracing_subscriber::fmt().with_max_level(level).init();
}
